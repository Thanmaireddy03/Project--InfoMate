package main

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"log"
	"math"
	"net/http"
	"os"
	"path/filepath"
	"sort"
	"strconv"
	"strings"
	"time"

	pdf "github.com/ledongthuc/pdf"
	"github.com/ollama/ollama/api"
)

type simpleMsg struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

type DocChunk struct {
	Page      int       `json:"page"`
	Text      string    `json:"text"`
	Embedding []float64 `json:"embedding"`
}

type DocIndex struct {
	DocName        string     `json:"doc_name"`
	EmbeddingModel string     `json:"embedding_model"`
	Chunks         []DocChunk `json:"chunks"`
	CreatedUnixSec int64      `json:"created_unix_sec"`
}

func main() {
	// chat flags
	model := flag.String("model", "gemma3:4b", "Ollama chat model tag")
	system := flag.String("system", "You are a concise assistant. Respond ONLY in English.", "system prompt")
	temp := flag.Float64("temp", 0.6, "temperature")
	session := flag.String("session", "default", "session name (persists chat across runs)")

	// RAG flags
	ragOn := flag.Bool("rag", true, "use retrieval-augmented generation when a PDF is loaded")
	embedModel := flag.String("embed_model", "nomic-embed-text", "Ollama embeddings model")
	topK := flag.Int("topk", 4, "number of chunks to retrieve from the PDF")
	flag.Parse()

	client, err := api.ClientFromEnvironment()
	if err != nil {
		log.Fatal(err)
	}

	// session messages
	sessPath := sessionPath(*session)
	messages, err := loadMessages(sessPath)
	if err != nil || len(messages) == 0 {
		messages = []api.Message{{Role: "system", Content: *system}}
		_ = saveMessages(sessPath, messages)
	} else {
		if messages[0].Role != "system" {
			messages = append([]api.Message{{Role: "system", Content: *system}}, messages...)
		} else {
			messages[0].Content = *system
		}
	}

	// load any PDF index for this session
	docIndex, _ := loadDocIndex(*session)

	in := bufio.NewReader(os.Stdin)
	fmt.Printf("Model: %s  |  Session: %s\n", *model, *session)
	fmt.Println("Commands: /help  /reset  /history [n]  /model TAG  /save  /exit")
	fmt.Println("/pdf PATH  /clearpdf  /rag on|off  /topk N  /embedmodel NAME  /context\n")

	for {
		fmt.Print("You > ")
		user, err := in.ReadString('\n')
		if err != nil {
			fmt.Println("\n(exit)")
			return
		}
		user = strings.TrimSpace(user)
		if user == "" {
			continue
		}

		switch {
		case user == "/exit" || user == "/quit":
			_ = saveMessages(sessPath, messages)
			fmt.Println("Bye!")
			return
		case user == "/save":
			if err := saveMessages(sessPath, messages); err != nil {
				fmt.Println("(save failed:", err, ")")
			} else {
				fmt.Println("(saved)")
			}
			continue
		case user == "/reset":
			messages = []api.Message{{Role: "system", Content: *system}}
			_ = saveMessages(sessPath, messages)
			fmt.Println("(context cleared)")
			continue
		case strings.HasPrefix(user, "/history"):
			n := 12
			f := strings.Fields(user)
			if len(f) == 2 {
				if v, e := strconv.Atoi(f[1]); e == nil && v > 0 {
					n = v
				}
			}
			printHistory(messages, n)
			continue
		case strings.HasPrefix(user, "/model "):
			*model = strings.TrimSpace(strings.TrimPrefix(user, "/model "))
			fmt.Printf("(model set to %s)\n", *model)
			continue
		case strings.HasPrefix(user, "/embedmodel "):
			*embedModel = strings.TrimSpace(strings.TrimPrefix(user, "/embedmodel "))
			fmt.Printf("(embedding model set to %s)\n", *embedModel)
			continue
		case strings.HasPrefix(user, "/topk "):
			val := strings.TrimSpace(strings.TrimPrefix(user, "/topk "))
			if v, e := strconv.Atoi(val); e == nil && v > 0 && v <= 10 {
				*topK = v
				fmt.Printf("(topk set to %d)\n", *topK)
			} else {
				fmt.Println("(usage: /topk 1..10)")
			}
			continue
		case strings.HasPrefix(user, "/rag "):
			onoff := strings.ToLower(strings.TrimSpace(strings.TrimPrefix(user, "/rag ")))
			if onoff == "on" {
				*ragOn = true
				fmt.Println("(RAG: ON)")
				continue
			}
			if onoff == "off" {
				*ragOn = false
				fmt.Println("(RAG: OFF)")
				continue
			}
			fmt.Println("(usage: /rag on|off)")
			continue
		case strings.HasPrefix(user, "/clearpdf"):
			docIndex = nil
			_ = removeDocIndex(*session)
			fmt.Println("(cleared PDF index)")
			continue
		case strings.HasPrefix(user, "/pdf "):
			path := strings.TrimSpace(strings.TrimPrefix(user, "/pdf "))
			if path == "" {
				fmt.Println("(usage: /pdf /path/to/file.pdf)")
				continue
			}
			newIdx, e := buildIndexFromPDF(client, path, *embedModel)
			if e != nil {
				fmt.Println("[pdf error]", e)
				continue
			}
			docIndex = newIdx
			if e := saveDocIndex(*session, docIndex); e != nil {
				fmt.Println("(warning: could not persist index:", e, ")")
			}
			*ragOn = true
			fmt.Printf("(indexed %s: %d chunks from %d pages using %s)\n", docIndex.DocName, len(docIndex.Chunks), pagesInIndex(docIndex), docIndex.EmbeddingModel)
			fmt.Println("PDF loaded. RAG is ON. You can now ask questions about its content.")
			continue
		case user == "/context":
			state := "OFF"
			if *ragOn {
				state = "ON"
			}
			chunks := 0
			name := "(none)"
			if docIndex != nil {
				chunks = len(docIndex.Chunks)
				name = docIndex.DocName
			}
			fmt.Printf("RAG: %s | PDF: %s | chunks: %d | topk: %d | embed: %s\n", state, name, chunks, *topK, *embedModel)
			continue
		}


		messages = append(messages, api.Message{Role: "user", Content: user})

		// Inject RAG context 
		reqMessages := messages
		if *ragOn && docIndex != nil && len(docIndex.Chunks) > 0 {
			ctxMsg, err := buildRAGContext(client, *embedModel, *topK, user, docIndex)
			if err != nil {
				fmt.Println("(RAG skipped:", err, ")")
			} else if ctxMsg.Content != "" {
				reqMessages = make([]api.Message, 0, len(messages)+1)
				reqMessages = append(reqMessages, messages[0])     // system
				reqMessages = append(reqMessages, ctxMsg)          // PDF context
				reqMessages = append(reqMessages, messages[1:]...) // rest
			}
		}

		req := &api.ChatRequest{
			Model:     *model,
			Messages:  reqMessages,
			Think:     thinkBool(false),
			KeepAlive: &api.Duration{Duration: 5 * time.Minute},
			Options: map[string]any{
				"temperature": *temp,
				"num_ctx":     8192,
				"num_predict": 512,
			},
		}

		var assistant strings.Builder
		err = client.Chat(context.Background(), req, func(res api.ChatResponse) error {
			if res.Message.Content != "" {
				fmt.Print(res.Message.Content)
				assistant.WriteString(res.Message.Content)
			}
			if res.Done {
				fmt.Println()
			}
			return nil
		})
		if err != nil {
			fmt.Printf("\n[error] %v\n", err)
			messages = messages[:len(messages)-1]
			continue
		}

		messages = append(messages, api.Message{Role: "assistant", Content: assistant.String()})
		_ = saveMessages(sessPath, messages)
	}
}


func sessionPath(name string) string {
	home, _ := os.UserHomeDir()
	dir := filepath.Join(home, ".llama-go", "sessions")
	_ = os.MkdirAll(dir, 0o755)
	return filepath.Join(dir, name+".json")
}

func toSimple(msgs []api.Message) []simpleMsg {
	out := make([]simpleMsg, 0, len(msgs))
	for _, m := range msgs {
		out = append(out, simpleMsg{Role: m.Role, Content: m.Content})
	}
	return out
}

func fromSimple(s []simpleMsg) []api.Message {
	out := make([]api.Message, 0, len(s))
	for _, m := range s {
		out = append(out, api.Message{Role: m.Role, Content: m.Content})
	}
	return out
}

func saveMessages(path string, msgs []api.Message) error {
	data, _ := json.MarshalIndent(toSimple(msgs), "", "  ")
	return os.WriteFile(path, data, 0o644)
}

func loadMessages(path string) ([]api.Message, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	var sm []simpleMsg
	if err := json.Unmarshal(data, &sm); err != nil {
		return nil, err
	}
	return fromSimple(sm), nil
}

func printHistory(msgs []api.Message, last int) {
	start := 0
	if len(msgs) > 1+2*last {
		start = len(msgs) - 2*last
	}
	for i := start; i < len(msgs); i++ {
		m := msgs[i]
		if m.Role == "user" {
			fmt.Printf("\nYou: %s\n", m.Content)
		} else if m.Role == "assistant" {
			fmt.Printf("AI : %s\n", m.Content)
		}
	}
	fmt.Println()
}


func thinkBool(b bool) *api.ThinkValue {
	v := new(api.ThinkValue)
	bs, _ := json.Marshal(b)
	_ = v.UnmarshalJSON(bs)
	return v
}


func buildIndexFromPDF(client *api.Client, path, embedModel string) (*DocIndex, error) {
	f, r, err := pdf.Open(path)
	if f != nil {
		defer f.Close()
	}
	if err != nil {
		return nil, fmt.Errorf("open pdf: %w", err)
	}
	total := r.NumPage()
	if total > 5 {
		return nil, fmt.Errorf("PDF has %d pages; limit is 5", total)
	}
	if total == 0 {
		return nil, fmt.Errorf("empty PDF")
	}

	var chunks []DocChunk
	for p := 1; p <= total; p++ {
		page := r.Page(p)
		if page.V.IsNull() {
			continue
		}
		text, _ := page.GetPlainText(nil)
		if strings.TrimSpace(text) == "" {
			continue
		}
		for _, piece := range splitChunks(text, 1000, 200) {
			emb, err := embedOnce(client, embedModel, piece)
			if err != nil {
				return nil, fmt.Errorf("embed: %w", err)
			}
			chunks = append(chunks, DocChunk{Page: p, Text: piece, Embedding: emb})
		}
	}
	idx := &DocIndex{
		DocName:        filepath.Base(path),
		EmbeddingModel: embedModel,
		Chunks:         chunks,
		CreatedUnixSec: time.Now().Unix(),
	}
	return idx, nil
}

func splitChunks(s string, maxLen, overlap int) []string {
	s = strings.Join(strings.Fields(s), " ")
	if len(s) <= maxLen {
		return []string{s}
	}
	var out []string
	for start := 0; start < len(s); {
		end := start + maxLen
		if end > len(s) {
			end = len(s)
		}
		out = append(out, s[start:end])
		if end == len(s) {
			break
		}
		start = end - overlap
		if start < 0 {
			start = 0
		}
	}
	return out
}

func embedOnce(client *api.Client, model, text string) ([]float64, error) {
	url := "http://localhost:11434/api/embeddings"
	payload := map[string]interface{}{
		"model":  model,
		"prompt": text,
	}
	body, _ := json.Marshal(payload)
	resp, err := http.Post(url, "application/json", bytes.NewReader(body))
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()
	var result struct {
		Embedding []float64 `json:"embedding"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		return nil, err
	}
	return result.Embedding, nil
}


func buildRAGContext(client *api.Client, embedModel string, topK int, question string, idx *DocIndex) (api.Message, error) {
	qv, err := embedOnce(client, embedModel, question)
	if err != nil {
		return api.Message{}, err
	}
	type scored struct {
		i     int
		score float64
	}
	best := make([]scored, 0, len(idx.Chunks))
	for i, ch := range idx.Chunks {
		best = append(best, scored{i, cosine(qv, ch.Embedding)})
	}
	sort.Slice(best, func(i, j int) bool { return best[i].score > best[j].score })
	if topK > len(best) {
		topK = len(best)
	}
	best = best[:topK]

	var b strings.Builder
	fmt.Fprintf(&b, "PDF context from %s (use ONLY this to answer; if insufficient, say: \"I don't have that info in the PDF.\")\n\n", idx.DocName)
	for _, sc := range best {
		ch := idx.Chunks[sc.i]
		fmt.Fprintf(&b, "— Page %d —\n%s\n\n", ch.Page, ch.Text)
	}
	return api.Message{Role: "system", Content: b.String()}, nil
}

func cosine(a, b []float64) float64 {
	if len(a) != len(b) || len(a) == 0 {
		return 0
	}
	var dot, na, nb float64
	for i := range a {
		dot += a[i] * b[i]
		na += a[i] * a[i]
		nb += b[i] * b[i]
	}
	if na == 0 || nb == 0 {
		return 0
	}
	return dot / (math.Sqrt(na) * math.Sqrt(nb))
}

func indexPath(session string) string {
	home, _ := os.UserHomeDir()
	dir := filepath.Join(home, ".llama-go", "index")
	_ = os.MkdirAll(dir, 0o755)
	return filepath.Join(dir, session+".json")
}

func saveDocIndex(session string, idx *DocIndex) error {
	if idx == nil {
		return nil
	}
	data, _ := json.MarshalIndent(idx, "", "  ")
	return os.WriteFile(indexPath(session), data, 0o644)
}

func loadDocIndex(session string) (*DocIndex, error) {
	data, err := os.ReadFile(indexPath(session))
	if err != nil {
		return nil, err
	}
	var idx DocIndex
	if e := json.Unmarshal(data, &idx); e != nil {
		return nil, e
	}
	return &idx, nil
}

func removeDocIndex(session string) error {
	return os.Remove(indexPath(session))
}

func pagesInIndex(idx *DocIndex) int {
	seen := map[int]struct{}{}
	for _, c := range idx.Chunks {
		seen[c.Page] = struct{}{}
	}
	return len(seen)
}
