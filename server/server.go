package main

import (
	"bytes"
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"log"
	"math"
	"net/http"
	"os"
	"path/filepath"
	"sort"
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

type ChatRequest struct {
	Session    string `json:"session"`
	Message    string `json:"message"`
	Model      string `json:"model"`
	RagEnabled bool   `json:"ragEnabled"`
}

type ChatResponse struct {
	Response string `json:"response"`
}

type SessionRequest struct {
	Name string `json:"name"`
}

type RAGRequest struct {
	Session string `json:"session"`
	Enabled bool   `json:"enabled"`
}

type ModelRequest struct {
	Session string `json:"session"`
	Model   string `json:"model"`
}

type ContextResponse struct {
	RagEnabled bool   `json:"ragEnabled"`
	Model      string `json:"model"`
	PdfName    string `json:"pdfName"`
	Chunks     int    `json:"chunks"`
}

type SessionsResponse struct {
	Sessions []string `json:"sessions"`
}

var (
	client     *api.Client
	docIndexes = make(map[string]*DocIndex)
	ragStates  = make(map[string]bool)
	models     = make(map[string]string)
)

func main() {
	port := flag.String("port", "8080", "HTTP server port")
	flag.Parse()

	var err error
	client, err = api.ClientFromEnvironment()
	if err != nil {
		log.Fatal(err)
	}

	// Initialize default values
	ragStates["default"] = true
	models["default"] = "gemma3:4b"

	http.HandleFunc("/api/chat", handleChat)
	http.HandleFunc("/api/sessions", handleSessions)
	http.HandleFunc("/api/context", handleContext)
	http.HandleFunc("/api/rag", handleRAG)
	http.HandleFunc("/api/model", handleModel)
	http.HandleFunc("/api/pdf", handlePDF)
	http.HandleFunc("/api/history", handleHistory)
	http.HandleFunc("/api/clear", handleClear)

	// Serve static files
	buildDir := "./frontend/build/"
	if _, err := os.Stat(buildDir); os.IsNotExist(err) {
		log.Fatalf("Frontend build directory not found: %s. Please run 'npm run build' in the frontend directory first.", buildDir)
	}
	http.Handle("/", http.FileServer(http.Dir(buildDir)))

	fmt.Printf("Server starting on port %s\n", *port)
	fmt.Printf("Serving frontend from: %s\n", buildDir)
	log.Fatal(http.ListenAndServe(":"+*port, nil))
}

func handleChat(w http.ResponseWriter, r *http.Request) {
	// Add CORS headers
	w.Header().Set("Access-Control-Allow-Origin", "*")
	w.Header().Set("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
	w.Header().Set("Access-Control-Allow-Headers", "Content-Type")

	if r.Method == "OPTIONS" {
		w.WriteHeader(http.StatusOK)
		return
	}

	if r.Method != http.MethodPost {
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}

	var req ChatRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, "Invalid JSON", http.StatusBadRequest)
		return
	}

	// Load session messages
	sessPath := sessionPath(req.Session)
	messages, err := loadMessages(sessPath)
	if err != nil || len(messages) == 0 {
		system := "You are a concise assistant. Respond ONLY in English."
		messages = []api.Message{{Role: "system", Content: system}}
		_ = saveMessages(sessPath, messages)
	}

	// Add user message
	messages = append(messages, api.Message{Role: "user", Content: req.Message})

	// Inject RAG context if enabled
	reqMessages := messages
	if req.RagEnabled {
		if docIndex, exists := docIndexes[req.Session]; exists && len(docIndex.Chunks) > 0 {
			ctxMsg, err := buildRAGContext(client, "nomic-embed-text", 4, req.Message, docIndex)
			if err == nil && ctxMsg.Content != "" {
				reqMessages = make([]api.Message, 0, len(messages)+1)
				reqMessages = append(reqMessages, messages[0])     // system
				reqMessages = append(reqMessages, ctxMsg)          // PDF context
				reqMessages = append(reqMessages, messages[1:]...) // rest
			}
		}
	}

	// Call Ollama
	fmt.Printf("Using model: %s for session: %s\n", req.Model, req.Session)
	chatReq := &api.ChatRequest{
		Model:    req.Model,
		Messages: reqMessages,
		Options: map[string]any{
			"temperature": 0.6,
			"num_ctx":     8192,
			"num_predict": 512,
		},
	}

	var response strings.Builder
	err = client.Chat(context.Background(), chatReq, func(res api.ChatResponse) error {
		if res.Message.Content != "" {
			response.WriteString(res.Message.Content)
		}
		return nil
	})

	if err != nil {
		http.Error(w, "Chat error: "+err.Error(), http.StatusInternalServerError)
		return
	}

	// Save assistant message
	messages = append(messages, api.Message{Role: "assistant", Content: response.String()})
	_ = saveMessages(sessPath, messages)

	// Return response
	resp := ChatResponse{Response: response.String()}
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(resp)
}

func handleSessions(w http.ResponseWriter, r *http.Request) {
	// Add CORS headers
	w.Header().Set("Access-Control-Allow-Origin", "*")
	w.Header().Set("Access-Control-Allow-Methods", "GET, POST, DELETE, OPTIONS")
	w.Header().Set("Access-Control-Allow-Headers", "Content-Type")

	if r.Method == "OPTIONS" {
		w.WriteHeader(http.StatusOK)
		return
	}

	if r.Method == http.MethodGet {
		// List sessions
		home, _ := os.UserHomeDir()
		sessionsDir := filepath.Join(home, ".llama-go", "sessions")
		files, _ := filepath.Glob(filepath.Join(sessionsDir, "*.json"))

		sessions := []string{"default"}
		for _, file := range files {
			name := strings.TrimSuffix(filepath.Base(file), ".json")
			if name != "default" {
				sessions = append(sessions, name)
			}
		}

		resp := SessionsResponse{Sessions: sessions}
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(resp)
	} else if r.Method == http.MethodPost {
		// Create session
		var req SessionRequest
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			http.Error(w, "Invalid JSON", http.StatusBadRequest)
			return
		}

		// Initialize session
		ragStates[req.Name] = true
		models[req.Name] = "gemma3:4b"

		w.WriteHeader(http.StatusCreated)
		json.NewEncoder(w).Encode(map[string]string{"status": "created"})
	} else if r.Method == http.MethodDelete {
		// Delete session
		session := r.URL.Query().Get("name")
		if session == "" || session == "default" {
			writeJSONError(w, http.StatusBadRequest, "Cannot delete default session")
			return
		}

		// Delete session file
		sessPath := sessionPath(session)
		_ = os.Remove(sessPath)

		// Delete index file
		_ = os.Remove(indexPath(session))

		// Remove from memory
		delete(ragStates, session)
		delete(models, session)
		delete(docIndexes, session)

		w.WriteHeader(http.StatusOK)
		json.NewEncoder(w).Encode(map[string]string{"status": "deleted"})
	}
}

func handleContext(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}

	session := r.URL.Query().Get("session")
	if session == "" {
		session = "default"
	}

	ragEnabled := ragStates[session]
	model := models[session]

	pdfName := ""
	chunks := 0
	if docIndex, exists := docIndexes[session]; exists {
		pdfName = docIndex.DocName
		chunks = len(docIndex.Chunks)
	}

	resp := ContextResponse{
		RagEnabled: ragEnabled,
		Model:      model,
		PdfName:    pdfName,
		Chunks:     chunks,
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(resp)
}

func handleRAG(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}

	var req RAGRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, "Invalid JSON", http.StatusBadRequest)
		return
	}

	ragStates[req.Session] = req.Enabled
	w.WriteHeader(http.StatusOK)
	json.NewEncoder(w).Encode(map[string]string{"status": "updated"})
}

func handleModel(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}

	var req ModelRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, "Invalid JSON", http.StatusBadRequest)
		return
	}

	models[req.Session] = req.Model
	fmt.Printf("Model changed to: %s for session: %s\n", req.Model, req.Session)
	w.WriteHeader(http.StatusOK)
	json.NewEncoder(w).Encode(map[string]string{"status": "updated"})
}

func handlePDF(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		writeJSONError(w, http.StatusMethodNotAllowed, "Method not allowed")
		return
	}

	// Parse multipart form
	err := r.ParseMultipartForm(32 << 20) // 32 MB max
	if err != nil {
		writeJSONError(w, http.StatusBadRequest, "Failed to parse form")
		return
	}

	file, _, err := r.FormFile("file")
	if err != nil {
		writeJSONError(w, http.StatusBadRequest, "No file uploaded")
		return
	}
	defer file.Close()

	session := r.FormValue("session")
	if session == "" {
		session = "default"
	}

	// Save uploaded file temporarily
	tempFile, err := os.CreateTemp("", "upload-*.pdf")
	if err != nil {
		writeJSONError(w, http.StatusInternalServerError, "Failed to create temp file")
		return
	}
	defer os.Remove(tempFile.Name())

	_, err = io.Copy(tempFile, file)
	if err != nil {
		writeJSONError(w, http.StatusInternalServerError, "Failed to save file")
		return
	}
	tempFile.Close()

	// Build index from PDF
	docIndex, err := buildIndexFromPDF(client, tempFile.Name(), "nomic-embed-text")
	if err != nil {
		writeJSONError(w, http.StatusBadRequest, "Failed to process PDF: "+err.Error())
		return
	}

	// Save index
	docIndexes[session] = docIndex
	_ = saveDocIndex(session, docIndex)
	ragStates[session] = true

	resp := map[string]string{
		"message": fmt.Sprintf("PDF processed: %d chunks from %d pages", len(docIndex.Chunks), pagesInIndex(docIndex)),
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(resp)
}

func handleHistory(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Access-Control-Allow-Origin", "*")
	if r.Method != http.MethodGet {
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}

	session := r.URL.Query().Get("session")
	if session == "" {
		session = "default"
	}

	messages, err := loadMessages(sessionPath(session))
	if err != nil {
		messages = []api.Message{}
	}

	// Convert to simple format for JSON
	simpleMsgs := toSimple(messages)
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]interface{}{"messages": simpleMsgs})
}

func handleClear(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Access-Control-Allow-Origin", "*")
	w.Header().Set("Access-Control-Allow-Methods", "POST, OPTIONS")
	if r.Method == "OPTIONS" {
		w.WriteHeader(http.StatusOK)
		return
	}
	if r.Method != http.MethodPost {
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}

	var req struct {
		Session string `json:"session"`
	}
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		writeJSONError(w, http.StatusBadRequest, "Invalid JSON")
		return
	}

	session := req.Session
	if session == "" {
		session = "default"
	}

	// Clear messages but keep system message
	sessPath := sessionPath(session)
	system := "You are a concise assistant. Respond ONLY in English."
	messages := []api.Message{{Role: "system", Content: system}}
	_ = saveMessages(sessPath, messages)

	w.WriteHeader(http.StatusOK)
	json.NewEncoder(w).Encode(map[string]string{"status": "cleared"})
}

func writeJSONError(w http.ResponseWriter, status int, message string) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	_ = json.NewEncoder(w).Encode(map[string]string{"error": message})
}

// Helper functions
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

func pagesInIndex(idx *DocIndex) int {
	seen := map[int]struct{}{}
	for _, c := range idx.Chunks {
		seen[c.Page] = struct{}{}
	}
	return len(seen)
}
