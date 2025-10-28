package main

import (
	"bufio"
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"log"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"time"

	"github.com/ollama/ollama/api"
)

type simpleMsg struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

func main() {
	model := flag.String("model", "gemma3:4b", "Ollama model tag")
	system := flag.String("system", "You are a concise assistant. Respond ONLY in English. "+
		"Do not greet, do not introduce yourself, and do not mention gemma3/OpenAI or any affiliations. "+
		"Answer directly and briefly unless the user asks for details.", "system prompt")
	temp := flag.Float64("temp", 0.6, "temperature")
	session := flag.String("session", "default", "session name (persists chat across runs)")
	flag.Parse()

	client, err := api.ClientFromEnvironment()
	if err != nil {
		log.Fatal(err)
	}

	sessPath := sessionPath(*session)

	messages, err := loadMessages(sessPath)
	if err != nil || len(messages) == 0 {
		messages = []api.Message{{Role: "system", Content: *system}}
		_ = saveMessages(sessPath, messages)
	}

	in := bufio.NewReader(os.Stdin)
	fmt.Printf("Model: %s  |  Session: %s\n", *model, *session)
	fmt.Println("Commands: /help  /reset  /history [n]  /model TAG  /save  /exit\n")

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
			fields := strings.Fields(user)
			if len(fields) == 2 {
				if v, e := strconv.Atoi(fields[1]); e == nil && v > 0 {
					n = v
				}
			}
			printHistory(messages, n)
			continue

		case strings.HasPrefix(user, "/model "):
			*model = strings.TrimSpace(strings.TrimPrefix(user, "/model "))
			fmt.Printf("(model set to %s)\n", *model)
			continue

		case user == "/help":
			fmt.Println(`/help        - show this help
/reset       - clear conversation context (keeps session file)
/history [n] - print last n exchanges
/model TAG   - switch model (e.g. deepseek-r1:8b)
/save        - persist now
/exit        - quit`)
			continue
		}

		messages = append(messages, api.Message{Role: "user", Content: user})

		req := &api.ChatRequest{
			Model:     *model,
			Messages:  messages,
			Think:     thinkBool(false),
			KeepAlive: &api.Duration{Duration: 5 * time.Minute},
			Options: map[string]any{
				"temperature": *temp,
				"num_ctx":     8192,
				"num_predict": 256,
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
	dir := filepath.Join(home, ".deepseek-go", "sessions")
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
	bs, _ := json.Marshal(b) // "true"/"false"
	_ = v.UnmarshalJSON(bs)
	return v
}

