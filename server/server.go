//go:build server
// +build server

package main

import (
	"encoding/json"
	"fmt"
	"log"
	"net/http"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"time"

	"github.com/ollama/ollama/api"
)

type ChatRequest struct {
	Model        string  `json:"model"`
	Session      string  `json:"session"`
	Message      string  `json:"message"`
	Temperature  float64 `json:"temperature"`
	MaxContext   int     `json:"maxContext"`
	MaxPredict   int     `json:"maxPredict"`
	SystemPrompt string  `json:"systemPrompt"`
}

type ChatResponse struct {
	Response string `json:"response"`
	Error    string `json:"error,omitempty"`
}

type SessionData struct {
	Messages []api.Message `json:"messages"`
}

var (
	client *api.Client
	mu     sync.RWMutex
)

func main() {
	var err error
	client, err = api.ClientFromEnvironment()
	if err != nil {
		log.Fatal("Failed to create Ollama client:", err)
	}

	http.HandleFunc("/chat", chatHandler)
	http.HandleFunc("/session/", getSessionHandler)
	http.HandleFunc("/reset/", resetHandler)

	fmt.Println("Starting server on http://localhost:3000")
	fmt.Println("Open index.html in your browser")

	if err := http.ListenAndServe(":8080", nil); err != nil {
		log.Fatal("Server failed to start:", err)
	}
}

func chatHandler(w http.ResponseWriter, r *http.Request) {
	if r.Method == http.MethodOptions {
		w.Header().Set("Access-Control-Allow-Origin", "*")
		w.Header().Set("Access-Control-Allow-Methods", "POST, GET, OPTIONS")
		w.Header().Set("Access-Control-Allow-Headers", "Content-Type")
		w.WriteHeader(http.StatusOK)
		return
	}

	if r.Method != http.MethodPost {
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}

	var req ChatRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, "Invalid request", http.StatusBadRequest)
		return
	}

	// CORS headers
	w.Header().Set("Access-Control-Allow-Origin", "*")
	w.Header().Set("Access-Control-Allow-Methods", "POST, GET, OPTIONS")
	w.Header().Set("Access-Control-Allow-Headers", "Content-Type")
	w.Header().Set("Content-Type", "application/json")

	// Load session messages
	sessPath := sessionPathServer(req.Session)
	messages, err := loadMessagesServer(sessPath)
	if err != nil || len(messages) == 0 {
		messages = []api.Message{{Role: "system", Content: req.SystemPrompt}}
		saveMessagesServer(sessPath, messages)
	}

	// Add user message
	messages = append(messages, api.Message{Role: "user", Content: req.Message})

	// Create chat request
	chatReq := &api.ChatRequest{
		Model:     req.Model,
		Messages:  messages,
		Think:     thinkBoolServer(false),
		KeepAlive: &api.Duration{Duration: 5 * time.Minute},
		Options: map[string]any{
			"temperature": req.Temperature,
			"num_ctx":     req.MaxContext,
			"num_predict": req.MaxPredict,
		},
	}

	// Stream response
	var assistantResponse strings.Builder
	err = client.Chat(r.Context(), chatReq, func(res api.ChatResponse) error {
		if res.Message.Content != "" {
			assistantResponse.WriteString(res.Message.Content)
		}
		return nil
	})

	var response ChatResponse
	if err != nil {
		response.Error = err.Error()
		json.NewEncoder(w).Encode(response)
		return
	}

	// Save assistant response
	assistantText := assistantResponse.String()
	messages = append(messages, api.Message{Role: "assistant", Content: assistantText})
	saveMessagesServer(sessPath, messages)

	response.Response = assistantText
	json.NewEncoder(w).Encode(response)
}
