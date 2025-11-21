# Infomate - AI Chat with PDF RAG

A modern web application that combines AI chat capabilities with PDF-based Retrieval-Augmented Generation (RAG) using Go backend and React frontend.

## Features

- **AI Chat**: Powered by Ollama with multiple model support
- **PDF RAG**: Upload PDFs and ask questions about their content
- **Session Management**: Create and switch between different chat sessions
- **RAG Toggle**: Enable/disable RAG functionality
- **Model Selection**: Switch between different LLM models
- **Persistent History**: Chat history is saved and restored
- **Modern UI**: Clean, responsive React interface

## Prerequisites

- Go 1.19+
- Node.js 16+
- Ollama running locally

## Installation

### 1. Clone and Setup Backend

```bash
git clone https://github.com/Thanmaireddy03/Project--InfoMate.git
cd llama-go
go mod tidy
```

### 2. Setup Frontend

```bash
cd frontend
npm install
npm run build
cd ..
```

### 3. Start Ollama

Make sure Ollama is running locally:
```bash
ollama serve
```

Pull a model:
```bash
ollama pull gemma3:4b
```

## Usage

### Command Line Interface

Run the original CLI version:
```bash
go run main.go
```

### Web Interface

Run the HTTP server:
```bash
go run cmd/server/main.go
```

Open your browser to `http://localhost:8080`

### Environment Variables

- `OLLAMA_HOST` - Ollama server URL (default: localhost:11434)

## Development

### Frontend Development

```bash
cd frontend
npm start
```

### Backend Development

```bash
go run cmd/server/main.go -port 8080
```

## File Structure

```
infomate/
├── main.go              # CLI version
├── cmd/
│   └── server/
│       └── main.go      # HTTP server version
├── frontend/            # React frontend
│   ├── public/
│   ├── src/
│   └── package.json
├── go.mod
└── README.md
```
