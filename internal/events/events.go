package events

import (
	"encoding/json"
	"fmt"
	"sync"
)

// Event types sent to the browser over SSE.
const (
	TypeStatus    = "status"    // generic status line
	TypeSearch    = "search"    // agent started a search
	TypeRead      = "read"      // agent is reading a page
	TypeFinding   = "finding"   // a finding was added
	TypeQuestion  = "question"  // agent asks the user mid-research
	TypeReport    = "report"    // final report ready
	TypeError     = "error"     // non-fatal error
	TypeDone      = "done"      // research finished
	TypeClarify   = "clarify"   // clarifying questions phase
	TypeBrief     = "brief"     // research brief ready
)

type Event struct {
	Type    string `json:"type"`
	Message string `json:"message,omitempty"`
	Data    any    `json:"data,omitempty"`
}

// Hub fans out events to all connected SSE clients.
type Hub struct {
	mu      sync.Mutex
	clients map[chan []byte]struct{}
}

func NewHub() *Hub {
	return &Hub{clients: make(map[chan []byte]struct{})}
}

func (h *Hub) Subscribe() chan []byte {
	ch := make(chan []byte, 64)
	h.mu.Lock()
	h.clients[ch] = struct{}{}
	h.mu.Unlock()
	return ch
}

func (h *Hub) Unsubscribe(ch chan []byte) {
	h.mu.Lock()
	delete(h.clients, ch)
	h.mu.Unlock()
}

func (h *Hub) Publish(ev Event) {
	data, err := json.Marshal(ev)
	if err != nil {
		return
	}
	msg := fmt.Sprintf("data: %s\n\n", data)
	h.mu.Lock()
	defer h.mu.Unlock()
	for ch := range h.clients {
		select {
		case ch <- []byte(msg):
		default:
		}
	}
}

func (h *Hub) Status(msg string) {
	h.Publish(Event{Type: TypeStatus, Message: msg})
}

func (h *Hub) Search(query string) {
	h.Publish(Event{Type: TypeSearch, Message: query})
}

func (h *Hub) Read(url string) {
	h.Publish(Event{Type: TypeRead, Message: url})
}

func (h *Hub) Finding(msg string) {
	h.Publish(Event{Type: TypeFinding, Message: msg})
}

func (h *Hub) Question(q string) {
	h.Publish(Event{Type: TypeQuestion, Message: q})
}

func (h *Hub) Report(md string) {
	h.Publish(Event{Type: TypeReport, Data: md})
}

func (h *Hub) Error(msg string) {
	h.Publish(Event{Type: TypeError, Message: msg})
}

func (h *Hub) Done() {
	h.Publish(Event{Type: TypeDone})
}

func (h *Hub) Clarify(questions []string) {
	h.Publish(Event{Type: TypeClarify, Data: questions})
}

func (h *Hub) Brief(brief string) {
	h.Publish(Event{Type: TypeBrief, Message: brief})
}
