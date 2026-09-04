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
	TypeLines     = "lines"     // research decomposed into lines
)

type Event struct {
	Type    string `json:"type"`
	Message string `json:"message,omitempty"`
	Data    any    `json:"data,omitempty"`
}

// Hub fans out events to SSE clients subscribed to a specific session.
type Hub struct {
	mu      sync.Mutex
	clients map[string]map[chan []byte]struct{}
}

func NewHub() *Hub {
	return &Hub{clients: make(map[string]map[chan []byte]struct{})}
}

// Subscribe registers a channel for a session and returns it.
func (h *Hub) Subscribe(sessionID string) chan []byte {
	ch := make(chan []byte, 64)
	h.mu.Lock()
	if h.clients[sessionID] == nil {
		h.clients[sessionID] = make(map[chan []byte]struct{})
	}
	h.clients[sessionID][ch] = struct{}{}
	h.mu.Unlock()
	return ch
}

func (h *Hub) Unsubscribe(sessionID string, ch chan []byte) {
	h.mu.Lock()
	if m := h.clients[sessionID]; m != nil {
		delete(m, ch)
		if len(m) == 0 {
			delete(h.clients, sessionID)
		}
	}
	h.mu.Unlock()
}

// Publish sends an event only to clients subscribed to the given session.
func (h *Hub) Publish(sessionID string, ev Event) {
	data, err := json.Marshal(ev)
	if err != nil {
		return
	}
	msg := fmt.Sprintf("data: %s\n\n", data)
	h.mu.Lock()
	defer h.mu.Unlock()
	for ch := range h.clients[sessionID] {
		select {
		case ch <- []byte(msg):
		default:
		}
	}
}

func (h *Hub) Status(sessionID, msg string) {
	h.Publish(sessionID, Event{Type: TypeStatus, Message: msg})
}

func (h *Hub) Search(sessionID, query string) {
	h.Publish(sessionID, Event{Type: TypeSearch, Message: query})
}

func (h *Hub) Read(sessionID, url string) {
	h.Publish(sessionID, Event{Type: TypeRead, Message: url})
}

func (h *Hub) Finding(sessionID, msg string) {
	h.Publish(sessionID, Event{Type: TypeFinding, Message: msg})
}

func (h *Hub) Question(sessionID, q string) {
	h.Publish(sessionID, Event{Type: TypeQuestion, Message: q})
}

func (h *Hub) Report(sessionID, md string) {
	h.Publish(sessionID, Event{Type: TypeReport, Data: md})
}

func (h *Hub) Error(sessionID, msg string) {
	h.Publish(sessionID, Event{Type: TypeError, Message: msg})
}

func (h *Hub) Done(sessionID string) {
	h.Publish(sessionID, Event{Type: TypeDone})
}

func (h *Hub) Clarify(sessionID string, questions []string) {
	h.Publish(sessionID, Event{Type: TypeClarify, Data: questions})
}

func (h *Hub) Brief(sessionID, brief string) {
	h.Publish(sessionID, Event{Type: TypeBrief, Message: brief})
}

func (h *Hub) Lines(sessionID string, titles []string) {
	h.Publish(sessionID, Event{Type: TypeLines, Data: titles})
}
