package research

import (
	"context"
	"fmt"
	"strings"
	"sync"
	"time"
)

// Finding is a single fact collected during research, tied to a source URL.
type Finding struct {
	Fact string `json:"fact"`
	URL  string `json:"url"`
}

// Session holds the state of one research run.
type Session struct {
	ID        string
	Topic     string
	Brief     string
	Questions []string
	Answers   []string

	mu         sync.Mutex
	Findings   []Finding
	ReadURLs   map[string]bool
	SearchCount int
	ReadCount   int
	StartTime   time.Time

	// Limits
	MaxSearches     int
	MaxPagesRead    int
	MaxTime         time.Duration
	MinSources      int
	MinDomains      int

	// Mid-research question handling
	PendingQuestion string
	AnswerCh        chan string
}

func NewSession(id, topic string, maxSearches, maxPagesRead int, maxTime time.Duration, minSources, minDomains int) *Session {
	return &Session{
		ID:          id,
		Topic:       topic,
		ReadURLs:    make(map[string]bool),
		StartTime:   time.Now(),
		MaxSearches: maxSearches,
		MaxPagesRead: maxPagesRead,
		MaxTime:     maxTime,
		MinSources:  minSources,
		MinDomains:  minDomains,
		AnswerCh:    make(chan string, 1),
	}
}

func (s *Session) AddFinding(fact, url string) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.Findings = append(s.Findings, Finding{Fact: fact, URL: url})
}

func (s *Session) MarkRead(url string) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.ReadURLs[url] = true
	s.ReadCount++
}

func (s *Session) AlreadyRead(url string) bool {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.ReadURLs[url]
}

func (s *Session) IncSearch() {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.SearchCount++
}

func (s *Session) SearchesUsed() int {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.SearchCount
}

func (s *Session) PagesRead() int {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.ReadCount
}

func (s *Session) TimeLeft() time.Duration {
	return s.MaxTime - time.Since(s.StartTime)
}

func (s *Session) Expired() bool {
	return time.Since(s.StartTime) >= s.MaxTime
}

// DistinctSources returns the number of distinct source URLs collected.
func (s *Session) DistinctSources() int {
	s.mu.Lock()
	defer s.mu.Unlock()
	seen := make(map[string]bool)
	for _, f := range s.Findings {
		seen[f.URL] = true
	}
	return len(seen)
}

// DistinctDomains returns the number of distinct domains among sources.
func (s *Session) DistinctDomains() int {
	s.mu.Lock()
	defer s.mu.Unlock()
	seen := make(map[string]bool)
	for _, f := range s.Findings {
		if d := domainOf(f.URL); d != "" {
			seen[d] = true
		}
	}
	return len(seen)
}

// ThinReport reports whether the collected material is too thin to write a
// proper report (per TZ: "тощий отчёт — не отчёт").
func (s *Session) ThinReport() bool {
	return s.DistinctSources() < s.MinSources || s.DistinctDomains() < s.MinDomains
}

func (s *Session) FindingsSnapshot() []Finding {
	s.mu.Lock()
	defer s.mu.Unlock()
	out := make([]Finding, len(s.Findings))
	copy(out, s.Findings)
	return out
}

// AskUser blocks until the user answers a mid-research question, or the
// context is cancelled.
func (s *Session) AskUser(ctx context.Context, question string) (string, error) {
	s.mu.Lock()
	s.PendingQuestion = question
	s.mu.Unlock()

	select {
	case ans := <-s.AnswerCh:
		s.mu.Lock()
		s.PendingQuestion = ""
		s.mu.Unlock()
		return ans, nil
	case <-ctx.Done():
		return "", ctx.Err()
	}
}

func domainOf(u string) string {
	u = strings.TrimPrefix(u, "https://")
	u = strings.TrimPrefix(u, "http://")
	u = strings.TrimPrefix(u, "www.")
	idx := strings.IndexAny(u, "/?#")
	if idx >= 0 {
		u = u[:idx]
	}
	return u
}

func (s *Session) String() string {
	return fmt.Sprintf("session %s topic=%q searches=%d pages=%d findings=%d",
		s.ID, s.Topic, s.SearchCount, s.ReadCount, len(s.Findings))
}
