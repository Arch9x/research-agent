package tools

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"
	"time"
)

// SearchResult is a single result from the Exa search API.
type SearchResult struct {
	ID            string `json:"id"`
	Title         string `json:"title"`
	URL           string `json:"url"`
	PublishedDate string `json:"publishedDate"`
	Author        string `json:"author"`
	Text          string `json:"text"`
}

// ExaClient wraps the Exa search API.
type ExaClient struct {
	apiKey     string
	endpoint   string
	numResults int
	http       *http.Client
}

func NewExaClient(apiKey, endpoint string, numResults int) *ExaClient {
	return &ExaClient{
		apiKey:     apiKey,
		endpoint:   endpoint,
		numResults: numResults,
		http:       &http.Client{Timeout: 30 * time.Second},
	}
}

type exaRequest struct {
	Query      string         `json:"query"`
	Type       string         `json:"type"`
	NumResults int            `json:"numResults"`
	Contents   map[string]any `json:"contents"`
}

type exaResponse struct {
	Results []SearchResult `json:"results"`
}

// Search runs a query and returns results with text content.
func (e *ExaClient) Search(ctx context.Context, query string) ([]SearchResult, error) {
	payload, err := json.Marshal(exaRequest{
		Query:      query,
		Type:       "auto",
		NumResults: e.numResults,
		Contents:   map[string]any{"text": true},
	})
	if err != nil {
		return nil, fmt.Errorf("marshal exa request: %w", err)
	}

	req, err := http.NewRequestWithContext(ctx, http.MethodPost, e.endpoint, bytes.NewReader(payload))
	if err != nil {
		return nil, fmt.Errorf("create exa request: %w", err)
	}
	req.Header.Set("x-api-key", e.apiKey)
	req.Header.Set("Content-Type", "application/json")

	resp, err := e.http.Do(req)
	if err != nil {
		return nil, fmt.Errorf("exa search: %w", err)
	}
	defer resp.Body.Close()

	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, fmt.Errorf("read exa response: %w", err)
	}
	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("exa search status %d: %.300s", resp.StatusCode, body)
	}

	var parsed exaResponse
	if err := json.Unmarshal(body, &parsed); err != nil {
		return nil, fmt.Errorf("parse exa response: %w", err)
	}
	return parsed.Results, nil
}

// FetchPage downloads a page and returns its text content (best effort).
func FetchPage(ctx context.Context, url string) (string, error) {
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, url, nil)
	if err != nil {
		return "", fmt.Errorf("create fetch request: %w", err)
	}
	req.Header.Set("User-Agent", "Mozilla/5.0 (research-agent)")

	client := &http.Client{Timeout: 20 * time.Second}
	resp, err := client.Do(req)
	if err != nil {
		return "", fmt.Errorf("fetch page: %w", err)
	}
	defer resp.Body.Close()

	body, err := io.ReadAll(io.LimitReader(resp.Body, 2<<20))
	if err != nil {
		return "", fmt.Errorf("read page: %w", err)
	}
	if resp.StatusCode != http.StatusOK {
		return "", fmt.Errorf("fetch page status %d", resp.StatusCode)
	}

	text := stripHTML(string(body))
	text = strings.Join(strings.Fields(text), " ")
	if len(text) > 8000 {
		text = text[:8000]
	}
	return text, nil
}

// stripHTML removes tags and decodes common entities (best effort).
func stripHTML(s string) string {
	var b strings.Builder
	inTag := false
	for i := 0; i < len(s); i++ {
		c := s[i]
		switch {
		case c == '<':
			inTag = true
		case c == '>':
			inTag = false
		case !inTag:
			b.WriteByte(c)
		}
	}
	out := b.String()
	out = strings.ReplaceAll(out, "&nbsp;", " ")
	out = strings.ReplaceAll(out, "&amp;", "&")
	out = strings.ReplaceAll(out, "&lt;", "<")
	out = strings.ReplaceAll(out, "&gt;", ">")
	out = strings.ReplaceAll(out, "&quot;", `"`)
	out = strings.ReplaceAll(out, "&#39;", "'")
	return out
}
