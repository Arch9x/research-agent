package llm

import (
	"context"
	"encoding/json"
	"fmt"

	"github.com/openai/openai-go/v3"
	"github.com/openai/openai-go/v3/option"
)

type Client struct {
	client *openai.Client
	model  string
}

func New(apiKey, baseURL, model string) *Client {
	opts := []option.RequestOption{
		option.WithAPIKey(apiKey),
	}
	if baseURL != "" {
		opts = append(opts, option.WithBaseURL(baseURL))
	}
	c := openai.NewClient(opts...)
	return &Client{
		client: &c,
		model:  model,
	}
}

// Model returns the configured model name.
func (c *Client) Model() string {
	return c.model
}

// RawClient exposes the underlying openai client for tool-calling loops.
func (c *Client) RawClient() *openai.Client {
	return c.client
}

// Chat sends a simple chat completion (no tools) and returns the text.
func (c *Client) Chat(ctx context.Context, system, user string) (string, error) {
	completion, err := c.client.Chat.Completions.New(ctx, openai.ChatCompletionNewParams{
		Model: openai.ChatModel(c.model),
		Messages: []openai.ChatCompletionMessageParamUnion{
			openai.SystemMessage(system),
			openai.UserMessage(user),
		},
	}, option.WithJSONSet("thinking", map[string]string{"type": "disabled"}))
	if err != nil {
		return "", fmt.Errorf("chat completion: %w", err)
	}
	if len(completion.Choices) == 0 {
		return "", fmt.Errorf("chat completion: no choices returned")
	}
	return completion.Choices[0].Message.Content, nil
}

// ChatJSON asks the model to produce JSON matching the given schema description
// and parses it into out. The schema is described in the prompt; we parse
// defensively (strip code fences if present).
func (c *Client) ChatJSON(ctx context.Context, system, user string, out any) error {
	text, err := c.Chat(ctx, system, user)
	if err != nil {
		return err
	}
	text = stripCodeFences(text)
	if err := json.Unmarshal([]byte(text), out); err != nil {
		return fmt.Errorf("parse JSON from model: %w; raw: %.500s", err, text)
	}
	return nil
}

func stripCodeFences(s string) string {
	s = trimSpace(s)
	if len(s) >= 7 && s[:3] == "```" {
		if idx := indexByte(s, '\n'); idx >= 0 {
			s = s[idx+1:]
		}
		if idx := lastIndex(s, "```"); idx >= 0 {
			s = s[:idx]
		}
	}
	return trimSpace(s)
}

func trimSpace(s string) string {
	start := 0
	for start < len(s) && (s[start] == ' ' || s[start] == '\t' || s[start] == '\n' || s[start] == '\r') {
		start++
	}
	end := len(s)
	for end > start && (s[end-1] == ' ' || s[end-1] == '\t' || s[end-1] == '\n' || s[end-1] == '\r') {
		end--
	}
	return s[start:end]
}

func indexByte(s string, b byte) int {
	for i := 0; i < len(s); i++ {
		if s[i] == b {
			return i
		}
	}
	return -1
}

func lastIndex(s, sub string) int {
	for i := len(s) - len(sub); i >= 0; i-- {
		if s[i:i+len(sub)] == sub {
			return i
		}
	}
	return -1
}
