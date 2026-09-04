package agents

import (
	"context"
	"encoding/json"
	"fmt"
	"strings"

	"github.com/openai/openai-go/v3"
	"github.com/openai/openai-go/v3/option"

	"research-agent/internal/events"
	"research-agent/internal/llm"
	"research-agent/internal/research"
	"research-agent/internal/tools"
)

const lineSystemPrompt = `You are a web research agent. You investigate ONE research line by searching the web, reading pages, and collecting facts with their source URLs.

Your research line:
%s

You have four tools:
1. search_tool(query) — search the web. Returns a list of results with title, url, and text.
2. read_tool(url) — read a specific page. Returns the page text.
3. add_finding_tool(fact, url) — record a factual finding tied to its source URL. Call this whenever you learn a concrete fact from a search result or a read page.
4. finish_tool(summary) — finish research. Provide a short summary of what you found and what you could NOT find out.

Rules:
- Investigate ONLY your research line. Do not drift into other topics.
- Work like a careful human researcher: start broad, then go narrow to fill gaps.
- Record findings as you go: each finding is a short factual statement tied to its source URL. Do NOT invent facts — only state what the source actually says. Use add_finding_tool for every important fact.
- If sources contradict each other, record BOTH sides as separate findings — do not smooth it over.
- If you have already read a page, do not read it again.
- When you have gathered enough, call finish_tool with a short summary of what you found and what you could NOT find out.

Respond by calling tools. Do not write prose outside tool calls.`

// LineResult is what a line sub-agent returns to the lead researcher.
type LineResult struct {
	Summary string
	Sources []string
	Err     error
}

// runLine runs one research line as a sub-agent. It writes findings into the
// shared session and returns a short summary plus the source URLs it used.
func runLine(ctx context.Context, sess *research.Session, line Line, llm *llm.Client, exa *tools.ExaClient, hub *events.Hub) LineResult {
	params := openai.ChatCompletionNewParams{
		Model: openai.ChatModel(llm.Model()),
		Messages: []openai.ChatCompletionMessageParamUnion{
			openai.SystemMessage(fmt.Sprintf(lineSystemPrompt, line.Question)),
		},
		Tools: lineTools(),
	}

	client := llm.RawClient()
	var sources []string

	for {
		if sess.Expired() {
			hub.Status(sess.ID, "⏱ Время исследования вышло, завершаю линию: "+line.Title)
			break
		}
		if sess.SearchesUsed() >= sess.MaxSearches {
			hub.Status(sess.ID, "Достигнут лимит поисков, завершаю линию: "+line.Title)
			break
		}

		completion, err := client.Chat.Completions.New(ctx, params, option.WithJSONSet("thinking", map[string]string{"type": "disabled"}))
		if err != nil {
			return LineResult{Err: fmt.Errorf("line %q completion: %w", line.Title, err)}
		}
		if len(completion.Choices) == 0 {
			return LineResult{Err: fmt.Errorf("line %q: no choices", line.Title)}
		}
		msg := completion.Choices[0].Message
		params.Messages = append(params.Messages, msg.ToParam())

		if len(msg.ToolCalls) == 0 {
			// No tool calls: agent is done (or just talking). Treat as finish.
			return LineResult{Summary: msg.Content, Sources: sources}
		}

		for _, tc := range msg.ToolCalls {
			if tc.Function.Name == "finish_tool" {
				var args struct {
					Summary string `json:"summary"`
				}
				_ = json.Unmarshal([]byte(tc.Function.Arguments), &args)
				return LineResult{Summary: args.Summary, Sources: sources}
			}

			result, err := executeLineTool(ctx, sess, tc, exa, hub, &sources)
			if err != nil {
				hub.Error(sess.ID, fmt.Sprintf("Инструмент %s: %v", tc.Function.Name, err))
				result = fmt.Sprintf("error: %v", err)
			}
			params.Messages = append(params.Messages, openai.ToolMessage(result, tc.ID))
		}
	}

	return LineResult{Summary: "Research stopped by limits.", Sources: sources}
}

// lineTools returns the tool definitions available to a line sub-agent.
func lineTools() []openai.ChatCompletionToolUnionParam {
	return []openai.ChatCompletionToolUnionParam{
		openai.ChatCompletionFunctionTool(openai.FunctionDefinitionParam{
			Name:        "search_tool",
			Description: openai.String("Search the web for information. Returns results with title, url, and text."),
			Parameters: openai.FunctionParameters{
				"type": "object",
				"properties": map[string]any{
					"query": map[string]string{"type": "string", "description": "the search query"},
				},
				"required": []string{"query"},
			},
		}),
		openai.ChatCompletionFunctionTool(openai.FunctionDefinitionParam{
			Name:        "read_tool",
			Description: openai.String("Read a specific web page. Returns the page text."),
			Parameters: openai.FunctionParameters{
				"type": "object",
				"properties": map[string]any{
					"url": map[string]string{"type": "string", "description": "the page URL to read"},
				},
				"required": []string{"url"},
			},
		}),
		openai.ChatCompletionFunctionTool(openai.FunctionDefinitionParam{
			Name:        "add_finding_tool",
			Description: openai.String("Record a factual finding tied to its source URL. Call this whenever you learn a concrete fact from a search result or a read page."),
			Parameters: openai.FunctionParameters{
				"type": "object",
				"properties": map[string]any{
					"fact": map[string]string{"type": "string", "description": "the factual statement"},
					"url":  map[string]string{"type": "string", "description": "the source URL this fact came from"},
				},
				"required": []string{"fact", "url"},
			},
		}),
		openai.ChatCompletionFunctionTool(openai.FunctionDefinitionParam{
			Name:        "finish_tool",
			Description: openai.String("Finish research. Provide a short summary of what you found and what you could NOT find out."),
			Parameters: openai.FunctionParameters{
				"type": "object",
				"properties": map[string]any{
					"summary": map[string]string{"type": "string", "description": "summary of findings and gaps"},
				},
				"required": []string{"summary"},
			},
		}),
	}
}

// executeLineTool runs a single tool call for a line sub-agent.
func executeLineTool(ctx context.Context, sess *research.Session, tc openai.ChatCompletionMessageToolCallUnion, exa *tools.ExaClient, hub *events.Hub, sources *[]string) (string, error) {
	switch tc.Function.Name {
	case "search_tool":
		var args struct {
			Query string `json:"query"`
		}
		if err := json.Unmarshal([]byte(tc.Function.Arguments), &args); err != nil {
			return "", fmt.Errorf("parse search args: %w", err)
		}
		sess.IncSearch()
		hub.Search(sess.ID, args.Query)

		results, err := exa.Search(ctx, args.Query)
		if err != nil {
			return "", err
		}
		var sb strings.Builder
		for i, res := range results {
			fmt.Fprintf(&sb, "[%d] %s\nURL: %s\n%s\n\n", i+1, res.Title, res.URL, truncate(res.Text, 1500))
		}
		return sb.String(), nil

	case "read_tool":
		var args struct {
			URL string `json:"url"`
		}
		if err := json.Unmarshal([]byte(tc.Function.Arguments), &args); err != nil {
			return "", fmt.Errorf("parse read args: %w", err)
		}
		if sess.AlreadyRead(args.URL) {
			return "This page was already read. Choose a different page.", nil
		}
		if sess.PagesRead() >= sess.MaxPagesRead {
			return "Page read limit reached. Do not read more pages.", nil
		}
		sess.MarkRead(args.URL)
		hub.Read(sess.ID, args.URL)

		text, err := tools.FetchPage(ctx, args.URL)
		if err != nil {
			return "", err
		}
		return text, nil

	case "add_finding_tool":
		var args struct {
			Fact string `json:"fact"`
			URL  string `json:"url"`
		}
		if err := json.Unmarshal([]byte(tc.Function.Arguments), &args); err != nil {
			return "", fmt.Errorf("parse finding args: %w", err)
		}
		args.Fact = strings.TrimSpace(args.Fact)
		args.URL = strings.TrimSpace(args.URL)
		if args.Fact == "" || args.URL == "" {
			return "error: both fact and url are required", nil
		}
		sess.AddFinding(args.Fact, args.URL)
		*sources = append(*sources, args.URL)
		hub.Finding(sess.ID, args.Fact)
		return "Finding recorded.", nil

	default:
		return "", fmt.Errorf("unknown tool %q", tc.Function.Name)
	}
}

func truncate(s string, n int) string {
	if len(s) <= n {
		return s
	}
	return s[:n] + "…"
}
