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

const researcherSystemPrompt = `You are a web research agent. You investigate a research brief by searching the web, reading pages, and collecting facts with their source URLs.

You have four tools:
1. search_tool(query) — search the web. Returns a list of results with title, url, and text.
2. read_tool(url) — read a specific page. Returns the page text.
3. add_finding_tool(fact, url) — record a factual finding tied to its source URL. Call this whenever you learn a concrete fact from a search result or a read page.
4. ask_user_tool(question) — ask the user a question when you hit a fork you cannot decide yourself. Use sparingly.

Rules:
- Work like a careful human researcher: start broad, then go narrow to fill gaps.
- After each search, decide what to read next. Read the most promising pages.
- Record findings as you go: each finding is a short factual statement tied to its source URL. Do NOT invent facts — only state what the source actually says. Use add_finding_tool for every important fact.
- If sources contradict each other, record BOTH sides as separate findings — do not smooth it over.
- Keep searching until you have enough distinct sources (at least 3 sources from at least 2 different domains) or until you hit your limits.
- If you have already read a page, do not read it again.
- When you have gathered enough, call finish_tool with a short summary of what you found and what you could NOT find out.

Respond by calling tools. Do not write prose outside tool calls.`

type researcherTools struct {
	exa   *tools.ExaClient
	hub   *events.Hub
	sess  *research.Session
	llm   *llm.Client
}

// Researcher runs the tool-calling research loop.
type Researcher struct {
	exa  *tools.ExaClient
	hub  *events.Hub
	llm  *llm.Client
}

func NewResearcher(exa *tools.ExaClient, hub *events.Hub, llm *llm.Client) *Researcher {
	return &Researcher{exa: exa, hub: hub, llm: llm}
}

// Run executes the research loop for a session until limits are hit or the
// agent finishes. Returns the final summary from the agent.
func (r *Researcher) Run(ctx context.Context, sess *research.Session) (string, error) {
	rt := &researcherTools{exa: r.exa, hub: r.hub, sess: sess, llm: r.llm}

	params := openai.ChatCompletionNewParams{
		Model: openai.ChatModel(r.llm.Model()),
		Messages: []openai.ChatCompletionMessageParamUnion{
			openai.SystemMessage(researcherSystemPrompt),
			openai.UserMessage(fmt.Sprintf("Research brief:\n%s", sess.Brief)),
		},
		Tools: []openai.ChatCompletionToolUnionParam{
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
				Name:        "ask_user_tool",
				Description: openai.String("Ask the user a question when you hit a fork you cannot decide yourself. Use sparingly."),
				Parameters: openai.FunctionParameters{
					"type": "object",
					"properties": map[string]any{
						"question": map[string]string{"type": "string", "description": "the question to ask the user"},
					},
					"required": []string{"question"},
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
		},
	}

	client := r.llm.RawClient()

	for {
		if sess.Expired() {
			r.hub.Status("⏱ Время исследования вышло, завершаю.")
			break
		}
		if sess.SearchesUsed() >= sess.MaxSearches {
			r.hub.Status("Достигнут лимит поисков, завершаю.")
			break
		}

		completion, err := client.Chat.Completions.New(ctx, params, option.WithJSONSet("thinking", map[string]string{"type": "disabled"}))
		if err != nil {
			return "", fmt.Errorf("researcher completion: %w", err)
		}
		if len(completion.Choices) == 0 {
			return "", fmt.Errorf("researcher: no choices")
		}
		msg := completion.Choices[0].Message
		params.Messages = append(params.Messages, msg.ToParam())

		if len(msg.ToolCalls) == 0 {
			// No tool calls: agent is done (or just talking). Treat as finish.
			r.hub.Status("Исследователь завершил работу.")
			return msg.Content, nil
		}

		for _, tc := range msg.ToolCalls {
			if tc.Function.Name == "finish_tool" {
				var args struct {
					Summary string `json:"summary"`
				}
				_ = json.Unmarshal([]byte(tc.Function.Arguments), &args)
				r.hub.Status("Исследователь завершил работу.")
				return args.Summary, nil
			}

			result, err := rt.execute(ctx, tc)
			if err != nil {
				r.hub.Error(fmt.Sprintf("Инструмент %s: %v", tc.Function.Name, err))
				result = fmt.Sprintf("error: %v", err)
			}
			params.Messages = append(params.Messages, openai.ToolMessage(result, tc.ID))
		}
	}

	return "Research stopped by limits.", nil
}

func (rt *researcherTools) execute(ctx context.Context, tc openai.ChatCompletionMessageToolCallUnion) (string, error) {
	switch tc.Function.Name {
	case "search_tool":
		var args struct {
			Query string `json:"query"`
		}
		if err := json.Unmarshal([]byte(tc.Function.Arguments), &args); err != nil {
			return "", fmt.Errorf("parse search args: %w", err)
		}
		rt.sess.IncSearch()
		rt.hub.Search(args.Query)

		results, err := rt.exa.Search(ctx, args.Query)
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
		if rt.sess.AlreadyRead(args.URL) {
			return "This page was already read. Choose a different page.", nil
		}
		if rt.sess.PagesRead() >= rt.sess.MaxPagesRead {
			return "Page read limit reached. Do not read more pages.", nil
		}
		rt.sess.MarkRead(args.URL)
		rt.hub.Read(args.URL)

		text, err := tools.FetchPage(ctx, args.URL)
		if err != nil {
			return "", err
		}
		return text, nil

	case "ask_user_tool":
		var args struct {
			Question string `json:"question"`
		}
		if err := json.Unmarshal([]byte(tc.Function.Arguments), &args); err != nil {
			return "", fmt.Errorf("parse ask args: %w", err)
		}
		rt.hub.Question(args.Question)
		ans, err := rt.sess.AskUser(ctx, args.Question)
		if err != nil {
			return "", err
		}
		return "User answered: " + ans, nil

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
		rt.sess.AddFinding(args.Fact, args.URL)
		rt.hub.Finding(args.Fact)
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
