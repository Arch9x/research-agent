package agents

import (
	"context"
	"encoding/json"
	"fmt"
	"strings"
	"sync"

	"github.com/openai/openai-go/v3"
	"github.com/openai/openai-go/v3/option"

	"research-agent/internal/events"
	"research-agent/internal/llm"
	"research-agent/internal/research"
	"research-agent/internal/tools"
)

const leadSystemPrompt = `You are the lead research agent. You coordinate a team of sub-agents, each investigating one research line. You do NOT search or read pages yourself.

You have two tools:
1. ask_user_tool(question) — ask the user a question when you hit a fork you cannot decide yourself. Use sparingly, before research lines start.
2. finish_tool(summary) — finish planning and start the research lines. Provide a short summary of the plan.

Rules:
- If the brief has a fork you cannot decide yourself, ask the user first.
- When you are ready, call finish_tool to start the research.

Respond by calling tools. Do not write prose outside tool calls.`

const synthesisSystemPrompt = `You are the lead research agent. Sub-agents have investigated several research lines and returned their summaries, and the collected findings (fact + source URL) are listed below. Synthesize them into one coherent picture.

Rules:
- Base your synthesis on BOTH the line summaries AND the collected findings. The findings are the ground truth — use them even if a line summary is thin or says it stopped by limits.
- Summarize what is known across all lines.
- Highlight contradictions explicitly: "X claims ..., but Y shows ...".
- List what could NOT be determined (gaps).
- Write in the same language as the research brief.
- Be concrete and factual. Do not add external knowledge.`

// Researcher runs the research: it plans lines, dispatches parallel
// sub-agents, and synthesizes their summaries.
type Researcher struct {
	exa     *tools.ExaClient
	hub     *events.Hub
	llm     *llm.Client
	planner *Planner
}

func NewResearcher(exa *tools.ExaClient, hub *events.Hub, llm *llm.Client, planner *Planner) *Researcher {
	return &Researcher{exa: exa, hub: hub, llm: llm, planner: planner}
}

// Run executes the research pipeline for a session: pre-planning questions,
// line planning, parallel line research, and synthesis. Returns the final
// summary.
func (r *Researcher) Run(ctx context.Context, sess *research.Session) (string, error) {
	// Phase 1: optional pre-planning question (ask_user_tool / finish_tool).
	if err := r.prePlan(ctx, sess); err != nil {
		return "", err
	}

	// Phase 2: plan research lines.
	lines, err := r.planner.Plan(ctx, sess.Brief)
	if err != nil {
		r.hub.Error(sess.ID, "Не удалось разбить на линии: "+err.Error())
	}
	if len(lines) == 0 {
		// Fallback: one line covering the whole brief.
		lines = []Line{{Title: sess.Topic, Question: sess.Brief}}
	}
	titles := make([]string, len(lines))
	for i, l := range lines {
		titles[i] = l.Title
	}
	r.hub.Lines(sess.ID, titles)

	// Phase 3: run all lines in parallel.
	results := make([]LineResult, len(lines))
	var wg sync.WaitGroup
	for i, line := range lines {
		wg.Add(1)
		go func(i int, line Line) {
			defer wg.Done()
			r.hub.Status(sess.ID, "Линия: "+line.Title)
			results[i] = runLine(ctx, sess, line, r.llm, r.exa, r.hub)
		}(i, line)
	}
	wg.Wait()

	// Phase 4: synthesize the line summaries.
	return r.synthesize(ctx, sess, lines, results)
}

// prePlan runs a short tool-calling loop where the lead agent may ask the user
// a question before research lines start.
func (r *Researcher) prePlan(ctx context.Context, sess *research.Session) error {
	params := openai.ChatCompletionNewParams{
		Model: openai.ChatModel(r.llm.Model()),
		Messages: []openai.ChatCompletionMessageParamUnion{
			openai.SystemMessage(leadSystemPrompt),
			openai.UserMessage(fmt.Sprintf("Research brief:\n%s", sess.Brief)),
		},
		Tools: []openai.ChatCompletionToolUnionParam{
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
				Description: openai.String("Finish planning and start the research lines."),
				Parameters: openai.FunctionParameters{
					"type": "object",
					"properties": map[string]any{
						"summary": map[string]string{"type": "string", "description": "short summary of the plan"},
					},
					"required": []string{"summary"},
				},
			}),
		},
	}

	client := r.llm.RawClient()
	for {
		if sess.Expired() {
			return nil
		}
		completion, err := client.Chat.Completions.New(ctx, params, option.WithJSONSet("thinking", map[string]string{"type": "disabled"}))
		if err != nil {
			return fmt.Errorf("lead completion: %w", err)
		}
		if len(completion.Choices) == 0 {
			return fmt.Errorf("lead: no choices")
		}
		msg := completion.Choices[0].Message
		params.Messages = append(params.Messages, msg.ToParam())

		if len(msg.ToolCalls) == 0 {
			return nil
		}
		for _, tc := range msg.ToolCalls {
			if tc.Function.Name == "finish_tool" {
				return nil
			}
			if tc.Function.Name == "ask_user_tool" {
				var args struct {
					Question string `json:"question"`
				}
				_ = json.Unmarshal([]byte(tc.Function.Arguments), &args)
				r.hub.Question(sess.ID, args.Question)
				ans, err := sess.AskUser(ctx, args.Question)
				if err != nil {
					return err
				}
				params.Messages = append(params.Messages, openai.ToolMessage("User answered: "+ans, tc.ID))
				continue
			}
			params.Messages = append(params.Messages, openai.ToolMessage(fmt.Sprintf("error: unknown tool %q", tc.Function.Name), tc.ID))
		}
	}
}

// synthesize combines all line summaries and collected findings into one
// coherent picture.
func (r *Researcher) synthesize(ctx context.Context, sess *research.Session, lines []Line, results []LineResult) (string, error) {
	var sb strings.Builder
	sb.WriteString("Research brief:\n")
	sb.WriteString(sess.Brief)
	sb.WriteString("\n\nLine summaries:\n")
	for i, res := range results {
		title := ""
		if i < len(lines) {
			title = lines[i].Title
		}
		sb.WriteString(fmt.Sprintf("--- Line %d: %s ---\n", i+1, title))
		if res.Err != nil {
			sb.WriteString("(line failed: " + res.Err.Error() + ")\n")
			continue
		}
		sb.WriteString(res.Summary)
		sb.WriteString("\n")
	}

	// Findings collected by the line sub-agents. These are the ground truth
	// even when a line stopped by limits and returned a thin summary.
	findings := sess.FindingsSnapshot()
	if len(findings) > 0 {
		sb.WriteString("\nCollected findings (fact — source URL):\n")
		for i, f := range findings {
			fmt.Fprintf(&sb, "%d. %s — %s\n", i+1, f.Fact, f.URL)
		}
	}

	synthesis, err := r.llm.Chat(ctx, synthesisSystemPrompt, sb.String())
	if err != nil {
		return "", fmt.Errorf("synthesis: %w", err)
	}
	sess.Synthesis = synthesis
	return synthesis, nil
}
