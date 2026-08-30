package agents

import (
	"context"
	"fmt"
	"strings"

	"research-agent/internal/llm"
)

const clarifySystemPrompt = `You are a research scoping assistant. Your job is to ask the user 2-4 smart, specific clarifying questions about their research topic BEFORE any research begins.

Rules:
- Questions must be concrete and derived from the user's topic, NOT generic ("please clarify", "tell me more").
- Ask about the dimensions that actually matter for this topic: e.g. for "REST vs gRPC" — performance vs developer experience, team size, browser clients, language ecosystem.
- Do NOT ask about things the user already stated.
- Ask 2-4 questions maximum.
- If the topic is already specific enough that research can begin without questions, set "need_clarification" to false and leave "questions" empty.

Respond with JSON only, no prose, matching this schema:
{"need_clarification": true/false, "questions": ["q1", "q2", ...]}`

type ClarifyResult struct {
	NeedClarification bool     `json:"need_clarification"`
	Questions         []string `json:"questions"`
}

// Clarifier asks the user 2-4 smart clarifying questions before research.
type Clarifier struct {
	llm *llm.Client
}

func NewClarifier(llm *llm.Client) *Clarifier {
	return &Clarifier{llm: llm}
}

// Clarify returns the questions to ask, or nil if none are needed.
func (c *Clarifier) Clarify(ctx context.Context, topic string, history []string) ([]string, error) {
	var sb strings.Builder
	sb.WriteString("Research topic: ")
	sb.WriteString(topic)
	sb.WriteString("\n\n")
	if len(history) > 0 {
		sb.WriteString("Conversation so far:\n")
		for _, h := range history {
			sb.WriteString("- ")
			sb.WriteString(h)
			sb.WriteString("\n")
		}
	}

	var res ClarifyResult
	if err := c.llm.ChatJSON(ctx, clarifySystemPrompt, sb.String(), &res); err != nil {
		return nil, fmt.Errorf("clarify: %w", err)
	}
	if !res.NeedClarification {
		return nil, nil
	}
	// Trim and drop empties.
	var out []string
	for _, q := range res.Questions {
		q = strings.TrimSpace(q)
		if q != "" {
			out = append(out, q)
		}
	}
	return out, nil
}
