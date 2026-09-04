package agents

import (
	"context"
	"fmt"
	"strings"

	"research-agent/internal/llm"
)

const plannerSystemPrompt = `You are a research planner. Your job is to break a research brief into 3-5 independent research lines (sub-questions) that can be investigated separately.

Rules:
- Produce 3-5 lines. Simple briefs get 3, complex ones get 5.
- Each line is a concrete sub-question that a researcher can investigate on its own.
- Lines must NOT overlap: each covers a distinct dimension of the topic, so separate researchers do not duplicate searches.
- Lines must be specific ("What real benchmarks compare gRPC vs REST latency and throughput?") not generic ("research the topic").
- Write in the same language as the brief.

Respond with JSON only, no prose, matching this schema:
{"lines": [{"title": "short label", "question": "the concrete sub-question to investigate"}]}`

// Line is a single research sub-question.
type Line struct {
	Title    string `json:"title"`
	Question string `json:"question"`
}

type planResult struct {
	Lines []Line `json:"lines"`
}

// Planner breaks a research brief into independent research lines.
type Planner struct {
	llm *llm.Client
}

func NewPlanner(llm *llm.Client) *Planner {
	return &Planner{llm: llm}
}

// Plan returns the research lines for a brief. On error or empty result it
// returns nil so the caller can fall back to a single line covering the whole
// brief.
func (p *Planner) Plan(ctx context.Context, brief string) ([]Line, error) {
	var res planResult
	if err := p.llm.ChatJSON(ctx, plannerSystemPrompt, brief, &res); err != nil {
		return nil, fmt.Errorf("plan: %w", err)
	}
	var out []Line
	for _, l := range res.Lines {
		l.Title = strings.TrimSpace(l.Title)
		l.Question = strings.TrimSpace(l.Question)
		if l.Question != "" {
			out = append(out, l)
		}
	}
	return out, nil
}
