package agents

import (
	"context"
	"fmt"
	"strings"

	"research-agent/internal/llm"
	"research-agent/internal/research"
)

const reporterSystemPrompt = `You are a research report writer. You write a final report based ONLY on the findings provided to you. You have no access to the internet and no tools.

Rules:
- The report's conclusion must be the FIRST line, not buried after an introduction.
- Every serious claim must be followed by a citation to its source URL in the form [Title](URL). Use ONLY URLs from the provided findings. Never invent or guess a URL.
- If sources contradict each other, show the contradiction explicitly: "X claims ..., but Y shows ...". Do not smooth it over.
- End with a section "## Что выяснить не удалось" listing honestly what could not be determined from the findings.
- If the findings are too thin to write a proper report, say so honestly instead of padding.
- Write in the same language as the research brief.
- Use Markdown. Be concrete and factual. Do not add external knowledge.`

// Reporter writes the final report from collected findings only.
type Reporter struct {
	llm *llm.Client
}

func NewReporter(llm *llm.Client) *Reporter {
	return &Reporter{llm: llm}
}

// Write produces the report markdown from the session's findings.
func (r *Reporter) Write(ctx context.Context, sess *research.Session) (string, error) {
	findings := sess.FindingsSnapshot()
	if len(findings) == 0 {
		return "## Отчёт\n\nНе удалось собрать ни одного источника по этой теме. Тема не раскрыта.", nil
	}

	var sb strings.Builder
	sb.WriteString("## Бриф\n\n")
	sb.WriteString(sess.Brief)
	sb.WriteString("\n\n## Находки\n\n")
	for i, f := range findings {
		fmt.Fprintf(&sb, "%d. %s — %s\n", i+1, f.Fact, f.URL)
	}

	report, err := r.llm.Chat(ctx, reporterSystemPrompt, sb.String())
	if err != nil {
		return "", fmt.Errorf("reporter: %w", err)
	}
	return verifyLinks(report, findings), nil
}

// verifyLinks makes it physically impossible for the report to contain a
// fabricated URL: every URL in the report must appear in the findings list.
// Any URL not in the list is stripped from the report.
func verifyLinks(report string, findings []research.Finding) string {
	allowed := make(map[string]bool)
	for _, f := range findings {
		allowed[f.URL] = true
	}
	var out strings.Builder
	rest := report
	for {
		open := strings.Index(rest, "](")
		if open < 0 {
			out.WriteString(rest)
			break
		}
		out.WriteString(rest[:open+1]) // up to and including "]"
		rest = rest[open+1:]          // rest starts with "("
		close := strings.Index(rest, ")")
		if close < 0 {
			out.WriteString(rest)
			break
		}
		url := rest[1:close]
		if allowed[url] {
			out.WriteString(rest[:close+1])
		}
		rest = rest[close+1:]
	}
	return out.String()
}
