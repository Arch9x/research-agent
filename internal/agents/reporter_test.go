package agents

import (
	"strings"
	"testing"

	"research-agent/internal/research"
)

func TestVerifyLinksKeepsAllowed(t *testing.T) {
	report := "Вывод: X [Источник](https://example.com/a)."
	findings := []research.Finding{
		{Fact: "X", URL: "https://example.com/a"},
	}
	out := verifyLinks(report, findings)
	if !strings.Contains(out, "https://example.com/a") {
		t.Fatalf("allowed URL was stripped: %q", out)
	}
}

func TestVerifyLinksStripsFabricated(t *testing.T) {
	report := "Вывод: X [Источник](https://example.com/a) и [Выдумка](https://fake.com/b)."
	findings := []research.Finding{
		{Fact: "X", URL: "https://example.com/a"},
	}
	out := verifyLinks(report, findings)
	if strings.Contains(out, "https://fake.com/b") {
		t.Fatalf("fabricated URL was kept: %q", out)
	}
	if !strings.Contains(out, "https://example.com/a") {
		t.Fatalf("allowed URL was stripped: %q", out)
	}
}

func TestVerifyLinksNoLinks(t *testing.T) {
	report := "Просто текст без ссылок."
	out := verifyLinks(report, nil)
	if out != report {
		t.Fatalf("expected unchanged, got %q", out)
	}
}
