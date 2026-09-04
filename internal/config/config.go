package config

import (
	"fmt"
	"os"
	"strconv"
	"strings"
)

type Config struct {
	Port            string
	OpenAIKey       string
	OpenAIBaseURL   string
	Model           string
	ExaKey          string
	ExaEndpoint     string
	ExaNumResults   int
	MaxSearches     int
	MaxPagesRead    int
	MaxResearchTime int
	MinSources      int
	MinDomains      int
	MaxClarifyRounds int
	ReportsDir      string
}

func Load() (*Config, error) {
	cfg := &Config{
		Port:             getEnv("PORT", "8080"),
		OpenAIKey:        getEnv("OPENAI_API_KEY", ""),
		OpenAIBaseURL:    getEnv("OPENAI_BASE_URL", "https://ollama.com/v1"),
		Model:            getEnv("MODEL", "deepseek-v4-flash:0731"),
		ExaKey:           getEnv("EXA_API_KEY", ""),
		ExaEndpoint:      getEnv("EXA_ENDPOINT", "https://api.exa.ai/search"),
		ExaNumResults:    getInt("EXA_NUM_RESULTS", 8),
		MaxSearches:      getInt("MAX_SEARCHES", 15),
		MaxPagesRead:     getInt("MAX_PAGES_READ", 20),
		MaxResearchTime:  getInt("MAX_RESEARCH_TIME_SECONDS", 300),
		MinSources:       getInt("MIN_SOURCES", 3),
		MinDomains:       getInt("MIN_DOMAINS", 2),
		MaxClarifyRounds: getInt("MAX_CLARIFY_ROUNDS", 3),
		ReportsDir:       getEnv("REPORTS_DIR", "reports"),
	}

	if cfg.OpenAIKey == "" {
		return nil, fmt.Errorf("OPENAI_API_KEY is required")
	}
	if cfg.ExaKey == "" {
		return nil, fmt.Errorf("EXA_API_KEY is required")
	}
	return cfg, nil
}

func getEnv(key, def string) string {
	if v := strings.TrimSpace(os.Getenv(key)); v != "" {
		return v
	}
	return def
}

func getInt(key string, def int) int {
	if v := strings.TrimSpace(os.Getenv(key)); v != "" {
		if n, err := strconv.Atoi(v); err == nil {
			return n
		}
	}
	return def
}
