package store

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"time"
)

// Report is a saved research report.
type Report struct {
	ID        string    `json:"id"`
	Topic     string    `json:"topic"`
	CreatedAt time.Time `json:"created_at"`
	Markdown  string    `json:"markdown"`
}

// Store persists reports as JSON files on disk.
type Store struct {
	dir string
}

func New(dir string) (*Store, error) {
	if err := os.MkdirAll(dir, 0o755); err != nil {
		return nil, fmt.Errorf("create reports dir: %w", err)
	}
	return &Store{dir: dir}, nil
}

func (s *Store) Save(r *Report) error {
	data, err := json.MarshalIndent(r, "", "  ")
	if err != nil {
		return fmt.Errorf("marshal report: %w", err)
	}
	path := filepath.Join(s.dir, r.ID+".json")
	if err := os.WriteFile(path, data, 0o644); err != nil {
		return fmt.Errorf("write report: %w", err)
	}
	return nil
}

func (s *Store) Get(id string) (*Report, error) {
	path := filepath.Join(s.dir, id+".json")
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("read report: %w", err)
	}
	var r Report
	if err := json.Unmarshal(data, &r); err != nil {
		return nil, fmt.Errorf("parse report: %w", err)
	}
	return &r, nil
}

func (s *Store) List() ([]Report, error) {
	entries, err := os.ReadDir(s.dir)
	if err != nil {
		return nil, fmt.Errorf("list reports: %w", err)
	}
	var reports []Report
	for _, e := range entries {
		if e.IsDir() || !strings.HasSuffix(e.Name(), ".json") {
			continue
		}
		data, err := os.ReadFile(filepath.Join(s.dir, e.Name()))
		if err != nil {
			continue
		}
		var r Report
		if err := json.Unmarshal(data, &r); err != nil {
			continue
		}
		reports = append(reports, r)
	}
	sort.Slice(reports, func(i, j int) bool {
		return reports[i].CreatedAt.After(reports[j].CreatedAt)
	})
	return reports, nil
}
