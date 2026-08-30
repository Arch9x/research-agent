package main

import (
	"context"
	"crypto/rand"
	"encoding/hex"
	"encoding/json"
	"log"
	"net/http"
	"strings"
	"sync"
	"time"

	"research-agent/internal/agents"
	"research-agent/internal/config"
	"research-agent/internal/events"
	"research-agent/internal/llm"
	"research-agent/internal/research"
	"research-agent/internal/store"
	"research-agent/internal/tools"
)

type server struct {
	cfg    *config.Config
	hub    *events.Hub
	store  *store.Store
	llm    *llm.Client
	exa    *tools.ExaClient
	clar   *agents.Clarifier
	res    *agents.Researcher
	repor  *agents.Reporter

	mu       sync.Mutex
	sessions map[string]*research.Session
}

func main() {
	cfg, err := config.Load()
	if err != nil {
		log.Fatalf("config: %v", err)
	}

	st, err := store.New(cfg.ReportsDir)
	if err != nil {
		log.Fatalf("store: %v", err)
	}

	llmClient := llm.New(cfg.OpenAIKey, cfg.OpenAIBaseURL, cfg.Model)
	exa := tools.NewExaClient(cfg.ExaKey, cfg.ExaEndpoint, cfg.ExaNumResults)
	hub := events.NewHub()

	s := &server{
		cfg:      cfg,
		hub:      hub,
		store:    st,
		llm:      llmClient,
		exa:      exa,
		clar:     agents.NewClarifier(llmClient),
		res:      agents.NewResearcher(exa, hub, llmClient),
		repor:    agents.NewReporter(llmClient),
		sessions: make(map[string]*research.Session),
	}

	mux := http.NewServeMux()
	mux.HandleFunc("/", s.handleIndex)
	mux.HandleFunc("/api/start", s.handleStart)
	mux.HandleFunc("/api/answer", s.handleAnswer)
	mux.HandleFunc("/api/events", s.handleEvents)
	mux.HandleFunc("/api/reports", s.handleReports)
	mux.HandleFunc("/api/reports/", s.handleReport)

	addr := ":" + cfg.Port
	log.Printf("listening on %s", addr)
	if err := http.ListenAndServe(addr, mux); err != nil {
		log.Fatalf("server: %v", err)
	}
}

func (s *server) handleIndex(w http.ResponseWriter, r *http.Request) {
	if r.URL.Path != "/" {
		http.NotFound(w, r)
		return
	}
	w.Header().Set("Content-Type", "text/html; charset=utf-8")
	w.Write([]byte(indexHTML))
}

func (s *server) handleStart(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
		return
	}
	var req struct {
		Topic string `json:"topic"`
	}
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, "bad request", http.StatusBadRequest)
		return
	}
	req.Topic = strings.TrimSpace(req.Topic)
	if req.Topic == "" {
		http.Error(w, "topic is required", http.StatusBadRequest)
		return
	}

	id := newID()
	sess := research.NewSession(id, req.Topic,
		s.cfg.MaxSearches, s.cfg.MaxPagesRead,
		time.Duration(s.cfg.MaxResearchTime)*time.Second,
		s.cfg.MinSources, s.cfg.MinDomains)

	s.mu.Lock()
	s.sessions[id] = sess
	s.mu.Unlock()

	go s.runResearch(id, sess)

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]string{"id": id})
}

func (s *server) handleAnswer(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
		return
	}
	var req struct {
		ID   string `json:"id"`
		Text string `json:"text"`
	}
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, "bad request", http.StatusBadRequest)
		return
	}
	s.mu.Lock()
	sess := s.sessions[req.ID]
	s.mu.Unlock()
	if sess == nil {
		http.Error(w, "session not found", http.StatusNotFound)
		return
	}
	select {
	case sess.AnswerCh <- req.Text:
	default:
	}
	w.WriteHeader(http.StatusOK)
}

func (s *server) handleEvents(w http.ResponseWriter, r *http.Request) {
	flusher, ok := w.(http.Flusher)
	if !ok {
		http.Error(w, "streaming unsupported", http.StatusInternalServerError)
		return
	}
	w.Header().Set("Content-Type", "text/event-stream")
	w.Header().Set("Cache-Control", "no-cache")
	w.Header().Set("Connection", "keep-alive")

	ch := s.hub.Subscribe()
	defer s.hub.Unsubscribe(ch)

	ctx := r.Context()
	for {
		select {
		case <-ctx.Done():
			return
		case msg := <-ch:
			if _, err := w.Write(msg); err != nil {
				return
			}
			flusher.Flush()
		}
	}
}

func (s *server) handleReports(w http.ResponseWriter, r *http.Request) {
	reports, err := s.store.List()
	if err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(reports)
}

func (s *server) handleReport(w http.ResponseWriter, r *http.Request) {
	id := strings.TrimPrefix(r.URL.Path, "/api/reports/")
	id = strings.TrimSuffix(id, "/")
	if id == "" {
		http.NotFound(w, r)
		return
	}
	rep, err := s.store.Get(id)
	if err != nil {
		http.NotFound(w, r)
		return
	}
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(rep)
}

// runResearch orchestrates the full pipeline: clarify → brief → research → report.
func (s *server) runResearch(id string, sess *research.Session) {
	ctx, cancel := context.WithTimeout(context.Background(), time.Duration(s.cfg.MaxResearchTime)*time.Second+2*time.Minute)
	defer cancel()

	s.hub.Status("Начинаю исследование: " + sess.Topic)

	// Phase 1: clarifying questions (one round of 2-4 questions).
	history := []string{}
	questions, err := s.clar.Clarify(ctx, sess.Topic, history)
	if err != nil {
		s.hub.Error("Не удалось сформулировать вопросы: " + err.Error())
	} else if len(questions) > 0 {
		sess.Questions = questions
		s.hub.Clarify(questions)
		// Wait for answers.
		answers := make([]string, 0, len(questions))
		for range questions {
			select {
			case ans := <-sess.AnswerCh:
				answers = append(answers, ans)
			case <-ctx.Done():
				s.hub.Error("Таймаут ожидания ответов.")
				return
			}
		}
		sess.Answers = append(sess.Answers, answers...)
		for i, a := range answers {
			q := ""
			if i < len(questions) {
				q = questions[i]
			}
			history = append(history, "Вопрос: "+q+"\nОтвет: "+a)
		}
	}

	// Phase 2: build research brief from topic + answers.
	brief, err := s.buildBrief(ctx, sess.Topic, history)
	if err != nil {
		s.hub.Error("Не удалось составить бриф: " + err.Error())
		brief = sess.Topic
	}
	sess.Brief = brief
	s.hub.Brief(brief)

	// Phase 3: research loop.
	summary, err := s.res.Run(ctx, sess)
	if err != nil {
		s.hub.Error("Ошибка исследования: " + err.Error())
	}

	// Phase 4: report.
	report, err := s.repor.Write(ctx, sess)
	if err != nil {
		s.hub.Error("Ошибка написания отчёта: " + err.Error())
		report = "## Отчёт\n\nНе удалось написать отчёт: " + err.Error()
	}

	// Save report.
	rep := &store.Report{
		ID:        id,
		Topic:     sess.Topic,
		CreatedAt: time.Now(),
		Markdown:  report,
	}
	if err := s.store.Save(rep); err != nil {
		s.hub.Error("Не удалось сохранить отчёт: " + err.Error())
	}

	s.hub.Report(report)
	s.hub.Status("Исследование завершено. Итог: " + summary)
	s.hub.Done()
}

func (s *server) buildBrief(ctx context.Context, topic string, history []string) (string, error) {
	var sb strings.Builder
	sb.WriteString("Research topic: " + topic + "\n")
	if len(history) > 0 {
		sb.WriteString("\nUser answers:\n")
		for _, h := range history {
			sb.WriteString("- " + h + "\n")
		}
	}
	const briefPrompt = `You are a research planner. Turn the research topic and the user's answers into a concrete research brief that a researcher agent will follow. Be specific about what to investigate, what dimensions matter, and what to look for. Write in the same language as the topic. Respond with plain text, no JSON.`
	return s.llm.Chat(ctx, briefPrompt, sb.String())
}

func newID() string {
	b := make([]byte, 8)
	_, _ = rand.Read(b)
	return hex.EncodeToString(b)
}

const indexHTML = `<!DOCTYPE html>
<html lang="ru">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Research Agent</title>
<style>
  body { font-family: system-ui, sans-serif; max-width: 800px; margin: 0 auto; padding: 20px; background: #0f1115; color: #e6e6e6; }
  h1 { font-size: 1.5rem; }
  input[type=text] { width: 100%; padding: 10px; border-radius: 6px; border: 1px solid #333; background: #1a1d24; color: #e6e6e6; }
  button { padding: 10px 18px; border-radius: 6px; border: none; background: #4f8cff; color: #fff; cursor: pointer; }
  button:disabled { opacity: .5; cursor: default; }
  #feed { margin-top: 20px; }
  .ev { padding: 8px 12px; margin: 4px 0; border-radius: 6px; background: #1a1d24; border-left: 3px solid #4f8cff; }
  .ev.search { border-left-color: #ffb84f; }
  .ev.read { border-left-color: #7bd88f; }
  .ev.finding { border-left-color: #c58bff; }
  .ev.question { border-left-color: #ff6b6b; }
  .ev.error { border-left-color: #ff4444; }
  .ev.report { border-left-color: #4f8cff; background: #1d2433; }
  .ev .t { font-size: .75rem; color: #888; }
  #questions { margin-top: 20px; }
  .q { padding: 10px; margin: 6px 0; background: #1a1d24; border-radius: 6px; }
  #report { margin-top: 20px; white-space: pre-wrap; background: #1d2433; padding: 16px; border-radius: 8px; }
  #reports { margin-top: 20px; }
  #reports a { color: #4f8cff; display: block; margin: 4px 0; }
  .hidden { display: none; }
</style>
</head>
<body>
<h1>Research Agent</h1>
<div>
  <input type="text" id="topic" placeholder="Введите тему исследования, например: стоит ли переезжать с REST на gRPC">
  <button id="startBtn">Исследовать</button>
</div>
<div id="questions" class="hidden"></div>
<div id="feed"></div>
<div id="report" class="hidden"></div>
<div id="reports"></div>

<script>
const feed = document.getElementById('feed');
const reportEl = document.getElementById('report');
const questionsEl = document.getElementById('questions');
const startBtn = document.getElementById('startBtn');
const topicEl = document.getElementById('topic');
let sessionId = null;
let pendingQuestions = [];

function addEvent(type, message, data) {
  const div = document.createElement('div');
  div.className = 'ev ' + type;
  const t = document.createElement('div');
  t.className = 't';
  t.textContent = new Date().toLocaleTimeString();
  div.appendChild(t);
  const m = document.createElement('div');
  m.textContent = message || '';
  div.appendChild(m);
  if (data && type === 'report') {
    const r = document.createElement('div');
    r.textContent = data;
    div.appendChild(r);
  }
  feed.appendChild(div);
  feed.scrollTop = feed.scrollHeight;
}

function showQuestions(questions) {
  pendingQuestions = questions;
  questionsEl.innerHTML = '';
  questionsEl.classList.remove('hidden');
  questions.forEach((q, i) => {
    const div = document.createElement('div');
    div.className = 'q';
    div.textContent = (i+1) + '. ' + q;
    const input = document.createElement('input');
    input.type = 'text';
    input.placeholder = 'Ваш ответ...';
    input.dataset.idx = i;
    div.appendChild(input);
    questionsEl.appendChild(div);
  });
  const btn = document.createElement('button');
  btn.textContent = 'Ответить';
  btn.onclick = async () => {
    const inputs = questionsEl.querySelectorAll('input');
    for (const inp of inputs) {
      const ans = inp.value.trim();
      if (!ans) { alert('Ответьте на все вопросы'); return; }
      await fetch('/api/answer', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({id: sessionId, text: ans})
      });
    }
    questionsEl.classList.add('hidden');
    questionsEl.innerHTML = '';
  };
  questionsEl.appendChild(btn);
}

async function start() {
  const topic = topicEl.value.trim();
  if (!topic) return;
  startBtn.disabled = true;
  feed.innerHTML = '';
  reportEl.classList.add('hidden');
  const res = await fetch('/api/start', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({topic})
  });
  const data = await res.json();
  sessionId = data.id;
  connectSSE();
}

function connectSSE() {
  const es = new EventSource('/api/events');
  es.onmessage = (e) => {
    const ev = JSON.parse(e.data);
    switch (ev.type) {
      case 'clarify': showQuestions(ev.data); break;
      case 'question': addEvent('question', '❓ ' + ev.message); break;
      case 'report':
        reportEl.textContent = ev.data;
        reportEl.classList.remove('hidden');
        break;
      case 'done':
        es.close();
        startBtn.disabled = false;
        loadReports();
        break;
      default:
        addEvent(ev.type, ev.message, ev.data);
    }
  };
  es.onerror = () => { /* keep alive */ };
}

async function loadReports() {
  const res = await fetch('/api/reports');
  const reports = await res.json();
  const el = document.getElementById('reports');
  el.innerHTML = '<h3>Прошлые отчёты</h3>';
  reports.forEach(r => {
    const a = document.createElement('a');
    a.href = '#';
    a.textContent = new Date(r.created_at).toLocaleString() + ' — ' + r.topic;
    a.onclick = (e) => {
      e.preventDefault();
      reportEl.textContent = r.markdown;
      reportEl.classList.remove('hidden');
    };
    el.appendChild(a);
  });
}

startBtn.onclick = start;
topicEl.addEventListener('keydown', (e) => { if (e.key === 'Enter') start(); });
loadReports();
</script>
</body>
</html>`
