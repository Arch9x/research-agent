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
	sid := r.URL.Query().Get("session")
	if sid == "" {
		http.Error(w, "session query param required", http.StatusBadRequest)
		return
	}
	w.Header().Set("Content-Type", "text/event-stream")
	w.Header().Set("Cache-Control", "no-cache")
	w.Header().Set("Connection", "keep-alive")

	ch := s.hub.Subscribe(sid)
	defer s.hub.Unsubscribe(sid, ch)

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

	s.hub.Status(id, "Начинаю исследование: "+sess.Topic)

	// Phase 1: clarifying questions (one round of 2-4 questions).
	history := []string{}
	questions, err := s.clar.Clarify(ctx, sess.Topic, history)
	if err != nil {
		s.hub.Error(id, "Не удалось сформулировать вопросы: "+err.Error())
	} else if len(questions) > 0 {
		sess.Questions = questions
		s.hub.Clarify(id, questions)
		// Wait for answers.
		answers := make([]string, 0, len(questions))
		for range questions {
			select {
			case ans := <-sess.AnswerCh:
				answers = append(answers, ans)
			case <-ctx.Done():
				s.hub.Error(id, "Таймаут ожидания ответов.")
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
		s.hub.Error(id, "Не удалось составить бриф: "+err.Error())
		brief = sess.Topic
	}
	sess.Brief = brief
	s.hub.Brief(id, brief)

	// Phase 3: research loop.
	summary, err := s.res.Run(ctx, sess)
	if err != nil {
		s.hub.Error(id, "Ошибка исследования: "+err.Error())
	}

	// Phase 4: report.
	report, err := s.repor.Write(ctx, sess)
	if err != nil {
		s.hub.Error(id, "Ошибка написания отчёта: "+err.Error())
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
		s.hub.Error(id, "Не удалось сохранить отчёт: "+err.Error())
	}

	s.hub.Report(id, report)
	s.hub.Status(id, "Исследование завершено. Итог: "+summary)
	s.hub.Done(id)
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
<script src="https://cdn.jsdelivr.net/npm/marked@12.0.2/marked.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/dompurify@3.1.6/dist/purify.min.js"></script>
<style>
  * { box-sizing: border-box; }
  body { margin: 0; font-family: system-ui, -apple-system, sans-serif; background: #0f1115; color: #e6e6e6; height: 100vh; display: flex; }
  #sidebar { width: 280px; min-width: 280px; background: #14171d; border-right: 1px solid #23272f; display: flex; flex-direction: column; }
  #sidebar h1 { font-size: 1.1rem; margin: 16px 16px 8px; }
  #sidebar .sub { font-size: .75rem; color: #888; margin: 0 16px 12px; }
  #newBtn { margin: 0 16px 12px; padding: 10px; border-radius: 8px; border: none; background: #4f8cff; color: #fff; cursor: pointer; font-size: .9rem; }
  #newBtn:hover { background: #3d7bf0; }
  #reports { flex: 1; overflow-y: auto; padding: 0 8px 16px; }
  #reports h3 { font-size: .75rem; text-transform: uppercase; letter-spacing: .05em; color: #888; margin: 8px 8px; }
  .rep-item { display: block; width: 100%; text-align: left; background: none; border: none; color: #c9d1d9; padding: 8px; border-radius: 6px; cursor: pointer; font-size: .85rem; }
  .rep-item:hover { background: #1d2129; }
  .rep-item .t { display: block; font-size: .7rem; color: #777; margin-top: 2px; }
  #main { flex: 1; display: flex; flex-direction: column; min-width: 0; }
  #chat { flex: 1; overflow-y: auto; padding: 24px; display: flex; flex-direction: column; gap: 12px; }
  .msg { max-width: 720px; padding: 12px 16px; border-radius: 12px; line-height: 1.5; font-size: .95rem; white-space: pre-wrap; word-wrap: break-word; }
  .msg.user { align-self: flex-end; background: #2b3a55; border-bottom-right-radius: 4px; }
  .msg.assistant { align-self: flex-start; background: #1a1d24; border-bottom-left-radius: 4px; }
  .msg.assistant .md { white-space: normal; }
  .msg.assistant .md h1, .msg.assistant .md h2, .msg.assistant .md h3 { margin: .6em 0 .3em; }
  .msg.assistant .md p { margin: .4em 0; }
  .msg.assistant .md ul, .msg.assistant .md ol { margin: .4em 0; padding-left: 1.4em; }
  .msg.assistant .md a { color: #7aa7ff; }
  .msg.assistant .md code { background: #23272f; padding: 1px 5px; border-radius: 4px; font-size: .85em; }
  .msg.assistant .md pre { background: #23272f; padding: 10px; border-radius: 8px; overflow-x: auto; }
  .msg.assistant .md blockquote { border-left: 3px solid #4f8cff; margin: .4em 0; padding-left: 12px; color: #aaa; }
  .msg.assistant .md table { border-collapse: collapse; margin: .5em 0; }
  .msg.assistant .md th, .msg.assistant .md td { border: 1px solid #333; padding: 6px 10px; }
  .typing { display: inline-flex; gap: 4px; align-items: center; }
  .typing span { width: 7px; height: 7px; border-radius: 50%; background: #888; animation: blink 1.2s infinite; }
  .typing span:nth-child(2) { animation-delay: .2s; }
  .typing span:nth-child(3) { animation-delay: .4s; }
  @keyframes blink { 0%, 80%, 100% { opacity: .25; } 40% { opacity: 1; } }
  .q-input { display: flex; gap: 8px; margin-top: 10px; }
  .q-input input { flex: 1; padding: 10px 12px; border-radius: 8px; border: 1px solid #333; background: #14171d; color: #e6e6e6; font-size: .9rem; }
  .q-input input:focus { outline: none; border-color: #4f8cff; }
  .q-input button { padding: 10px 16px; border-radius: 8px; border: none; background: #4f8cff; color: #fff; cursor: pointer; font-size: .9rem; }
  .q-input button:disabled { opacity: .5; cursor: default; }
  #composer { padding: 16px 24px; border-top: 1px solid #23272f; background: #0f1115; }
  #composer form { display: flex; gap: 8px; max-width: 720px; margin: 0 auto; }
  #composer input { flex: 1; padding: 12px 14px; border-radius: 10px; border: 1px solid #333; background: #14171d; color: #e6e6e6; font-size: .95rem; }
  #composer input:focus { outline: none; border-color: #4f8cff; }
  #composer button { padding: 12px 20px; border-radius: 10px; border: none; background: #4f8cff; color: #fff; cursor: pointer; font-size: .95rem; }
  #composer button:disabled { opacity: .5; cursor: default; }
  .hidden { display: none !important; }
  .empty { color: #666; text-align: center; margin-top: 20vh; font-size: .95rem; }
</style>
</head>
<body>
<div id="sidebar">
  <h1>Research Agent</h1>
  <div class="sub">Мини-Perplexity</div>
  <button id="newBtn">Новое исследование</button>
  <div id="reports"><h3>Прошлые отчёты</h3></div>
</div>
<div id="main">
  <div id="chat"></div>
  <div id="composer">
    <form id="composerForm">
      <input type="text" id="topic" placeholder="Введите тему исследования, например: стоит ли переезжать с REST на gRPC" autocomplete="off">
      <button id="startBtn" type="submit">Исследовать</button>
    </form>
  </div>
</div>

<script>
const chat = document.getElementById('chat');
const composerForm = document.getElementById('composerForm');
const topicEl = document.getElementById('topic');
const startBtn = document.getElementById('startBtn');
const newBtn = document.getElementById('newBtn');
const reportsEl = document.getElementById('reports');

let sessionId = null;
let es = null;
let busy = false;
let pendingQuestions = [];
let questionIdx = 0;
let awaitingAnswer = false;

function addMsg(role, html) {
  const div = document.createElement('div');
  div.className = 'msg ' + role;
  div.innerHTML = html;
  chat.appendChild(div);
  scrollToBottom();
  return div;
}

function addUserMsg(text) {
  const div = document.createElement('div');
  div.className = 'msg user';
  div.textContent = text;
  chat.appendChild(div);
  scrollToBottom();
  return div;
}

function addAssistantMsg(html) {
  const div = document.createElement('div');
  div.className = 'msg assistant';
  div.innerHTML = html;
  chat.appendChild(div);
  scrollToBottom();
  return div;
}

function addTyping() {
  const div = document.createElement('div');
  div.className = 'msg assistant';
  div.innerHTML = '<div class="typing"><span></span><span></span><span></span></div>';
  chat.appendChild(div);
  scrollToBottom();
  return div;
}

function scrollToBottom() {
  chat.scrollTop = chat.scrollHeight;
}

function renderMarkdown(md) {
  if (typeof marked !== 'undefined' && typeof DOMPurify !== 'undefined') {
    return DOMPurify.sanitize(marked.parse(md));
  }
  const div = document.createElement('div');
  div.textContent = md;
  return div.innerHTML;
}

function showQuestion(q, total) {
  awaitingAnswer = true;
  const div = addAssistantMsg('<div>Вопрос ' + questionIdx + ' из ' + total + '</div><div>' + escapeHtml(q) + '</div>');
  const row = document.createElement('div');
  row.className = 'q-input';
  const input = document.createElement('input');
  input.type = 'text';
  input.placeholder = 'Ваш ответ...';
  input.autocomplete = 'off';
  const btn = document.createElement('button');
  btn.textContent = 'Ответить';
  btn.disabled = true;
  input.addEventListener('input', () => { btn.disabled = !input.value.trim(); });
  const submit = async () => {
    const ans = input.value.trim();
    if (!ans) return;
    input.disabled = true;
    btn.disabled = true;
    addUserMsg(ans);
    await fetch('/api/answer', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({id: sessionId, text: ans})
    });
    questionIdx++;
    if (questionIdx < pendingQuestions.length) {
      showQuestion(pendingQuestions[questionIdx], pendingQuestions.length);
    } else {
      awaitingAnswer = false;
      pendingQuestions = [];
      questionIdx = 0;
      addTyping();
    }
  };
  btn.onclick = submit;
  input.addEventListener('keydown', (e) => { if (e.key === 'Enter') submit(); });
  row.appendChild(input);
  row.appendChild(btn);
  div.appendChild(row);
  input.focus();
}

function escapeHtml(s) {
  const div = document.createElement('div');
  div.textContent = s;
  return div.innerHTML;
}

function setBusy(b) {
  busy = b;
  startBtn.disabled = b;
  topicEl.disabled = b;
}

function resetChat() {
  chat.innerHTML = '';
  const empty = document.createElement('div');
  empty.className = 'empty';
  empty.textContent = 'Введите тему исследования ниже.';
  chat.appendChild(empty);
}

async function start() {
  const topic = topicEl.value.trim();
  if (!topic || busy) return;
  setBusy(true);
  chat.innerHTML = '';
  addUserMsg(topic);
  topicEl.value = '';
  const res = await fetch('/api/start', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({topic})
  });
  if (!res.ok) {
    addAssistantMsg('<div>Не удалось начать исследование.</div>');
    setBusy(false);
    return;
  }
  const data = await res.json();
  sessionId = data.id;
  connectSSE();
}

function connectSSE() {
  if (es) es.close();
  es = new EventSource('/api/events?session=' + encodeURIComponent(sessionId));
  es.onmessage = (e) => {
    const ev = JSON.parse(e.data);
    switch (ev.type) {
      case 'clarify':
        pendingQuestions = ev.data || [];
        questionIdx = 0;
        if (pendingQuestions.length > 0) {
          showQuestion(pendingQuestions[0], pendingQuestions.length);
        } else {
          addTyping();
        }
        break;
      case 'question':
        if (!awaitingAnswer) {
          pendingQuestions = [ev.message];
          questionIdx = 0;
          showQuestion(ev.message, 1);
        }
        break;
      case 'report':
        addAssistantMsg('<div class="md">' + renderMarkdown(ev.data) + '</div>');
        break;
      case 'done':
        es.close();
        es = null;
        setBusy(false);
        loadReports();
        break;
      case 'error':
        addAssistantMsg('<div>⚠️ ' + escapeHtml(ev.message || 'Ошибка') + '</div>');
        break;
      default:
        break;
    }
  };
  es.onerror = () => { /* keep alive */ };
}

async function loadReports() {
  const res = await fetch('/api/reports');
  if (!res.ok) return;
  const reports = await res.json();
  reportsEl.innerHTML = '<h3>Прошлые отчёты</h3>';
  if (reports.length === 0) {
    const d = document.createElement('div');
    d.className = 'rep-item';
    d.textContent = 'Пока пусто';
    reportsEl.appendChild(d);
    return;
  }
  reports.forEach(r => {
    const b = document.createElement('button');
    b.className = 'rep-item';
    const t = document.createElement('span');
    t.className = 't';
    t.textContent = new Date(r.created_at).toLocaleString();
    b.appendChild(document.createTextNode(r.topic));
    b.appendChild(t);
    b.onclick = () => {
      chat.innerHTML = '';
      addUserMsg(r.topic);
      addAssistantMsg('<div class="md">' + renderMarkdown(r.markdown) + '</div>');
    };
    reportsEl.appendChild(b);
  });
}

composerForm.addEventListener('submit', (e) => { e.preventDefault(); start(); });
newBtn.onclick = () => {
  if (es) { es.close(); es = null; }
  sessionId = null;
  setBusy(false);
  resetChat();
  topicEl.focus();
};
resetChat();
loadReports();
</script>
</body>
</html>`
