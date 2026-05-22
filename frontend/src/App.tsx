import { useState, useRef, useEffect, useCallback } from 'react';
import ReactMarkdown from 'react-markdown';
import { Send, RefreshCw, Microscope, ExternalLink, Brain } from 'lucide-react';
import './index.css';

// ── Types ─────────────────────────────────────────────────────────────────────

interface ChatMessage {
  id: string;
  type: 'query' | 'answer' | 'error';
  content: string;
  sources?: string[];
}

interface LoadingStage {
  id: string;
  label: string;
  icon: string;
  status: 'pending' | 'active' | 'done';
}

const STAGES: Omit<LoadingStage, 'status'>[] = [
  { id: 'plan',      label: 'Decomposing into sub-questions',   icon: '⬡' },
  { id: 'search',    label: 'Searching the web via Tavily',      icon: '◎' },
  { id: 'retrieve',  label: 'Querying local knowledge base',     icon: '◈' },
  { id: 'synthesize',label: 'Synthesizing & evaluating results', icon: '◇' },
  { id: 'respond',   label: 'Generating cited answer',           icon: '◉' },
];

const SUGGESTIONS = [
  { icon: '🧬', text: 'How does retrieval-augmented generation work?' },
  { icon: '🤖', text: 'What are the key differences between GPT-4 and Gemini?' },
  { icon: '⚡', text: 'Explain transformer architecture in deep learning' },
  { icon: '🔍', text: 'What frameworks are used to build AI agents?' },
];

const API_BASE = 'http://localhost:8000';

// ── Helpers ───────────────────────────────────────────────────────────────────

function generateId(): string {
  return Math.random().toString(36).slice(2, 10);
}



// ── Loading Indicator ─────────────────────────────────────────────────────────

function LoadingIndicator() {
  const [stageIndex, setStageIndex] = useState(0);

  useEffect(() => {
    const timer = setInterval(() => {
      setStageIndex(prev => Math.min(prev + 1, STAGES.length - 1));
    }, 3500);
    return () => clearInterval(timer);
  }, []);

  return (
    <div className="loading-card">
      <div className="loading-stages">
        {STAGES.map((stage, idx) => {
          const status =
            idx < stageIndex ? 'done' :
            idx === stageIndex ? 'active' : 'pending';
          return (
            <div key={stage.id} className={`loading-stage ${status}`}>
              <span className="stage-icon">{status === 'done' ? '✓' : stage.icon}</span>
              <span>{stage.label}</span>
            </div>
          );
        })}
      </div>
    </div>
  );
}

// ── Answer Card ───────────────────────────────────────────────────────────────

function AnswerCard({ content, sources }: { content: string; sources?: string[] }) {
  return (
    <div className="answer-card">
      <div className="answer-header">
        <span className="answer-icon"><Brain size={13} /></span>
        <span className="answer-label">Research Answer</span>
      </div>
      <div className="answer-body">
        <ReactMarkdown>{content}</ReactMarkdown>
        {sources && sources.length > 0 && (
          <div className="sources-section">
            <div className="sources-title">Sources</div>
            <div className="source-list">
              {sources.map((url, i) => (
                <div key={i} className="source-item">
                  <span className="source-num">[{i + 1}]</span>
                  <a href={url} target="_blank" rel="noopener noreferrer" title={url}>
                    {url.replace(/^https?:\/\//, '').slice(0, 60)}{url.length > 65 ? '…' : ''}
                  </a>
                  <ExternalLink size={11} style={{ opacity: 0.4, flexShrink: 0 }} />
                </div>
              ))}
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

// ── Main App ──────────────────────────────────────────────────────────────────

export default function App() {
  const [sessionId, setSessionId] = useState<string>(() => generateId() + generateId());
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [input, setInput] = useState('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const textareaRef = useRef<HTMLTextAreaElement>(null);
  const bottomRef = useRef<HTMLDivElement>(null);

  // Auto-scroll to bottom when messages change
  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages, loading]);

  // Auto-resize textarea
  useEffect(() => {
    const ta = textareaRef.current;
    if (ta) {
      ta.style.height = 'auto';
      ta.style.height = Math.min(ta.scrollHeight, 160) + 'px';
    }
  }, [input]);

  const startNewSession = useCallback(() => {
    setSessionId(generateId() + generateId());
    setMessages([]);
    setError(null);
    textareaRef.current?.focus();
  }, []);

  const submit = useCallback(async (topic: string) => {
    const trimmed = topic.trim();
    if (!trimmed || loading) return;

    const queryId = generateId();
    setMessages(prev => [...prev, { id: queryId, type: 'query', content: trimmed }]);
    setInput('');
    setLoading(true);
    setError(null);

    try {
      const res = await fetch(`${API_BASE}/api/research`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ topic: trimmed, thread_id: sessionId }),
      });

      if (!res.ok) {
        const errData = await res.json().catch(() => ({}));
        throw new Error(errData.detail ?? `Server error ${res.status}`);
      }

      const data = await res.json();
      setMessages(prev => [
        ...prev,
        { id: generateId(), type: 'answer', content: data.answer, sources: data.sources },
      ]);
    } catch (err: unknown) {
      const msg = err instanceof Error ? err.message : 'Unknown error';
      setMessages(prev => [
        ...prev,
        { id: generateId(), type: 'error', content: msg },
      ]);
    } finally {
      setLoading(false);
    }
  }, [loading, sessionId]);

  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      submit(input);
    }
  };

  return (
    <div className="app-wrapper">
      {/* ── Header ── */}
      <header className="header">
        <div className="header-inner">
          <div className="logo">
            <div className="logo-icon"><Microscope size={17} /></div>
            <span className="logo-text">Research Agent</span>
          </div>
        </div>
      </header>

      {/* ── Main ── */}
      <main className="main">
        {/* Hero */}
        <section className="hero">
          <h1 className="hero-title">AI-Powered <span className="highlight">Research Assistant</span></h1>
          <p className="hero-subtitle">
            Ask any question. The agent decomposes it into sub-questions, searches the web,
            retrieves knowledge, and delivers a comprehensive cited answer.
          </p>
        </section>

        {/* Session indicator */}
        <div className="session-card">
          <div className="session-info">
            <span className="session-dot" />
            <span>Session active</span>
          </div>
          <button
            id="btn-new-session"
            className="btn-new-session"
            onClick={startNewSession}
            title="Start a fresh research session"
          >
            <RefreshCw size={12} /> New Session
          </button>
        </div>

        {/* Error banner */}
        {error && (
          <div className="error-banner" role="alert">
            ⚠ {error}
          </div>
        )}

        {/* Chat history */}
        {messages.length === 0 && !loading ? (
          <div className="empty-state">
            <div className="suggestions-grid">
              {SUGGESTIONS.map((s, i) => (
                <button
                  key={i}
                  id={`suggestion-${i}`}
                  className="suggestion-card"
                  onClick={() => submit(s.text)}
                >
                  <span className="suggestion-icon">{s.icon}</span>
                  <span className="suggestion-text">{s.text}</span>
                </button>
              ))}
            </div>
          </div>
        ) : (
          <div className="chat-history">
            {messages.map(msg => {
              if (msg.type === 'query') {
                return (
                  <div key={msg.id} className="query-bubble">
                    {msg.content}
                  </div>
                );
              }
              if (msg.type === 'answer') {
                return (
                  <AnswerCard key={msg.id} content={msg.content} sources={msg.sources} />
                );
              }
              if (msg.type === 'error') {
                return (
                  <div key={msg.id} className="error-banner" role="alert">
                    ⚠ {msg.content}
                  </div>
                );
              }
              return null;
            })}
            {loading && <LoadingIndicator />}
          </div>
        )}

        {/* Bottom anchor for auto-scroll */}
        <div ref={bottomRef} />

        {/* Input area */}
        <div className="input-area">
          <form
            className="input-form"
            onSubmit={e => { e.preventDefault(); submit(input); }}
          >
            <div className="input-wrapper">
              <textarea
                ref={textareaRef}
                id="research-input"
                className="input-field"
                placeholder="Ask a research question… (Enter to send, Shift+Enter for newline)"
                value={input}
                onChange={e => setInput(e.target.value)}
                onKeyDown={handleKeyDown}
                disabled={loading}
                rows={1}
              />
            </div>
            <button
              id="btn-submit"
              type="submit"
              className="btn-submit"
              disabled={loading || !input.trim()}
              title="Send"
            >
              <Send size={18} />
            </button>
          </form>
        </div>
      </main>
    </div>
  );
}
