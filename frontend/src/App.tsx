import { useState, useRef, useEffect, useCallback } from 'react';
import {
  ArrowUpRight, BookOpen, ChevronDown, CircleAlert, GitBranch, Layers3, LibraryBig, Moon, Network, Plus, Sun, ThumbsUp,
} from 'lucide-react';
import Composer from './Composer';
import InkAnswer from './InkAnswer';

interface ChatMessage {
  id: string;
  type: 'query' | 'answer' | 'error';
  content: string;
  sources?: string[];
}

const SUGGESTIONS = [
  { icon: LibraryBig, text: 'How does retrieval-augmented generation work?' },
  { icon: GitBranch, text: 'What are the key differences between GPT-4 and Gemini?' },
  { icon: Layers3, text: 'Explain transformer architecture in deep learning' },
  { icon: Network, text: 'What frameworks are used to build AI agents?' },
];

const LOCAL_API_BASE = 'http://localhost:8000';
const API_BASE = import.meta.env.VITE_API_URL?.trim().replace(/\/+$/, '') || LOCAL_API_BASE;

function generateId(): string {
  return Math.random().toString(36).slice(2, 10);
}

function LoadingIndicator() {
  return (
    <div className="loading-card" role="status">
      <span className="loading-ink" aria-hidden="true" />
      <span>Working on your answer…</span>
    </div>
  );
}

function AnswerCard({ id, content, sources, latest, answerRef }: {
  id: string;
  content: string;
  sources?: string[];
  latest: boolean;
  answerRef: React.RefObject<HTMLElement | null>;
}) {
  const [expanded, setExpanded] = useState(true);
  const [liked, setLiked] = useState(false);
  const sourcesId = `sources-${id}`;

  return (
    <article className="answer-card" ref={latest ? answerRef : undefined}>
      <InkAnswer content={content} animate={latest} />
      <footer className="answer-tools">
        {!!sources?.length && (
          <h3 className="sources-title">
            <button type="button" className="sources-toggle" aria-expanded={expanded} aria-controls={sourcesId}
              onClick={() => setExpanded(open => !open)}>
              Sources <span className="source-count">{String(sources.length).padStart(2, '0')}</span>
              <ChevronDown size={16} aria-hidden="true" />
            </button>
          </h3>
        )}
        <button type="button" className="like-button" aria-label="Like answer" aria-pressed={liked}
          title="Like answer" onClick={() => setLiked(value => !value)}>
          <ThumbsUp size={18} aria-hidden="true" />
        </button>
      </footer>
      {!!sources?.length && (
        <div id={sourcesId} className="sources-panel" data-expanded={expanded} inert={!expanded}>
          <div className="sources-inner">
            <ol className="source-list" role="list">
              {sources.map((url, index) => (
                <li key={index} className="source-item">
                  <span className="source-num">[{index + 1}]</span>
                  {/^https?:\/\//i.test(url) ? (
                    <a href={url} target="_blank" rel="noopener noreferrer" title={url}>
                      {url.replace(/^https?:\/\//i, '')}
                      <ArrowUpRight size={14} aria-hidden="true" />
                    </a>
                  ) : (
                    <span className="local-source" title={url}>{url}</span>
                  )}
                </li>
              ))}
            </ol>
          </div>
        </div>
      )}
    </article>
  );
}

export default function App() {
  const [sessionId, setSessionId] = useState(() => generateId() + generateId());
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [input, setInput] = useState('');
  const [loading, setLoading] = useState(false);
  const [dark, setDark] = useState(() => localStorage.getItem('research-theme') === 'dark');
  const textareaRef = useRef<HTMLTextAreaElement>(null);
  const bottomRef = useRef<HTMLDivElement>(null);
  const answerRef = useRef<HTMLElement>(null);
  const lastMessage = messages.at(-1);

  useEffect(() => {
    if (messages.length === 0) return;
    const reduceMotion = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
    const behavior = reduceMotion ? 'auto' : 'smooth';
    if (lastMessage?.type === 'answer') {
      answerRef.current?.scrollIntoView({ block: 'start', behavior });
    } else {
      bottomRef.current?.scrollIntoView({ block: 'nearest', behavior });
    }
  }, [lastMessage, loading, messages.length]);

  useEffect(() => {
    localStorage.setItem('research-theme', dark ? 'dark' : 'light');
    document.documentElement.dataset.theme = dark ? 'dark' : 'light';
  }, [dark]);

  const startNewSession = useCallback(() => {
    setSessionId(generateId() + generateId());
    setMessages([]);
    setInput('');
    textareaRef.current?.focus();
  }, []);

  const submit = useCallback(async (topic: string) => {
    const trimmed = topic.trim();
    if (!trimmed || loading) return;

    setMessages(prev => [...prev, { id: generateId(), type: 'query', content: trimmed }]);
    setInput('');
    setLoading(true);

    try {
      const res = await fetch(`${API_BASE}/api/research`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ topic: trimmed, thread_id: sessionId }),
      }).catch((cause: unknown) => {
        if (!(cause instanceof TypeError)) throw cause;
        const guidance = API_BASE === LOCAL_API_BASE
          ? 'From the repository root, run python -m uvicorn api:app --port 8000 and retry.'
          : 'Check that the backend is running and allows requests from this site.';
        throw new Error(`Could not reach the Research API at ${API_BASE}. ${guidance}`, { cause });
      });

      if (res.status === 429) {
        const retryAfter = Number(res.headers.get('Retry-After'));
        const wait = retryAfter > 0 ? `in ${retryAfter} seconds` : 'in a moment';
        throw new Error(`Too many research requests. Try again ${wait}.`);
      }
      if (res.status === 504) {
        throw new Error('The research request timed out. Try again with a narrower question.');
      }
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
      setMessages(prev => [...prev, { id: generateId(), type: 'error', content: msg }]);
    } finally {
      setLoading(false);
    }
  }, [loading, sessionId]);

  return (
    <div className="app-wrapper">
      <header className="header">
        <div className="header-inner">
          <div className="logo">
            <span className="logo-icon"><BookOpen size={20} aria-hidden="true" /></span>
            <span className="logo-text">Research Agent</span>
          </div>
          <button type="button" className="theme-button" aria-label={dark ? 'Light mode' : 'Dark mode'}
            onClick={() => setDark(value => !value)}>
            {dark ? <Sun size={17} aria-hidden="true" /> : <Moon size={17} aria-hidden="true" />}
            <span>{dark ? 'Light' : 'Dark'}</span>
          </button>
        </div>
      </header>

      <main className="main">
        <section className="hero">
          <p className="eyebrow">The research desk</p>
          <h1 className="hero-title"><span>Research</span>{' '}<em>Assistant</em></h1>
          <p className="hero-subtitle">
            Ask a question, explore the evidence, and read the sources behind each answer.
          </p>
        </section>

        <div className="session-card">
          <span className="session-info"><span className="session-dot" aria-hidden="true" /> Session active</span>
          <button type="button" id="btn-new-session" className="btn-new-session"
            onClick={startNewSession} disabled={loading}>
            <Plus size={16} aria-hidden="true" /> New session
          </button>
        </div>

        <span className="sr-only" aria-live="polite">
          {lastMessage?.type === 'answer' ? 'Research answer ready.' : ''}
        </span>
        {messages.length === 0 ? (
          <section className="empty-state" aria-labelledby="suggestions-title">
            <h2 className="eyebrow" id="suggestions-title">Start with a question</h2>
            <div className="suggestions-grid">
              {SUGGESTIONS.map(({ icon: Icon, text }, index) => (
                <button type="button" key={text} id={`suggestion-${index}`}
                  className="suggestion-card" onClick={() => submit(text)}>
                  <span className="suggestion-meta" aria-hidden="true">
                    <Icon size={19} /> <span>{String(index + 1).padStart(2, '0')}</span>
                  </span>
                  <span className="suggestion-text">{text}</span>
                </button>
              ))}
            </div>
          </section>
        ) : (
          <section className="chat-history" aria-label="Research conversation">
            {messages.map(message => {
              if (message.type === 'answer') {
                return <AnswerCard key={message.id} id={message.id} content={message.content}
                  sources={message.sources} latest={message === lastMessage} answerRef={answerRef} />;
              }
              if (message.type === 'error') {
                return <div key={message.id} className="error-banner" role="alert">
                  <CircleAlert size={19} aria-hidden="true" /> {message.content}
                </div>;
              }
              return <div key={message.id} className="query-bubble">{message.content}</div>;
            })}
            {loading && <LoadingIndicator />}
          </section>
        )}
        <div className="scroll-anchor" ref={bottomRef} />
        <Composer value={input} loading={loading} onChange={setInput}
          onSubmit={() => submit(input)} textareaRef={textareaRef} />
      </main>
    </div>
  );
}
