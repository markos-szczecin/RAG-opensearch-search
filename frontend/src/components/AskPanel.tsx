import { FormEvent, useState } from 'react'
import { ask } from '../api'
import type { AnswerMode, AskResponse, RetrievalMode, UserRole } from '../types'

const MODES: RetrievalMode[] = ['hybrid', 'keyword', 'vector']
const ANSWER_MODES: AnswerMode[] = ['brief', 'detailed', 'step_by_step', 'agent']
const ROLES: UserRole[] = ['customer', 'agent', 'compliance', 'admin']

const CONFIDENCE_COLORS: Record<string, string> = {
  grounded: '#16a34a',
  cautious: '#ca8a04',
  refused: '#dc2626',
}

export function AskPanel() {
  const [query, setQuery] = useState('')
  const [mode, setMode] = useState<RetrievalMode>('hybrid')
  const [answerMode, setAnswerMode] = useState<AnswerMode>('brief')
  const [role, setRole] = useState<UserRole>('customer')
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [response, setResponse] = useState<AskResponse | null>(null)

  async function handleSubmit(e: FormEvent) {
    e.preventDefault()
    if (!query.trim()) return
    setLoading(true)
    setError(null)
    setResponse(null)
    try {
      const res = await ask({
        query: query.trim(),
        user_role: role,
        retrieval_mode: mode,
        answer_mode: answerMode,
      })
      setResponse(res)
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Unknown error')
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="panel">
      <form onSubmit={handleSubmit} className="form">
        <div className="field">
          <textarea
            className="query-input query-textarea"
            placeholder="Ask a question about the knowledge base…"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            disabled={loading}
            rows={3}
          />
        </div>

        <div className="controls">
          <label className="control-group">
            <span>Retrieval</span>
            <select value={mode} onChange={(e) => setMode(e.target.value as RetrievalMode)} disabled={loading}>
              {MODES.map((m) => (
                <option key={m} value={m}>{m}</option>
              ))}
            </select>
          </label>

          <label className="control-group">
            <span>Answer style</span>
            <select value={answerMode} onChange={(e) => setAnswerMode(e.target.value as AnswerMode)} disabled={loading}>
              {ANSWER_MODES.map((m) => (
                <option key={m} value={m}>{m.replace('_', ' ')}</option>
              ))}
            </select>
          </label>

          <label className="control-group">
            <span>Role</span>
            <select value={role} onChange={(e) => setRole(e.target.value as UserRole)} disabled={loading}>
              {ROLES.map((r) => (
                <option key={r} value={r}>{r}</option>
              ))}
            </select>
          </label>

          <button type="submit" className="btn-primary" disabled={loading || !query.trim()}>
            {loading ? 'Thinking…' : 'Ask'}
          </button>
        </div>
      </form>

      {error && <div className="error-box">{error}</div>}

      {response && (
        <div className="ask-response">
          <div className="answer-header">
            <span
              className="confidence-badge"
              style={{ backgroundColor: CONFIDENCE_COLORS[response.confidence] }}
            >
              {response.confidence}
            </span>
            <span className="answer-meta">
              {response.retrieval_mode} · {response.latency_ms.toFixed(0)} ms ·{' '}
              {response.tokens.total} tokens
            </span>
          </div>

          <div className="answer-body">{response.answer}</div>

          {response.citations.length > 0 && (
            <div className="citations">
              <h4 className="citations-title">Sources</h4>
              {response.citations.map((c, i) => (
                <div key={c.chunk_id} className="citation">
                  <span className="citation-num">[{i + 1}]</span>
                  <span className="citation-title">{c.title}</span>
                  <span className="citation-score">score {c.score.toFixed(3)}</span>
                  <span className="result-path">{c.source_path}</span>
                </div>
              ))}
            </div>
          )}
        </div>
      )}
    </div>
  )
}
