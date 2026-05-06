import { FormEvent, useState } from 'react'
import { search } from '../api'
import type { RetrievalMode, SearchResponse, UserRole } from '../types'

const MODES: RetrievalMode[] = ['hybrid', 'keyword', 'vector']
const ROLES: UserRole[] = ['customer', 'agent', 'compliance', 'admin']

export function SearchPanel() {
  const [query, setQuery] = useState('')
  const [mode, setMode] = useState<RetrievalMode>('hybrid')
  const [role, setRole] = useState<UserRole>('customer')
  const [topK, setTopK] = useState(5)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [response, setResponse] = useState<SearchResponse | null>(null)

  async function handleSubmit(e: FormEvent) {
    e.preventDefault()
    if (!query.trim()) return
    setLoading(true)
    setError(null)
    setResponse(null)
    try {
      const res = await search(mode, { query: query.trim(), top_k: topK, user_role: role })
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
          <input
            className="query-input"
            type="text"
            placeholder="Search the knowledge base…"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            disabled={loading}
          />
        </div>

        <div className="controls">
          <label className="control-group">
            <span>Mode</span>
            <select value={mode} onChange={(e) => setMode(e.target.value as RetrievalMode)} disabled={loading}>
              {MODES.map((m) => (
                <option key={m} value={m}>{m}</option>
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

          <label className="control-group">
            <span>Top-K</span>
            <select value={topK} onChange={(e) => setTopK(Number(e.target.value))} disabled={loading}>
              {[3, 5, 10, 20].map((k) => (
                <option key={k} value={k}>{k}</option>
              ))}
            </select>
          </label>

          <button type="submit" className="btn-primary" disabled={loading || !query.trim()}>
            {loading ? 'Searching…' : 'Search'}
          </button>
        </div>
      </form>

      {error && <div className="error-box">{error}</div>}

      {response && (
        <div className="results">
          <div className="results-meta">
            {response.total} results · {response.retrieval_mode} · {response.latency_ms.toFixed(0)} ms
          </div>
          {response.results.map((r) => (
            <div key={r.chunk_id} className="result-card">
              <div className="result-header">
                <span className="result-title">{r.title}</span>
                <span className="result-score">score {r.score.toFixed(3)}</span>
              </div>
              <div className="result-meta">
                <span className="badge">{r.doc_type}</span>
                <span className="badge">{r.access_level}</span>
                <span className="result-path">{r.source_path}</span>
              </div>
              {r.highlight ? (
                <p
                  className="result-highlight"
                  dangerouslySetInnerHTML={{ __html: r.highlight }}
                />
              ) : (
                <p className="result-content">{r.content.slice(0, 300)}{r.content.length > 300 ? '…' : ''}</p>
              )}
            </div>
          ))}
          {response.results.length === 0 && (
            <p className="empty">No results found.</p>
          )}
        </div>
      )}
    </div>
  )
}
