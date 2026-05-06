import type { AskRequest, AskResponse, RetrievalMode, SearchRequest, SearchResponse } from './types'

const BASE = '/api'

async function post<T>(path: string, body: unknown): Promise<T> {
  const res = await fetch(`${BASE}${path}`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  })
  if (!res.ok) {
    const detail = await res.json().catch(() => ({ detail: res.statusText }))
    throw new Error(detail.detail ?? res.statusText)
  }
  return res.json() as Promise<T>
}

export function search(mode: RetrievalMode, req: SearchRequest): Promise<SearchResponse> {
  return post<SearchResponse>(`/search/${mode}`, req)
}

export function ask(req: AskRequest): Promise<AskResponse> {
  return post<AskResponse>('/ask', req)
}
