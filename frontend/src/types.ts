export type RetrievalMode = 'keyword' | 'vector' | 'hybrid'
export type AnswerMode = 'brief' | 'detailed' | 'step_by_step' | 'agent'
export type UserRole = 'customer' | 'agent' | 'compliance' | 'admin'

export interface SearchFilters {
  language?: string
  doc_type?: string
  access_level?: string
  status?: string
  department?: string
}

export interface SearchRequest {
  query: string
  filters?: SearchFilters
  top_k?: number
  user_role?: string
}

export interface SearchResult {
  chunk_id: string
  doc_id: string
  title: string
  content: string
  score: number
  source_path: string
  access_level: string
  doc_type: string
  highlight?: string
}

export interface SearchResponse {
  results: SearchResult[]
  total: number
  retrieval_mode: RetrievalMode
  latency_ms: number
}

export interface AskRequest {
  query: string
  user_role?: string
  retrieval_mode?: RetrievalMode
  answer_mode?: AnswerMode
  chat_history?: Array<{ role: string; content: string }>
}

export interface Citation {
  doc_id: string
  title: string
  chunk_id: string
  score: number
  source_path: string
}

export interface TokenUsage {
  context: number
  answer: number
  total: number
}

export interface AskResponse {
  answer: string
  citations: Citation[]
  retrieval_mode: RetrievalMode
  tokens: TokenUsage
  confidence: 'grounded' | 'cautious' | 'refused'
  latency_ms: number
}
