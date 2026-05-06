import { useState } from 'react'
import { AskPanel } from './components/AskPanel'
import { SearchPanel } from './components/SearchPanel'

type Tab = 'search' | 'ask'

export default function App() {
  const [tab, setTab] = useState<Tab>('search')

  return (
    <div className="app">
      <header className="header">
        <div className="header-inner">
          <h1 className="logo">RAG Search</h1>
          <p className="subtitle">Fintech knowledge base · hybrid retrieval</p>
        </div>
      </header>

      <main className="main">
        <nav className="tabs">
          <button
            className={`tab ${tab === 'search' ? 'tab-active' : ''}`}
            onClick={() => setTab('search')}
          >
            Search
          </button>
          <button
            className={`tab ${tab === 'ask' ? 'tab-active' : ''}`}
            onClick={() => setTab('ask')}
          >
            Ask
          </button>
        </nav>

        {tab === 'search' ? <SearchPanel /> : <AskPanel />}
      </main>
    </div>
  )
}
