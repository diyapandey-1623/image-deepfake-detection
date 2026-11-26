import React, { useState, useEffect } from 'react'
import { History as HistoryIcon, RefreshCw } from 'lucide-react'
import ScanTable from '../components/ScanTable'
import { getHistory } from '../lib/api'

export default function History() {
  const [scans, setScans] = useState([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)

  const loadHistory = async () => {
    setLoading(true)
    setError(null)
    try {
      const data = await getHistory(100)
      setScans(Array.isArray(data) ? data : [])
    } catch (err) {
      setError(err.message || 'Could not load scan history.')
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => {
    loadHistory()
  }, [])

  return (
    <div className="max-w-5xl mx-auto px-4 sm:px-6 py-8 sm:py-10 animate-fade-in">
      <div className="flex flex-col sm:flex-row items-start sm:items-center justify-between pb-5 border-b border-border gap-4 mb-6">
        <div>
          <div className="flex items-center gap-2">
            <HistoryIcon className="w-4 h-4 text-text" strokeWidth={1.75} />
            <h1 className="font-semibold text-xl text-text tracking-tight">
              Scan history
            </h1>
          </div>
          <p className="text-xs text-text-muted mt-0.5">
            Audit log of verified image inputs, confidence scores, and diagnostic records.
          </p>
        </div>

        <button
          onClick={loadHistory}
          disabled={loading}
          className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-card border border-border text-xs font-medium text-text hover:bg-surface-subtle transition-colors shadow-subtle focus-ring disabled:opacity-50"
        >
          <RefreshCw className={`w-3.5 h-3.5 text-text-muted ${loading ? 'animate-spin' : ''}`} />
          Refresh
        </button>
      </div>

      {loading ? (
        <div className="py-20 text-center">
          <div className="w-6 h-6 border-2 border-text border-t-transparent rounded-full animate-spin mx-auto mb-2.5" />
          <p className="text-xs text-text-muted font-mono">Retrieving audit records...</p>
        </div>
      ) : error ? (
        <div className="p-6 rounded-xl bg-card border border-border text-center shadow-subtle max-w-md mx-auto">
          <p className="text-xs font-medium text-text mb-1">Could not connect</p>
          <p className="text-xs text-text-muted mb-4">{error}</p>
          <button
            onClick={loadHistory}
            className="px-3.5 py-1.5 rounded-lg bg-text text-white text-xs font-medium focus-ring"
          >
            Try again
          </button>
        </div>
      ) : (
        <ScanTable scans={scans} />
      )}
    </div>
  )
}
