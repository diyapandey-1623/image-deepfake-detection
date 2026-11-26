import React, { useState, useEffect } from 'react'

export default function HistoryPage({ apiBase, onSelectScan }) {
  const [history, setHistory] = useState([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)

  const fetchHistory = async () => {
    setLoading(true)
    setError(null)
    try {
      const res = await fetch(`${apiBase}/api/history`)
      if (!res.ok) {
        throw new Error(`Failed to load scan history (HTTP ${res.status})`)
      }
      const data = await res.json()
      setHistory(data)
    } catch (err) {
      setError(err.message || 'Could not connect to history service.')
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => {
    fetchHistory()
  }, [apiBase])

  const downloadPdf = (reportId) => {
    window.open(`${apiBase}/api/report/${reportId}/pdf`, '_blank')
  }

  return (
    <div className="bg-slate-900/80 border border-slate-800 rounded-2xl p-6 sm:p-8 backdrop-blur-sm shadow-2xl">
      <div className="flex flex-col sm:flex-row items-start sm:items-center justify-between pb-6 border-b border-slate-800 gap-4">
        <div>
          <h2 className="text-xl font-bold text-white flex items-center gap-2">
            Scan Investigation History
            <span className="text-xs px-2.5 py-0.5 rounded-full bg-cyan-950 text-cyan-400 border border-cyan-800/60 font-mono">
              SQLite Audit Log
            </span>
          </h2>
          <p className="text-xs text-slate-400 mt-1">
            Persisted history of all deepfake verification scans and generated forensic artifacts.
          </p>
        </div>

        <button
          onClick={fetchHistory}
          disabled={loading}
          className="px-3.5 py-1.5 bg-slate-800 hover:bg-slate-700 text-slate-200 text-xs font-semibold rounded-lg border border-slate-700 transition-colors flex items-center gap-1.5"
        >
          <svg className={`w-3.5 h-3.5 ${loading ? 'animate-spin' : ''}`} fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15" />
          </svg>
          Refresh Log
        </button>
      </div>

      {loading ? (
        <div className="py-16 text-center text-slate-400">
          <div className="w-8 h-8 border-2 border-cyan-500 border-t-transparent rounded-full animate-spin mx-auto mb-3" />
          <p className="text-sm font-medium">Retrieving scan history from database...</p>
        </div>
      ) : error ? (
        <div className="my-8 p-4 rounded-xl bg-rose-950/30 border border-rose-800/50 text-rose-300 text-sm text-center">
          {error}
        </div>
      ) : history.length === 0 ? (
        <div className="py-16 text-center text-slate-500">
          <svg className="w-12 h-12 mx-auto mb-3 opacity-40" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="1.5" d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
          </svg>
          <p className="text-base font-semibold text-slate-300">No Scans Recorded Yet</p>
          <p className="text-xs text-slate-500 mt-1">Upload an image in the detector to generate your first forensic report.</p>
        </div>
      ) : (
        <div className="mt-6 overflow-x-auto">
          <table className="w-full text-left text-xs border-collapse">
            <thead>
              <tr className="border-b border-slate-800 text-slate-400 uppercase tracking-wider font-mono">
                <th className="py-3 px-3">Report ID</th>
                <th className="py-3 px-3">File Name</th>
                <th className="py-3 px-3">Verdict</th>
                <th className="py-3 px-3">Fake Prob</th>
                <th className="py-3 px-3">Faces</th>
                <th className="py-3 px-3">Latency</th>
                <th className="py-3 px-3">Timestamp</th>
                <th className="py-3 px-3 text-right">Actions</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-800/60 font-mono">
              {history.map((scan) => {
                const normVerdict = String(scan.verdict).toLowerCase()
                let badgeClass = 'text-emerald-400 bg-emerald-950/40 border-emerald-800/60'
                if (normVerdict === 'fake' || normVerdict === 'deepfake') {
                  badgeClass = 'text-rose-400 bg-rose-950/40 border-rose-800/60'
                } else if (normVerdict === 'uncertain') {
                  badgeClass = 'text-amber-400 bg-amber-950/40 border-amber-800/60'
                }

                return (
                  <tr key={scan.report_id} className="hover:bg-slate-800/40 transition-colors">
                    <td className="py-3 px-3 font-bold text-cyan-400">{scan.report_id}</td>
                    <td className="py-3 px-3 text-slate-200 truncate max-w-[140px]" title={scan.filename}>
                      {scan.filename}
                    </td>
                    <td className="py-3 px-3">
                      <span className={`px-2 py-0.5 rounded text-[10px] font-bold uppercase border ${badgeClass}`}>
                        {scan.verdict}
                      </span>
                    </td>
                    <td className="py-3 px-3 font-semibold text-slate-200">
                      {Math.round(scan.fake_probability * 100)}%
                    </td>
                    <td className="py-3 px-3 text-slate-300">
                      {scan.face_detected ? `${scan.total_faces || 1} Face` : 'Full Image'}
                    </td>
                    <td className="py-3 px-3 text-slate-400">
                      {scan.processing_time_ms ? `${scan.processing_time_ms.toFixed(0)} ms` : 'N/A'}
                    </td>
                    <td className="py-3 px-3 text-slate-400 text-[11px]">
                      {scan.created_at ? scan.created_at.slice(0, 16).replace('T', ' ') : 'N/A'}
                    </td>
                    <td className="py-3 px-3 text-right">
                      <button
                        onClick={() => downloadPdf(scan.report_id)}
                        className="px-2.5 py-1 bg-cyan-950/80 hover:bg-cyan-900 text-cyan-300 border border-cyan-800/80 rounded font-sans text-xs transition-colors inline-flex items-center gap-1"
                        title="Download official PDF forensic report"
                      >
                        <svg className="w-3 h-3" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M12 10v6m0 0l-3-3m3 3l3-3m2 8H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
                        </svg>
                        PDF
                      </button>
                    </td>
                  </tr>
                )
              })}
            </tbody>
          </table>
        </div>
      )}
    </div>
  )
}
