import React, { useState } from 'react'
import { Link } from 'react-router-dom'
import { Search, FileText, ArrowRight } from 'lucide-react'
import { getVerdictData, isFakeVerdict } from '../lib/verdict'
import { formatDateTime } from '../lib/format'
import { getReportPdfUrl } from '../lib/api'

export default function ScanTable({ scans = [] }) {
  const [search, setSearch] = useState('')
  const [filterVerdict, setFilterVerdict] = useState('all')

  const filtered = scans.filter((scan) => {
    const isFake = isFakeVerdict(scan.fake_probability, scan.verdict)
    if (filterVerdict === 'fake' && !isFake) return false
    if (filterVerdict === 'real' && isFake) return false

    if (search.trim()) {
      const q = search.toLowerCase()
      const matchName = String(scan.filename || '').toLowerCase().includes(q)
      const matchId = String(scan.report_id || '').toLowerCase().includes(q)
      return matchName || matchId
    }

    return true
  })

  return (
    <div className="bg-card border border-border rounded-xl shadow-subtle overflow-hidden">
      {/* Search and Filters Header */}
      <div className="p-3.5 sm:p-4 border-b border-border flex flex-col sm:flex-row items-stretch sm:items-center justify-between gap-2.5 bg-surface-subtle/50">
        <div className="relative flex-1 max-w-sm">
          <Search className="w-3.5 h-3.5 absolute left-3 top-1/2 -translate-y-1/2 text-text-muted" />
          <input
            type="text"
            value={search}
            onChange={(e) => setSearch(e.target.value)}
            placeholder="Search by filename or report ID..."
            className="w-full pl-8 pr-3 py-1.5 text-xs rounded-lg border border-border bg-white text-text focus-ring font-mono"
          />
        </div>

        <div className="flex items-center gap-1.5 overflow-x-auto">
          {['all', 'fake', 'real'].map((type) => (
            <button
              key={type}
              onClick={() => setFilterVerdict(type)}
              className={`px-2.5 py-1 rounded-md text-xs font-medium capitalize transition-colors focus-ring whitespace-nowrap ${
                filterVerdict === type
                  ? 'bg-text text-white font-medium'
                  : 'bg-white text-text-muted hover:text-text border border-border'
              }`}
            >
              {type === 'all' ? 'All records' : type}
            </button>
          ))}
        </div>
      </div>

      {filtered.length === 0 ? (
        /* Empty State */
        <div className="py-14 px-4 text-center">
          <div className="w-10 h-10 rounded-lg bg-surface-subtle border border-border flex items-center justify-center mx-auto mb-2.5 text-text-muted">
            <FileText className="w-5 h-5" strokeWidth={1.75} />
          </div>
          <h3 className="font-medium text-sm text-text mb-1">
            No matching scan records
          </h3>
          <p className="text-xs text-text-muted max-w-sm mx-auto mb-4">
            {scans.length === 0
              ? 'No inspections recorded yet. Upload an image to view forensic results.'
              : 'Try adjusting the search query or verdict filters.'}
          </p>
          <Link
            to="/detect"
            className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-text hover:bg-zinc-800 text-white text-xs font-medium transition-colors shadow-subtle focus-ring"
          >
            Inspect an image
            <ArrowRight className="w-3.5 h-3.5" />
          </Link>
        </div>
      ) : (
        <>
          {/* Desktop Table View */}
          <div className="hidden md:block overflow-x-auto">
            <table className="w-full text-left text-xs border-collapse">
              <thead>
                <tr className="border-b border-border text-text-muted font-medium bg-surface-subtle/50 text-[11px]">
                  <th className="py-2.5 px-3.5">Filename</th>
                  <th className="py-2.5 px-3.5">Verdict</th>
                  <th className="py-2.5 px-3.5">Probability</th>
                  <th className="py-2.5 px-3.5">Faces</th>
                  <th className="py-2.5 px-3.5">Timestamp</th>
                  <th className="py-2.5 px-3.5 text-right">Actions</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-border font-mono">
                {filtered.map((scan) => {
                  const v = getVerdictData(scan.fake_probability, scan.verdict)
                  const facesCount = scan.total_faces ?? scan.faces?.length ?? (scan.face_detected || scan.checks?.face_detected ? 1 : 0)
                  const pdfUrl = getReportPdfUrl(scan.report_id)

                  return (
                    <tr key={scan.report_id} className="hover:bg-surface-subtle/40 transition-colors">
                      <td className="py-2.5 px-3.5">
                        <div className="font-sans font-medium text-text truncate max-w-[200px]" title={scan.filename}>
                          {scan.filename || 'Untitled'}
                        </div>
                        <div className="text-[10px] text-text-muted">
                          {scan.report_id?.slice(0, 8)}
                        </div>
                      </td>

                      <td className="py-2.5 px-3.5 font-sans">
                        <span
                          className={`inline-flex items-center px-2 py-0.5 rounded text-[11px] font-medium border ${v.bgColor} ${v.textColor} ${v.borderColor}`}
                        >
                          {v.label}
                        </span>
                      </td>

                      <td className="py-2.5 px-3.5 font-medium text-text tabular-nums">
                        {v.percent}%
                      </td>

                      <td className="py-2.5 px-3.5 text-text-muted font-sans">
                        {facesCount} {facesCount === 1 ? 'face' : 'faces'}
                      </td>

                      <td className="py-2.5 px-3.5 text-text-muted text-[11px]">
                        {formatDateTime(scan.created_at)}
                      </td>

                      <td className="py-2.5 px-3.5 text-right font-sans">
                        <div className="inline-flex items-center gap-1.5">
                          <Link
                            to={`/result/${scan.report_id}`}
                            className="px-2 py-1 rounded text-text hover:bg-surface-subtle text-xs transition-colors"
                          >
                            View
                          </Link>
                          <a
                            href={pdfUrl}
                            target="_blank"
                            rel="noopener noreferrer"
                            className="px-2 py-1 rounded text-text-muted hover:text-text hover:bg-surface-subtle text-xs transition-colors"
                          >
                            PDF
                          </a>
                        </div>
                      </td>
                    </tr>
                  )
                })}
              </tbody>
            </table>
          </div>

          {/* Mobile Card List View */}
          <div className="md:hidden divide-y divide-border">
            {filtered.map((scan) => {
              const v = getVerdictData(scan.fake_probability, scan.verdict)
              return (
                <div key={scan.report_id} className="p-3.5 space-y-2">
                  <div className="flex items-start justify-between gap-2">
                    <div className="min-w-0">
                      <h4 className="font-medium text-xs text-text truncate">
                        {scan.filename || 'Untitled'}
                      </h4>
                      <p className="text-[10px] font-mono text-text-muted">
                        ID: {scan.report_id?.slice(0, 8)} · {formatDateTime(scan.created_at)}
                      </p>
                    </div>
                    <span
                      className={`inline-flex items-center px-2 py-0.5 rounded text-[11px] font-medium border shrink-0 ${v.bgColor} ${v.textColor} ${v.borderColor}`}
                    >
                      {v.label} ({v.percent}%)
                    </span>
                  </div>

                  <div className="flex items-center justify-end gap-2 pt-1">
                    <Link
                      to={`/result/${scan.report_id}`}
                      className="px-2.5 py-1 rounded bg-surface-subtle text-text text-xs font-medium"
                    >
                      View report
                    </Link>
                  </div>
                </div>
              )
            })}
          </div>
        </>
      )}
    </div>
  )
}
