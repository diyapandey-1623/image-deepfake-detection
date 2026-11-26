import React, { useState, useEffect } from 'react'
import { useParams, useLocation, Link } from 'react-router-dom'
import { Download, Link as LinkIcon, ArrowLeft, Check, AlertCircle } from 'lucide-react'
import VerdictCard from '../components/VerdictCard'
import ImageCompare from '../components/ImageCompare'
import ReasonsList from '../components/ReasonsList'
import FacesGrid from '../components/FacesGrid'
import ChecksGrid from '../components/ChecksGrid'
import Disclaimer from '../components/Disclaimer'
import { getReportById, getReportPdfUrl } from '../lib/api'

export default function Result() {
  const { id } = useParams()
  const location = useLocation()

  const [result, setResult] = useState(location.state?.result || null)
  const [previewUrl] = useState(location.state?.previewUrl || null)
  const [loading, setLoading] = useState(!result)
  const [error, setError] = useState(null)
  const [copied, setCopied] = useState(false)

  useEffect(() => {
    return () => {
      if (previewUrl && previewUrl.startsWith('blob:')) {
        URL.revokeObjectURL(previewUrl)
      }
    }
  }, [previewUrl])

  useEffect(() => {
    if (!result && id) {
      setLoading(true)
      setError(null)
      getReportById(id)
        .then((data) => {
          if (data) {
            setResult(data)
          } else {
            setError(`Report "${id}" could not be found.`)
          }
        })
        .catch((err) => {
          setError(err.message || 'Failed to retrieve report.')
        })
        .finally(() => setLoading(false))
    }
  }, [id, result])

  const handleCopyLink = () => {
    navigator.clipboard.writeText(window.location.href)
    setCopied(true)
    setTimeout(() => setCopied(false), 2000)
  }

  if (loading) {
    return (
      <div className="max-w-3xl mx-auto px-4 py-20 text-center">
        <div className="w-8 h-8 border-2 border-text border-t-transparent rounded-full animate-spin mx-auto mb-3" />
        <h3 className="font-semibold text-sm text-text">Loading forensic report</h3>
        <p className="text-xs text-text-muted mt-1 font-mono">Retrieving diagnostics and multi-domain metadata.</p>
      </div>
    )
  }

  if (error || !result) {
    return (
      <div className="max-w-md mx-auto px-4 py-16 text-center animate-fade-in">
        <div className="w-10 h-10 rounded-lg bg-verdict-fakeBg text-verdict-fake flex items-center justify-center mx-auto mb-3">
          <AlertCircle className="w-5 h-5" strokeWidth={1.75} />
        </div>
        <h3 className="font-semibold text-base text-text mb-1">Report not found</h3>
        <p className="text-xs text-text-muted mb-5 leading-relaxed">
          {error || "Unable to locate this scan result."}
        </p>
        <Link
          to="/detect"
          className="inline-flex items-center gap-1.5 px-4 py-2 rounded-lg bg-text text-white text-xs font-medium focus-ring"
        >
          <ArrowLeft className="w-3.5 h-3.5" />
          Inspect an image
        </Link>
      </div>
    )
  }

  const pdfUrl = getReportPdfUrl(result.report_id)

  return (
    <div className="max-w-4xl mx-auto px-4 sm:px-6 py-8 sm:py-10 space-y-5 animate-fade-in">
      {/* Navigation & Action Header */}
      <div className="flex flex-col sm:flex-row items-start sm:items-center justify-between gap-3 pb-2">
        <Link
          to="/detect"
          className="inline-flex items-center gap-1.5 text-xs text-text-muted hover:text-text transition-colors focus-ring rounded py-1"
        >
          <ArrowLeft className="w-3.5 h-3.5" />
          Inspect another image
        </Link>

        <div className="flex items-center gap-2 w-full sm:w-auto">
          {/* Copy Link Button */}
          <button
            onClick={handleCopyLink}
            className="flex-1 sm:flex-initial inline-flex items-center justify-center gap-1.5 px-3 py-1.5 rounded-lg bg-card border border-border text-xs font-medium text-text hover:bg-surface-subtle transition-colors shadow-subtle focus-ring"
          >
            {copied ? <Check className="w-3.5 h-3.5 text-verdict-real" /> : <LinkIcon className="w-3.5 h-3.5 text-text-muted" />}
            {copied ? 'Copied' : 'Copy link'}
          </button>

          {/* Download PDF Button */}
          <a
            href={pdfUrl}
            target="_blank"
            rel="noopener noreferrer"
            className="flex-1 sm:flex-initial inline-flex items-center justify-center gap-1.5 px-3.5 py-1.5 rounded-lg bg-text hover:bg-zinc-800 text-white text-xs font-medium transition-colors shadow-subtle focus-ring"
          >
            <Download className="w-3.5 h-3.5" />
            PDF export
          </a>
        </div>
      </div>

      {/* 1. Verdict Card */}
      <VerdictCard result={result} />

      {/* 2. Image Comparison (Original vs Heatmap) */}
      <ImageCompare
        originalUrl={previewUrl || result.image_url}
        heatmapUrl={result.heatmap_base64}
      />

      {/* 3. Forensic Rationale */}
      <ReasonsList result={result} />

      {/* 4. Faces Localized Grid */}
      <FacesGrid
        faces={result.faces}
        faceDetected={result.checks?.face_detected ?? true}
      />

      {/* 5. Technical Diagnostic Checks */}
      <ChecksGrid
        checks={result.checks}
        modelName={result.model_name}
        processingTimeMs={result.processing_time_ms}
      />

      {/* 6. Forensic Notice Disclaimer */}
      <Disclaimer />
    </div>
  )
}
