import React, { useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { AlertCircle } from 'lucide-react'
import Uploader from '../components/Uploader'
import ImagePreview from '../components/ImagePreview'
import ProgressSteps from '../components/ProgressSteps'
import { analyzeImage } from '../lib/api'

export default function Home() {
  const navigate = useNavigate()
  const [selectedFile, setSelectedFile] = useState(null)
  const [previewUrl, setPreviewUrl] = useState(null)
  const [isAnalyzing, setIsAnalyzing] = useState(false)
  const [errorMessage, setErrorMessage] = useState(null)
  const [abortController, setAbortController] = useState(null)

  const handleFileSelect = (file) => {
    setSelectedFile(file)
    setErrorMessage(null)
    const url = URL.createObjectURL(file)
    setPreviewUrl(url)
  }

  const handleRemove = () => {
    if (previewUrl) {
      URL.revokeObjectURL(previewUrl)
    }
    setSelectedFile(null)
    setPreviewUrl(null)
    setErrorMessage(null)
  }

  const handleCancel = () => {
    if (abortController) {
      abortController.abort()
    }
    setIsAnalyzing(false)
  }

  const handleAnalyze = async () => {
    if (!selectedFile) return

    setIsAnalyzing(true)
    setErrorMessage(null)
    const controller = new AbortController()
    setAbortController(controller)

    try {
      const result = await analyzeImage(selectedFile, { signal: controller.signal })
      navigate(`/result/${result.report_id}`, {
        state: {
          result,
          previewUrl,
          filename: selectedFile.name,
        },
      })
    } catch (err) {
      if (err.name !== 'AbortError') {
        setErrorMessage(err.message || 'Inspection failed. Please verify the backend connection.')
      }
    } finally {
      setIsAnalyzing(false)
      setAbortController(null)
    }
  }

  return (
    <div className="max-w-3xl mx-auto px-4 sm:px-6 py-12 sm:py-16">
      {/* Hero Section */}
      <div className="max-w-2xl mx-auto mb-10 text-center">
        <h1 className="text-2xl sm:text-3xl font-semibold text-text tracking-tight mb-3">
          Forensic facial image verification
        </h1>
        <p className="text-xs sm:text-sm text-text-muted leading-relaxed max-w-xl mx-auto">
          Multi-domain inspection evaluating spatial facial features, 2D Fourier power spectra, and Error Level Analysis (ELA) to detect synthetic generation and composite manipulation.
        </p>
      </div>

      {/* Error Alert */}
      {errorMessage && (
        <div className="mb-6 p-3.5 rounded-lg bg-verdict-fakeBg border border-verdict-fakeBorder text-verdict-fake text-xs flex items-start justify-between gap-3 animate-fade-in max-w-xl mx-auto">
          <div className="flex items-start gap-2.5">
            <AlertCircle className="w-4 h-4 shrink-0 mt-0.5" />
            <div>
              <p className="font-medium">Analysis error</p>
              <p className="mt-0.5 opacity-90">{errorMessage}</p>
            </div>
          </div>
          <button
            onClick={() => setErrorMessage(null)}
            className="text-verdict-fake font-semibold hover:underline shrink-0 text-xs focus-ring"
          >
            Dismiss
          </button>
        </div>
      )}

      {/* Main Upload / Analyzing Interaction Card */}
      <div className="max-w-xl mx-auto mb-10">
        {isAnalyzing ? (
          <ProgressSteps previewUrl={previewUrl} onCancel={handleCancel} />
        ) : selectedFile ? (
          <ImagePreview
            file={selectedFile}
            previewUrl={previewUrl}
            onAnalyze={handleAnalyze}
            onRemove={handleRemove}
            disabled={isAnalyzing}
          />
        ) : (
          <Uploader onFileSelect={handleFileSelect} disabled={isAnalyzing} />
        )}
      </div>

      {/* Forensic Pipeline Pillars */}
      <div className="max-w-2xl mx-auto grid grid-cols-1 sm:grid-cols-3 gap-4 pt-6 border-t border-border text-left">
        <div className="p-3.5 rounded-lg bg-card border border-border">
          <div className="text-[11px] font-mono text-text-muted mb-1">01 · Spatial CNN</div>
          <div className="text-xs font-medium text-text mb-1">Appearance consistency</div>
          <p className="text-[11px] text-text-muted leading-normal">
            Extracts blending seams, unnatural skin textures, and facial boundary margins.
          </p>
        </div>

        <div className="p-3.5 rounded-lg bg-card border border-border">
          <div className="text-[11px] font-mono text-text-muted mb-1">02 · Fourier 2D</div>
          <div className="text-xs font-medium text-text mb-1">Spectral frequency</div>
          <p className="text-[11px] text-text-muted leading-normal">
            Detects deconvolution grid artifacts and high-frequency power spectrum decay anomalies.
          </p>
        </div>

        <div className="p-3.5 rounded-lg bg-card border border-border">
          <div className="text-[11px] font-mono text-text-muted mb-1">03 · Cross-Attention</div>
          <div className="text-xs font-medium text-text mb-1">Multi-domain fusion</div>
          <p className="text-[11px] text-text-muted leading-normal">
            Bidirectional cross-attention validates whether spatial cues match frequency fingerprints.
          </p>
        </div>
      </div>
    </div>
  )
}
