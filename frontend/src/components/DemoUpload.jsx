import React, { useEffect, useState } from 'react'
import { Link } from 'react-router-dom'
import { AlertCircle, UploadCloud } from 'lucide-react'
import { analyzeImage } from '../lib/api'
import { getVerdictData } from '../lib/verdict'

const PRESET_RESULTS = {
  real: {
    verdict: 'REAL',
    score: 12.3,
    confidence: 87.7,
    metrics: {
      faceDetected: true,
      spectralArtifacts: 'None detected (Natural)',
    },
  },
  fake: {
    verdict: 'FAKE',
    score: 87.4,
    confidence: 87.4,
    metrics: {
      faceDetected: true,
      spectralArtifacts: 'Anomalies detected',
    },
  },
}

export default function DemoUpload() {
  const [previewSrc, setPreviewSrc] = useState(null)
  const [isProcessing, setIsProcessing] = useState(false)
  const [errorMessage, setErrorMessage] = useState(null)
  const [reportId, setReportId] = useState(null)
  const [result, setResult] = useState(PRESET_RESULTS.fake)

  useEffect(() => {
    return () => {
      if (previewSrc && previewSrc.startsWith('blob:')) {
        URL.revokeObjectURL(previewSrc)
      }
    }
  }, [previewSrc])

  const handleFile = async (file) => {
    if (!file) return
    setPreviewSrc(URL.createObjectURL(file))
    setIsProcessing(true)
    setErrorMessage(null)
    setReportId(null)

    try {
      const data = await analyzeImage(file)
      const v = getVerdictData(data.fake_probability, data.verdict)
      const isReal = v.type === 'real'
      const confidenceVal = Math.round((isReal ? 100 - v.percent : v.percent) * 10) / 10

      setReportId(data.report_id)
      setResult({
        verdict: isReal ? 'REAL' : 'FAKE',
        score: Math.round((data.fake_probability ?? 0) * 1000) / 10,
        confidence: Math.max(confidenceVal, 50.0),
        metrics: {
          faceDetected: data.checks?.face_detected ?? false,
          spectralArtifacts: data.checks?.ela_flag ? 'Compression delta flagged' : 'Uniform compression',
        },
      })
    } catch (err) {
      setErrorMessage(err.message || 'Backend inference failed.')
    } finally {
      setIsProcessing(false)
    }
  }

  const loadSample = (kind) => {
    setErrorMessage(null)
    setReportId(null)
    setPreviewSrc(null)
    setResult(kind === 'real' ? PRESET_RESULTS.real : PRESET_RESULTS.fake)
  }

  return (
    <section id="demo-upload" className="bg-[#FFF9F7] text-[#0B1A5C] py-20 sm:py-28 border-b-2 border-[#0B1A5C]">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="text-center max-w-2xl mx-auto mb-12 sm:mb-16">
          <div className="inline-block px-3 py-1 bg-[#0F45E3] text-[#C8FF00] font-mono text-xs font-bold uppercase tracking-widest-mono mb-3 shadow-[2px_2px_0px_#0B1A5C]">
            INTERACTIVE FORENSIC SANDBOX
          </div>
          <h2 className="font-heading font-extrabold text-4xl sm:text-6xl tracking-tightest leading-tight text-[#0B1A5C]">
            Test an image live.
          </h2>
          <p className="font-mono text-xs sm:text-sm text-[#0B1A5C]/75 mt-3">
            Drop your own file or click a preset specimen below to inspect the classifier in real-time.
          </p>

          <div className="flex items-center justify-center gap-3 mt-5 font-mono text-xs">
            <button
              onClick={() => loadSample('fake')}
              className="px-3 py-1.5 bg-white border border-[#0B1A5C] shadow-[2px_2px_0px_#0B1A5C] hover:bg-[#FF4D4D] hover:text-white transition-colors font-bold"
            >
              Preset: Sample Fake
            </button>
            <button
              onClick={() => loadSample('real')}
              className="px-3 py-1.5 bg-white border border-[#0B1A5C] shadow-[2px_2px_0px_#0B1A5C] hover:bg-[#1FD17A] hover:text-[#0B1A5C] transition-colors font-bold"
            >
              Preset: Sample Real
            </button>
          </div>
        </div>

        <div className="max-w-3xl mx-auto">
          <div
            onDragOver={(e) => e.preventDefault()}
            onDrop={(e) => {
              e.preventDefault()
              handleFile(e.dataTransfer.files?.[0])
            }}
            className="relative bg-white border-3 border-dashed border-[#0B1A5C] p-8 sm:p-12 text-center shadow-hard-navy-lg transition-colors hover:bg-[#FFFDFB]"
          >
            <input
              type="file"
              id="face-file-input"
              accept="image/jpeg,image/png,image/webp"
              onChange={(e) => handleFile(e.target.files?.[0])}
              className="hidden"
            />

            <div className="w-16 h-16 mx-auto mb-4 bg-[#C8FF00] border-2 border-[#0B1A5C] shadow-[3px_3px_0px_#0B1A5C] flex items-center justify-center">
              <UploadCloud className="w-8 h-8 text-[#0B1A5C] stroke-[2.5]" />
            </div>

            <h3 className="font-heading font-extrabold text-2xl sm:text-3xl text-[#0B1A5C] tracking-tight mb-2">
              Drag & Drop face image here
            </h3>
            <p className="font-mono text-xs sm:text-sm text-[#0B1A5C]/70 mb-6 max-w-md mx-auto">
              Supports standard formats: JPG, PNG, WEBP (cropped automatically to face bounding box).
            </p>

            <label
              htmlFor="face-file-input"
              className="cursor-pointer inline-flex items-center gap-3 px-8 py-3.5 bg-[#C8FF00] text-[#0B1A5C] font-mono text-sm sm:text-base font-extrabold uppercase tracking-wider border-2 border-[#0B1A5C] shadow-hard-navy btn-press"
            >
              <span>{isProcessing ? 'Analyzing...' : 'Detect Image →'}</span>
            </label>
          </div>

          {errorMessage && (
            <div className="mt-4 p-3.5 bg-red-50 border-2 border-[#FF4D4D] text-[#B91C1C] font-mono text-xs flex items-center gap-2">
              <AlertCircle className="w-4 h-4 shrink-0" />
              <span>{errorMessage}</span>
            </div>
          )}

          <div className="mt-8 bg-white border-3 border-[#0B1A5C] p-6 sm:p-8 shadow-hard-navy-lg">
            <div className="flex items-center justify-between border-b-2 border-[#0B1A5C] pb-4 mb-6">
              <span className="font-mono text-xs font-extrabold uppercase tracking-widest text-[#0B1A5C]">
                MODEL INFERENCE VERDICT
              </span>
              <span className="font-mono text-[11px] bg-[#0B1A5C] text-white px-2.5 py-0.5">
                VIT-BASE-PATCH16-224
              </span>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-12 gap-8 items-center">
              <div className="md:col-span-5 text-center md:text-left border-b md:border-b-0 md:border-r border-[#0B1A5C]/20 pb-6 md:pb-0 md:pr-6">
                <div className="font-mono text-[11px] text-[#0B1A5C]/70 font-bold uppercase mb-1">
                  CLASSIFICATION VERDICT
                </div>

                {result.verdict === 'FAKE' ? (
                  <div className="inline-block bg-[#FF4D4D] text-white font-heading font-extrabold text-5xl sm:text-6xl px-6 py-2 border-3 border-[#0B1A5C] shadow-[4px_4px_0px_#0B1A5C] tracking-tight">
                    FAKE
                  </div>
                ) : (
                  <div className="inline-block bg-[#1FD17A] text-[#0B1A5C] font-heading font-extrabold text-5xl sm:text-6xl px-6 py-2 border-3 border-[#0B1A5C] shadow-[4px_4px_0px_#0B1A5C] tracking-tight">
                    REAL
                  </div>
                )}

                <div className="mt-3 font-mono text-xs text-[#0B1A5C] font-bold">
                  Fake Score: <span className="text-lg">{result.score.toFixed(1)} / 100</span>
                </div>
              </div>

              <div className="md:col-span-7 space-y-4">
                <div>
                  <div className="flex justify-between font-mono text-xs font-bold text-[#0B1A5C] mb-1.5">
                    <span>Confidence Score</span>
                    <span>{result.confidence.toFixed(1)}%</span>
                  </div>
                  <div className="w-full h-5 bg-[#0B1A5C]/10 border-2 border-[#0B1A5C] p-0.5">
                    <div
                      className={`h-full transition-all duration-500 ${
                        result.verdict === 'FAKE' ? 'bg-[#FF4D4D]' : 'bg-[#1FD17A]'
                      }`}
                      style={{ width: `${result.confidence}%` }}
                    />
                  </div>
                </div>

                <div className="grid grid-cols-2 gap-2 pt-2 font-mono text-[11px] text-[#0B1A5C]">
                  <div className="p-2 bg-[#FFF9F7] border border-[#0B1A5C]">
                    <span className="block text-[10px] text-[#0B1A5C]/60">FACE LOCALIZATION</span>
                    <span className="font-bold">
                      {result.metrics.faceDetected ? 'MTCNN Cropped (224px)' : 'Full Frame Fallback'}
                    </span>
                  </div>
                  <div className="p-2 bg-[#FFF9F7] border border-[#0B1A5C]">
                    <span className="block text-[10px] text-[#0B1A5C]/60">ELA INTEGRITY</span>
                    <span className="font-bold">{result.metrics.spectralArtifacts}</span>
                  </div>
                </div>

                {reportId && (
                  <div className="flex items-center justify-end font-mono text-[10px] pt-1">
                    <Link
                      to={`/result/${reportId}`}
                      className="text-[#0F45E3] font-bold underline hover:text-[#0B1A5C]"
                    >
                      View Full Forensic Report →
                    </Link>
                  </div>
                )}
              </div>
            </div>
          </div>
        </div>
      </div>
    </section>
  )
}
