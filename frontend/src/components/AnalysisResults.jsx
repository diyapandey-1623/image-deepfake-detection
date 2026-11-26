import React, { useState } from 'react'
import ScoreBar from './ScoreBar'

export default function AnalysisResults({ results, previewUrl, onReset, apiBase = '' }) {
  if (!results) return null

  const {
    report_id,
    verdict = 'uncertain',
    confidence,
    fake_probability,
    probability_fake,
    faces = [],
    heatmap_base64,
    checks = {},
    model_name,
    scores,
    inconsistency_score,
    metadata,
    warnings = [],
    visuals
  } = results

  const [activeTab, setActiveTab] = useState('scores') // 'scores' or 'inspector'

  // Normalize fake probability & confidence
  const rawFakeProb = fake_probability ?? probability_fake ?? 0.5
  const rawConf = confidence !== undefined ? confidence : Math.max(rawFakeProb, 1 - rawFakeProb)
  const confPercent = Math.round(rawConf * 100)
  const fakePercent = Math.round(rawFakeProb * 100)

  // Verdict normalization (lowercase or uppercase)
  const normVerdict = String(verdict).toLowerCase()

  // Badge styling based on verdict
  let badgeStyle = 'bg-emerald-500/20 border-emerald-500/40 text-emerald-400 shadow-emerald-950/40'
  let badgeLabel = 'REAL / AUTHENTIC'
  let badgeSubtitle = 'No significant generative manipulation signatures detected.'

  if (normVerdict === 'fake' || normVerdict === 'deepfake') {
    badgeStyle = 'bg-rose-500/20 border-rose-500/40 text-rose-400 shadow-rose-950/40'
    badgeLabel = 'DEEPFAKE / SYNTHETIC'
    badgeSubtitle = 'Deepfake or AI-generated manipulation patterns detected.'
  } else if (normVerdict === 'uncertain') {
    badgeStyle = 'bg-amber-500/20 border-amber-500/40 text-amber-300 shadow-amber-950/40'
    badgeLabel = 'UNCERTAIN / BORDERLINE'
    badgeSubtitle = 'Confidence falls within the indeterminate threshold band (0.40 - 0.60).'
  }

  const downloadPdf = () => {
    if (report_id) {
      window.open(`${apiBase}/api/report/${report_id}/pdf`, '_blank')
    }
  }

  return (
    <div className="bg-slate-900/80 border border-slate-800 rounded-2xl p-6 sm:p-8 backdrop-blur-sm shadow-2xl">
      {/* Top Header: Verdict & Primary Confidence */}
      <div className="flex flex-col lg:flex-row items-start lg:items-center justify-between gap-6 pb-6 border-b border-slate-800">
        <div className="flex items-center gap-4">
          <div className={`px-5 py-2.5 rounded-xl border text-sm sm:text-base font-extrabold tracking-wider uppercase shadow-lg ${badgeStyle}`}>
            {badgeLabel}
          </div>
          <div>
            <div className="flex items-baseline gap-2">
              <span className="text-3xl font-extrabold font-mono text-slate-100">{confPercent}%</span>
              <span className="text-xs uppercase tracking-wider font-semibold text-slate-400">Confidence</span>
              <span className="text-xs font-mono text-slate-500 ml-2">({fakePercent}% Fake Prob)</span>
            </div>
            <p className="text-xs text-slate-400 mt-0.5">{badgeSubtitle}</p>
          </div>
        </div>

        <div className="flex flex-wrap items-center gap-3">
          {/* Tab Selector */}
          <div className="flex bg-slate-950/80 p-1 rounded-xl border border-slate-800">
            <button
              onClick={() => setActiveTab('scores')}
              className={`px-3.5 py-1.5 text-xs font-semibold rounded-lg transition-colors ${
                activeTab === 'scores' ? 'bg-cyan-500 text-slate-950 font-bold' : 'text-slate-400 hover:text-slate-200'
              }`}
            >
              Analysis Breakdown
            </button>
            <button
              onClick={() => setActiveTab('inspector')}
              className={`px-3.5 py-1.5 text-xs font-semibold rounded-lg transition-colors ${
                activeTab === 'inspector' ? 'bg-cyan-500 text-slate-950 font-bold' : 'text-slate-400 hover:text-slate-200'
              }`}
            >
              Original vs Heatmap
            </button>
          </div>

          {/* PDF Download Button */}
          {report_id && (
            <button
              onClick={downloadPdf}
              className="px-3.5 py-2 bg-cyan-950/80 hover:bg-cyan-900 text-cyan-300 text-sm font-semibold rounded-xl border border-cyan-800/80 transition-colors flex items-center gap-2"
              title="Download formal forensic PDF report"
            >
              <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M12 10v6m0 0l-3-3m3 3l3-3m2 8H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
              </svg>
              Download PDF
            </button>
          )}

          <button
            onClick={onReset}
            className="px-4 py-2 bg-slate-800 hover:bg-slate-700 text-slate-200 text-sm font-semibold rounded-xl border border-slate-700 transition-colors flex items-center gap-2"
          >
            <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15" />
            </svg>
            Analyze Another
          </button>
        </div>
      </div>

      {activeTab === 'scores' ? (
        /* Breakdown Tab */
        <div className="grid grid-cols-1 lg:grid-cols-12 gap-6 mt-6">
          {/* Left Column: Image preview + face bbox overlay */}
          <div className="lg:col-span-5 flex flex-col gap-4">
            <div className="relative rounded-xl overflow-hidden border border-slate-800 bg-black/40 flex items-center justify-center min-h-[260px]">
              {previewUrl ? (
                <img src={previewUrl} alt="Analyzed Target" className="w-full max-h-[360px] object-contain" />
              ) : (
                <div className="text-slate-500 text-sm">Image preview unavailable</div>
              )}
              <div className="absolute bottom-2 left-2 px-2.5 py-1 bg-black/70 backdrop-blur-md rounded-md text-[11px] font-mono text-slate-300 border border-slate-700/50">
                {checks.face_detected ? `MTCNN Face Detected (${faces.length} face${faces.length === 1 ? '' : 's'})` : 'Full Image Scan (No Face)'}
              </div>
            </div>

            {/* Metadata Card */}
            <div className="bg-slate-950/60 border border-slate-800/80 rounded-xl p-3.5 text-xs text-slate-300 font-mono space-y-1.5">
              {report_id && (
                <div className="flex justify-between">
                  <span className="text-slate-400">Report ID:</span>
                  <span className="text-cyan-400">{report_id}</span>
                </div>
              )}
              {model_name && (
                <div className="flex justify-between">
                  <span className="text-slate-400">Active Model:</span>
                  <span className="text-slate-200 truncate max-w-[200px]" title={model_name}>{model_name}</span>
                </div>
              )}
              <div className="flex justify-between">
                <span className="text-slate-400">Face Detected:</span>
                <span className={checks.face_detected ? 'text-emerald-400' : 'text-amber-400'}>
                  {checks.face_detected ? 'Yes (+20% margin crop)' : 'No (Analyzed Full Image)'}
                </span>
              </div>
              <div className="flex justify-between">
                <span className="text-slate-400">ELA Modification Flag:</span>
                <span className={checks.ela_flag ? 'text-rose-400' : 'text-emerald-400'}>
                  {checks.ela_flag ? 'Flagged (compression delta)' : 'Normal'}
                </span>
              </div>
              <div className="flex justify-between">
                <span className="text-slate-400">Metadata / EXIF Tags:</span>
                <span>{checks.metadata_flags && checks.metadata_flags.length > 0 ? checks.metadata_flags.join(', ') : 'None / Clean'}</span>
              </div>
            </div>
          </div>

          {/* Right Column: Per-Modality / Component Breakdown */}
          <div className="lg:col-span-7 flex flex-col gap-3.5">
            <h4 className="text-xs font-bold uppercase tracking-wider text-slate-400">
              Model Inference & Forensic Checks
            </h4>

            {scores ? (
              <>
                <ScoreBar
                  label="Spatial RGB Stream"
                  score={scores.spatial ?? rawFakeProb}
                  description="Deep CNN features extracted from facial appearance looking for unnatural artifacts."
                />
                <ScoreBar
                  label="Frequency 2D FFT Stream"
                  score={scores.frequency ?? 0.5}
                  description="Fourier spectral analysis exposing periodic grid spikes and abnormal roll-off."
                />
                <ScoreBar
                  label="Handcrafted & ELA Forensics"
                  score={scores.forensic ?? 0.5}
                  description="Physics-grounded features: Error Level Analysis and chrominance consistency."
                />
                <ScoreBar
                  label="Cross-Modal Fusion Score"
                  score={scores.fusion ?? rawFakeProb}
                  description="Cross-attention fusion evaluating inter-stream consistency."
                />
              </>
            ) : (
              <>
                <ScoreBar
                  label="Deepfake Probability (Primary Model)"
                  score={rawFakeProb}
                  description={`Softmax output probability computed by ${model_name || 'deep learning backbone'}.`}
                />
                <ScoreBar
                  label="Authentic / Real Probability"
                  score={1.0 - rawFakeProb}
                  description="Inverse probability reflecting natural biometric consistency."
                />
                {faces.map((f, idx) => (
                  <ScoreBar
                    key={idx}
                    label={`Face #${idx + 1} Detection BBox [${f.bbox.join(', ')}]`}
                    score={f.fake_probability}
                    description={`Localized crop probability for detected face #${idx + 1}.`}
                  />
                ))}
              </>
            )}
          </div>
        </div>
      ) : (
        /* Explainability / Visual Inspector Tab */
        <div className="mt-6">
          <div className="mb-4">
            <h4 className="text-sm font-bold text-slate-200">Model Explainability & Grad-CAM Heatmap</h4>
            <p className="text-xs text-slate-400">
              Visual activation heatmaps highlight the salient spatial regions that influenced the model's classification decision.
            </p>
          </div>

          {heatmap_base64 ? (
            <div className="space-y-6">
              {/* Side-by-side Original vs Grad-CAM Heatmap */}
              <div className="grid grid-cols-1 md:grid-cols-2 gap-6 bg-slate-950/80 border border-slate-800 rounded-xl p-5">
                <div className="flex flex-col items-center">
                  <span className="text-xs font-semibold text-slate-300 mb-2">Original Analyzed Image</span>
                  {previewUrl ? (
                    <img
                      src={previewUrl}
                      alt="Original Target"
                      className="max-h-[280px] w-auto object-contain rounded-lg border border-slate-700 shadow-md"
                    />
                  ) : (
                    <div className="h-[240px] flex items-center justify-center text-slate-500 text-xs">Preview unavailable</div>
                  )}
                  <span className="text-[11px] text-slate-400 mt-2 text-center">Input face crop or full image feed</span>
                </div>

                <div className="flex flex-col items-center">
                  <span className="text-xs font-semibold text-cyan-400 mb-2">Grad-CAM Explainability Heatmap</span>
                  <img
                    src={heatmap_base64}
                    alt="Grad-CAM Heatmap"
                    className="max-h-[280px] w-auto object-contain rounded-lg border border-cyan-800/80 shadow-md shadow-cyan-950/50"
                  />
                  <span className="text-[11px] text-slate-400 mt-2 text-center">
                    Warmer colors (red/yellow) highlight regions driving the model's decision.
                  </span>
                </div>
              </div>

              {/* Forensic Interpretation & Reasons */}
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4 text-xs font-mono">
                <div className="p-4 bg-slate-950/60 rounded-xl border border-slate-800 space-y-2">
                  <div className="font-bold text-slate-200 uppercase tracking-wider text-[11px]">Primary Decision Reasons:</div>
                  <p className="text-slate-300 font-sans leading-relaxed">
                    {normVerdict === 'fake' || normVerdict === 'deepfake'
                      ? `Model detected high-probability synthetic artifacts (${fakePercent}%). Attention highlights unnatural blending gradients or boundary seams.`
                      : normVerdict === 'real'
                      ? `Authentic biometric skin textures and coherent lighting verified (${100 - fakePercent}% authentic probability). No anomalous activation hotspots.`
                      : `Signals fall in the uncertain transition zone (${fakePercent}%). Discrepancies may stem from compression or sensor noise.`}
                  </p>
                </div>

                <div className="p-4 bg-slate-950/60 rounded-xl border border-slate-800 space-y-2">
                  <div className="font-bold text-slate-200 uppercase tracking-wider text-[11px]">Technical Checks Summary:</div>
                  <ul className="text-slate-300 font-sans space-y-1">
                    <li>• <b>Face Localization:</b> {checks.face_detected ? `MTCNN cropped face (+20% margin)` : `Full image fallback (no face)`}</li>
                    <li>• <b>Error Level Analysis:</b> {checks.ela_flag ? `Flagged (compression delta at Q=90)` : `Normal (uniform quantization)`}</li>
                    <li>• <b>Digital Fingerprints:</b> {checks.metadata_flags && checks.metadata_flags.length > 0 ? checks.metadata_flags.join(', ') : `Clean (no editing tags)`}</li>
                    <li>• <b>Active Backbone:</b> {model_name || 'Standardized deep learning architecture'}</li>
                  </ul>
                </div>
              </div>
            </div>
          ) : (
            <div className="text-slate-400 text-sm py-12 text-center bg-slate-950/50 rounded-xl border border-slate-800 space-y-2">
              <p className="font-semibold text-slate-300">
                {model_name && model_name.includes('vit')
                  ? 'Grad-CAM is not applicable to Vision Transformer (ViT) patch attention backbones.'
                  : 'Heatmap visualization is unavailable for this session or model architecture.'}
              </p>
              <p className="text-xs text-slate-500">
                Classification is based on full biometric feature embeddings.
              </p>
            </div>
          )}
        </div>
      )}

      {/* Warnings & Metadata Flags */}
      {checks.metadata_flags && checks.metadata_flags.length > 0 && (
        <div className="mt-6 p-4 rounded-xl bg-amber-950/20 border border-amber-800/40 text-amber-200/90 text-xs space-y-1">
          <div className="font-bold flex items-center gap-1.5 text-amber-400">
            <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z" />
            </svg>
            Forensic Metadata Observations:
          </div>
          {checks.metadata_flags.map((flag, idx) => (
            <div key={idx} className="pl-5 text-slate-300">• {flag}</div>
          ))}
        </div>
      )}
    </div>
  )
}
