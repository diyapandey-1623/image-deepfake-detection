import React, { useState } from 'react'
import { Eye, Activity } from 'lucide-react'

export default function ImageCompare({ originalUrl, heatmapUrl }) {
  const [blend, setBlend] = useState(50) // 0: 100% original, 100: 100% heatmap
  const [activeTab, setActiveTab] = useState('split') // 'split' or 'slider'

  if (!originalUrl) return null

  if (!heatmapUrl) {
    return (
      <div className="bg-card border border-border rounded-xl p-5 sm:p-6 shadow-subtle">
        <h3 className="font-semibold text-sm text-text mb-1">
          Analyzed image
        </h3>
        <p className="text-xs text-text-muted mb-4">
          Visual input analyzed by the model.
        </p>
        <div className="max-w-md mx-auto rounded-lg overflow-hidden border border-border bg-surface-subtle flex items-center justify-center p-2">
          <img src={originalUrl} alt="Original" className="max-h-72 w-auto object-contain rounded" />
        </div>
        <p className="text-[11px] text-text-muted text-center mt-3 font-mono">
          Note: Grad-CAM heatmap overlay not available for this backbone.
        </p>
      </div>
    )
  }

  return (
    <div className="bg-card border border-border rounded-xl p-5 sm:p-6 shadow-subtle">
      <div className="flex flex-col sm:flex-row items-start sm:items-center justify-between pb-4 gap-3 border-b border-border">
        <div>
          <h3 className="font-semibold text-sm sm:text-base text-text">
            Visual inspection & Grad-CAM heatmap
          </h3>
          <p className="text-xs text-text-muted mt-0.5">
            Compare original facial features against neural activation focus regions.
          </p>
        </div>

        {/* Display Mode Toggle */}
        <div className="flex bg-surface-subtle p-0.5 rounded-lg border border-border text-xs font-medium">
          <button
            onClick={() => setActiveTab('split')}
            className={`px-3 py-1 rounded-md transition-colors ${
              activeTab === 'split' ? 'bg-white text-text font-semibold shadow-subtle' : 'text-text-muted hover:text-text'
            }`}
          >
            Side by side
          </button>
          <button
            onClick={() => setActiveTab('slider')}
            className={`px-3 py-1 rounded-md transition-colors ${
              activeTab === 'slider' ? 'bg-white text-text font-semibold shadow-subtle' : 'text-text-muted hover:text-text'
            }`}
          >
            Blend slider
          </button>
        </div>
      </div>

      {activeTab === 'split' ? (
        /* Side by Side Mode */
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4 pt-4">
          {/* Original Card */}
          <div className="flex flex-col">
            <span className="text-xs font-medium text-text mb-2 flex items-center gap-1.5">
              <Eye className="w-3.5 h-3.5 text-text-muted" />
              Original input
            </span>
            <div className="w-full h-64 sm:h-72 rounded-lg overflow-hidden border border-border bg-surface-subtle flex items-center justify-center p-2">
              <img
                src={originalUrl}
                alt="Original uploaded"
                className="max-h-full max-w-full object-contain rounded"
              />
            </div>
            <span className="text-[11px] font-mono text-text-muted mt-2 text-center">Unmodified facial photograph</span>
          </div>

          {/* Heatmap Card */}
          <div className="flex flex-col">
            <span className="text-xs font-medium text-text mb-2 flex items-center gap-1.5">
              <Activity className="w-3.5 h-3.5 text-text" />
              Grad-CAM activation overlay
            </span>
            <div className="w-full h-64 sm:h-72 rounded-lg overflow-hidden border border-border bg-surface-subtle flex items-center justify-center p-2">
              <img
                src={heatmapUrl}
                alt="Grad-CAM activation overlay"
                className="max-h-full max-w-full object-contain rounded"
              />
            </div>
            <span className="text-[11px] font-mono text-text-muted mt-2 text-center">
              <span className="text-red-500 font-bold">Red: High (&ge;50%)</span> &middot; <span className="text-yellow-600 font-bold">Yellow: Moderate (25-50%)</span> &middot; <span className="text-blue-500 font-bold">Blue: Low (&lt;25%)</span>
            </span>
          </div>
        </div>
      ) : (
        /* Blend Slider Mode */
        <div className="pt-4 max-w-lg mx-auto space-y-4">
          <div className="relative w-full h-72 sm:h-80 rounded-lg overflow-hidden border border-border bg-surface-subtle flex items-center justify-center p-2">
            <img
              src={originalUrl}
              alt="Original"
              className="absolute max-h-full max-w-full object-contain"
            />
            <img
              src={heatmapUrl}
              alt="Grad-CAM"
              className="absolute max-h-full max-w-full object-contain transition-opacity duration-75"
              style={{ opacity: blend / 100 }}
            />
          </div>

          <div className="space-y-1.5">
            <div className="flex justify-between text-xs text-text-muted font-mono">
              <span>Original (0%)</span>
              <span className="text-text font-medium">Blend: {blend}%</span>
              <span>Heatmap (100%)</span>
            </div>
            <input
              type="range"
              min="0"
              max="100"
              value={blend}
              onChange={(e) => setBlend(Number(e.target.value))}
              className="w-full accent-text cursor-pointer"
            />
          </div>
        </div>
      )}
    </div>
  )
}
