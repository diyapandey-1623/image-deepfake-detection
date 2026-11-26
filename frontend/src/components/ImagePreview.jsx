import React from 'react'
import { Trash2, ArrowRight, Check } from 'lucide-react'
import { formatBytes } from '../lib/format'

export default function ImagePreview({ file, previewUrl, onAnalyze, onRemove, disabled = false }) {
  if (!file || !previewUrl) return null

  return (
    <div className="bg-card border border-border rounded-xl p-4 sm:p-5 w-full max-w-xl mx-auto shadow-subtle animate-fade-in">
      <div className="flex flex-col sm:flex-row items-center gap-4">
        {/* Thumbnail Viewport */}
        <div className="w-24 h-24 sm:w-28 sm:h-28 rounded-lg overflow-hidden bg-surface-subtle border border-border shrink-0 flex items-center justify-center">
          <img
            src={previewUrl}
            alt={file.name || 'Selected file'}
            className="w-full h-full object-cover"
          />
        </div>

        {/* File Meta Info */}
        <div className="flex-1 text-center sm:text-left min-w-0 w-full">
          <div className="flex items-center justify-center sm:justify-start gap-1.5 text-xs font-medium text-verdict-real mb-1">
            <Check className="w-3.5 h-3.5" strokeWidth={2.2} />
            Ready for forensic inspection
          </div>
          <h4 className="font-medium text-sm text-text truncate" title={file.name}>
            {file.name}
          </h4>
          <p className="text-xs text-text-muted font-mono mt-0.5">
            {formatBytes(file.size)} · {file.type || 'image/jpeg'}
          </p>

          <div className="flex items-center justify-center sm:justify-start gap-2 mt-3.5">
            <button
              onClick={onAnalyze}
              disabled={disabled}
              className="inline-flex items-center justify-center gap-1.5 px-4 py-2 rounded-lg bg-text hover:bg-zinc-800 text-white text-xs font-medium transition-colors focus-ring disabled:opacity-50"
            >
              Run inspection
              <ArrowRight className="w-3.5 h-3.5" />
            </button>

            <button
              onClick={onRemove}
              disabled={disabled}
              className="inline-flex items-center justify-center gap-1.5 px-3 py-2 rounded-lg bg-surface-subtle hover:bg-zinc-200 text-text-muted hover:text-verdict-fake text-xs font-medium transition-colors focus-ring disabled:opacity-50"
              title="Remove this image and choose another"
            >
              <Trash2 className="w-3.5 h-3.5" />
              Remove
            </button>
          </div>
        </div>
      </div>
    </div>
  )
}
