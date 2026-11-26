import React, { useState } from 'react'
import { CheckCircle2, AlertTriangle, HelpCircle, Cpu, Clock, Layers, FileCode, CheckSquare } from 'lucide-react'

export default function ChecksGrid({ checks = {}, modelName = '', processingTimeMs = 0 }) {
  const [tooltip, setTooltip] = useState(null)

  const items = [
    {
      id: 'face',
      label: 'Face localization (MTCNN)',
      value: checks.face_detected ? 'Face localized (+20% margin)' : 'Full image fallback',
      status: checks.face_detected ? 'pass' : 'neutral',
      icon: Layers,
      tooltip: 'MTCNN searches for human eye, nose, and mouth landmarks to extract standardized crops.',
    },
    {
      id: 'ela',
      label: 'Error Level Analysis (ELA)',
      value: checks.ela_flag ? 'Flagged compression delta' : 'Uniform compression residual',
      status: checks.ela_flag ? 'warn' : 'pass',
      icon: CheckSquare,
      tooltip: 'Measures compression differences across the image to reveal spliced or pasted regions.',
    },
    {
      id: 'metadata',
      label: 'Metadata & EXIF inspection',
      value: checks.metadata_flags && checks.metadata_flags.length > 0 ? checks.metadata_flags.join(', ') : 'Clean camera tags',
      status: checks.metadata_flags && checks.metadata_flags.length > 0 ? 'warn' : 'pass',
      icon: FileCode,
      tooltip: 'Audits header information for software modification signatures like Photoshop or AI stamps.',
    },
    {
      id: 'model',
      label: 'Neural backbone',
      value: modelName || 'Multimodal cross-attention',
      status: 'pass',
      icon: Cpu,
      tooltip: 'The specific machine learning model currently active for computing latent feature representations.',
    },
    {
      id: 'time',
      label: 'Analysis latency',
      value: processingTimeMs ? `${processingTimeMs.toFixed(0)} ms` : 'Fast (<150 ms)',
      status: 'pass',
      icon: Clock,
      tooltip: 'End-to-end processing time taken by the server to inspect the image and return the verdict.',
    },
  ]

  return (
    <div className="bg-card border border-border rounded-xl p-5 sm:p-6 shadow-subtle">
      <h3 className="font-semibold text-sm sm:text-base text-text mb-0.5">
        Technical forensic checks
      </h3>
      <p className="text-xs text-text-muted mb-4">
        Automated pipeline diagnostics executed during the scan.
      </p>

      <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
        {items.map((item) => {
          const IconComp = item.icon
          const isPass = item.status === 'pass'
          const isWarn = item.status === 'warn'

          return (
            <div
              key={item.id}
              onMouseEnter={() => setTooltip(item.id)}
              onMouseLeave={() => setTooltip(null)}
              onClick={() => setTooltip(tooltip === item.id ? null : item.id)}
              className="relative p-3 rounded-lg bg-surface-subtle border border-border/80 transition-colors cursor-pointer group"
            >
              <div className="flex items-start justify-between gap-2">
                <div className="flex items-center gap-2">
                  <div className="w-5 h-5 rounded bg-white border border-border flex items-center justify-center text-text-muted">
                    <IconComp className="w-3 h-3" strokeWidth={1.75} />
                  </div>
                  <span className="text-xs font-medium text-text truncate">
                    {item.label}
                  </span>
                </div>

                <div className="shrink-0">
                  {isWarn ? (
                    <AlertTriangle className="w-3.5 h-3.5 text-verdict-uncertain" strokeWidth={1.75} />
                  ) : isPass ? (
                    <CheckCircle2 className="w-3.5 h-3.5 text-verdict-real" strokeWidth={1.75} />
                  ) : (
                    <HelpCircle className="w-3.5 h-3.5 text-text-muted" strokeWidth={1.75} />
                  )}
                </div>
              </div>

              <div className="mt-1.5 text-xs font-mono text-text-muted truncate">
                {item.value}
              </div>

              {tooltip === item.id && (
                <div className="absolute z-20 bottom-full left-2 right-2 mb-2 p-2 bg-zinc-900 text-zinc-100 text-[11px] rounded-md shadow-modal animate-fade-in font-sans">
                  {item.tooltip}
                </div>
              )}
            </div>
          )
        })}
      </div>
    </div>
  )
}
