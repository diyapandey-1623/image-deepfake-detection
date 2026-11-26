import React from 'react'
import { User, Users } from 'lucide-react'
import { getVerdictData } from '../lib/verdict'

export default function FacesGrid({ faces = [], faceDetected = true }) {
  if (!faceDetected || faces.length === 0) {
    return (
      <div className="bg-card border border-border rounded-xl p-5 shadow-subtle">
        <div className="flex items-center gap-2 mb-1.5">
          <User className="w-4 h-4 text-text-muted" strokeWidth={1.75} />
          <h3 className="font-semibold text-sm text-text">
            Faces localized
          </h3>
        </div>
        <p className="text-xs text-text-muted">
          No distinct human face detected. Full image analyzed.
        </p>
      </div>
    )
  }

  return (
    <div className="bg-card border border-border rounded-xl p-5 sm:p-6 shadow-subtle">
      <div className="flex items-center justify-between pb-3 border-b border-border mb-3.5">
        <div className="flex items-center gap-2">
          <Users className="w-4 h-4 text-text-muted" strokeWidth={1.75} />
          <div>
            <h3 className="font-semibold text-sm sm:text-base text-text">
              Faces localized ({faces.length})
            </h3>
            <p className="text-xs text-text-muted">
              MTCNN cropped regions with +20% contextual boundary margin.
            </p>
          </div>
        </div>
      </div>

      <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 gap-3">
        {faces.map((f, idx) => {
          const v = getVerdictData(f.fake_probability)
          const bboxStr = f.bbox ? `[${f.bbox.join(', ')}]` : 'N/A'

          return (
            <div
              key={idx}
              className="p-3 rounded-lg bg-surface-subtle border border-border flex items-center justify-between gap-3"
            >
              <div>
                <div className="text-xs font-medium text-text">Face #{idx + 1}</div>
                <div className="text-[11px] font-mono text-text-muted mt-0.5">
                  BBox: {bboxStr}
                </div>
              </div>

              <div className="text-right">
                <span
                  className={`inline-flex items-center px-2 py-0.5 rounded text-[11px] font-mono font-medium border ${v.bgColor} ${v.textColor} ${v.borderColor}`}
                >
                  {v.percent}% {v.label}
                </span>
              </div>
            </div>
          )
        })}
      </div>
    </div>
  )
}
