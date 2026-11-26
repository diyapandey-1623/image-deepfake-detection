import React from 'react'
import { CheckCircle2, AlertTriangle, Info, BrainCircuit, Activity } from 'lucide-react'
import { deriveReasons, isFakeVerdict } from '../lib/verdict'

export default function ReasonsList({ result }) {
  if (!result) return null

  const reasons = deriveReasons(result)
  const rationale = result.forensic_rationale
  const regionalInfluence = result.regional_influence || []
  const isFake = isFakeVerdict(result.fake_probability, result.verdict)

  return (
    <div className="space-y-4">
      {/* 1. Primary Model Decision Rationale */}
      <div className="bg-card border border-border rounded-xl p-5 sm:p-6 shadow-subtle">
        <div className="flex items-center gap-2 mb-3">
          <BrainCircuit className="w-4 h-4 text-primary" strokeWidth={2} />
          <h3 className="font-semibold text-sm sm:text-base text-text">
            Forensic Decision Rationale
          </h3>
          <span className="ml-auto text-[11px] font-mono font-medium px-2 py-0.5 rounded bg-surface-subtle border border-border text-text-muted">
            {isFake ? 'MANIPULATION SIGNATURE' : 'BIOMETRIC INTEGRITY'}
          </span>
        </div>

        {rationale ? (
          <div className={`p-4 rounded-lg border text-xs sm:text-sm leading-relaxed ${
            isFake 
              ? 'bg-red-500/5 border-red-500/20 text-slate-800' 
              : 'bg-emerald-500/5 border-emerald-500/20 text-slate-800'
          }`}>
            <p className="font-normal">
              {rationale}
            </p>
          </div>
        ) : (
          <p className="text-xs text-text-muted leading-relaxed">
            Neural inspection evaluated spatial high-frequency characteristics, lighting consistency, and facial organ boundaries.
          </p>
        )}
      </div>

      {/* 2. Regional Facial Influence Analysis (High: Red, Moderate: Yellow, Low: Blue) */}
      {regionalInfluence.length > 0 && (
        <div className="bg-card border border-border rounded-xl p-5 sm:p-6 shadow-subtle">
          <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2 mb-4 pb-3 border-b border-border">
            <div className="flex items-center gap-2">
              <Activity className="w-4 h-4 text-text" strokeWidth={2} />
              <h3 className="font-semibold text-sm sm:text-base text-text">
                Regional Facial Influence Analysis
              </h3>
            </div>
            {/* Color Legend */}
            <div className="flex items-center gap-3 text-[11px] font-mono">
              <span className="flex items-center gap-1.5">
                <span className="w-2.5 h-2.5 rounded-full bg-red-500 inline-block" />
                <span className="text-text-muted font-bold">Red: High</span>
              </span>
              <span className="flex items-center gap-1.5">
                <span className="w-2.5 h-2.5 rounded-full bg-yellow-400 inline-block" />
                <span className="text-text-muted font-bold">Yellow: Moderate</span>
              </span>
              <span className="flex items-center gap-1.5">
                <span className="w-2.5 h-2.5 rounded-full bg-blue-500 inline-block" />
                <span className="text-text-muted font-bold">Blue: Low</span>
              </span>
            </div>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
            {regionalInfluence.map((zone, idx) => {
              const lvl = (zone.influence_level || 'Low').toLowerCase()
              const isHigh = lvl.includes('high') || zone.color?.toLowerCase().includes('red')
              const isMod = lvl.includes('mod') || zone.color?.toLowerCase().includes('yellow')

              // Badge styling tokens
              let badgeBg = 'bg-blue-500/10 text-blue-600 border-blue-500/30'
              let barColor = 'bg-blue-500'
              let dotColor = 'bg-blue-500'
              let levelLabel = 'Low Influence'

              if (isHigh) {
                badgeBg = 'bg-red-500/10 text-red-600 border-red-500/30'
                barColor = 'bg-red-500'
                dotColor = 'bg-red-500'
                levelLabel = 'High Influence'
              } else if (isMod) {
                badgeBg = 'bg-yellow-500/15 text-yellow-700 border-yellow-500/40'
                barColor = 'bg-yellow-400'
                dotColor = 'bg-yellow-400'
                levelLabel = 'Moderate Influence'
              }

              return (
                <div
                  key={idx}
                  className="p-3.5 rounded-lg bg-surface-subtle border border-border/80 flex flex-col justify-between"
                >
                  <div className="flex items-center justify-between gap-2 mb-1.5">
                    <span className="text-xs font-semibold text-text">
                      {zone.region}
                    </span>
                    <span className={`inline-flex items-center gap-1 px-2 py-0.5 text-[10px] font-mono font-bold uppercase rounded border ${badgeBg}`}>
                      <span className={`w-1.5 h-1.5 rounded-full ${dotColor}`} />
                      {levelLabel}
                    </span>
                  </div>

                  {/* Score bar */}
                  <div className="my-2">
                    <div className="flex justify-between text-[11px] font-mono text-text-muted mb-1">
                      <span>Activation Focus</span>
                      <span className="font-semibold text-text">{zone.score}%</span>
                    </div>
                    <div className="w-full h-1.5 bg-slate-200 rounded-full overflow-hidden">
                      <div
                        className={`h-full rounded-full transition-all duration-300 ${barColor}`}
                        style={{ width: `${Math.min(100, Math.max(5, zone.score))}%` }}
                      />
                    </div>
                  </div>

                  {zone.description && (
                    <p className="text-[11px] text-text-muted mt-1 leading-relaxed">
                      {zone.description}
                    </p>
                  )}
                </div>
              )
            })}
          </div>
        </div>
      )}

      {/* 3. Multi-Domain Verification Factors */}
      <div className="bg-card border border-border rounded-xl p-5 sm:p-6 shadow-subtle">
        <div className="flex items-center gap-2 mb-3.5">
          <Info className="w-4 h-4 text-text-muted" strokeWidth={1.75} />
          <h3 className="font-semibold text-sm sm:text-base text-text">
            Forensic Verification Factors
          </h3>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
          {reasons.map((item) => (
            <div
              key={item.id}
              className="flex items-start gap-2.5 p-3 rounded-lg bg-surface-subtle border border-border/80"
            >
              <div className="shrink-0 mt-0.5">
                {item.isWarning ? (
                  <AlertTriangle className="w-4 h-4 text-verdict-uncertain" strokeWidth={1.75} />
                ) : (
                  <CheckCircle2 className="w-4 h-4 text-verdict-real" strokeWidth={1.75} />
                )}
              </div>
              <div>
                <h4 className="text-xs font-semibold text-text mb-0.5">
                  {item.title}
                </h4>
                <p className="text-xs text-text-muted leading-relaxed">
                  {item.text}
                </p>
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  )
}
