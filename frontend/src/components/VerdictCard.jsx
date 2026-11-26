import React from 'react'
import { CheckCircle2, AlertTriangle, AlertOctagon, HelpCircle } from 'lucide-react'
import { getVerdictData } from '../lib/verdict'
import ConfidenceMeter from './ConfidenceMeter'

export default function VerdictCard({ result }) {
  if (!result) return null

  const { fake_probability, verdict } = result
  const v = getVerdictData(fake_probability, verdict)

  let VerdictIcon = HelpCircle
  if (v.type === 'fake') VerdictIcon = AlertOctagon
  if (v.type === 'real') VerdictIcon = CheckCircle2
  if (v.type === 'uncertain') VerdictIcon = AlertTriangle

  return (
    <div className="bg-card border border-border rounded-xl p-5 sm:p-6 shadow-subtle animate-fade-in">
      {/* Top Banner Row */}
      <div className="flex flex-col sm:flex-row items-start sm:items-center justify-between gap-4 pb-5 border-b border-border">
        <div className="flex items-center gap-3">
          <div className={`w-10 h-10 rounded-lg flex items-center justify-center shrink-0 ${v.bgColor} ${v.textColor} border ${v.borderColor}`}>
            <VerdictIcon className="w-5 h-5" strokeWidth={1.75} />
          </div>

          <div>
            <div className="flex items-center gap-2">
              <span className={`text-lg sm:text-xl font-semibold tracking-tight ${v.textColor}`}>
                {v.label}
              </span>
              <span className={`text-[11px] font-mono px-2 py-0.5 rounded border ${v.confidenceColor}`}>
                {v.confidenceTier} confidence
              </span>
            </div>
            <p className="text-xs text-text-muted mt-0.5 font-mono">
              Report ID: {result.report_id?.slice(0, 8) || 'Scan'} · {result.model_name || 'Fusion Cross-Attention CNN'}
            </p>
          </div>
        </div>

        {/* Big Percentage Callout */}
        <div className="sm:text-right">
          <div className="text-2xl sm:text-3xl font-semibold text-text tracking-tight tabular-nums">
            {v.percent}%
          </div>
          <div className="text-[11px] font-mono text-text-muted uppercase tracking-wider">
            Synthetic probability
          </div>
        </div>
      </div>

      {/* Meter Bar Section */}
      <div className="pt-5 space-y-2.5">
        <ConfidenceMeter percent={v.percent} verdictType={v.type} barColor={v.barColor} />

        <p className="text-xs text-text-muted leading-relaxed pt-1">
          {v.explanation}
        </p>
      </div>
    </div>
  )
}
