import React from 'react'

export default function ConfidenceMeter({ percent, verdictType, barColor }) {
  const clamped = Math.max(0, Math.min(100, Number(percent) || 0))

  return (
    <div className="w-full space-y-2">
      {/* Precision Calibrated Scale Track */}
      <div className="relative h-2.5 w-full bg-surface-subtle rounded border border-border overflow-hidden">
        {/* Fill bar */}
        <div
          className={`h-full transition-all duration-500 ease-out ${barColor}`}
          style={{ width: `${clamped}%` }}
        />

        {/* Threshold divider ticks at 40% and 60% */}
        <div
          className="absolute top-0 bottom-0 w-[1px] bg-zinc-400/80 z-10"
          style={{ left: '40%' }}
          title="Authentic threshold (40%)"
        />
        <div
          className="absolute top-0 bottom-0 w-[1px] bg-zinc-400/80 z-10"
          style={{ left: '60%' }}
          title="Synthetic threshold (60%)"
        />
      </div>

      {/* Axis markers with tabular precision */}
      <div className="flex justify-between items-center text-[10px] font-mono text-text-muted px-0.5">
        <span>0% Authentic</span>
        <span className="text-zinc-500">40% threshold</span>
        <span className="text-zinc-500">60% threshold</span>
        <span>100% Synthetic</span>
      </div>
    </div>
  )
}
