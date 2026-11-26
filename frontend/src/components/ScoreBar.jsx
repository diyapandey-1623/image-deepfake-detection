import React from 'react'

export default function ScoreBar({ label, score, description, icon }) {
  // score is 0.0 to 1.0 (probability of fake)
  const percentage = Math.round(score * 100)
  
  // Color scale: Green (low probability of fake) to Amber (medium) to Red (high probability of fake)
  let barColor = 'from-emerald-500 to-teal-400'
  let textColor = 'text-emerald-400'
  let statusText = 'Authentic'

  if (score > 0.60) {
    barColor = 'from-rose-500 to-red-600'
    textColor = 'text-rose-400'
    statusText = 'Synthetic Anomaly'
  } else if (score >= 0.40) {
    barColor = 'from-amber-400 to-amber-500'
    textColor = 'text-amber-400'
    statusText = 'Borderline'
  }

  return (
    <div className="bg-slate-900/60 border border-slate-800/80 rounded-xl p-4 transition-all hover:border-slate-700">
      <div className="flex items-center justify-between mb-1.5">
        <div className="flex items-center gap-2">
          {icon && <span className="text-slate-400">{icon}</span>}
          <span className="text-sm font-semibold text-slate-200">{label}</span>
        </div>
        <div className="flex items-baseline gap-1.5">
          <span className={`text-base font-bold font-mono ${textColor}`}>{percentage}%</span>
          <span className="text-[11px] font-medium text-slate-400 uppercase tracking-wider">({statusText})</span>
        </div>
      </div>

      <div className="w-full bg-slate-800/90 rounded-full h-2.5 overflow-hidden p-0.5 border border-slate-700/50">
        <div
          className={`h-full rounded-full bg-gradient-to-r ${barColor} transition-all duration-700 ease-out`}
          style={{ width: `${Math.max(5, percentage)}%` }}
        />
      </div>

      {description && (
        <p className="mt-2 text-xs text-slate-400 leading-relaxed font-sans">{description}</p>
      )}
    </div>
  )
}
