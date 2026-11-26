import React from 'react'
import { ShieldAlert } from 'lucide-react'

export default function Disclaimer() {
  return (
    <div className="bg-surface-subtle border border-border rounded-xl p-4 text-xs text-text-muted flex items-start gap-3">
      <ShieldAlert className="w-4 h-4 text-zinc-500 shrink-0 mt-0.5" strokeWidth={1.75} />
      <div className="space-y-0.5">
        <h4 className="font-semibold text-text text-xs">
          Forensic research advisory
        </h4>
        <p className="leading-relaxed text-[11px]">
          Results are statistical estimates and should assist, not replace, human forensic verification. Extreme compression, blur, or novel generative models may introduce false positives or negatives.
        </p>
      </div>
    </div>
  )
}
