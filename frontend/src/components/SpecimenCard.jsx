import React from 'react'
import { Check, ShieldAlert, Sparkles, Activity } from 'lucide-react'

export default function SpecimenCard() {
  const checks = [
    { label: 'Skin texture', detail: 'consistent', status: 'PASS', limeText: 'NORMAL' },
    { label: 'Eye reflections', detail: 'mismatch', status: 'ALERT', limeText: 'ARTIFACT' },
    { label: 'Edge blending', detail: 'artifacts found', status: 'WARN', limeText: 'SUSPICIOUS' },
  ]

  return (
    <div className="relative w-full max-w-md mx-auto lg:max-w-none group">
      
      {/* Circular Rotating Lime Badge in top-right corner */}
      <div className="absolute -top-6 -right-6 z-20 w-24 h-24 sm:w-28 sm:h-28 pointer-events-none select-none">
        <div className="relative w-full h-full">
          {/* Rotating outer ring with text */}
          <div className="absolute inset-0 animate-spin-badge">
            <svg viewBox="0 0 100 100" className="w-full h-full">
              <path
                id="circlePath"
                d="M 50, 50 m -37, 0 a 37,37 0 1,1 74,0 a 37,37 0 1,1 -74,0"
                fill="none"
              />
              <text className="font-mono text-[9px] font-bold fill-[#0B1A5C] uppercase tracking-[0.24em]">
                <textPath href="#circlePath" startOffset="0%">
                  VERIFIED BY MODEL · FORENSIC AI ·
                </textPath>
              </text>
            </svg>
          </div>

          {/* Central Lime Circle with Checkmark */}
          <div className="absolute inset-0 m-auto w-10 h-10 sm:w-12 sm:h-12 bg-[#C8FF00] border-2 border-[#0B1A5C] shadow-[2px_2px_0px_#0B1A5C] rounded-full flex items-center justify-center">
            <Check className="w-5 h-5 sm:w-6 sm:h-6 text-[#0B1A5C] stroke-[3.5]" />
          </div>
        </div>
      </div>

      {/* Main White Specimen Card with Hard Navy Offset Shadow and Subtle Hover Tilt */}
      <div className="relative bg-white text-[#0B1A5C] border-3 border-[#0B1A5C] shadow-hard-navy-lg sm:shadow-hard-navy-xl p-5 sm:p-7 rotate-1 sm:rotate-2 group-hover:rotate-0 transition-transform duration-300 ease-out">
        
        {/* Top Header Label */}
        <div className="flex items-center justify-between border-b-2 border-[#0B1A5C] pb-3 mb-5 font-mono text-[11px] font-bold tracking-widest uppercase">
          <span className="flex items-center gap-1.5 text-[#0F45E3]">
            <Activity className="w-3.5 h-3.5 stroke-[2.5]" />
            SPECIMEN · ILLUSTRATIVE
          </span>
          <span className="bg-[#0B1A5C] text-white px-2 py-0.5 text-[10px]">
            REF #884-FL
          </span>
        </div>

        {/* Vector Forensic Face Placeholder (NO stock photos) */}
        <div className="relative w-full h-44 sm:h-48 bg-[#FFF9F7] border-2 border-[#0B1A5C] overflow-hidden mb-5 flex items-center justify-center">
          {/* Grid lines */}
          <div className="absolute inset-0 bg-[linear-gradient(to_right,#0B1A5C10_1px,transparent_1px),linear-gradient(to_bottom,#0B1A5C10_1px,transparent_1px)] bg-[size:16px_16px]" />

          {/* Forensic Facial Vector Silhouette & Crosshairs */}
          <svg
            viewBox="0 0 200 200"
            className="w-36 h-36 relative z-10 text-[#0B1A5C]"
            fill="none"
            stroke="currentColor"
            strokeWidth="1.5"
            strokeLinecap="round"
            strokeLinejoin="round"
          >
            {/* Outer face contour */}
            <path d="M 60 45 C 60 20, 140 20, 140 45 C 140 85, 135 150, 100 175 C 65 150, 60 85, 60 45 Z" fill="#0B1A5C08" strokeWidth="2" />
            
            {/* Eyes */}
            <path d="M 72 82 Q 85 75 95 82 Q 85 88 72 82 Z" fill="#C8FF0040" stroke="#0B1A5C" strokeWidth="1.8" />
            <circle cx="84" cy="81.5" r="3" fill="#0B1A5C" />
            <path d="M 105 82 Q 115 75 128 82 Q 115 88 105 82 Z" fill="#FF4D4D40" stroke="#FF4D4D" strokeWidth="2" />
            <circle cx="116" cy="81.5" r="3" fill="#FF4D4D" />

            {/* Nose & Mouth */}
            <path d="M 100 84 L 97 112 L 105 112" stroke="#0B1A5C" strokeWidth="1.8" />
            <path d="M 85 136 Q 100 144 115 136" stroke="#0B1A5C" strokeWidth="2" />

            {/* Forensic Mesh Grid and Bounding Artifact Box */}
            <rect x="100" y="70" width="35" height="24" stroke="#FF4D4D" strokeWidth="1.5" strokeDasharray="3 3" fill="#FF4D4D15" />
            <circle cx="100" cy="70" r="2.5" fill="#FF4D4D" />
            <circle cx="135" cy="70" r="2.5" fill="#FF4D4D" />
            <circle cx="100" cy="94" r="2.5" fill="#FF4D4D" />
            <circle cx="135" cy="94" r="2.5" fill="#FF4D4D" />

            {/* Crosshairs & Target Indicators */}
            <line x1="100" y1="20" x2="100" y2="35" stroke="#0F45E3" strokeWidth="1.5" />
            <line x1="100" y1="165" x2="100" y2="180" stroke="#0F45E3" strokeWidth="1.5" />
            <line x1="30" y1="100" x2="45" y2="100" stroke="#0F45E3" strokeWidth="1.5" />
            <line x1="155" y1="100" x2="170" y2="100" stroke="#0F45E3" strokeWidth="1.5" />
          </svg>

          {/* Floating Forensic Tag in Placeholder */}
          <div className="absolute bottom-2 left-2 bg-[#0B1A5C] text-[#C8FF00] font-mono text-[9px] px-2 py-0.5 font-bold uppercase tracking-widest">
            SYNTHETIC PATTERN DETECTED
          </div>
          <div className="absolute top-2 right-2 bg-[#FF4D4D] text-white font-mono text-[9px] px-2 py-0.5 font-bold uppercase">
            FLAG · EYES
          </div>
        </div>

        {/* 3 Rows of Forensic Checks */}
        <div className="space-y-2.5 mb-6">
          {checks.map((c, i) => (
            <div
              key={i}
              className="flex items-center justify-between border-b border-[#0B1A5C]/15 pb-2 text-xs font-mono"
            >
              <div className="flex items-center gap-2">
                <span className="w-1.5 h-1.5 bg-[#0B1A5C] rounded-none" />
                <span className="font-bold text-[#0B1A5C]">{c.label}</span>
                <span className="text-[#0B1A5C]/70">· {c.detail}</span>
              </div>
              <span className="bg-[#C8FF00] text-[#0B1A5C] px-2 py-0.5 text-[10px] font-extrabold uppercase tracking-wide border border-[#0B1A5C]">
                {c.limeText}
              </span>
            </div>
          ))}
        </div>

        {/* Score & Verdict Box: Big Number "87" with "FAKE SCORE" label */}
        <div className="bg-[#FFF9F7] border-2 border-[#0B1A5C] p-4 flex items-center justify-between">
          <div>
            <div className="font-mono text-[11px] font-extrabold uppercase tracking-widest text-[#0B1A5C]">
              FAKE SCORE
            </div>
            <div className="text-[10px] font-mono text-[#0B1A5C]/70 mt-0.5">
              Confidence threshold: 99.2%
            </div>
          </div>

          <div className="flex items-baseline gap-1">
            <span className="font-heading font-extrabold text-5xl sm:text-6xl text-[#0B1A5C] leading-none tracking-tight">
              87
            </span>
            <span className="font-mono font-bold text-sm text-[#FF4D4D]">
              /100
            </span>
          </div>
        </div>

      </div>
    </div>
  )
}
