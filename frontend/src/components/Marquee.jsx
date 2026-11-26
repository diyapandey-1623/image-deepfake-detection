import React from 'react'

export default function Marquee() {
  const items = [
    { text: 'REAL', arrow: '→', highlight: false },
    { text: 'FAKE', arrow: '→', highlight: true },
    { text: 'CHECKED', arrow: '→', highlight: false },
    { text: 'REAL', arrow: '→', highlight: false },
    { text: 'FAKE', arrow: '→', highlight: true },
    { text: 'CHECKED', arrow: '→', highlight: false },
    { text: 'REAL', arrow: '→', highlight: false },
    { text: 'FAKE', arrow: '→', highlight: true },
    { text: 'CHECKED', arrow: '→', highlight: false },
    { text: 'REAL', arrow: '→', highlight: false },
    { text: 'FAKE', arrow: '→', highlight: true },
    { text: 'CHECKED', arrow: '→', highlight: false },
  ]

  return (
    <div className="relative w-full bg-[#0B1A5C] text-white py-3.5 sm:py-4 border-b-2 border-[#0B1A5C] overflow-hidden select-none">
      <div className="animate-marquee flex items-center whitespace-nowrap">
        {/* Sequence 1 */}
        <div className="flex items-center gap-6 sm:gap-8 px-4">
          {items.map((item, idx) => (
            <div key={`m1-${idx}`} className="flex items-center gap-6 sm:gap-8 font-mono text-base sm:text-xl font-extrabold tracking-widest uppercase">
              <span className={item.highlight ? 'text-[#C8FF00]' : 'text-white'}>
                {item.text}
              </span>
              <span className="text-[#C8FF00] font-mono text-lg sm:text-2xl">
                {item.arrow}
              </span>
            </div>
          ))}
        </div>

        {/* Duplicate sequence for seamless loop */}
        <div className="flex items-center gap-6 sm:gap-8 px-4" aria-hidden="true">
          {items.map((item, idx) => (
            <div key={`m2-${idx}`} className="flex items-center gap-6 sm:gap-8 font-mono text-base sm:text-xl font-extrabold tracking-widest uppercase">
              <span className={item.highlight ? 'text-[#C8FF00]' : 'text-white'}>
                {item.text}
              </span>
              <span className="text-[#C8FF00] font-mono text-lg sm:text-2xl">
                {item.arrow}
              </span>
            </div>
          ))}
        </div>
      </div>
    </div>
  )
}
