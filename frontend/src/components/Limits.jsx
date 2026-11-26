import React from 'react'
import { AlertCircle, Scale, ShieldCheck, EyeOff } from 'lucide-react'

export default function Limits() {
  const cards = [
    {
      title: 'Probabilistic, Not Definitive',
      desc: 'Our deep learning classifier produces a calibrated probability based on pattern anomalies. It does not provide absolute cryptographic proof of origin.',
      icon: Scale,
    },
    {
      title: 'Benchmark Accuracy',
      desc: 'Evaluated at XX% test accuracy on held-out forensic face datasets. Real-world performance varies across extreme lighting and heavy compression.',
      icon: ShieldCheck,
      placeholder: true,
    },
    {
      title: 'No Accusations Without Review',
      desc: 'Predictions should serve as an initial triage indicator. Never use an automated verdict as the sole justification to accuse or penalize an individual.',
      icon: AlertCircle,
    },
  ]

  return (
    <section id="limits" className="bg-[#0F45E3] text-white py-20 sm:py-28 border-b-2 border-[#0B1A5C]">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        
        {/* Section Header */}
        <div className="max-w-3xl mb-14 sm:mb-16">
          <div className="inline-block px-3 py-1 bg-[#0B1A5C] text-[#C8FF00] font-mono text-xs font-bold uppercase tracking-widest-mono mb-3 shadow-[2px_2px_0px_#0B1A5C]">
            ETHICAL USE · OPERATIONAL BOUNDARIES
          </div>
          <h2 className="font-heading font-extrabold text-4xl sm:text-6xl tracking-tightest leading-tight text-white">
            No detector is perfect.
          </h2>
          <p className="font-serif italic text-xl sm:text-2xl text-[#FFF9F7]/90 mt-3 max-w-2xl leading-relaxed">
            "We believe in radical forensic transparency. Here is what you need to know before interpreting any Fake Score."
          </p>
        </div>

        {/* 3 Editorial Cards with Hard Shadows */}
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6 sm:gap-8">
          {cards.map((card, i) => {
            const Icon = card.icon
            return (
              <div
                key={i}
                className="bg-[#FFF9F7] text-[#0B1A5C] border-3 border-[#0B1A5C] p-6 sm:p-8 shadow-hard-navy-lg flex flex-col justify-between"
              >
                <div>
                  <div className="w-12 h-12 bg-[#0F45E3] border-2 border-[#0B1A5C] flex items-center justify-center mb-5 text-[#C8FF00] shadow-[2px_2px_0px_#0B1A5C]">
                    <Icon className="w-6 h-6 stroke-[2.2]" />
                  </div>

                  <h3 className="font-heading font-extrabold text-2xl text-[#0B1A5C] tracking-tight mb-3">
                    {card.title}
                  </h3>

                  <p className="font-mono text-xs sm:text-sm text-[#0B1A5C]/80 leading-relaxed">
                    {card.desc}
                  </p>
                </div>

                {card.placeholder && (
                  <div className="mt-6 pt-4 border-t border-[#0B1A5C]/20 font-mono text-[10px] text-[#0F45E3] font-bold uppercase tracking-wider">
                    [PLACEHOLDER: FILL WITH FINAL BENCHMARK METRIC]
                  </div>
                )}
              </div>
            )
          })}
        </div>

      </div>
    </section>
  )
}
