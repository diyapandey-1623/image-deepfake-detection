import React from 'react'
import { Link } from 'react-router-dom'
import { ArrowRight, ChevronDown } from 'lucide-react'
import SpecimenCard from './SpecimenCard'

export default function Hero() {
  return (
    <section className="relative bg-[#0F45E3] text-white pt-10 pb-16 sm:pt-16 sm:pb-24 border-b-2 border-[#0B1A5C] overflow-hidden">
      
      {/* Background Subtle Grid / Forensic Crosses */}
      <div className="absolute inset-0 opacity-10 pointer-events-none bg-[radial-gradient(#FFFFFF_1px,transparent_1px)] [background-size:24px_24px]" />

      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 relative z-10">
        <div className="grid grid-cols-1 lg:grid-cols-12 gap-12 lg:gap-8 items-center">
          
          {/* Left Column: Bold Typography & CTAs */}
          <div className="lg:col-span-7 space-y-6 sm:space-y-8">
            
            {/* Small Mono Label */}
            <div className="inline-flex items-center gap-2 px-3 py-1 bg-[#0B1A5C] border border-[#C8FF00]/40 text-[#C8FF00] font-mono text-[11px] sm:text-xs font-bold uppercase tracking-widest-mono shadow-[2px_2px_0px_#0B1A5C]">
              <span className="w-2 h-2 bg-[#C8FF00] rounded-none animate-pulse" />
              DEEPFAKE DETECTION · AI × FACE FORENSICS
            </div>

            {/* Huge Headline */}
            <h1 className="font-heading font-extrabold text-5xl sm:text-7xl lg:text-8xl tracking-tightest leading-[0.95] text-white">
              <span className="block">
                Faces can{' '}
                <span className="relative inline-block text-white">
                  lie.
                  {/* Lime Strikethrough Bar */}
                  <span className="absolute left-0 top-1/2 -translate-y-1/2 w-full h-3 sm:h-4.5 bg-[#C8FF00] -rotate-3" />
                </span>
              </span>
              <span className="block mt-2 sm:mt-3">
                Pixels{' '}
                <span className="inline-block bg-[#C8FF00] text-[#0B1A5C] px-3 sm:px-5 py-0.5 sm:py-1 -rotate-1 border-2 border-[#0B1A5C] shadow-[4px_4px_0px_#0B1A5C]">
                  prove.
                </span>
              </span>
            </h1>

            {/* Italic Serif Subtext */}
            <p className="font-serif italic text-2xl sm:text-3xl text-[#FFF9F7]/95 max-w-xl leading-snug">
              "We don't trust the face. We read the pixels. Every image gets a Fake Score."
            </p>

            {/* CTAs */}
            <div className="pt-2 flex flex-wrap items-center gap-5 sm:gap-6">
              
              {/* Lime Button: Check an image -> */}
              <a
                href="#demo-upload"
                className="inline-flex items-center gap-3 px-7 py-4 bg-[#C8FF00] text-[#0B1A5C] font-mono text-sm sm:text-base font-extrabold uppercase tracking-wider border-2 border-[#0B1A5C] shadow-hard-navy btn-press focus:outline-none focus-visible:ring-2 focus-visible:ring-white"
              >
                <span>Check an image</span>
                <ArrowRight className="w-5 h-5 stroke-[3]" />
              </a>

              {/* Ghost Link: HOW IT WORKS v */}
              <a
                href="#how-it-works"
                className="inline-flex items-center gap-2 text-[#FFF9F7] hover:text-[#C8FF00] font-mono text-xs sm:text-sm font-bold uppercase tracking-widest underline underline-offset-8 transition-colors focus:outline-none focus-visible:ring-1 focus-visible:ring-[#C8FF00]"
              >
                <span>HOW IT WORKS</span>
                <ChevronDown className="w-4 h-4 stroke-[2.5]" />
              </a>

            </div>

            {/* Metadata Badges */}
            <div className="pt-4 flex flex-wrap items-center gap-6 text-[11px] font-mono text-white/75 border-t border-white/15">
              <div>
                <span className="text-[#C8FF00] font-bold">01/</span> ZERO LOGGING
              </div>
              <div>
                <span className="text-[#C8FF00] font-bold">02/</span> GRAD-CAM HEATMAPS
              </div>
              <div>
                <span className="text-[#C8FF00] font-bold">03/</span> FREQUENCY FFT CHECK
              </div>
            </div>

          </div>

          {/* Right Column: Specimen Card */}
          <div className="lg:col-span-5 flex justify-center lg:justify-end">
            <SpecimenCard />
          </div>

        </div>
      </div>

    </section>
  )
}
