import React from 'react'
import { UploadCloud, ScanFace, Cpu, Eye } from 'lucide-react'

export default function ChecksList() {
  const steps = [
    {
      step: '01',
      title: 'Upload',
      desc: 'Drop a JPG, PNG or WEBP face photo.',
      tag: 'IMAGE',
      icon: UploadCloud,
    },
    {
      step: '02',
      title: 'Find the face',
      desc: 'Face is detected and cropped so the model sees only what matters.',
      tag: 'DETECT',
      icon: ScanFace,
    },
    {
      step: '03',
      title: 'Classify',
      desc: 'A trained model scores the face from 0 to 100.',
      tag: 'MODEL',
      icon: Cpu,
    },
    {
      step: '04',
      title: 'Explain',
      desc: 'A heatmap shows which regions looked fake.',
      tag: 'GRAD-CAM',
      icon: Eye,
    },
  ]

  return (
    <section id="how-it-works" className="bg-[#FFF9F7] text-[#0B1A5C] py-20 sm:py-28 border-b-2 border-[#0B1A5C]">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        
        {/* Section Header */}
        <div className="max-w-3xl mb-16 sm:mb-20">
          <div className="font-mono text-xs font-bold uppercase tracking-widest text-[#0F45E3] mb-3">
            METHODOLOGY · FORENSIC PIPELINE
          </div>
          <h2 className="font-heading font-extrabold text-4xl sm:text-6xl lg:text-7xl tracking-tightest leading-none text-[#0B1A5C]">
            Four checks.{' '}
            <span className="font-serif italic font-normal text-[#0F45E3] block sm:inline">
              Zero guesswork.
            </span>
          </h2>
          <p className="font-mono text-xs sm:text-sm text-[#0B1A5C]/75 mt-4 max-w-xl leading-relaxed">
            Every file undergoes a deterministic four-tier inspection to isolate synthetic generative noise, frequency distortions, and spatial seams.
          </p>
        </div>

        {/* Four Rows separated by thin lines */}
        <div className="border-t-2 border-[#0B1A5C]">
          {steps.map((item, idx) => {
            const IconComponent = item.icon
            return (
              <div
                key={item.step}
                className="group py-8 sm:py-10 border-b border-[#0B1A5C]/25 transition-colors hover:bg-white px-2 sm:px-6 -mx-2 sm:-mx-6"
              >
                <div className="grid grid-cols-1 md:grid-cols-12 gap-4 md:gap-8 items-center">
                  
                  {/* Step Number + Title */}
                  <div className="md:col-span-5 flex items-center gap-4 sm:gap-6">
                    <span className="font-mono text-sm sm:text-base font-extrabold text-[#0F45E3] w-8">
                      {item.step}
                    </span>
                    <div className="w-10 h-10 border-2 border-[#0B1A5C] bg-[#FFF9F7] group-hover:bg-[#C8FF00] flex items-center justify-center transition-colors">
                      <IconComponent className="w-5 h-5 text-[#0B1A5C] stroke-[2.2]" />
                    </div>
                    <h3 className="font-heading font-extrabold text-2xl sm:text-3xl text-[#0B1A5C] tracking-tight">
                      {item.title}
                    </h3>
                  </div>

                  {/* Description */}
                  <div className="md:col-span-5">
                    <p className="font-mono text-xs sm:text-sm text-[#0B1A5C]/80 leading-relaxed">
                      {item.desc}
                    </p>
                  </div>

                  {/* Mono Tag on the Right */}
                  <div className="md:col-span-2 flex md:justify-end">
                    <span className="inline-block px-3 py-1 font-mono text-xs font-bold uppercase tracking-widest text-[#0B1A5C] bg-white border border-[#0B1A5C] group-hover:bg-[#C8FF00] shadow-[2px_2px_0px_#0B1A5C] transition-all">
                      {item.tag}
                    </span>
                  </div>

                </div>
              </div>
            )
          })}
        </div>

      </div>
    </section>
  )
}
