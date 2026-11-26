import React, { useState } from 'react'
import { Layers, CheckSquare, FileCode, Cpu, HelpCircle, ChevronDown, ChevronUp } from 'lucide-react'

export default function HowItWorks() {
  const [openFaq, setOpenFaq] = useState(0)

  const faqs = [
    {
      q: 'Can deepfake detectors make mistakes?',
      a: 'Yes. Deepfake detection models are probabilistic statistical estimators, not absolute proof. High compression, blur, unusual lighting, or new generative AI tools may cause authentic images to look suspicious or synthetic images to appear authentic.',
    },
    {
      q: 'What is a Grad-CAM heatmap?',
      a: 'Grad-CAM (Gradient-weighted Class Activation Mapping) produces a visual heatmap showing which parts of the face the neural network looked at most closely. Warmer colors (red, orange, yellow) indicate features that heavily influenced the decision.',
    },
    {
      q: 'What does Error Level Analysis (ELA) check?',
      a: 'ELA re-saves the image at a known JPEG compression quality (Q=90) and calculates the pixel difference. Areas spliced, edited, or digitally altered after original capture usually show different compression errors than untouched areas.',
    },
    {
      q: 'Is my uploaded photo kept private?',
      a: 'Yes. Images uploaded through DeepCheck are processed strictly in temporary volatile memory and purged immediately after the report is generated. No images are retained or used to train models.',
    },
    {
      q: 'What do the decision thresholds mean?',
      a: 'Scores above 60% are classified as "Likely Fake", scores below 40% are "Likely Real", and scores between 40% and 60% are classified as "Uncertain", warranting human investigator review.',
    },
  ]

  return (
    <div className="max-w-4xl mx-auto px-4 sm:px-6 py-10 sm:py-14 space-y-10 animate-fade-in">
      {/* Intro Header */}
      <div className="max-w-2xl mx-auto text-center">
        <h1 className="font-semibold text-2xl sm:text-3xl text-text tracking-tight mb-2.5">
          Forensic detection methodology
        </h1>
        <p className="text-xs sm:text-sm text-text-muted leading-relaxed">
          Understanding the multi-domain inspection pipeline, spectral transformations, and forensic baselines.
        </p>
      </div>

      {/* 4 Core Pillars */}
      <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
        <div className="bg-card border border-border rounded-xl p-5 shadow-subtle">
          <div className="w-8 h-8 rounded-lg bg-surface-subtle border border-border flex items-center justify-center mb-3 text-text">
            <Layers className="w-4 h-4" strokeWidth={1.75} />
          </div>
          <h3 className="font-medium text-sm text-text mb-1">
            1. Facial alignment & MTCNN margin
          </h3>
          <p className="text-xs text-text-muted leading-relaxed">
            Faces are localized using MTCNN with a +20% margin to ensure blending seams along the jawline, forehead, and hairline are retained rather than clipped.
          </p>
        </div>

        <div className="bg-card border border-border rounded-xl p-5 shadow-subtle">
          <div className="w-8 h-8 rounded-lg bg-surface-subtle border border-border flex items-center justify-center mb-3 text-text">
            <Cpu className="w-4 h-4" strokeWidth={1.75} />
          </div>
          <h3 className="font-medium text-sm text-text mb-1">
            2. 2D Fourier power spectrum (FFT)
          </h3>
          <p className="text-xs text-text-muted leading-relaxed">
            Generators rely on transposed convolution upsamplers, inducing periodic grid spikes and power decay discrepancies across the 2D Fourier magnitude spectrum.
          </p>
        </div>

        <div className="bg-card border border-border rounded-xl p-5 shadow-subtle">
          <div className="w-8 h-8 rounded-lg bg-surface-subtle border border-border flex items-center justify-center mb-3 text-text">
            <CheckSquare className="w-4 h-4" strokeWidth={1.75} />
          </div>
          <h3 className="font-medium text-sm text-text mb-1">
            3. Error Level Analysis (ELA)
          </h3>
          <p className="text-xs text-text-muted leading-relaxed">
            When synthetic faces are inserted into authentic backgrounds, the spliced region exhibits a distinct compression delta when re-quantized at Q=90.
          </p>
        </div>

        <div className="bg-card border border-border rounded-xl p-5 shadow-subtle">
          <div className="w-8 h-8 rounded-lg bg-surface-subtle border border-border flex items-center justify-center mb-3 text-text">
            <FileCode className="w-4 h-4" strokeWidth={1.75} />
          </div>
          <h3 className="font-medium text-sm text-text mb-1">
            4. Bidirectional cross-attention
          </h3>
          <p className="text-xs text-text-muted leading-relaxed">
            Spatial representations query spectral fingerprints and vice versa, exposing inconsistencies where high-frequency spikes fail to match genuine facial edges.
          </p>
        </div>
      </div>

      {/* FAQ Accordion */}
      <div className="pt-6 border-t border-border">
        <h2 className="font-semibold text-lg sm:text-xl text-text mb-5 text-center">
          Frequently asked questions
        </h2>

        <div className="space-y-2.5 max-w-2xl mx-auto">
          {faqs.map((faq, idx) => {
            const isOpen = openFaq === idx
            return (
              <div
                key={idx}
                className="bg-card border border-border rounded-lg overflow-hidden shadow-subtle"
              >
                <button
                  onClick={() => setOpenFaq(isOpen ? null : idx)}
                  className="w-full p-4 text-left flex items-center justify-between gap-4 font-medium text-xs sm:text-sm text-text focus-ring"
                >
                  <span>{faq.q}</span>
                  {isOpen ? (
                    <ChevronUp className="w-3.5 h-3.5 text-text-muted shrink-0" />
                  ) : (
                    <ChevronDown className="w-3.5 h-3.5 text-text-muted shrink-0" />
                  )}
                </button>
                {isOpen && (
                  <div className="px-4 pb-4 pt-1 text-xs text-text-muted leading-relaxed border-t border-border/60">
                    {faq.a}
                  </div>
                )}
              </div>
            )
          })}
        </div>
      </div>
    </div>
  )
}
