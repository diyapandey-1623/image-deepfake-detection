import React, { useState, useEffect } from 'react'
import { Check, Loader2, X } from 'lucide-react'

const STEPS = [
  { id: 1, label: 'Face detection & boundary margin extraction' },
  { id: 2, label: 'Computing 2D Fourier power spectrum & ELA' },
  { id: 3, label: 'Bidirectional cross-attention feature fusion' },
  { id: 4, label: 'Generating Grad-CAM forensic visualization' },
]

export default function ProgressSteps({ previewUrl, onCancel }) {
  const [currentStep, setCurrentStep] = useState(1)

  useEffect(() => {
    const t1 = setTimeout(() => setCurrentStep(2), 650)
    const t2 = setTimeout(() => setCurrentStep(3), 1300)
    const t3 = setTimeout(() => setCurrentStep(4), 1950)

    return () => {
      clearTimeout(t1)
      clearTimeout(t2)
      clearTimeout(t3)
    }
  }, [])

  return (
    <div className="bg-card border border-border rounded-xl p-6 sm:p-7 max-w-md mx-auto shadow-subtle animate-fade-in text-center">
      {/* Thumbnail Preview */}
      {previewUrl && (
        <div className="w-16 h-16 mx-auto rounded-lg overflow-hidden border border-border bg-surface-subtle mb-4">
          <img src={previewUrl} alt="Target" className="w-full h-full object-cover" />
        </div>
      )}

      <h3 className="font-semibold text-sm text-text mb-1">
        Executing forensic pipeline
      </h3>
      <p className="text-xs text-text-muted mb-5">
        Multi-domain spectral and spatial analysis in progress.
      </p>

      {/* Checklist */}
      <div className="space-y-2.5 text-left max-w-xs mx-auto mb-5">
        {STEPS.map((step) => {
          const isDone = currentStep > step.id
          const isCurrent = currentStep === step.id

          return (
            <div key={step.id} className="flex items-center gap-2.5 text-xs">
              <div
                className={`w-5 h-5 rounded-md flex items-center justify-center shrink-0 text-[10px] font-mono transition-colors ${
                  isDone
                    ? 'bg-verdict-real text-white'
                    : isCurrent
                    ? 'bg-text text-white'
                    : 'bg-surface-subtle text-text-muted border border-border'
                }`}
              >
                {isDone ? (
                  <Check className="w-3 h-3" strokeWidth={2.5} />
                ) : isCurrent ? (
                  <Loader2 className="w-3 h-3 animate-spin" />
                ) : (
                  step.id
                )}
              </div>
              <span
                className={`transition-colors ${
                  isDone
                    ? 'text-text'
                    : isCurrent
                    ? 'text-text font-medium'
                    : 'text-text-muted/60'
                }`}
              >
                {step.label}
              </span>
            </div>
          )
        })}
      </div>

      {/* Cancel Button */}
      {onCancel && (
        <button
          onClick={onCancel}
          className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-md text-xs text-text-muted hover:text-text hover:bg-surface-subtle transition-colors focus-ring"
        >
          <X className="w-3.5 h-3.5" />
          Cancel analysis
        </button>
      )}
    </div>
  )
}
