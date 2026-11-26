import React, { useState, useEffect, useRef } from 'react'
import { Upload, AlertCircle } from 'lucide-react'

const MAX_SIZE_BYTES = 10 * 1024 * 1024 // 10 MB
const ALLOWED_TYPES = ['image/jpeg', 'image/jpg', 'image/png', 'image/webp']
const ALLOWED_EXTS = ['.jpg', '.jpeg', '.png', '.webp']

export default function Uploader({ onFileSelect, disabled = false }) {
  const [isDragOver, setIsDragOver] = useState(false)
  const [errorMessage, setErrorMessage] = useState(null)
  const fileInputRef = useRef(null)

  const validateAndProcessFile = (file) => {
    setErrorMessage(null)
    if (!file) return

    const name = file.name.toLowerCase()
    const hasValidExt = ALLOWED_EXTS.some((ext) => name.endsWith(ext))
    const hasValidMime = ALLOWED_TYPES.includes(file.type)

    if (!hasValidExt && !hasValidMime) {
      setErrorMessage('Unsupported format. Please upload a JPG, PNG, or WEBP image.')
      return
    }

    if (file.size > MAX_SIZE_BYTES) {
      const sizeMb = (file.size / (1024 * 1024)).toFixed(1)
      setErrorMessage(`File exceeds 10 MB limit (${sizeMb} MB).`)
      return
    }

    if (file.size === 0) {
      setErrorMessage('The uploaded file is empty (0 bytes).')
      return
    }

    onFileSelect(file)
  }

  const handleDragOver = (e) => {
    e.preventDefault()
    e.stopPropagation()
    if (!disabled) setIsDragOver(true)
  }

  const handleDragLeave = (e) => {
    e.preventDefault()
    e.stopPropagation()
    setIsDragOver(false)
  }

  const handleDrop = (e) => {
    e.preventDefault()
    e.stopPropagation()
    setIsDragOver(false)
    if (disabled) return

    const files = e.dataTransfer?.files
    if (files && files.length > 0) {
      validateAndProcessFile(files[0])
    }
  }

  useEffect(() => {
    const handlePaste = (e) => {
      if (disabled) return
      const items = e.clipboardData?.items
      if (!items) return

      for (let i = 0; i < items.length; i++) {
        if (items[i].type.indexOf('image') !== -1) {
          const blob = items[i].getAsFile()
          if (blob) {
            const ext = blob.type.split('/')[1] || 'png'
            const file = new File([blob], `clipboard_${Date.now()}.${ext}`, { type: blob.type })
            validateAndProcessFile(file)
            break
          }
        }
      }
    }

    window.addEventListener('paste', handlePaste)
    return () => window.removeEventListener('paste', handlePaste)
  }, [disabled])

  return (
    <div className="w-full">
      <input
        ref={fileInputRef}
        type="file"
        accept=".jpg,.jpeg,.png,.webp,image/jpeg,image/png,image/webp"
        className="hidden"
        disabled={disabled}
        onChange={(e) => {
          if (e.target.files && e.target.files[0]) {
            validateAndProcessFile(e.target.files[0])
          }
        }}
      />

      <div
        onClick={() => !disabled && fileInputRef.current?.click()}
        onDragOver={handleDragOver}
        onDragLeave={handleDragLeave}
        onDrop={handleDrop}
        role="button"
        tabIndex={disabled ? -1 : 0}
        onKeyDown={(e) => {
          if ((e.key === 'Enter' || e.key === ' ') && !disabled) {
            fileInputRef.current?.click()
          }
        }}
        className={`relative cursor-pointer border border-dashed rounded-xl p-8 sm:p-10 text-center transition-colors bg-card ${
          isDragOver
            ? 'border-text bg-surface-subtle'
            : 'border-border hover:border-zinc-400 hover:bg-surface-subtle/50'
        } ${disabled ? 'opacity-50 cursor-not-allowed' : ''} focus-ring`}
      >
        <div className="flex flex-col items-center justify-center max-w-sm mx-auto">
          <div className="w-10 h-10 rounded-lg bg-surface-subtle border border-border text-text flex items-center justify-center mb-3">
            <Upload className="w-5 h-5" strokeWidth={1.75} />
          </div>

          <h3 className="font-medium text-sm text-text mb-1">
            Drop image to inspect, or browse
          </h3>
          <p className="text-xs text-text-muted mb-3 leading-relaxed">
            Upload a facial photo for multi-domain forensic verification.
          </p>

          <div className="text-[11px] font-mono text-text-muted">
            JPG · PNG · WEBP (max 10 MB) · <span className="font-sans text-zinc-500">Paste with Ctrl+V</span>
          </div>
        </div>
      </div>

      {errorMessage && (
        <div className="mt-3 p-3 rounded-lg bg-verdict-fakeBg border border-verdict-fakeBorder text-verdict-fake text-xs flex items-center gap-2 animate-fade-in">
          <AlertCircle className="w-3.5 h-3.5 shrink-0" />
          <span>{errorMessage}</span>
        </div>
      )}
    </div>
  )
}
