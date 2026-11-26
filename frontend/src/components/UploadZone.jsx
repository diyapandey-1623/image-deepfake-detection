import React, { useRef, useState } from 'react'

export default function UploadZone({ onFileSelected, isAnalyzing, currentStage }) {
  const [isDragOver, setIsDragOver] = useState(false)
  const fileInputRef = useRef(null)

  const handleDragOver = (e) => {
    e.preventDefault()
    setIsDragOver(true)
  }

  const handleDragLeave = () => {
    setIsDragOver(false)
  }

  const handleDrop = (e) => {
    e.preventDefault()
    setIsDragOver(false)
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      validateAndSelect(e.dataTransfer.files[0])
    }
  }

  const handleInputChange = (e) => {
    if (e.target.files && e.target.files[0]) {
      validateAndSelect(e.target.files[0])
    }
  }

  const validateAndSelect = (file) => {
    if (!file.type.startsWith('image/')) {
      alert('Please upload a valid image file (JPEG, PNG, WEBP).')
      return
    }
    onFileSelected(file)
  }

  return (
    <div
      onDragOver={handleDragOver}
      onDragLeave={handleDragLeave}
      onDrop={handleDrop}
      onClick={() => !isAnalyzing && fileInputRef.current?.click()}
      className={`relative group cursor-pointer border-2 border-dashed rounded-2xl p-8 sm:p-12 text-center transition-all duration-300 ${
        isDragOver
          ? 'border-cyan-400 bg-cyan-950/20 scale-[1.01]'
          : 'border-slate-700/80 hover:border-cyan-500/60 bg-slate-900/40 hover:bg-slate-900/60'
      } ${isAnalyzing ? 'pointer-events-none opacity-80' : ''}`}
    >
      <input
        ref={fileInputRef}
        type="file"
        accept="image/jpeg,image/png,image/webp,image/bmp"
        className="hidden"
        onChange={handleInputChange}
      />

      <div className="flex flex-col items-center justify-center">
        {/* Upload Icon */}
        <div className="w-16 h-16 rounded-2xl bg-gradient-to-tr from-cyan-500/20 to-blue-500/20 border border-cyan-500/30 flex items-center justify-center mb-4 group-hover:scale-110 transition-transform">
          <svg className="w-8 h-8 text-cyan-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l1.586-1.586a2 2 0 012.828 0L20 14m-6-6h.01M6 20h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2z" />
          </svg>
        </div>

        <h3 className="text-lg font-bold text-slate-100 mb-1">
          {isAnalyzing ? currentStage : 'Drop facial photo here or click to browse'}
        </h3>
        <p className="text-sm text-slate-400 max-w-sm mb-4">
          Accepts JPEG, PNG, or WEBP up to 25MB. Single portrait with clear face yields optimal forensic accuracy.
        </p>

        {isAnalyzing && (
          <div className="w-full max-w-xs mt-3">
            <div className="w-full bg-slate-800 rounded-full h-2 overflow-hidden border border-slate-700">
              <div className="h-full bg-gradient-to-r from-cyan-500 via-blue-500 to-purple-500 animate-pulse w-full rounded-full" />
            </div>
          </div>
        )}
      </div>
    </div>
  )
}
