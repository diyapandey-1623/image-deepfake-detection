import React from 'react'
import { Link } from 'react-router-dom'
import { Check, Shield } from 'lucide-react'

export default function Footer() {
  return (
    <footer className="bg-[#0B1A5C] text-white border-t-3 border-[#0B1A5C] py-14 sm:py-16">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        
        <div className="flex flex-col md:flex-row items-start md:items-center justify-between gap-8 pb-12 border-b border-white/15">
          
          {/* Left: Brand */}
          <div className="space-y-3">
            <Link to="/" className="flex items-center gap-3 group focus:outline-none focus-visible:ring-2 focus-visible:ring-[#C8FF00]">
              <div className="w-9 h-9 rounded-full bg-[#C8FF00] border-2 border-white flex items-center justify-center">
                <Check className="w-5 h-5 text-[#0B1A5C] stroke-[3.5]" />
              </div>
              <span className="font-heading font-extrabold text-2xl text-white tracking-tight">
                FAKELENS
              </span>
            </Link>
            <p className="font-mono text-xs text-white/70 max-w-sm leading-relaxed">
              Open face forensics & deepfake detection powered by Vision Transformers and Fourier analysis.
            </p>
          </div>

          {/* Right: Nav Links */}
          <div className="flex flex-wrap items-center gap-6 sm:gap-8 font-mono text-xs font-bold uppercase tracking-widest">
            <Link to="/" className="text-white hover:text-[#C8FF00] transition-colors">
              01 OVERVIEW
            </Link>
            <Link to="/detect" className="text-white hover:text-[#C8FF00] transition-colors">
              02 DETECT
            </Link>
            <Link to="/history" className="text-white hover:text-[#C8FF00] transition-colors">
              03 HISTORY
            </Link>
            <a href="#limits" className="text-white hover:text-[#C8FF00] transition-colors">
              04 ABOUT
            </a>
          </div>

        </div>

        {/* Bottom Notice & Ethical Stance */}
        <div className="pt-8 flex flex-col sm:flex-row items-start sm:items-center justify-between gap-4 font-mono text-xs text-white/60">
          <div className="flex items-center gap-2 text-[#C8FF00] font-bold">
            <Shield className="w-4 h-4 stroke-[2.5]" />
            <span>Only analyze images you have the right to use.</span>
          </div>

          <div className="text-[11px] text-white/40">
            © {new Date().getFullYear()} FakeLens Systems. MIT License · Face Forensics.
          </div>
        </div>

      </div>
    </footer>
  )
}
