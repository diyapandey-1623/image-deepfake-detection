import React, { useState } from 'react'
import { Link, useLocation } from 'react-router-dom'
import { Check, Search, Menu, X, ArrowUpRight } from 'lucide-react'

export default function Navbar() {
  const location = useLocation()
  const [mobileMenuOpen, setMobileMenuOpen] = useState(false)

  const isCurrent = (path) => {
    if (path === '/' && (location.pathname === '/' || location.pathname === '')) return true
    if (path !== '/' && location.pathname.startsWith(path)) return true
    return false
  }

  const navLinks = [
    { label: '01 OVERVIEW', to: '/' },
    { label: '02 DETECT', to: '/detect' },
    { label: '03 HISTORY', to: '/history' },
    { label: '04 ABOUT', to: '/#limits' },
  ]

  return (
    <header className="sticky top-0 z-50 bg-[#0F45E3] border-b-2 border-[#0B1A5C] text-white">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex items-center justify-between h-20">
          
          {/* Brand Logo with Lime Circle + Checkmark */}
          <Link to="/" className="flex items-center gap-3 group focus:outline-none focus-visible:ring-2 focus-visible:ring-[#C8FF00]">
            <div className="w-10 h-10 rounded-full bg-[#C8FF00] border-2 border-[#0B1A5C] shadow-[2px_2px_0px_#0B1A5C] flex items-center justify-center transition-transform group-hover:scale-105">
              <Check className="w-6 h-6 text-[#0B1A5C] stroke-[3.5]" />
            </div>
            <div className="flex flex-col">
              <span className="font-heading font-extrabold text-2xl tracking-tight text-white leading-none">
                FAKELENS
              </span>
              <span className="font-mono text-[10px] text-[#C8FF00] tracking-widest-mono uppercase mt-0.5">
                FORENSICS · AI
              </span>
            </div>
          </Link>

          {/* Desktop Navigation Links */}
          <nav className="hidden md:flex items-center gap-8 font-mono text-xs uppercase tracking-widest">
            {navLinks.map((link) => {
              const active = isCurrent(link.to)
              return (
                <Link
                  key={link.label}
                  to={link.to}
                  className={`relative py-2 font-semibold transition-colors hover:text-[#C8FF00] focus:outline-none focus-visible:ring-1 focus-visible:ring-[#C8FF00] ${
                    active ? 'text-white' : 'text-white/80'
                  }`}
                >
                  {link.label}
                  {active && (
                    <span className="absolute bottom-0 left-0 w-full h-[3px] bg-[#C8FF00] rounded-none" />
                  )}
                </Link>
              )
            })}
          </nav>

          {/* Right Action: Search-Style Button */}
          <div className="hidden md:flex items-center gap-4">
            <Link
              to="/detect"
              className="inline-flex items-center gap-2.5 px-4 py-2.5 bg-white text-[#0B1A5C] font-mono text-xs font-bold uppercase border-2 border-[#0B1A5C] shadow-[3px_3px_0px_#0B1A5C] hover:bg-[#C8FF00] transition-colors btn-press focus:outline-none focus-visible:ring-2 focus-visible:ring-[#C8FF00]"
            >
              <Search className="w-3.5 h-3.5 text-[#0B1A5C] stroke-[2.5]" />
              <span>SEARCH / SCAN IMAGE</span>
              <span className="bg-[#0B1A5C] text-white text-[9px] px-1.5 py-0.5 ml-1 font-mono">⌘K</span>
            </Link>
          </div>

          {/* Mobile Menu Hamburger */}
          <div className="flex md:hidden">
            <button
              onClick={() => setMobileMenuOpen(!mobileMenuOpen)}
              className="p-2 border-2 border-[#0B1A5C] bg-[#C8FF00] text-[#0B1A5C] shadow-[2px_2px_0px_#0B1A5C] focus:outline-none"
              aria-label="Toggle Navigation Menu"
            >
              {mobileMenuOpen ? <X className="w-5 h-5 stroke-[2.5]" /> : <Menu className="w-5 h-5 stroke-[2.5]" />}
            </button>
          </div>

        </div>
      </div>

      {/* Mobile Menu Dropdown */}
      {mobileMenuOpen && (
        <div className="md:hidden border-t-2 border-[#0B1A5C] bg-[#0F45E3] px-4 pt-4 pb-6 space-y-3 font-mono text-sm">
          {navLinks.map((link) => (
            <Link
              key={link.label}
              to={link.to}
              onClick={() => setMobileMenuOpen(false)}
              className="block py-2 px-3 border-b border-white/10 hover:bg-[#0B1A5C] hover:text-[#C8FF00] transition-colors"
            >
              {link.label}
            </Link>
          ))}
          <div className="pt-2">
            <Link
              to="/detect"
              onClick={() => setMobileMenuOpen(false)}
              className="w-full flex items-center justify-center gap-2 py-3 bg-[#C8FF00] text-[#0B1A5C] font-bold border-2 border-[#0B1A5C] shadow-[3px_3px_0px_#0B1A5C]"
            >
              <Search className="w-4 h-4" />
              <span>SCAN IMAGE NOW</span>
            </Link>
          </div>
        </div>
      )}
    </header>
  )
}
