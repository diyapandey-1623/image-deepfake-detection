import React from 'react'
import Navbar from '../components/Navbar'
import Hero from '../components/Hero'
import Marquee from '../components/Marquee'
import ChecksList from '../components/ChecksList'
import DemoUpload from '../components/DemoUpload'
import Limits from '../components/Limits'
import Footer from '../components/Footer'

export default function Overview() {
  return (
    <div className="min-h-screen bg-[#0F45E3] text-white flex flex-col justify-between selection:bg-[#C8FF00] selection:text-[#0B1A5C]">
      {/* 1. Top Nav (Sticky, Blue) */}
      <Navbar />

      <main className="flex-1 w-full">
        {/* 2. Hero (Blue background, Headline, Lie struck through, Prove highlight, SpecimenCard) */}
        <Hero />

        {/* 3. Marquee strip (Full width endlessly scrolling bold text) */}
        <Marquee />

        {/* 4. "Four checks. Zero guesswork." section (Paper background, navy text) */}
        <ChecksList />

        {/* 5. Live demo teaser (Dashed drag & drop, lime Detect button, REAL/FAKE preview) */}
        <DemoUpload />

        {/* 6. Honest-limits section (Blue background, "No detector is perfect.", XX% placeholder) */}
        <Limits />
      </main>

      {/* 7. Footer (Logo, links, "Only analyze images you have the right to use.") */}
      <Footer />
    </div>
  )
}
