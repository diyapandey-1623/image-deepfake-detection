import React from 'react'
import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom'
import Navbar from './components/Navbar'
import Footer from './components/Footer'
import Overview from './pages/Overview'
import Home from './pages/Home'
import Result from './pages/Result'
import History from './pages/History'

// Shell wrapper for sub-pages to maintain uniform FakeLens navigation & footer
function PageLayout({ children }) {
  return (
    <div className="min-h-screen bg-[#FFF9F7] text-[#0B1A5C] flex flex-col justify-between selection:bg-[#C8FF00] selection:text-[#0B1A5C]">
      <Navbar />
      <div className="flex-1 w-full py-8">
        {children}
      </div>
      <Footer />
    </div>
  )
}

export default function App() {
  return (
    <BrowserRouter>
      <Routes>
        {/* FakeLens Overview Landing Page */}
        <Route path="/" element={<Overview />} />

        {/* Detect / Dedicated Full Scan Page */}
        <Route
          path="/detect"
          element={
            <PageLayout>
              <Home />
            </PageLayout>
          }
        />

        {/* Result Report Page */}
        <Route
          path="/result/:id"
          element={
            <PageLayout>
              <Result />
            </PageLayout>
          }
        />

        {/* History Scan Log Page */}
        <Route
          path="/history"
          element={
            <PageLayout>
              <History />
            </PageLayout>
          }
        />

        {/* Fallback to Overview */}
        <Route path="*" element={<Navigate to="/" replace />} />
      </Routes>
    </BrowserRouter>
  )
}
