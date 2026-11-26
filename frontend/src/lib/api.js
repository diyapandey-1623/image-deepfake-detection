/**
 * src/lib/api.js
 * --------------
 * Lightweight API client for FastAPI backend with automatic timeout,
 * friendly error transformation, and optional mock data fallback.
 */

import { MOCK_RESULTS, MOCK_HISTORY } from '../mocks/mockResult'

const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000'
const USE_MOCK = import.meta.env.VITE_USE_MOCK === 'true'

// In-memory session cache for reports viewed during current session
const sessionReportCache = new Map()

/**
 * Uploads an image file to POST /analyze with an optional 30s timeout.
 */
export async function analyzeImage(file, options = {}) {
  // If mock mode is explicitly enabled
  if (USE_MOCK) {
    await new Promise((res) => setTimeout(res, 2200))
    const isLikelyFake = file.name.toLowerCase().includes('fake') || file.name.toLowerCase().includes('synthetic')
    const mock = isLikelyFake ? MOCK_RESULTS.fake : MOCK_RESULTS.real
    const report = {
      ...mock,
      report_id: `rep_${Math.random().toString(36).substring(2, 10)}`,
      filename: file.name,
      created_at: new Date().toISOString(),
    }
    sessionReportCache.set(report.report_id, report)
    return report
  }

  const formData = new FormData()
  formData.append('file', file)

  const controller = new AbortController()
  const timeoutId = setTimeout(() => controller.abort(), 30000) // 30s timeout

  try {
    const response = await fetch(`${API_BASE_URL}/analyze`, {
      method: 'POST',
      body: formData,
      signal: options.signal || controller.signal,
    })

    clearTimeout(timeoutId)

    if (!response.ok) {
      let message = 'Failed to analyze image'
      try {
        const errorData = await response.json()
        message = errorData.detail || message
      } catch (e) {
        if (response.status === 413) {
          message = 'File size exceeds maximum limit of 10 MB.'
        } else if (response.status === 503) {
          message = 'The AI model is currently initializing. Please try again in a few seconds.'
        } else {
          message = `Server returned error (HTTP ${response.status})`
        }
      }
      throw new Error(message)
    }

    const data = await response.json()
    // Cache for immediate viewing
    sessionReportCache.set(data.report_id, data)
    return data
  } catch (err) {
    clearTimeout(timeoutId)
    if (err.name === 'AbortError') {
      throw new Error('Analysis timed out after 30 seconds. Please try again with a smaller image.')
    }
    if (err.message && (err.message.includes('Failed to fetch') || err.message.includes('NetworkError'))) {
      throw new Error("Can't reach the server, please try again or verify the backend is running.")
    }
    throw err
  }
}

/**
 * Fetches recent scan records from GET /history
 */
export async function getHistory(limit = 50) {
  if (USE_MOCK) {
    await new Promise((res) => setTimeout(res, 300))
    return MOCK_HISTORY
  }

  try {
    const response = await fetch(`${API_BASE_URL}/history?limit=${limit}`)
    if (!response.ok) {
      throw new Error(`Failed to load history (HTTP ${response.status})`)
    }
    return await response.json()
  } catch (err) {
    if (err.message && (err.message.includes('Failed to fetch') || err.message.includes('NetworkError'))) {
      // Fallback to in-memory session cache if server down
      const cached = Array.from(sessionReportCache.values()).map((r, idx) => ({
        id: idx + 1,
        report_id: r.report_id,
        filename: r.filename || 'uploaded_image.jpg',
        verdict: r.verdict,
        fake_probability: r.fake_probability,
        face_detected: r.checks?.face_detected ?? true,
        total_faces: r.faces?.length || 1,
        metadata_flags: r.checks?.metadata_flags || [],
        ela_flag: r.checks?.ela_flag ?? false,
        model_name: r.model_name,
        processing_time_ms: r.processing_time_ms || 120,
        created_at: r.created_at,
      }))
      if (cached.length > 0) return cached
      throw new Error("Can't reach the server, please try again.")
    }
    throw err
  }
}

/**
 * Retrieves a specific scan report by report ID.
 */
export async function getReportById(reportId) {
  // Check memory session cache first
  if (sessionReportCache.has(reportId)) {
    return sessionReportCache.get(reportId)
  }

  // Look in history list
  try {
    const history = await getHistory(100)
    const match = history.find((h) => h.report_id === reportId)
    if (match) {
      return {
        report_id: match.report_id,
        verdict: match.verdict,
        fake_probability: match.fake_probability,
        faces: [],
        heatmap_base64: null,
        checks: {
          face_detected: match.face_detected,
          metadata_flags: match.metadata_flags || [],
          ela_flag: match.ela_flag,
        },
        model_name: match.model_name,
        processing_time_ms: match.processing_time_ms,
        created_at: match.created_at,
        filename: match.filename,
      }
    }
  } catch (e) {
    // Continue to mock fallback
  }

  if (USE_MOCK) {
    return MOCK_RESULTS.fake
  }

  return null
}

/**
 * Returns the download URL for a scan's PDF report.
 */
export function getReportPdfUrl(reportId) {
  return `${API_BASE_URL}/report/${reportId}/pdf`
}
