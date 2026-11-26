/**
 * API client for the FastAPI deepfake detection backend.
 */

import { MOCK_RESULTS, MOCK_HISTORY } from '../mocks/mockResult'

export const API_BASE_URL =
  import.meta.env.VITE_API_URL || (import.meta.env.PROD ? '' : 'http://127.0.0.1:8000')
const USE_MOCK = import.meta.env.VITE_USE_MOCK === 'true'
const REQUEST_TIMEOUT_MS = 30000

const sessionReportCache = new Map()

function isNetworkError(err) {
  return Boolean(
    err?.message &&
    (err.message.includes('Failed to fetch') || err.message.includes('NetworkError'))
  )
}

async function parseErrorMessage(response) {
  try {
    const errorData = await response.json()
    if (errorData?.detail) return errorData.detail
  } catch {
    // Non-JSON error body; fall back to HTTP status message below.
  }
  if (response.status === 413) return 'File size exceeds maximum limit of 10 MB.'
  if (response.status === 503) return 'The AI model is currently initializing. Please try again in a few seconds.'
  return `Server returned error (HTTP ${response.status})`
}

export async function analyzeImage(file, options = {}) {
  if (USE_MOCK) {
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
  const timeoutId = setTimeout(() => controller.abort(), REQUEST_TIMEOUT_MS)

  if (options.signal) {
    if (options.signal.aborted) {
      controller.abort()
    } else {
      options.signal.addEventListener('abort', () => controller.abort(), { once: true })
    }
  }

  try {
    const response = await fetch(`${API_BASE_URL}/analyze`, {
      method: 'POST',
      body: formData,
      signal: controller.signal,
    })
    clearTimeout(timeoutId)

    if (!response.ok) {
      throw new Error(await parseErrorMessage(response))
    }

    const data = await response.json()
    sessionReportCache.set(data.report_id, data)
    return data
  } catch (err) {
    clearTimeout(timeoutId)
    if (err.name === 'AbortError') {
      if (options.signal?.aborted) {
        throw err
      }
      throw new Error('Analysis timed out after 30 seconds. Please try again with a smaller image.')
    }
    if (isNetworkError(err)) {
      throw new Error("Can't reach the server, please verify the backend is running.")
    }
    throw err
  }
}

export async function getHistory(limit = 50) {
  if (USE_MOCK) {
    return MOCK_HISTORY
  }

  try {
    const response = await fetch(`${API_BASE_URL}/history?limit=${limit}`)
    if (!response.ok) {
      throw new Error(`Failed to load history (HTTP ${response.status})`)
    }
    return await response.json()
  } catch (err) {
    if (isNetworkError(err)) {
      throw new Error("Can't reach the server, please verify the backend is running.")
    }
    throw err
  }
}

export async function getReportById(reportId) {
  if (sessionReportCache.has(reportId)) {
    return sessionReportCache.get(reportId)
  }

  const history = await getHistory(100)
  const match = history.find((h) => h.report_id === reportId)
  if (!match) return null

  return {
    report_id: match.report_id,
    verdict: match.verdict,
    fake_probability: match.fake_probability,
    faces: match.faces || [],
    total_faces: match.total_faces,
    heatmap_base64: match.heatmap_base64 || null,
    regional_influence: match.regional_influence || [],
    forensic_rationale: match.forensic_rationale || '',
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

export function getReportPdfUrl(reportId) {
  return `${API_BASE_URL}/report/${reportId}/pdf`
}
