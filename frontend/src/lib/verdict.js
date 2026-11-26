/**
 * src/lib/verdict.js
 * ------------------
 * Centralized helpers for computing user-friendly verdict classifications,
 * calibrated confidence tiers, and human-readable explanations.
 */

import { DECISION_THRESHOLD } from './class_config'

/**
 * Calibrates the verdict label, styling tokens, and confidence tier.
 * @param {number} fakeProbability - Value between 0.0 and 1.0
 * @param {string} rawVerdict - Optional verdict override from server
 */
export function getVerdictData(fakeProbability = 0.5, rawVerdict = null) {
  const prob = Number(fakeProbability) || 0.0
  const percent = Math.round(prob * 100)

  // Standardized binary decision threshold: >= 0.50 = Fake, < 0.50 = Real
  const isExplicitFake = String(rawVerdict).toLowerCase() === 'fake' || String(rawVerdict).toLowerCase() === 'deepfake'
  const isExplicitReal = String(rawVerdict).toLowerCase() === 'real'
  const isFake = isExplicitFake || (!isExplicitReal && prob >= DECISION_THRESHOLD)

  let type = isFake ? 'fake' : 'real'
  let label = isFake ? 'Likely Fake' : 'Likely Real'
  let textColor = isFake ? 'text-verdict-fake' : 'text-verdict-real'
  let bgColor = isFake ? 'bg-verdict-fakeBg' : 'bg-verdict-realBg'
  let borderColor = isFake ? 'border-verdict-fakeBorder' : 'border-verdict-realBorder'
  let barColor = isFake ? 'bg-verdict-fake' : 'bg-verdict-real'

  // Confidence tier based on distance from the 0.5 boundary
  // < 0.2: Low | 0.2 - 0.4: Medium | > 0.4: High
  const dist = Math.abs(prob - 0.5)
  let confidenceTier = 'Low'
  let confidenceColor = 'text-slate-600 bg-slate-100 border-slate-200'

  if (dist >= 0.4) {
    confidenceTier = 'High'
    confidenceColor = 'text-primary-dark bg-primary-light/50 border-primary/30'
  } else if (dist >= 0.2) {
    confidenceTier = 'Medium'
    confidenceColor = 'text-amber-700 bg-amber-50 border-amber-200'
  }

  // Plain-English sentence explaining the score
  let explanation = ''
  if (type === 'fake') {
    explanation = `The model is ${percent}% sure this image is AI-generated or manipulated. Higher scores indicate greater likelihood of manipulation.`
  } else if (type === 'real') {
    const authenticPercent = 100 - percent
    explanation = `The model is ${authenticPercent}% confident this photo is an authentic camera photograph without synthetic modifications.`
  } else {
    explanation = `The detector returned a borderline score (${percent}%). The image exhibits mixed signals and cannot be decisively verified.`
  }

  return {
    type,
    label,
    percent,
    prob,
    textColor,
    bgColor,
    borderColor,
    barColor,
    confidenceTier,
    confidenceColor,
    explanation,
  }
}

/**
 * Derives 3-4 plain-English bullet reasons from technical checks.
 */
export function deriveReasons(result) {
  if (!result) return []

  const reasons = []
  const prob = result.fake_probability ?? 0.5
  const checks = result.checks || {}
  const faces = result.faces || []

  if (prob > 0.6) {
    reasons.push({
      id: 'high-prob',
      title: 'Structural Generator Patterns',
      text: `Deep neural pattern analysis detected synthetic texture artifacts characteristic of generative models.`,
      isWarning: true,
    })
  } else if (prob < 0.4) {
    reasons.push({
      id: 'low-prob',
      title: 'Natural Biometric Coherence',
      text: `Facial features, natural lighting gradients, and skin micropores match authentic photographic camera captures.`,
      isWarning: false,
    })
  } else {
    reasons.push({
      id: 'borderline',
      title: 'Ambiguous Sensory Signatures',
      text: `Features fall into the middle threshold band. Heavy compression, low resolution, or lighting may mimic synthetic noise.`,
      isWarning: true,
    })
  }

  // Check 2: Error Level Analysis
  if (checks.ela_flag) {
    reasons.push({
      id: 'ela',
      title: 'Compression Inconsistencies (ELA)',
      text: `Error Level Analysis revealed uneven JPEG compression history, suggesting composite editing or facial region splicing.`,
      isWarning: true,
    })
  } else {
    reasons.push({
      id: 'ela-clean',
      title: 'Uniform Compression Residuals',
      text: `Re-compression delta is uniform across the entire surface with no regional compression discontinuities.`,
      isWarning: false,
    })
  }

  // Check 3: Metadata Flags
  if (checks.metadata_flags && checks.metadata_flags.length > 0) {
    reasons.push({
      id: 'metadata',
      title: 'Software Modification Signatures',
      text: `File metadata contained digital editor signatures (${checks.metadata_flags.join(', ')}).`,
      isWarning: true,
    })
  } else {
    reasons.push({
      id: 'metadata-clean',
      title: 'Clean File Metadata',
      text: `No third-party image manipulation software tags or artificial generator stamps were detected.`,
      isWarning: false,
    })
  }

  // Check 4: Face Localization
  if (checks.face_detected && faces.length > 0) {
    const highestFace = Math.max(...faces.map(f => f.fake_probability || 0))
    reasons.push({
      id: 'faces',
      title: `Face Localization (${faces.length} detected)`,
      text: `High-resolution facial crops were successfully aligned and analyzed individually (peak score: ${Math.round(highestFace * 100)}%).`,
      isWarning: highestFace > 0.6,
    })
  } else {
    reasons.push({
      id: 'no-face',
      title: 'Global Scene Analysis',
      text: `No human face was localized with MTCNN; the system evaluated the entire image frame.`,
      isWarning: false,
    })
  }

  return reasons
}

/**
 * Format bytes into human-readable string.
 */
export function formatBytes(bytes, decimals = 1) {
  if (!bytes || bytes === 0) return '0 Bytes'
  const k = 1024
  const dm = decimals < 0 ? 0 : decimals
  const sizes = ['Bytes', 'KB', 'MB', 'GB']
  const i = Math.floor(Math.log(bytes) / Math.log(k))
  return parseFloat((bytes / Math.pow(k, i)).toFixed(dm)) + ' ' + sizes[i]
}

/**
 * Format ISO datetime into readable local time.
 */
export function formatDateTime(isoString) {
  if (!isoString) return 'Just now'
  try {
    const date = new Date(isoString)
    return new Intl.DateTimeFormat('en-US', {
      month: 'short',
      day: 'numeric',
      year: 'numeric',
      hour: 'numeric',
      minute: '2-digit',
    }).format(date)
  } catch (e) {
    return isoString
  }
}
