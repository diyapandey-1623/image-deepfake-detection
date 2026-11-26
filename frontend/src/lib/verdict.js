/**
 * Centralized helpers for binary verdict classification, confidence tiers,
 * and human-readable forensic explanations.
 */

import { DECISION_THRESHOLD } from './class_config'

export function isFakeVerdict(fakeProbability = 0.5, rawVerdict = null) {
  const verdictLower = String(rawVerdict || '').toLowerCase()
  if (verdictLower === 'fake' || verdictLower === 'deepfake') return true
  if (verdictLower === 'real') return false
  return (Number(fakeProbability) || 0.0) >= DECISION_THRESHOLD
}

export function getVerdictData(fakeProbability = 0.5, rawVerdict = null) {
  const prob = Number(fakeProbability) || 0.0
  const percent = Math.round(prob * 100)
  const isFake = isFakeVerdict(prob, rawVerdict)

  const type = isFake ? 'fake' : 'real'
  const label = isFake ? 'Likely Fake' : 'Likely Real'
  const textColor = isFake ? 'text-verdict-fake' : 'text-verdict-real'
  const bgColor = isFake ? 'bg-verdict-fakeBg' : 'bg-verdict-realBg'
  const borderColor = isFake ? 'border-verdict-fakeBorder' : 'border-verdict-realBorder'
  const barColor = isFake ? 'bg-verdict-fake' : 'bg-verdict-real'

  const dist = Math.abs(prob - DECISION_THRESHOLD)
  let confidenceTier = 'Low'
  let confidenceColor = 'text-slate-600 bg-slate-100 border-slate-200'

  if (dist >= 0.4) {
    confidenceTier = 'High'
    confidenceColor = 'text-primary-dark bg-primary-light/50 border-primary/30'
  } else if (dist >= 0.2) {
    confidenceTier = 'Medium'
    confidenceColor = 'text-amber-700 bg-amber-50 border-amber-200'
  }

  const explanation = isFake
    ? `The model is ${percent}% sure this image is AI-generated or manipulated. Higher scores indicate greater likelihood of manipulation.`
    : `The model is ${100 - percent}% confident this photo is an authentic camera photograph without synthetic modifications.`

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

export function deriveReasons(result) {
  if (!result) return []

  const reasons = []
  const prob = result.fake_probability ?? 0.5
  const checks = result.checks || {}
  const faces = result.faces || []
  const isFake = isFakeVerdict(prob, result.verdict)

  if (isFake) {
    reasons.push({
      id: 'high-prob',
      title: 'Structural Generator Patterns',
      text: 'Deep neural pattern analysis detected synthetic texture artifacts characteristic of generative models.',
      isWarning: true,
    })
  } else {
    reasons.push({
      id: 'low-prob',
      title: 'Natural Biometric Coherence',
      text: 'Facial features, natural lighting gradients, and skin micropores match authentic photographic camera captures.',
      isWarning: false,
    })
  }

  if (checks.ela_flag) {
    reasons.push({
      id: 'ela-flag',
      title: 'Inconsistent Compression Artifacts',
      text: 'Error Level Analysis detected uneven JPEG recompression residuals across facial regions.',
      isWarning: true,
    })
  } else {
    reasons.push({
      id: 'ela-clean',
      title: 'Uniform Compression Residual',
      text: 'Error Level Analysis confirms consistent image compression across the analyzed frame.',
      isWarning: false,
    })
  }

  if (checks.metadata_flags && checks.metadata_flags.length > 0) {
    reasons.push({
      id: 'meta-flag',
      title: 'Editing Software / Header Flags',
      text: `Metadata audit flagged: ${checks.metadata_flags.join(', ')}.`,
      isWarning: true,
    })
  } else {
    reasons.push({
      id: 'meta-clean',
      title: 'No Software Manipulation Tags',
      text: 'No known editing software or AI generator signatures were found in image metadata.',
      isWarning: false,
    })
  }

  if (faces.length > 1) {
    const highestFace = Math.max(...faces.map((f) => f.fake_probability || 0))
    reasons.push({
      id: 'multi-face',
      title: `Multiple Faces Analyzed (${faces.length})`,
      text: highestFace >= DECISION_THRESHOLD
        ? `Primary manipulation indicators detected on one or more faces (max score ${(highestFace * 100).toFixed(0)}%).`
        : 'All detected faces scored within authentic camera ranges.',
      isWarning: highestFace >= DECISION_THRESHOLD,
    })
  } else if (checks.face_detected === false) {
    reasons.push({
      id: 'no-face',
      title: 'Full Frame Fallback Analysis',
      text: 'No prominent face was localized by MTCNN; evaluation was performed across the full image.',
      isWarning: false,
    })
  } else {
    reasons.push({
      id: 'face-localized',
      title: 'Facial Boundary Alignment',
      text: 'MTCNN localized the primary face with a +20% boundary margin to preserve hairline and jawline context.',
      isWarning: false,
    })
  }

  return reasons
}
