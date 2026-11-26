/**
 * src/mocks/mockResult.js
 * -----------------------
 * Deterministic mock data for testing UI states, mobile responsiveness,
 * and component behavior without an active backend.
 */

// Simple 1x1 transparent/colored PNG base64 for fallback preview
export const DUMMY_HEATMAP_BASE64 =
  'data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=='

export const MOCK_RESULTS = {
  fake: {
    report_id: 'rep_demo_fake_882',
    verdict: 'fake',
    fake_probability: 0.884,
    faces: [
      {
        bbox: [120, 80, 240, 240],
        fake_probability: 0.884,
      },
    ],
    heatmap_base64: DUMMY_HEATMAP_BASE64,
    checks: {
      face_detected: true,
      metadata_flags: ['Photoshop CS6', 'ExifTool Stamp'],
      ela_flag: true,
    },
    model_name: 'custom_pytorch (efficientnet_b0)',
    processing_time_ms: 114.2,
    created_at: new Date().toISOString(),
  },

  real: {
    report_id: 'rep_demo_real_104',
    verdict: 'real',
    fake_probability: 0.126,
    faces: [
      {
        bbox: [110, 75, 230, 230],
        fake_probability: 0.126,
      },
    ],
    heatmap_base64: null,
    checks: {
      face_detected: true,
      metadata_flags: [],
      ela_flag: false,
    },
    model_name: 'dima806/deepfake_vs_real_image_detection',
    processing_time_ms: 98.4,
    created_at: new Date(Date.now() - 3600000 * 2).toISOString(),
  },
}

export const MOCK_HISTORY = [
  {
    id: 1,
    report_id: 'rep_demo_fake_882',
    filename: 'portrait_ai_avatar.png',
    verdict: 'fake',
    fake_probability: 0.884,
    face_detected: true,
    total_faces: 1,
    metadata_flags: ['Photoshop CS6'],
    ela_flag: true,
    model_name: 'custom_pytorch (efficientnet_b0)',
    processing_time_ms: 114.2,
    created_at: new Date().toISOString(),
  },
  {
    id: 2,
    report_id: 'rep_demo_real_104',
    filename: 'office_headshot.jpg',
    verdict: 'real',
    fake_probability: 0.126,
    face_detected: true,
    total_faces: 1,
    metadata_flags: [],
    ela_flag: false,
    model_name: 'custom_pytorch (efficientnet_b0)',
    processing_time_ms: 98.4,
    created_at: new Date(Date.now() - 3600000 * 4).toISOString(),
  },
  {
    id: 3,
    report_id: 'rep_demo_unc_551',
    filename: 'blurred_profile_picture.webp',
    verdict: 'uncertain',
    fake_probability: 0.512,
    face_detected: false,
    total_faces: 0,
    metadata_flags: ['WebP Lossy'],
    ela_flag: true,
    model_name: 'prithivMLmods/Deep-Fake-Detector-v2-Model',
    processing_time_ms: 142.8,
    created_at: new Date(Date.now() - 3600000 * 24).toISOString(),
  },
  {
    id: 4,
    report_id: 'rep_demo_fake_331',
    filename: 'synthetic_conference_speaker.jpg',
    verdict: 'fake',
    fake_probability: 0.941,
    face_detected: true,
    total_faces: 1,
    metadata_flags: ['Midjourney Generated'],
    ela_flag: true,
    model_name: 'custom_pytorch (efficientnet_b0)',
    processing_time_ms: 108.5,
    created_at: new Date(Date.now() - 3600000 * 48).toISOString(),
  },
]
