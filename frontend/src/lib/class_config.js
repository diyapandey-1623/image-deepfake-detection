/**
 * Single source of truth for Deepfake Detection class mappings in frontend.
 * Synchronized with backend/class_config.py and src/class_config.py.
 */
export const CLASS_TO_IDX = { real: 0, fake: 1 };
export const IDX_TO_LABEL = { 0: 'Real', 1: 'Fake' };
export const REAL_CLASS_IDX = 0;
export const FAKE_CLASS_IDX = 1;
export const DECISION_THRESHOLD = 0.50;
