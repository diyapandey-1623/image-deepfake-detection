"""
backend/class_config.py
-----------------------
Single source of truth for Deepfake Detection class mappings.
Synchronized across training, evaluation, backend inference, and frontend.
"""

# Binary class mapping
CLASS_TO_IDX = {"real": 0, "fake": 1}
IDX_TO_LABEL = {0: "Real", 1: "Fake"}
LABEL_TO_IDX = {"Real": 0, "Fake": 1}

REAL_LABEL = "Real"
FAKE_LABEL = "Fake"

REAL_CLASS_IDX = 0
FAKE_CLASS_IDX = 1

# Standard classification threshold: >= 0.50 is Fake, < 0.50 is Real
DECISION_THRESHOLD = 0.50
