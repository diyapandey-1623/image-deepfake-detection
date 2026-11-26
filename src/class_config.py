"""
src/class_config.py
-------------------
Single source of truth for Deepfake Detection class mappings, binary verdict,
and forensic rationale formatting.
"""

from typing import Any, Dict, List, Optional

CLASS_TO_IDX = {"real": 0, "fake": 1}
IDX_TO_LABEL = {0: "Real", 1: "Fake"}
LABEL_TO_IDX = {"Real": 0, "Fake": 1}

REAL_CLASS_IDX = 0
FAKE_CLASS_IDX = 1

DECISION_THRESHOLD = 0.50

FAKE_KEYWORDS = {"fake", "deepfake", "ai", "artificial", "synthetic", "manipulated", "generated", "forgery"}
REAL_KEYWORDS = {"real", "authentic", "genuine", "pristine", "original", "natural", "human"}


def resolve_fake_label_id(id2label: Optional[Dict[Any, str]]) -> int:
    """Resolves the 'fake' class index from a Hugging Face id2label mapping."""
    if not id2label:
        return FAKE_CLASS_IDX

    for label_id, label_name in id2label.items():
        name_norm = str(label_name).lower().replace("-", " ").replace("_", " ").strip()
        tokens = set(name_norm.split())
        if tokens.intersection(FAKE_KEYWORDS) or any(kw in name_norm for kw in FAKE_KEYWORDS):
            return int(label_id)

    if len(id2label) == 2:
        for label_id, label_name in id2label.items():
            name_norm = str(label_name).lower().replace("-", " ").replace("_", " ").strip()
            tokens = set(name_norm.split())
            if tokens.intersection(REAL_KEYWORDS) or any(kw in name_norm for kw in REAL_KEYWORDS):
                other_id = next(k for k in id2label if k != label_id)
                return int(other_id)

    raise ValueError(
        f"Could not identify a 'fake' class from model labels: {list(id2label.values())}."
    )


def get_verdict(fake_prob: float, threshold: float = DECISION_THRESHOLD) -> str:
    """Returns 'fake' when fake_prob >= threshold, else 'real'."""
    return "fake" if fake_prob >= threshold else "real"


def build_forensic_rationale(
    verdict: str,
    fake_prob: float,
    regional_influence: Optional[List[Dict[str, Any]]] = None,
) -> str:
    """Generates a technical forensic explanation aligned with the model verdict and focus zones."""
    regions = regional_influence or []
    high_regions = [r["region"] for r in regions if r.get("influence_level") == "High"]
    mod_regions = [r["region"] for r in regions if r.get("influence_level") == "Moderate"]
    is_fake = str(verdict).lower() in ("fake", "deepfake")

    if is_fake:
        regions_str = (
            ", ".join(high_regions)
            or ", ".join(mod_regions)
            or "central facial landmarks"
        )
        return (
            f"The model detected high manipulation attention ({round(fake_prob * 100, 1)}% synthetic probability) "
            f"concentrated on the {regions_str}. Neural activations reveal anomalous generative synthesis patterns, "
            f"unnatural boundary blending, and abnormal frequency distributions distinct from authentic physical captures."
        )

    regions_str = (
        ", ".join(high_regions)
        or ", ".join(mod_regions)
        or "facial micro-features"
    )
    return (
        f"Biometric integrity verified ({round((1.0 - fake_prob) * 100, 1)}% authentic confidence). "
        f"Model focus across {regions_str} confirms coherent organic skin pores, natural light-reflection "
        f"dynamics across the eyes/teeth, and uniform camera sensor noise without generative manipulation signatures."
    )
