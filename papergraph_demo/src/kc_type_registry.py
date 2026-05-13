from __future__ import annotations


TEXT_KC_TYPES = {
    "problem",
    "method",
    "mechanism",
    "dataset",
    "experiment",
    "evaluation",
    "ablation",
    "result",
    "conclusion",
    "limitation",
    "background",
    "central_claim",
    "algorithm",
    "analysis",
    "motivation",
}

MULTIMODAL_KC_TYPES = {
    "table_result",
    "table_comparison",
    "table_ablation",
    "visual_component",
    "visual_mechanism",
    "visual_pipeline",
    "chart_trend",
    "multimodal_limitation",
}

VALID_KC_TYPES = TEXT_KC_TYPES | MULTIMODAL_KC_TYPES


def valid_kc_type(value: object, allowed_types: set[str] | None = None) -> str | None:
    text = str(value or "").strip()
    allowed = allowed_types or VALID_KC_TYPES
    return text if text in allowed else None
