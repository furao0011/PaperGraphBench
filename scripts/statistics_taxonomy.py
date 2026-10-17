"""The paper's four failure categories, using its original exact mapping.

This is a reporting taxonomy, not the Judge's raw event-family schema. Unmapped
raw types remain explicitly uncategorized; no family or subtype is guessed.
"""

from __future__ import annotations


PAPER_CATEGORIES = (
    "contradicted_kc_claim",
    "challenge_false_premise_acceptance",
    "challenge_overclaim_acceptance",
    "challenge_wrong_relation_acceptance",
)
PAPER_CATEGORY_LABELS = {
    "contradicted_kc_claim": {"en": "Contrad. KC", "zh": "与知识组件矛盾"},
    "challenge_false_premise_acceptance": {"en": "False Prem.", "zh": "错误前提接受"},
    "challenge_overclaim_acceptance": {"en": "Overclaim", "zh": "过度主张接受"},
    "challenge_wrong_relation_acceptance": {"en": "Wrong Rel.", "zh": "错误关系接受"},
}
PAPER_CATEGORY_COLUMNS = (
    "model", "status", "task_total", "completed_tasks", "incomplete_tasks",
    *PAPER_CATEGORIES, "categorized_event_count", "unmapped_event_count", "event_count",
)
_FAILURE_MAPPING = {
    "contradicted_kc_claim": "contradicted_kc_claim",
    "review_regression": "contradicted_kc_claim",
    "challenge_false_premise_acceptance": "challenge_false_premise_acceptance",
    "thread_premise_mutation_acceptance": "challenge_false_premise_acceptance",
    "challenge_overclaim_acceptance": "challenge_overclaim_acceptance",
    "thread_overclaim_acceptance": "challenge_overclaim_acceptance",
    "challenge_wrong_relation_acceptance": "challenge_wrong_relation_acceptance",
    "thread_wrong_bridge_acceptance": "challenge_wrong_relation_acceptance",
    "thread_reasoning_hallucination": "challenge_wrong_relation_acceptance",
}


def collapsed_failure_type(raw_type: str | None) -> str | None:
    """Return only an exact paper category match; other types are unmapped."""
    return _FAILURE_MAPPING.get(raw_type) if isinstance(raw_type, str) else None


def _task_key(value: dict) -> tuple[str, str, str]:
    key = tuple(value.get(name) for name in ("round", "model", "paper_id"))
    if any(not isinstance(item, str) or not item.strip() for item in key):
        raise ValueError("Paper-category records require round, model and paper_id strings.")
    return key


def paper_category_rows(events: list[dict], rows: list[dict]) -> list[dict]:
    """Count completed-task events per model, preserving the paper's four bins.

    Events use ``type`` from ``collect_classification``. The round/model/paper/
    event-ID scope prevents merging unrelated events with the same local ID.
    """
    tasks, output = {}, {}
    for row in rows:
        key = _task_key(row)
        if key in tasks:
            raise ValueError(f"Duplicate paper-category task: {key}")
        tasks[key] = row
        model = row["model"]
        result = output.setdefault(model, {"model": model, "status": "completed", "task_total": 0,
            "completed_tasks": 0, "incomplete_tasks": 0, **{category: 0 for category in PAPER_CATEGORIES},
            "categorized_event_count": 0, "unmapped_event_count": 0, "event_count": 0})
        result["task_total"] += 1
        result["completed_tasks" if row["status"] == "completed" else "incomplete_tasks"] += 1
        if row["status"] != "completed":
            result["status"] = "incomplete"
    seen = set()
    for event in events:
        key = _task_key(event)
        if key not in tasks:
            raise ValueError(f"Paper-category event has no task: {key}")
        if tasks[key]["status"] != "completed":
            continue
        event_id = event.get("event_id")
        if not isinstance(event_id, str) or not event_id.strip():
            raise ValueError("Paper-category events require a nonempty event_id.")
        scoped_id = (*key, event_id)
        if scoped_id in seen:
            raise ValueError(f"Duplicate scoped paper-category event: {scoped_id}")
        seen.add(scoped_id)
        result = output[event["model"]]
        result["event_count"] += 1
        category = collapsed_failure_type(event.get("type"))
        if category is None:
            result["unmapped_event_count"] += 1
        else:
            result[category] += 1
            result["categorized_event_count"] += 1
    return [output[model] for model in sorted(output)]
