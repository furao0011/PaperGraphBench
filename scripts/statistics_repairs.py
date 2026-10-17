"""Reconstruct paper repair tasks from recorded repair contexts and answers."""

from __future__ import annotations

from collections import defaultdict

from statistics_trajectory import coverage_complete, events, kc_ids, validated_turns


def _context(turn: dict) -> dict:
    context = turn.get("repair_context", {})
    if not isinstance(context, dict):
        raise ValueError(f"Turn {turn.get('turn_id')} repair_context must be an object")
    return context


def _success(turn: dict, repair_type: str) -> bool:
    if repair_type == "hallucination":
        state = turn["judge_result"].get("state")
        return state == "SELF_CORRECTED" or (state in {"MAIN_PROGRESS", "CHALLENGE_RESISTED", "THREAD_PROGRESS"} and not events(turn))
    if coverage_complete(turn):
        return True
    remaining = _context(turn).get("remaining_kc_ids", [])
    if not isinstance(remaining, list) or any(not isinstance(item, str) or not item.strip() for item in remaining):
        raise ValueError(f"Turn {turn.get('turn_id')} remaining_kc_ids must be a list of nonempty strings")
    return bool(remaining) and not (set(remaining) & kc_ids(turn, "missing"))


def extract_repair_tasks(trajectory: dict) -> list[dict]:
    """One task per repair type/root turn/root question; stop at first success."""
    turns = validated_turns(trajectory)
    indices = {turn["turn_id"]: index for index, turn in enumerate(turns, start=1)}
    grouped = defaultdict(list)
    for index, turn in enumerate(turns, start=1):
        context = _context(turn)
        if not context.get("repair_type") and not context.get("root_turn_id"):
            continue
        for field in ("repair_type", "root_turn_id", "root_question_id"):
            if not isinstance(context.get(field), str) or not context[field].strip():
                raise ValueError(f"Turn {turn['turn_id']} repair context requires {field}")
        if context["repair_type"] not in {"detail", "hallucination"}:
            raise ValueError(f"Turn {turn['turn_id']} has unsupported repair_type")
        root = context["root_turn_id"]
        if root not in indices or indices[root] >= index:
            raise ValueError(f"Turn {turn['turn_id']} repair root must reference an earlier recorded turn")
        if turns[indices[root] - 1]["question_id"] != context["root_question_id"]:
            raise ValueError(f"Turn {turn['turn_id']} repair root question disagrees with root turn")
        grouped[tuple(context[field] for field in ("repair_type", "root_turn_id", "root_question_id"))].append((index, turn))
    result = []
    for (kind, root, question), items in sorted(grouped.items()):
        success = next(((index, turn) for index, turn in items if _success(turn, kind)), None)
        end_index, final_turn = success if success is not None else items[-1]
        attempts = sum(index <= end_index for index, _ in items)
        result.append({"repair_task_id": f"{kind}::{root}::{question}", "repair_type": kind,
                       "root_turn_id": root, "root_question_id": question,
                       "root_question_type": _context(items[0][1]).get("root_question_type"),
                       "start_turn_index": indices[root], "end_turn_index": end_index,
                       "attempts": attempts, "turn_span": end_index - indices[root],
                       "success": success is not None, "exhausted": attempts >= 3 and success is None,
                       "success_turn_id": final_turn["turn_id"] if success is not None else None,
                       "success_turn_index": end_index if success is not None else None,
                       "final_turn_id": final_turn["turn_id"], "final_judge_state": final_turn["judge_result"].get("state")})
    return result
