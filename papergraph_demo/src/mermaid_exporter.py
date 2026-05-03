from __future__ import annotations


def export_master_graph_mermaid(master_graph: dict) -> str:
    lines = ["graph TD"]
    for kc in master_graph.get("kc_nodes", []):
        label = kc["short_label"].replace('"', "'")
        lines.append(f'    {kc["kc_id"]}["{kc["kc_id"]}: {label}"]')
    for edge in master_graph.get("reasoning_edges", []):
        lines.append(f'    {edge["source"]} -- {edge["relation"]} --> {edge["target"]}')
    return "\n".join(lines) + "\n"


def export_macro_spine_mermaid(macro_spine: dict) -> str:
    lines = ["graph TD"]
    for macro in sorted(macro_spine.get("macro_nodes", []), key=lambda item: int(item.get("order") or 0)):
        macro_id = macro.get("macro_id", "")
        title = _safe_label(macro.get("title", macro_id))
        role = _safe_label(macro.get("role", ""))
        lines.append(f'    {macro_id}["{macro_id}: {title}<br/>{role}"]')
    for edge in macro_spine.get("macro_edges", []):
        source = edge.get("source")
        target = edge.get("target")
        relation = _safe_label(edge.get("relation", "relates_to"))
        if source and target:
            lines.append(f"    {source} -- {relation} --> {target}")
    return "\n".join(lines) + "\n"


def export_reasoning_threads_mermaid(reasoning_threads: dict) -> str:
    threads = reasoning_threads.get("threads", reasoning_threads if isinstance(reasoning_threads, list) else [])
    lines = ["graph TD"]
    for thread in threads:
        thread_id = thread.get("thread_id", "RT")
        label = _safe_label(thread.get("thread_type", "thread"))
        lines.append(f'    {thread_id}["{thread_id}: {label}"]')
        steps = thread.get("planned_turns", [])
        for step in steps:
            step_id = step.get("thread_turn_id")
            if not step_id:
                continue
            role = _safe_label(step.get("role", "step"))
            targets = ",".join(step.get("target_kc_ids", []))
            lines.append(f'    {step_id}["{step_id}: {role}<br/>{_safe_label(targets)}"]')
            lines.append(f"    {thread_id} --> {step_id}")
        _add_thread_step_edges(lines, steps)
    return "\n".join(lines) + "\n"


def export_final_state_mermaid(master_graph: dict, eval_state: dict) -> str:
    icon = {
        "lit": "✅",
        "corrected": "🔁",
        "missing": "⚠️",
        "hallucinated": "❌",
        "failed": "❌",
        "unlit": "⬜",
    }
    lines = ["graph TD"]
    for kc in master_graph.get("kc_nodes", []):
        st = eval_state.get("kc_states", {}).get(kc["kc_id"], {}).get("status", "unlit")
        label = kc["short_label"].replace('"', "'")
        lines.append(f'    {kc["kc_id"]}["{icon.get(st, "⬜")} {kc["kc_id"]}: {label}"]')
    for edge in master_graph.get("reasoning_edges", []):
        lines.append(f'    {edge["source"]} --> {edge["target"]}')
    return "\n".join(lines) + "\n"


def export_final_thread_state_mermaid(master_graph: dict, eval_state: dict) -> str:
    lines = ["graph TD"]
    thread_states = eval_state.get("thread_states", {})
    for thread in master_graph.get("reasoning_threads", []):
        thread_id = thread.get("thread_id", "RT")
        state = thread_states.get(thread_id, {})
        status = _safe_label(state.get("status", "not_started"))
        lines.append(f'    {thread_id}["{thread_id}: {status}"]')
        completed = set(state.get("completed_steps", []))
        for step in thread.get("planned_turns", []):
            step_id = step.get("thread_turn_id")
            if not step_id:
                continue
            role = _safe_label(step.get("role", "step"))
            step_status = "done" if step_id in completed else "pending"
            lines.append(f'    {step_id}["{step_id}: {role}<br/>{step_status}"]')
            lines.append(f"    {thread_id} --> {step_id}")
        _add_thread_step_edges(lines, thread.get("planned_turns", []))
    return "\n".join(lines) + "\n"


def _add_thread_step_edges(lines: list[str], steps: list[dict]) -> None:
    by_role = {step.get("role"): step.get("thread_turn_id") for step in steps}
    premise = by_role.get("establish_premise")
    evidence = by_role.get("establish_evidence")
    bridge = by_role.get("bridge_reasoning")
    review = by_role.get("review_consistency")
    if premise and bridge:
        lines.append(f"    {premise} --> {bridge}")
    if evidence and bridge:
        lines.append(f"    {evidence} --> {bridge}")
    if bridge and review:
        lines.append(f"    {bridge} --> {review}")
    if not bridge:
        ordered = [step.get("thread_turn_id") for step in steps if step.get("thread_turn_id")]
        for left, right in zip(ordered, ordered[1:]):
            lines.append(f"    {left} --> {right}")


def _safe_label(value: object) -> str:
    return str(value or "").replace('"', "'").replace("\n", " ")[:120]
