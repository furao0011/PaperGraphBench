from __future__ import annotations


def export_master_graph_mermaid(master_graph: dict) -> str:
    lines = ["graph TD"]
    for kc in master_graph.get("kc_nodes", []):
        label = kc["short_label"].replace('"', "'")
        lines.append(f'    {kc["kc_id"]}["{kc["kc_id"]}: {label}"]')
    for edge in master_graph.get("reasoning_edges", []):
        lines.append(f'    {edge["source"]} -- {edge["relation"]} --> {edge["target"]}')
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

