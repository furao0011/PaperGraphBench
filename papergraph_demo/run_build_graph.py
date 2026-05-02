import json
import os
from pathlib import Path

from src.active_kc_selector import select_active_kcs
from src.config import load_settings
from src.graph_builder import build_master_graph
from src.graph_builder import build_reasoning_edges_for_kcs
from src.kc_bank_builder import build_kc_bank
from src.kc_bank_builder import finalize_kc_bank_scores
from src.kc_extractor import extract_kc_candidates_by_sections
from src.macro_extractor import extract_macro_spine
from src.mermaid_exporter import export_master_graph_mermaid
from src.model_client import ModelConfig, OpenAICompatClient
from src.paper_parser import load_paper_text, load_paper_text_from_dir, split_into_sections
from src.progress import log, span
from src.reasoning_thread_builder import build_reasoning_threads


BASE_DIR = Path(__file__).resolve().parent
PAPER_PATH = BASE_DIR / "data" / "papers" / "demo_paper.md"
PAPER_DIR_PATH = BASE_DIR.parent / "util_example" / "output1"
GRAPH_PATH = BASE_DIR / "data" / "graphs" / "master_graph.json"
MASTER_MMD_PATH = BASE_DIR / "data" / "graphs" / "master_graph.mmd"
SECTIONS_PATH = BASE_DIR / "data" / "graphs" / "sections.json"
MACRO_SPINE_PATH = BASE_DIR / "data" / "graphs" / "macro_spine.json"
KC_BANK_PATH = BASE_DIR / "data" / "graphs" / "kc_bank.json"
ACTIVE_KC_PATH = BASE_DIR / "data" / "graphs" / "active_kc.json"
REASONING_THREADS_PATH = BASE_DIR / "data" / "graphs" / "reasoning_threads.json"


def main() -> None:
    project_root = BASE_DIR.parent
    settings = load_settings(project_root)
    log("build_graph configuration loaded", base_dir=BASE_DIR)

    input_dir = Path(os.getenv("PAPER_INPUT_DIR", str(PAPER_DIR_PATH)))
    input_file = Path(os.getenv("PAPER_INPUT_FILE", str(PAPER_PATH)))

    if input_dir.exists():
        log("loading paper from directory", input_dir=input_dir)
        with span("load paper directory"):
            paper_text = load_paper_text_from_dir(input_dir)
        paper_id = input_dir.name
        paper_text_path = str(input_dir)
    elif input_file.exists():
        log("loading paper from file", input_file=input_file)
        with span("load paper file"):
            paper_text = load_paper_text(input_file)
        paper_id = input_file.stem
        paper_text_path = str(input_file)
    else:
        raise FileNotFoundError(
            f"No valid input found. Checked directory: {input_dir}, file: {input_file}"
        )

    allow_offline_fallback = os.getenv("ALLOW_OFFLINE_FALLBACK", "false").lower() in {"1", "true", "yes", "on"}
    client = OpenAICompatClient(
        ModelConfig(
            api_key=settings.api_key,
            base_url=settings.base_url,
            llm_model=settings.llm_model,
            embed_base_url=settings.embed_base_url,
            embed_model=settings.embed_model,
        )
    )
    if not client.is_ready() and not allow_offline_fallback:
        raise RuntimeError("Online graph construction requires API_KEY, BASE_URL, and LLM_MODEL. Set ALLOW_OFFLINE_FALLBACK=true only for local debugging.")

    with span("split paper into sections", paper_chars=len(paper_text)):
        sections = split_into_sections(paper_text)
    log("sections ready", count=len(sections))
    SECTIONS_PATH.parent.mkdir(parents=True, exist_ok=True)
    SECTIONS_PATH.write_text(json.dumps({"paper_id": paper_id, "sections": sections}, ensure_ascii=False, indent=2), encoding="utf-8")
    log("sections written", path=SECTIONS_PATH)
    with span("extract macro spine", sections=len(sections)):
        macro_spine = extract_macro_spine(paper_id, sections, client)
    MACRO_SPINE_PATH.write_text(json.dumps(macro_spine, ensure_ascii=False, indent=2), encoding="utf-8")
    log("macro spine written", path=MACRO_SPINE_PATH)
    with span("extract KC candidates", sections=len(sections)):
        kc_candidates = extract_kc_candidates_by_sections(
            sections,
            client,
            allow_offline_fallback=allow_offline_fallback,
            macro_spine=macro_spine,
        )
    log("KC candidates extracted", count=len(kc_candidates))
    with span("build KC Bank", candidates=len(kc_candidates)):
        kc_bank = build_kc_bank(
            paper_id=paper_id,
            candidates=kc_candidates,
            macro_spine=macro_spine,
            client=client,
            allow_offline_fallback=allow_offline_fallback,
        )
    with span("build KC Bank reasoning edges", bank_kcs=len(kc_bank.get("kc_nodes", []))):
        bank_edges = build_reasoning_edges_for_kcs(
            kc_bank["kc_nodes"],
            _macro_nodes_with_bank_kcs(macro_spine, kc_bank),
            client,
            allow_offline_fallback=allow_offline_fallback,
        )
    finalize_kc_bank_scores(kc_bank, macro_spine, bank_edges)
    with span("select Active KCs", bank_kcs=len(kc_bank.get("kc_nodes", []))):
        active_kc = select_active_kcs(kc_bank, macro_spine)
    KC_BANK_PATH.write_text(json.dumps(kc_bank, ensure_ascii=False, indent=2), encoding="utf-8")
    log("KC Bank written", path=KC_BANK_PATH, kcs=len(kc_bank.get("kc_nodes", [])))
    ACTIVE_KC_PATH.write_text(json.dumps(active_kc, ensure_ascii=False, indent=2), encoding="utf-8")
    log("Active KCs written", path=ACTIVE_KC_PATH, active=len(active_kc.get("active_kc_ids", [])))
    with span("build master graph", kcs=len(active_kc.get("kc_nodes", []))):
        graph = build_master_graph(
            paper_id=paper_id,
            paper_text_path=paper_text_path,
            kcs=active_kc["kc_nodes"],
            client=client,
            allow_offline_fallback=allow_offline_fallback,
            macro_spine=macro_spine,
            kc_bank_path="data/graphs/kc_bank.json",
            active_kc_path="data/graphs/active_kc.json",
            precomputed_reasoning_edges=bank_edges,
        )
    with span("build reasoning threads", paths=len(graph.get("reasoning_paths", []))):
        reasoning_threads = build_reasoning_threads(
            paper_id=paper_id,
            macro_spine=macro_spine,
            active_kc=active_kc,
            reasoning_edges=graph.get("reasoning_edges", []),
            reasoning_paths=graph.get("reasoning_paths", []),
            client=client,
        )
    REASONING_THREADS_PATH.write_text(json.dumps(reasoning_threads, ensure_ascii=False, indent=2), encoding="utf-8")
    graph["reasoning_threads_path"] = "data/graphs/reasoning_threads.json"
    graph["reasoning_threads"] = reasoning_threads.get("threads", [])
    log(
        "master graph ready",
        macros=len(graph.get("macro_nodes", [])),
        kcs=len(graph.get("kc_nodes", [])),
        edges=len(graph.get("reasoning_edges", [])),
        paths=len(graph.get("reasoning_paths", [])),
        threads=len(graph.get("reasoning_threads", [])),
    )

    GRAPH_PATH.parent.mkdir(parents=True, exist_ok=True)
    GRAPH_PATH.write_text(
        json.dumps(graph, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    MASTER_MMD_PATH.write_text(export_master_graph_mermaid(graph), encoding="utf-8")
    log("graph artifacts written", graph=GRAPH_PATH, mermaid=MASTER_MMD_PATH)
    print(f"Sections written: {SECTIONS_PATH}")
    print(f"Macro spine generated: {MACRO_SPINE_PATH}")
    print(f"KC Bank generated: {KC_BANK_PATH}")
    print(f"Active KC generated: {ACTIVE_KC_PATH}")
    print(f"Reasoning threads generated: {REASONING_THREADS_PATH}")
    print(f"Master graph generated: {GRAPH_PATH}")


def _macro_nodes_with_bank_kcs(macro_spine: dict, kc_bank: dict) -> list[dict]:
    by_macro = {m["macro_id"]: [] for m in macro_spine.get("macro_nodes", [])}
    for kc in kc_bank.get("kc_nodes", []):
        by_macro.setdefault(kc.get("macro_id"), []).append(kc["kc_id"])
    out = []
    for macro in macro_spine.get("macro_nodes", []):
        item = dict(macro)
        item["kc_ids"] = by_macro.get(macro.get("macro_id"), [])
        out.append(item)
    return out


if __name__ == "__main__":
    main()
