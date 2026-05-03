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
from src.mermaid_exporter import export_macro_spine_mermaid, export_master_graph_mermaid, export_reasoning_threads_mermaid
from src.model_client import ModelConfig, OpenAICompatClient
from src.paper_parser import load_paper_text, load_paper_text_from_dir, split_into_sections
from src.progress import log, span
from src.reasoning_thread_builder import build_reasoning_threads


BASE_DIR = Path(__file__).resolve().parent
PAPER_PATH = BASE_DIR / "data" / "papers" / "demo_paper.md"
PAPER_DIR_PATH = BASE_DIR.parent / "util_example" / "output1"
GRAPH_PATH = BASE_DIR / "data" / "graphs" / "master_graph.json"
MASTER_MMD_PATH = BASE_DIR / "data" / "graphs" / "master_graph.mmd"
MACRO_SPINE_MMD_PATH = BASE_DIR / "data" / "graphs" / "macro_spine.mmd"
REASONING_THREADS_MMD_PATH = BASE_DIR / "data" / "graphs" / "reasoning_threads.mmd"
SECTIONS_PATH = BASE_DIR / "data" / "graphs" / "sections.json"
MACRO_SPINE_PATH = BASE_DIR / "data" / "graphs" / "macro_spine.json"
KC_CANDIDATES_PATH = BASE_DIR / "data" / "graphs" / "kc_candidates.json"
KC_BANK_PATH = BASE_DIR / "data" / "graphs" / "kc_bank.json"
BANK_EDGES_PATH = BASE_DIR / "data" / "graphs" / "kc_bank_reasoning_edges.json"
ACTIVE_KC_PATH = BASE_DIR / "data" / "graphs" / "active_kc.json"
REASONING_THREADS_PATH = BASE_DIR / "data" / "graphs" / "reasoning_threads.json"
BUILD_CHECKPOINT_PATH = BASE_DIR / "data" / "graphs" / "build_graph_checkpoint.json"


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

    allow_offline_fallback = _env_bool("ALLOW_OFFLINE_FALLBACK")
    resume = _env_bool("PAPERGRAPH_RESUME") or _env_bool("BUILD_GRAPH_RESUME")
    restart = _env_bool("PAPERGRAPH_RESTART") or _env_bool("BUILD_GRAPH_RESTART")
    checkpoint_path = Path(os.getenv("BUILD_GRAPH_CHECKPOINT_PATH", str(BUILD_CHECKPOINT_PATH)))
    client = OpenAICompatClient(
        ModelConfig(
            api_key=settings.api_key,
            base_url=settings.base_url,
            llm_model=settings.llm_model,
            embed_api_key=settings.embed_api_key,
            embed_base_url=settings.embed_base_url,
            embed_model=settings.embed_model,
        )
    )
    if not client.is_ready() and not allow_offline_fallback:
        raise RuntimeError("Online graph construction requires API_KEY, BASE_URL, and LLM_MODEL. Set ALLOW_OFFLINE_FALLBACK=true only for local debugging.")
    log("build graph resume settings", resume=resume, restart=restart, checkpoint=checkpoint_path)

    SECTIONS_PATH.parent.mkdir(parents=True, exist_ok=True)

    sections_payload = _load_resumable_json(SECTIONS_PATH, paper_id, resume, restart, "sections")
    if sections_payload:
        sections = sections_payload["sections"]
    else:
        with span("split paper into sections", paper_chars=len(paper_text)):
            sections = split_into_sections(paper_text)
        _write_json(SECTIONS_PATH, {"paper_id": paper_id, "sections": sections})
        _write_build_checkpoint(checkpoint_path, paper_id, "sections", resume=resume, restart=restart)
    log("sections ready", count=len(sections))

    macro_spine = _load_resumable_json(MACRO_SPINE_PATH, paper_id, resume, restart, "macro spine")
    if not macro_spine:
        with span("extract macro spine", sections=len(sections)):
            macro_spine = extract_macro_spine(paper_id, sections, client)
        _write_json(MACRO_SPINE_PATH, macro_spine)
        _write_build_checkpoint(checkpoint_path, paper_id, "macro_spine", resume=resume, restart=restart)
    log("macro spine ready", path=MACRO_SPINE_PATH)

    kc_candidates_payload = _load_resumable_json(KC_CANDIDATES_PATH, paper_id, resume, restart, "KC candidates")
    if kc_candidates_payload:
        kc_candidates = kc_candidates_payload["kc_candidates"]
    else:
        with span("extract KC candidates", sections=len(sections)):
            kc_candidates = extract_kc_candidates_by_sections(
                sections,
                client,
                allow_offline_fallback=allow_offline_fallback,
                macro_spine=macro_spine,
            )
        _write_json(KC_CANDIDATES_PATH, {"paper_id": paper_id, "kc_candidates": kc_candidates})
        _write_build_checkpoint(checkpoint_path, paper_id, "kc_candidates", resume=resume, restart=restart)
    log("KC candidates ready", count=len(kc_candidates))

    kc_bank = _load_resumable_json(KC_BANK_PATH, paper_id, resume, restart, "KC Bank")
    if not kc_bank:
        with span("build KC Bank", candidates=len(kc_candidates)):
            kc_bank = build_kc_bank(
                paper_id=paper_id,
                candidates=kc_candidates,
                macro_spine=macro_spine,
                client=client,
                allow_offline_fallback=allow_offline_fallback,
            )
        _write_json(KC_BANK_PATH, kc_bank)
        _write_build_checkpoint(checkpoint_path, paper_id, "kc_bank_base", resume=resume, restart=restart)

    bank_edges_payload = _load_resumable_json(BANK_EDGES_PATH, paper_id, resume, restart, "KC Bank reasoning edges")
    if bank_edges_payload:
        bank_edges = bank_edges_payload["reasoning_edges"]
    else:
        with span("build KC Bank reasoning edges", bank_kcs=len(kc_bank.get("kc_nodes", []))):
            bank_edges = build_reasoning_edges_for_kcs(
                kc_bank["kc_nodes"],
                _macro_nodes_with_bank_kcs(macro_spine, kc_bank),
                client,
                allow_offline_fallback=allow_offline_fallback,
            )
        _write_json(BANK_EDGES_PATH, {"paper_id": paper_id, "reasoning_edges": bank_edges})
        _write_build_checkpoint(checkpoint_path, paper_id, "kc_bank_reasoning_edges", resume=resume, restart=restart)

    if not _kc_bank_has_final_scores(kc_bank):
        finalize_kc_bank_scores(kc_bank, macro_spine, bank_edges)
        _write_json(KC_BANK_PATH, kc_bank)
        _write_build_checkpoint(checkpoint_path, paper_id, "kc_bank_scored", resume=resume, restart=restart)

    active_kc = _load_resumable_json(ACTIVE_KC_PATH, paper_id, resume, restart, "Active KC")
    if not active_kc:
        with span("select Active KCs", bank_kcs=len(kc_bank.get("kc_nodes", []))):
            active_kc = select_active_kcs(kc_bank, macro_spine)
        _write_json(KC_BANK_PATH, kc_bank)
        _write_json(ACTIVE_KC_PATH, active_kc)
        _write_build_checkpoint(checkpoint_path, paper_id, "active_kc", resume=resume, restart=restart)
    _sync_active_flags(kc_bank, active_kc)
    _write_json(KC_BANK_PATH, kc_bank)
    log("KC Bank ready", path=KC_BANK_PATH, kcs=len(kc_bank.get("kc_nodes", [])))
    log("Active KCs ready", path=ACTIVE_KC_PATH, active=len(active_kc.get("active_kc_ids", [])))

    graph = _load_resumable_json(GRAPH_PATH, paper_id, resume, restart, "master graph")
    if not graph:
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
        _write_build_checkpoint(checkpoint_path, paper_id, "master_graph_base", resume=resume, restart=restart)

    reasoning_threads = _load_resumable_json(REASONING_THREADS_PATH, paper_id, resume, restart, "reasoning threads")
    if not reasoning_threads:
        with span("build reasoning threads", paths=len(graph.get("reasoning_paths", []))):
            reasoning_threads = build_reasoning_threads(
                paper_id=paper_id,
                macro_spine=macro_spine,
                active_kc=active_kc,
                reasoning_edges=graph.get("reasoning_edges", []),
                reasoning_paths=graph.get("reasoning_paths", []),
                client=client,
            )
        _write_json(REASONING_THREADS_PATH, reasoning_threads)
        _write_build_checkpoint(checkpoint_path, paper_id, "reasoning_threads", resume=resume, restart=restart)
    graph["reasoning_threads_path"] = "data/graphs/reasoning_threads.json"
    graph["reasoning_threads"] = reasoning_threads.get("threads", [])
    _annotate_macro_bank_counts(graph, kc_bank, active_kc)
    log(
        "master graph ready",
        macros=len(graph.get("macro_nodes", [])),
        kcs=len(graph.get("kc_nodes", [])),
        edges=len(graph.get("reasoning_edges", [])),
        paths=len(graph.get("reasoning_paths", [])),
        threads=len(graph.get("reasoning_threads", [])),
    )

    _write_json(GRAPH_PATH, graph)
    MASTER_MMD_PATH.write_text(export_master_graph_mermaid(graph), encoding="utf-8")
    MACRO_SPINE_MMD_PATH.write_text(export_macro_spine_mermaid(macro_spine), encoding="utf-8")
    REASONING_THREADS_MMD_PATH.write_text(export_reasoning_threads_mermaid(reasoning_threads), encoding="utf-8")
    _write_build_checkpoint(checkpoint_path, paper_id, "completed", resume=resume, restart=restart)
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


def _annotate_macro_bank_counts(graph: dict, kc_bank: dict, active_kc: dict) -> None:
    bank_counts: dict[str, int] = {}
    for kc in kc_bank.get("kc_nodes", []):
        macro_id = kc.get("macro_id")
        if macro_id:
            bank_counts[macro_id] = bank_counts.get(macro_id, 0) + 1
    active_counts = {
        macro_id: len(kc_ids)
        for macro_id, kc_ids in active_kc.get("macro_active_kcs", {}).items()
    }
    for macro in graph.get("macro_nodes", []):
        macro_id = macro.get("macro_id")
        macro["bank_kc_count"] = bank_counts.get(macro_id, 0)
        macro["active_kc_count"] = active_counts.get(macro_id, len(macro.get("kc_ids", [])))


def _env_bool(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


def _load_resumable_json(
    path: Path,
    paper_id: str,
    resume: bool,
    restart: bool,
    label: str,
) -> dict | None:
    if restart or not resume or not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise RuntimeError(f"Cannot resume {label}; failed to read {path}: {type(exc).__name__}: {exc}") from exc
    if payload.get("paper_id") != paper_id:
        log("resume artifact ignored due to paper_id mismatch", label=label, path=path)
        return None
    log("resume artifact loaded", label=label, path=path)
    return payload


def _kc_bank_has_final_scores(kc_bank: dict) -> bool:
    for node in kc_bank.get("kc_nodes", []):
        scores = node.get("importance_scores") or node.get("scores") or {}
        if "final_importance_score" not in scores:
            return False
    return bool(kc_bank.get("kc_nodes"))


def _sync_active_flags(kc_bank: dict, active_kc: dict) -> None:
    active_ids = set(active_kc.get("active_kc_ids", []))
    for node in kc_bank.get("kc_nodes", []):
        flags = node.setdefault("flags", {})
        active = node.get("kc_id") in active_ids
        flags["active_for_question_generation"] = active
        flags["active_for_core_metrics"] = active
        flags.setdefault("usable_for_claim_verification", True)


def _write_build_checkpoint(
    path: Path,
    paper_id: str,
    stage: str,
    resume: bool,
    restart: bool,
) -> None:
    _write_json(
        path,
        {
            "paper_id": paper_id,
            "last_completed_stage": stage,
            "resume_requested": resume,
            "restart_requested": restart,
            "artifacts": {
                "sections": str(SECTIONS_PATH),
                "macro_spine": str(MACRO_SPINE_PATH),
                "kc_candidates": str(KC_CANDIDATES_PATH),
                "kc_bank": str(KC_BANK_PATH),
                "kc_bank_reasoning_edges": str(BANK_EDGES_PATH),
                "active_kc": str(ACTIVE_KC_PATH),
                "reasoning_threads": str(REASONING_THREADS_PATH),
                "master_graph": str(GRAPH_PATH),
                "master_mermaid": str(MASTER_MMD_PATH),
                "macro_spine_mermaid": str(MACRO_SPINE_MMD_PATH),
                "reasoning_threads_mermaid": str(REASONING_THREADS_MMD_PATH),
            },
        },
    )


if __name__ == "__main__":
    main()
