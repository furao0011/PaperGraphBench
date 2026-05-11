import json
import os
from pathlib import Path

from src.active_kc_selector import select_active_kcs
from src.config import load_settings
from src.edge_candidate_builder import (
    build_adjacent_macro_edge_candidates,
    build_macro_edge_candidates,
    build_thread_candidate_edges,
    build_unit_edge_candidates,
)
from src.edge_coverage_report import attach_reasoning_path_coverage, build_edge_coverage_report
from src.edge_verifier import verify_edge_candidates
from src.extraction_unit_builder import decompose_extraction_units
from src.graph_builder import build_master_graph
from src.graph_builder import build_reasoning_edges_for_kcs
from src.kc_bank_builder import build_kc_bank
from src.kc_bank_builder import finalize_kc_bank_scores
from src.kc_bank_builder import append_kc_candidates_to_bank
from src.kc_extractor import extract_kc_candidates_by_sections
from src.macro_extractor import extract_macro_spine
from src.mermaid_exporter import export_macro_spine_mermaid, export_master_graph_mermaid, export_reasoning_threads_mermaid
from src.model_client import ModelConfig, OpenAICompatClient
from src.multimodal_asset_grouper import group_multimodal_assets
from src.multimodal_asset_normalizer import normalize_multimodal_assets
from src.multimodal_explainer import build_vision_client, explain_multimodal_assets
from src.multimodal_html_group_analyzer import analyze_multimodal_html_groups
from src.multimodal_kc_extractor import extract_multimodal_kc_candidates
from src.multimodal_unit_builder import augment_extraction_units_with_multimodal_units
from src.paper_block_aligner import align_blocks_to_sections
from src.paper_block_parser import load_paper_bundle_from_dir, load_paper_bundle_from_file
from src.paper_parser import split_into_sections
from src.progress import log, span
from src.reasoning_thread_builder import build_reasoning_threads
from src.unit_kc_extractor import extract_kc_candidates_by_units


BASE_DIR = Path(__file__).resolve().parent
MASTER_GRAPH_BUILDER_VERSION = "v4_text_active_policy"
EDGE_ARTIFACT_SIGNATURE_VERSION = "v1_multimodal_virtual_units"
PAPER_PATH = BASE_DIR / "data" / "papers" / "demo_paper.md"
PAPER_DIR_PATH = BASE_DIR.parent / "util_example" / "output1"
GRAPH_PATH = BASE_DIR / "data" / "graphs" / "master_graph.json"
MASTER_MMD_PATH = BASE_DIR / "data" / "graphs" / "master_graph.mmd"
MACRO_SPINE_MMD_PATH = BASE_DIR / "data" / "graphs" / "macro_spine.mmd"
REASONING_THREADS_MMD_PATH = BASE_DIR / "data" / "graphs" / "reasoning_threads.mmd"
SECTIONS_PATH = BASE_DIR / "data" / "graphs" / "sections.json"
MACRO_SPINE_PATH = BASE_DIR / "data" / "graphs" / "macro_spine.json"
EXTRACTION_UNITS_PATH = BASE_DIR / "data" / "graphs" / "extraction_units.json"
KC_CANDIDATES_PATH = BASE_DIR / "data" / "graphs" / "kc_candidates.json"
KC_BANK_PATH = BASE_DIR / "data" / "graphs" / "kc_bank.json"
EDGE_CANDIDATE_UNITS_PATH = BASE_DIR / "data" / "graphs" / "edge_candidate_units.json"
EDGE_CANDIDATE_MACRO_PATH = BASE_DIR / "data" / "graphs" / "edge_candidate_macro.json"
EDGE_CANDIDATE_CROSS_MACRO_PATH = BASE_DIR / "data" / "graphs" / "edge_candidate_cross_macro.json"
EDGE_CANDIDATE_THREAD_PATH = BASE_DIR / "data" / "graphs" / "edge_candidate_thread.json"
VERIFIED_EDGES_PATH = BASE_DIR / "data" / "graphs" / "verified_edges.json"
EDGE_VERIFICATION_LOG_PATH = BASE_DIR / "data" / "graphs" / "edge_verification_log.json"
EDGE_COVERAGE_REPORT_PATH = BASE_DIR / "data" / "graphs" / "edge_coverage_report.json"
BANK_EDGES_PATH = BASE_DIR / "data" / "graphs" / "kc_bank_reasoning_edges.json"
ACTIVE_KC_PATH = BASE_DIR / "data" / "graphs" / "active_kc.json"
REASONING_THREADS_PATH = BASE_DIR / "data" / "graphs" / "reasoning_threads.json"
BUILD_CHECKPOINT_PATH = BASE_DIR / "data" / "graphs" / "build_graph_checkpoint.json"
MULTIMODAL_DIR = BASE_DIR / "data" / "multimodal"
PAPER_BLOCKS_PATH = MULTIMODAL_DIR / "paper_blocks.json"
MULTIMODAL_ASSET_GROUPS_PATH = MULTIMODAL_DIR / "multimodal_asset_groups.json"
MULTIMODAL_ASSETS_PATH = MULTIMODAL_DIR / "multimodal_assets.json"
MULTIMODAL_ASSET_EXPLANATIONS_PATH = MULTIMODAL_DIR / "multimodal_asset_explanations.json"
MULTIMODAL_KC_CANDIDATES_PATH = MULTIMODAL_DIR / "multimodal_kc_candidates.json"


def main() -> None:
    project_root = BASE_DIR.parent
    settings = load_settings(project_root)
    log("build_graph configuration loaded", base_dir=BASE_DIR)

    input_dir = Path(os.getenv("PAPER_INPUT_DIR", str(PAPER_DIR_PATH)))
    input_file = Path(os.getenv("PAPER_INPUT_FILE", str(PAPER_PATH)))

    if input_dir.exists():
        log("loading paper from directory", input_dir=input_dir)
        with span("load paper directory"):
            paper_bundle = load_paper_bundle_from_dir(input_dir)
            paper_text = paper_bundle["clean_text"]
        paper_id = input_dir.name
        paper_text_path = str(input_dir)
    elif input_file.exists():
        log("loading paper from file", input_file=input_file)
        with span("load paper file"):
            paper_bundle = load_paper_bundle_from_file(input_file)
            paper_text = paper_bundle["clean_text"]
        paper_id = input_file.stem
        paper_text_path = str(input_file)
    else:
        raise FileNotFoundError(
            f"No valid input found. Checked directory: {input_dir}, file: {input_file}"
        )

    allow_offline_fallback = _env_bool("ALLOW_OFFLINE_FALLBACK")
    multimodal_enabled = _env_bool("MULTIMODAL_ENABLED")
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

    paper_blocks_payload = None
    if multimodal_enabled:
        with span("align paper blocks to sections", blocks=len(paper_bundle.get("blocks", []))):
            paper_blocks_payload = align_blocks_to_sections(
                paper_id=paper_id,
                blocks=paper_bundle.get("blocks", []),
                sections=sections,
            )
        _write_json(PAPER_BLOCKS_PATH, paper_blocks_payload)
        diagnostics = paper_blocks_payload.get("diagnostics", {})
        if diagnostics.get("image_path_missing_count", 0):
            raise RuntimeError(f"Multimodal image path diagnostics failed: {diagnostics}")
        log(
            "multimodal paper blocks ready",
            path=PAPER_BLOCKS_PATH,
            blocks=paper_blocks_payload.get("summary", {}).get("block_count", 0),
            diagnostics=diagnostics,
        )
    else:
        log("multimodal block parsing disabled", env="MULTIMODAL_ENABLED")

    macro_spine = _load_resumable_json(MACRO_SPINE_PATH, paper_id, resume, restart, "macro spine")
    if not macro_spine:
        with span("extract macro spine", sections=len(sections)):
            macro_spine = extract_macro_spine(paper_id, sections, client)
        _write_json(MACRO_SPINE_PATH, macro_spine)
        _write_build_checkpoint(checkpoint_path, paper_id, "macro_spine", resume=resume, restart=restart)
    log("macro spine ready", path=MACRO_SPINE_PATH)

    multimodal_asset_groups_payload = None
    multimodal_assets_payload = None
    multimodal_asset_explanations_payload = None
    if multimodal_enabled:
        if paper_blocks_payload is None:
            raise RuntimeError("MULTIMODAL_ENABLED=true requires aligned paper blocks.")
        with span("group multimodal assets", blocks=len(paper_blocks_payload.get("blocks", []))):
            multimodal_asset_groups_payload = group_multimodal_assets(
                paper_id=paper_id,
                blocks=paper_blocks_payload.get("blocks", []),
                sections=sections,
            )
        with span("analyze multimodal HTML groups", groups=len(multimodal_asset_groups_payload.get("asset_groups", []))):
            multimodal_asset_groups_payload = analyze_multimodal_html_groups(
                paper_id=paper_id,
                asset_groups=multimodal_asset_groups_payload,
                client=client,
            )
        _write_json(MULTIMODAL_ASSET_GROUPS_PATH, multimodal_asset_groups_payload)
        with span("normalize multimodal assets", groups=len(multimodal_asset_groups_payload.get("asset_groups", []))):
            multimodal_assets_payload = normalize_multimodal_assets(
                paper_id=paper_id,
                asset_groups=multimodal_asset_groups_payload,
                macro_spine=macro_spine,
            )
        _write_json(MULTIMODAL_ASSETS_PATH, multimodal_assets_payload)
        with span("explain multimodal assets", assets=len(multimodal_assets_payload.get("assets", []))):
            vision_client = build_vision_client(
                embed_api_key=settings.embed_api_key,
                vision_api_key=settings.vision_api_key,
                vision_base_url=settings.vision_base_url,
                vision_model=settings.vision_model,
            )
            multimodal_asset_explanations_payload = explain_multimodal_assets(
                paper_id=paper_id,
                assets_payload=multimodal_assets_payload,
                text_client=client,
                vision_client=vision_client,
            )
        _write_json(MULTIMODAL_ASSET_EXPLANATIONS_PATH, multimodal_asset_explanations_payload)
        log(
            "multimodal assets ready",
            groups=multimodal_asset_groups_payload.get("summary", {}).get("asset_group_count", 0),
            assets=multimodal_assets_payload.get("summary", {}).get("asset_count", 0),
            explanations=multimodal_asset_explanations_payload.get("summary", {}).get("asset_explanation_count", 0),
            macro_unresolved=multimodal_assets_payload.get("summary", {}).get("macro_unresolved_count", 0),
        )

    extraction_units_enabled = _env_bool("EXTRACTION_UNIT_ENABLED", True)
    extraction_units = None
    if extraction_units_enabled:
        extraction_units = _load_resumable_json(
            EXTRACTION_UNITS_PATH,
            paper_id,
            resume,
            restart,
            "Extraction Units",
        )
        if not extraction_units:
            with span("decompose extraction units", sections=len(sections)):
                extraction_units = decompose_extraction_units(paper_id, sections, client)
            _write_json(EXTRACTION_UNITS_PATH, extraction_units)
            _write_build_checkpoint(checkpoint_path, paper_id, "extraction_units", resume=resume, restart=restart)
        log("Extraction Units ready", path=EXTRACTION_UNITS_PATH, units=len(extraction_units.get("units", [])))
    else:
        log("Extraction Unit decomposition disabled", env="EXTRACTION_UNIT_ENABLED")

    kc_extraction_source = os.getenv("KC_EXTRACTION_SOURCE", "unit").strip().lower() or "unit"
    if kc_extraction_source not in {"unit", "section"}:
        raise ValueError("KC_EXTRACTION_SOURCE must be 'unit' or 'section'.")
    if kc_extraction_source == "unit" and not extraction_units:
        raise RuntimeError("KC_EXTRACTION_SOURCE=unit requires EXTRACTION_UNIT_ENABLED=true.")
    multimodal_kc_enabled = multimodal_enabled and _env_bool("MULTIMODAL_KC_ENABLED", True)
    requested_kc_extraction_source = (
        f"{kc_extraction_source}+multimodal" if multimodal_kc_enabled else kc_extraction_source
    )

    kc_candidates_payload = _load_resumable_json(KC_CANDIDATES_PATH, paper_id, resume, restart, "KC candidates")
    if kc_candidates_payload and kc_candidates_payload.get("extraction_source") not in {
        kc_extraction_source,
        requested_kc_extraction_source,
    }:
        log(
            "resume artifact ignored due to KC extraction source mismatch",
            path=KC_CANDIDATES_PATH,
            artifact_source=kc_candidates_payload.get("extraction_source"),
            requested_source=requested_kc_extraction_source,
        )
        kc_candidates_payload = None
    if kc_candidates_payload:
        kc_candidates = kc_candidates_payload["kc_candidates"]
    else:
        if kc_extraction_source == "unit":
            with span("extract KC candidates from units", units=len(extraction_units.get("units", []))):
                kc_candidates_payload = extract_kc_candidates_by_units(
                    extraction_units,
                    macro_spine,
                    client,
                    return_metadata=True,
                )
                kc_candidates = kc_candidates_payload["kc_candidates"]
        else:
            with span("extract KC candidates from sections", sections=len(sections)):
                kc_candidates = extract_kc_candidates_by_sections(
                    sections,
                    client,
                    allow_offline_fallback=allow_offline_fallback,
                    macro_spine=macro_spine,
                )
            kc_candidates_payload = {
                "paper_id": paper_id,
                "extraction_source": kc_extraction_source,
                "kc_candidates": kc_candidates,
            }
        kc_candidates_payload["paper_id"] = paper_id
        kc_candidates_payload["extraction_source"] = kc_extraction_source
        _write_json(KC_CANDIDATES_PATH, kc_candidates_payload)
        _write_build_checkpoint(checkpoint_path, paper_id, "kc_candidates", resume=resume, restart=restart)

    if multimodal_kc_enabled:
        already_has_multimodal = bool(kc_candidates_payload.get("multimodal_kc_enabled"))
        if not already_has_multimodal:
            if multimodal_asset_explanations_payload is None:
                if not MULTIMODAL_ASSET_EXPLANATIONS_PATH.exists():
                    raise FileNotFoundError(
                        f"MULTIMODAL_KC_ENABLED=true requires asset explanations: {MULTIMODAL_ASSET_EXPLANATIONS_PATH}"
                    )
                multimodal_asset_explanations_payload = json.loads(
                    MULTIMODAL_ASSET_EXPLANATIONS_PATH.read_text(encoding="utf-8")
                )
            with span(
                "extract multimodal KC candidates",
                assets=multimodal_asset_explanations_payload.get("summary", {}).get("asset_explanation_count", 0),
            ):
                multimodal_kc_candidates_payload = extract_multimodal_kc_candidates(
                    paper_id=paper_id,
                    asset_explanations_payload=multimodal_asset_explanations_payload,
                    client=client,
                )
            _write_json(MULTIMODAL_KC_CANDIDATES_PATH, multimodal_kc_candidates_payload)
            multimodal_candidates = multimodal_kc_candidates_payload["kc_candidates"]
            kc_candidates = kc_candidates + multimodal_candidates
            kc_candidates_payload["kc_candidates"] = kc_candidates
            kc_candidates_payload["multimodal_kc_enabled"] = True
            kc_candidates_payload["multimodal_kc_candidates_path"] = "data/multimodal/multimodal_kc_candidates.json"
            kc_candidates_payload["multimodal_candidate_count"] = len(multimodal_candidates)
            kc_candidates_payload["extraction_source"] = f"{kc_extraction_source}+multimodal"
            _write_json(KC_CANDIDATES_PATH, kc_candidates_payload)
            _write_build_checkpoint(checkpoint_path, paper_id, "multimodal_kc_candidates", resume=resume, restart=restart)
        log(
            "multimodal KC candidates ready",
            enabled=True,
            multimodal_count=kc_candidates_payload.get("multimodal_candidate_count", 0),
        )
    elif kc_candidates_payload.get("multimodal_kc_enabled"):
        kc_candidates = [
            candidate
            for candidate in kc_candidates
            if not candidate.get("modality", {}).get("is_multimodal")
        ]
        kc_candidates_payload["kc_candidates"] = kc_candidates
        kc_candidates_payload["multimodal_kc_enabled"] = False
        kc_candidates_payload["multimodal_candidate_count"] = 0
        kc_candidates_payload["extraction_source"] = kc_extraction_source
        _write_json(KC_CANDIDATES_PATH, kc_candidates_payload)
    log("KC candidates ready", source=kc_extraction_source, count=len(kc_candidates))
    kc_candidates_signature = _kc_candidates_signature(kc_candidates)

    kc_bank = _load_resumable_json(KC_BANK_PATH, paper_id, resume, restart, "KC Bank")
    if kc_bank and not _kc_bank_matches_requested_source(kc_bank, kc_extraction_source):
        log(
            "resume artifact ignored due to KC Bank source mismatch",
            path=KC_BANK_PATH,
            requested_source=kc_extraction_source,
        )
        kc_bank = None
    if kc_bank and kc_bank.get("kc_candidates_signature") != kc_candidates_signature:
        log(
            "resume artifact ignored due to KC candidate signature mismatch",
            path=KC_BANK_PATH,
        )
        kc_bank = None
    if kc_bank and multimodal_kc_enabled and not _kc_bank_has_multimodal_kcs(kc_bank):
        multimodal_candidates_for_bank = [
            candidate
            for candidate in kc_candidates
            if candidate.get("modality", {}).get("is_multimodal")
        ]
        with span("append multimodal KCs to KC Bank", candidates=len(multimodal_candidates_for_bank)):
            kc_bank = append_kc_candidates_to_bank(
                kc_bank=kc_bank,
                candidates=multimodal_candidates_for_bank,
                macro_spine=macro_spine,
                client=client,
                allow_offline_fallback=allow_offline_fallback,
            )
        kc_bank["kc_candidates_signature"] = kc_candidates_signature
        _write_json(KC_BANK_PATH, kc_bank)
        _write_build_checkpoint(checkpoint_path, paper_id, "kc_bank_multimodal_extension", resume=resume, restart=restart)
    if not kc_bank:
        with span("build KC Bank", candidates=len(kc_candidates)):
            kc_bank = build_kc_bank(
                paper_id=paper_id,
                candidates=kc_candidates,
                macro_spine=macro_spine,
                client=client,
                allow_offline_fallback=allow_offline_fallback,
            )
        kc_bank["kc_candidates_signature"] = kc_candidates_signature
        _write_json(KC_BANK_PATH, kc_bank)
        _write_build_checkpoint(checkpoint_path, paper_id, "kc_bank_base", resume=resume, restart=restart)

    edge_extraction_units = augment_extraction_units_with_multimodal_units(extraction_units, kc_bank)
    kc_bank_signature = _kc_bank_signature(kc_bank)
    log(
        "edge extraction units ready",
        base_units=len(extraction_units.get("units", [])),
        units=len(edge_extraction_units.get("units", [])),
        multimodal_units=edge_extraction_units.get("metadata", {}).get("multimodal_virtual_unit_count", 0),
    )

    edge_candidates_for_base_verification: list[dict] = []
    requested_base_edge_layers: list[str] = []
    edge_layer_enabled = (
        _env_bool("EDGE_UNIT_ENABLED", True)
        or _env_bool("EDGE_MACRO_INTERNAL_ENABLED", True)
        or _env_bool("EDGE_ADJACENT_MACRO_ENABLED", True)
        or _env_bool("EDGE_THREAD_CANDIDATE_ENABLED", True)
    )
    graph_edge_source = os.getenv("GRAPH_REASONING_EDGE_SOURCE", "verified").strip().lower() or "verified"
    if graph_edge_source not in {"verified", "legacy"}:
        raise ValueError("GRAPH_REASONING_EDGE_SOURCE must be 'verified' or 'legacy'.")
    if graph_edge_source == "verified" and not edge_layer_enabled:
        raise RuntimeError("GRAPH_REASONING_EDGE_SOURCE=verified requires at least one EDGE_* construction layer.")
    if edge_layer_enabled and not extraction_units:
        raise RuntimeError("v2 edge construction requires extraction_units.json.")

    if _env_bool("EDGE_UNIT_ENABLED", True):
        requested_base_edge_layers.append("unit")
        unit_edge_candidates_payload = _load_resumable_json(
            EDGE_CANDIDATE_UNITS_PATH,
            paper_id,
            resume,
            restart,
            "Unit edge candidates",
        )
        if not _payload_has_kc_bank_signature(
            unit_edge_candidates_payload,
            kc_bank_signature,
            "Unit edge candidates",
            EDGE_CANDIDATE_UNITS_PATH,
        ):
            unit_edge_candidates_payload = None
        if unit_edge_candidates_payload:
            unit_edge_candidates = unit_edge_candidates_payload["edge_candidates"]
        else:
            with span("build Unit edge candidates", bank_kcs=len(kc_bank.get("kc_nodes", []))):
                unit_edge_candidates_payload = build_unit_edge_candidates(
                    paper_id=paper_id,
                    kc_bank=kc_bank,
                    extraction_units=edge_extraction_units,
                    client=client,
                )
                unit_edge_candidates = unit_edge_candidates_payload["edge_candidates"]
                unit_edge_candidates_payload["kc_bank_signature"] = kc_bank_signature
            _write_json(EDGE_CANDIDATE_UNITS_PATH, unit_edge_candidates_payload)
            _write_build_checkpoint(checkpoint_path, paper_id, "edge_candidate_units", resume=resume, restart=restart)
        edge_candidates_for_base_verification.extend(unit_edge_candidates)
    else:
        log("Unit edge construction disabled", env="EDGE_UNIT_ENABLED")

    if _env_bool("EDGE_MACRO_INTERNAL_ENABLED", True):
        requested_base_edge_layers.append("macro")
        macro_edge_candidates_payload = _load_resumable_json(
            EDGE_CANDIDATE_MACRO_PATH,
            paper_id,
            resume,
            restart,
            "Macro edge candidates",
        )
        if not _payload_has_kc_bank_signature(
            macro_edge_candidates_payload,
            kc_bank_signature,
            "Macro edge candidates",
            EDGE_CANDIDATE_MACRO_PATH,
        ):
            macro_edge_candidates_payload = None
        if macro_edge_candidates_payload:
            macro_edge_candidates = macro_edge_candidates_payload["edge_candidates"]
        else:
            with span("build Macro edge candidates", bank_kcs=len(kc_bank.get("kc_nodes", []))):
                macro_edge_candidates_payload = build_macro_edge_candidates(
                    paper_id=paper_id,
                    kc_bank=kc_bank,
                    macro_spine=macro_spine,
                    extraction_units=edge_extraction_units,
                    client=client,
                )
                macro_edge_candidates = macro_edge_candidates_payload["edge_candidates"]
                macro_edge_candidates_payload["kc_bank_signature"] = kc_bank_signature
            _write_json(EDGE_CANDIDATE_MACRO_PATH, macro_edge_candidates_payload)
            _write_build_checkpoint(checkpoint_path, paper_id, "edge_candidate_macro", resume=resume, restart=restart)
        edge_candidates_for_base_verification.extend(macro_edge_candidates)
    else:
        log("Macro internal edge construction disabled", env="EDGE_MACRO_INTERNAL_ENABLED")

    if _env_bool("EDGE_ADJACENT_MACRO_ENABLED", True):
        requested_base_edge_layers.append("adjacent_macro")
        adjacent_edge_candidates_payload = _load_resumable_json(
            EDGE_CANDIDATE_CROSS_MACRO_PATH,
            paper_id,
            resume,
            restart,
            "Adjacent Macro edge candidates",
        )
        if not _payload_has_kc_bank_signature(
            adjacent_edge_candidates_payload,
            kc_bank_signature,
            "Adjacent Macro edge candidates",
            EDGE_CANDIDATE_CROSS_MACRO_PATH,
        ):
            adjacent_edge_candidates_payload = None
        if adjacent_edge_candidates_payload:
            adjacent_edge_candidates = adjacent_edge_candidates_payload["edge_candidates"]
        else:
            with span("build Adjacent Macro edge candidates", bank_kcs=len(kc_bank.get("kc_nodes", []))):
                adjacent_edge_candidates_payload = build_adjacent_macro_edge_candidates(
                    paper_id=paper_id,
                    kc_bank=kc_bank,
                    macro_spine=macro_spine,
                    extraction_units=edge_extraction_units,
                    client=client,
                )
                adjacent_edge_candidates = adjacent_edge_candidates_payload["edge_candidates"]
                adjacent_edge_candidates_payload["kc_bank_signature"] = kc_bank_signature
            _write_json(EDGE_CANDIDATE_CROSS_MACRO_PATH, adjacent_edge_candidates_payload)
            _write_build_checkpoint(checkpoint_path, paper_id, "edge_candidate_cross_macro", resume=resume, restart=restart)
        edge_candidates_for_base_verification.extend(adjacent_edge_candidates)
    else:
        log("Adjacent Macro edge construction disabled", env="EDGE_ADJACENT_MACRO_ENABLED")

    thread_edges_enabled = _env_bool("EDGE_THREAD_CANDIDATE_ENABLED", True)
    requested_edge_layers = sorted(set(requested_base_edge_layers + (["thread"] if thread_edges_enabled else [])))
    verified_edges_payload = None
    if requested_edge_layers:
        verified_edges_payload = _load_resumable_json(
            VERIFIED_EDGES_PATH,
            paper_id,
            resume,
            restart,
            "verified edges",
        )
        if not _payload_has_kc_bank_signature(
            verified_edges_payload,
            kc_bank_signature,
            "verified edges",
            VERIFIED_EDGES_PATH,
        ):
            verified_edges_payload = None
        if verified_edges_payload and verified_edges_payload.get("source_layers") != requested_edge_layers:
            log(
                "resume artifact ignored due to verified edge layer mismatch",
                path=VERIFIED_EDGES_PATH,
                artifact_layers=verified_edges_payload.get("source_layers"),
                requested_layers=requested_edge_layers,
            )
            verified_edges_payload = None
        if not verified_edges_payload:
            with span("verify base edge candidates", candidates=len(edge_candidates_for_base_verification)):
                base_verification_payload = verify_edge_candidates(
                    paper_id=paper_id,
                    edge_candidates=edge_candidates_for_base_verification,
                    kc_bank=kc_bank,
                    extraction_units=edge_extraction_units,
                    client=client,
                )
            verified_edges = base_verification_payload["verified_edges"]
            verification_log = base_verification_payload["verification_log"]
            summary = dict(base_verification_payload["summary"])

            if thread_edges_enabled:
                thread_edge_candidates_payload = _load_resumable_json(
                    EDGE_CANDIDATE_THREAD_PATH,
                    paper_id,
                    resume,
                    restart,
                    "Thread edge candidates",
                )
                if not _payload_has_kc_bank_signature(
                    thread_edge_candidates_payload,
                    kc_bank_signature,
                    "Thread edge candidates",
                    EDGE_CANDIDATE_THREAD_PATH,
                ):
                    thread_edge_candidates_payload = None
                if thread_edge_candidates_payload and thread_edge_candidates_payload.get(
                    "verified_edge_signature"
                ) != _verified_edge_signature(verified_edges):
                    log(
                        "resume artifact ignored due to verified edge signature mismatch",
                        label="Thread edge candidates",
                        path=EDGE_CANDIDATE_THREAD_PATH,
                    )
                    thread_edge_candidates_payload = None
                if thread_edge_candidates_payload:
                    thread_edge_candidates = thread_edge_candidates_payload["edge_candidates"]
                else:
                    with span("build Thread candidate edges", verified_edges=len(verified_edges)):
                        thread_edge_candidates_payload = build_thread_candidate_edges(
                            paper_id=paper_id,
                            kc_bank=kc_bank,
                            macro_spine=macro_spine,
                            extraction_units=edge_extraction_units,
                            verified_edges=verified_edges,
                            client=client,
                        )
                        thread_edge_candidates = thread_edge_candidates_payload["edge_candidates"]
                        thread_edge_candidates_payload["kc_bank_signature"] = kc_bank_signature
                        thread_edge_candidates_payload["verified_edge_signature"] = _verified_edge_signature(verified_edges)
                    _write_json(EDGE_CANDIDATE_THREAD_PATH, thread_edge_candidates_payload)
                    _write_build_checkpoint(checkpoint_path, paper_id, "edge_candidate_thread", resume=resume, restart=restart)

                with span("verify Thread edge candidates", candidates=len(thread_edge_candidates)):
                    thread_verification_payload = verify_edge_candidates(
                        paper_id=paper_id,
                        edge_candidates=thread_edge_candidates,
                        kc_bank=kc_bank,
                        extraction_units=edge_extraction_units,
                        client=client,
                    )
                verified_edges = _renumber_verified_edges(
                    verified_edges + thread_verification_payload["verified_edges"]
                )
                verification_log = verification_log + thread_verification_payload["verification_log"]
                summary = _merge_edge_verification_summaries(
                    summary,
                    thread_verification_payload["summary"],
                )
            else:
                log("Thread candidate edge construction disabled", env="EDGE_THREAD_CANDIDATE_ENABLED")

            verified_edges_payload = {
                "paper_id": paper_id,
                "source_layers": requested_edge_layers,
                "kc_bank_signature": kc_bank_signature,
                "verified_edges": verified_edges,
                "summary": summary,
            }
            _write_json(VERIFIED_EDGES_PATH, verified_edges_payload)
            _write_json(
                EDGE_VERIFICATION_LOG_PATH,
                {
                    "paper_id": paper_id,
                    "source_layers": requested_edge_layers,
                    "kc_bank_signature": kc_bank_signature,
                    "verification_log": verification_log,
                    "summary": summary,
                },
            )
            _write_build_checkpoint(checkpoint_path, paper_id, "verified_edges", resume=resume, restart=restart)
        log(
            "verified edges ready",
            layers=",".join(requested_edge_layers),
            candidates=verified_edges_payload.get("summary", {}).get("candidate_count", 0),
            verified=len(verified_edges_payload.get("verified_edges", [])),
        )

    if graph_edge_source == "verified":
        if not verified_edges_payload:
            raise RuntimeError("GRAPH_REASONING_EDGE_SOURCE=verified requires verified_edges.json.")
        graph_reasoning_edges = verified_edges_payload.get("verified_edges", [])
        if not graph_reasoning_edges:
            raise RuntimeError("verified_edges.json contains no verified edges; cannot build v2 Master Graph.")
        coverage_report = _load_resumable_json(
            EDGE_COVERAGE_REPORT_PATH,
            paper_id,
            resume,
            restart,
            "edge coverage report",
        )
        if coverage_report and coverage_report.get("verified_edge_count") != len(graph_reasoning_edges):
            log(
                "resume artifact ignored due to edge coverage count mismatch",
                path=EDGE_COVERAGE_REPORT_PATH,
                artifact_edges=coverage_report.get("verified_edge_count"),
                verified_edges=len(graph_reasoning_edges),
            )
            coverage_report = None
        if not coverage_report:
            with span("build edge coverage report", verified_edges=len(graph_reasoning_edges)):
                coverage_report = build_edge_coverage_report(
                    paper_id=paper_id,
                    macro_spine=macro_spine,
                    kc_bank=kc_bank,
                    verified_edges=graph_reasoning_edges,
                )
            _write_json(EDGE_COVERAGE_REPORT_PATH, coverage_report)
            _write_build_checkpoint(checkpoint_path, paper_id, "edge_coverage_report", resume=resume, restart=restart)
        log(
            "edge coverage report ready",
            path=EDGE_COVERAGE_REPORT_PATH,
            empty_macro_pairs=len(coverage_report.get("empty_macro_pairs", [])),
            isolated_kcs=coverage_report.get("kc_coverage", {}).get("isolated_kc_count", 0),
        )
    else:
        coverage_report = None
        bank_edges_payload = _load_resumable_json(BANK_EDGES_PATH, paper_id, resume, restart, "KC Bank reasoning edges")
        if not _payload_has_kc_bank_signature(
            bank_edges_payload,
            kc_bank_signature,
            "KC Bank reasoning edges",
            BANK_EDGES_PATH,
        ):
            bank_edges_payload = None
        if bank_edges_payload:
            graph_reasoning_edges = bank_edges_payload["reasoning_edges"]
        else:
            with span("build KC Bank reasoning edges", bank_kcs=len(kc_bank.get("kc_nodes", []))):
                graph_reasoning_edges = build_reasoning_edges_for_kcs(
                    kc_bank["kc_nodes"],
                    _macro_nodes_with_bank_kcs(macro_spine, kc_bank),
                    client,
                    allow_offline_fallback=allow_offline_fallback,
                )
            _write_json(
                BANK_EDGES_PATH,
                {
                    "paper_id": paper_id,
                    "kc_bank_signature": kc_bank_signature,
                    "reasoning_edges": graph_reasoning_edges,
                },
            )
            _write_build_checkpoint(checkpoint_path, paper_id, "kc_bank_reasoning_edges", resume=resume, restart=restart)

    score_signature = _score_signature(graph_edge_source, graph_reasoning_edges)
    if not _kc_bank_has_final_scores(kc_bank) or kc_bank.get("score_metadata", {}).get("graph_signature") != score_signature:
        finalize_kc_bank_scores(kc_bank, macro_spine, graph_reasoning_edges)
        kc_bank["score_metadata"] = {
            "reasoning_edge_source": graph_edge_source,
            "graph_signature": score_signature,
            "reasoning_edge_count": len(graph_reasoning_edges),
            "edge_coverage_report_path": "data/graphs/edge_coverage_report.json" if coverage_report else None,
            "kc_bank_signature": kc_bank_signature,
        }
        if isinstance(kc_bank.get("extension_metadata"), dict):
            kc_bank["extension_metadata"]["final_scores_stale"] = False
            kc_bank["extension_metadata"]["stale_reason"] = None
        _write_json(KC_BANK_PATH, kc_bank)
        _write_build_checkpoint(checkpoint_path, paper_id, "kc_bank_scored", resume=resume, restart=restart)

    active_kc = _load_resumable_json(ACTIVE_KC_PATH, paper_id, resume, restart, "Active KC")
    if active_kc and active_kc.get("source_score_signature") != score_signature:
        log(
            "resume artifact ignored due to Active KC score signature mismatch",
            path=ACTIVE_KC_PATH,
            requested_score_signature=score_signature,
        )
        active_kc = None
    if active_kc and active_kc.get("selection_policy", {}).get("include_multimodal") != _env_bool(
        "ACTIVE_KC_INCLUDE_MULTIMODAL",
        False,
    ):
        log(
            "resume artifact ignored due to Active KC multimodal policy mismatch",
            path=ACTIVE_KC_PATH,
            include_multimodal=_env_bool("ACTIVE_KC_INCLUDE_MULTIMODAL", False),
        )
        active_kc = None
    if not active_kc:
        with span("select Active KCs", bank_kcs=len(kc_bank.get("kc_nodes", []))):
            active_kc = select_active_kcs(kc_bank, macro_spine)
        active_kc["source_score_signature"] = score_signature
        active_kc["reasoning_edge_source"] = graph_edge_source
        _write_json(KC_BANK_PATH, kc_bank)
        _write_json(ACTIVE_KC_PATH, active_kc)
        _write_build_checkpoint(checkpoint_path, paper_id, "active_kc", resume=resume, restart=restart)
    _sync_active_flags(kc_bank, active_kc)
    _write_json(KC_BANK_PATH, kc_bank)
    log("KC Bank ready", path=KC_BANK_PATH, kcs=len(kc_bank.get("kc_nodes", [])))
    log("Active KCs ready", path=ACTIVE_KC_PATH, active=len(active_kc.get("active_kc_ids", [])))

    graph = _load_resumable_json(GRAPH_PATH, paper_id, resume, restart, "master graph")
    graph_kc_source = os.getenv(
        "MASTER_GRAPH_KC_SOURCE",
        "bank" if graph_edge_source == "verified" else "active",
    ).strip().lower() or "bank"
    if graph_kc_source not in {"bank", "active"}:
        raise ValueError("MASTER_GRAPH_KC_SOURCE must be 'bank' or 'active'.")
    graph_kcs = kc_bank["kc_nodes"] if graph_kc_source == "bank" else active_kc["kc_nodes"]
    if graph and graph.get("diagnostics", {}).get("graph_signature") != _master_graph_signature(
        graph_edge_source,
        graph_kc_source,
        graph_kcs,
        graph_reasoning_edges,
    ):
        log(
            "resume artifact ignored due to Master Graph signature mismatch",
            path=GRAPH_PATH,
            edge_source=graph_edge_source,
        )
        graph = None
    if not graph:
        with span("build master graph", kcs=len(graph_kcs)):
            graph = build_master_graph(
                paper_id=paper_id,
                paper_text_path=paper_text_path,
                kcs=graph_kcs,
                client=client,
                allow_offline_fallback=allow_offline_fallback,
                macro_spine=macro_spine,
                kc_bank_path="data/graphs/kc_bank.json",
                active_kc_path="data/graphs/active_kc.json",
                precomputed_reasoning_edges=graph_reasoning_edges,
                reasoning_edge_source=graph_edge_source,
                edge_coverage_report_path="data/graphs/edge_coverage_report.json" if coverage_report else None,
            )
            graph.setdefault("diagnostics", {})["graph_signature"] = _master_graph_signature(
                graph_edge_source,
                graph_kc_source,
                graph_kcs,
                graph_reasoning_edges,
            )
            graph["diagnostics"]["master_graph_kc_source"] = graph_kc_source
        _write_build_checkpoint(checkpoint_path, paper_id, "master_graph_base", resume=resume, restart=restart)
    if multimodal_enabled:
        graph["multimodal_assets_path"] = "data/multimodal/multimodal_assets.json"
        graph["multimodal_asset_groups_path"] = "data/multimodal/multimodal_asset_groups.json"
        graph["multimodal_asset_explanations_path"] = "data/multimodal/multimodal_asset_explanations.json"
        graph.setdefault("diagnostics", {})["multimodal_enabled"] = True
        graph["diagnostics"]["multimodal_summary"] = (
            multimodal_assets_payload.get("summary", {}) if multimodal_assets_payload else {}
        )
        graph["diagnostics"]["multimodal_explanation_summary"] = (
            multimodal_asset_explanations_payload.get("summary", {}) if multimodal_asset_explanations_payload else {}
        )
    if coverage_report:
        coverage_report = attach_reasoning_path_coverage(coverage_report, graph.get("reasoning_paths", []))
        _write_json(EDGE_COVERAGE_REPORT_PATH, coverage_report)

    reasoning_threads = _load_resumable_json(REASONING_THREADS_PATH, paper_id, resume, restart, "reasoning threads")
    if reasoning_threads and (
        reasoning_threads.get("source_graph_signature") != graph.get("diagnostics", {}).get("graph_signature")
        or reasoning_threads.get("thread_builder_version") != "v2_verified_edges"
        or not _reasoning_threads_have_v2_steps(reasoning_threads)
    ):
        log(
            "resume artifact ignored due to Reasoning Thread graph signature mismatch",
            path=REASONING_THREADS_PATH,
        )
        reasoning_threads = None
    if not reasoning_threads:
        with span("build reasoning threads", paths=len(graph.get("reasoning_paths", []))):
            reasoning_threads = build_reasoning_threads(
                paper_id=paper_id,
                macro_spine=macro_spine,
                active_kc=active_kc,
                reasoning_edges=graph.get("reasoning_edges", []),
                reasoning_paths=graph.get("reasoning_paths", []),
                client=client,
                edge_coverage_report=coverage_report,
            )
        reasoning_threads["source_graph_signature"] = graph.get("diagnostics", {}).get("graph_signature")
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
    if multimodal_enabled:
        print(f"Paper blocks generated: {PAPER_BLOCKS_PATH}")
        print(f"Multimodal asset groups generated: {MULTIMODAL_ASSET_GROUPS_PATH}")
        print(f"Multimodal assets generated: {MULTIMODAL_ASSETS_PATH}")
        print(f"Multimodal asset explanations generated: {MULTIMODAL_ASSET_EXPLANATIONS_PATH}")
    if _env_bool("EXTRACTION_UNIT_ENABLED", True):
        print(f"Extraction Units generated: {EXTRACTION_UNITS_PATH}")
    print(f"KC Bank generated: {KC_BANK_PATH}")
    if graph_edge_source == "verified":
        print(f"Edge coverage report generated: {EDGE_COVERAGE_REPORT_PATH}")
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


def _kc_bank_matches_requested_source(kc_bank: dict, source: str) -> bool:
    nodes = kc_bank.get("kc_nodes", [])
    if not nodes:
        return False
    if source == "unit":
        return all(str(node.get("unit_id", "")).strip() for node in nodes)
    return True


def _kc_bank_has_multimodal_kcs(kc_bank: dict) -> bool:
    return any(
        bool(node.get("modality", {}).get("is_multimodal"))
        for node in kc_bank.get("kc_nodes", [])
        if isinstance(node, dict)
    )


def _kc_candidates_signature(candidates: list[dict]) -> dict:
    return {
        "signature_version": "v1_kc_candidates",
        "candidate_count": len(candidates),
        "candidate_refs": [
            {
                "candidate_id": candidate.get("candidate_id"),
                "unit_id": candidate.get("unit_id"),
                "source_window_id": candidate.get("source_window_id"),
                "macro_id": candidate.get("macro_id"),
                "type": candidate.get("type"),
                "asset_id": candidate.get("asset_id"),
                "asset_type": candidate.get("asset_type"),
                "is_multimodal": bool(candidate.get("modality", {}).get("is_multimodal")),
                "claim": candidate.get("claim"),
            }
            for candidate in candidates
        ],
    }


def _kc_bank_signature(kc_bank: dict) -> dict:
    nodes = kc_bank.get("kc_nodes", [])
    return {
        "signature_version": EDGE_ARTIFACT_SIGNATURE_VERSION,
        "kc_count": len(nodes),
        "kc_refs": [
            {
                "kc_id": node.get("kc_id"),
                "source_candidate_id": node.get("source_candidate_id"),
                "unit_id": node.get("unit_id"),
                "macro_id": node.get("macro_id"),
                "type": node.get("type"),
                "asset_id": node.get("asset_id"),
                "asset_type": node.get("asset_type"),
                "is_multimodal": bool(node.get("modality", {}).get("is_multimodal")),
            }
            for node in nodes
        ],
    }


def _payload_has_kc_bank_signature(payload: dict | None, expected_signature: dict, label: str, path: Path) -> bool:
    if payload is None:
        return False
    if payload.get("kc_bank_signature") == expected_signature:
        return True
    log(
        "resume artifact ignored due to KC Bank signature mismatch",
        label=label,
        path=path,
    )
    return False


def _verified_edge_signature(edges: list[dict]) -> dict:
    return {
        "edge_count": len(edges),
        "edge_refs": [
            [
                edge.get("edge_id"),
                edge.get("source"),
                edge.get("target"),
                edge.get("relation"),
                edge.get("source_layer"),
            ]
            for edge in edges
        ],
    }


def _score_signature(edge_source: str, reasoning_edges: list[dict]) -> dict:
    return {
        "reasoning_edge_source": edge_source,
        "edge_ids": [edge.get("edge_id") for edge in reasoning_edges],
        "edge_count": len(reasoning_edges),
        "edge_sources": [
            [
                edge.get("edge_id"),
                edge.get("source"),
                edge.get("target"),
                edge.get("relation"),
                edge.get("source_layer"),
            ]
            for edge in reasoning_edges
        ],
    }


def _master_graph_signature(edge_source: str, kc_source: str, graph_kcs: list[dict], reasoning_edges: list[dict]) -> dict:
    return {
        "master_graph_builder_version": MASTER_GRAPH_BUILDER_VERSION,
        "reasoning_edge_source": edge_source,
        "master_graph_kc_source": kc_source,
        "graph_kc_ids": [kc.get("kc_id") for kc in graph_kcs],
        "edge_ids": [edge.get("edge_id") for edge in reasoning_edges],
        "edge_count": len(reasoning_edges),
    }


def _reasoning_threads_have_v2_steps(reasoning_threads: dict) -> bool:
    threads = reasoning_threads.get("threads", [])
    if not threads:
        return False
    for thread in threads:
        if not thread.get("edge_sequence"):
            return False
        for step in thread.get("planned_turns", []):
            if "supporting_edge_ids" not in step or not step.get("expected_reasoning"):
                return False
            if step.get("role") == "bridge_reasoning" and not step.get("supporting_edge_ids"):
                return False
    return True


def _sync_active_flags(kc_bank: dict, active_kc: dict) -> None:
    active_ids = set(active_kc.get("active_kc_ids", []))
    for node in kc_bank.get("kc_nodes", []):
        flags = node.setdefault("flags", {})
        active = node.get("kc_id") in active_ids
        flags["active_for_question_generation"] = active
        flags["active_for_core_metrics"] = active
        flags.setdefault("usable_for_claim_verification", True)


def _renumber_verified_edges(edges: list[dict]) -> list[dict]:
    out = []
    for idx, edge in enumerate(edges, start=1):
        item = dict(edge)
        item["edge_id"] = f"E{idx}"
        out.append(item)
    return out


def _merge_edge_verification_summaries(left: dict, right: dict) -> dict:
    return {
        "candidate_count": int(left.get("candidate_count", 0)) + int(right.get("candidate_count", 0)),
        "verified_count": int(left.get("verified_count", 0)) + int(right.get("verified_count", 0)),
        "rejected_count": int(left.get("rejected_count", 0)) + int(right.get("rejected_count", 0)),
    }


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
                "extraction_units": str(EXTRACTION_UNITS_PATH),
                "kc_candidates": str(KC_CANDIDATES_PATH),
                "kc_bank": str(KC_BANK_PATH),
                "edge_candidate_units": str(EDGE_CANDIDATE_UNITS_PATH),
                "edge_candidate_macro": str(EDGE_CANDIDATE_MACRO_PATH),
                "edge_candidate_cross_macro": str(EDGE_CANDIDATE_CROSS_MACRO_PATH),
                "edge_candidate_thread": str(EDGE_CANDIDATE_THREAD_PATH),
                "verified_edges": str(VERIFIED_EDGES_PATH),
                "edge_verification_log": str(EDGE_VERIFICATION_LOG_PATH),
                "edge_coverage_report": str(EDGE_COVERAGE_REPORT_PATH),
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
