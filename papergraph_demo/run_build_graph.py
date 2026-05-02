import json
import os
from pathlib import Path

from src.config import load_settings
from src.graph_builder import build_master_graph
from src.kc_extractor import extract_kcs_by_sections_with_online_fallback
from src.macro_extractor import extract_macro_spine
from src.mermaid_exporter import export_master_graph_mermaid
from src.model_client import ModelConfig, OpenAICompatClient
from src.paper_parser import load_paper_text, load_paper_text_from_dir, split_into_sections
from src.progress import log, span


BASE_DIR = Path(__file__).resolve().parent
PAPER_PATH = BASE_DIR / "data" / "papers" / "demo_paper.md"
PAPER_DIR_PATH = BASE_DIR.parent / "util_example" / "output1"
GRAPH_PATH = BASE_DIR / "data" / "graphs" / "master_graph.json"
MASTER_MMD_PATH = BASE_DIR / "data" / "graphs" / "master_graph.mmd"
SECTIONS_PATH = BASE_DIR / "data" / "graphs" / "sections.json"
MACRO_SPINE_PATH = BASE_DIR / "data" / "graphs" / "macro_spine.json"


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
    with span("extract KCs", sections=len(sections)):
        kcs = extract_kcs_by_sections_with_online_fallback(
            sections,
            client,
            allow_offline_fallback=allow_offline_fallback,
            macro_spine=macro_spine,
        )
    log("KCs extracted", count=len(kcs))
    with span("build master graph", kcs=len(kcs)):
        graph = build_master_graph(
            paper_id=paper_id,
            paper_text_path=paper_text_path,
            kcs=kcs,
            client=client,
            allow_offline_fallback=allow_offline_fallback,
            macro_spine=macro_spine,
        )
    log(
        "master graph ready",
        macros=len(graph.get("macro_nodes", [])),
        kcs=len(graph.get("kc_nodes", [])),
        edges=len(graph.get("reasoning_edges", [])),
        paths=len(graph.get("reasoning_paths", [])),
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
    print(f"Master graph generated: {GRAPH_PATH}")


if __name__ == "__main__":
    main()
