import json
import os
from pathlib import Path

from src.config import load_settings
from src.graph_builder import build_master_graph
from src.kc_extractor import extract_kcs_by_sections_with_online_fallback
from src.mermaid_exporter import export_master_graph_mermaid
from src.model_client import ModelConfig, OpenAICompatClient
from src.paper_parser import load_paper_text, load_paper_text_from_dir, split_into_sections


BASE_DIR = Path(__file__).resolve().parent
PAPER_PATH = BASE_DIR / "data" / "papers" / "demo_paper.md"
PAPER_DIR_PATH = BASE_DIR.parent / "util_example" / "output1"
GRAPH_PATH = BASE_DIR / "data" / "graphs" / "master_graph.json"
MASTER_MMD_PATH = BASE_DIR / "data" / "graphs" / "master_graph.mmd"
SECTIONS_PATH = BASE_DIR / "data" / "graphs" / "sections.json"


def main() -> None:
    project_root = BASE_DIR.parent
    settings = load_settings(project_root)

    input_dir = Path(os.getenv("PAPER_INPUT_DIR", str(PAPER_DIR_PATH)))
    input_file = Path(os.getenv("PAPER_INPUT_FILE", str(PAPER_PATH)))

    if input_dir.exists():
        paper_text = load_paper_text_from_dir(input_dir)
        paper_id = input_dir.name
        paper_text_path = str(input_dir)
    elif input_file.exists():
        paper_text = load_paper_text(input_file)
        paper_id = input_file.stem
        paper_text_path = str(input_file)
    else:
        raise FileNotFoundError(
            f"No valid input found. Checked directory: {input_dir}, file: {input_file}"
        )

    client = None
    if settings.api_key and settings.base_url and settings.llm_model:
        client = OpenAICompatClient(
            ModelConfig(
                api_key=settings.api_key,
                base_url=settings.base_url,
                llm_model=settings.llm_model,
            )
        )

    sections = split_into_sections(paper_text)
    SECTIONS_PATH.parent.mkdir(parents=True, exist_ok=True)
    SECTIONS_PATH.write_text(json.dumps({"paper_id": paper_id, "sections": sections}, ensure_ascii=False, indent=2), encoding="utf-8")
    kcs = extract_kcs_by_sections_with_online_fallback(sections, client)
    graph = build_master_graph(paper_id=paper_id, paper_text_path=paper_text_path, kcs=kcs, client=client)

    GRAPH_PATH.parent.mkdir(parents=True, exist_ok=True)
    GRAPH_PATH.write_text(
        json.dumps(graph, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    MASTER_MMD_PATH.write_text(export_master_graph_mermaid(graph), encoding="utf-8")
    print(f"Sections written: {SECTIONS_PATH}")
    print(f"Master graph generated: {GRAPH_PATH}")


if __name__ == "__main__":
    main()
