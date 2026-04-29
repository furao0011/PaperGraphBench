import json
from pathlib import Path

from src.config import load_settings
from src.model_client import ModelConfig, OpenAICompatClient
from src.question_generator import generate_questions_with_online_fallback


BASE_DIR = Path(__file__).resolve().parent
GRAPH_PATH = BASE_DIR / "data" / "graphs" / "master_graph.json"
QUESTION_PATH = BASE_DIR / "data" / "questions" / "question_templates.json"


def main() -> None:
    if not GRAPH_PATH.exists():
        raise FileNotFoundError(f"Master graph not found: {GRAPH_PATH}")

    settings = load_settings(BASE_DIR.parent)
    client = OpenAICompatClient(
        ModelConfig(settings.api_key, settings.base_url, settings.llm_model)
    )
    graph = json.loads(GRAPH_PATH.read_text(encoding="utf-8"))
    bundle = generate_questions_with_online_fallback(graph, client)
    payload = {
        "paper_id": graph.get("paper_id", "unknown"),
        "main_questions": bundle["main_questions"],
        "multi_hop_questions": bundle["multi_hop_questions"],
        "reserved_followup_templates": bundle["reserved_followup_templates"],
    }

    QUESTION_PATH.parent.mkdir(parents=True, exist_ok=True)
    QUESTION_PATH.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Question templates generated: {QUESTION_PATH}")


if __name__ == "__main__":
    main()
