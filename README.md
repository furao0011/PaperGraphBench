# paper2Bench

This workspace uses `uv` for Python environment and dependency management.

## Quick start

```powershell
uv venv .venv
.venv\Scripts\activate
uv run python --version
```

## Demo scaffold

The PaperGraph demo scaffold lives in `papergraph_demo/`.

## Run order (Guidance-aligned)

1. Build graph:

```powershell
uv run python papergraph_demo\run_build_graph.py
```

Outputs:
- `papergraph_demo/data/graphs/sections.json`
- `papergraph_demo/data/graphs/master_graph.json`
- `papergraph_demo/data/graphs/master_graph.mmd`

2. Generate questions:

```powershell
uv run python papergraph_demo\run_generate_questions.py
```

Output:
- `papergraph_demo/data/questions/question_templates.json`

3. Run evaluation:

```powershell
uv run python papergraph_demo\run_evaluation.py
```

Outputs:
- `papergraph_demo/data/outputs/dialogue_trajectory.json`
- `papergraph_demo/data/outputs/evaluation_report.json`
- `papergraph_demo/data/graphs/eval_state_graph.json`
- `papergraph_demo/data/graphs/final_state_graph.mmd`

## Runtime parameters

Input selection:
- `PAPER_INPUT_DIR`: OCR markdown directory (`doc_*.md`).
- `PAPER_INPUT_FILE`: single markdown file.

Online model switches:
- `USE_ONLINE_EVAL=true|false`: whether answers and judge use online model in evaluation.

Prompt-driven extraction/generation controls:
- `ONLINE_SECTION_LIMIT` (default `6`): max sections for online KC extraction.
- `ONLINE_KC_WORKERS` (default `4`): concurrent workers for section-level KC extraction.
- `RUBRIC_ONLINE_WORKERS` (default `4`): concurrent workers for rubric generation.
- `RUBRIC_ONLINE_BUDGET`:
  - integer `N`: first N KCs online rubric;
  - `0`: local rubric only;
  - `all` / `full` / `unlimited` / `-1`: all KCs online rubric.

API config in `.env`:
- `API_KEY`
- `BASE_URL`
- `LLM_MODEL`
- `EMBED_MODEL`
- `WORKING_DIR`
- `EMBED_DIM`
- `EMBED_MAX_TOKENS`

## Recommended stable config (online-heavy)

```powershell
$env:ONLINE_SECTION_LIMIT='4'
$env:ONLINE_KC_WORKERS='3'
$env:RUBRIC_ONLINE_BUDGET='all'
$env:RUBRIC_ONLINE_WORKERS='3'
uv run python papergraph_demo\run_build_graph.py
```
