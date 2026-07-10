# PaperGraph-Bench

PaperGraph-Bench is the benchmark construction and evaluation pipeline used by the EMNLP 2026 paper draft under `EMNLP2026/acl-style-files-master`. It turns OCR-parsed scientific papers into graph-guided benchmark instances, then runs multi-turn evaluation over macro questions, challenge questions, and reasoning-thread questions.

## Quick Start

### 1. Prepare the environment

Use Python 3.11+ with `uv`.

```powershell
uv sync
Copy-Item .env.example .env
```

Fill `.env` before running the formal pipeline:

- `API_KEY`, `BASE_URL`, `LLM_MODEL`: construction, question generation, and judge model.
- `EMBED_API_KEY`, `EMBED_BASE_URL`, `EMBED_MODEL`: embedding model.
- `VISION_API_KEY`, `VISION_BASE_URL`, `VISION_MODEL`: vision model for multimodal assets and visual challenge questions.
- `PADDLE_OCR_API_URL`, `PADDLE_OCR_TOKEN`: PDF OCR service.
- `EVAL_TARGET_API_KEY`, `EVAL_TARGET_BASE_URL`, `EVAL_TARGET_MODEL`: evaluated target model.

Formal runs should use online models. Do not enable `ALLOW_OFFLINE_FALLBACK` or `ALLOW_MOCK_EVAL` for paper results.

### 2. Parse PDFs into OCR papers

Put input PDFs under `pdfInput/`, then run:

```powershell
uv run python papergraph_demo\run_parse_pdf.py
```

For one PDF, set the input file and paper id explicitly:

```powershell
$env:PAPERGRAPH_PDF_INPUT_FILE='pdfInput\your-paper.pdf'
$env:PAPER_ID='your-paper-id'
uv run python papergraph_demo\run_parse_pdf.py
```

Output goes to `rawPaper/<paper_id>/`.

### 3. Build the paper graph

Use the `<paper_id>` folder produced by OCR:

```powershell
$env:PAPER_ID='your-paper-id'
uv run python papergraph_demo\run_build_graph.py
```

Key outputs are written to `papergraph_demo/data/<paper_id>/`, including:

- `master_graph.json`
- `multimodal_assets.json`
- `multimodal_asset_explanations.json`
- `paper_clean_text.md`
- `paper_eval_context.md`

### 4. Generate benchmark questions

```powershell
$env:PAPER_ID='your-paper-id'
uv run python papergraph_demo\run_generate_questions.py
```

The main output is:

- `papergraph_demo/data/<paper_id>/question_templates.json`

This file contains the graph-linked macro questions, challenge questions, and reasoning-thread seeds used by the evaluator.

### 5. Run multi-turn evaluation

```powershell
$env:PAPER_ID='your-paper-id'
$env:USE_ONLINE_EVAL='true'
uv run python papergraph_demo\run_evaluation.py
```

Each evaluation job now owns an isolated model-and-paper directory:

```text
eval_result/<target_model>/<paper_id>/
|-- dialogue_trajectory.json
|-- evaluation_report.json
+-- cache/
    |-- evaluation_checkpoint.json
    |-- eval_state_graph.json
    +-- claim_verification_log.json
```

The checkpoint and intermediate state no longer write into papergraph_demo/data/<paper_id>/, so different models can evaluate the same paper concurrently.

Run every paper/model pair concurrently with:

```powershell
uv run python papergraph_demo\run_batch_evaluation.py --workers 4 --continue-on-failure
```

Completed reports are skipped. Interrupted jobs resume from their own cache/evaluation_checkpoint.json; add --force only when every selected job should restart.

### Parallel dataset construction

Different papers can be built concurrently, while graph construction and question generation remain ordered inside each paper:

```powershell
uv run python papergraph_demo\run_batch_dataset.py --workers 2 --continue-on-failure
```

Use --paper-ids <paper_id...> for a subset, --skip-graph or --skip-questions for one stage, and --dry-run to inspect the schedule.

The no-graph ablation uses the same paper/model isolation:

```powershell
uv run python papergraph_demo\run_generate_textonly_questions.py --workers 2
uv run python papergraph_demo\run_textonly_evaluation.py --workers 4
```

For individual entry scripts, set PAPERGRAPH_RESUME=true to resume and use PAPERGRAPH_RESTART=true only when old stage artifacts should be rebuilt.

## Repository Data Layout

- `rawPaper/`: OCR-parsed papers. Each paper folder contains `doc_*.md`, extracted images, and `papergraph_ocr_manifest.json`.
- `data/`: released PaperGraph-Bench dataset artifacts. Each paper folder contains the dataset paper graph, multimodal figure/table asset references, and graph-linked questions used by the paper.
- `eval_result/`: evaluation results grouped by target model and paper, including `dialogue_trajectory.json` and `evaluation_report.json`.
- `papergraph_demo/data/`: local runtime outputs produced by the four entry scripts. These files follow the same schema as the released dataset artifacts.

## Main Entry Points

```text
papergraph_demo/run_parse_pdf.py
papergraph_demo/run_build_graph.py
papergraph_demo/run_generate_questions.py
papergraph_demo/run_evaluation.py
papergraph_demo/run_batch_dataset.py
papergraph_demo/run_batch_evaluation.py
papergraph_demo/run_generate_textonly_questions.py
papergraph_demo/run_textonly_evaluation.py
```

## License

The repository code is released under the Apache License 2.0. See `LICENSE`.

Dataset artifacts in `data/` and evaluation outputs in `eval_result/` are released for research reproduction with this project. OCR-derived paper text and extracted images in `rawPaper/` may remain subject to the copyright and usage terms of the original papers.
