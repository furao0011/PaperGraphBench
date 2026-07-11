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

Put input PDFs under `pdfInput/`, then process the whole directory with paper-level workers:

```powershell
$env:PAPERGRAPH_PROGRESS='true'
uv run python papergraph_demo\run_parse_pdf.py --all-pdfs --workers 2 --continue-on-failure
```

`--all-pdfs` intentionally ignores any stale single-paper `PAPER_ID` and
`PAPERGRAPH_PDF_INPUT_FILE` values. Each filename stem is normalized into its
stable `paper_id`, and output is written to `rawPaper/<paper_id>/`. Completed
OCR directories are validated and skipped; use `--overwrite` only for an
intentional rebuild.

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

Run formal evaluation for every paper selected from `rawPaper/`:

```powershell
$env:PAPERGRAPH_PROGRESS='true'
uv run python papergraph_demo\run_batch_evaluation.py --workers 2 --continue-on-failure
```

A worker owns one paper and evaluates its models sequentially. Different papers
run concurrently. The default model set is `gpt-5-mini`, `gpt-5`, Doubao Pro,
and Doubao Mini; override it with `--models` or `EVAL_BATCH_MODELS`.
Completed reports are skipped, and interrupted runs resume from their isolated
`cache/evaluation_checkpoint.json`. Per-model logs are stored under
`logs/main_evaluation/<paper_id>/`; add `--force` only for an intentional
restart.

### Parallel dataset construction

Different papers under `rawPaper/` can be built concurrently, while graph
construction and question generation remain ordered inside each paper:

```powershell
$env:PAPERGRAPH_PROGRESS='true'
uv run python papergraph_demo\run_batch_dataset.py --workers 2 --continue-on-failure
```

Use `--skip-questions` to build only graphs, or `--skip-graph` to generate
questions from completed graphs. Use `--paper-ids <paper_id...>` for a subset
and `--dry-run` to inspect the schedule. Child-process logs are isolated under
`logs/main_dataset/<paper_id>/`.

### No-graph ablation

The no-graph ablation scans every eligible paper under the repository-root `data/<paper_id>/` directory. Enable the Rich live panel with `PAPERGRAPH_PROGRESS=true`:

```powershell
$env:PAPERGRAPH_PROGRESS='true'
uv run python papergraph_demo/run_generate_textonly_questions.py --workers 2 --continue-on-failure
uv run python papergraph_demo/run_textonly_evaluation.py --workers 4 --continue-on-failure
```

The build command reads `paper_clean_text.md`, `multimodal_assets.json`, and `multimodal_asset_explanations.json`. It writes the final package to:

```text
data/<paper_id>/textonly_question_templates.json
```

Each paper is one build job. A worker takes the next queued paper after finishing its current paper. The generated package contains macro questions, text challenge questions, and multimodal challenge questions.

Challenge construction follows the full filtering shape without using the graph:

```text
paper text / asset summaries
-> macro questions
-> text + multimodal challenge plan pools
-> natural challenge question generation
-> usability judge
-> solver trials
-> target-failure matching
-> easy-plan rejection or question revision
-> accepted / rejected / human-review outputs
```

The default pools contain 40 text plans and 40 multimodal plans, with 10 accepted questions required from each pool. Candidate JSON is validated immediately; invalid multimodal plan entries are regenerated in place with their original positions, exact validation errors, and legal asset-id list, while valid entries are preserved. This runs for up to `TEXTONLY_GENERATION_SCHEMA_ATTEMPTS` attempts (default 3). No asset is assigned automatically. Text solver trials use the common model API; multimodal solver trials use the configured vision API and original image attachments. Build checkpoints and audit artifacts are written under:

```text
data/<paper_id>/cache/textonly/
|-- generation_candidates.json
|-- challenge_plans.json
|-- challenge_loop_text.json
|-- challenge_loop_multimodal.json
|-- challenge_questions_raw.json
|-- challenge_questions_filtered.json
|-- challenge_questions_need_human_review.json
|-- challenge_questions_rejected.json
+-- challenge_solver_trials.json
```

Evaluation uses paper-level workers. Each worker runs `gpt-5-mini`, `gpt-5`, Doubao Pro, and Doubao Mini sequentially for one paper before taking the next paper. Macro questions establish dialogue context; accepted text and multimodal challenges are distributed after macro turns. Every new turn receives the complete preceding question-answer history. Repair, detail follow-up, hallucination follow-up, thread, and review tasks are disabled.

Because no graph or KC targets exist, the report does not calculate graph/KC coverage. Macro expected points are used only by the judge to classify the turn and detect unsupported claims; they are not aggregated as coverage. The primary metrics are challenge failure/resistance/incomplete rates, text and multimodal failure rates, per-type and per-failure-mode counts/rates, challenge hallucination events, source solver-filter statistics, total turns, and response lengths.

Evaluation outputs are isolated by model and paper:

```text
eval_resultTextOnly/<model>/<paper_id>/
|-- dialogue_trajectory.json
|-- evaluation_report.json
+-- cache/textonly_evaluation_checkpoint.json
```

Completed model/paper results are skipped, and incomplete results resume from their own checkpoint.
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
