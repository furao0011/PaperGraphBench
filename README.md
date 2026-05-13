# paper2Bench

Paper2Bench uses the PaperGraph pipeline to turn a PDF paper into a graph, generate questions, and evaluate one target model in a reproducible dialogue trajectory.

The project is intentionally runnable with `uv`; the current code path only uses Python's standard library plus online OpenAI-compatible APIs, so dependency setup should stay light.

## Quick Start

```powershell
uv sync
uv run python --version
```

Create a local `.env` from `.env.example`, then fill the API settings:

- Construction / generation / judge model: `API_KEY`, `BASE_URL`, `LLM_MODEL`
- Embedding model: `EMBED_API_KEY`, `EMBED_BASE_URL`, `EMBED_MODEL`
- Vision model: `VISION_API_KEY`, `VISION_BASE_URL`, `VISION_MODEL`
- Evaluated model: `EVAL_TARGET_API_KEY`, `EVAL_TARGET_BASE_URL`, `EVAL_TARGET_MODEL`
- OCR service: `PADDLE_OCR_API_URL`, `PADDLE_OCR_TOKEN`

Do not commit `.env`; keep real API keys local.

## Formal Run Order

The formal entry path is PDF-first:

1. Put PDF files under `pdfInput/`.
2. Run OCR. Parsed markdown and images are written to `rawPaper/<paper_id>/`.
3. Set `PAPER_ID` to that `<paper_id>`.
4. Build the graph, generate questions, and run evaluation. Final artifacts are written under `papergraph_demo/data/<paper_id>/`.

When `PAPER_ID` is empty during OCR, each PDF's sanitized filename becomes its `paper_id`. If the filename is not a usable identifier, set `PAPER_ID` and OCR a single PDF through `PAPERGRAPH_PDF_INPUT_FILE`.

### 1. OCR PDF

```powershell
uv run python papergraph_demo\run_parse_pdf.py
```

For a single file:

```powershell
$env:PAPERGRAPH_PDF_INPUT_FILE='pdfInput\your-paper.pdf'
$env:PAPER_ID='your-paper-id'
uv run python papergraph_demo\run_parse_pdf.py
```

The OCR step writes `rawPaper/<paper_id>/papergraph_ocr_manifest.json`, which records the `PAPER_ID` and `PAPER_INPUT_DIR` values expected by later stages.

### 2. Build Graph

```powershell
$env:PAPER_ID='your-paper-id'
uv run python papergraph_demo\run_build_graph.py
```

Core final artifacts:

- `papergraph_demo/data/<paper_id>/master_graph.json`
- `papergraph_demo/data/<paper_id>/multimodal_assets.json`
- `papergraph_demo/data/<paper_id>/multimodal_asset_explanations.json`
- `papergraph_demo/data/<paper_id>/paper_clean_text.md`
- `papergraph_demo/data/<paper_id>/paper_eval_context.md`

Intermediate graph artifacts and checkpoints are written under `papergraph_demo/data/<paper_id>/cache/`.

### 3. Generate Questions

```powershell
$env:PAPER_ID='your-paper-id'
uv run python papergraph_demo\run_generate_questions.py
```

Core final artifact:

- `papergraph_demo/data/<paper_id>/question_templates.json`

Challenge plans, solver trials, and generation caches are written under `papergraph_demo/data/<paper_id>/cache/questions/`.

### 4. Run Evaluation

```powershell
$env:PAPER_ID='your-paper-id'
uv run python papergraph_demo\run_evaluation.py
```

Core final artifacts:

- `papergraph_demo/data/<paper_id>/dialogue_trajectory.json`
- `papergraph_demo/data/<paper_id>/evaluation_report.json`
- `papergraph_demo/data/<paper_id>/eval_state_graph.json`

Evaluation checkpoints and Mermaid state graphs are written under `papergraph_demo/data/<paper_id>/cache/evaluation/`.

## Optional Local Smoke Test

`util_example/output1/` may exist in a local development checkout as a small parsed-paper sample. It is ignored by Git and is not part of the formal public flow. To use it intentionally:

```powershell
$env:PAPER_ID='output1'
$env:PAPER_INPUT_DIR='util_example\output1'
uv run python papergraph_demo\run_build_graph.py
```

Question schema:
- `macro_main_questions`: v1 Macro main questions.
- `thread_question_seeds`: planned thread question goals; bridge/review questions are generated at evaluation time from real dialogue history.
- `review_question_seeds`: reserved for end-of-dialogue review planning.
- `main_questions`: compatibility alias for the current evaluation runner.
- `multi_hop_questions`: compatibility field; v1 moves multi-turn reasoning to `thread_question_seeds`.
- `challenge_plans_path`: path to v2 structured Challenge Plans for later challenge question generation.
- `challenge_plan_summary`: count summary for generated Challenge Plans.
- `challenge_questions_raw_path`: path to v2 natural-language raw challenge questions before solver filtering.
- `challenge_question_raw_summary`: count summary for raw challenge questions.
- `challenge_questions_filtered_path`: path to challenge questions that fooled at least one configured solver.
- `challenge_solver_trials_path`: path to per-question solver answers and judge decisions.
- `challenge_filter_summary`: count summary for solver-based challenge filtering.
- `challenge_questions`: filtered challenge question bank used by later challenge scheduling.
- `challenge_scheduler_config`: scheduler-facing config for later Macro/Thread challenge insertion.

## Runtime parameters

### API config in `.env`

LLM:
- `API_KEY`: API key for chat/completions.
- `BASE_URL`: OpenAI-compatible chat base URL.
- `LLM_MODEL`: model used for extraction, scoring, judging, and generation.

Embedding:
- `EMBED_API_KEY`: API key for embeddings.
- `EMBED_BASE_URL`: OpenAI-compatible embedding base URL.
- `EMBED_MODEL`: embedding model, for example `text-embedding-v4`.
- `EMBED_BATCH_SIZE` (default `10`): embedding request batch size. Keep `<=10` for DashScope `text-embedding-v4`.
- `EMBED_DIM` (default `1024`): embedding dimension metadata.
- `EMBED_MAX_TOKENS` (default `8192`): embedding token-limit metadata.

Workspace:
- `WORKING_DIR` (default `./working`): reserved working directory setting.
- `USE_ONLINE_KC_EXTRACT`: legacy/reserved switch; current strict online build path is controlled by model config and `ALLOW_OFFLINE_FALLBACK`.

### Input selection

- `PAPER_INPUT_DIR`: OCR markdown directory (`doc_*.md`).
- `PAPER_INPUT_FILE`: single markdown file.

### Global execution controls

- `PAPERGRAPH_PROGRESS=true`: print progress logs.
- `PAPERGRAPH_RESUME=true`: reuse matching stage artifacts/checkpoints and continue.
- `PAPERGRAPH_RESTART=true`: ignore existing artifacts/checkpoints and rebuild/regenerate/rerun.
- `ALLOW_OFFLINE_FALLBACK=true`: allow local fallback paths for debugging only.
- `ALLOW_MOCK_EVAL=true`: allow mock target answers in evaluation only.

`RESUME` and `RESTART` are intentionally separate. `RESUME` means “continue from existing artifacts”; `RESTART` means “force a clean run even if artifacts exist.” In ordinary interrupted runs, use only `PAPERGRAPH_RESUME=true`. Use `PAPERGRAPH_RESTART=true` when old cached artifacts may be stale or incompatible.

### Build graph controls

- `BUILD_GRAPH_RESUME=true`: resume graph construction only.
- `BUILD_GRAPH_RESTART=true`: restart graph construction only.
- `BUILD_GRAPH_CHECKPOINT_PATH`: override graph-build checkpoint path.

Macro Spine:
- `MACRO_TARGET_COUNT` (default `8`): preferred dynamic Macro count.
- `MACRO_MIN_COUNT` (default `6`): minimum Macro count.
- `MACRO_MAX_COUNT` (default `12`): maximum Macro count.

Extraction Units:
- `EXTRACTION_UNIT_ENABLED` (default `true`): decompose sections into semantic Extraction Units before KC extraction.
- `UNIT_DECOMP_MAX_CHARS` (default `12000`): max chars sent to one decomposition request before coarse windowing.
- `UNIT_DECOMP_WINDOW_CHARS` (default `10000`): coarse window char limit for long sections.
- `UNIT_DECOMP_WINDOW_OVERLAP_PARAGRAPHS` (default `1`): paragraph overlap between coarse windows.
- `UNIT_MAX_CHARS_SOFT` (default `2500`): soft prompt hint for unit size; validation does not split units by this value.
- `UNIT_DECOMP_WORKERS` (default `3`): concurrent section/window decomposition workers.

KC extraction and scoring:
- `KC_EXTRACTION_SOURCE` (default `unit`): `unit` uses `extraction_units.json`; `section` keeps the legacy section/chunk extractor.
- `UNIT_KC_WORKERS` (default `4`): concurrent workers for Unit-level full KC extraction.
- `KC_PER_UNIT_LIMIT` (default `0`): optional Unit-level KC response cap; `0` means no limit.
- `KC_PER_UNIT_HARD_CAP` (default `20`): validation cap for one Unit response; `0` disables the cap.
- `ONLINE_SECTION_LIMIT` (default `0`): legacy section/chunk extractor only; max sections for online KC candidate extraction.
- `ONLINE_KC_WORKERS` (default `4`): legacy section/chunk extractor only; concurrent workers.
- `KC_EXTRACTION_CHUNK_CHARS` (default `7000`): max text length for one KC extraction chunk.
- `KC_EXTRACTION_CHUNK_OVERLAP_CHARS` (default `600`): overlap between long-section chunks.
- `KC_PER_EXTRACTION_CHUNK` (default `5`): legacy section/chunk extractor only; max KC candidates accepted from one chunk.
- `KC_BANK_MAX` (default `0`): maximum number of KC candidates kept in KC Bank; `0` means keep all candidates.
- KC Bank does not semantic-merge candidates. It only adds conservative duplicate metadata.
- `KC_NEAR_DUPLICATE_JACCARD` (default `0.92`): token Jaccard threshold for near-exact duplicate grouping inside the same Macro.
- `KC_NEAR_DUPLICATE_MIN_TOKENS` (default `5`): minimum normalized token count for near-exact duplicate grouping.
- `RUBRIC_ONLINE_WORKERS` (default `4`): concurrent workers for rubric generation.
- `RUBRIC_ONLINE_BUDGET`:
  - integer `N`: first N KCs online rubric in legacy graph-builder paths;
  - `0`: local rubric only when fallback is allowed;
  - `all` / `full` / `unlimited` / `-1`: all KCs online rubric.

Active KC:
- `ACTIVE_KC_TARGET` (default `30`): target Active KC count.
- `ACTIVE_KC_MIN_PER_MACRO` (default `2`): minimum Active KCs per Macro.
- `ACTIVE_KC_CRITICAL_MIN_PER_MACRO` (default `3`): minimum Active KCs for critical Macros.
- `ACTIVE_KC_THRESHOLD` (default `0.65`): importance threshold for Active KC candidates.

Reasoning Threads:
- `REASONING_THREAD_TARGET` (default `4`): preferred thread count.
- `REASONING_THREAD_MIN` (default `2`): minimum thread count.
- `REASONING_THREAD_MAX` (default `5`): maximum thread count.
- Reasoning Thread v2 uses only verified edges inside the Active KC subgraph. Each `bridge_reasoning` planned turn must include `supporting_edge_ids`, and each planned turn records `expected_reasoning`.

Edge construction:
- `GRAPH_REASONING_EDGE_SOURCE` (default `verified`): `verified` uses `verified_edges.json` for scoring and Master Graph reasoning edges; `legacy` explicitly uses the older KC Bank reasoning edge generator.
- `MASTER_GRAPH_KC_SOURCE` (default `bank` when `GRAPH_REASONING_EDGE_SOURCE=verified`, otherwise `active`): controls whether `master_graph.json.kc_nodes` contains all KC Bank nodes or only Active KC nodes.
- `EDGE_UNIT_ENABLED` (default `true`): build and verify Unit-internal local edge candidates.
- `EDGE_UNIT_WORKERS` (default `3`): concurrent workers for Unit edge candidate generation.
- `EDGE_MACRO_INTERNAL_ENABLED` (default `true`): build and verify Macro-internal edge candidates across different Units.
- `EDGE_MACRO_WORKERS` (default `3`): concurrent workers for Macro-internal edge candidate generation.
- `EDGE_MACRO_BATCH_KCS` (default `30`): maximum KC count in one Macro edge candidate batch.
- `EDGE_ADJACENT_MACRO_ENABLED` (default `true`): build and verify adjacent-Macro progression edge candidates.
- `EDGE_ADJACENT_MACRO_WORKERS` (default `3`): concurrent workers for adjacent-Macro edge generation.
- `EDGE_ADJACENT_MACRO_TOP_KCS` (default `12`): representative KCs per Macro side for adjacent-Macro edge generation.
- `EDGE_THREAD_CANDIDATE_ENABLED` (default `true`): build and verify cross-Macro Thread candidate edges after lower-layer verified edges exist.
- `EDGE_THREAD_CANDIDATE_KCS` (default `40`): representative KC count passed to Thread candidate edge generation.
- `EDGE_THREAD_VERIFIED_HINTS` (default `80`): lower-layer verified edge hints passed to Thread candidate edge generation.
- `EDGE_VERIFY_WORKERS` (default `3`): concurrent workers for edge verification.
- `EDGE_COVERAGE_MIN_MACRO_INCIDENT_EDGES` (default `1`): minimum incident verified edges expected for each Macro in `edge_coverage_report.json`.

v2 graph construction uses paragraph-id evidence for Extraction Units, KC evidence, and edge evidence. Prompts select paragraph IDs, while Python reconstructs `source_text` / `evidence` from the original text so strict validation does not depend on the model copying spans byte-for-byte.

### Question generation controls

- `QUESTION_RESUME=true`: resume question generation cache only.
- `QUESTION_RESTART=true`: ignore question generation cache only.
- `QUESTION_CACHE_PATH`: override question-generation cache path.
- `CHALLENGE_ACCEPT_TARGET` (default `10`): accepted challenge question target for the loop.
- `CHALLENGE_MAX_ATTEMPTS_PER_PLAN` (default `3`): maximum question-regeneration attempts for one Challenge Plan.
- `CHALLENGE_PLAN_POOL_TARGET` (default falls back to `CHALLENGE_PLAN_TARGET`, currently `30`): structured Challenge Plan pool size before random loop sampling.
- `CHALLENGE_PLAN_POOL_PER_TYPE_LIMIT` (default falls back to `CHALLENGE_PLAN_PER_TYPE_LIMIT`, currently `12`): per-type cap for the Challenge Plan pool.
- `CHALLENGE_RANDOM_SEED`: optional seed for deterministic random Challenge Plan order.
- `CHALLENGE_PLAN_EVIDENCE_MAX_CHARS` (default `1200`): evidence text cap stored in each Challenge Plan.
- `CHALLENGE_LOOP_CACHE_PATH`: override the closed-loop challenge generation cache path.
- `CHALLENGE_SOLVER_COUNT` (default `3`): number of solver models used to test each raw challenge question.
- `CHALLENGE_SOLVER_MODELS`: comma-separated solver model names. The count must match `CHALLENGE_SOLVER_COUNT`; when empty, the common `LLM_MODEL` is repeated.
- `CHALLENGE_SOLVER_TEMPERATURE` (default `1.5`): shared solver temperature. Current supported range is `0` to `2`.
- `CHALLENGE_SOLVER_WORKERS` (default = solver count): concurrent workers for solver calls inside one challenge question.
- `CHALLENGE_JUDGE_TEMPERATURE` (default `0.1`): judge temperature for deciding whether each solver answer falls into the target failure mode.
- `CHALLENGE_META_JUDGE_TEMPERATURE` (default `0.1`): judge temperature for question usability and plan-easiness diagnosis.

Challenge generation is a closed loop: randomly sample one structured plan, generate one question, judge whether the question is usable, run solver trials if usable, accept it only when at least one solver matches the target failure mode, otherwise diagnose whether the plan itself is too easy. Easy plans are blacklisted; fixable easy questions are regenerated with the diagnosis as revision feedback. Solver diversity is controlled by `CHALLENGE_SOLVER_MODELS`, while all solvers share `CHALLENGE_SOLVER_TEMPERATURE`. Each solver receives the same prompt shape as one evaluation turn: full original paper, empty dialogue history, and current challenge question.

### Evaluation controls

- `USE_ONLINE_EVAL=true|false`: whether target answers and judge use online model in evaluation.
- `EVAL_RESUME=true`: resume evaluation checkpoint only.
- `EVAL_RESTART=true`: ignore evaluation checkpoint only.
- `EVAL_CHECKPOINT_PATH`: override evaluation checkpoint path.
- `EVAL_MAX_TURNS`: stop evaluation after N turns; `0` or empty means no explicit cap. Use `0` when you want all generated challenge questions to be usable during one evaluation run.
- `EVAL_PAPER_CHAR_LIMIT` (default `0`): truncate original paper context; `0` means no truncation.
- `EVAL_MISLEADING_PER_MACRO` (default `1`, max `2`): misleading follow-ups per Macro.
- `EVAL_REVIEW_AT_END` (default `2`, max `3`): review follow-ups at the end.
- `EVAL_THREAD_TURNS_PER_CHECK` (default `1`, max `3`): maximum ready Thread turns inserted after each Macro/follow-up checkpoint.
- `EVAL_CHALLENGE_PER_MACRO` (default `1`, supports `all`): maximum pre-mined Macro-level challenge questions inserted for each Macro checkpoint. Set to `all` to keep scheduling until the filtered challenge bank is exhausted or the global turn cap is reached.
- `EVAL_CHALLENGE_PER_THREAD` (default `1`): maximum pre-mined Thread-level challenge questions inserted for each Thread.

Evaluation completion status:
- `eval_state_graph.json.global_state.evaluation_status` is one of `not_started`, `running`, `completed`, `failed`, `stopped_by_max_turns`, or `interrupted`.
- `completion_reason` explains why the evaluation ended.
- `completed_at_turn` records the last turn number when the terminal state was written.

Global Claim Verification:
- `CLAIM_VERIFY_ENABLED=true`: verify extra paper-related claims in each model answer.
- Claim retrieval uses the configured online embedding model (`EMBED_API_KEY`, `EMBED_BASE_URL`, `EMBED_MODEL`) and cosine similarity. Token-overlap retrieval is not used.
- `CLAIM_RETRIEVE_TOP_KC` (default `5`): top KC candidates retrieved by embedding similarity for each extracted claim.
- `CLAIM_RETRIEVE_TOP_EVIDENCE` (default `5`): top evidence spans re-ranked by embedding similarity and passed to the verifier.
- `CLAIM_VERIFY_MAX_CLAIMS_PER_TURN` (default `8`): max atomic claims extracted from one answer.

### Model request controls

- `PAPERGRAPH_LLM_TIMEOUT_S` / `LLM_TIMEOUT_S` (default `300`): single request timeout.
- `PAPERGRAPH_LLM_MAX_RETRIES` / `LLM_MAX_RETRIES` (default `2`): retry count after a failed request.
- `PAPERGRAPH_LLM_RETRY_SLEEP_S` / `LLM_RETRY_SLEEP_S` (default `5`): retry sleep in seconds.

## Recommended stable config (online-heavy)

```powershell
$env:ONLINE_SECTION_LIMIT='0'
$env:ONLINE_KC_WORKERS='3'
$env:RUBRIC_ONLINE_BUDGET='all'
$env:RUBRIC_ONLINE_WORKERS='3'
uv run python papergraph_demo\run_build_graph.py
```
