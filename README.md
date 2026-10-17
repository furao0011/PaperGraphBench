# PaperGraph-Bench

[English](README.md) | [简体中文](README.zh-CN.md)

PaperGraph-Bench constructs a graph-guided benchmark from scientific papers and evaluates a target model through multi-turn reading tasks. The standard pipeline parses a PDF when needed, builds a paper graph, generates graph-linked questions, and records dialogue-level coverage, challenge, hallucination, and repair results.

All standard stages run automatically, including graph verification, question validation, solver filtering, and dialogue judging.

For test operators, see the [Indicator 1.3 evaluation plan (Chinese)](docs/evaluation_plan.zh-CN.md) for the four evaluation dimensions, fine-grained hallucination classification, commands, result files, and acceptance criteria.

## Recommended: evaluate existing datasets and generate statistics

> **Running `start.ps1` / `start.sh` without a command evaluates the 12 released paper packages in `data/` and exports per-paper results, model summaries, and fine-grained hallucination classifications. No separate extraction script is needed. It does not rebuild graphs, regenerate questions, or run OCR. Explicit `test` does the same.**

Install [uv](https://docs.astral.sh/uv/), use Python 3.11+, and run from a complete repository checkout:

```powershell
.\start.ps1
```

On macOS/Linux:

```bash
bash ./start.sh
```

**Before evaluation, supply your own API key, base URL, and model ID for these two roles in `config.env`.** The first launch creates a blank file and lists the fields to fill. Edit it, then repeat the same command. Missing settings stop execution before model requests.

| Required service | API key | Provider API base URL | Callable model ID |
| --- | --- | --- | --- |
| Evaluation judge | `API_KEY` | `BASE_URL` | `LLM_MODEL` |
| Evaluated target model | `EVAL_TARGET_API_KEY` | `EVAL_TARGET_BASE_URL` | `EVAL_TARGET_MODEL` |

The judge must support JSON responses; the target must accept images. Use your provider's API root URL, not a website or full `/chat/completions` endpoint. **No vision-generation, embedding, or OCR service is required by default.** Embeddings are additionally needed only if you enable `CLAIM_VERIFY_ENABLED=true`. API access is not bundled; evaluation consumes your provider quota.

### Common testing commands

```powershell
# Test one existing package first
.\start.ps1 test --paper-id ConVis

# Three independent evaluations of all existing packages, followed by statistics
.\start.ps1 test --workers 2 --repeats 3

# Compare models; replace these IDs with models available to your account
.\start.ps1 test --models model-a model-b --workers 2

# Preview selected papers, models and output paths without model requests
.\start.ps1 test --dry-run
```

No command defaults to `test`: all papers in `data/`, one round per model, one worker, followed by statistics. Options also work without `test`, for example `.\start.ps1 --workers 2`. Use `--data-root` for another existing dataset and `--config` for a private configuration. On macOS/Linux, replace `.\start.ps1` with `bash ./start.sh`. If Windows blocks execution, use `powershell -NoProfile -ExecutionPolicy Bypass -File .\start.ps1`.

### Results and statistics

Each test creates `outputs/testing/<session>/` and prints the exact path:

| File or directory | Purpose |
| --- | --- |
| `statistics/four_dimensions.csv` | Paper-aligned four-dimension table, eleven rows per model, with values, aggregation rules, denominators and task status; start here |
| `statistics/failure_category_by_model.csv` | Per-model counts for the paper's four hallucination categories and unmapped events; start here |
| `statistics/summary.csv` | Canonical paper metric values, valid counts and task completeness |
| `statistics/per_paper.csv` | Per-round, per-paper metrics, numerators, denominators and completion status |
| `statistics/summary.json` | Machine-readable statistics |
| `statistics/repair_tasks.csv` | Actual follow-up task groups, success decisions and attempts until first success or termination |
| `statistics/classification_events.csv`, `classification_events.json` | New trajectory events, paper categories, raw labels, origins and detection-time snapshots |
| `statistics/classification_paths.csv` | Original `family/type/subtype` path counts for debugging raw labels |
| `statistics/classification_summary.csv`, `classification_summary.json` | Per-model classification completeness and issue counts; JSON also includes classification-path distributions |
| `round-01/evaluation/<model>/<paper_id>/` | Per-task `evaluation_report.json`, `dialogue_trajectory.json` and state cache |
| `testing.json`, `main.log`, `logs/` | Planned task manifest, main log and stage logs |

Statistics follow the paper: coverage uses initial macro answers, not final global KC state; challenge rates include only ordinary `challenge_question` turns, excluding thread challenges, with Text/MM splits; hallucinations use mean events per paper and event-bearing turns; detail/hallucination repair uses actual `repair_context` task groups, success and attempts rather than final event-closure ratios.

Coverage, macro completion, mean events and mean total turns weight papers equally, averaging repetitions within each paper first. Challenges, event-bearing-turn rate/density, repair success/attempts and answer length pool their numerators and denominators instead of averaging paper ratios. Canonical `summary.csv` values use paper metric names such as `one_turn_kc_coverage`, `challenge_failure_rate`, and `mean_hallucination_events`, without a `_mean` suffix; the four-dimension table provides `value` and `aggregation`. Zero denominators are N/A; missing or incomplete tasks remain separate.

The paper reports four hallucination categories: **KC-evidence contradiction (Contr.KC), false-premise acceptance (False prem.), overclaim acceptance (Overclaim), and wrong-relation acceptance (Wrong rel.)**. Thread premise mutations merge into false premises; thread overclaims into overclaims; wrong bridges and generic thread-reasoning hallucinations into wrong relations; review regressions into KC contradictions. Mapping uses the exact original `hallucination_type`. Unmapped labels, including `fabricated_claim` and generic `challenge_failure`, remain explicit rather than being guessed into a category. Section 4 of the [evaluation plan](docs/evaluation_plan.zh-CN.md) lists the full mapping. Raw `family/type/subtype` fields are technical traceability labels, not additional paper categories.

Classification reads new events from each trajectory turn, validates unique event IDs within a task, and preserves label issues. `status_at_detection`, `followup_turns_at_detection`, and `resolved_at_turn_at_detection` are detection-time snapshots, not final states; actual repair is exported in `repair_tasks.csv`. **Statistics require only completed reports and trajectories, never final-state caches or reconstructed state.** With no events, classification completeness is N/A; complete or distinct label fields do not prove three semantic levels.

Start with `four_dimensions.csv` and `failure_category_by_model.csv`, and use `repair_tasks.csv` to inspect repair. The default launch creates all eleven statistics files above. Category counts sum new events across completed papers and independent runs; they are not mean events per paper. The four counts plus `unmapped_event_count` equal the total event count. A successful API call does not validate answers or labels.

Existing results can be summarized offline using **only `evaluation_report.json` and `dialogue_trajectory.json`, without API settings, benchmark packages, state caches, or model calls**:

```powershell
uv run --locked python scripts/summarize_results.py --result-root eval_result
```

This exports the same eleven files as default evaluation of a new model. `--result-root` also accepts `outputs/evaluation`, a test session, or one round's results. Historical directories without a manifest have an unknown planned total. Missing reports/trajectories are incomplete tasks; corrupt or inconsistent data are errors. Missing optional caches are allowed.

For the original fixed paper experiment, Section 3.1 of the [evaluation plan](docs/evaluation_plan.zh-CN.md) shows offline verification with `scripts/verify_paper_results.py`. All 150 core checks match; the known GLM answer-length discrepancy (1737.1 recomputed vs. 1736.1 in the paper) is reported separately without changing results. New discrepancies fail verification. This reference check is not a performance requirement for new models.

## Optional: API configuration for rebuilding benchmarks

> **You must provide your own API keys, provider base URLs, and model IDs in the root `config.env`. No model services or credentials are bundled. Calls consume your provider quota and may incur charges.**

The full graph → questions → evaluation workflow needs all four groups:

| Service | API key | API base URL | Model ID |
| --- | --- | --- | --- |
| Text model and evaluation judge | `API_KEY` | `BASE_URL` | `LLM_MODEL` |
| Vision model | `VISION_API_KEY` | `VISION_BASE_URL` | `VISION_MODEL` |
| Embedding model | `EMBED_API_KEY` | `EMBED_BASE_URL` | `EMBED_MODEL` |
| Evaluated target model | `EVAL_TARGET_API_KEY` | `EVAL_TARGET_BASE_URL` | `EVAL_TARGET_MODEL` |

Obtain the key, API root URL, and callable model ID from your provider. Do not use a website address or a full `/chat/completions` or `/embeddings` endpoint. The target must support image input for multimodal evaluation. If groups share a service, explicitly fill each group even when the key and URL are the same. Evaluation of existing packages needs only the text/judge and target groups by default. See [Configuration](docs/configuration.md) for details.

## Optional: construct a new benchmark

**Construction requires an explicit `run`, `build`, `questions`, or `measure` command.** Running the launcher without a command only tests existing packages. Skip this section if you do not need a new benchmark.

Install [uv](https://docs.astral.sh/uv/) and use Python 3.11+. From the repository root, run:

```powershell
.\start.ps1 run --paper-id ConVis
```

On macOS/Linux:

```bash
bash ./start.sh run --paper-id ConVis
```

The launcher prepares dependencies and creates `config.env` if absent. **On the first run, fill the API settings listed above, then run the same command again.** Existing configuration is never overwritten. Missing configuration stops execution before model requests and lists the fields to fill.

Once configured, that single command builds the ConVis graph, generates questions, and evaluates the configured target model. The first full build can make many paid API requests. For other runs:

```powershell
# Full measurement: build once, freeze inputs, evaluate three independent rounds
.\start.ps1 measure --workers 2

# Process all parsed papers once
.\start.ps1 run --all --workers 2

# Preview the construction plan without model requests
.\start.ps1 run --all --dry-run
```

On macOS/Linux, replace `.\start.ps1` with `bash ./start.sh`. Measurement sessions are isolated under `outputs/measurement/<session>/`, including the dataset, round results, input hashes, and main log. Use `measure --repeats 1` for one evaluation round or `measure --paper-id ConVis` for one paper.

Use `--config path/to/config.env` to select a different configuration. If Windows blocks script execution, use `powershell -NoProfile -ExecutionPolicy Bypass -File .\start.ps1 run --paper-id ConVis` for that launch only. Individual `uv run python run.py ...` commands remain available below for development and stage-level control.

## Inputs and outputs

| Path | Purpose |
| --- | --- |
| `pdfInput/` | Optional input PDFs for OCR. |
| `rawPaper/<paper_id>/` | OCR pages, extracted images, and completion manifest. |
| `data/<paper_id>/` | Paper graph, multimodal assets, clean text, and graph-linked question package. |
| `outputs/evaluation/<model>/<paper_id>/` | New dialogue trajectories, reports, and evaluation checkpoints. |
| `outputs/logs/` | New pipeline logs. |
| `outputs/testing/<session>/` | Existing-dataset test rounds, statistics and isolated logs. |
| `outputs/measurement/<session>/` | Isolated measurement dataset, round results, manifests, and logs. |
| `eval_result/` | Historical paper evaluation results; not the destination for new runs. |

Original paper text and figures remain subject to their source licenses.

Released packages use repository-relative paths with forward slashes. Their master graphs contain the full KC bank, so evaluating a released package does not require unpublished graph-building caches.

## More ways to run

The unified entry point also supports individual construction stages and multi-model evaluation:

```powershell
uv run python run.py build --paper-id ConVis
uv run python run.py questions --paper-id ConVis
uv run python run.py evaluate --paper-id ConVis --models gpt-5-mini gpt-5
uv run python run.py run --all --workers 2 --models gpt-5-mini gpt-5
uv run python run.py check --stage build
```

`check --stage` accepts `parse`, `build`, `questions`, `evaluate`, or `all`. A requested stage fails when a required input or credential is missing; it does not silently switch to an offline or mock result. The lower-level scripts in `papergraph_demo/` remain available for development and ablations, while `run.py` is the supported end-to-end interface.

For the source tree, stage modules, and artifact contracts, see [Architecture](docs/architecture.md).

## Verification

```powershell
uv run python -m unittest discover -s tests -v
```

Tests use the released ConVis package, isolated temporary configurations, and a local HTTP provider simulator. They do not contact external model APIs or need real API keys.

## License

The code is released under the Apache License 2.0 license; see [LICENSE](LICENSE). Dataset artifacts and evaluation outputs are for research reproduction, subject to the copyright and redistribution terms of the original papers and figures.
