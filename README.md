# PaperGraph-Bench

[English](README.md) | [简体中文](README.zh-CN.md)

PaperGraph-Bench constructs a graph-guided benchmark from scientific papers and evaluates a target model through multi-turn reading tasks. The standard pipeline parses a PDF when needed, builds a paper graph, generates graph-linked questions, and records dialogue-level coverage, challenge, hallucination, and repair results.

All standard stages run automatically, including graph verification, question validation, solver filtering, and dialogue judging.

For test operators, see the [Existing-dataset evaluation and statistics plan (Chinese)](docs/evaluation_plan.zh-CN.md) for the procedure, metric definitions, and acceptance criteria.

## Recommended: evaluate existing datasets and generate statistics

> **Running `start.ps1` / `start.sh` without a command evaluates the 12 released paper packages in `data/` and exports statistics. It does not rebuild graphs, regenerate questions, or run OCR. The explicit `test` command does the same.**

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
| `statistics/summary.csv` | Main model-level metrics, standard deviations, valid-paper counts and completeness statistics; start here |
| `statistics/per_paper.csv` | Per-round, per-paper metrics, numerators, denominators and completion status |
| `statistics/summary.json` | Machine-readable statistics |
| `round-01/evaluation/<model>/<paper_id>/` | Per-task `evaluation_report.json`, `dialogue_trajectory.json` and state cache |
| `testing.json`, `main.log`, `logs/` | Planned task manifest, main log and stage logs |

Statistics cover KC coverage, macro completion, challenge failure, hallucination-turn proportion, event density, event closure and omission completion. Main means weight papers equally. Zero denominators are blank/N/A; missing or incomplete tasks are identified rather than scored as zero. Repeated runs are averaged within each paper before averaging across papers.

Existing reports can also be summarized separately, **without model API calls**:

```powershell
uv run --locked python scripts/summarize_results.py --result-root outputs/evaluation
```

Replace `--result-root` with a test session or historical result root. Without a planned-task manifest, statistics describe the reports found and do not assert full dataset completion.

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
