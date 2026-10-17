# Pipeline architecture

PaperGraph-Bench has four ordered stages. The root `start.ps1` and `start.sh` launchers prepare dependencies and configuration. Their default is existing-dataset evaluation and statistics; an explicit `run` command invokes the construction pipeline through `run.py`. The implementation lives in `papergraph_demo/`.

Running `start.ps1` / `bash start.sh` without a command is equivalent to `test`: it evaluates new targets on all existing `data/` packages and exports paper-aligned statistics without construction. Sessions under `outputs/testing/` contain the plan, independent rounds, logs, and eleven statistics files. Start with `four_dimensions.csv` and `failure_category_by_model.csv`; `repair_tasks.csv` exposes actual repair groups and attempts. New evaluations need Judge and target API settings, but `scripts/summarize_results.py --result-root eval_result` computes the same outputs offline using only completed report/trajectory pairs, with no credentials, graph packages, final-state dependency, or state reconstruction.

Statistics are separated into report/trajectory validation, paper metric and repair-task aggregation, four-dimension presentation, trajectory-event classification, and export. The paper dimensions are initial macro-answer KC coverage and completion; ordinary-challenge failure overall/Text/MM (excluding thread challenges); mean events per paper and event-bearing-turn rate; and actual detail/hallucination repair success plus attempts until first success or termination. No final global KC coverage, state-name hallucination proxy, event-closure rate, or coverage-gap closure is substituted for these definitions.

The four-dimension table has eleven rows per model and explicit `value`/`aggregation` fields. Coverage, macro completion, mean events and mean total turns average repetitions within papers before weighting papers equally; challenges, event-bearing-turn rate/density, repair ratios/rounds and answer length use pooled count sums. Canonical summary values have the paper's metric names without a `_mean` suffix. New events come directly from each turn's `judge_result.hallucination_events`; duplicate event IDs within a task are errors rather than silently deduplicated. Detection snapshots are exported as `status_at_detection`, `followup_turns_at_detection`, and `resolved_at_turn_at_detection`, never as final state. Actual repair grouping uses `(repair_type, root_turn_id, root_question_id)` from follow-up records.

The paper categories are Contr.KC, False prem., Overclaim, and Wrong rel. Classification uses exact original `hallucination_type` mappings: review regressions merge into KC contradictions; thread premise mutations into false premises; thread overclaims into overclaims; wrong bridges and generic thread-reasoning hallucinations into wrong relations. Other labels, such as `fabricated_claim` and generic `challenge_failure`, are retained as unmapped events rather than guessed into a category. Category totals are raw event sums across completed runs, not paper-weighted metric means. Per-event `paper_category` and `paper_category_label` expose this projection while retaining the original family/type/subtype labels for debugging; those raw labels are not additional paper categories or proof of three semantic hierarchy levels.

Runtime evaluation can still write checkpoints and state caches for scheduling and resume. They are optional runtime artifacts, not offline analysis inputs. Completed report/trajectory identity, turn counts and IDs are checked; partial tasks remain separate and zero denominators are N/A. Offline regression and local mock-service tests validate software behavior without being presented as real new-model performance.

```text
PDF ──parse──> rawPaper/<paper_id>/
                  │
                  └──build──> data/<paper_id>/master_graph.json
                                    │
                                    └──questions──> data/<paper_id>/question_templates.json
                                                          │
                                                          └──evaluate──> outputs/evaluation/<model>/<paper_id>/
```

The `build` stage reads OCR text and extracted figure/table assets, creates macro-stage and knowledge-component (KC) objects, verifies reasoning edges, and writes a paper graph plus clean evaluation context. The `questions` stage creates macro coverage questions, text and multimodal challenge questions, and reasoning-thread seeds. Challenge candidates pass automatic usability checks and solver trials. Rejected candidates and review flags are saved as diagnostic outputs. The evaluator reads the graph and final question package, schedules multi-turn questions and follow-ups, judges answers, updates graph state, and writes a dialogue trajectory and report.

## Source tree

```text
config.example.env       API settings and standard defaults
start.ps1 / start.sh     Quick start and first-run configuration checks
run.py                   Public command entry point
scripts/                 Startup, measurement orchestration and dataset utilities
docs/                    Configuration and architecture
papergraph_demo/
  run_*.py               Stage implementations and optional experiments
  src/                   Graph, question, model-client and evaluation modules
  prompts/               Versioned model instructions and JSON schemas
rawPaper/                Parsed source papers and extracted images
data/                    Released/generated graph and question packages
outputs/                 New reports, checkpoints and logs (not committed)
eval_result/             Historical paper results
tests/                   Offline configuration, path and pipeline checks
```

The main source modules follow the pipeline:

| Stage | Modules in `papergraph_demo/src/` |
| --- | --- |
| Configuration and startup | `config.py`, `preflight.py`, `api_probe.py`, `pipeline_*.py` |
| Graph construction | `macro_extractor.py`, `kc_bank_builder.py`, `edge_verifier.py`, `graph_builder.py` |
| Multimodal evidence | `multimodal_explainer.py`, `multimodal_kc_extractor.py`, `multimodal_question_assets.py` |
| Questions and automatic filtering | `question_generator.py`, `challenge_plan_builder.py`, `challenge_loop.py`, `challenge_filter.py` |
| Dialogue evaluation | `eval_turn_runner.py`, `evaluation_stage_runner.py`, `judge.py`, `state_updater.py` |
| Artifacts and reporting | `artifact_layout.py`, `artifact_paths.py`, `reporter.py` |

## Stage boundaries

| Command | Reads | Produces |
| --- | --- | --- |
| `run.py run --pdf <path>` | A PDF | OCR output, graph, questions, evaluation result |
| `run.py run --paper-id <id>` | `rawPaper/<id>/` | Graph, questions, evaluation result |
| `run.py build --paper-id <id>` | Completed OCR output | `data/<id>/master_graph.json` and associated paper artifacts |
| `run.py questions --paper-id <id>` | Graph and associated paper artifacts | `data/<id>/question_templates.json` and question diagnostics |
| `run.py evaluate --paper-id <id>` | An existing package in `data/<id>/` | `outputs/evaluation/<model>/<id>/dialogue_trajectory.json` and `evaluation_report.json` |

`run --all --workers N` processes different papers concurrently but keeps stages ordered within each paper. `--models` selects explicit evaluation targets; without it, the configured `EVAL_TARGET_MODEL` is used. `--dry-run` prints the planned work without executing it.

## Artifact roles

`rawPaper/` contains OCR-derived source material. `data/` is the canonical graph-and-question package location for both released packages and newly built paper packages. `outputs/evaluation/` and `outputs/logs/` hold new runtime results; `eval_result/` is historical paper output. Per-paper and per-model isolation prevents simultaneous evaluations from sharing a checkpoint.

The lower-level `papergraph_demo/run_*.py` programs implement individual stages and research ablations. They are not required as separate manual steps for the standard workflow. For service requirements and safe preflight commands, see [Configuration](configuration.md).

`start.ps1 measure` / `bash start.sh measure` creates an isolated measurement session, snapshots source papers, builds graphs and questions once, freezes input hashes, and evaluates independent rounds. Its session `main.log` and `logs/` directories keep the batch and per-stage logs together. `PAPERGRAPH_LOG_ROOT` selects an alternative stage-log directory; ordinary runs default to `outputs/logs/`.
