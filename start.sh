#!/usr/bin/env bash
# One-command Unix entry. Quoted argument forwarding never evaluates user input.
set -euo pipefail
for argument in "$@"; do
  if [[ "$argument" == '--help' || "$argument" == '-h' ]]; then
    printf '%s\n' \
      'PaperGraph-Bench quick start:' \
      '  bash start.sh                         DEFAULT: all existing data + statistics' \
      '  bash start.sh --workers 2             Test existing data with two workers' \
      '  bash start.sh test --workers 2        Explicit equivalent of default test' \
      '  bash start.sh test --paper-id ConVis  One existing paper; no rebuilding' \
      '  bash start.sh run --paper-id ConVis   Explicit full pipeline for ConVis' \
      '  bash start.sh run --all --workers 2   Explicit pipeline for all parsed papers' \
      '  bash start.sh evaluate --all          Evaluate existing data packages' \
      '  bash start.sh measure --workers 2     Build and run repeated measurements' \
      '  bash start.sh --config private.env    Select another configuration file' \
      '  bash start.sh run --pdf paper.pdf     Include PDF OCR (additional API)' \
      'test defaults to all data/ papers, one round; --repeats N repeats evaluation.' \
      'No command (or options only) selects test; rebuilding requires an explicit command.' \
      'Testing needs Text/Judge + Target API settings, not construction APIs.' \
      '--dry-run makes no requests or configuration files; other commands match run.py.' \
      'First run creates config.env. You MUST fill your own API keys, BASE URLs,' \
      'and model IDs there before running; missing settings stop before API calls.' \
      'Requires uv: https://docs.astral.sh/uv/getting-started/installation/'
    exit 0
  fi
done
if ! command -v uv >/dev/null 2>&1; then
  printf '%s\n' 'uv is required. Install from https://docs.astral.sh/uv/getting-started/installation/ and rerun.' >&2
  exit 1
fi
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
exec uv run --locked python scripts/start.py "$@"
