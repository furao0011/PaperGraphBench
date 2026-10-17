# One-command Windows entry. All arguments are forwarded without shell evaluation.
$ErrorActionPreference = 'Stop'
if (($args -contains '--help') -or ($args -contains '-h')) {
    Write-Output @'
PaperGraph-Bench quick start (run from any working directory):
  .\start.ps1                         DEFAULT: test all existing data + export statistics
  .\start.ps1 --workers 2             Test existing data with two concurrent papers
  .\start.ps1 test --workers 2        Explicit equivalent of the default test command
  .\start.ps1 test --paper-id ConVis  Test one existing paper; never rebuild packages
  .\start.ps1 run --paper-id ConVis   Explicitly build, generate questions and evaluate
  .\start.ps1 run --all --workers 2   Explicit full pipeline for all parsed papers
  .\start.ps1 evaluate --all          Evaluate existing data packages
  .\start.ps1 measure --workers 2     Build packages and run repeated measurements
  .\start.ps1 --config private.env    Select another configuration file
  .\start.ps1 run --pdf paper.pdf     Include PDF OCR (additional API required)
test defaults to all data/ papers, one round; use --repeats N for independent runs.
No command (or options only) selects test. Rebuilding requires run/build/questions/measure.
Testing needs Text/Judge and Target API credentials; no construction API required.
Use --dry-run to print a plan without requests. Other commands match run.py.
The first run creates config.env. You MUST provide your own API keys, BASE URLs,
and model IDs there before running; missing settings stop startup before API calls.
Requires uv: https://docs.astral.sh/uv/getting-started/installation/
'@
    exit 0
}
if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
    [Console]::Error.WriteLine('uv is required. Install it, then rerun this script:')
    [Console]::Error.WriteLine('https://docs.astral.sh/uv/getting-started/installation/')
    exit 1
}
$ExitCode = 1
Push-Location -LiteralPath $PSScriptRoot
try {
    & uv run --locked python scripts/start.py @args
    $ExitCode = $LASTEXITCODE
}
finally {
    Pop-Location
}
exit $ExitCode
