# One-command Windows entry. All arguments are forwarded without shell evaluation.
$ErrorActionPreference = 'Stop'
if (($args -contains '--help') -or ($args -contains '-h')) {
    Write-Output @'
PaperGraph-Bench quick start (run from any working directory):
  .\start.ps1                         Full pipeline for the public ConVis example
  .\start.ps1 --all --workers 2       Full pipeline for all parsed papers
  .\start.ps1 evaluate --all          Evaluate existing data packages
  .\start.ps1 measure --workers 2     Build packages and run repeated measurements
  .\start.ps1 --config private.env    Select another configuration file
  .\start.ps1 run --pdf paper.pdf     Include PDF OCR (additional API required)
Arguments are the same as run.py. Use --dry-run to print a plan without requests.
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
