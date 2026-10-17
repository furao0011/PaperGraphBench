"""Public command-line entry point for the PaperGraph-Bench AI pipeline."""

from pathlib import Path
import sys


sys.path.insert(0, str(Path(__file__).resolve().parent / "papergraph_demo"))

from src.pipeline_cli import main


if __name__ == "__main__":
    raise SystemExit(main())
