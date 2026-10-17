"""Configure once, then launch the existing pipeline without setup commands."""

from __future__ import annotations

import os
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "papergraph_demo"))

from src.config import load_environment
from src.pipeline_cli import main as pipeline_main, parser


def startup_arguments(argv: list[str]) -> tuple[list[str], object]:
    """Default to the public ConVis example; retain the existing CLI grammar."""
    cli = parser()
    commands = next(action.choices for action in cli._actions if action.dest == "command")
    tokens = list(argv)
    index = 0
    while index < len(tokens):
        if tokens[index] == "--config":
            index += 2
        elif tokens[index].startswith("--config="):
            index += 1
        else:
            break
    if index < len(tokens) and tokens[index] in {"measure", "test"}:
        command = tokens.pop(index)
        if command == "test":
            from evaluate_dataset import parser as command_parser
        else:
            from run_measurement import parser as command_parser
        args = command_parser().parse_args(tokens)
        args.command = command
        return tokens, args
    if index == len(tokens) or tokens[index] not in commands:
        tokens.insert(index, "run")
    args = cli.parse_args(tokens)
    if args.command in {"run", "build", "questions", "evaluate"}:
        if not (args.paper_id or args.all or args.pdf or args.input):
            tokens.extend(["--paper-id", "ConVis"])
            args.paper_id = "ConVis"
    return tokens, args


def configuration_path(config: str | None) -> Path:
    path = Path(config or os.getenv("PAPERGRAPH_CONFIG", "").strip() or "config.env").expanduser()
    return (path if path.is_absolute() else ROOT / path).resolve()


def create_configuration(path: Path) -> bool:
    """Create a blank private file, never overwrite an existing configuration."""
    if path.exists():
        if not path.is_file():
            raise ValueError(f"Configuration path is not a file: {path}")
        return False
    template = (ROOT / "config.example.env").read_text(encoding="utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as destination:
        destination.write(template)
    return True


def print_requirements(args, path: Path) -> None:
    print(f"\n{'=' * 72}\nREQUIRED BEFORE RUNNING: provide YOUR API credentials and model IDs.")
    print(f"Configuration file: {path}")
    print("The repository does NOT include usable API keys, service URLs or models.")
    stages = {
        "run": ["text", "vision", "embedding", "target"],
        "measure": ["text", "vision", "embedding", "target"],
        "build": ["text", "vision", "embedding"],
        "questions": ["text", "vision"],
        "evaluate": ["text", "target"],
        "test": ["text", "target"],
        "parse": [],
        "check": {
            "all": ["text", "vision", "embedding", "target"],
            "build": ["text", "vision", "embedding"],
            "questions": ["text", "vision"],
            "evaluate": ["text", "target"],
            "parse": [],
        }.get(getattr(args, "stage", "all"), []),
    }.get(args.command, [])
    required = {
        "text": "Text / judge: API_KEY + BASE_URL + LLM_MODEL",
        "vision": "Vision:       VISION_API_KEY + VISION_BASE_URL + VISION_MODEL",
        "embedding": "Embedding:    EMBED_API_KEY + EMBED_BASE_URL + EMBED_MODEL",
        "target": "Target:       EVAL_TARGET_API_KEY + EVAL_TARGET_BASE_URL + EVAL_TARGET_MODEL",
    }
    for name in stages:
        print(f"  {required[name]}")
    if args.command in {"evaluate", "test"} or (args.command == "check" and args.stage == "evaluate"):
        print("  Embedding is additionally required only if CLAIM_VERIFY_ENABLED=true.")
    if args.command == "parse" or getattr(args, "pdf", None) or (args.command == "check" and args.stage == "parse"):
        print("  PDF OCR: PADDLE_OCR_API_URL + PADDLE_OCR_TOKEN")
    print("Use OpenAI-compatible API BASE URLs, not /chat/completions endpoints.")
    print("One provider may supply multiple roles; fill each required section.")
    print("Existing environment variables override this file. Key values are never printed.")
    print("Startup validates configuration locally before any paid model request.")
    print(f"{'=' * 72}\n", flush=True)


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if "--help" in argv or "-h" in argv:
        print("Quick start: start.ps1 / start.sh [test|run|measure|evaluate|build|questions] [options]")
        print("RECOMMENDED FOR TESTING: start.ps1 test --workers 2 (existing data + automatic statistics)")
        print("No command defaults to: run --paper-id ConVis")
        print("Examples: --all --workers 2 | evaluate --all | measure --workers 2 | --config private.env")
        print("First run creates config.env; fill your own API keys, BASE URLs and model IDs, then rerun.\n")
        if "test" in argv:
            from evaluate_dataset import parser as test_parser
            test_parser().print_help()
        elif "measure" in argv:
            from run_measurement import parser as measurement_parser
            measurement_parser().print_help()
        else:
            parser().print_help()
        return 0
    tokens, args = startup_arguments(argv)
    if args.command == "test":
        from evaluate_dataset import main as command_main
    elif args.command == "measure":
        from run_measurement import main as command_main
    else:
        command_main = pipeline_main
    if args.command == "init":
        return pipeline_main(tokens)
    path = configuration_path(args.config)
    print_requirements(args, path)
    try:
        if getattr(args, "dry_run", False):
            print("Dry run: no configuration file will be created and no APIs will be called.")
            return command_main(tokens)
        if create_configuration(path):
            print(f"Created blank private configuration: {path}\nFill the required fields above and rerun the same command.", flush=True)
        os.environ["PAPERGRAPH_CONFIG"] = str(path)
        load_environment(ROOT, path)
        if args.command == "check":
            return pipeline_main(tokens)
        stage = {"run": "all", "measure": "all", "test": "evaluate", "build": "build", "questions": "questions", "evaluate": "evaluate", "parse": "parse"}[args.command]
        check = ["check", "--stage", stage]
        if getattr(args, "models", None) and args.command in {"run", "measure", "test", "evaluate"}:
            check.extend(["--models", *args.models])
        status = pipeline_main(check)
        if status:
            return status
        if args.command == "run" and args.pdf:
            status = pipeline_main(["check", "--stage", "parse"])
            if status:
                return status
        return command_main(tokens)
    except (OSError, ValueError) as exc:
        print(f"Startup failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
