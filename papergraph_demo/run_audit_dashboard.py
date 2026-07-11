from __future__ import annotations

import argparse
import json
import mimetypes
import os
import re
import threading
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any


BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent
DEFAULT_DATA_ROOT = PROJECT_ROOT / "EMNLP2026" / "data"
DEFAULT_EVAL_ROOT = PROJECT_ROOT / "eval_result"
DEFAULT_ANNOTATION_PATH = PROJECT_ROOT / "EMNLP2026" / "audit_annotations.json"


def main() -> None:
    args = _parse_args()
    app = AuditApp(
        data_root=Path(args.data_root),
        eval_root=Path(args.eval_root),
        annotation_path=Path(args.annotation_path),
    )
    server = ThreadingHTTPServer((args.host, args.port), _handler(app))
    url = f"http://{args.host}:{server.server_port}"
    print(f"[audit] serving {url}", flush=True)
    print(f"[audit] data_root={app.data_root}", flush=True)
    print(f"[audit] eval_root={app.eval_root}", flush=True)
    print(f"[audit] annotations={app.annotation_path}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[audit] stopped", flush=True)
    finally:
        server.server_close()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Serve a local PaperGraph audit dashboard.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--data-root", default=str(DEFAULT_DATA_ROOT))
    parser.add_argument("--eval-root", default=str(DEFAULT_EVAL_ROOT))
    parser.add_argument("--annotation-path", default=str(DEFAULT_ANNOTATION_PATH))
    return parser.parse_args()


class AuditApp:
    def __init__(self, data_root: Path, eval_root: Path, annotation_path: Path) -> None:
        self.data_root = data_root.resolve()
        self.eval_root = eval_root.resolve()
        self.annotation_path = annotation_path.resolve()
        self.lock = threading.Lock()

    def config(self) -> dict:
        return {
            "data_root": str(self.data_root),
            "eval_root": str(self.eval_root),
            "annotation_path": str(self.annotation_path),
        }

    def update_config(self, payload: dict) -> dict:
        data_root = Path(str(payload.get("data_root") or self.data_root)).resolve()
        eval_root = Path(str(payload.get("eval_root") or self.eval_root)).resolve()
        annotation_path = Path(str(payload.get("annotation_path") or self.annotation_path)).resolve()
        if not data_root.exists() or not data_root.is_dir():
            raise FileNotFoundError(f"data_root does not exist or is not a directory: {data_root}")
        if not eval_root.exists() or not eval_root.is_dir():
            raise FileNotFoundError(f"eval_root does not exist or is not a directory: {eval_root}")
        with self.lock:
            self.data_root = data_root
            self.eval_root = eval_root
            self.annotation_path = annotation_path
        return self.config()

    def overview(self) -> dict:
        papers = self._papers()
        models = self._models()
        annotations = self._load_annotations()
        paper_rows = []
        for paper_id in papers:
            package_counts = self._package_counts(paper_id)
            model_count = sum(1 for model in models if self._result_paths(model, paper_id)[0].exists())
            paper_rows.append(
                {
                    "paper_id": paper_id,
                    "models_with_results": model_count,
                    "package_counts": package_counts,
                    "annotation_counts": _annotation_counts_for_paper(annotations, paper_id),
                }
            )
        model_rows = []
        for model in models:
            available = [paper_id for paper_id in papers if self._result_paths(model, paper_id)[0].exists()]
            model_rows.append({"model": model, "paper_count": len(available)})
        return {
            "data_root": str(self.data_root),
            "eval_root": str(self.eval_root),
            "annotation_path": str(self.annotation_path),
            "papers": paper_rows,
            "models": model_rows,
            "totals": {
                "paper_count": len(papers),
                "model_count": len(models),
                "annotation_count": len(annotations.get("items", {})),
                "judge_annotation_count": len(annotations.get("judge_items", {})),
            },
        }

    def paper(self, paper_id: str) -> dict:
        root = self._paper_root(paper_id)
        graph = _read_json(root / "master_graph.json")
        questions = _read_json(root / "question_templates.json")
        assets = _read_json(root / "multimodal_assets.json", default={"assets": []})
        annotations = self._load_annotations()
        by_kc = {kc.get("kc_id"): kc for kc in graph.get("kc_nodes", []) if kc.get("kc_id")}
        packages = (
            self._macro_packages(paper_id, questions, by_kc, annotations)
            + self._challenge_packages(paper_id, questions, by_kc, annotations, "challenge_questions")
            + self._challenge_packages(paper_id, questions, by_kc, annotations, "thread_challenge_questions")
            + self._thread_packages(paper_id, questions, by_kc, annotations)
        )
        return {
            "paper_id": paper_id,
            "paper_title": graph.get("paper_title") or paper_id,
            "counts": {
                "macro_nodes": len(graph.get("macro_nodes", [])),
                "kc_nodes": len(graph.get("kc_nodes", [])),
                "reasoning_edges": len(graph.get("reasoning_edges", [])),
                "reasoning_threads": len(graph.get("reasoning_threads", [])),
                "assets": len(assets.get("assets", [])),
                "packages": len(packages),
            },
            "packages": packages,
            "assets": _asset_summaries(assets),
        }

    def package_detail(self, paper_id: str, package_id: str) -> dict:
        paper = self.paper(paper_id)
        root = self._paper_root(paper_id)
        graph = _read_json(root / "master_graph.json")
        questions = _read_json(root / "question_templates.json")
        by_kc = {kc.get("kc_id"): kc for kc in graph.get("kc_nodes", []) if kc.get("kc_id")}
        question = _find_question(questions, package_id)
        if question is None:
            raise KeyError(f"Unknown package/question id: {package_id}")
        target_kcs = [by_kc[kc_id] for kc_id in question.get("target_kc_ids", []) if kc_id in by_kc]
        annotation = self._load_annotations().get("items", {}).get(_annotation_key(paper_id, package_id), {})
        return {
            "paper_id": paper_id,
            "package": next((p for p in paper["packages"] if p["package_id"] == package_id), {}),
            "question": question,
            "target_kcs": [_kc_detail(kc) for kc in target_kcs],
            "asset_references": question.get("asset_references", []),
            "checklist_schema": _package_checklist_schema(
                next((p for p in paper["packages"] if p["package_id"] == package_id), {}).get("package_type"),
                question,
            ),
            "annotation": annotation,
        }

    def result(self, model: str, paper_id: str) -> dict:
        report_path, trajectory_path = self._result_paths(model, paper_id)
        report = _read_json(report_path)
        trajectory = _read_json(trajectory_path)
        turns = trajectory.get("turns", [])
        return {
            "model": model,
            "paper_id": paper_id,
            "report": report,
            "turns": [_turn_summary(turn) for turn in turns],
            "turn_count": len(turns),
        }

    def turn_detail(self, model: str, paper_id: str, turn_id: str) -> dict:
        _, trajectory_path = self._result_paths(model, paper_id)
        trajectory = _read_json(trajectory_path)
        for turn in trajectory.get("turns", []):
            if str(turn.get("turn_id")) == turn_id or str(turn.get("question_id")) == turn_id:
                return turn
        raise KeyError(f"Unknown turn: {turn_id}")

    def save_annotation(self, payload: dict) -> dict:
        paper_id = str(payload.get("paper_id") or "").strip()
        package_id = str(payload.get("package_id") or payload.get("question_id") or "").strip()
        if not paper_id or not package_id:
            raise ValueError("Annotation requires paper_id and package_id.")
        status = _normalize_package_status(payload.get("status"))
        item = {
            "paper_id": paper_id,
            "package_id": package_id,
            "status": status,
            "checklist": _normalize_checklist(payload.get("checklist", {})),
            "error_types": _list_strings(payload.get("error_types", [])),
            "notes": str(payload.get("notes") or "").strip(),
            "reviewer": str(payload.get("reviewer") or "").strip(),
            "updated_at": _nowish(),
        }
        with self.lock:
            data = self._load_annotations()
            data.setdefault("items", {})[_annotation_key(paper_id, package_id)] = item
            self._write_annotations(data)
        return item

    def calibration_sample(self, per_bucket: int = 8) -> dict:
        per_bucket = max(1, min(int(per_bucket or 8), 50))
        annotations = self._load_annotations()
        buckets: dict[str, list[dict]] = {}
        for model in self._models():
            for paper_id in self._papers():
                _, trajectory_path = self._result_paths(model, paper_id)
                if not trajectory_path.exists():
                    continue
                trajectory = _read_json(trajectory_path, default={"turns": []})
                for turn in trajectory.get("turns", []):
                    item = _calibration_item(model, paper_id, turn, annotations)
                    if item:
                        buckets.setdefault(item["bucket"], []).append(item)
        selected = []
        bucket_counts = {}
        for bucket in sorted(buckets):
            items = _balanced_bucket_items(buckets[bucket], per_bucket)
            bucket_counts[bucket] = {"available": len(buckets[bucket]), "selected": len(items)}
            selected.extend(items)
        selected.sort(key=lambda item: (item["bucket"], item["model"], item["paper_id"], item["turn_number"]))
        return {
            "per_bucket": per_bucket,
            "bucket_counts": bucket_counts,
            "items": selected,
            "total_selected": len(selected),
        }

    def calibration_turn_detail(self, model: str, paper_id: str, turn_id: str) -> dict:
        turn = self.turn_detail(model, paper_id, turn_id)
        graph = _read_json(self._paper_root(paper_id) / "master_graph.json")
        by_kc = {kc.get("kc_id"): kc for kc in graph.get("kc_nodes", []) if kc.get("kc_id")}
        target_kcs = [by_kc[kc_id] for kc_id in turn.get("target_kc_ids", []) if kc_id in by_kc]
        annotation = self._load_annotations().get("judge_items", {}).get(_judge_annotation_key(model, paper_id, turn_id), {})
        return {
            "model": model,
            "paper_id": paper_id,
            "turn": turn,
            "target_kcs": [_kc_detail(kc) for kc in target_kcs],
            "annotation": annotation,
        }

    def save_judge_annotation(self, payload: dict) -> dict:
        model = str(payload.get("model") or "").strip()
        paper_id = str(payload.get("paper_id") or "").strip()
        turn_id = str(payload.get("turn_id") or "").strip()
        if not model or not paper_id or not turn_id:
            raise ValueError("Judge annotation requires model, paper_id, and turn_id.")
        item = {
            "model": model,
            "paper_id": paper_id,
            "turn_id": turn_id,
            "question_id": str(payload.get("question_id") or "").strip(),
            "covered_kc_ids": _list_strings(payload.get("covered_kc_ids", [])),
            "challenge_outcome": _enum_or_na(payload.get("challenge_outcome"), {"pass", "fail", "incomplete", "na"}),
            "hallucination_present": _enum_or_na(payload.get("hallucination_present"), {"yes", "no", "unclear", "na"}),
            "hallucination_type": _enum_or_na(
                payload.get("hallucination_type"),
                {"false_premise", "overclaim", "wrong_relation", "contradicted_kc", "fabricated_claim", "other", "none", "na"},
            ),
            "repair_success": _enum_or_na(payload.get("repair_success"), {"success", "fail", "unclear", "na"}),
            "notes": str(payload.get("notes") or "").strip(),
            "reviewer": str(payload.get("reviewer") or "").strip(),
            "updated_at": _nowish(),
        }
        with self.lock:
            data = self._load_annotations()
            data.setdefault("judge_items", {})[_judge_annotation_key(model, paper_id, turn_id)] = item
            self._write_annotations(data)
        return item

    def asset(self, raw_path: str) -> tuple[bytes, str]:
        path = Path(raw_path)
        if not path.is_absolute():
            path = (PROJECT_ROOT / raw_path).resolve()
        else:
            path = path.resolve()
        if not path.exists() or not path.is_file():
            raise FileNotFoundError(str(path))
        mime = mimetypes.guess_type(str(path))[0] or "application/octet-stream"
        if not mime.startswith("image/"):
            raise ValueError(f"Only image assets can be served, got {mime}.")
        return path.read_bytes(), mime

    def _papers(self) -> list[str]:
        if not self.data_root.exists():
            return []
        return sorted(
            path.name
            for path in self.data_root.iterdir()
            if path.is_dir() and (path / "question_templates.json").exists() and (path / "master_graph.json").exists()
        )

    def _models(self) -> list[str]:
        if not self.eval_root.exists():
            return []
        return sorted(path.name for path in self.eval_root.iterdir() if path.is_dir())

    def _paper_root(self, paper_id: str) -> Path:
        root = (self.data_root / paper_id).resolve()
        if not _is_relative_to(root, self.data_root):
            raise ValueError("Invalid paper_id.")
        if not root.exists():
            raise FileNotFoundError(f"Unknown paper_id: {paper_id}")
        return root

    def _result_paths(self, model: str, paper_id: str) -> tuple[Path, Path]:
        root = (self.eval_root / model / paper_id).resolve()
        if not _is_relative_to(root, self.eval_root):
            raise ValueError("Invalid model or paper_id.")
        return root / "evaluation_report.json", root / "dialogue_trajectory.json"

    def _package_counts(self, paper_id: str) -> dict:
        root = self._paper_root(paper_id)
        questions = _read_json(root / "question_templates.json")
        challenges = questions.get("challenge_questions", [])
        thread_challenges = questions.get("thread_challenge_questions", [])
        mm_challenges = [q for q in challenges + thread_challenges if _requires_mm(q)]
        return {
            "macro": len(questions.get("macro_main_questions", [])),
            "challenge": len(challenges),
            "thread_challenge": len(thread_challenges),
            "thread_seed": len(questions.get("thread_question_seeds", [])),
            "mm_challenge": len(mm_challenges),
        }

    def _load_annotations(self) -> dict:
        if not self.annotation_path.exists():
            return {"schema_version": "audit_annotations.v2", "items": {}, "judge_items": {}}
        data = _read_json(self.annotation_path, default={"schema_version": "audit_annotations.v2", "items": {}, "judge_items": {}})
        data.setdefault("schema_version", "audit_annotations.v2")
        data.setdefault("items", {})
        data.setdefault("judge_items", {})
        return data

    def _write_annotations(self, data: dict) -> None:
        self.annotation_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.annotation_path.with_suffix(self.annotation_path.suffix + ".tmp")
        tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        tmp.replace(self.annotation_path)

    def _macro_packages(self, paper_id: str, questions: dict, by_kc: dict, annotations: dict) -> list[dict]:
        out = []
        for question in questions.get("macro_main_questions", []):
            out.append(_package_summary(paper_id, question, "Macro-KC", by_kc, annotations))
        return out

    def _challenge_packages(self, paper_id: str, questions: dict, by_kc: dict, annotations: dict, key: str) -> list[dict]:
        label = "Thread Challenge" if key == "thread_challenge_questions" else "Challenge"
        out = []
        for question in questions.get(key, []):
            out.append(_package_summary(paper_id, question, label, by_kc, annotations))
        return out

    def _thread_packages(self, paper_id: str, questions: dict, by_kc: dict, annotations: dict) -> list[dict]:
        out = []
        for question in questions.get("thread_question_seeds", []):
            out.append(_package_summary(paper_id, question, "Thread Seed", by_kc, annotations))
        return out


def _handler(app: AuditApp):
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            try:
                parsed = urllib.parse.urlparse(self.path)
                query = urllib.parse.parse_qs(parsed.query)
                if parsed.path == "/":
                    self._send_html(INDEX_HTML)
                elif parsed.path == "/api/config":
                    self._send_json(app.config())
                elif parsed.path == "/api/overview":
                    self._send_json(app.overview())
                elif parsed.path == "/api/paper":
                    self._send_json(app.paper(_one(query, "paper_id")))
                elif parsed.path == "/api/package":
                    self._send_json(app.package_detail(_one(query, "paper_id"), _one(query, "package_id")))
                elif parsed.path == "/api/result":
                    self._send_json(app.result(_one(query, "model"), _one(query, "paper_id")))
                elif parsed.path == "/api/turn":
                    self._send_json(app.turn_detail(_one(query, "model"), _one(query, "paper_id"), _one(query, "turn_id")))
                elif parsed.path == "/api/judge_sample":
                    per_bucket = int((query.get("per_bucket") or ["8"])[0] or "8")
                    self._send_json(app.calibration_sample(per_bucket=per_bucket))
                elif parsed.path == "/api/judge_turn":
                    self._send_json(app.calibration_turn_detail(_one(query, "model"), _one(query, "paper_id"), _one(query, "turn_id")))
                elif parsed.path == "/asset":
                    blob, mime = app.asset(_one(query, "path"))
                    self._send_bytes(blob, mime)
                else:
                    self.send_error(404, "Not found")
            except Exception as exc:
                self._send_json({"error": f"{type(exc).__name__}: {exc}"}, status=500)

        def do_POST(self) -> None:
            try:
                parsed = urllib.parse.urlparse(self.path)
                if parsed.path not in {"/api/annotation", "/api/judge_annotation", "/api/config"}:
                    self.send_error(404, "Not found")
                    return
                length = int(self.headers.get("Content-Length", "0") or "0")
                payload = json.loads(self.rfile.read(length).decode("utf-8") or "{}")
                if parsed.path == "/api/annotation":
                    self._send_json(app.save_annotation(payload))
                elif parsed.path == "/api/judge_annotation":
                    self._send_json(app.save_judge_annotation(payload))
                else:
                    self._send_json(app.update_config(payload))
            except Exception as exc:
                self._send_json({"error": f"{type(exc).__name__}: {exc}"}, status=500)

        def log_message(self, fmt: str, *args: Any) -> None:
            return

        def _send_json(self, payload: Any, status: int = 200) -> None:
            blob = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(blob)))
            self.end_headers()
            self.wfile.write(blob)

        def _send_html(self, html: str) -> None:
            blob = html.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(blob)))
            self.end_headers()
            self.wfile.write(blob)

        def _send_bytes(self, blob: bytes, mime: str) -> None:
            self.send_response(200)
            self.send_header("Content-Type", mime)
            self.send_header("Content-Length", str(len(blob)))
            self.end_headers()
            self.wfile.write(blob)

    return Handler


def _package_summary(paper_id: str, question: dict, package_type: str, by_kc: dict, annotations: dict) -> dict:
    target_ids = _list_strings(question.get("target_kc_ids", []))
    target_kcs = [by_kc[kc_id] for kc_id in target_ids if kc_id in by_kc]
    annotation = annotations.get("items", {}).get(_annotation_key(paper_id, question.get("question_id")), {})
    return {
        "package_id": question.get("question_id"),
        "package_type": package_type,
        "question_type": question.get("question_type"),
        "macro_id": question.get("macro_id") or question.get("preferred_macro_id"),
        "thread_id": question.get("thread_id") or question.get("target_thread_id"),
        "challenge_type": question.get("challenge_type"),
        "target_failure_mode": question.get("target_failure_mode"),
        "requires_multimodal_input": _requires_mm(question),
        "target_kc_count": len(target_ids),
        "target_kc_preview": [_short(kc.get("full_claim") or kc.get("claim") or "", 120) for kc in target_kcs[:3]],
        "question_text": question.get("question_text") or question.get("question_goal") or "",
        "asset_ids": [ref.get("asset_id") for ref in question.get("asset_references", []) if ref.get("asset_id")],
        "annotation_status": _display_package_status(annotation.get("status", "unreviewed")),
        "annotation_key": _annotation_key(paper_id, question.get("question_id")) if paper_id else None,
    }


def _kc_detail(kc: dict) -> dict:
    return {
        "kc_id": kc.get("kc_id"),
        "macro_id": kc.get("macro_id"),
        "type": kc.get("type"),
        "full_claim": kc.get("full_claim"),
        "evidence": kc.get("evidence") or kc.get("evidence_spans") or [],
        "rubric": kc.get("rubric") or kc.get("grading_rubric") or {},
        "forbidden_claims": kc.get("forbidden_claims", []),
        "asset_id": kc.get("asset_id"),
        "asset_type": kc.get("asset_type"),
        "modality": kc.get("modality", {}),
    }


def _asset_summaries(assets_payload: dict) -> list[dict]:
    out = []
    for asset in assets_payload.get("assets", []):
        attachments = asset.get("attachments") or []
        out.append(
            {
                "asset_id": asset.get("asset_id"),
                "asset_type": asset.get("asset_type"),
                "caption": _short(asset.get("caption") or "", 180),
                "macro_id": asset.get("macro_id"),
                "image_paths": asset.get("image_paths", []),
                "attachment_count": len(attachments),
            }
        )
    return out


def _turn_summary(turn: dict) -> dict:
    judge = turn.get("judge_result", {}) or {}
    challenge = judge.get("challenge_result") or {}
    return {
        "turn_id": turn.get("turn_id"),
        "question_id": turn.get("question_id"),
        "question_type": turn.get("question_type"),
        "macro_id": turn.get("macro_id"),
        "challenge_type": turn.get("challenge_type"),
        "requires_multimodal_input": bool(turn.get("requires_multimodal_input")),
        "state": judge.get("state"),
        "covered": len(judge.get("covered_kc_ids", []) or []),
        "missing": len(judge.get("missing_kc_ids", []) or []),
        "challenge_failed": challenge.get("failed"),
        "challenge_resisted": challenge.get("resisted"),
        "hallucination_events": len(judge.get("hallucination_events", []) or []),
        "question_text": _short(turn.get("question_text") or "", 160),
    }


def _calibration_item(model: str, paper_id: str, turn: dict, annotations: dict) -> dict | None:
    bucket = _calibration_bucket(turn)
    if not bucket:
        return None
    judge = turn.get("judge_result", {}) or {}
    key = _judge_annotation_key(model, paper_id, turn.get("turn_id"))
    return {
        "key": key,
        "bucket": bucket,
        "model": model,
        "paper_id": paper_id,
        "turn_id": turn.get("turn_id"),
        "turn_number": _turn_number(turn.get("turn_id")),
        "question_id": turn.get("question_id"),
        "question_type": turn.get("question_type"),
        "requires_multimodal_input": bool(turn.get("requires_multimodal_input")),
        "state": judge.get("state"),
        "covered_count": len(judge.get("covered_kc_ids", []) or []),
        "missing_count": len(judge.get("missing_kc_ids", []) or []),
        "challenge_failed": (judge.get("challenge_result") or {}).get("failed"),
        "hallucination_events": len(judge.get("hallucination_events", []) or []),
        "question_text": _short(turn.get("question_text") or "", 180),
        "annotation_status": "done" if key in annotations.get("judge_items", {}) else "unreviewed",
    }


def _calibration_bucket(turn: dict) -> str | None:
    qtype = str(turn.get("question_type") or "")
    is_mm = bool(turn.get("requires_multimodal_input"))
    hallucination_events = turn.get("judge_result", {}).get("hallucination_events", []) or []
    if qtype in {"macro_main_question", "main"}:
        return "macro_mm" if is_mm else "macro_text"
    if qtype in {"challenge_question", "thread_challenge_question"}:
        return "challenge_mm" if is_mm else "challenge_text"
    if qtype == "hallucination_followup":
        return "hallucination_repair"
    if qtype == "detail_followup":
        return "detail_repair"
    if qtype.startswith("thread_") or qtype == "review_followup":
        return "thread_review"
    if hallucination_events:
        return "hallucination_event"
    return None


def _balanced_bucket_items(items: list[dict], limit: int) -> list[dict]:
    by_model: dict[str, list[dict]] = {}
    for item in sorted(items, key=lambda x: (x["model"], x["paper_id"], x["turn_number"])):
        by_model.setdefault(item["model"], []).append(item)
    selected = []
    while len(selected) < limit:
        progressed = False
        for model in sorted(by_model):
            if by_model[model] and len(selected) < limit:
                selected.append(by_model[model].pop(0))
                progressed = True
        if not progressed:
            break
    return selected


def _package_checklist_schema(package_type: str | None, question: dict) -> list[dict]:
    package_type = str(package_type or "")
    if package_type == "Macro-KC":
        checks = [
            ("question_targets_main_content", "问题确实在问论文主干内容"),
            ("target_kcs_are_core", "target KCs 是该问题应覆盖的核心点"),
            ("target_kcs_supported", "target KCs 有论文证据支持"),
        ]
    elif package_type in {"Challenge", "Thread Challenge"}:
        checks = [
            ("trap_is_real", "challenge 确实包含明确陷阱"),
            ("failure_mode_unsupported", "false premise / overclaim / wrong relation 不被论文支持"),
            ("expected_behavior_clear", "正确行为明确：reject / qualify / correct"),
        ]
    elif package_type == "Thread Seed":
        checks = [
            ("reasoning_relation_valid", "两个 KC/edge 之间是真 reasoning relation，不只是主题相关"),
        ]
    else:
        checks = []
    if _requires_mm(question):
        checks.extend(
            [
                ("modality_necessary", "问题确实需要看 figure/table"),
                ("asset_supports_expected_answer", "figure/table 支持 expected answer"),
            ]
        )
    return [{"id": key, "label": label, "options": ["yes", "no", "unclear"]} for key, label in checks]


def _find_question(questions: dict, question_id: str) -> dict | None:
    for key in ("macro_main_questions", "challenge_questions", "thread_challenge_questions", "thread_question_seeds"):
        for question in questions.get(key, []):
            if str(question.get("question_id")) == question_id:
                return question
    return None


def _annotation_counts_for_paper(annotations: dict, paper_id: str) -> dict:
    counts: dict[str, int] = {}
    prefix = paper_id + "::"
    for key, item in annotations.get("items", {}).items():
        if not key.startswith(prefix):
            continue
        status = item.get("status", "unreviewed")
        counts[status] = counts.get(status, 0) + 1
    return counts


def _annotation_key(paper_id: object, package_id: object) -> str:
    return f"{paper_id}::{package_id}"


def _judge_annotation_key(model: object, paper_id: object, turn_id: object) -> str:
    return f"{model}::{paper_id}::{turn_id}"


def _normalize_package_status(value: object) -> str:
    raw = str(value or "unreviewed").strip().lower()
    aliases = {"revised": "revise", "removed": "remove"}
    raw = aliases.get(raw, raw)
    if raw not in {"unreviewed", "valid", "revise", "remove"}:
        raise ValueError("status must be one of unreviewed, valid, revise, remove.")
    return raw


def _display_package_status(value: object) -> str:
    try:
        return _normalize_package_status(value)
    except ValueError:
        return "unreviewed"


def _normalize_checklist(value: object) -> dict:
    if not isinstance(value, dict):
        return {}
    out = {}
    for key, raw in value.items():
        item = str(raw or "").strip().lower()
        if item in {"yes", "no", "unclear"}:
            out[str(key)] = item
    return out


def _enum_or_na(value: object, allowed: set[str]) -> str:
    item = str(value or "na").strip().lower()
    if item not in allowed:
        raise ValueError(f"Invalid annotation value {item!r}; expected one of {sorted(allowed)}.")
    return item


def _turn_number(value: object) -> int:
    match = re.search(r"(\d+)$", str(value or ""))
    return int(match.group(1)) if match else 0


def _requires_mm(question: dict) -> bool:
    return bool(question.get("requires_multimodal_input") or question.get("asset_references"))


def _read_json(path: Path, default: Any | None = None) -> Any:
    if not path.exists():
        if default is not None:
            return default
        raise FileNotFoundError(str(path))
    return json.loads(path.read_text(encoding="utf-8"))


def _list_strings(values: Any) -> list[str]:
    if not isinstance(values, list):
        return []
    return [str(item) for item in values if str(item).strip()]


def _short(text: str, limit: int) -> str:
    text = re.sub(r"\s+", " ", str(text or "")).strip()
    return text if len(text) <= limit else text[: limit - 1] + "..."


def _one(query: dict[str, list[str]], name: str) -> str:
    value = (query.get(name) or [""])[0]
    value = urllib.parse.unquote(value)
    if not value:
        raise ValueError(f"Missing query parameter: {name}")
    return value


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def _nowish() -> str:
    from datetime import datetime

    return datetime.now().astimezone().isoformat(timespec="seconds")


INDEX_HTML = r"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>PaperGraph Audit</title>
  <style>
    :root {
      color-scheme: light;
      --bg: #f6f7f9;
      --panel: #ffffff;
      --line: #d8dde6;
      --text: #1d2430;
      --muted: #657083;
      --blue: #1f6feb;
      --green: #16833a;
      --red: #c4332f;
      --amber: #9a6700;
      --chip: #eef2f7;
    }
    * { box-sizing: border-box; }
    body { margin: 0; font-family: ui-sans-serif, system-ui, "Segoe UI", Arial, sans-serif; background: var(--bg); color: var(--text); }
    header { height: 56px; display: flex; align-items: center; gap: 16px; padding: 0 18px; background: #111827; color: #fff; }
    header h1 { font-size: 17px; margin: 0; font-weight: 650; }
    header .sub { color: #cbd5e1; font-size: 13px; }
    .layout { display: grid; grid-template-columns: 320px 1fr; min-height: calc(100vh - 56px); }
    aside { border-right: 1px solid var(--line); background: #fff; padding: 14px; overflow: auto; }
    main { padding: 14px; overflow: auto; }
    label { display: block; font-size: 12px; color: var(--muted); margin: 10px 0 6px; }
    select, input, textarea { width: 100%; border: 1px solid var(--line); border-radius: 6px; background: #fff; padding: 8px 10px; color: var(--text); }
    textarea { min-height: 74px; resize: vertical; }
    button { border: 1px solid var(--line); background: #fff; border-radius: 6px; padding: 8px 10px; cursor: pointer; }
    button.primary { background: var(--blue); border-color: var(--blue); color: #fff; }
    button:hover { filter: brightness(0.98); }
    .tabs { display: flex; gap: 8px; margin-bottom: 12px; }
    .tabs button.active { background: #111827; color: #fff; border-color: #111827; }
    .grid { display: grid; gap: 12px; }
    .grid.cols { grid-template-columns: repeat(4, minmax(0, 1fr)); }
    .panel { background: var(--panel); border: 1px solid var(--line); border-radius: 8px; padding: 12px; }
    .metric { font-size: 26px; font-weight: 720; }
    .muted { color: var(--muted); }
    .row { display: flex; gap: 8px; align-items: center; flex-wrap: wrap; }
    .split { display: grid; grid-template-columns: minmax(320px, 0.45fr) minmax(420px, 0.55fr); gap: 12px; }
    .list { max-height: 68vh; overflow: auto; display: grid; gap: 8px; }
    .item { border: 1px solid var(--line); border-radius: 7px; padding: 10px; background: #fff; cursor: pointer; }
    .item:hover, .item.selected { border-color: var(--blue); box-shadow: 0 0 0 1px rgba(31,111,235,.15); }
    .title { font-weight: 650; margin-bottom: 4px; }
    .chips { display: flex; gap: 6px; flex-wrap: wrap; margin: 6px 0; }
    .chip { padding: 2px 7px; border-radius: 999px; background: var(--chip); font-size: 12px; color: #2f3a4b; }
    .chip.green { background: #daf5e3; color: var(--green); }
    .chip.red { background: #fde2df; color: var(--red); }
    .chip.amber { background: #fff1c7; color: var(--amber); }
    .choice-grid { display: grid; gap: 8px; margin: 8px 0; }
    .choice-line { display: grid; grid-template-columns: minmax(180px, 1fr) auto; gap: 10px; align-items: center; padding: 8px; border: 1px solid var(--line); border-radius: 7px; background: #fbfcfe; }
    .choice-buttons { display: flex; gap: 4px; }
    .choice-buttons label { margin: 0; font-size: 12px; color: var(--text); }
    .choice-buttons input { width: auto; margin-right: 3px; }
    .detail h2 { margin: 4px 0 8px; font-size: 18px; }
    .detail h3 { margin: 14px 0 6px; font-size: 14px; }
    pre { white-space: pre-wrap; word-break: break-word; background: #0f172a; color: #dbeafe; border-radius: 7px; padding: 10px; max-height: 360px; overflow: auto; }
    .kc { border-left: 3px solid var(--blue); padding: 8px 10px; background: #f8fafc; margin: 8px 0; }
    .asset-img { max-width: 100%; max-height: 280px; border: 1px solid var(--line); border-radius: 7px; object-fit: contain; background: #fff; }
    .table { width: 100%; border-collapse: collapse; font-size: 13px; }
    .table th, .table td { border-bottom: 1px solid var(--line); padding: 7px; text-align: left; vertical-align: top; }
    .table th { color: var(--muted); font-weight: 600; background: #f8fafc; position: sticky; top: 0; }
    .hidden { display: none; }
    @media (max-width: 1000px) {
      .layout, .split, .grid.cols { grid-template-columns: 1fr; }
      aside { border-right: 0; border-bottom: 1px solid var(--line); }
    }
  </style>
</head>
<body>
<header>
  <h1>PaperGraph Audit</h1>
  <span class="sub">Evaluation-facing packages, trajectories, judge results</span>
</header>
<div class="layout">
  <aside>
    <label>Paper</label>
    <select id="paperSelect"></select>
    <label>Model</label>
    <select id="modelSelect"></select>
    <label>Package filter</label>
    <select id="typeFilter">
      <option value="">All packages</option>
      <option>Macro-KC</option>
      <option>Challenge</option>
      <option>Thread Challenge</option>
      <option>Thread Seed</option>
    </select>
    <label>Text search</label>
    <input id="searchBox" placeholder="question, KC, challenge type" />
    <div class="panel" style="margin-top:12px">
      <div class="title">Data Sources</div>
      <label>EMNLP2026 data root</label>
      <input id="dataRootInput" />
      <label>eval_result root</label>
      <input id="evalRootInput" />
      <label>annotation file</label>
      <input id="annPathInput" />
      <button class="primary" style="margin-top:8px" onclick="applyConfig()">Apply paths</button>
      <div id="roots" style="font-size:12px; word-break:break-all; margin-top:8px"></div>
    </div>
  </aside>
  <main>
    <div class="tabs">
      <button id="tabOverview" class="active">Overview</button>
      <button id="tabPackages">Packages</button>
      <button id="tabResults">Results</button>
      <button id="tabJudge">Judge</button>
    </div>
    <section id="viewOverview"></section>
    <section id="viewPackages" class="hidden"></section>
    <section id="viewResults" class="hidden"></section>
    <section id="viewJudge" class="hidden"></section>
  </main>
</div>
<script>
let state = { overview: null, paper: null, result: null, calibration: null, selectedPackage: null, selectedTurn: null, selectedJudgeKey: null, tab: 'overview' };
const $ = id => document.getElementById(id);
const enc = encodeURIComponent;

async function api(path) {
  const res = await fetch(path);
  const data = await res.json();
  if (!res.ok || data.error) throw new Error(data.error || res.statusText);
  return data;
}

async function post(path, payload) {
  const res = await fetch(path, { method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify(payload) });
  const data = await res.json();
  if (!res.ok || data.error) throw new Error(data.error || res.statusText);
  return data;
}

function chip(text, cls='') { return `<span class="chip ${cls}">${escapeHtml(text ?? '')}</span>`; }
function escapeHtml(s) { return String(s ?? '').replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c])); }
function pct(v) { return v == null ? 'n/a' : (v * 100).toFixed(1) + '%'; }

async function loadOverview() {
  state.overview = await api('/api/overview');
  $('paperSelect').innerHTML = state.overview.papers.map(p => `<option>${escapeHtml(p.paper_id)}</option>`).join('');
  $('modelSelect').innerHTML = state.overview.models.map(m => `<option>${escapeHtml(m.model)}</option>`).join('');
  $('dataRootInput').value = state.overview.data_root;
  $('evalRootInput').value = state.overview.eval_root;
  $('annPathInput').value = state.overview.annotation_path;
  $('roots').innerHTML = `<b>data</b>: ${escapeHtml(state.overview.data_root)}<br><b>eval</b>: ${escapeHtml(state.overview.eval_root)}<br><b>annotations</b>: ${escapeHtml(state.overview.annotation_path)}`;
  await loadPaper();
  await loadResult();
  await loadCalibration();
  render();
}

async function loadPaper() {
  const paper = $('paperSelect').value;
  if (!paper) return;
  state.paper = await api(`/api/paper?paper_id=${enc(paper)}`);
  state.selectedPackage = state.paper.packages[0]?.package_id || null;
}

async function loadResult() {
  const paper = $('paperSelect').value, model = $('modelSelect').value;
  if (!paper || !model) return;
  try {
    state.result = await api(`/api/result?paper_id=${enc(paper)}&model=${enc(model)}`);
    state.selectedTurn = state.result.turns[0]?.turn_id || null;
  } catch (err) {
    state.result = { error: err.message, turns: [], report: null };
  }
}

async function loadCalibration() {
  try {
    state.calibration = await api('/api/judge_sample?per_bucket=8');
    state.selectedJudgeKey = state.calibration.items[0]?.key || null;
  } catch (err) {
    state.calibration = { error: err.message, items: [] };
  }
}

async function applyConfig() {
  await post('/api/config', {
    data_root: $('dataRootInput').value,
    eval_root: $('evalRootInput').value,
    annotation_path: $('annPathInput').value
  });
  state = { overview: null, paper: null, result: null, calibration: null, selectedPackage: null, selectedTurn: null, selectedJudgeKey: null, tab: 'overview' };
  await loadOverview();
}

function render() {
  document.querySelectorAll('.tabs button').forEach(b => b.classList.remove('active'));
  $('tab' + cap(state.tab)).classList.add('active');
  ['Overview','Packages','Results','Judge'].forEach(name => $('view' + name).classList.toggle('hidden', state.tab !== name.toLowerCase()));
  renderOverview();
  renderPackages();
  renderResults();
  renderJudge();
}
function cap(s) { return s.charAt(0).toUpperCase() + s.slice(1); }

function renderOverview() {
  if (!state.overview || !state.paper) return;
  const p = state.paper;
  const rows = state.overview.papers.map(row => `
    <tr><td>${escapeHtml(row.paper_id)}</td><td>${row.models_with_results}</td><td>${row.package_counts.macro}</td><td>${row.package_counts.challenge}</td><td>${row.package_counts.mm_challenge}</td><td>${Object.entries(row.annotation_counts || {}).map(([k,v]) => `${k}:${v}`).join(', ')}</td></tr>
  `).join('');
  $('viewOverview').innerHTML = `
    <div class="grid cols">
      <div class="panel"><div class="muted">Papers</div><div class="metric">${state.overview.totals.paper_count}</div></div>
      <div class="panel"><div class="muted">Models</div><div class="metric">${state.overview.totals.model_count}</div></div>
      <div class="panel"><div class="muted">Current Packages</div><div class="metric">${p.counts.packages}</div></div>
      <div class="panel"><div class="muted">Annotations</div><div class="metric">${state.overview.totals.annotation_count}</div></div>
    </div>
    <div class="panel" style="margin-top:12px">
      <div class="title">Selected Paper: ${escapeHtml(p.paper_title)}</div>
      <div class="chips">
        ${chip('KC ' + p.counts.kc_nodes)} ${chip('edges ' + p.counts.reasoning_edges)} ${chip('threads ' + p.counts.reasoning_threads)} ${chip('assets ' + p.counts.assets)}
      </div>
    </div>
    <div class="panel" style="margin-top:12px; max-height:48vh; overflow:auto">
      <table class="table"><thead><tr><th>paper</th><th>models</th><th>macro</th><th>challenge</th><th>MM challenge</th><th>audit</th></tr></thead><tbody>${rows}</tbody></table>
    </div>
  `;
}

function filteredPackages() {
  if (!state.paper) return [];
  const typ = $('typeFilter').value;
  const q = $('searchBox').value.toLowerCase();
  return state.paper.packages.filter(p => {
    if (typ && p.package_type !== typ) return false;
    const hay = [p.package_id,p.package_type,p.question_text,p.challenge_type,p.target_failure_mode,(p.target_kc_preview||[]).join(' ')].join(' ').toLowerCase();
    return !q || hay.includes(q);
  });
}

async function selectPackage(id) {
  state.selectedPackage = id;
  renderPackages();
}

async function renderPackages() {
  if (!state.paper) return;
  const packages = filteredPackages();
  const items = packages.map(p => `
    <div class="item ${p.package_id===state.selectedPackage?'selected':''}" onclick="selectPackage('${escapeJs(p.package_id)}')">
      <div class="title">${escapeHtml(p.package_id)} · ${escapeHtml(p.package_type)}</div>
      <div class="muted">${escapeHtml(p.question_text)}</div>
      <div class="chips">
        ${chip('KC ' + p.target_kc_count)}
        ${p.requires_multimodal_input ? chip('MM','amber') : chip('text')}
        ${p.challenge_type ? chip(p.challenge_type) : ''}
        ${chip(p.annotation_status, p.annotation_status==='valid'?'green':p.annotation_status==='remove'?'red':p.annotation_status==='revise'?'amber':'')}
      </div>
    </div>
  `).join('');
  $('viewPackages').innerHTML = `
    <div class="split">
      <div class="panel"><div class="title">Packages (${packages.length})</div><div class="list">${items}</div></div>
      <div class="panel detail" id="packageDetail">Loading...</div>
    </div>`;
  if (state.selectedPackage) await renderPackageDetail();
}
function escapeJs(s) { return String(s ?? '').replace(/\\/g,'\\\\').replace(/'/g,"\\'"); }

async function renderPackageDetail() {
  const box = $('packageDetail');
  try {
    const d = await api(`/api/package?paper_id=${enc($('paperSelect').value)}&package_id=${enc(state.selectedPackage)}`);
    const q = d.question || {};
    const ann = d.annotation || {};
    const annStatus = normalizePackageStatus(ann.status);
    box.innerHTML = `
      <h2>${escapeHtml(state.selectedPackage)}</h2>
      <div class="chips">${chip(d.package.package_type)} ${q.requires_multimodal_input || q.asset_references?.length ? chip('multimodal','amber') : chip('text')}</div>
      <h3>Question</h3><div>${escapeHtml(q.question_text || q.question_goal || '')}</div>
      <h3>Audit Decision</h3>
      <div class="row">
        ${packageDecisionButton('valid', annStatus)}
        ${packageDecisionButton('revise', annStatus)}
        ${packageDecisionButton('remove', annStatus)}
      </div>
      <div class="muted" style="margin-top:8px">Current: ${escapeHtml(annStatus)}</div>
      <h3>Target KCs</h3>
      ${d.target_kcs.map(kc => `<div class="kc"><b>${escapeHtml(kc.kc_id)}</b> ${escapeHtml(kc.full_claim || '')}<br><span class="muted">${escapeHtml(JSON.stringify(kc.rubric || {}))}</span></div>`).join('') || '<div class="muted">No target KC details.</div>'}
      <h3>Assets</h3>
      ${renderAssets(d.asset_references || [])}
      <h3>Raw Question JSON</h3><pre>${escapeHtml(JSON.stringify(q, null, 2))}</pre>
    `;
  } catch (err) {
    box.innerHTML = `<div class="chip red">${escapeHtml(err.message)}</div>`;
  }
}

function packageDecisionButton(status, current) {
  const cls = status === current ? 'primary' : '';
  return `<button class="${cls}" onclick="savePackageDecision('${status}')">${escapeHtml(status)}</button>`;
}

function normalizePackageStatus(status) {
  if (status === 'revised') return 'revise';
  if (status === 'removed') return 'remove';
  return status || 'unreviewed';
}

function renderAssets(refs) {
  if (!refs.length) return '<div class="muted">No linked asset.</div>';
  return refs.map(ref => `
    <div class="panel">
      <b>${escapeHtml(ref.asset_id)}</b> ${escapeHtml(ref.asset_type || '')}
      <div class="muted">${escapeHtml(ref.caption || ref.summary || '')}</div>
      ${(ref.attachments||[]).map(a => a.type === 'image' && a.path ? `<img class="asset-img" src="/asset?path=${enc(a.path)}" onerror="this.replaceWith(document.createTextNode('image missing: ${escapeJs(a.path)}'))">` : `<pre>${escapeHtml(a.content || '')}</pre>`).join('')}
    </div>
  `).join('');
}

async function savePackageDecision(status) {
  const payload = {
    paper_id: $('paperSelect').value,
    package_id: state.selectedPackage,
    status
  };
  await post('/api/annotation', payload);
  await loadPaper();
  state.selectedPackage = payload.package_id;
  renderPackages();
}

function renderResults() {
  if (!state.result) return;
  if (state.result.error) {
    $('viewResults').innerHTML = `<div class="panel"><span class="chip red">${escapeHtml(state.result.error)}</span></div>`;
    return;
  }
  const report = state.result.report || {};
  const summary = report.summary || {};
  const challenge = report.challenge_metrics || report.thread_challenge_metrics || {};
  const turns = state.result.turns || [];
  const rows = turns.map(t => `
    <tr onclick="selectTurn('${escapeJs(t.turn_id)}')" style="cursor:pointer">
      <td>${escapeHtml(t.turn_id)}</td><td>${escapeHtml(t.question_type)}</td><td>${escapeHtml(t.state)}</td><td>${t.covered}/${t.missing}</td><td>${t.challenge_failed===true?'fail':t.challenge_resisted===true?'resist':''}</td><td>${t.hallucination_events}</td><td>${escapeHtml(t.question_text)}</td>
    </tr>
  `).join('');
  $('viewResults').innerHTML = `
    <div class="grid cols">
      <div class="panel"><div class="muted">Turns</div><div class="metric">${summary.total_turns ?? state.result.turn_count}</div></div>
      <div class="panel"><div class="muted">Status</div><div class="metric" style="font-size:20px">${escapeHtml(summary.evaluation_status || 'n/a')}</div></div>
      <div class="panel"><div class="muted">Challenge Fail</div><div class="metric">${escapeHtml(challenge.challenge_failures ?? challenge.failed_count ?? 'n/a')}</div></div>
      <div class="panel"><div class="muted">Failed</div><div class="metric">${summary.failed ? 'yes' : 'no'}</div></div>
    </div>
    <div class="split" style="margin-top:12px">
      <div class="panel" style="max-height:64vh; overflow:auto"><table class="table"><thead><tr><th>turn</th><th>type</th><th>state</th><th>cov/miss</th><th>challenge</th><th>hall.</th><th>question</th></tr></thead><tbody>${rows}</tbody></table></div>
      <div class="panel detail" id="turnDetail"><pre>${escapeHtml(JSON.stringify(report, null, 2))}</pre></div>
    </div>
  `;
}

async function selectTurn(turnId) {
  const box = $('turnDetail');
  box.innerHTML = 'Loading...';
  const t = await api(`/api/turn?paper_id=${enc($('paperSelect').value)}&model=${enc($('modelSelect').value)}&turn_id=${enc(turnId)}`);
  box.innerHTML = `<h2>${escapeHtml(turnId)}</h2><h3>Question</h3><div>${escapeHtml(t.question_text || '')}</div><h3>Answer</h3><div>${escapeHtml(t.model_answer || '')}</div><h3>Judge</h3><pre>${escapeHtml(JSON.stringify(t.judge_result || {}, null, 2))}</pre>`;
}

function renderJudge() {
  if (!state.calibration) return;
  if (state.calibration.error) {
    $('viewJudge').innerHTML = `<div class="panel"><span class="chip red">${escapeHtml(state.calibration.error)}</span></div>`;
    return;
  }
  const items = state.calibration.items || [];
  const buckets = Object.entries(state.calibration.bucket_counts || {}).map(([k,v]) => chip(`${k} ${v.selected}/${v.available}`)).join('');
  const list = items.map(item => `
    <div class="item ${item.key===state.selectedJudgeKey?'selected':''}" onclick="selectJudgeItem('${escapeJs(item.key)}')">
      <div class="title">${escapeHtml(item.bucket)} · ${escapeHtml(item.model)} · ${escapeHtml(item.turn_id)}</div>
      <div class="muted">${escapeHtml(item.paper_id)}</div>
      <div class="muted">${escapeHtml(item.question_text)}</div>
      <div class="chips">
        ${chip(item.question_type)}
        ${item.requires_multimodal_input ? chip('MM','amber') : chip('text')}
        ${chip(item.state || 'state?')}
        ${item.annotation_status === 'done' ? chip('annotated','green') : chip('unreviewed')}
      </div>
    </div>
  `).join('');
  $('viewJudge').innerHTML = `
    <div class="panel" style="margin-bottom:12px">
      <div class="row">
        <div class="title">Judge Calibration Sample (${items.length})</div>
        <button onclick="reloadCalibration()">Refresh sample</button>
      </div>
      <div class="chips">${buckets}</div>
    </div>
    <div class="split">
      <div class="panel"><div class="list">${list}</div></div>
      <div class="panel detail" id="judgeDetail">Select a calibration turn.</div>
    </div>
  `;
  if (state.selectedJudgeKey) renderJudgeDetail();
}

async function reloadCalibration() {
  await loadCalibration();
  renderJudge();
}

async function selectJudgeItem(key) {
  state.selectedJudgeKey = key;
  renderJudge();
}

async function renderJudgeDetail() {
  const box = $('judgeDetail');
  const item = (state.calibration.items || []).find(x => x.key === state.selectedJudgeKey);
  if (!item) return;
  box.innerHTML = 'Loading...';
  try {
    const d = await api(`/api/judge_turn?model=${enc(item.model)}&paper_id=${enc(item.paper_id)}&turn_id=${enc(item.turn_id)}`);
    const t = d.turn || {};
    const ann = d.annotation || {};
    const covered = new Set(ann.covered_kc_ids || []);
    box.innerHTML = `
      <h2>${escapeHtml(item.turn_id)} · ${escapeHtml(item.bucket)}</h2>
      <div class="chips">${chip(item.model)} ${chip(item.paper_id)} ${chip(t.question_type)} ${t.requires_multimodal_input ? chip('MM','amber') : chip('text')}</div>
      <h3>Question</h3><div>${escapeHtml(t.question_text || '')}</div>
      <h3>Model Answer</h3><div>${escapeHtml(t.model_answer || '')}</div>
      <h3>Covered KCs</h3>
      <div class="choice-grid">
        ${d.target_kcs.map(kc => `
          <label class="choice-line" style="grid-template-columns:auto 1fr">
            <input type="checkbox" class="judge-kc" value="${escapeHtml(kc.kc_id)}" ${covered.has(kc.kc_id) ? 'checked' : ''}>
            <span><b>${escapeHtml(kc.kc_id)}</b> ${escapeHtml(kc.full_claim || '')}</span>
          </label>
        `).join('') || '<div class="muted">No target KCs.</div>'}
      </div>
      <h3>Judge Calibration Labels</h3>
      ${judgeSelect('challengeOutcome','Challenge fail/pass', ann.challenge_outcome || 'na', ['na','pass','fail','incomplete'])}
      ${judgeSelect('hallucinationPresent','Hallucination?', ann.hallucination_present || 'na', ['na','no','yes','unclear'])}
      ${judgeSelect('hallucinationType','Hallucination type', ann.hallucination_type || 'na', ['na','none','false_premise','overclaim','wrong_relation','contradicted_kc','fabricated_claim','other'])}
      ${judgeSelect('repairSuccess','Repair success', ann.repair_success || 'na', ['na','success','fail','unclear'])}
      <label>Reviewer</label><input id="judgeReviewer" value="${escapeHtml(ann.reviewer || '')}">
      <label>Notes</label><textarea id="judgeNotes">${escapeHtml(ann.notes || '')}</textarea>
      <button class="primary" onclick="saveJudgeAnnotation()">Save judge annotation</button>
      <h3>Auto Judge Output</h3><pre>${escapeHtml(JSON.stringify(t.judge_result || {}, null, 2))}</pre>
    `;
  } catch (err) {
    box.innerHTML = `<span class="chip red">${escapeHtml(err.message)}</span>`;
  }
}

function judgeSelect(id, label, value, options) {
  return `<label>${escapeHtml(label)}</label><select id="${id}">${options.map(o => `<option ${value===o?'selected':''}>${escapeHtml(o)}</option>`).join('')}</select>`;
}

async function saveJudgeAnnotation() {
  const item = (state.calibration.items || []).find(x => x.key === state.selectedJudgeKey);
  if (!item) return;
  const covered = Array.from(document.querySelectorAll('.judge-kc:checked')).map(x => x.value);
  await post('/api/judge_annotation', {
    model: item.model,
    paper_id: item.paper_id,
    turn_id: item.turn_id,
    question_id: item.question_id,
    covered_kc_ids: covered,
    challenge_outcome: $('challengeOutcome').value,
    hallucination_present: $('hallucinationPresent').value,
    hallucination_type: $('hallucinationType').value,
    repair_success: $('repairSuccess').value,
    reviewer: $('judgeReviewer').value,
    notes: $('judgeNotes').value
  });
  await loadCalibration();
  renderJudge();
}

['Overview','Packages','Results','Judge'].forEach(name => {
  $('tab' + name).onclick = () => { state.tab = name.toLowerCase(); render(); };
});
$('paperSelect').onchange = async () => { await loadPaper(); await loadResult(); render(); };
$('modelSelect').onchange = async () => { await loadResult(); render(); };
$('typeFilter').onchange = render;
$('searchBox').oninput = render;

loadOverview().catch(err => {
  document.body.innerHTML = `<pre>${escapeHtml(err.stack || err.message)}</pre>`;
});
</script>
</body>
</html>
"""


if __name__ == "__main__":
    main()
