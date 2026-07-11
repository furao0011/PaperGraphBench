from __future__ import annotations

import sys
import threading
from dataclasses import dataclass

from rich.console import Console
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
)

from src.progress import enabled as progress_enabled


@dataclass(frozen=True)
class BatchTask:
    paper_id: str
    total: int


class PaperBatchProgress:
    def __init__(self, label: str, tasks: list[BatchTask]) -> None:
        self.label = label
        self.tasks = tasks
        self.console = Console(file=sys.stderr)
        self.dynamic = progress_enabled() and self.console.is_terminal
        self._lock = threading.Lock()
        self._task_ids: dict[str, int] = {}
        self._totals = {task.paper_id: task.total for task in tasks}
        self._progress = Progress(
            SpinnerColumn(),
            TextColumn("[bold]{task.fields[paper_id]}", justify="left"),
            BarColumn(),
            TaskProgressColumn(),
            TextColumn("{task.fields[status]}", justify="left"),
            TimeElapsedColumn(),
            console=self.console,
            transient=False,
            redirect_stdout=True,
            redirect_stderr=True,
        )

    def __enter__(self) -> PaperBatchProgress:
        if self.dynamic:
            self._progress.start()
            for task in self.tasks:
                self._task_ids[task.paper_id] = self._progress.add_task(
                    self.label,
                    total=max(1, task.total),
                    paper_id=_short_paper_id(task.paper_id),
                    status="queued",
                )
        else:
            self.console.print(
                f"[{self.label}] queued papers={len(self.tasks)}",
                markup=False,
                highlight=False,
            )
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        if self.dynamic:
            self._progress.stop()

    def start(self, paper_id: str, status: str) -> None:
        self.update(paper_id, status=status, emit=True)

    def update(
        self,
        paper_id: str,
        *,
        status: str,
        advance: int = 0,
        completed: int | None = None,
        emit: bool = False,
    ) -> None:
        with self._lock:
            if self.dynamic:
                kwargs: dict[str, object] = {"status": status}
                if advance:
                    kwargs["advance"] = advance
                if completed is not None:
                    kwargs["completed"] = completed
                self._progress.update(self._task_ids[paper_id], **kwargs)
                return
            if emit:
                self.console.print(
                    f"[{self.label}] {status} | paper_id={_short_paper_id(paper_id)}",
                    markup=False,
                    highlight=False,
                )

    def finish(self, paper_id: str, status: str) -> None:
        total = self._totals[paper_id]
        self.update(paper_id, status=status, completed=max(1, total), emit=True)

    def fail(self, paper_id: str, error: BaseException) -> None:
        self.update(
            paper_id,
            status=f"failed: {type(error).__name__}: {error}",
            emit=True,
        )

def _short_paper_id(paper_id: str, limit: int = 32) -> str:
    if len(paper_id) <= limit:
        return paper_id
    tail = 10
    return f"{paper_id[: limit - tail - 3]}...{paper_id[-tail:]}"
