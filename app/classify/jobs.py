"""Background classification jobs with real progress and cancellation.

Classification is the slow phase — one model call per question — so it runs off
the request thread and reports genuine elapsed/remaining time rather than an
animated placeholder. Estimates come from the running average of questions
actually completed, so they get more accurate as the job proceeds.

Cancelling stops before the next question starts. Everything already classified
is kept: a programmer who cancels a long run should still have the work that
finished.
"""

from __future__ import annotations

import logging
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Callable, Literal

from app.classify.classifier import ClassificationOutcome, classify_question
from app.classify.lines import question_lines
from app.classify.ollama import OllamaClient
from app.models.document import ParsedDocument
from app.models.survey import Question

logger = logging.getLogger(__name__)

JobState = Literal["running", "completed", "cancelled", "failed"]

#: Jobs are discarded this long after finishing, so a long-lived server does
#: not accumulate completed jobs forever.
JOB_RETENTION_SECONDS = 3600


@dataclass
class ClassifyJob:
    """One batch of questions being classified."""

    id: str
    labels: list[str]
    threshold: float
    state: JobState = "running"
    completed: int = 0
    started_at: float = field(default_factory=time.monotonic)
    finished_at: float | None = None
    current_label: str | None = None
    error: str | None = None
    questions: list[Question] = field(default_factory=list)
    fallback_count: int = 0
    _cancel: threading.Event = field(default_factory=threading.Event)

    @property
    def total(self) -> int:
        return len(self.labels)

    @property
    def elapsed_seconds(self) -> float:
        end = self.finished_at if self.finished_at is not None else time.monotonic()
        return max(0.0, end - self.started_at)

    @property
    def estimated_remaining_seconds(self) -> float | None:
        """Projected from the running average, or ``None`` before the first result.

        No estimate is better than a fabricated one: until at least one question
        has finished there is nothing to extrapolate from.
        """
        if self.state != "running" or not self.completed:
            return None
        remaining = self.total - self.completed
        if remaining <= 0:
            return 0.0
        return (self.elapsed_seconds / self.completed) * remaining

    @property
    def cancelled(self) -> bool:
        return self._cancel.is_set()

    def cancel(self) -> None:
        self._cancel.set()

    def status(self) -> dict:
        remaining = self.estimated_remaining_seconds
        return {
            "job_id": self.id,
            "state": self.state,
            "completed": self.completed,
            "total": self.total,
            "current_label": self.current_label,
            "elapsed_seconds": round(self.elapsed_seconds, 1),
            "estimated_remaining_seconds": None if remaining is None else round(remaining, 1),
            "fallback_count": self.fallback_count,
            "error": self.error,
        }


class JobManager:
    """Owns running jobs and the thread each one runs on."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._jobs: dict[str, ClassifyJob] = {}

    def get(self, job_id: str) -> ClassifyJob | None:
        with self._lock:
            return self._jobs.get(job_id)

    def start(
        self,
        document: ParsedDocument,
        labels: list[str],
        client: OllamaClient,
        threshold: float,
        on_batch: Callable[[list[Question]], None] | None = None,
    ) -> ClassifyJob:
        """Begin classifying ``labels`` from ``document`` on a worker thread.

        ``on_batch`` receives the classified questions as each one finishes, so
        results are durable before the job ends — including when it is cancelled.
        """
        job = ClassifyJob(id=uuid.uuid4().hex[:12], labels=list(labels), threshold=threshold)
        with self._lock:
            self._prune_locked()
            self._jobs[job.id] = job

        thread = threading.Thread(
            target=self._run,
            args=(job, document, client, on_batch),
            name=f"classify-{job.id}",
            daemon=True,
        )
        thread.start()
        return job

    def _run(self, job, document, client, on_batch) -> None:
        boundaries = {b.label: b for b in document.questions if b.label}
        try:
            for label in job.labels:
                if job.cancelled:
                    break
                job.current_label = label

                boundary = boundaries.get(label)
                lines = question_lines(document, boundary) if boundary else []
                outcome: ClassificationOutcome = classify_question(
                    label, lines, client, job.threshold
                )

                job.questions.append(outcome.question)
                if outcome.used_fallback:
                    job.fallback_count += 1
                job.completed += 1

                if on_batch:
                    on_batch([outcome.question])
        except Exception as exc:  # pragma: no cover - defensive
            logger.exception("Classification job %s failed", job.id)
            job.state = "failed"
            job.error = str(exc)
        else:
            job.state = "cancelled" if job.cancelled else "completed"
        finally:
            job.current_label = None
            job.finished_at = time.monotonic()

    def _prune_locked(self) -> None:
        now = time.monotonic()
        for job_id, job in list(self._jobs.items()):
            if job.finished_at and now - job.finished_at > JOB_RETENTION_SECONDS:
                del self._jobs[job_id]

    def clear(self) -> None:
        """Cancel and forget every job. Used between tests."""
        with self._lock:
            for job in self._jobs.values():
                job.cancel()
            self._jobs.clear()


job_manager = JobManager()
