"""Standalone background worker process for Ticket Intelligence Platform.

Polls the jobs table using SELECT ... FOR UPDATE SKIP LOCKED without busy-waiting.
Handles clean shutdown on SIGINT / SIGTERM.

Usage:
    python -m app.worker.main
"""

import argparse
import logging
import os
import signal
import sys
import threading
import time
from typing import Any
import uuid
import psycopg
from dotenv import load_dotenv

from app.config import get_settings
import app.repositories.jobs as job_repo
import app.repositories.kb as kb_repo
from app.worker.handler import process_claimed_job, process_claimed_article_job

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("worker")


class WorkerService:
    """Queue worker service executing jobs across tenants."""

    def __init__(
        self,
        db_url: str,
        worker_id: str | None = None,
        poll_interval: float = 1.0,
        lease_seconds: int = 300,
        recovery_interval: float = 60.0,
        enable_embeddings: bool | None = None,
        embedding_adapter: Any | None = None,
    ) -> None:
        self.db_url = db_url
        self.worker_id = worker_id or f"worker_{uuid.uuid4().hex[:8]}"
        self.poll_interval = poll_interval
        self.lease_seconds = lease_seconds
        self.recovery_interval = recovery_interval
        self._stop_event = threading.Event()
        self._last_recovery_time = 0.0

        cfg = get_settings()
        self.enable_embeddings = (
            enable_embeddings if enable_embeddings is not None else cfg.enable_embeddings
        )
        self.embedding_adapter = embedding_adapter
        self._next_queue_turn = "ticket"

    def stop(self) -> None:
        """Signal the worker to stop gracefully."""
        logger.info(f"Worker {self.worker_id}: received shutdown signal.")
        self._stop_event.set()

    @property
    def is_stopped(self) -> bool:
        return self._stop_event.is_set()

    def run_one_cycle(self, conn: psycopg.Connection, tenant_id: uuid.UUID | None = None) -> str | None:
        """Execute one cycle: recover stale jobs if interval elapsed, then claim and process one job.

        Alternates fairly between ticket jobs and article jobs using round-robin polling.

        Returns:
            str | None: Result status if a job was claimed ('completed', 'requeued', etc.),
                        or None if the queue was empty.
        """
        now = time.monotonic()
        if now - self._last_recovery_time >= self.recovery_interval:
            try:
                with conn.transaction():
                    recovered_ticket = job_repo.recover_stale_jobs(
                        conn=conn,
                        lease_seconds=self.lease_seconds,
                        tenant_id=tenant_id,
                    )
                if recovered_ticket > 0:
                    logger.info(f"Worker {self.worker_id}: recovered {recovered_ticket} stale ticket job(s).")
            except Exception as e:
                logger.error(f"Worker {self.worker_id}: error during ticket stale job recovery: {e}")

            if self.enable_embeddings:
                try:
                    with conn.transaction():
                        recovered_article = kb_repo.recover_stale_article_jobs(
                            conn=conn,
                            lease_seconds=self.lease_seconds,
                            tenant_id=tenant_id,
                        )
                    if recovered_article > 0:
                        logger.info(f"Worker {self.worker_id}: recovered {recovered_article} stale article job(s).")
                except Exception as e:
                    logger.warning(f"Worker {self.worker_id}: article stale job recovery skipped/failed: {e}")

            self._last_recovery_time = now

        allowed_types: tuple[str, ...] = (
            ("ticket_ingest", "ticket_embedding")
            if self.enable_embeddings
            else ("ticket_ingest",)
        )

        job_claimed = None
        job_kind: str | None = None

        # Determine attempt order based on fair round-robin queue turn
        if not self.enable_embeddings:
            attempt_order = ["ticket"]
        elif self._next_queue_turn == "ticket":
            attempt_order = ["ticket", "article"]
        else:
            attempt_order = ["article", "ticket"]

        for queue_type in attempt_order:
            if queue_type == "ticket":
                try:
                    with conn.transaction():
                        t_job = job_repo.claim_next_job(
                            conn=conn,
                            worker_id=self.worker_id,
                            lease_seconds=self.lease_seconds,
                            tenant_id=tenant_id,
                            allowed_types=allowed_types,
                        )
                    if t_job:
                        job_claimed = t_job
                        job_kind = "ticket"
                        if self.enable_embeddings:
                            self._next_queue_turn = "article"
                        break
                except Exception as e:
                    logger.error(f"Worker {self.worker_id}: error claiming ticket job: {e}")
                    return None
            elif queue_type == "article":
                try:
                    with conn.transaction():
                        a_job = kb_repo.claim_next_article_job(
                            conn=conn,
                            worker_id=self.worker_id,
                            lease_seconds=self.lease_seconds,
                            tenant_id=tenant_id,
                        )
                    if a_job:
                        job_claimed = a_job
                        job_kind = "article"
                        if self.enable_embeddings:
                            self._next_queue_turn = "ticket"
                        break
                except Exception as e:
                    logger.warning(
                        f"Worker {self.worker_id}: error claiming article job (table may not be migrated yet): {e}"
                    )
                    if self.enable_embeddings:
                        self._next_queue_turn = "ticket"
                    continue

        if not job_claimed or not job_kind:
            return None

        if job_kind == "ticket":
            logger.info(f"Worker {self.worker_id}: claimed ticket job {job_claimed.id} (type={job_claimed.type}, gen={job_claimed.attempts})")
            result = process_claimed_job(
                conn=conn,
                job=job_claimed,
                worker_id=self.worker_id,
                lease_seconds=self.lease_seconds,
                enable_embeddings=self.enable_embeddings,
                embedding_adapter=self.embedding_adapter,
            )
            logger.info(f"Worker {self.worker_id}: ticket job {job_claimed.id} finished with status '{result}'")
            return result
        else:
            logger.info(f"Worker {self.worker_id}: claimed article job {job_claimed.id} (type={job_claimed.type}, gen={job_claimed.attempts})")
            result = process_claimed_article_job(
                conn=conn,
                job=job_claimed,
                worker_id=self.worker_id,
                lease_seconds=self.lease_seconds,
                enable_embeddings=self.enable_embeddings,
                embedding_adapter=self.embedding_adapter,
            )
            logger.info(f"Worker {self.worker_id}: article job {job_claimed.id} finished with status '{result}'")
            return result

    def start(self) -> None:
        """Run worker loop until stopped."""
        logger.info(f"Worker {self.worker_id} started. Poll interval: {self.poll_interval}s.")

        # Preload trusted model artifact once per worker process
        try:
            from app.ml.loader import get_model_manager
            get_model_manager().load_model()
            logger.info("Trusted ML model loaded and verified once on worker process startup.")
        except Exception as exc:
            logger.warning(f"Could not preload model artifact on worker start (will retry on job): {exc}")

        # Preload embedding adapter once per worker process if enabled
        if self.enable_embeddings and self.embedding_adapter is None:
            try:
                from app.ml.embeddings import EmbeddingModelAdapter
                adapter = EmbeddingModelAdapter()
                adapter.load()
                self.embedding_adapter = adapter
                logger.info("Trusted embedding model adapter loaded and verified once on worker process startup.")
            except Exception as exc:
                logger.warning(f"Could not preload embedding model on worker start (will retry on job): {exc}")

        with psycopg.connect(self.db_url, autocommit=True) as conn:
            while not self._stop_event.is_set():
                result = self.run_one_cycle(conn)
                if result is None:
                    # Queue is empty; wait poll_interval without busy-looping
                    self._stop_event.wait(self.poll_interval)


def main() -> None:
    """CLI entrypoint."""
    load_dotenv()
    parser = argparse.ArgumentParser(description="Ticket Intelligence Platform Queue Worker.")
    parser.add_argument("--poll-interval", type=float, default=1.0, help="Seconds to wait when queue is empty.")
    parser.add_argument("--lease-seconds", type=int, default=300, help="Lease expiration duration in seconds.")
    args = parser.parse_args()

    cfg = get_settings()
    db_url = cfg.database_url.get_secret_value()

    worker = WorkerService(
        db_url=db_url,
        poll_interval=args.poll_interval,
        lease_seconds=args.lease_seconds,
    )

    def handle_signal(sig, frame):
        worker.stop()

    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    try:
        worker.start()
    except Exception as exc:
        logger.critical(f"Worker fatal crash: {exc}")
        sys.exit(1)


if __name__ == "__main__":
    main()
