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
import uuid
import psycopg
from dotenv import load_dotenv

from app.config import get_settings
import app.repositories.jobs as job_repo
from app.worker.handler import process_claimed_job

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
    ) -> None:
        self.db_url = db_url
        self.worker_id = worker_id or f"worker_{uuid.uuid4().hex[:8]}"
        self.poll_interval = poll_interval
        self.lease_seconds = lease_seconds
        self.recovery_interval = recovery_interval
        self._stop_event = threading.Event()
        self._last_recovery_time = 0.0

    def stop(self) -> None:
        """Signal the worker to stop gracefully."""
        logger.info(f"Worker {self.worker_id}: received shutdown signal.")
        self._stop_event.set()

    @property
    def is_stopped(self) -> bool:
        return self._stop_event.is_set()

    def run_one_cycle(self, conn: psycopg.Connection, tenant_id: uuid.UUID | None = None) -> str | None:
        """Execute one cycle: recover stale jobs if interval elapsed, then claim and process one job.

        Returns:
            str | None: Result status if a job was claimed ('completed', 'requeued', etc.),
                        or None if the queue was empty.
        """
        now = time.monotonic()
        if now - self._last_recovery_time >= self.recovery_interval:
            try:
                with conn.transaction():
                    recovered = job_repo.recover_stale_jobs(
                        conn=conn,
                        lease_seconds=self.lease_seconds,
                        tenant_id=tenant_id,
                    )
                if recovered > 0:
                    logger.info(f"Worker {self.worker_id}: recovered {recovered} stale job(s).")
            except Exception as e:
                logger.error(f"Worker {self.worker_id}: error during stale job recovery: {e}")
            self._last_recovery_time = now

        # Claim next eligible queued job
        job: job_repo.JobRecord | None = None
        try:
            with conn.transaction():
                job = job_repo.claim_next_job(
                    conn=conn,
                    worker_id=self.worker_id,
                    lease_seconds=self.lease_seconds,
                    tenant_id=tenant_id,
                )
        except Exception as e:
            logger.error(f"Worker {self.worker_id}: error claiming job: {e}")
            return None

        if not job:
            return None

        logger.info(f"Worker {self.worker_id}: claimed job {job.id} (type={job.type}, gen={job.attempts})")
        result = process_claimed_job(
            conn=conn,
            job=job,
            worker_id=self.worker_id,
            lease_seconds=self.lease_seconds,
        )
        logger.info(f"Worker {self.worker_id}: job {job.id} finished with status '{result}'")
        return result

    def start(self) -> None:
        """Run worker loop until stopped."""
        logger.info(f"Worker {self.worker_id} started. Poll interval: {self.poll_interval}s.")

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
