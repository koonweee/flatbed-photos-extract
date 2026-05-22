"""Background scan scheduler for the web app."""

from __future__ import annotations

import asyncio
import concurrent.futures
import time
from collections import deque
from pathlib import Path

from extractor import append_metadata, process_scan

from .database import (
    claim_next_pending_file,
    complete_file,
    fail_file_and_job,
    finalize_job_if_finished,
    reset_interrupted_work,
)
from .logging import log_job_event
from .results import ensure_job_zip


class JobScheduler:
    def __init__(
        self,
        database_path: Path,
        data_dir: Path,
        max_parallel_scans: int,
        write_debug: bool,
        debug_panel_width: int | None,
        poll_seconds: float = 0.1,
    ) -> None:
        self.database_path = database_path
        self.data_dir = data_dir
        self.max_parallel_scans = max(1, max_parallel_scans)
        self.write_debug = write_debug
        self.debug_panel_width = debug_panel_width
        self.poll_seconds = poll_seconds
        self._stop = asyncio.Event()
        self._loop_task: asyncio.Task[None] | None = None
        self._active_tasks: set[asyncio.Task[None]] = set()
        self._durations: deque[float] = deque(maxlen=20)
        self._metadata_lock = asyncio.Lock()
        self._executor = concurrent.futures.ThreadPoolExecutor(max_workers=self.max_parallel_scans + 1)

    @property
    def average_seconds_per_scan(self) -> float | None:
        if not self._durations:
            return None
        return sum(self._durations) / len(self._durations)

    async def start(self) -> None:
        reset_interrupted_work(self.database_path)
        self._stop.clear()
        log_job_event("scheduler_started", "scheduler", max_parallel_scans=self.max_parallel_scans)
        self._loop_task = asyncio.create_task(self._run(), name="flatbed-job-scheduler")

    async def stop(self) -> None:
        self._stop.set()
        if self._active_tasks:
            try:
                await asyncio.wait_for(
                    asyncio.gather(*self._active_tasks, return_exceptions=True),
                    timeout=5,
                )
            except asyncio.TimeoutError:
                for task in self._active_tasks:
                    task.cancel()
        if self._loop_task is not None:
            self._loop_task.cancel()
            await asyncio.gather(self._loop_task, return_exceptions=True)
        self._executor.shutdown(wait=False, cancel_futures=True)
        log_job_event("scheduler_stopped", "scheduler")

    async def _run(self) -> None:
        while not self._stop.is_set():
            self._active_tasks = {task for task in self._active_tasks if not task.done()}
            while len(self._active_tasks) < self.max_parallel_scans:
                claim = await self._call_blocking(claim_next_pending_file, self.database_path)
                if claim is None:
                    break
                task = asyncio.create_task(self._process_claim(dict(claim)))
                self._active_tasks.add(task)
            try:
                await asyncio.wait_for(self._stop.wait(), timeout=self.poll_seconds)
            except asyncio.TimeoutError:
                pass

    async def _process_claim(self, claim: dict) -> None:
        file_id = claim["file_id"]
        job_id = claim["job_id"]
        input_path = self.data_dir / claim["path"]
        job_dir = self.data_dir / "jobs" / job_id
        photos_dir = job_dir / "output" / "photos"
        debug_dir = job_dir / "output" / "debug"
        metadata_path = job_dir / "output" / "metadata.csv"
        started = time.perf_counter()
        log_job_event("scan_started", job_id, file_id=file_id, input_path=str(claim["path"]))

        try:
            result = await self._call_blocking(
                process_scan,
                input_path,
                photos_dir,
                debug_dir,
                input_path.stem,
                self.write_debug,
                self.debug_panel_width,
            )
        except Exception as exc:
            await self._call_blocking(fail_file_and_job, self.database_path, file_id, job_id, str(exc))
            log_job_event("scan_failed", job_id, file_id=file_id, error=str(exc))
            return

        duration_seconds = max(0.001, time.perf_counter() - started)
        self._durations.append(duration_seconds)
        async with self._metadata_lock:
            await self._call_blocking(append_metadata, metadata_path, result.detections)
        await self._call_blocking(complete_file, self.database_path, file_id)
        log_job_event(
            "scan_completed",
            job_id,
            file_id=file_id,
            duration_seconds=round(duration_seconds, 3),
            detections=len(result.detections),
        )
        job_status = await self._call_blocking(finalize_job_if_finished, self.database_path, job_id)
        if job_status:
            log_job_event("job_finished", job_id, status=job_status)
        if job_status == "completed":
            try:
                await self._call_blocking(ensure_job_zip, self.data_dir, job_id)
            except ValueError:
                pass

    async def _call_blocking(self, func, *args):
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(self._executor, func, *args)

def eta_label(status: str, total_files: int, completed_files: int, average_seconds: float | None) -> str:
    remaining = max(0, total_files - completed_files)
    if status == "queued":
        return "Pending"
    if status != "running" or remaining == 0:
        return ""
    if average_seconds is None:
        return "Calculating"
    seconds = max(1, round(remaining * average_seconds))
    if seconds < 60:
        return f"about {seconds}s"
    minutes = max(1, round(seconds / 60))
    return f"about {minutes}m"
