import asyncio
import sqlite3
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import httpx

from web import create_app
from web.config import Settings
from web.database import create_job_with_files, init_db


def make_app(data_dir: Path, max_parallel_scans: int = 1):
    settings = Settings(
        app_password="secret",
        session_secret="test-session-secret",
        data_dir=data_dir,
        min_free_disk_gb=0,
        max_parallel_scans=max_parallel_scans,
        write_debug=False,
        scheduler_poll_seconds=0.01,
    )
    return create_app(settings)


async def authenticated_client(app):
    transport = httpx.ASGITransport(app=app)
    client = httpx.AsyncClient(transport=transport, base_url="http://test")
    await client.post("/login", data={"password": "secret"}, follow_redirects=False)
    return client


def rows(database_path: Path, query: str):
    with sqlite3.connect(database_path) as connection:
        connection.row_factory = sqlite3.Row
        return connection.execute(query).fetchall()


async def upload(client, title: str, filenames: list[str]):
    files = [
        ("files", (filename, b"image-bytes", "image/jpeg"))
        for filename in filenames
    ]
    return await client.post(
        "/upload",
        data={"title": title},
        files=files,
        follow_redirects=False,
    )


async def wait_for(predicate, timeout: float = 2.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        await asyncio.sleep(0.02)
    raise AssertionError("Timed out waiting for condition")


def test_two_jobs_keep_chronological_active_queue_order(monkeypatch, tmp_path):
    release = threading.Event()

    def fake_process_scan(*args, **kwargs):
        release.wait(timeout=1)
        return SimpleNamespace(detections=[])

    monkeypatch.setattr("web.scheduler.process_scan", fake_process_scan)

    async def run():
        app = make_app(tmp_path, max_parallel_scans=1)
        async with app.router.lifespan_context(app):
            client = await authenticated_client(app)
            try:
                await upload(client, "First job", ["first.jpg"])
                await upload(client, "Second job", ["second.jpg"])
                await wait_for(
                    lambda: rows(
                        tmp_path / "app.db",
                        "SELECT COUNT(*) AS count FROM jobs WHERE status IN ('queued', 'running')",
                    )[0]["count"] == 2
                )
                response = await client.get("/active")
                release.set()
                await wait_for(
                    lambda: rows(
                        tmp_path / "app.db",
                        "SELECT COUNT(*) AS count FROM jobs WHERE status IN ('queued', 'running')",
                    )[0]["count"] == 0,
                    timeout=3,
                )
            finally:
                release.set()
                await client.aclose()
        return response

    response = asyncio.run(run())

    assert response.status_code == 200
    assert response.text.index("First job") < response.text.index("Second job")


def test_max_parallel_scans_one_allows_only_one_running_scan(monkeypatch, tmp_path):
    lock = threading.Lock()
    active = 0
    max_seen = 0

    def fake_process_scan(*args, **kwargs):
        nonlocal active, max_seen
        with lock:
            active += 1
            max_seen = max(max_seen, active)
        time.sleep(0.08)
        with lock:
            active -= 1
        return SimpleNamespace(detections=[])

    monkeypatch.setattr("web.scheduler.process_scan", fake_process_scan)

    async def run():
        app = make_app(tmp_path, max_parallel_scans=1)
        async with app.router.lifespan_context(app):
            client = await authenticated_client(app)
            try:
                await upload(client, "Limited", ["a.jpg", "b.jpg"])
                await wait_for(
                    lambda: rows(tmp_path / "app.db", "SELECT status FROM jobs")[0]["status"] == "completed",
                    timeout=3,
                )
            finally:
                await client.aclose()

    asyncio.run(run())

    assert max_seen == 1


def test_max_parallel_scans_two_allows_two_running_scans(monkeypatch, tmp_path):
    lock = threading.Lock()
    active = 0
    max_seen = 0

    def fake_process_scan(*args, **kwargs):
        nonlocal active, max_seen
        with lock:
            active += 1
            max_seen = max(max_seen, active)
        time.sleep(0.1)
        with lock:
            active -= 1
        return SimpleNamespace(detections=[])

    monkeypatch.setattr("web.scheduler.process_scan", fake_process_scan)

    async def run():
        app = make_app(tmp_path, max_parallel_scans=2)
        async with app.router.lifespan_context(app):
            client = await authenticated_client(app)
            try:
                await upload(client, "Parallel", ["a.jpg", "b.jpg"])
                await wait_for(
                    lambda: rows(tmp_path / "app.db", "SELECT status FROM jobs")[0]["status"] == "completed",
                    timeout=3,
                )
            finally:
                await client.aclose()

    asyncio.run(run())

    assert max_seen == 2


def test_max_parallel_scans_two_spreads_slots_across_jobs(monkeypatch, tmp_path):
    release = threading.Event()

    def fake_process_scan(*args, **kwargs):
        release.wait(timeout=1)
        return SimpleNamespace(detections=[])

    monkeypatch.setattr("web.scheduler.process_scan", fake_process_scan)

    first_paths = []
    second_paths = []
    for name in ("a.jpg", "b.jpg", "c.jpg"):
        path = tmp_path / "jobs" / "first-job" / "inputs" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"image")
        first_paths.append(str(path.relative_to(tmp_path)))
    path = tmp_path / "jobs" / "second-job" / "inputs" / "only.jpg"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"image")
    second_paths.append(str(path.relative_to(tmp_path)))

    init_db(tmp_path / "app.db")
    create_job_with_files(tmp_path / "app.db", "first-job", "First multi-scan job", first_paths)
    create_job_with_files(tmp_path / "app.db", "second-job", "Second single-scan job", second_paths)

    async def run():
        app = make_app(tmp_path, max_parallel_scans=2)
        async with app.router.lifespan_context(app):
            try:
                await wait_for(
                    lambda: rows(
                        tmp_path / "app.db",
                        "SELECT COUNT(*) AS count FROM files WHERE status = 'running'",
                    )[0]["count"] == 2,
                    timeout=3,
                )
                running = rows(
                    tmp_path / "app.db",
                    "SELECT job_id, COUNT(*) AS count FROM files WHERE status = 'running' GROUP BY job_id",
                )
            finally:
                release.set()
                await wait_for(
                    lambda: rows(
                        tmp_path / "app.db",
                        "SELECT COUNT(*) AS count FROM jobs WHERE status IN ('queued', 'running')",
                    )[0]["count"] == 0,
                    timeout=3,
                )
        return running

    running = asyncio.run(run())

    assert {row["job_id"]: row["count"] for row in running} == {
        "first-job": 1,
        "second-job": 1,
    }


def test_scan_failure_marks_file_and_job_failed_with_message(monkeypatch, tmp_path):
    def fake_process_scan(input_path, *args, **kwargs):
        if "bad" in Path(input_path).name:
            raise RuntimeError("simulated extraction failure")
        return SimpleNamespace(detections=[])

    monkeypatch.setattr("web.scheduler.process_scan", fake_process_scan)

    async def run():
        app = make_app(tmp_path, max_parallel_scans=1)
        async with app.router.lifespan_context(app):
            client = await authenticated_client(app)
            try:
                await upload(client, "Failure", ["bad.jpg"])
                await wait_for(
                    lambda: rows(tmp_path / "app.db", "SELECT status FROM jobs")[0]["status"] == "failed",
                    timeout=3,
                )
                response = await client.get("/completed")
            finally:
                await client.aclose()
        return response

    response = asyncio.run(run())

    job = rows(tmp_path / "app.db", "SELECT status, error_message FROM jobs")[0]
    file_row = rows(tmp_path / "app.db", "SELECT status, error_message FROM files")[0]
    assert job["status"] == "failed"
    assert "simulated extraction failure" in job["error_message"]
    assert file_row["status"] == "failed"
    assert "simulated extraction failure" in response.text


def test_eta_visible_for_running_and_pending_for_queued(monkeypatch, tmp_path):
    release = threading.Event()

    def fake_process_scan(*args, **kwargs):
        release.wait(timeout=1)
        return SimpleNamespace(detections=[])

    monkeypatch.setattr("web.scheduler.process_scan", fake_process_scan)

    async def run():
        app = make_app(tmp_path, max_parallel_scans=1)
        async with app.router.lifespan_context(app):
            client = await authenticated_client(app)
            try:
                await upload(client, "Running ETA", ["running.jpg"])
                await upload(client, "Queued ETA", ["queued.jpg"])
                await wait_for(
                    lambda: rows(
                        tmp_path / "app.db",
                        "SELECT COUNT(*) AS count FROM jobs WHERE status = 'running'",
                    )[0]["count"] == 1
                )
                response = await client.get("/active")
                release.set()
                await wait_for(
                    lambda: rows(
                        tmp_path / "app.db",
                        "SELECT COUNT(*) AS count FROM jobs WHERE status IN ('queued', 'running')",
                    )[0]["count"] == 0,
                    timeout=3,
                )
            finally:
                release.set()
                await client.aclose()
        return response

    response = asyncio.run(run())

    assert response.status_code == 200
    assert "Running ETA" in response.text
    assert "Calculating" in response.text
    assert "Queued ETA" in response.text
    assert "Pending" in response.text
