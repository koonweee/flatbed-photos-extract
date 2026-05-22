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


def test_abort_queued_job_before_processing(monkeypatch, tmp_path):
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
                await upload(client, "Running job", ["running.jpg"])
                await upload(client, "Queued job", ["queued.jpg"])
                await wait_for(
                    lambda: rows(
                        tmp_path / "app.db",
                        "SELECT COUNT(*) AS count FROM jobs WHERE status = 'queued'",
                    )[0]["count"] == 1
                )
                queued_id = rows(
                    tmp_path / "app.db",
                    "SELECT id FROM jobs WHERE title = 'Queued job'",
                )[0]["id"]
                response = await client.post(
                    f"/jobs/{queued_id}/abort",
                    headers={"HX-Request": "true"},
                )
            finally:
                release.set()
                await client.aclose()
        return response

    response = asyncio.run(run())

    job = rows(tmp_path / "app.db", "SELECT status, error_message FROM jobs WHERE title = 'Queued job'")[0]
    file_row = rows(
        tmp_path / "app.db",
        "SELECT status, error_message FROM files WHERE job_id = (SELECT id FROM jobs WHERE title = 'Queued job')",
    )[0]
    assert response.status_code == 200
    assert job["status"] == "cancelled"
    assert "Cancelled by user" in job["error_message"]
    assert file_row["status"] == "cancelled"
    assert "Queued job" not in response.text


def test_abort_running_job_cancels_pending_scans(monkeypatch, tmp_path):
    release = threading.Event()
    processed: list[str] = []

    def fake_process_scan(input_path, *args, **kwargs):
        processed.append(Path(input_path).name)
        release.wait(timeout=1)
        return SimpleNamespace(detections=[])

    monkeypatch.setattr("web.scheduler.process_scan", fake_process_scan)

    async def run():
        app = make_app(tmp_path, max_parallel_scans=1)
        async with app.router.lifespan_context(app):
            client = await authenticated_client(app)
            try:
                await upload(client, "Abort running", ["a.jpg", "b.jpg", "c.jpg"])
                await wait_for(
                    lambda: rows(
                        tmp_path / "app.db",
                        "SELECT COUNT(*) AS count FROM files WHERE status = 'running'",
                    )[0]["count"] == 1
                )
                job_id = rows(tmp_path / "app.db", "SELECT id FROM jobs")[0]["id"]
                response = await client.post(
                    f"/jobs/{job_id}/abort",
                    headers={"HX-Request": "true"},
                )
                release.set()
                await wait_for(
                    lambda: rows(tmp_path / "app.db", "SELECT status FROM jobs")[0]["status"] == "cancelled",
                    timeout=3,
                )
            finally:
                release.set()
                await client.aclose()
        return response

    response = asyncio.run(run())

    statuses = [row["status"] for row in rows(tmp_path / "app.db", "SELECT status FROM files ORDER BY id")]
    assert response.status_code == 200
    assert rows(tmp_path / "app.db", "SELECT status FROM jobs")[0]["status"] == "cancelled"
    assert statuses == ["cancelled", "cancelled", "cancelled"]
    assert len(processed) == 1


def test_delete_completed_job_removes_database_rows_and_folder(tmp_path):
    database_path = tmp_path / "app.db"
    init_db(database_path)
    job_dir = tmp_path / "jobs" / "completed-job"
    input_path = job_dir / "inputs" / "scan.jpg"
    input_path.parent.mkdir(parents=True, exist_ok=True)
    input_path.write_bytes(b"image")
    create_job_with_files(database_path, "completed-job", "Completed", [str(input_path.relative_to(tmp_path))])
    with sqlite3.connect(database_path) as connection:
        connection.execute("UPDATE jobs SET status = 'completed', completed_at = CURRENT_TIMESTAMP")
        connection.execute("UPDATE files SET status = 'completed', completed_at = CURRENT_TIMESTAMP")

    async def run():
        app = make_app(tmp_path)
        async with app.router.lifespan_context(app):
            client = await authenticated_client(app)
            try:
                return await client.post(
                    "/jobs/completed-job/delete",
                    headers={"HX-Request": "true"},
                )
            finally:
                await client.aclose()

    response = asyncio.run(run())

    assert response.status_code == 200
    assert not rows(database_path, "SELECT * FROM jobs")
    assert not rows(database_path, "SELECT * FROM files")
    assert not job_dir.exists()


def test_delete_completed_job_keeps_database_rows_when_folder_delete_fails(monkeypatch, tmp_path):
    database_path = tmp_path / "app.db"
    init_db(database_path)
    job_dir = tmp_path / "jobs" / "stubborn-job"
    input_path = job_dir / "inputs" / "scan.jpg"
    input_path.parent.mkdir(parents=True, exist_ok=True)
    input_path.write_bytes(b"image")
    create_job_with_files(database_path, "stubborn-job", "Stubborn", [str(input_path.relative_to(tmp_path))])
    with sqlite3.connect(database_path) as connection:
        connection.execute("UPDATE jobs SET status = 'completed', completed_at = CURRENT_TIMESTAMP")
        connection.execute("UPDATE files SET status = 'completed', completed_at = CURRENT_TIMESTAMP")

    def fail_rmtree(path):
        raise OSError("simulated remove failure")

    monkeypatch.setattr("web.app.shutil.rmtree", fail_rmtree)

    async def run():
        app = make_app(tmp_path)
        async with app.router.lifespan_context(app):
            client = await authenticated_client(app)
            try:
                return await client.post(
                    "/jobs/stubborn-job/delete",
                    headers={"HX-Request": "true"},
                )
            finally:
                await client.aclose()

    response = asyncio.run(run())

    assert response.status_code == 500
    assert "Job files could not be deleted" in response.text
    assert rows(database_path, "SELECT * FROM jobs")
    assert rows(database_path, "SELECT * FROM files")
    assert job_dir.exists()


def test_delete_running_job_is_blocked_then_abort_allows_delete(monkeypatch, tmp_path):
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
                await upload(client, "Delete flow", ["a.jpg", "b.jpg"])
                await wait_for(
                    lambda: rows(
                        tmp_path / "app.db",
                        "SELECT status FROM jobs",
                    )[0]["status"] == "running"
                )
                job_id = rows(tmp_path / "app.db", "SELECT id FROM jobs")[0]["id"]
                blocked = await client.post(f"/jobs/{job_id}/delete")
                await client.post(f"/jobs/{job_id}/abort")
                release.set()
                await wait_for(
                    lambda: rows(tmp_path / "app.db", "SELECT status FROM jobs")[0]["status"] == "cancelled",
                    timeout=3,
                )
                deleted = await client.post(f"/jobs/{job_id}/delete")
            finally:
                release.set()
                await client.aclose()
        return blocked, deleted, job_id

    blocked, deleted, job_id = asyncio.run(run())

    assert blocked.status_code == 409
    assert "Abort the job before deleting it" in blocked.text
    assert deleted.status_code == 303
    assert not rows(tmp_path / "app.db", "SELECT * FROM jobs")
    assert not (tmp_path / "jobs" / job_id).exists()


def test_restart_after_cancellation_finalizes_interrupted_running_scan(tmp_path):
    database_path = tmp_path / "app.db"
    init_db(database_path)
    job_dir = tmp_path / "jobs" / "cancel-job"
    first = job_dir / "inputs" / "a.jpg"
    second = job_dir / "inputs" / "b.jpg"
    first.parent.mkdir(parents=True, exist_ok=True)
    first.write_bytes(b"image")
    second.write_bytes(b"image")
    create_job_with_files(
        database_path,
        "cancel-job",
        "Interrupted cancellation",
        [str(first.relative_to(tmp_path)), str(second.relative_to(tmp_path))],
    )
    with sqlite3.connect(database_path) as connection:
        connection.execute("UPDATE jobs SET status = 'cancelling', error_message = 'Cancellation requested.'")
        connection.execute(
            "UPDATE files SET status = 'running' WHERE path = ?",
            (str(first.relative_to(tmp_path)),),
        )
        connection.execute(
            "UPDATE files SET status = 'cancelled', error_message = 'Skipped after job was cancelled.' WHERE path = ?",
            (str(second.relative_to(tmp_path)),),
        )

    async def run():
        app = make_app(tmp_path, max_parallel_scans=1)
        async with app.router.lifespan_context(app):
            await wait_for(
                lambda: rows(database_path, "SELECT status FROM jobs")[0]["status"] == "cancelled",
                timeout=3,
            )

    asyncio.run(run())

    assert rows(database_path, "SELECT status FROM jobs")[0]["status"] == "cancelled"
    assert [row["status"] for row in rows(database_path, "SELECT status FROM files ORDER BY id")] == [
        "cancelled",
        "cancelled",
    ]
