import asyncio
import json
import logging
import sqlite3
from pathlib import Path
from types import SimpleNamespace

import httpx

from web import create_app
from web.config import Settings, load_settings
from web.database import create_job_with_files, init_db


PNG_BYTES = (
    b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01"
    b"\x08\x02\x00\x00\x00\x90wS\xde\x00\x00\x00\x0cIDATx\x9cc\xf8\xff\xff?"
    b"\x00\x05\xfe\x02\xfeA\xde\xfc\x96\x00\x00\x00\x00IEND\xaeB`\x82"
)


def make_app(data_dir: Path):
    settings = Settings(
        app_password="secret",
        session_secret="test-session-secret",
        data_dir=data_dir,
        min_free_disk_gb=0,
        max_parallel_scans=1,
        write_debug=False,
        scheduler_poll_seconds=0.01,
    )
    return create_app(settings)


async def authenticated_client(app):
    transport = httpx.ASGITransport(app=app)
    client = httpx.AsyncClient(transport=transport, base_url="http://test")
    await client.post("/login", data={"password": "secret"}, follow_redirects=False)
    return client


async def wait_for(predicate, timeout: float = 2.0):
    deadline = asyncio.get_running_loop().time() + timeout
    while asyncio.get_running_loop().time() < deadline:
        if predicate():
            return
        await asyncio.sleep(0.02)
    raise AssertionError("Timed out waiting for condition")


def rows(database_path: Path, query: str):
    with sqlite3.connect(database_path) as connection:
        connection.row_factory = sqlite3.Row
        return connection.execute(query).fetchall()


def test_health_endpoint_initializes_database_without_auth(tmp_path):
    async def run():
        app = make_app(tmp_path)
        transport = httpx.ASGITransport(app=app)
        async with app.router.lifespan_context(app):
            async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
                return await client.get("/health")

    response = asyncio.run(run())

    assert response.status_code == 200
    assert response.json()["status"] == "ok"
    assert response.json()["database"] == "ok"
    assert (tmp_path / "app.db").is_file()


def test_job_lifecycle_logs_json_events(monkeypatch, caplog, tmp_path):
    def fake_process_scan(input_path, photos_dir, debug_dir, source_stem, write_debug=True, debug_panel_width=None):
        photos_dir.mkdir(parents=True, exist_ok=True)
        output_path = photos_dir / f"{source_stem}-01.png"
        output_path.write_bytes(PNG_BYTES)
        return SimpleNamespace(detections=[])

    monkeypatch.setattr("web.scheduler.process_scan", fake_process_scan)
    caplog.set_level(logging.INFO, logger="web")

    async def run():
        app = make_app(tmp_path)
        async with app.router.lifespan_context(app):
            client = await authenticated_client(app)
            try:
                await client.post(
                    "/upload",
                    data={"title": "Logged job"},
                    files={"files": ("scan.png", PNG_BYTES, "image/png")},
                    follow_redirects=False,
                )
                await wait_for(
                    lambda: rows(tmp_path / "app.db", "SELECT status FROM jobs")[0]["status"] == "completed",
                    timeout=3,
                )
            finally:
                await client.aclose()

    asyncio.run(run())

    events = [json.loads(record.message) for record in caplog.records if record.name == "web"]
    event_names = {event["event"] for event in events}
    job_id = rows(tmp_path / "app.db", "SELECT id FROM jobs")[0]["id"]
    assert {"job_created", "scan_started", "scan_completed", "job_finished"}.issubset(event_names)
    assert any(event["event"] == "job_created" and event["job_id"] == job_id for event in events)
    assert any(event["event"] == "job_finished" and event["status"] == "completed" for event in events)


def test_restart_requeues_interrupted_running_job(monkeypatch, tmp_path):
    database_path = tmp_path / "app.db"
    init_db(database_path)
    input_path = tmp_path / "jobs" / "restart-job" / "inputs" / "scan.jpg"
    input_path.parent.mkdir(parents=True)
    input_path.write_bytes(b"image")
    create_job_with_files(database_path, "restart-job", "Restart queued", [str(input_path.relative_to(tmp_path))])
    with sqlite3.connect(database_path) as connection:
        connection.execute("UPDATE jobs SET status = 'running', started_at = CURRENT_TIMESTAMP")
        connection.execute("UPDATE files SET status = 'running', started_at = CURRENT_TIMESTAMP")

    def no_claims(*args, **kwargs):
        return None

    monkeypatch.setattr("web.scheduler.claim_next_pending_file", no_claims)

    async def run():
        app = make_app(tmp_path)
        async with app.router.lifespan_context(app):
            client = await authenticated_client(app)
            try:
                active = await client.get("/active")
            finally:
                await client.aclose()
        return active

    active = asyncio.run(run())

    assert active.status_code == 200
    assert "Restart queued" in active.text
    assert rows(database_path, "SELECT status FROM jobs")[0]["status"] == "queued"
    assert rows(database_path, "SELECT status FROM files")[0]["status"] == "pending"


def test_load_settings_uses_documented_deployment_defaults(monkeypatch, tmp_path):
    monkeypatch.setenv("APP_PASSWORD", "secret")
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    for name in ("MAX_PARALLEL_SCANS", "WRITE_DEBUG", "DEBUG_PANEL_WIDTH", "MIN_FREE_DISK_GB"):
        monkeypatch.delenv(name, raising=False)

    settings = load_settings()

    assert settings.max_parallel_scans == 2
    assert settings.write_debug is True
    assert settings.debug_panel_width is None
    assert settings.min_free_disk_gb == 10


def test_readme_documents_docker_usage_and_env_vars():
    readme = Path("README.md").read_text()

    assert "make prod-build" in readme
    assert "/health" in readme
    assert "./data" in readme
    for name in (
        "APP_PASSWORD",
        "DATA_DIR",
        "MAX_PARALLEL_SCANS",
        "WRITE_DEBUG",
        "DEBUG_PANEL_WIDTH",
        "MIN_FREE_DISK_GB",
    ):
        assert f"`{name}`" in readme


def test_docker_compose_exposes_operational_env_vars():
    compose = Path("docker-compose.yml").read_text()

    assert "APP_PASSWORD" in compose
    assert "./data:/app/data" in compose
    for name in ("MAX_PARALLEL_SCANS", "WRITE_DEBUG", "DEBUG_PANEL_WIDTH", "MIN_FREE_DISK_GB"):
        assert name in compose


def test_docker_packaging_includes_models_and_ignores_generated_artifacts():
    dockerfile = Path("Dockerfile").read_text()
    dockerignore = Path(".dockerignore").read_text()

    assert "COPY extractor ./extractor" in dockerfile
    assert Path("extractor/models/face_detection_yunet_2023mar.onnx").is_file()
    for pattern in (".venv/", "__pycache__/", ".pytest_cache/", "output/", "data/"):
        assert pattern in dockerignore


def test_responsive_css_covers_job_lists_uploads_and_galleries():
    css = Path("web/static/app.css").read_text()
    jobs_table = Path("web/templates/_jobs_table.html").read_text()

    assert "@media (max-width: 720px)" in css
    assert "@media (max-width: 420px)" in css
    assert "td::before" in css
    assert ".selected-file" in css
    assert ".download-actions" in css
    assert ".gallery" in css
    assert 'data-label="Actions"' in jobs_table
