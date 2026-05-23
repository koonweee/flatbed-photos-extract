import asyncio
import json
import logging
import sqlite3
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import httpx

from web import create_app
from web.app import format_completion_duration, format_display_datetime, format_job_title
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
    upload = Path("web/templates/upload.html").read_text()
    detail = Path("web/templates/job_detail.html").read_text()

    assert "@media (max-width: 720px)" in css
    assert "@media (max-width: 420px)" in css
    assert ".job-card" in css
    assert "position: sticky" in css
    assert "z-index: 10" in css
    assert ".job-list-head" not in css
    assert ".job-summary" in css
    assert ".selected-file" in css
    assert "selected-files-heading" in upload
    assert "data-clear-files" in upload
    assert "No files selected." not in upload
    assert 'data-clear-files hidden' in upload
    assert "data-view-control" in detail
    assert "gallery-list" in css
    assert ".download-actions" in css
    assert ".gallery" in css
    assert "job-actions" in jobs_table
    assert "data-gallery-open-key" in jobs_table
    assert "data-gallery-key" in jobs_table
    assert "See completed jobs" in jobs_table
    assert "action-icon" in jobs_table
    assert 'data-lucide="trash-2"' in jobs_table
    assert 'data-lucide="external-link"' in jobs_table
    assert "job-list-head" not in jobs_table
    assert "<span>Inputs</span>" not in jobs_table
    assert 'aria-label="Download all"' in jobs_table
    assert "scale(2.15)" in css
    assert "gallery-item img {\n  aspect-ratio: 1;\n  border-radius" in css


def test_relative_date_labels_drop_seconds():
    now = datetime(2026, 5, 22, 14, 30, 15)

    assert format_display_datetime("2026-05-22 09:07:44", now) == "Today at 9:07 AM"
    assert format_display_datetime("2026-05-21 18:05:12", now) == "Yesterday at 6:05 PM"
    assert format_display_datetime("2026-05-18 00:01:59", now) == "4d ago at 12:01 AM"
    assert format_display_datetime("bad 10:04:22") == "bad 10:04"


def test_completion_duration_labels():
    assert format_completion_duration("2026-05-22 09:07:00", "2026-05-22 09:07:12") == "12 seconds"
    assert format_completion_duration("2026-05-22 09:07:00", "2026-05-22 09:08:01") == "1 min 1 second"
    assert format_completion_duration("2026-05-22 09:07:00", "2026-05-22 09:09:00") == "2 min 0 seconds"


def test_blank_job_title_display_falls_back_to_created_at_label():
    assert format_job_title(None, "Today at 9:07 AM") == "Today at 9:07 AM"
    assert format_job_title("   ", "Today at 9:07 AM") == "Today at 9:07 AM"
    assert format_job_title("Family scans", "Today at 9:07 AM") == "Family scans"


def test_theme_override_script_and_css_are_present():
    base = Path("web/templates/base.html").read_text()
    css = Path("web/static/app.css").read_text()

    assert 'params.get("theme")' in base
    assert 'window.localStorage.setItem(storageKey, requestedTheme)' in base
    assert 'window.localStorage.removeItem(storageKey)' in base
    assert "lucide@1.16.0" in base
    assert "window.lucide.createIcons" in base
    assert "data-theme-toggle" in base
    assert "data-menu-toggle" in base
    assert "data-primary-nav" in base
    assert ':root[data-theme="dark"]' in css
    assert ':root:not([data-theme="light"])' in css


def test_dark_mode_component_tokens_cover_shared_ui():
    css = Path("web/static/app.css").read_text()

    for token in (
        "--nav-link",
        "--danger-solid",
        "--error-border",
        "--neutral-chip-bg",
        "--running-bg",
        "--cancelled-bg",
        "--progress-bg",
        "--control-active-bg",
        "--media-placeholder-bg",
        "--hover-shadow",
    ):
        assert token in css
    for rule in (
        "nav a {\n  border-bottom: 2px solid transparent;\n  color: var(--nav-link);",
        "button.danger {\n  background: var(--danger-solid);",
        ".summary-chip {\n  align-items: center;\n  background: var(--neutral-chip-bg);",
        ".status-running {\n  background: var(--running-bg);",
        ".progress {\n  background: var(--progress-bg);",
        ".section-tools .is-active {\n  background: var(--control-active-bg);",
    ):
        assert rule in css
