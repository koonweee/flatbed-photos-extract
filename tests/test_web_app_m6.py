import asyncio
import sqlite3
import time
import zipfile
from pathlib import Path
from types import SimpleNamespace

import httpx

from extractor import write_metadata
from web import create_app
from web.config import Settings
from web.database import create_job_with_files, init_db


PNG_BYTES = (
    b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01"
    b"\x08\x02\x00\x00\x00\x90wS\xde\x00\x00\x00\x0cIDATx\x9cc\xf8\xff\xff?"
    b"\x00\x05\xfe\x02\xfeA\xde\xfc\x96\x00\x00\x00\x00IEND\xaeB`\x82"
)


def make_app(data_dir: Path, write_debug: bool = True):
    settings = Settings(
        app_password="secret",
        session_secret="test-session-secret",
        data_dir=data_dir,
        min_free_disk_gb=0,
        max_parallel_scans=1,
        write_debug=write_debug,
        scheduler_poll_seconds=0.01,
    )
    return create_app(settings)


async def authenticated_client(app):
    transport = httpx.ASGITransport(app=app)
    client = httpx.AsyncClient(transport=transport, base_url="http://test")
    await client.post("/login", data={"password": "secret"}, follow_redirects=False)
    return client


async def upload(client, title: str, filenames: list[str]):
    files = [
        ("files", (filename, PNG_BYTES, "image/png"))
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


def rows(database_path: Path, query: str):
    with sqlite3.connect(database_path) as connection:
        connection.row_factory = sqlite3.Row
        return connection.execute(query).fetchall()


def fake_process_scan(input_path, photos_dir, debug_dir, source_stem, write_debug=True, debug_panel_width=None):
    photos_dir.mkdir(parents=True, exist_ok=True)
    outputs = []
    for index in range(1, 3):
        output_path = photos_dir / f"{source_stem}-{index:02d}.png"
        output_path.write_bytes(PNG_BYTES)
        outputs.append(output_path)
    if write_debug:
        debug_dir.mkdir(parents=True, exist_ok=True)
        (debug_dir / f"{source_stem}_debug.png").write_bytes(PNG_BYTES)
    return SimpleNamespace(
        detections=[
            fake_detection(Path(input_path).name, source_stem, index, output_path.name)
            for index, output_path in enumerate(outputs, start=1)
        ]
    )


def fake_detection(source_file: str, source_stem: str, index: int, filename: str) -> dict:
    return {
        "source_file": source_file,
        "source_stem": source_stem,
        "source_photo_index": index,
        "filename": filename,
        "bbox": (0, 0, 10, 10),
        "quad": ((0, 0), (10, 0), (10, 10), (0, 10)),
        "width": 10,
        "height": 10,
        "trimmed_width": 10,
        "trimmed_height": 10,
        "trim_left": 0,
        "trim_top": 0,
        "trim_right": 0,
        "trim_bottom": 0,
        "dark_edge_ratio_before": 0.0,
        "dark_edge_ratio_after": 0.0,
        "angle": 0.0,
        "orientation_deg": 0,
        "orientation_score": 1.0,
        "orientation_margin": 1.0,
        "face_count": 0,
        "orientation_method": "test",
        "needs_review": False,
        "orientation_scores": [{"rotation": 0, "score": 1.0, "face_count": 0}],
        "yunet_orientation_deg": 0,
        "yunet_orientation_score": 1.0,
        "yunet_orientation_margin": 1.0,
        "yunet_face_count": 0,
        "yunet_orientation_scores": [{"rotation": 0, "score": 1.0, "face_count": 0}],
        "gyroscope_orientation_deg": 0,
        "gyroscope_orientation_score": 1.0,
        "gyroscope_orientation_margin": 1.0,
        "gyroscope_orientation_scores": [{"rotation": 0, "score": 1.0}],
        "refined": False,
        "refine_reason": "test",
        "area": 100.0,
    }


def test_completed_list_links_to_detail_with_debug_enabled(monkeypatch, tmp_path):
    monkeypatch.setattr("web.scheduler.process_scan", fake_process_scan)

    async def run():
        app = make_app(tmp_path, write_debug=True)
        async with app.router.lifespan_context(app):
            client = await authenticated_client(app)
            try:
                await upload(client, "Family scans", ["scan-a.png", "scan-b.png"])
                await wait_for(
                    lambda: rows(tmp_path / "app.db", "SELECT status FROM jobs")[0]["status"] == "completed",
                    timeout=3,
                )
                job_id = rows(tmp_path / "app.db", "SELECT id FROM jobs")[0]["id"]
                completed = await client.get("/completed")
                detail = await client.get(f"/jobs/{job_id}")
                input_media = await client.get(f"/jobs/{job_id}/media/inputs/001-scan-a.png")
            finally:
                await client.aclose()
        return completed, detail, input_media, job_id

    completed, detail, input_media, job_id = asyncio.run(run())

    assert completed.status_code == 200
    assert f'data-href="/jobs/{job_id}"' in completed.text
    assert f'href="/jobs/{job_id}"' in completed.text
    assert detail.status_code == 200
    assert "Input scans" in detail.text
    assert "Extracted photos" in detail.text
    assert "scan-a-01.png" in detail.text
    assert "scan-b-02.png" in detail.text
    assert "Debug" in detail.text
    assert "001-scan-a_debug.png" in detail.text
    assert "/download/metadata" in detail.text
    assert "/download/photos.zip" in detail.text
    assert input_media.status_code == 200
    assert input_media.content == PNG_BYTES


def test_detail_omits_debug_viewer_when_debug_disabled(monkeypatch, tmp_path):
    monkeypatch.setattr("web.scheduler.process_scan", fake_process_scan)

    async def run():
        app = make_app(tmp_path, write_debug=False)
        async with app.router.lifespan_context(app):
            client = await authenticated_client(app)
            try:
                await upload(client, "No debug", ["scan.png"])
                await wait_for(
                    lambda: rows(tmp_path / "app.db", "SELECT status FROM jobs")[0]["status"] == "completed",
                    timeout=3,
                )
                job_id = rows(tmp_path / "app.db", "SELECT id FROM jobs")[0]["id"]
                return await client.get(f"/jobs/{job_id}")
            finally:
                await client.aclose()

    detail = asyncio.run(run())

    assert detail.status_code == 200
    assert "Extracted photos" in detail.text
    assert "Debug" not in detail.text
    assert "_debug.png" not in detail.text


def test_completed_job_zip_contains_output_photos_and_metadata(monkeypatch, tmp_path):
    monkeypatch.setattr("web.scheduler.process_scan", fake_process_scan)

    async def run():
        app = make_app(tmp_path, write_debug=True)
        async with app.router.lifespan_context(app):
            client = await authenticated_client(app)
            try:
                await upload(client, "Zip me", ["first.png", "second.png"])
                await wait_for(
                    lambda: rows(tmp_path / "app.db", "SELECT status FROM jobs")[0]["status"] == "completed",
                    timeout=3,
                )
                job_id = rows(tmp_path / "app.db", "SELECT id FROM jobs")[0]["id"]
                response = await client.get(f"/jobs/{job_id}/download/photos.zip")
            finally:
                await client.aclose()
        return response, job_id

    response, job_id = asyncio.run(run())
    zip_path = tmp_path / "jobs" / job_id / "downloads" / "photos.zip"

    assert response.status_code == 200
    assert zip_path.is_file()
    with zipfile.ZipFile(zip_path) as archive:
        names = set(archive.namelist())
        assert {
            "photos/001-first-01.png",
            "photos/001-first-02.png",
            "photos/002-second-01.png",
            "photos/002-second-02.png",
            "metadata.csv",
        }.issubset(names)
        metadata = archive.read("metadata.csv").decode()
    assert "filename" in metadata
    assert "001-first-01.png" in metadata


def test_restart_preserves_existing_metadata_when_pending_scan_completes(monkeypatch, tmp_path):
    monkeypatch.setattr("web.scheduler.process_scan", fake_process_scan)

    database_path = tmp_path / "app.db"
    init_db(database_path)
    job_dir = tmp_path / "jobs" / "restart-job"
    first_input = job_dir / "inputs" / "001-first.png"
    second_input = job_dir / "inputs" / "002-second.png"
    first_output = job_dir / "output" / "photos" / "001-first-01.png"
    first_input.parent.mkdir(parents=True)
    first_output.parent.mkdir(parents=True)
    first_input.write_bytes(PNG_BYTES)
    second_input.write_bytes(PNG_BYTES)
    first_output.write_bytes(PNG_BYTES)
    write_metadata(
        job_dir / "output" / "metadata.csv",
        [fake_detection("001-first.png", "001-first", 1, "001-first-01.png")],
    )
    create_job_with_files(
        database_path,
        "restart-job",
        "Restart job",
        [str(first_input.relative_to(tmp_path)), str(second_input.relative_to(tmp_path))],
    )
    with sqlite3.connect(database_path) as connection:
        connection.execute(
            "UPDATE files SET status = 'completed', completed_at = CURRENT_TIMESTAMP WHERE path = ?",
            (str(first_input.relative_to(tmp_path)),),
        )

    async def run():
        app = make_app(tmp_path, write_debug=True)
        async with app.router.lifespan_context(app):
            await wait_for(
                lambda: rows(database_path, "SELECT status FROM jobs")[0]["status"] == "completed",
                timeout=3,
            )

    asyncio.run(run())

    metadata = (job_dir / "output" / "metadata.csv").read_text()
    zip_path = job_dir / "downloads" / "photos.zip"
    assert "001-first-01.png" in metadata
    assert "002-second-01.png" in metadata
    assert zip_path.is_file()
    with zipfile.ZipFile(zip_path) as archive:
        names = set(archive.namelist())
        zipped_metadata = archive.read("metadata.csv").decode()
    assert "photos/001-first-01.png" in names
    assert "photos/002-second-01.png" in names
    assert "001-first-01.png" in zipped_metadata
    assert "002-second-01.png" in zipped_metadata


def test_row_keydown_ignores_interactive_controls():
    script = Path("web/templates/base.html").read_text()

    assert script.count('event.target.closest("a, button, form, input")') == 2


def test_download_routes_reject_symlink_escape(tmp_path):
    database_path = tmp_path / "app.db"
    init_db(database_path)
    job_dir = tmp_path / "jobs" / "escape-job"
    input_path = job_dir / "inputs" / "scan.png"
    metadata_link = job_dir / "output" / "metadata.csv"
    downloads_link = job_dir / "downloads"
    outside_dir = tmp_path / "outside"
    outside_dir.mkdir()
    input_path.parent.mkdir(parents=True)
    metadata_link.parent.mkdir(parents=True)
    input_path.write_bytes(PNG_BYTES)
    (outside_dir / "metadata.csv").write_text("outside")
    metadata_link.symlink_to(outside_dir / "metadata.csv")
    downloads_link.symlink_to(outside_dir, target_is_directory=True)
    create_job_with_files(database_path, "escape-job", "Escape", [str(input_path.relative_to(tmp_path))])
    with sqlite3.connect(database_path) as connection:
        connection.execute("UPDATE jobs SET status = 'completed', completed_at = CURRENT_TIMESTAMP")
        connection.execute("UPDATE files SET status = 'completed', completed_at = CURRENT_TIMESTAMP")

    async def run():
        app = make_app(tmp_path, write_debug=True)
        async with app.router.lifespan_context(app):
            client = await authenticated_client(app)
            try:
                metadata = await client.get("/jobs/escape-job/download/metadata")
                archive = await client.get("/jobs/escape-job/download/photos.zip")
            finally:
                await client.aclose()
        return metadata, archive

    metadata, archive = asyncio.run(run())

    assert metadata.status_code == 404
    assert archive.status_code == 404
    assert not (outside_dir / "photos.zip").exists()


def test_detail_css_contains_mobile_gallery_rules():
    css = Path("web/static/app.css").read_text()

    assert "@media (max-width: 720px)" in css
    assert ".gallery" in css
    assert "grid-template-columns" in css
