import asyncio
import sqlite3
from pathlib import Path

import httpx

from web import create_app
from web.config import Settings


def make_app(data_dir: Path, min_free_disk_gb: float = 0):
    settings = Settings(
        app_password="secret",
        session_secret="test-session-secret",
        data_dir=data_dir,
        min_free_disk_gb=min_free_disk_gb,
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


def test_upload_one_image_creates_queued_job_and_input_file(tmp_path):
    async def run():
        app = make_app(tmp_path)
        async with app.router.lifespan_context(app):
            client = await authenticated_client(app)
            try:
                response = await client.post(
                    "/upload",
                    data={"title": "Family scans"},
                    files={"files": ("scan.jpg", b"jpeg-bytes", "image/jpeg")},
                    follow_redirects=False,
                )
            finally:
                await client.aclose()
        return response

    response = asyncio.run(run())

    assert response.status_code == 303
    assert response.headers["location"] == "/active"
    job = rows(tmp_path / "app.db", "SELECT id, title, status FROM jobs")[0]
    file_row = rows(tmp_path / "app.db", "SELECT job_id, role, path, status FROM files")[0]
    assert job["title"] == "Family scans"
    assert job["status"] == "queued"
    assert file_row["job_id"] == job["id"]
    assert file_row["role"] == "input"
    assert file_row["status"] in {"pending", "running", "completed"}
    assert (tmp_path / file_row["path"]).read_bytes() == b"jpeg-bytes"


def test_upload_multiple_images_creates_one_job_with_all_inputs(tmp_path):
    async def run():
        app = make_app(tmp_path)
        async with app.router.lifespan_context(app):
            client = await authenticated_client(app)
            try:
                response = await client.post(
                    "/upload",
                    data={"title": "Batch"},
                    files=[
                        ("files", ("first.png", b"png", "image/png")),
                        ("files", ("second.tif", b"tiff", "image/tiff")),
                    ],
                    follow_redirects=False,
                )
            finally:
                await client.aclose()
        return response

    response = asyncio.run(run())

    assert response.status_code == 303
    jobs = rows(tmp_path / "app.db", "SELECT id, status FROM jobs")
    files = rows(tmp_path / "app.db", "SELECT job_id, path FROM files ORDER BY path")
    assert len(jobs) == 1
    assert jobs[0]["status"] == "queued"
    assert len(files) == 2
    assert {file["job_id"] for file in files} == {jobs[0]["id"]}
    assert {Path(file["path"]).name for file in files} == {"001-first.png", "002-second.tif"}


def test_empty_upload_is_rejected(tmp_path):
    async def run():
        app = make_app(tmp_path)
        async with app.router.lifespan_context(app):
            client = await authenticated_client(app)
            try:
                response = await client.post(
                    "/upload",
                    data={"title": "No files"},
                    follow_redirects=False,
                )
            finally:
                await client.aclose()
        return response

    response = asyncio.run(run())

    assert response.status_code == 400
    assert "Select at least one image file." in response.text
    assert rows(tmp_path / "app.db", "SELECT * FROM jobs") == []


def test_blank_title_is_stored_null_and_displayed_from_created_at(tmp_path):
    async def run():
        app = make_app(tmp_path)
        async with app.router.lifespan_context(app):
            client = await authenticated_client(app)
            try:
                response = await client.post(
                    "/upload",
                    data={"title": "   "},
                    files={"files": ("scan.webp", b"webp", "image/webp")},
                    follow_redirects=False,
                )
                active = await client.get("/active")
            finally:
                await client.aclose()
        return response, active

    response, active = asyncio.run(run())

    assert response.status_code == 303
    job = rows(tmp_path / "app.db", "SELECT title FROM jobs")[0]
    assert job["title"] is None
    assert "Today at" in active.text
    assert "Untitled upload" not in active.text


def test_low_disk_space_rejects_upload(monkeypatch, tmp_path):
    monkeypatch.setattr("web.app.free_disk_gb", lambda path: 0.25)

    async def run():
        app = make_app(tmp_path, min_free_disk_gb=1)
        async with app.router.lifespan_context(app):
            client = await authenticated_client(app)
            try:
                response = await client.post(
                    "/upload",
                    data={"title": "Blocked"},
                    files={"files": ("scan.jpg", b"jpeg", "image/jpeg")},
                    follow_redirects=False,
                )
            finally:
                await client.aclose()
        return response

    response = asyncio.run(run())

    assert response.status_code == 507
    assert "free disk space is below 1 GB" in response.text
    assert rows(tmp_path / "app.db", "SELECT * FROM jobs") == []
    assert not (tmp_path / "jobs").exists()
