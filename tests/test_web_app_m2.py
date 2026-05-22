import asyncio
from pathlib import Path

import httpx

from web import create_app
from web.config import Settings, load_settings
from web.database import insert_job


def make_app(data_dir: Path):
    settings = Settings(
        app_password="secret",
        session_secret="test-session-secret",
        data_dir=data_dir,
    )
    return create_app(settings)


def test_unauthenticated_request_redirects_to_login(tmp_path):
    async def run():
        app = make_app(tmp_path)
        transport = httpx.ASGITransport(app=app)
        async with app.router.lifespan_context(app):
            async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
                response = await client.get("/upload", follow_redirects=False)
        return response

    response = asyncio.run(run())

    assert response.status_code == 303
    assert response.headers["location"] == "/login"


def test_wrong_password_is_rejected(tmp_path):
    async def run():
        app = make_app(tmp_path)
        transport = httpx.ASGITransport(app=app)
        async with app.router.lifespan_context(app):
            async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
                response = await client.post(
                    "/login",
                    data={"password": "wrong"},
                    follow_redirects=False,
                )
        return response

    response = asyncio.run(run())

    assert response.status_code == 401
    assert "Invalid password." in response.text


def test_correct_password_creates_usable_session(tmp_path):
    async def run():
        app = make_app(tmp_path)
        transport = httpx.ASGITransport(app=app)
        async with app.router.lifespan_context(app):
            async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
                login = await client.post(
                    "/login",
                    data={"password": "secret"},
                    follow_redirects=False,
                )
                upload = await client.get("/upload")
                active = await client.get("/active")
                completed = await client.get("/completed")
        return login, upload, active, completed

    login, upload, active, completed = asyncio.run(run())

    assert login.status_code == 303
    assert "session=" in login.headers["set-cookie"]
    assert upload.status_code == 200
    assert "Upload scans" in upload.text
    assert active.status_code == 200
    assert "Active jobs" in active.text
    assert completed.status_code == 200
    assert "Completed jobs" in completed.text


def test_database_initializes_and_persists_jobs(tmp_path):
    async def initialize():
        app = make_app(tmp_path)
        transport = httpx.ASGITransport(app=app)
        async with app.router.lifespan_context(app):
            async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
                response = await client.post(
                    "/login",
                    data={"password": "secret"},
                    follow_redirects=False,
                )
        return response

    response = asyncio.run(initialize())
    assert response.status_code == 303

    database_path = tmp_path / "app.db"
    assert database_path.exists()
    insert_job(database_path, "job-1", "First job", "completed")

    async def restart_and_fetch_completed():
        app = make_app(tmp_path)
        transport = httpx.ASGITransport(app=app)
        async with app.router.lifespan_context(app):
            async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
                response = await client.post(
                    "/login",
                    data={"password": "secret"},
                    follow_redirects=False,
                )
                assert response.status_code == 303
                completed = await client.get("/completed")
        return completed

    completed = asyncio.run(restart_and_fetch_completed())

    assert completed.status_code == 200
    assert "First job" in completed.text


def test_missing_app_password_disables_auth(monkeypatch, tmp_path):
    monkeypatch.delenv("APP_PASSWORD", raising=False)
    monkeypatch.delenv("SESSION_SECRET", raising=False)
    monkeypatch.setenv("DATA_DIR", str(tmp_path))

    settings = load_settings()

    assert settings.app_password == ""


def test_passwordless_app_skips_login(tmp_path):
    async def run():
        app = create_app(Settings(data_dir=tmp_path))
        transport = httpx.ASGITransport(app=app)
        async with app.router.lifespan_context(app):
            async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
                response = await client.get("/upload", follow_redirects=False)
        return response

    response = asyncio.run(run())

    assert response.status_code == 200
    assert "Upload scans" in response.text


def test_settings_loads_dotenv_without_overriding_environment(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("APP_PASSWORD", raising=False)
    monkeypatch.delenv("DATA_DIR", raising=False)
    (tmp_path / ".env").write_text("APP_PASSWORD=from-file\nDATA_DIR=from-dotenv\n")

    settings = load_settings()

    assert settings.app_password == "from-file"
    assert settings.data_dir == Path("from-dotenv")

    monkeypatch.setenv("APP_PASSWORD", "from-env")

    settings = load_settings()

    assert settings.app_password == "from-env"
