"""FastAPI application skeleton."""

from __future__ import annotations

import re
import secrets
import shutil
from contextlib import asynccontextmanager
from datetime import datetime
from mimetypes import guess_type
from pathlib import Path
from typing import AsyncIterator
from uuid import uuid4

from fastapi import Depends, FastAPI, File, Form, Request, UploadFile
from fastapi.responses import HTMLResponse, RedirectResponse, Response
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.middleware.sessions import SessionMiddleware

from .config import Settings, load_settings
from .database import (
    abort_job,
    create_job_with_files,
    delete_job,
    get_job,
    init_db,
    list_files_for_job,
    list_jobs,
)
from .logging import log_job_event
from .results import IMAGE_EXTENSIONS, ensure_job_zip
from .scheduler import JobScheduler, eta_label

ACTIVE_STATUSES = ("queued", "running", "cancelling")
COMPLETED_STATUSES = ("completed", "failed", "cancelled")
ALLOWED_IMAGE_EXTENSIONS = IMAGE_EXTENSIONS
BYTES_PER_GB = 1024**3

WEB_DIR = Path(__file__).resolve().parent
TEMPLATES = Jinja2Templates(directory=str(WEB_DIR / "templates"))
STATIC_DIR = WEB_DIR / "static"


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or load_settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        init_db(settings.database_path)
        scheduler = JobScheduler(
            settings.database_path,
            settings.data_dir,
            settings.max_parallel_scans,
            settings.write_debug,
            settings.debug_panel_width,
            settings.scheduler_poll_seconds,
        )
        app.state.scheduler = scheduler
        await scheduler.start()
        yield
        await scheduler.stop()

    app = FastAPI(lifespan=lifespan)
    app.state.settings = settings
    app.add_middleware(
        SessionMiddleware,
        secret_key=settings.session_secret,
        same_site="lax",
        https_only=False,
    )
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

    @app.exception_handler(LoginRedirect)
    async def login_redirect_handler(request: Request, exc: LoginRedirect) -> RedirectResponse:
        return RedirectResponse("/login", status_code=303)

    @app.get("/", response_class=HTMLResponse)
    async def index(request: Request) -> RedirectResponse:
        if is_authenticated(request):
            return RedirectResponse("/upload", status_code=303)
        return RedirectResponse("/login", status_code=303)

    @app.get("/health")
    async def health() -> dict:
        init_db(settings.database_path)
        return {
            "status": "ok",
            "data_dir": str(settings.data_dir),
            "database": "ok",
        }

    @app.get("/login", response_class=HTMLResponse)
    async def login_form(request: Request) -> HTMLResponse:
        if is_authenticated(request):
            return RedirectResponse("/upload", status_code=303)
        return TEMPLATES.TemplateResponse(request, "login.html", {"error": None})

    @app.post("/login", response_class=HTMLResponse)
    async def login(request: Request, password: str = Form("")) -> HTMLResponse:
        if settings.app_password and secrets.compare_digest(password, settings.app_password):
            request.session["authenticated"] = True
            return RedirectResponse("/upload", status_code=303)
        return TEMPLATES.TemplateResponse(
            request,
            "login.html",
            {"error": "Invalid password."},
            status_code=401,
        )

    @app.post("/logout")
    async def logout(request: Request) -> RedirectResponse:
        request.session.clear()
        return RedirectResponse("/login", status_code=303)

    @app.get("/upload", response_class=HTMLResponse)
    async def upload(request: Request, _: None = Depends(require_auth)) -> HTMLResponse:
        return render_upload(request)

    @app.post("/upload", response_class=HTMLResponse)
    async def create_upload_job(
        request: Request,
        _: None = Depends(require_auth),
        title: str = Form(""),
        files: list[UploadFile] = File(default_factory=list),
    ) -> HTMLResponse:
        settings.data_dir.mkdir(parents=True, exist_ok=True)
        free_gb = free_disk_gb(settings.data_dir)
        if free_gb < settings.min_free_disk_gb:
            return render_upload(
                request,
                error=(
                    "Uploads are paused because free disk space is below "
                    f"{settings.min_free_disk_gb:g} GB."
                ),
                status_code=507,
            )

        selected_files = [file for file in files if file.filename]
        error = validate_uploads(selected_files)
        if error:
            return render_upload(request, error=error, status_code=400)

        job_id = uuid4().hex
        job_title = normalize_title(title)
        inputs_dir = settings.data_dir / "jobs" / job_id / "inputs"
        inputs_dir.mkdir(parents=True, exist_ok=True)

        stored_paths: list[str] = []
        for index, upload_file in enumerate(selected_files, start=1):
            filename = unique_input_filename(upload_file.filename or f"scan-{index}", index)
            destination = inputs_dir / filename
            with destination.open("wb") as output:
                shutil.copyfileobj(upload_file.file, output)
            stored_paths.append(str(destination.relative_to(settings.data_dir)))

        create_job_with_files(settings.database_path, job_id, job_title, stored_paths)
        log_job_event(
            "job_created",
            job_id,
            title=job_title,
            input_count=len(stored_paths),
        )
        return RedirectResponse("/active", status_code=303)

    @app.get("/active", response_class=HTMLResponse)
    async def active_jobs(request: Request, _: None = Depends(require_auth)) -> HTMLResponse:
        jobs = prepare_jobs(request, ACTIVE_STATUSES)
        return TEMPLATES.TemplateResponse(
            request,
            "jobs.html",
            {
                "active": "active",
                "title": "Active jobs",
                "empty_message": "No active jobs.",
                "jobs": jobs,
                "status_counts": status_counts(jobs),
                "job_mode": "active",
                "auto_refresh": True,
            },
        )

    @app.get("/active/fragment", response_class=HTMLResponse)
    async def active_jobs_fragment(request: Request, _: None = Depends(require_auth)) -> HTMLResponse:
        return render_jobs_fragment(request, ACTIVE_STATUSES, "No active jobs.", auto_refresh=True)

    @app.post("/jobs/{job_id}/abort", response_class=HTMLResponse)
    async def abort_job_route(job_id: str, request: Request, _: None = Depends(require_auth)) -> HTMLResponse:
        result = abort_job(settings.database_path, job_id)
        log_job_event("job_abort_requested", job_id, result=result)
        if is_htmx(request):
            status_code = 404 if result == "not_found" else 200
            return render_jobs_fragment(
                request,
                ACTIVE_STATUSES,
                "No active jobs.",
                auto_refresh=True,
                status_code=status_code,
            )
        return RedirectResponse("/active", status_code=303)

    @app.get("/completed", response_class=HTMLResponse)
    async def completed_jobs(request: Request, _: None = Depends(require_auth)) -> HTMLResponse:
        jobs = prepare_jobs(request, COMPLETED_STATUSES)
        return TEMPLATES.TemplateResponse(
            request,
            "jobs.html",
            {
                "active": "completed",
                "title": "Completed jobs",
                "empty_message": "No completed jobs.",
                "jobs": jobs,
                "status_counts": status_counts(jobs),
                "job_mode": "completed",
            },
        )

    @app.get("/completed/fragment", response_class=HTMLResponse)
    async def completed_jobs_fragment(request: Request, _: None = Depends(require_auth)) -> HTMLResponse:
        return render_jobs_fragment(request, COMPLETED_STATUSES, "No completed jobs.")

    @app.get("/jobs/{job_id}", response_class=HTMLResponse)
    async def job_detail(job_id: str, request: Request, _: None = Depends(require_auth)) -> HTMLResponse:
        job = get_job(settings.database_path, job_id)
        if job is None:
            return HTMLResponse("Job not found.", status_code=404)
        if job["status"] not in COMPLETED_STATUSES:
            return RedirectResponse("/active", status_code=303)
        job_data = dict(job)
        decorate_job_dates(job_data)

        return TEMPLATES.TemplateResponse(
            request,
            "job_detail.html",
            {
                "active": "completed",
                "job": job_data,
                **build_job_result(settings.data_dir, settings.database_path, job_id),
            },
        )

    @app.get("/jobs/{job_id}/media/{relative_path:path}")
    async def job_media(
        job_id: str,
        relative_path: str,
        request: Request,
        _: None = Depends(require_auth),
    ):
        job = get_job(settings.database_path, job_id)
        if job is None:
            return HTMLResponse("Job not found.", status_code=404)
        file_path = resolve_job_file(settings.data_dir, job_id, relative_path)
        if file_path is None or not file_path.is_file():
            return HTMLResponse("File not found.", status_code=404)
        media_type = guess_type(file_path.name)[0] or "application/octet-stream"
        return Response(file_path.read_bytes(), media_type=media_type)

    @app.get("/jobs/{job_id}/download/metadata")
    async def download_metadata(
        job_id: str,
        request: Request,
        _: None = Depends(require_auth),
    ):
        job = get_job(settings.database_path, job_id)
        if job is None or job["status"] not in COMPLETED_STATUSES:
            return HTMLResponse("Job not found.", status_code=404)
        metadata_path = resolve_job_file(settings.data_dir, job_id, "output/metadata.csv")
        if metadata_path is None or not metadata_path.is_file():
            return HTMLResponse("Metadata not found.", status_code=404)
        return file_download_response(metadata_path, "metadata.csv", "text/csv")

    @app.get("/jobs/{job_id}/download/photos.zip")
    async def download_photos_zip(
        job_id: str,
        request: Request,
        _: None = Depends(require_auth),
    ):
        job = get_job(settings.database_path, job_id)
        if job is None or job["status"] != "completed":
            return HTMLResponse("Download not found.", status_code=404)
        try:
            ensure_job_zip(settings.data_dir, job_id)
        except ValueError:
            return HTMLResponse("Download not found.", status_code=404)
        zip_path = resolve_job_file(settings.data_dir, job_id, "downloads/photos.zip")
        if zip_path is None or not zip_path.is_file():
            return HTMLResponse("Download not found.", status_code=404)
        return file_download_response(zip_path, "photos.zip", "application/zip")

    @app.post("/jobs/{job_id}/delete", response_class=HTMLResponse)
    async def delete_job_route(job_id: str, request: Request, _: None = Depends(require_auth)) -> HTMLResponse:
        job = get_job(settings.database_path, job_id)
        if job is not None and job["status"] not in COMPLETED_STATUSES:
            if is_htmx(request):
                return render_jobs_fragment(
                    request,
                    ACTIVE_STATUSES,
                    "No active jobs.",
                    auto_refresh=True,
                    status_code=409,
                    notice="Abort the job before deleting it.",
                )
            return HTMLResponse("Abort the job before deleting it.", status_code=409)

        if job is None:
            result = "not_found"
        else:
            job_dir = settings.data_dir / "jobs" / job_id
            if job_dir.exists():
                try:
                    shutil.rmtree(job_dir)
                except OSError:
                    if is_htmx(request):
                        return render_jobs_fragment(
                            request,
                            COMPLETED_STATUSES,
                            "No completed jobs.",
                            status_code=500,
                            notice="Job files could not be deleted. The job was kept.",
                        )
                    return HTMLResponse(
                        "Job files could not be deleted. The job was kept.",
                        status_code=500,
                    )

            result = delete_job(settings.database_path, job_id)
            log_job_event("job_deleted", job_id, result=result)

        if is_htmx(request):
            status_code = 404 if result == "not_found" else 200
            return render_jobs_fragment(
                request,
                COMPLETED_STATUSES,
                "No completed jobs.",
                status_code=status_code,
            )
        return RedirectResponse("/completed", status_code=303)

    return app


def prepare_jobs(request: Request, statuses: tuple[str, ...]) -> list[dict]:
    settings: Settings = request.app.state.settings
    scheduler = getattr(request.app.state, "scheduler", None)
    average_seconds = scheduler.average_seconds_per_scan if scheduler else None
    jobs = []
    for row in list_jobs(settings.database_path, statuses):
        job = dict(row)
        decorate_job_dates(job)
        job["files"] = [dict(file) for file in list_files_for_job(settings.database_path, row["id"])]
        for file in job["files"]:
            file_path = Path(file["path"])
            file["name"] = file_path.name
            file["url"] = f"/jobs/{row['id']}/media/inputs/{file_path.name}"
        total_files = int(job["total_files"] or 0)
        completed_files = int(job["completed_files"] or 0)
        job["progress_label"] = f"{completed_files}/{total_files}"
        job["progress_percent"] = round((completed_files / total_files) * 100) if total_files else 0
        job["eta"] = eta_label(row["status"], total_files, completed_files, average_seconds)
        photos_dir = settings.data_dir / "jobs" / row["id"] / "output" / "photos"
        job["output_count"] = sum(
            1
            for path in (photos_dir.iterdir() if photos_dir.is_dir() else [])
            if path.is_file() and path.suffix.lower() in ALLOWED_IMAGE_EXTENSIONS
        )
        job["zip_exists"] = (settings.data_dir / "jobs" / row["id"] / "downloads" / "photos.zip").is_file()
        jobs.append(job)
    return jobs


def render_jobs_fragment(
    request: Request,
    statuses: tuple[str, ...],
    empty_message: str,
    auto_refresh: bool = False,
    status_code: int = 200,
    notice: str | None = None,
) -> HTMLResponse:
    jobs = prepare_jobs(request, statuses)
    return TEMPLATES.TemplateResponse(
        request,
        "_jobs_table.html",
        {
            "empty_message": empty_message,
            "jobs": jobs,
            "status_counts": status_counts(jobs),
            "job_mode": "active" if auto_refresh else "completed",
            "auto_refresh": auto_refresh,
            "notice": notice,
        },
        status_code=status_code,
    )


def status_counts(jobs: list[dict]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for job in jobs:
        status = str(job["status"])
        counts[status] = counts.get(status, 0) + 1
    return counts


def decorate_job_dates(job: dict) -> None:
    job["created_at_display"] = format_display_datetime(job.get("created_at"))
    job["completed_at_display"] = format_display_datetime(job.get("completed_at"))
    job["completed_duration_display"] = format_completion_duration(job.get("created_at"), job.get("completed_at"))
    job["title_display"] = format_job_title(job.get("title"), job["created_at_display"])


def format_job_title(title: str | None, created_at_display: str) -> str:
    cleaned = title.strip() if title else ""
    return cleaned or created_at_display or "Untitled upload"


def format_display_datetime(value: str | None, now: datetime | None = None) -> str:
    if not value:
        return ""
    parsed = parse_database_datetime(value)
    if parsed is None:
        return trim_seconds(value)

    now = now or datetime.now()
    days_ago = (now.date() - parsed.date()).days
    time_label = parsed.strftime("%I:%M %p").lstrip("0")
    if days_ago == 0:
        return f"Today at {time_label}"
    if days_ago == 1:
        return f"Yesterday at {time_label}"
    if days_ago > 1:
        return f"{days_ago}d ago at {time_label}"
    if days_ago == -1:
        return f"Tomorrow at {time_label}"
    return f"In {abs(days_ago)}d at {time_label}"


def format_completion_duration(start_value: str | None, end_value: str | None) -> str:
    if not start_value or not end_value:
        return ""
    start = parse_database_datetime(start_value)
    end = parse_database_datetime(end_value)
    if start is None or end is None:
        return ""
    total_seconds = max(0, round((end - start).total_seconds()))
    minutes, seconds = divmod(total_seconds, 60)
    if minutes:
        return f"{minutes} min {seconds} {pluralize('second', seconds)}"
    return f"{seconds} {pluralize('second', seconds)}"


def pluralize(word: str, count: int) -> str:
    return word if count == 1 else f"{word}s"


def parse_database_datetime(value: str) -> datetime | None:
    normalized = value.strip().replace("Z", "+00:00")
    try:
        parsed = datetime.fromisoformat(normalized)
    except ValueError:
        return None
    if parsed.tzinfo is not None:
        parsed = parsed.astimezone().replace(tzinfo=None)
    return parsed


def trim_seconds(value: str) -> str:
    return re.sub(r"(\d{1,2}:\d{2}):\d{2}", r"\1", value)


def build_job_result(data_dir: Path, database_path: Path, job_id: str) -> dict:
    job_dir = data_dir / "jobs" / job_id
    inputs_dir = job_dir / "inputs"
    output_dir = job_dir / "output"
    photos_dir = output_dir / "photos"
    debug_dir = output_dir / "debug"
    metadata_path = output_dir / "metadata.csv"
    zip_path = job_dir / "downloads" / "photos.zip"
    job = get_job(database_path, job_id)

    if job is not None and job["status"] == "completed":
        try:
            ensure_job_zip(data_dir, job_id)
        except ValueError:
            pass

    input_files = [
        media_item(data_dir, path, job_id)
        for path in sorted(inputs_dir.iterdir() if inputs_dir.is_dir() else [])
        if path.is_file()
    ]
    output_photos = [
        media_item(data_dir, path, job_id)
        for path in sorted(photos_dir.iterdir() if photos_dir.is_dir() else [])
        if path.is_file() and path.suffix.lower() in ALLOWED_IMAGE_EXTENSIONS
    ]
    debug_images = [
        media_item(data_dir, path, job_id)
        for path in sorted(debug_dir.glob("*.png") if debug_dir.is_dir() else [])
        if path.is_file()
    ]

    return {
        "input_files": input_files,
        "output_photos": output_photos,
        "debug_images": debug_images,
        "metadata_exists": metadata_path.is_file(),
        "zip_exists": zip_path.is_file(),
    }


def media_item(data_dir: Path, path: Path, job_id: str) -> dict:
    job_dir = data_dir / "jobs" / job_id
    return {
        "name": path.name,
        "url": f"/jobs/{job_id}/media/{path.relative_to(job_dir).as_posix()}",
    }


def resolve_job_file(data_dir: Path, job_id: str, relative_path: str) -> Path | None:
    job_dir = (data_dir / "jobs" / job_id).resolve()
    candidate = (job_dir / relative_path).resolve()
    try:
        candidate.relative_to(job_dir)
    except ValueError:
        return None
    return candidate


def file_download_response(path: Path, filename: str, media_type: str) -> Response:
    return Response(
        path.read_bytes(),
        media_type=media_type,
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


def is_authenticated(request: Request) -> bool:
    settings: Settings = request.app.state.settings
    if not settings.app_password:
        return True
    return request.session.get("authenticated") is True


def is_htmx(request: Request) -> bool:
    return request.headers.get("HX-Request", "").lower() == "true"


async def require_auth(request: Request) -> None:
    if not is_authenticated(request):
        raise LoginRedirect()


class LoginRedirect(Exception):
    pass


def render_upload(
    request: Request,
    error: str | None = None,
    status_code: int = 200,
) -> HTMLResponse:
    return TEMPLATES.TemplateResponse(
        request,
        "upload.html",
        {"active": "upload", "error": error},
        status_code=status_code,
    )


def free_disk_gb(path: Path) -> float:
    return shutil.disk_usage(path).free / BYTES_PER_GB


def validate_uploads(files: list[UploadFile]) -> str | None:
    if not files:
        return "Select at least one image file."

    for file in files:
        suffix = Path(file.filename or "").suffix.lower()
        if suffix not in ALLOWED_IMAGE_EXTENSIONS:
            return "Only image files are supported."
        if file.content_type and not file.content_type.startswith("image/"):
            return "Only image files are supported."
    return None


def normalize_title(title: str) -> str | None:
    cleaned = title.strip()
    if cleaned:
        return cleaned
    return None


def unique_input_filename(original_name: str, index: int) -> str:
    path = Path(original_name)
    stem = re.sub(r"[^A-Za-z0-9._-]+", "-", path.stem).strip(".-") or "scan"
    suffix = path.suffix.lower()
    return f"{index:03d}-{stem}{suffix}"
