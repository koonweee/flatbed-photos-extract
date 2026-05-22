"""Helpers for completed job result files."""

from __future__ import annotations

import zipfile
from pathlib import Path


IMAGE_EXTENSIONS = {".bmp", ".gif", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}


def ensure_job_zip(data_dir: Path, job_id: str) -> Path:
    job_dir = (data_dir / "jobs" / job_id).resolve()
    photos_dir = job_dir / "output" / "photos"
    metadata_path = job_dir / "output" / "metadata.csv"
    zip_path = job_dir / "downloads" / "photos.zip"
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    _ensure_inside_job(job_dir, zip_path.parent)

    photo_paths = [
        path
        for path in sorted(photos_dir.iterdir() if photos_dir.is_dir() else [])
        if _is_inside_job_file(job_dir, path) and path.suffix.lower() in IMAGE_EXTENSIONS
    ]
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for photo_path in photo_paths:
            archive.write(photo_path, f"photos/{photo_path.name}")
        if _is_inside_job_file(job_dir, metadata_path):
            archive.write(metadata_path, "metadata.csv")
    return zip_path


def _is_inside_job_file(job_dir: Path, path: Path) -> bool:
    if not path.is_file():
        return False
    try:
        _ensure_inside_job(job_dir, path)
    except ValueError:
        return False
    return True


def _ensure_inside_job(job_dir: Path, path: Path) -> None:
    path.resolve().relative_to(job_dir)
