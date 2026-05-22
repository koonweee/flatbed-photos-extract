"""Configuration for the FastAPI web app."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Settings:
    app_password: str = ""
    session_secret: str = "passwordless-local-session-secret"
    data_dir: Path = Path("data")
    min_free_disk_gb: float = 10.0
    max_parallel_scans: int = 2
    write_debug: bool = True
    debug_panel_width: int | None = None
    scheduler_poll_seconds: float = 0.1

    @property
    def database_path(self) -> Path:
        return self.data_dir / "app.db"


class ConfigurationError(RuntimeError):
    """Raised when required web app configuration is missing."""


def load_settings() -> Settings:
    load_dotenv()
    app_password = os.environ.get("APP_PASSWORD", "")
    session_secret = os.environ.get("SESSION_SECRET") or app_password or "passwordless-local-session-secret"
    data_dir = Path(os.environ.get("DATA_DIR", "data")).expanduser()
    min_free_disk_gb = float(os.environ.get("MIN_FREE_DISK_GB", "10"))
    max_parallel_scans = max(1, int(os.environ.get("MAX_PARALLEL_SCANS", "2")))
    write_debug = os.environ.get("WRITE_DEBUG", "true").lower() not in {"0", "false", "no", "off"}
    debug_panel_width_value = os.environ.get("DEBUG_PANEL_WIDTH")
    debug_panel_width = int(debug_panel_width_value) if debug_panel_width_value else None
    return Settings(
        app_password=app_password,
        session_secret=session_secret,
        data_dir=data_dir,
        min_free_disk_gb=min_free_disk_gb,
        max_parallel_scans=max_parallel_scans,
        write_debug=write_debug,
        debug_panel_width=debug_panel_width,
    )


def load_dotenv(path: Path = Path(".env")) -> None:
    if not path.exists():
        return

    for line in path.read_text().splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        key = key.strip()
        value = value.strip()
        if not key:
            continue
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
            value = value[1:-1]
        os.environ.setdefault(key, value)
