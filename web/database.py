"""SQLite persistence for web jobs and files."""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Iterable


SCHEMA = """
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS jobs (
    id TEXT PRIMARY KEY,
    title TEXT NOT NULL,
    status TEXT NOT NULL,
    error_message TEXT,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    started_at TEXT,
    completed_at TEXT
);

CREATE TABLE IF NOT EXISTS files (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    job_id TEXT NOT NULL,
    role TEXT NOT NULL,
    path TEXT NOT NULL,
    status TEXT NOT NULL,
    error_message TEXT,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    started_at TEXT,
    completed_at TEXT,
    FOREIGN KEY (job_id) REFERENCES jobs(id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_jobs_status_created_at
    ON jobs(status, created_at);

CREATE INDEX IF NOT EXISTS idx_files_job_id
    ON files(job_id);
"""


def connect(database_path: Path) -> sqlite3.Connection:
    connection = sqlite3.connect(database_path)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    return connection


def init_db(database_path: Path) -> None:
    database_path.parent.mkdir(parents=True, exist_ok=True)
    with connect(database_path) as connection:
        connection.executescript(SCHEMA)
        migrate_schema(connection)


def migrate_schema(connection: sqlite3.Connection) -> None:
    add_columns(
        connection,
        "jobs",
        {
            "error_message": "TEXT",
            "started_at": "TEXT",
            "completed_at": "TEXT",
        },
    )
    add_columns(
        connection,
        "files",
        {
            "error_message": "TEXT",
            "started_at": "TEXT",
            "completed_at": "TEXT",
        },
    )


def add_columns(connection: sqlite3.Connection, table: str, columns: dict[str, str]) -> None:
    existing = {
        row["name"]
        for row in connection.execute(f"PRAGMA table_info({table})").fetchall()
    }
    for name, definition in columns.items():
        if name not in existing:
            connection.execute(f"ALTER TABLE {table} ADD COLUMN {name} {definition}")


def list_jobs(database_path: Path, statuses: Iterable[str]) -> list[sqlite3.Row]:
    status_values = list(statuses)
    placeholders = ", ".join("?" for _ in status_values)
    query = (
        "SELECT id, title, status, error_message, created_at, updated_at, started_at, completed_at, "
        "(SELECT COUNT(*) FROM files WHERE files.job_id = jobs.id AND role = 'input') AS total_files, "
        "(SELECT COUNT(*) FROM files WHERE files.job_id = jobs.id AND role = 'input' AND status = 'completed') AS completed_files, "
        "(SELECT COUNT(*) FROM files WHERE files.job_id = jobs.id AND role = 'input' AND status = 'failed') AS failed_files, "
        "(SELECT COUNT(*) FROM files WHERE files.job_id = jobs.id AND role = 'input' AND status = 'running') AS running_files "
        f"FROM jobs WHERE status IN ({placeholders}) "
        "ORDER BY CASE WHEN status IN ('queued', 'running', 'cancelling') THEN created_at END ASC, "
        "created_at DESC"
    )
    with connect(database_path) as connection:
        return connection.execute(query, status_values).fetchall()


def get_job(database_path: Path, job_id: str) -> sqlite3.Row | None:
    with connect(database_path) as connection:
        return connection.execute(
            "SELECT id, title, status, error_message, created_at, updated_at, started_at, completed_at "
            "FROM jobs WHERE id = ?",
            (job_id,),
        ).fetchone()


def insert_job(database_path: Path, job_id: str, title: str, status: str) -> None:
    with connect(database_path) as connection:
        connection.execute(
            "INSERT INTO jobs (id, title, status) VALUES (?, ?, ?)",
            (job_id, title, status),
        )


def create_job_with_files(
    database_path: Path,
    job_id: str,
    title: str,
    file_paths: Iterable[str],
) -> None:
    with connect(database_path) as connection:
        connection.execute(
            "INSERT INTO jobs (id, title, status) VALUES (?, ?, ?)",
            (job_id, title, "queued"),
        )
        connection.executemany(
            "INSERT INTO files (job_id, role, path, status) VALUES (?, ?, ?, ?)",
            ((job_id, "input", path, "pending") for path in file_paths),
        )


def list_files_for_job(database_path: Path, job_id: str) -> list[sqlite3.Row]:
    with connect(database_path) as connection:
        return connection.execute(
            "SELECT id, job_id, role, path, status, error_message, created_at, updated_at, started_at, completed_at "
            "FROM files WHERE job_id = ? ORDER BY id",
            (job_id,),
        ).fetchall()


def claim_next_pending_file(database_path: Path) -> sqlite3.Row | None:
    with connect(database_path) as connection:
        connection.execute("BEGIN IMMEDIATE")
        row = connection.execute(
            """
            SELECT
              files.id AS file_id,
              files.job_id,
              files.path,
              jobs.title,
              jobs.created_at,
              (
                SELECT COUNT(*)
                FROM files AS running_files
                WHERE running_files.job_id = jobs.id
                  AND running_files.role = 'input'
                  AND running_files.status = 'running'
              ) AS running_count
            FROM files
            JOIN jobs ON jobs.id = files.job_id
            WHERE files.role = 'input'
              AND files.status = 'pending'
              AND jobs.status IN ('queued', 'running')
            ORDER BY
              CASE WHEN running_count = 0 THEN 0 ELSE 1 END ASC,
              jobs.created_at ASC,
              files.id ASC
            LIMIT 1
            """
        ).fetchone()
        if row is None:
            connection.commit()
            return None

        connection.execute(
            "UPDATE jobs SET status = 'running', updated_at = CURRENT_TIMESTAMP, "
            "started_at = COALESCE(started_at, CURRENT_TIMESTAMP) WHERE id = ?",
            (row["job_id"],),
        )
        connection.execute(
            "UPDATE files SET status = 'running', updated_at = CURRENT_TIMESTAMP, "
            "started_at = CURRENT_TIMESTAMP WHERE id = ?",
            (row["file_id"],),
        )
        connection.commit()
        return row


def complete_file(database_path: Path, file_id: int) -> None:
    with connect(database_path) as connection:
        row = connection.execute(
            """
            SELECT files.job_id, jobs.status AS job_status
            FROM files
            JOIN jobs ON jobs.id = files.job_id
            WHERE files.id = ?
            """,
            (file_id,),
        ).fetchone()
        if row is None:
            return
        if row["job_status"] == "cancelling":
            connection.execute(
                "UPDATE files SET status = 'cancelled', error_message = 'Cancelled by user.', "
                "updated_at = CURRENT_TIMESTAMP, completed_at = CURRENT_TIMESTAMP WHERE id = ?",
                (file_id,),
            )
            return
        connection.execute(
            "UPDATE files SET status = 'completed', updated_at = CURRENT_TIMESTAMP, "
            "completed_at = CURRENT_TIMESTAMP WHERE id = ?",
            (file_id,),
        )


def fail_file_and_job(database_path: Path, file_id: int, job_id: str, message: str) -> None:
    with connect(database_path) as connection:
        job = connection.execute("SELECT status FROM jobs WHERE id = ?", (job_id,)).fetchone()
        if job is not None and job["status"] == "cancelling":
            connection.execute(
                "UPDATE files SET status = 'cancelled', error_message = 'Cancelled by user.', "
                "updated_at = CURRENT_TIMESTAMP, completed_at = CURRENT_TIMESTAMP WHERE id = ?",
                (file_id,),
            )
            return
        connection.execute(
            "UPDATE files SET status = 'failed', error_message = ?, updated_at = CURRENT_TIMESTAMP, "
            "completed_at = CURRENT_TIMESTAMP WHERE id = ?",
            (message, file_id),
        )
        connection.execute(
            "UPDATE files SET status = 'cancelled', error_message = 'Skipped after another scan failed.', "
            "updated_at = CURRENT_TIMESTAMP, completed_at = CURRENT_TIMESTAMP "
            "WHERE job_id = ? AND role = 'input' AND status = 'pending'",
            (job_id,),
        )
        connection.execute(
            "UPDATE jobs SET status = 'failed', error_message = ?, updated_at = CURRENT_TIMESTAMP, "
            "completed_at = CURRENT_TIMESTAMP WHERE id = ?",
            (message, job_id),
        )


def finalize_job_if_finished(database_path: Path, job_id: str) -> str | None:
    with connect(database_path) as connection:
        job = connection.execute("SELECT status FROM jobs WHERE id = ?", (job_id,)).fetchone()
        if job is None or job["status"] in {"failed", "cancelled", "completed"}:
            return job["status"] if job is not None else None
        counts = connection.execute(
            """
            SELECT
              SUM(CASE WHEN status IN ('pending', 'running') THEN 1 ELSE 0 END) AS active_count,
              SUM(CASE WHEN status = 'failed' THEN 1 ELSE 0 END) AS failed_count,
              SUM(CASE WHEN status = 'cancelled' THEN 1 ELSE 0 END) AS cancelled_count
            FROM files
            WHERE job_id = ? AND role = 'input'
            """,
            (job_id,),
        ).fetchone()
        if counts["active_count"]:
            return None
        if counts["failed_count"]:
            status = "failed"
        elif counts["cancelled_count"]:
            status = "cancelled"
        else:
            status = "completed"
        connection.execute(
            "UPDATE jobs SET status = ?, updated_at = CURRENT_TIMESTAMP, completed_at = CURRENT_TIMESTAMP "
            "WHERE id = ?",
            (status, job_id),
        )
        return status


def abort_job(database_path: Path, job_id: str) -> str:
    with connect(database_path) as connection:
        connection.execute("BEGIN IMMEDIATE")
        job = connection.execute("SELECT status FROM jobs WHERE id = ?", (job_id,)).fetchone()
        if job is None:
            connection.rollback()
            return "not_found"
        if job["status"] in {"completed", "failed", "cancelled"}:
            connection.rollback()
            return "finished"
        running_count = connection.execute(
            "SELECT COUNT(*) AS count FROM files WHERE job_id = ? AND role = 'input' AND status = 'running'",
            (job_id,),
        ).fetchone()["count"]
        pending_message = "Cancelled before processing." if running_count == 0 else "Skipped after job was cancelled."
        connection.execute(
            "UPDATE files SET status = 'cancelled', error_message = ?, updated_at = CURRENT_TIMESTAMP, "
            "completed_at = CURRENT_TIMESTAMP WHERE job_id = ? AND role = 'input' AND status = 'pending'",
            (pending_message, job_id),
        )
        if running_count:
            connection.execute(
                "UPDATE jobs SET status = 'cancelling', error_message = 'Cancellation requested.', "
                "updated_at = CURRENT_TIMESTAMP WHERE id = ?",
                (job_id,),
            )
            connection.commit()
            return "cancelling"
        connection.execute(
            "UPDATE jobs SET status = 'cancelled', error_message = 'Cancelled by user.', "
            "updated_at = CURRENT_TIMESTAMP, completed_at = CURRENT_TIMESTAMP WHERE id = ?",
            (job_id,),
        )
        connection.commit()
        return "cancelled"


def delete_job(database_path: Path, job_id: str) -> str:
    with connect(database_path) as connection:
        connection.execute("BEGIN IMMEDIATE")
        job = connection.execute("SELECT status FROM jobs WHERE id = ?", (job_id,)).fetchone()
        if job is None:
            connection.rollback()
            return "not_found"
        if job["status"] not in {"completed", "failed", "cancelled"}:
            connection.rollback()
            return "active"
        connection.execute("DELETE FROM jobs WHERE id = ?", (job_id,))
        connection.commit()
        return "deleted"


def reset_interrupted_work(database_path: Path) -> None:
    with connect(database_path) as connection:
        connection.execute(
            "UPDATE files SET status = 'pending', updated_at = CURRENT_TIMESTAMP, started_at = NULL "
            "WHERE status = 'running' AND job_id IN (SELECT id FROM jobs WHERE status = 'running')"
        )
        connection.execute(
            "UPDATE files SET status = 'cancelled', error_message = 'Cancelled during interrupted processing.', "
            "updated_at = CURRENT_TIMESTAMP, completed_at = CURRENT_TIMESTAMP "
            "WHERE status = 'running' AND job_id IN (SELECT id FROM jobs WHERE status = 'cancelling')"
        )
        connection.execute(
            "UPDATE jobs SET status = 'queued', updated_at = CURRENT_TIMESTAMP, started_at = NULL "
            "WHERE status = 'running'"
        )
        job_ids = [
            row["id"]
            for row in connection.execute(
                "SELECT id FROM jobs WHERE status IN ('queued', 'running', 'cancelling')"
            ).fetchall()
        ]

    for job_id in job_ids:
        finalize_job_if_finished(database_path, job_id)
