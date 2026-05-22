"""Structured logging helpers for operational job events."""

from __future__ import annotations

import json
import logging
from typing import Any


logger = logging.getLogger("web")


def log_job_event(event: str, job_id: str, **fields: Any) -> None:
    payload = {
        "event": event,
        "job_id": job_id,
        **fields,
    }
    logger.info(json.dumps(payload, sort_keys=True, default=str))
