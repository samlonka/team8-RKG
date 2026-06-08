"""
api/ingest_jobs.py — Async tenant Excel ingest job store and worker.
"""

from __future__ import annotations

import threading
import traceback
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

from ingest_vendor import TenantIngestResult, run_tenant_ingest

JobStatus = Literal["pending", "running", "completed", "failed"]

JOBS_DIR = Path(__file__).parent.parent / "data" / "ingest_jobs"
JOBS: dict[str, "IngestJob"] = {}
_LOCK = threading.Lock()


@dataclass
class IngestJob:
    job_id:       str
    status:       JobStatus
    source_path:  str
    output_path:  str | None = None
    created_at:   str = ""
    started_at:   str | None = None
    completed_at: str | None = None
    error:        str | None = None
    result:       dict | None = None


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _result_to_dict(result: TenantIngestResult) -> dict:
    return {
        "run_id":       result.run_id,
        "source_file":  result.source_file,
        "output_file":  result.output_file,
        "started_at":   result.started_at,
        "completed_at": result.completed_at,
        "summary":      result.summary,
        "alerts":       result.alerts,
        "rows":         [asdict(r) for r in result.rows],
    }


def save_upload_and_create_job(content: bytes, suffix: str = ".xlsx") -> IngestJob:
    """Persist uploaded Excel and register a pending ingest job."""
    JOBS_DIR.mkdir(parents=True, exist_ok=True)
    job_id = str(uuid.uuid4())
    dest = JOBS_DIR / f"{job_id}_input{suffix}"
    dest.write_bytes(content)
    job = IngestJob(
        job_id=job_id,
        status="pending",
        source_path=str(dest),
        created_at=_now(),
    )
    with _LOCK:
        JOBS[job_id] = job
    return job


def get_job(job_id: str) -> IngestJob | None:
    with _LOCK:
        return JOBS.get(job_id)


def _run_job(job_id: str, skip_validation: bool) -> None:
    with _LOCK:
        job = JOBS.get(job_id)
        if job is None:
            return
        job.status = "running"
        job.started_at = _now()

    try:
        out_path = str(
            JOBS_DIR / f"{job_id}_output.xlsx"
        )
        result = run_tenant_ingest(
            job.source_path,
            output_file=out_path,
            skip_validation=skip_validation,
            write_graph=True,
        )
        _sync_postgres_tenant(job.source_path)
        with _LOCK:
            job = JOBS[job_id]
            job.status = "completed"
            job.completed_at = _now()
            job.output_path = result.output_file
            job.result = _result_to_dict(result)
    except Exception as exc:
        with _LOCK:
            job = JOBS[job_id]
            job.status = "failed"
            job.completed_at = _now()
            job.error = f"{exc}\n{traceback.format_exc()}"


def _sync_postgres_tenant(source_path: str) -> None:
    """Upsert uploaded tenant Excel into PostgreSQL tenant tables."""
    try:
        from data.postgres_store import sync_tenant_xlsx

        sync_tenant_xlsx(source_path)
    except Exception as exc:
        print(f"[ingest] PostgreSQL sync failed (Neo4j ingest succeeded): {exc}")


def start_job_async(job_id: str, skip_validation: bool = False) -> None:
    thread = threading.Thread(
        target=_run_job,
        args=(job_id, skip_validation),
        daemon=True,
        name=f"ingest-{job_id[:8]}",
    )
    thread.start()


def job_to_response(job: IngestJob) -> dict:
    return {
        "job_id":       job.job_id,
        "status":       job.status,
        "created_at":   job.created_at,
        "started_at":   job.started_at,
        "completed_at": job.completed_at,
        "source_path":  job.source_path,
        "output_path":  job.output_path,
        "error":        job.error,
        "result":       job.result,
    }
