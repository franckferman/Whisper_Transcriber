#!/usr/bin/env python3
# web/app.py

"""
FastAPI Web Interface for whispr

Description:
Provides a browser-based UI to submit transcription jobs, monitor progress
via WebSocket, and download results. Does not modify TranscriptionManager;
attaches/detaches a custom logging.Handler to capture log output.

Created By  : Franck FERMAN
Version     : 2.0.0
"""

import asyncio
import importlib
import logging
import os
import shutil
import sys
import tempfile
import threading
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Dict, List, Optional

# Make the parent directory importable so `transcriber` package resolves
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fastapi import FastAPI, File, Form, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    asyncio.create_task(_cleanup_old_jobs())
    yield

app = FastAPI(title="whispr Web UI", lifespan=lifespan)

_STATIC_DIR = Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=str(_STATIC_DIR)), name="static")

# ---------------------------------------------------------------------------
# Job store
# ---------------------------------------------------------------------------

class JobRecord:
    """Tracks state, log messages, and output files for a single transcription job."""

    def __init__(self, job_id: str, output_dir: str) -> None:
        self.job_id = job_id
        self.output_dir = output_dir
        self.status: str = "pending"   # pending | running | done | error
        self.log_messages: List[dict] = []
        self.output_files: Dict[str, str] = {}  # fmt -> absolute path
        self.created_at: float = time.time()
        self._lock = threading.Lock()

    def append_log(self, level: str, msg: str) -> None:
        with self._lock:
            self.log_messages.append({"level": level, "msg": msg})

    def snapshot_logs(self, from_index: int) -> List[dict]:
        with self._lock:
            return list(self.log_messages[from_index:])


_jobs: Dict[str, JobRecord] = {}
_jobs_lock = threading.Lock()


def _get_job(job_id: str) -> Optional[JobRecord]:
    with _jobs_lock:
        return _jobs.get(job_id)


def _register_job(record: JobRecord) -> None:
    with _jobs_lock:
        _jobs[record.job_id] = record


# ---------------------------------------------------------------------------
# Custom logging handler
# ---------------------------------------------------------------------------

class _JobLogHandler(logging.Handler):
    """Appends log records to a JobRecord's message list."""

    def __init__(self, record: JobRecord) -> None:
        super().__init__()
        self._record = record

    def emit(self, log_record: logging.LogRecord) -> None:
        try:
            msg = self.format(log_record)
            self._record.append_log(log_record.levelname, msg)
        except Exception:
            self.handleError(log_record)


# ---------------------------------------------------------------------------
# Backend detection
# ---------------------------------------------------------------------------

def _detect_backends() -> dict:
    """
    Detect which transcription backends are currently available.

    Returns a dict keyed by backend name with:
      available   (bool)
      reason      (str | None)  -- why it's unavailable
      install_hint (str | None)
    """
    result = {}

    # faster_whisper
    try:
        importlib.import_module("faster_whisper")
        result["faster_whisper"] = {"available": True, "reason": None, "install_hint": None}
    except ImportError:
        result["faster_whisper"] = {
            "available": False,
            "reason": "faster-whisper package not installed",
            "install_hint": "pip install faster-whisper",
        }

    # openai
    try:
        importlib.import_module("openai")
        result["openai"] = {"available": True, "reason": None, "install_hint": None}
    except ImportError:
        result["openai"] = {
            "available": False,
            "reason": "openai package not installed",
            "install_hint": "pip install openai",
        }

    # whisper_cpp -- binary detection
    project_root = Path(__file__).resolve().parent.parent
    candidate_paths = [
        str(project_root / "whisper.cpp" / "main"),
        str(project_root / "whisper.cpp" / "build" / "bin" / "whisper-cli"),
        shutil.which("whisper") or "",
        shutil.which("whisper-cli") or "",
    ]
    detected_binary: Optional[str] = None
    for p in candidate_paths:
        if p and Path(p).is_file() and os.access(p, os.X_OK):
            detected_binary = p
            break

    # whisper.cpp model detection: scan models directory for .bin files
    detected_model: Optional[str] = None
    models_dir = project_root / "whisper.cpp" / "models"
    if models_dir.is_dir():
        bins = sorted(models_dir.glob("ggml-*.bin"))
        if bins:
            detected_model = str(bins[0])

    if detected_binary:
        result["whisper_cpp"] = {
            "available": True,
            "reason": None,
            "install_hint": None,
            "binary_path": detected_binary,
            "model_path": detected_model,
        }
    else:
        result["whisper_cpp"] = {
            "available": False,
            "reason": "whisper.cpp binary not found",
            "install_hint": (
                "Run: make whisper-cpp-setup  "
                "(clones and compiles whisper.cpp)"
            ),
            "binary_path": None,
            "model_path": detected_model,
        }

    return result


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/", response_class=HTMLResponse)
async def index() -> FileResponse:
    return FileResponse(str(_STATIC_DIR / "index.html"))


@app.get("/api/backends")
async def get_backends() -> JSONResponse:
    return JSONResponse(_detect_backends())


@app.get("/api/status/{job_id}")
async def job_status(job_id: str) -> JSONResponse:
    record = _get_job(job_id)
    if record is None:
        return JSONResponse({"error": "job not found"}, status_code=404)
    return JSONResponse({"job_id": job_id, "status": record.status})


@app.post("/api/transcribe")
async def start_transcribe(
    source_type: str = Form(...),
    backend: str = Form(...),
    language: Optional[str] = Form(None),
    formats: str = Form("txt"),
    workers: int = Form(2),
    openai_key: Optional[str] = Form(None),
    fw_model: Optional[str] = Form(None),
    whisper_binary: Optional[str] = Form(None),
    whisper_model: Optional[str] = Form(None),
    url: Optional[str] = Form(None),
    file: Optional[UploadFile] = File(None),
) -> JSONResponse:
    """
    Start a transcription job.

    Saves the uploaded file (or records the URL), creates a JobRecord,
    and launches the TranscriptionManager in a background thread.
    Returns immediately with {job_id}.
    """
    job_id = str(uuid.uuid4())
    output_dir = tempfile.mkdtemp(prefix=f"wt_job_{job_id[:8]}_")
    record = JobRecord(job_id=job_id, output_dir=output_dir)
    _register_job(record)

    # Resolve output formats
    fmt_list = [f.strip() for f in formats.split(",") if f.strip()]
    if not fmt_list:
        fmt_list = ["txt"]

    # Language
    lang = language if language and language != "auto" else None

    # Persist uploaded file if needed
    input_file: Optional[str] = None
    input_url: Optional[str] = None

    if source_type == "file" and file is not None:
        upload_path = os.path.join(output_dir, file.filename or "upload")
        content = await file.read()
        with open(upload_path, "wb") as fh:
            fh.write(content)
        input_file = upload_path
    elif source_type == "url" and url:
        input_url = url
    else:
        return JSONResponse({"error": "Invalid source_type or missing file/url"}, status_code=400)

    # Build config dict
    config_dict: dict = {
        "backend": backend,
        "language": lang,
        "workers": workers,
        "output_formats": fmt_list,
        "output_dir": output_dir,
        "output_prefix": "transcript",
    }
    if input_file:
        config_dict["input_file"] = input_file
    if input_url:
        config_dict["input_url"] = input_url

    if backend == "faster_whisper" and fw_model:
        config_dict["faster_whisper_model"] = fw_model

    if backend == "openai" and openai_key:
        config_dict["openai_api_key"] = openai_key

    if backend == "whisper_cpp":
        if whisper_binary:
            config_dict["whisper_cpp_binary"] = whisper_binary
        if whisper_model:
            config_dict["whisper_cpp_model"] = whisper_model

    # Launch in background thread
    t = threading.Thread(
        target=_run_job,
        args=(record, config_dict),
        daemon=True,
    )
    t.start()

    return JSONResponse({"job_id": job_id})


@app.get("/api/download/{job_id}/{fmt}")
async def download_result(job_id: str, fmt: str) -> FileResponse:
    record = _get_job(job_id)
    if record is None:
        return JSONResponse({"error": "job not found"}, status_code=404)
    path = record.output_files.get(fmt)
    if not path or not Path(path).is_file():
        return JSONResponse({"error": f"format '{fmt}' not available"}, status_code=404)
    return FileResponse(
        path,
        filename=Path(path).name,
        media_type="application/octet-stream",
    )


@app.websocket("/ws/{job_id}")
async def websocket_progress(ws: WebSocket, job_id: str) -> None:
    """
    Stream job progress to the browser.

    Polls the job's log buffer every 500 ms, pushing new messages as JSON.
    Sends a final {"type": "done"} or {"type": "error"} frame when complete.
    """
    await ws.accept()
    record = _get_job(job_id)
    if record is None:
        await ws.send_json({"type": "error", "msg": "Job not found"})
        await ws.close()
        return

    log_cursor = 0

    try:
        while True:
            # Push any new log lines
            new_entries = record.snapshot_logs(log_cursor)
            for entry in new_entries:
                await ws.send_json({"type": "log", "level": entry["level"], "msg": entry["msg"]})
            log_cursor += len(new_entries)

            if record.status == "done":
                await ws.send_json({
                    "type": "done",
                    "formats": list(record.output_files.keys()),
                    "job_id": job_id,
                })
                break
            elif record.status == "error":
                # Grab last log message as the error summary if available
                last_logs = record.snapshot_logs(0)
                err_msg = last_logs[-1]["msg"] if last_logs else "Unknown error"
                await ws.send_json({"type": "error", "msg": err_msg})
                break

            await asyncio.sleep(0.5)

    except WebSocketDisconnect:
        pass
    finally:
        try:
            await ws.close()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Background job runner
# ---------------------------------------------------------------------------

def _run_job(record: JobRecord, config_dict: dict) -> None:
    """
    Run TranscriptionManager in a worker thread.

    Attaches a custom log handler to the root logger (and the 'transcriber'
    logger specifically) before the run, detaches afterwards.
    """
    from transcriber.config import TranscriptionConfig
    from transcriber.managers.transcription import TranscriptionManager

    record.status = "running"

    handler = _JobLogHandler(record)
    handler.setFormatter(logging.Formatter("%(message)s"))

    # Attach to root so all transcriber.* loggers are captured
    root_logger = logging.getLogger()
    transcriber_logger = logging.getLogger("transcriber")
    root_logger.addHandler(handler)
    transcriber_logger.addHandler(handler)

    # Also make sure the level is low enough to capture INFO
    prev_root_level = root_logger.level
    if root_logger.level == 0 or root_logger.level > logging.INFO:
        root_logger.setLevel(logging.INFO)

    try:
        config = TranscriptionConfig.from_dict(config_dict)
        manager = TranscriptionManager(config)
        manager.run()

        # Collect output files
        output_dir = Path(config.output_dir)
        for fmt in config.output_formats:
            # OutputFormatter writes <prefix>.<fmt>
            candidate = output_dir / f"{config.output_prefix}.{fmt}"
            if candidate.is_file():
                record.output_files[fmt] = str(candidate)

        if not record.output_files:
            # Fallback: scan for any recognised file
            for p in output_dir.iterdir():
                if p.suffix.lstrip(".") in {"txt", "srt", "vtt", "json"} and p.is_file():
                    record.output_files[p.suffix.lstrip(".")] = str(p)

        record.status = "done"
        record.append_log("INFO", "Transcription complete.")

    except Exception as exc:
        record.status = "error"
        record.append_log("ERROR", f"Job failed: {exc}")
    finally:
        root_logger.removeHandler(handler)
        transcriber_logger.removeHandler(handler)
        root_logger.setLevel(prev_root_level)


# ---------------------------------------------------------------------------
# Periodic cleanup (jobs older than 1 hour)
# ---------------------------------------------------------------------------

_CLEANUP_INTERVAL = 3600  # seconds
_JOB_TTL = 3600


async def _cleanup_old_jobs() -> None:
    while True:
        await asyncio.sleep(_CLEANUP_INTERVAL)
        now = time.time()
        to_delete = []
        with _jobs_lock:
            for jid, rec in _jobs.items():
                if now - rec.created_at > _JOB_TTL:
                    to_delete.append(jid)
        for jid in to_delete:
            rec = _get_job(jid)
            if rec and Path(rec.output_dir).exists():
                try:
                    shutil.rmtree(rec.output_dir)
                except OSError:
                    pass
            with _jobs_lock:
                _jobs.pop(jid, None)


