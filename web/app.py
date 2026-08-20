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
import re
import shutil
import sys
import tempfile
import threading
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Dict, List, Optional

# Make the parent dir (for `transcriber`) and this dir (for `security`) importable
_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent))
sys.path.insert(0, str(_HERE))

from fastapi import (
    Depends,
    FastAPI,
    File,
    Form,
    HTTPException,
    Request,
    UploadFile,
    WebSocket,
    WebSocketDisconnect,
)
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

import security as sec

logger = logging.getLogger("whispr.web")

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


@app.middleware("http")
async def _security_headers(request: Request, call_next):
    resp = await call_next(request)
    resp.headers.setdefault("X-Content-Type-Options", "nosniff")
    resp.headers.setdefault("X-Frame-Options", "DENY")
    resp.headers.setdefault("Referrer-Policy", "no-referrer")
    resp.headers.setdefault(
        "Content-Security-Policy",
        "default-src 'self'; img-src 'self' data:; "
        "style-src 'self' 'unsafe-inline'; script-src 'self'; "
        "connect-src 'self'; frame-ancestors 'none'",
    )
    return resp


def require_auth(request: Request) -> None:
    """Gate for the JSON API. No-op unless WHISPR_AUTH_TOKEN is set."""
    # header preferred; query param is the fallback for <a> downloads
    supplied = sec.token_from_headers(request.headers) or request.query_params.get("token")
    if not sec.token_ok(supplied):
        raise HTTPException(status_code=401, detail="unauthorized")

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


def _public_backends() -> dict:
    """Backend view for the browser -> availability only, no server paths."""
    data = _detect_backends()
    public = {}
    for name, info in data.items():
        entry = {
            "available": info.get("available"),
            "reason": info.get("reason"),
            "install_hint": info.get("install_hint"),
        }
        if name == "whisper_cpp":
            # server manages the paths now; only say whether a model is present
            entry["model_available"] = bool(info.get("model_path"))
        public[name] = entry
    return public


@app.get("/api/backends")
async def get_backends(_: None = Depends(require_auth)) -> JSONResponse:
    return JSONResponse(_public_backends())


@app.get("/api/status/{job_id}")
async def job_status(job_id: str, _: None = Depends(require_auth)) -> JSONResponse:
    record = _get_job(job_id)
    if record is None:
        return JSONResponse({"error": "job not found"}, status_code=404)
    return JSONResponse({"job_id": job_id, "status": record.status})


_VALID_BACKENDS = {"whisper_cpp", "faster_whisper", "openai"}
_VALID_FORMATS = {"txt", "srt", "vtt", "json"}
_FW_MODELS = {"tiny", "base", "small", "medium", "large-v2"}
_LANG_RE = re.compile(r"^[a-zA-Z]{2,3}(-[a-zA-Z]{2,4})?$")


async def _save_upload(file: UploadFile, output_dir: str) -> str:
    """Stream an upload to disk under a sanitised name, capped at MAX_UPLOAD_BYTES."""
    name = sec.safe_upload_name(file.filename)
    dest = os.path.join(output_dir, name)

    # belt-and-suspenders: never let the write escape the job dir
    root = os.path.realpath(output_dir)
    if os.path.commonpath([os.path.realpath(dest), root]) != root:
        raise HTTPException(status_code=400, detail="invalid filename")

    total = 0
    with open(dest, "wb") as fh:
        while True:
            chunk = await file.read(1024 * 1024)
            if not chunk:
                break
            total += len(chunk)
            if total > sec.MAX_UPLOAD_BYTES:
                fh.close()
                os.remove(dest)
                raise HTTPException(status_code=413, detail="upload too large")
            fh.write(chunk)
    return dest


@app.post("/api/transcribe")
async def start_transcribe(
    source_type: str = Form(...),
    backend: str = Form(...),
    language: Optional[str] = Form(None),
    formats: str = Form("txt"),
    workers: int = Form(2),
    openai_key: Optional[str] = Form(None),
    fw_model: Optional[str] = Form(None),
    url: Optional[str] = Form(None),
    file: Optional[UploadFile] = File(None),
    _: None = Depends(require_auth),
) -> JSONResponse:
    """
    Start a transcription job.

    Saves the uploaded file (or records the URL), creates a JobRecord,
    and launches the TranscriptionManager in a background thread.
    Returns immediately with {job_id}.

    Note: whisper.cpp binary/model paths are NOT taken from the request -- they
    are pinned to what the server detected, so a client can't point the server
    at an arbitrary executable.
    """
    if backend not in _VALID_BACKENDS:
        return JSONResponse({"error": "invalid backend"}, status_code=400)

    # Output formats -> validate against the allowed set
    fmt_list = [f.strip().lower() for f in formats.split(",") if f.strip()]
    fmt_list = [f for f in fmt_list if f in _VALID_FORMATS]
    if not fmt_list:
        fmt_list = ["txt"]

    # Language: keep only well-formed ISO-ish codes (blocks ${..} interpolation etc.)
    lang = None
    if language and language != "auto":
        if not _LANG_RE.match(language):
            return JSONResponse({"error": "invalid language code"}, status_code=400)
        lang = language

    # Workers: clamp to a sane server-side range (client hints are advisory)
    try:
        workers = max(1, min(int(workers), sec.MAX_WORKERS))
    except (TypeError, ValueError):
        workers = 2

    job_id = str(uuid.uuid4())
    output_dir = tempfile.mkdtemp(prefix=f"wt_job_{job_id[:8]}_")

    config_dict: dict = {
        "backend": backend,
        "language": lang,
        "workers": workers,
        "output_formats": fmt_list,
        "output_dir": output_dir,
        "output_prefix": "transcript",
    }

    # Resolve the input source
    if source_type == "file" and file is not None:
        config_dict["input_file"] = await _save_upload(file, output_dir)
    elif source_type == "url" and url:
        if not sec.ALLOW_URL_FETCH:
            shutil.rmtree(output_dir, ignore_errors=True)
            return JSONResponse({"error": "remote URL fetching is disabled"}, status_code=403)
        ok, reason = sec.is_safe_remote_url(url)
        if not ok:
            shutil.rmtree(output_dir, ignore_errors=True)
            return JSONResponse({"error": f"URL rejected: {reason}"}, status_code=400)
        config_dict["input_url"] = url
    else:
        shutil.rmtree(output_dir, ignore_errors=True)
        return JSONResponse({"error": "Invalid source_type or missing file/url"}, status_code=400)

    # Backend-specific settings
    if backend == "faster_whisper" and fw_model:
        if fw_model not in _FW_MODELS:
            shutil.rmtree(output_dir, ignore_errors=True)
            return JSONResponse({"error": "invalid faster-whisper model"}, status_code=400)
        config_dict["faster_whisper_model"] = fw_model

    if backend == "openai":
        # No env fallback here: an anonymous web caller must bring their own key,
        # otherwise the server's OPENAI_API_KEY would get spent for them.
        if not openai_key:
            shutil.rmtree(output_dir, ignore_errors=True)
            return JSONResponse(
                {"error": "OpenAI backend requires an API key"}, status_code=400
            )
        config_dict["openai_api_key"] = openai_key

    if backend == "whisper_cpp":
        det = _detect_backends()["whisper_cpp"]
        if not det.get("available") or not det.get("binary_path"):
            shutil.rmtree(output_dir, ignore_errors=True)
            return JSONResponse(
                {"error": "whisper.cpp is not available on this server"}, status_code=400
            )
        if not det.get("model_path"):
            shutil.rmtree(output_dir, ignore_errors=True)
            return JSONResponse(
                {"error": "no whisper.cpp model installed on this server"}, status_code=400
            )
        config_dict["whisper_cpp_binary"] = det["binary_path"]
        config_dict["whisper_cpp_model"] = det["model_path"]

    record = JobRecord(job_id=job_id, output_dir=output_dir)
    _register_job(record)

    # Launch in background thread
    t = threading.Thread(
        target=_run_job,
        args=(record, config_dict),
        daemon=True,
    )
    t.start()

    return JSONResponse({"job_id": job_id})


@app.get("/api/download/{job_id}/{fmt}")
async def download_result(
    job_id: str, fmt: str, _: None = Depends(require_auth)
) -> FileResponse:
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
    # Auth (if enabled): token via ?token= or X-Whispr-Token, checked before accept
    if sec.AUTH_TOKEN is not None:
        supplied = ws.query_params.get("token") or sec.token_from_headers(ws.headers)
        if not sec.token_ok(supplied):
            await ws.close(code=1008)  # policy violation
            return

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
        # interpolate=False: this dict carries user input, don't expand ${ENV}
        config = TranscriptionConfig.from_dict(config_dict, interpolate=False)
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

    except Exception:
        # full detail to the server log; only a generic line to the client
        # (raw exceptions leak filesystem paths and remote-fetch errors)
        record.status = "error"
        logger.exception("Job %s failed", record.job_id)
        record.append_log("ERROR", "Job failed. See server logs for details.")
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


