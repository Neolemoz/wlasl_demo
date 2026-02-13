import os
import subprocess
import time
import uuid
from pathlib import Path
from threading import Lock

from fastapi import FastAPI, File, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from . import paths
from .infer import InferenceError, infer_video

app = FastAPI()

MAX_UPLOAD_BYTES = 25 * 1024 * 1024
MAX_DURATION_SECONDS = 10.0
UPLOAD_CHUNK_SIZE = 1024 * 1024
RATE_LIMIT_WINDOW_SECONDS = 60.0
RATE_LIMIT_MAX_REQUESTS = 30
KEEP_UPLOADS = os.getenv("KEEP_UPLOADS", "").lower() in {"1", "true", "yes"}

_rate_limit_lock = Lock()
_rate_limit_state: dict[str, list[float]] = {}

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health() -> dict:
    return {"ok": True}


def _json_error(status_code: int, error: str, code: str, hint: str | None = None) -> JSONResponse:
    content = {"error": error, "code": code}
    if hint:
        content["hint"] = hint
    return JSONResponse(status_code=status_code, content=content)


def _is_rate_limited(client_ip: str) -> bool:
    now = time.time()
    cutoff = now - RATE_LIMIT_WINDOW_SECONDS
    with _rate_limit_lock:
        timestamps = [ts for ts in _rate_limit_state.get(client_ip, []) if ts >= cutoff]
        if len(timestamps) >= RATE_LIMIT_MAX_REQUESTS:
            _rate_limit_state[client_ip] = timestamps
            return True
        timestamps.append(now)
        _rate_limit_state[client_ip] = timestamps
        return False


def _probe_duration_seconds(video_path: Path) -> float | None:
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=nw=1:nk=1",
        str(video_path),
    ]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=2, check=False)
    except Exception:
        return None
    if proc.returncode != 0:
        return None
    output = (proc.stdout or "").strip()
    if not output:
        return None
    try:
        return float(output.splitlines()[0].strip())
    except Exception:
        return None


@app.post("/infer")
async def infer(
    request: Request,
    file: UploadFile = File(...),
    topk: int = 5,
    mock: bool = True,
    weights: str = "weights/model.ts",
    labels: str = "weights/labels.json",
    num_classes: int = 100,
    confidence_threshold: float = 0.50,
    margin_threshold: float = 0.15,
) -> dict:
    out_path: Path | None = None
    response: dict | JSONResponse | None = None
    try:
        client_ip = request.client.host if request.client and request.client.host else "unknown"
        if _is_rate_limited(client_ip):
            response = _json_error(
                status_code=429,
                error="Rate limit exceeded",
                code="RATE_LIMITED",
                hint="Try again later",
            )
        elif not file.filename:
            response = _json_error(
                status_code=400,
                error="missing filename",
                code="BAD_REQUEST",
                hint="Provide a file upload.",
            )
        else:
            name = file.filename.lower()
            if not (name.endswith(".mp4") or name.endswith(".webm")):
                response = _json_error(
                    status_code=400,
                    error="unsupported file type",
                    code="BAD_EXTENSION",
                    hint="Upload .mp4 or .webm.",
                )
            else:
                content_length = request.headers.get("content-length")
                if content_length:
                    try:
                        if int(content_length) > MAX_UPLOAD_BYTES:
                            response = _json_error(status_code=400, error="File too large", code="FILE_TOO_LARGE")
                    except ValueError:
                        pass

                if response is None:
                    paths.ensure_dirs()
                    uploads_dir = paths.OUTPUTS_DIR / "uploads"
                    uploads_dir.mkdir(parents=True, exist_ok=True)

                    if not mock:
                        weights_path = Path(weights)
                        if not weights_path.exists():
                            response = _json_error(
                                status_code=400,
                                error="model weights not found",
                                code="WEIGHTS_MISSING",
                                hint="Place TorchScript at weights/model.ts or pass weights=...",
                            )

                if response is None:
                    file_id = uuid.uuid4().hex
                    suffix = ".webm" if name.endswith(".webm") else ".mp4"
                    out_path = uploads_dir / f"{file_id}{suffix}"

                    total_written = 0
                    with out_path.open("wb") as f:
                        while True:
                            chunk = await file.read(UPLOAD_CHUNK_SIZE)
                            if not chunk:
                                break
                            total_written += len(chunk)
                            if total_written > MAX_UPLOAD_BYTES:
                                response = _json_error(
                                    status_code=400, error="File too large", code="FILE_TOO_LARGE"
                                )
                                break
                            f.write(chunk)

                if out_path is not None and response is None:
                    duration_seconds = _probe_duration_seconds(out_path)
                    if duration_seconds is not None and duration_seconds > MAX_DURATION_SECONDS:
                        response = _json_error(
                            status_code=400,
                            error="Video too long",
                            code="DURATION_TOO_LONG",
                            hint="Max 10s",
                        )

                if out_path is not None and response is None:
                    response = infer_video(
                        input_path=str(out_path),
                        topk=topk,
                        mock=mock,
                        weights_path=weights,
                        labels_path=labels,
                        num_classes=num_classes,
                        confidence_threshold=confidence_threshold,
                        margin_threshold=margin_threshold,
                    )
    except InferenceError as exc:
        response = _json_error(status_code=400, error=exc.message, code="INFERENCE_ERROR", hint=exc.hint)
    except ValueError as exc:
        message = str(exc)
        if message in {"DECODE_FAILED", "No frames read from video."}:
            response = _json_error(
                status_code=400,
                error=message,
                code="DECODE_ERROR",
                hint="Check video decoding.",
            )
        else:
            response = _json_error(
                status_code=400,
                error=message,
                code="BAD_REQUEST",
            )
    except Exception as exc:
        response = _json_error(
            status_code=400,
            error=f"Failed to run inference: {exc}",
            code="INFER_FAIL",
        )
    finally:
        await file.close()
        if out_path is not None and not KEEP_UPLOADS:
            try:
                out_path.unlink()
            except FileNotFoundError:
                pass
            except Exception:
                pass
    if response is None:
        response = _json_error(status_code=400, error="BAD_REQUEST", code="BAD_REQUEST")
    return response
