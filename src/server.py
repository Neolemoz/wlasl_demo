import uuid
from pathlib import Path

from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from . import paths
from .infer import InferenceError, infer_video

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health() -> dict:
    return {"ok": True}


@app.post("/infer")
def infer(
    file: UploadFile = File(...),
    topk: int = 5,
    mock: bool = True,
    weights: str = "weights/model.ts",
    labels: str = "weights/labels.json",
    num_classes: int = 100,
) -> dict:
    if not file.filename:
        return JSONResponse(
            status_code=400,
            content={"error": "missing filename", "hint": "Provide a file upload.", "code": "BAD_REQUEST"},
        )
    name = file.filename.lower()
    if not (name.endswith(".mp4") or name.endswith(".webm")):
        return JSONResponse(
            status_code=400,
            content={
                "error": "unsupported file type",
                "hint": "Upload .mp4 or .webm.",
                "code": "BAD_EXTENSION",
            },
        )

    paths.ensure_dirs()
    uploads_dir = paths.OUTPUTS_DIR / "uploads"
    uploads_dir.mkdir(parents=True, exist_ok=True)

    if not mock:
        weights_path = Path(weights)
        if not weights_path.exists():
            return JSONResponse(
                status_code=400,
                content={
                    "error": "model weights not found",
                    "hint": "Place TorchScript at weights/model.ts or pass weights=...",
                    "code": "WEIGHTS_MISSING",
                },
            )

    file_id = uuid.uuid4().hex
    suffix = ".webm" if name.endswith(".webm") else ".mp4"
    out_path = uploads_dir / f"{file_id}{suffix}"

    try:
        with out_path.open("wb") as f:
            while True:
                chunk = file.file.read(1024 * 1024)
                if not chunk:
                    break
                f.write(chunk)
    finally:
        file.file.close()

    try:
        result = infer_video(
            input_path=str(out_path),
            topk=topk,
            mock=mock,
            weights_path=weights,
            labels_path=labels,
            num_classes=num_classes,
        )
        return result
    except InferenceError as exc:
        content = {"error": exc.message, "code": "INFERENCE_ERROR"}
        if exc.hint:
            content["hint"] = exc.hint
        return JSONResponse(status_code=400, content=content)
    except ValueError as exc:
        return JSONResponse(
            status_code=400,
            content={"error": str(exc), "hint": "Check video decoding.", "code": "DECODE_ERROR"},
        )
    except Exception as exc:
        return JSONResponse(
            status_code=400,
            content={"error": f"Failed to run inference: {exc}", "code": "INFER_FAIL"},
        )
