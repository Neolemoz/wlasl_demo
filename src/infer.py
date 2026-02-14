import argparse
import hashlib
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

from . import paths


DEFAULT_NUM_LABELS = 100
MODEL_INPUT_FRAMES = 32
DEFAULT_NUM_FRAMES = MODEL_INPUT_FRAMES
MOCK_FALLBACK_LABELS = ["videos", "other", "misc", "alt", "bg", "noise", "dummy"]


class InferenceError(Exception):
    def __init__(self, message: str, hint: str | None = None, code: int | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.hint = hint
        self.code = code


def load_labels(labels_path: Path, num_classes: int) -> list[str]:
    if not labels_path.exists():
        print(f"HINT: labels not found at {labels_path}. Using mock_<i> labels.")
        print("      To use real labels, place weights/labels.json (list of strings) or pass --labels <path>.")
        return [f"mock_{i}" for i in range(num_classes)]

    try:
        with labels_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as exc:
        print(f"HINT: labels file malformed at {labels_path}. Using mock_<i> labels.")
        return [f"mock_{i}" for i in range(num_classes)]

    if isinstance(data, list) and all(isinstance(x, str) for x in data):
        return data

    if isinstance(data, dict):
        items = []
        try:
            for k, v in data.items():
                items.append((int(k), str(v)))
        except Exception as exc:
            print(f"HINT: labels file malformed at {labels_path}. Using mock_<i> labels.")
            return [f"mock_{i}" for i in range(num_classes)]
        items.sort(key=lambda x: x[0])
        return [label for _, label in items]

    print(f"HINT: labels file malformed at {labels_path}. Using mock_<i> labels.")
    return [f"mock_{i}" for i in range(num_classes)]


def _resize_rgb(frame_bgr: np.ndarray, size: int = 224) -> np.ndarray:
    frame = cv2.resize(frame_bgr, (size, size), interpolation=cv2.INTER_AREA)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = frame.astype(np.float32) / 255.0
    return frame


def decode_video_to_tensor(video_path: Path, num_frames: int = MODEL_INPUT_FRAMES) -> tuple[torch.Tensor, dict]:
    cap = None
    frames: list[np.ndarray] = []
    fps = None
    width = None
    height = None
    try:
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise ValueError("DECODE_FAILED")

        fps = float(cap.get(cv2.CAP_PROP_FPS)) or None
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or None
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or None

        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                break
            frames.append(_resize_rgb(frame))
    except Exception:
        raise ValueError("DECODE_FAILED")
    finally:
        if cap is not None:
            cap.release()

    if not frames:
        raise ValueError("DECODE_FAILED")

    total = len(frames)
    if total >= num_frames:
        indices = np.linspace(0, total - 1, num_frames).round().astype(int)
        indices = np.clip(indices, 0, total - 1)
        frames = [frames[int(i)] for i in indices]
    else:
        frames = frames * (num_frames // total) + frames[: (num_frames % total)]

    arr = np.stack(frames, axis=0)  # [T, H, W, C]
    arr = np.transpose(arr, (0, 3, 1, 2))  # [T, C, H, W]
    arr = np.expand_dims(arr, axis=0)  # [1, T, C, H, W]
    meta = {
        "frames": int(num_frames),
        "fps": fps,
        "width": width,
        "height": height,
    }
    return torch.from_numpy(arr).float(), meta


def _fingerprint_from_tensor(video_tensor: torch.Tensor) -> np.ndarray:
    data = video_tensor.squeeze(0).permute(0, 2, 3, 1).cpu().numpy()  # [T, H, W, C]
    mean_rgb = data.mean(axis=(0, 1, 2))
    std_rgb = data.std(axis=(0, 1, 2))
    if data.shape[0] > 1:
        diffs = np.abs(data[1:] - data[:-1])
        motion = float(diffs.mean())
    else:
        motion = 0.0
    luma = (0.299 * data[..., 0] + 0.587 * data[..., 1] + 0.114 * data[..., 2]).mean()
    features = np.concatenate([mean_rgb, std_rgb, np.array([motion, luma], dtype=np.float32)])
    return features.astype(np.float32)


def mock_predict_probs(video_tensor: torch.Tensor, num_labels: int) -> tuple[np.ndarray, int, float]:
    features = _fingerprint_from_tensor(video_tensor)
    digest = hashlib.sha256(features.tobytes()).digest()
    seed = int.from_bytes(digest[:8], "little", signed=False)
    rng = np.random.default_rng(seed)

    if num_labels <= 0:
        raise ValueError("num_labels must be >= 1")

    # Build deterministic pseudo-scores, then map through softmax(log(score)).
    score_weights = rng.uniform(0.01, 0.05, size=(num_labels,)).astype(np.float32)
    if num_labels >= 2:
        top2 = rng.choice(num_labels, size=2, replace=False)
        s1 = 1.00 + float(rng.uniform(-0.02, 0.02))
        s2 = 0.60 + float(rng.uniform(-0.02, 0.02))
        if s2 >= s1:
            s2 = max(0.01, s1 - 0.01)
        score_weights[top2[0]] = np.float32(s1)
        score_weights[top2[1]] = np.float32(s2)
    else:
        score_weights[0] = np.float32(0.80)

    logits = np.log(np.maximum(score_weights, 1e-6))

    temperature = 0.95 + (seed % 26) / 100.0
    scaled = logits / temperature
    scaled = scaled - np.max(scaled)
    exp = np.exp(scaled)
    probs = exp / exp.sum()

    probs = np.maximum(probs, 1e-6)
    probs = probs / probs.sum()
    return probs.astype(np.float32), seed, float(temperature)


def load_torchscript_model(weights_path: Path) -> torch.jit.ScriptModule:
    return torch.jit.load(str(weights_path), map_location="cpu")


def run_inference(
    video_tensor: torch.Tensor,
    labels: list[str],
    weights_path: Path,
    use_mock: bool,
) -> tuple[np.ndarray, int | None, float | None]:
    if use_mock:
        probs, seed, temp = mock_predict_probs(video_tensor, len(labels))
        return probs, seed, temp

    if not weights_path.exists():
        raise InferenceError(
            message=f"ERROR: model weights not found: {weights_path}",
            hint="HINT: place a TorchScript model at weights/model.ts or pass --weights <path>\n"
            "      TorchScript expected (torch.jit.load on CPU).",
            code=2,
        )

    model = load_torchscript_model(weights_path)
    model.eval()
    with torch.no_grad():
        outputs = model(video_tensor)
    if isinstance(outputs, (list, tuple)):
        outputs = outputs[0]
    if isinstance(outputs, dict):
        outputs = outputs.get("logits", None)
    if outputs is None:
        raise RuntimeError("Model output is invalid; expected logits tensor.")
    if outputs.ndim == 2:
        outputs = outputs[0]
    probs = torch.softmax(outputs, dim=-1).cpu().numpy()
    return probs.astype(np.float32), None, None


def _adjust_topk_for_display(
    topk_probs: np.ndarray, min_tail: float = 0.02, max_sum: float = 0.90
) -> np.ndarray:
    display = topk_probs.copy()
    if display.size >= 3:
        p1 = float(display[0])
        tail_floor = min_tail
        if tail_floor >= p1 * 0.8:
            tail_floor = max(0.005, p1 * 0.6)
        display[2:] = np.maximum(display[2:], tail_floor)

    target_sum = min(max_sum, float(topk_probs.sum()))
    if target_sum <= 0:
        return display

    scale = target_sum / float(display.sum())
    display = display * scale

    if display.size >= 2:
        p1 = float(display[0])
        if display[1:].max() >= p1:
            display[1:] = np.minimum(display[1:], p1 * 0.9)
            display = display * (target_sum / float(display.sum()))

    return display


def decide_unknown(topk_list: list[dict], T: float, M: float) -> tuple[str, str | None]:
    # Assumes topk_list sorted by score desc.
    if len(topk_list) < 2:
        return "unknown", "ambiguous"
    top1_prob = float(topk_list[0]["score"])
    if top1_prob < T:
        return "unknown", "low_confidence"
    top2_prob = float(topk_list[1]["score"])
    if (top1_prob - top2_prob) < M:
        return "unknown", "ambiguous"
    return "ok", None


def extract_topk(labels: list[str], probs: np.ndarray, topk: int) -> list[dict]:
    if not labels:
        return []
    requested = max(1, int(topk))
    min_required = min(2, len(labels))
    k = min(len(labels), max(requested, min_required))
    indices = np.argsort(-probs)[:k]
    return [{"label": labels[int(idx)], "score": float(probs[int(idx)])} for idx in indices]


def format_topk(labels: list[str], probs: np.ndarray, topk: int, mock_display: bool = False) -> str:
    topk = min(topk, len(labels))
    indices = np.argsort(-probs)[:topk]
    display_probs = probs[indices]
    if mock_display:
        display_probs = _adjust_topk_for_display(display_probs)
    lines = []
    for rank, idx in enumerate(indices, start=1):
        label = labels[int(idx)]
        prob = float(display_probs[rank - 1])
        lines.append(f"  {rank}) {label:<12} {prob:.2f}")
    return "\n".join(lines), int(indices[0]), float(display_probs[0])


def _infer_core(
    input_path: str,
    topk: int,
    mock: bool,
    weights_path: str | None,
    labels_path: str | None,
    num_classes: int,
) -> tuple[np.ndarray, list[str], dict, str, int | None, float | None]:
    paths.ensure_dirs()

    input_p = Path(input_path)
    if not input_p.exists():
        raise InferenceError(message=f"ERROR: input not found: {input_p}", code=1)

    labels_p = Path(labels_path) if labels_path else paths.WEIGHTS_DIR / "labels.json"
    weights_p = Path(weights_path) if weights_path else paths.WEIGHTS_DIR / "model.ts"

    labels = load_labels(labels_p, max(1, int(num_classes)))
    if mock and len(labels) < 2:
        extended = [labels[0] if labels else MOCK_FALLBACK_LABELS[0]]
        for candidate in MOCK_FALLBACK_LABELS:
            if candidate not in extended:
                extended.append(candidate)
        labels = extended
    video_tensor, meta = decode_video_to_tensor(input_p)

    probs, seed, temp = run_inference(
        video_tensor=video_tensor,
        labels=labels,
        weights_path=weights_p,
        use_mock=mock,
    )

    meta = {
        "seed": seed,
        "temp": temp,
        "frames": int(meta.get("frames", 0)),
        "fps": meta.get("fps", None),
        "width": meta.get("width", None),
        "height": meta.get("height", None),
    }
    mode = "MOCK" if mock else "REAL"
    return probs, labels, meta, mode, seed, temp


def infer_video(
    input_path: str,
    topk: int = 5,
    mock: bool = True,
    weights_path: str | None = None,
    labels_path: str | None = None,
    num_classes: int = 100,
    confidence_threshold: float = 0.50,
    margin_threshold: float = 0.15,
) -> dict:
    # --- AUTO num_classes from labels.json when REAL ---
    if not mock:
        try:
            lp = Path(labels_path) if labels_path else (paths.WEIGHTS_DIR / "labels.json")
            data = json.loads(lp.read_text(encoding="utf-8"))
            if isinstance(data, list):
                num_classes = len(data)
            elif isinstance(data, dict) and data:
                num_classes = max(int(k) for k in data.keys()) + 1
        except Exception:
            pass
    # -----------------------------------------------

    probs, labels, meta, mode, _, _ = _infer_core(
        input_path=input_path,
        topk=topk,
        mock=mock,
        weights_path=weights_path,
        labels_path=labels_path,
        num_classes=num_classes,
    )

    topk_list = extract_topk(labels, probs, topk)
    top1_score = float(topk_list[0]["score"])
    margin = None
    if len(topk_list) >= 2:
        margin = float(topk_list[0]["score"] - topk_list[1]["score"])
    status, reason = decide_unknown(
        topk_list,
        T=float(confidence_threshold),
        M=float(margin_threshold),
    )

    return {
        "status": status,
        "reason": reason,
        "top1": {
            "label": topk_list[0]["label"],
            "score": top1_score,
        },
        "topk": topk_list,
        "margin": margin,
        "confidence_threshold": float(confidence_threshold),
        "margin_threshold": float(margin_threshold),
        "mode": mode,
        "input": input_path,
        "meta": meta,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CPU-only offline inference.")
    parser.add_argument("--input", type=str, required=False, help="Path to input mp4.")
    parser.add_argument("--topk", type=int, default=5)
    parser.add_argument("--num_classes", type=int, default=DEFAULT_NUM_LABELS)
    parser.add_argument("--weights", type=str, default=str(paths.WEIGHTS_DIR / "model.ts"))
    parser.add_argument("--labels", type=str, default=str(paths.WEIGHTS_DIR / "labels.json"))
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--mock", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.device.lower() != "cpu":
        print("ERROR: only --device cpu is supported.")
        sys.exit(1)

    input_path = Path(args.input) if args.input else None
    weights_path = Path(args.weights)
    labels_path = Path(args.labels)

    if not args.dry_run and input_path is None:
        print("ERROR: --input is required unless --dry_run is set.")
        sys.exit(1)

    if args.dry_run:
        print("DRY_RUN: path validation")
        if input_path is not None:
            print(f"input: {input_path.resolve()}")
        print(f"weights: {weights_path.resolve()}")
        print(f"labels: {labels_path.resolve()}")
        sys.exit(0)

    num_classes = max(1, int(args.num_classes))
    if input_path is None:
        print("ERROR: --input is required unless --dry_run is set.")
        sys.exit(1)

    try:
        probs, labels, meta, tag, seed, temp = _infer_core(
            input_path=str(input_path),
            topk=max(1, args.topk),
            mock=args.mock,
            weights_path=str(weights_path),
            labels_path=str(labels_path),
            num_classes=num_classes,
        )
    except InferenceError as exc:
        print(exc.message)
        if exc.hint:
            print(exc.hint)
        sys.exit(exc.code if exc.code is not None else 1)
    except ValueError:
        print("ERROR: video opened but no frames were read.")
        sys.exit(3)
    except Exception as exc:
        print(f"ERROR: failed to decode video: {exc}")
        sys.exit(3)

    if args.mock and seed is not None and temp is not None:
        print(f"[MOCK] seed={seed} temp={temp:.2f}")

    topk = min(max(1, args.topk), len(labels))
    topk_text, top1_idx, top1_prob = format_topk(labels, probs, topk, mock_display=args.mock)
    print(f"[{tag}] input: {input_path}")
    print(f"pred: {labels[top1_idx]} ({top1_prob:.2f})")
    print(f"top-{topk}:")
    print(topk_text)
    print(f"SUMMARY: mode={tag} input={input_path} top1={labels[top1_idx]}({top1_prob:.2f}) topk={topk}")


if __name__ == "__main__":
    main()
