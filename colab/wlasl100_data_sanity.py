"""WLASL100 Data Sanity Check (Colab-ready).

Run in Colab or locally after setting the parameter constants below.
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# =========================
# Parameters (edit these)
# =========================
DATA_ROOT = "/content/data"
ANN_PATH = "/content/data/annotations.json"
LABELS_PATH = "/content/out/labels.json"
SPLIT = "train"  # "train" or "val"
SAMPLE_VIS_N = 5
DECODE_CHECK_N = 50
SEED = 1337
ALLOW_CSV = False


VIDEO_KEYS = [
    "video_relpath",
    "video_path",
    "video",
    "path",
    "file",
]
LABEL_KEYS = ["label", "gloss", "word", "class", "class_name"]
ID_KEYS = ["class_id", "label_id", "id", "target"]


def _as_int(value: Any) -> int | None:
    try:
        return int(value)
    except Exception:
        return None


def load_label_mapping(labels_path: Path) -> tuple[dict[str, int], dict[int, str]]:
    with labels_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    label_to_id: dict[str, int] = {}
    id_to_label: dict[int, str] = {}

    if isinstance(data, list):
        for idx, label in enumerate(data):
            label_s = str(label)
            label_to_id[label_s] = idx
            id_to_label[idx] = label_s
    elif isinstance(data, dict):
        numeric_key_dict = all(str(k).isdigit() for k in data.keys())
        if numeric_key_dict:
            for k, v in data.items():
                idx = int(k)
                label_s = str(v)
                label_to_id[label_s] = idx
                id_to_label[idx] = label_s
        else:
            for k, v in data.items():
                label_s = str(k)
                idx = int(v)
                label_to_id[label_s] = idx
                id_to_label[idx] = label_s
    else:
        raise ValueError(f"Unsupported LABELS_PATH format: {type(data)}")

    # Hard checks for WLASL100 mapping used by training/export.
    assert len(label_to_id) == len(id_to_label), "Duplicate labels or ids in mapping"
    assert len(label_to_id) == 100, f"Expected 100 classes, got {len(label_to_id)}"
    ids_sorted = sorted(id_to_label.keys())
    assert ids_sorted == list(range(100)), "Class ids must be contiguous [0..99]"

    return label_to_id, id_to_label


def _get_first(d: dict[str, Any], keys: list[str]) -> Any:
    for key in keys:
        if key in d and d[key] is not None:
            return d[key]
    return None


def _schema_preview_json(data: Any) -> None:
    print("Annotation schema preview:")
    if isinstance(data, list):
        print(f"- root type: list (len={len(data)})")
        if data and isinstance(data[0], dict):
            print(f"- first item keys: {sorted(list(data[0].keys()))[:20]}")
    elif isinstance(data, dict):
        print(f"- root type: dict")
        print(f"- top-level keys: {sorted(list(data.keys()))[:30]}")
        for key in ["train", "val", "test", "annotations", "data", "items", "videos"]:
            if isinstance(data.get(key), list):
                print(f"- key '{key}' is list (len={len(data[key])})")
                if data[key] and isinstance(data[key][0], dict):
                    print(f"- first {key} item keys: {sorted(list(data[key][0].keys()))[:20]}")
                break
    else:
        print(f"- root type: {type(data)}")


def _extract_json_records(data: Any, split: str) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []

    # Repo-first: list of items containing label/gloss/word + video/path/file.
    def parse_item(item: dict[str, Any]) -> dict[str, Any] | None:
        rel = _get_first(item, VIDEO_KEYS)
        label = _get_first(item, LABEL_KEYS)
        class_id = _get_first(item, ID_KEYS)
        item_split = item.get("split")

        if item_split is not None and str(item_split).lower() != split.lower():
            return None
        if rel is None:
            return None

        rec: dict[str, Any] = {"video_relpath": str(rel)}
        if label is not None:
            rec["label"] = str(label)
        cid = _as_int(class_id)
        if cid is not None:
            rec["class_id"] = cid
        return rec

    if isinstance(data, dict):
        # Explicit split key first.
        if isinstance(data.get(split), list):
            data = data[split]
        else:
            for key in ["annotations", "data", "items", "videos"]:
                if isinstance(data.get(key), list):
                    data = data[key]
                    break

    if isinstance(data, list):
        for item in data:
            if not isinstance(item, dict):
                continue
            parsed = parse_item(item)
            if parsed is not None:
                records.append(parsed)

    return records


def load_annotations(ann_path: Path, split: str) -> list[dict[str, Any]]:
    suffix = ann_path.suffix.lower()
    if suffix == ".json":
        with ann_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        _schema_preview_json(data)
        records = _extract_json_records(data, split=split)
        if not records:
            raise ValueError("ANNOTATION_FORMAT_UNRECOGNIZED")
        return records

    if suffix == ".csv":
        if not ALLOW_CSV:
            raise ValueError("CSV_NOT_SUPPORTED_IN_PR4")
        raise ValueError("ANNOTATION_FORMAT_UNRECOGNIZED")

    raise ValueError(f"Unsupported annotation format: {ann_path}")


def enrich_and_validate_records(
    records: list[dict[str, Any]],
    data_root: Path,
    label_to_id: dict[str, int],
    id_to_label: dict[int, str],
) -> list[dict[str, Any]]:
    unknown_labels: list[str] = []
    unknown_ids: list[int] = []
    enriched: list[dict[str, Any]] = []

    for rec in records:
        rel = str(rec["video_relpath"])
        p = Path(rel)
        abs_path = p if p.is_absolute() else (data_root / p)

        class_id = rec.get("class_id")
        label = rec.get("label")

        if label is not None:
            if label not in label_to_id:
                unknown_labels.append(label)
                continue
            class_id = label_to_id[label]

        if class_id is not None:
            class_id = int(class_id)
            if class_id not in id_to_label:
                unknown_ids.append(class_id)
                continue
            label = id_to_label[class_id]

        if class_id is None and label is None:
            continue

        enriched.append(
            {
                "video_relpath": rel,
                "video_path": str(abs_path),
                "label": str(label),
                "class_id": int(class_id),
            }
        )

    print("\nMapping alignment report:")
    print(f"- unknown_labels_count: {len(unknown_labels)}")
    if unknown_labels:
        print(f"- unknown_labels_examples: {sorted(set(unknown_labels))[:10]}")
    print(f"- unknown_ids_count: {len(unknown_ids)}")
    if unknown_ids:
        print(f"- unknown_ids_examples: {sorted(set(unknown_ids))[:10]}")

    return enriched


def distribution_report(df: pd.DataFrame, id_to_label: dict[int, str]) -> None:
    counts = df.groupby("class_id").size().sort_values()
    counts_df = counts.rename("clips").reset_index()
    counts_df["label"] = counts_df["class_id"].map(id_to_label)
    counts_df = counts_df[["class_id", "label", "clips"]]

    min_v = int(counts.min()) if not counts.empty else 0
    med_v = float(counts.median()) if not counts.empty else 0.0
    max_v = int(counts.max()) if not counts.empty else 0

    print("\nClips-per-class summary:")
    print(f"- min clips/class: {min_v}")
    print(f"- median clips/class: {med_v:.2f}")
    print(f"- max clips/class: {max_v}")

    print("\nTop 10 smallest classes:")
    print(counts_df.head(10).to_string(index=False))

    print("\nTop 10 largest classes:")
    print(counts_df.tail(10).sort_values("clips", ascending=False).to_string(index=False))


def decode_subset_health(df: pd.DataFrame, seed: int, decode_check_n: int) -> tuple[pd.DataFrame, dict[str, Any]]:
    exists_mask = df["video_path"].map(lambda p: Path(p).exists())
    missing_file_count = int((~exists_mask).sum())
    existing_df = df[exists_mask].copy()

    rng = random.Random(seed)
    subset_n = min(decode_check_n, len(existing_df))
    subset_idx = rng.sample(list(existing_df.index), subset_n) if subset_n > 0 else []
    subset_df = existing_df.loc[subset_idx].copy()

    decode_ok_rows: list[dict[str, Any]] = []
    decode_failed_count = 0

    duration_values: list[float] = []
    invalid_fps_or_frames = 0

    for row in subset_df.to_dict(orient="records"):
        path = row["video_path"]
        cap = cv2.VideoCapture(path)
        try:
            if not cap.isOpened():
                decode_failed_count += 1
                continue

            ok, frame = cap.read()
            if not ok or frame is None:
                decode_failed_count += 1
                continue

            fps = float(cap.get(cv2.CAP_PROP_FPS))
            frame_count = float(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            duration_sec = None
            if fps > 0 and frame_count > 0:
                duration_sec = frame_count / fps
                duration_values.append(duration_sec)
            else:
                invalid_fps_or_frames += 1

            row["fps"] = fps if fps > 0 else np.nan
            row["frame_count"] = frame_count if frame_count > 0 else np.nan
            row["duration_sec"] = duration_sec if duration_sec is not None else np.nan
            decode_ok_rows.append(row)
        finally:
            cap.release()

    metrics = {
        "missing_file_count": missing_file_count,
        "decode_failed_count": decode_failed_count,
        "decode_checked_count": subset_n,
        "invalid_fps_or_frame_count": invalid_fps_or_frames,
    }

    if duration_values:
        arr = np.array(duration_values, dtype=np.float64)
        metrics["duration_min_sec"] = float(arr.min())
        metrics["duration_median_sec"] = float(np.median(arr))
        metrics["duration_max_sec"] = float(arr.max())
    else:
        metrics["duration_min_sec"] = np.nan
        metrics["duration_median_sec"] = np.nan
        metrics["duration_max_sec"] = np.nan

    return pd.DataFrame(decode_ok_rows), metrics


def sample_frames_for_vis(video_path: str, n_frames: int = 4) -> list[np.ndarray]:
    cap = cv2.VideoCapture(video_path)
    try:
        if not cap.isOpened():
            return []

        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total > 0:
            idx = np.linspace(0, max(0, total - 1), n_frames).round().astype(int)
            idx = np.clip(idx, 0, max(0, total - 1))
            frames = []
            for i in idx:
                cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
                ok, frame = cap.read()
                if ok and frame is not None:
                    frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            return frames

        frames = []
        while len(frames) < n_frames:
            ok, frame = cap.read()
            if not ok or frame is None:
                break
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        return frames
    finally:
        cap.release()


def visualize_samples(decoded_ok_df: pd.DataFrame, sample_vis_n: int, seed: int) -> None:
    if decoded_ok_df.empty:
        print("\nVisualization skipped: no decodable samples in subset.")
        return

    rng = random.Random(seed)
    vis_n = min(sample_vis_n, len(decoded_ok_df))
    picks = rng.sample(list(decoded_ok_df.index), vis_n)

    print(f"\nVisualizing {vis_n} random videos (seed={seed})...")
    for idx in picks:
        row = decoded_ok_df.loc[idx]
        frames = sample_frames_for_vis(row["video_path"], n_frames=4)
        if not frames:
            continue

        fig, axes = plt.subplots(1, len(frames), figsize=(4 * len(frames), 3))
        if len(frames) == 1:
            axes = [axes]

        title = f"{Path(row['video_path']).name} | label={row['label']} | class_id={row['class_id']}"
        fig.suptitle(title)

        for ax, frame in zip(axes, frames):
            ax.imshow(frame)
            ax.axis("off")
        plt.tight_layout()
        plt.show()


def main() -> None:
    np.random.seed(SEED)

    data_root = Path(DATA_ROOT)
    ann_path = Path(ANN_PATH)
    labels_path = Path(LABELS_PATH)

    print("=== WLASL100 Data Sanity Check ===")
    print(f"DATA_ROOT={data_root}")
    print(f"ANN_PATH={ann_path}")
    print(f"LABELS_PATH={labels_path}")
    print(f"SPLIT={SPLIT}")
    print(f"SAMPLE_VIS_N={SAMPLE_VIS_N} DECODE_CHECK_N={DECODE_CHECK_N} SEED={SEED}")
    print(f"ALLOW_CSV={ALLOW_CSV}")

    assert data_root.exists(), f"DATA_ROOT not found: {data_root}"
    assert ann_path.exists(), f"ANN_PATH not found: {ann_path}"
    assert labels_path.exists(), f"LABELS_PATH not found: {labels_path}"

    label_to_id, id_to_label = load_label_mapping(labels_path)
    print("\nLabel mapping summary:")
    print(f"- num_classes: {len(label_to_id)}")
    first10 = sorted(id_to_label.items(), key=lambda x: x[0])[:10]
    print(f"- first_10_labels: {first10}")

    raw_records = load_annotations(ann_path, split=SPLIT)
    print(f"\nLoaded annotation records: {len(raw_records)}")

    records = enrich_and_validate_records(raw_records, data_root, label_to_id, id_to_label)
    df = pd.DataFrame(records)
    if df.empty:
        raise RuntimeError("No valid records after mapping alignment checks.")

    print("\nSplit summary:")
    print(f"- total_clips: {len(df)}")
    print(f"- total_classes_in_split: {df['class_id'].nunique()}")

    distribution_report(df, id_to_label)

    decoded_ok_df, metrics = decode_subset_health(df, seed=SEED, decode_check_n=DECODE_CHECK_N)
    print("\nFile/decode health:")
    print(f"- missing_file_count: {metrics['missing_file_count']}")
    print(f"- decode_failed_count: {metrics['decode_failed_count']}")
    print(f"- decode_checked_count: {metrics['decode_checked_count']}")

    print("\nDuration stats (decoded subset):")
    print("NOTE: duration stats use OpenCV metadata and may be unreliable for some codecs.")
    print(f"- min_sec: {metrics['duration_min_sec']}")
    print(f"- median_sec: {metrics['duration_median_sec']}")
    print(f"- max_sec: {metrics['duration_max_sec']}")
    print(f"- invalid_fps_or_frame_count: {metrics['invalid_fps_or_frame_count']}")
    decode_checked = int(metrics["decode_checked_count"])
    invalid_count = int(metrics["invalid_fps_or_frame_count"])
    if decode_checked > 0 and (invalid_count / decode_checked) > 0.2:
        print("WARNING: Many videos have invalid FPS/frame_count; consider ffprobe in later phases.")

    visualize_samples(decoded_ok_df, sample_vis_n=SAMPLE_VIS_N, seed=SEED)

    print("\nDONE checklist:")
    print("- [x] prints min/median/max clips per class")
    print("- [x] prints 10 smallest / 10 largest classes")
    print("- [x] shows 5 sample videos (frames) when decodable samples exist")
    print("- [x] prints missing_file_count + decode_failed_count")
    print("- [x] confirms label<->id mapping used for training/export")


if __name__ == "__main__":
    main()
