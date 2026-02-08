import csv
import hashlib
import json
from pathlib import Path

VIDEO_EXTS = {".mp4", ".webm", ".avi"}
ANNOTATION_CANDIDATES = {
    "labels.json",
    "glossary.json",
    "classes.json",
    "annotations.json",
    "metadata.json",
    "train.json",
    "test.json",
    "splits.json",
}


def _iter_video_files(root: Path) -> list[Path]:
    return [p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in VIDEO_EXTS]


def _find_annotation_file(root: Path) -> Path | None:
    for name in ANNOTATION_CANDIDATES:
        cand = root / name
        if cand.exists():
            return cand
    for p in root.rglob("*"):
        if p.suffix.lower() == ".json":
            return p
    for p in root.rglob("*"):
        if p.suffix.lower() == ".csv":
            return p
    return None


def _hash_split(path: str) -> float:
    h = hashlib.sha256(path.encode("utf-8")).hexdigest()
    return int(h[:8], 16) / 0xFFFFFFFF


def _normalize_label(val: str) -> str:
    return str(val).strip()


def _extract_from_json(data: object, root: Path) -> list[dict]:
    samples: list[dict] = []

    def add_sample(p: str | Path, label: str) -> None:
        if not p:
            return
        path = Path(p)
        if not path.is_absolute():
            path = root / path
        if path.exists() and path.suffix.lower() in VIDEO_EXTS:
            samples.append({"path": str(path), "label": _normalize_label(label)})

    if isinstance(data, list):
        for item in data:
            if isinstance(item, dict):
                label = item.get("label") or item.get("gloss") or item.get("word")
                video = item.get("video") or item.get("path") or item.get("file") or item.get("url")
                if label and video:
                    add_sample(video, label)
    elif isinstance(data, dict):
        # common nested formats
        for key in ["annotations", "data", "items", "videos"]:
            if isinstance(data.get(key), list):
                samples.extend(_extract_from_json(data.get(key), root))
                if samples:
                    return samples
        # mapping: label -> list of videos
        for label, videos in data.items():
            if isinstance(videos, list):
                for v in videos:
                    add_sample(v, label)
    return samples


def _extract_from_csv(path: Path, root: Path) -> list[dict]:
    samples: list[dict] = []
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            label = row.get("label") or row.get("gloss") or row.get("word")
            video = row.get("video") or row.get("path") or row.get("file")
            if not label or not video:
                continue
            p = Path(video)
            if not p.is_absolute():
                p = root / p
            if p.exists() and p.suffix.lower() in VIDEO_EXTS:
                samples.append({"path": str(p), "label": _normalize_label(label)})
    return samples


def load_wlasl_dataset(
    data_root: str | Path,
    max_classes: int | None = None,
    max_samples_per_class: int | None = None,
) -> tuple[list[tuple[str, int]], list[tuple[str, int]], list[str]]:
    root = Path(data_root)
    if not root.exists():
        raise RuntimeError(f"data_root not found: {root}")

    ann_path = _find_annotation_file(root)
    samples: list[dict] = []
    if ann_path is not None:
        if ann_path.suffix.lower() == ".csv":
            samples = _extract_from_csv(ann_path, root)
        elif ann_path.suffix.lower() == ".json":
            with ann_path.open("r", encoding="utf-8") as f:
                data = json.load(f)
            samples = _extract_from_json(data, root)

    if not samples:
        video_files = _iter_video_files(root)
        for vf in video_files:
            label = vf.parent.name
            samples.append({"path": str(vf), "label": _normalize_label(label)})

    if not samples:
        raise RuntimeError(
            "Failed to build samples. Ensure annotations include video paths and labels."
        )

    # limit classes and samples per class
    class_to_samples: dict[str, list[dict]] = {}
    for s in samples:
        class_to_samples.setdefault(s["label"], []).append(s)

    labels = sorted(class_to_samples.keys())
    if max_classes is not None:
        labels = labels[: max(1, int(max_classes))]

    limited: list[dict] = []
    for label in labels:
        items = class_to_samples[label]
        if max_samples_per_class is not None:
            items = items[: max(1, int(max_samples_per_class))]
        limited.extend(items)

    label_to_idx = {label: idx for idx, label in enumerate(labels)}
    train: list[tuple[str, int]] = []
    val: list[tuple[str, int]] = []
    for s in limited:
        split_val = _hash_split(s["path"])
        sample = (s["path"], label_to_idx[s["label"]])
        if split_val < 0.9:
            train.append(sample)
        else:
            val.append(sample)

    return train, val, labels
