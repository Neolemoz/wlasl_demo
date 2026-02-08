from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = REPO_ROOT / "data"
WEIGHTS_DIR = REPO_ROOT / "weights"
VIDEOS_DIR = REPO_ROOT / "videos"
OUTPUTS_DIR = REPO_ROOT / "outputs"
CACHE_DIR = REPO_ROOT / "cache"

def ensure_dirs() -> None:
    for p in [DATA_DIR, WEIGHTS_DIR, VIDEOS_DIR, OUTPUTS_DIR, CACHE_DIR]:
        p.mkdir(parents=True, exist_ok=True)
