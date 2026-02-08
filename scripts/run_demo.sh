#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

DRY_RUN=0
VIDEO_PATH=""
TOPK=5
WEBCAM=0
SECONDS=2
DEVICE=0
FPS=30
PREVIEW=0
MOCK=0
LABELS_PATH=""
WEIGHTS_PATH=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry_run) DRY_RUN=1; shift ;;
    --video) VIDEO_PATH="${2:-}"; shift 2 ;;
    --topk) TOPK="${2:-5}"; shift 2 ;;
    --webcam) WEBCAM=1; shift ;;
    --seconds) SECONDS="${2:-2}"; shift 2 ;;
    --device) DEVICE="${2:-0}"; shift 2 ;;
    --fps) FPS="${2:-30}"; shift 2 ;;
    --preview) PREVIEW=1; shift ;;
    --mock) MOCK=1; shift ;;
    --labels) LABELS_PATH="${2:-}"; shift 2 ;;
    --weights) WEIGHTS_PATH="${2:-}"; shift 2 ;;
    *) echo "Unknown arg: $1"; exit 1 ;;
  esac
done

if [[ ! -d ".venv" ]]; then
  echo "ERROR: .venv not found. Run: bash scripts/setup_venv.sh"
  exit 2
fi

source .venv/bin/activate

echo "== 1) Healthcheck =="
python scripts/healthcheck.py ${VIDEO_PATH:+--video "$VIDEO_PATH"}

echo
if [[ $DRY_RUN -eq 1 ]]; then
  echo "DRY_RUN: validation only"
  if [[ -n "$VIDEO_PATH" ]]; then
    if [[ -f "$VIDEO_PATH" ]]; then
      echo "Video: OK ($VIDEO_PATH)"
    else
      echo "Video: MISSING ($VIDEO_PATH)"
    fi
  fi
  if [[ $WEBCAM -eq 1 ]]; then
    python3 -m src.webcam_record --dry_run --device "$DEVICE"
  fi
  exit 0
fi

echo "== 2) Offline inference =="
if [[ -n "$VIDEO_PATH" ]]; then
  ARGS=(--input "$VIDEO_PATH" --topk "$TOPK")
  if [[ $MOCK -eq 1 ]]; then
    ARGS+=(--mock)
  fi
  if [[ -n "$LABELS_PATH" ]]; then
    ARGS+=(--labels "$LABELS_PATH")
  fi
  if [[ -n "$WEIGHTS_PATH" ]]; then
    ARGS+=(--weights "$WEIGHTS_PATH")
  fi
  python3 -m src.infer "${ARGS[@]}"
else
  echo "No --video provided."
fi

echo
echo "== 3) Webcam demo =="
if [[ $WEBCAM -eq 1 ]]; then
  PREVIEW_FLAG=""
  if [[ $PREVIEW -eq 1 ]]; then
    PREVIEW_FLAG="--preview"
  fi
  if [[ $PREVIEW -eq 1 ]]; then
    QT_LOGGING_RULES="*.debug=false;qt.qpa.*=false" QT_QPA_PLATFORM="xcb" OPENCV_LOG_LEVEL="ERROR" \
      python3 -m src.webcam_record --out outputs/webcam.mp4 --seconds "$SECONDS" --device "$DEVICE" --fps "$FPS" $PREVIEW_FLAG
  else
    python3 -m src.webcam_record --out outputs/webcam.mp4 --seconds "$SECONDS" --device "$DEVICE" --fps "$FPS" $PREVIEW_FLAG
  fi
  if [[ -f "outputs/webcam.mp4" ]]; then
    ARGS=(--input outputs/webcam.mp4 --topk "$TOPK")
    if [[ -n "$LABELS_PATH" ]]; then
      ARGS+=(--labels "$LABELS_PATH")
    fi
    if [[ -n "$WEIGHTS_PATH" ]]; then
      ARGS+=(--weights "$WEIGHTS_PATH")
    fi
    if [[ $MOCK -eq 1 || -z "$WEIGHTS_PATH" ]]; then
      ARGS+=(--mock)
    fi
    python3 -m src.infer "${ARGS[@]}"
  elif [[ -f "outputs/webcam.avi" ]]; then
    ARGS=(--input outputs/webcam.avi --topk "$TOPK")
    if [[ -n "$LABELS_PATH" ]]; then
      ARGS+=(--labels "$LABELS_PATH")
    fi
    if [[ -n "$WEIGHTS_PATH" ]]; then
      ARGS+=(--weights "$WEIGHTS_PATH")
    fi
    if [[ $MOCK -eq 1 || -z "$WEIGHTS_PATH" ]]; then
      ARGS+=(--mock)
    fi
    python3 -m src.infer "${ARGS[@]}"
  else
    echo "ERROR: webcam output not found."
    exit 3
  fi
else
  echo "Use --webcam to record and run inference."
fi

echo
echo "DONE"
