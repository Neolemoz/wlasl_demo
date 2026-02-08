import argparse
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

os.environ.setdefault("OPENCV_LOG_LEVEL", "ERROR")

import cv2

from . import paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Record a short webcam clip to file.")
    parser.add_argument("--out", type=str, default=str(paths.OUTPUTS_DIR / "webcam.mp4"))
    parser.add_argument("--seconds", type=int, default=2)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--preview", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    return parser.parse_args()


def _open_camera(device: int) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(device)
    if not cap.isOpened():
        print("ERROR: Unable to open webcam.")
        print("Hint: try another device index (e.g., --device 1)")
        print("Hint: list devices with: v4l2-ctl --list-devices")
        sys.exit(4)
    return cap


def _best_effort_size(path: Path) -> str:
    try:
        size = path.stat().st_size
        return f"{size} bytes"
    except Exception:
        return "unknown size"


def start_ffplay_preview(width: int, height: int, fps: int) -> subprocess.Popen | None:
    if shutil.which("ffplay") is None:
        return None
    cmd = [
        "ffplay",
        "-loglevel",
        "error",
        "-f",
        "rawvideo",
        "-pixel_format",
        "bgr24",
        "-video_size",
        f"{width}x{height}",
        "-framerate",
        str(fps),
        "-i",
        "-",
    ]
    try:
        return subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except Exception:
        return None


def send_frame_to_ffplay(proc: subprocess.Popen | None, frame) -> bool:
    if proc is None or proc.stdin is None:
        return False
    try:
        proc.stdin.write(frame.tobytes())
        return True
    except Exception:
        return False


def main() -> None:
    args = parse_args()
    paths.ensure_dirs()

    out_path = Path(args.out)
    seconds = max(1, int(args.seconds))
    fps = max(1, int(args.fps))

    cap = _open_camera(args.device)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, int(args.width))
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, int(args.height))
    cap.set(cv2.CAP_PROP_FPS, int(fps))

    if args.dry_run:
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        actual_fps = cap.get(cv2.CAP_PROP_FPS)
        print("DRY_RUN: webcam opened")
        print(f"device: {args.device}")
        print(f"resolution: {width}x{height}")
        print(f"fps: {actual_fps:.2f}")
        cap.release()
        sys.exit(0)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, fps, (int(args.width), int(args.height)))
    fallback_used = False

    if not writer.isOpened():
        fallback_used = True
        out_path = out_path.with_suffix(".avi")
        fourcc = cv2.VideoWriter_fourcc(*"XVID")
        writer = cv2.VideoWriter(str(out_path), fourcc, fps, (int(args.width), int(args.height)))
        if not writer.isOpened():
            cap.release()
            print("ERROR: Failed to open video writer (mp4v and XVID).")
            sys.exit(5)

    if fallback_used:
        print(f"Writer fallback: saving to {out_path}")

    target_frames = seconds * fps
    frames_written = 0
    start = time.monotonic()
    preview_enabled = bool(args.preview)
    preview_warned = False
    ffplay_proc = None

    if preview_enabled:
        ffplay_proc = start_ffplay_preview(int(args.width), int(args.height), fps)
        if ffplay_proc is None:
            print("HINT: ffplay not found, falling back to OpenCV preview (may print Qt warnings).")

    try:
        while frames_written < target_frames:
            ok, frame = cap.read()
            if not ok or frame is None:
                print("WARNING: failed to read frame; stopping early.")
                break
            writer.write(frame)
            frames_written += 1
            if preview_enabled:
                if ffplay_proc is not None:
                    ok_send = send_frame_to_ffplay(ffplay_proc, frame)
                    if not ok_send:
                        ffplay_proc = None
                else:
                    try:
                        cv2.imshow("WLASL Demo - Webcam", frame)
                        if cv2.waitKey(1) & 0xFF == ord("q"):
                            print("Preview stopped by user.")
                            break
                    except Exception:
                        if not preview_warned:
                            print("WARNING: preview unavailable; continuing without window.")
                            preview_warned = True
                        preview_enabled = False
    except KeyboardInterrupt:
        print("Recording interrupted. Finalizing output.")
    finally:
        cap.release()
        writer.release()
        if preview_enabled or preview_warned:
            try:
                cv2.destroyAllWindows()
            except Exception:
                pass
        if ffplay_proc is not None:
            try:
                if ffplay_proc.stdin:
                    ffplay_proc.stdin.close()
                ffplay_proc.terminate()
                ffplay_proc.wait(timeout=2)
            except Exception:
                pass

    elapsed = time.monotonic() - start
    print(f"Saved: {out_path} ({_best_effort_size(out_path)})")
    print(f"Frames: {frames_written} in {elapsed:.2f}s")


if __name__ == "__main__":
    main()
