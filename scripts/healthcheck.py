#!/usr/bin/env python3
import argparse
import os
import platform
import shutil
import sys
from pathlib import Path

def check_ffmpeg() -> bool:
    return shutil.which("ffmpeg") is not None

def check_imports() -> dict:
    out = {}
    try:
        import cv2  # noqa
        out["opencv"] = True
    except Exception as e:
        out["opencv"] = f"FAIL: {e}"

    try:
        import torch  # noqa
        out["torch"] = True
    except Exception as e:
        out["torch"] = f"FAIL: {e}"

    return out

def check_video(video_path: Path) -> dict:
    import cv2
    res = {"path": str(video_path), "ok": False}

    if not video_path.exists():
        res["error"] = "File does not exist"
        return res

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        res["error"] = "OpenCV could not open video (codec/path issue)"
        return res

    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    ret, frame = cap.read()
    cap.release()
    if not ret or frame is None:
        res["error"] = "Could open container but failed to read first frame"
        return res

    res.update({"ok": True, "fps": fps, "width": w, "height": h})
    return res

def check_webcam(device_index: int) -> dict:
    import cv2
    res = {"device": device_index, "ok": False}

    cap = cv2.VideoCapture(device_index)
    if not cap.isOpened():
        res["error"] = "Could not open webcam (busy / permission / wrong index)"
        return res

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    ret, frame = cap.read()
    cap.release()

    if not ret or frame is None:
        res["error"] = "Opened webcam but failed to read a frame"
        return res

    res.update({"ok": True, "width": w, "height": h})
    return res

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", type=str, default=None, help="Optional mp4 path to validate decoding")
    ap.add_argument("--webcam", action="store_true", help="Validate webcam can be opened")
    ap.add_argument("--device", type=int, default=0, help="Webcam device index, default 0")
    args = ap.parse_args()

    print("=== WLASL Demo Healthcheck ===")
    print(f"Python: {sys.version.split()[0]}")
    print(f"Platform: {platform.platform()}")
    print(f"CWD: {os.getcwd()}")

    ffmpeg_ok = check_ffmpeg()
    print(f"ffmpeg: {'OK' if ffmpeg_ok else 'MISSING'}")
    if not ffmpeg_ok:
        print("  Hint: sudo apt install -y ffmpeg")

    imports = check_imports()
    for k, v in imports.items():
        print(f"{k}: {v}")

    vres = None
    wres = None

    if args.video:
        vres = check_video(Path(args.video))
        print("video_check:", vres)

    if args.webcam:
        wres = check_webcam(args.device)
        print("webcam_check:", wres)
        if not wres.get("ok"):
            print("  Hints:")
            print("   - list devices: v4l2-ctl --list-devices")
            print("   - try another index: --device 1")
            print("   - close apps using the camera (Zoom/Browser/etc.)")

    core_ok = (imports.get("opencv") is True) and (imports.get("torch") is True) and ffmpeg_ok
    if not core_ok:
        sys.exit(2)

    if args.video and (not vres or not vres.get("ok", False)):
        sys.exit(3)
    if args.webcam and (not wres or not wres.get("ok", False)):
        sys.exit(4)

    print("OK: healthcheck passed")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
