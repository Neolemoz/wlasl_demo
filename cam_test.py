import cv2
import time

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Camera not opened")
    exit()

# warmup
for _ in range(10):
    cap.read()

t0 = time.time()
n = 60
ok = 0

for _ in range(n):
    ret, frame = cap.read()
    if ret and frame is not None:
        ok += 1

dt = time.time() - t0
fps = ok / dt if dt > 0 else 0

if ok > 0:
    h, w = frame.shape[:2]
else:
    h, w = 0, 0

print("Read frames:", ok, "/", n)
print("Frame size:", w, "x", h)
print("Approx FPS:", round(fps, 1))

cap.release()
