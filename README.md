# wlasl_demo

## Offline Inference

CPU-only, offline inference via `src/infer.py`. This step supports:
- Dry run (validate paths)
- Mock inference (deterministic, no weights)
- Real inference (TorchScript only)

### Dry run
```bash
python3 -m src.infer --input videos/sample.mp4 --dry_run
```

### Mock inference (no weights)
```bash
python3 -m src.infer --input videos/sample.mp4 --mock --topk 5
```

### Real inference (TorchScript only)
Place a TorchScript model and labels manually under `weights/` (CPU-only).
```bash
python3 -m src.infer --input videos/sample.mp4 --weights weights/model.ts --labels weights/labels.json
```

Notes:
- No automatic downloads. Provide weights and labels manually.
- CPU-only inference. `--device cpu` is the only supported option.

## Webcam demo

Dry run webcam check:
```bash
python3 -m src.webcam_record --dry_run
```

Run webcam demo:
```bash
./scripts/run_demo.sh --webcam
```

Run with preview (press q to stop). `--preview` uses ffplay when available for a clean preview window:
```bash
./scripts/run_demo.sh --webcam --preview
```

Optional args:
- `--seconds <int>`
- `--device <int>`
- `--fps <int>`

The demo saves to `outputs/webcam.mp4` (or `outputs/webcam.avi` fallback) and runs mock inference by default.

## Demo entrypoint

Offline (mock):
```bash
./scripts/run_demo.sh --video videos/sample.mp4 --mock
```

Offline (real, requires weights):
```bash
./scripts/run_demo.sh --video videos/sample.mp4
```

Webcam (preview + mock):
```bash
./scripts/run_demo.sh --webcam --preview --mock
```

Dry run examples:
```bash
./scripts/run_demo.sh --video videos/sample.mp4 --dry_run
./scripts/run_demo.sh --webcam --dry_run
```

## Labels & Weights

Labels file:
- `weights/labels.json` (list of strings or dict)

Weights file:
- `weights/model.ts` (TorchScript)

Examples:
```bash
./scripts/run_demo.sh --video videos/sample.mp4 --mock
./scripts/run_demo.sh --video videos/sample.mp4 --weights weights/model.ts --labels weights/labels.json
./scripts/run_demo.sh --webcam --preview --mock
```

## Local Inference API

Install deps:
```bash
pip install -r requirements.txt
```

Run server:
```bash
./scripts/run_server.sh
```

Test:
```bash
curl -s http://127.0.0.1:8000/health
curl -s -X POST "http://127.0.0.1:8000/infer?mock=true&topk=5" -F "file=@videos/sample.mp4"
```

Notes:
- `mock=true` by default for MVP.
- Weights and labels must be placed manually under `weights/`.

## Web UI (Next.js)

Start API:
```bash
./scripts/run_server.sh
```

Start web:
```bash
./scripts/run_web.sh
```

Open:
```bash
http://127.0.0.1:3000
```

## Demoday (One Command)

Steps:
1. Start everything:
```bash
./scripts/demoday.sh
```
2. Open browser:
```bash
http://127.0.0.1:3000
```

Demo flow:
- Upload MP4/WebM OR Record 2s Webcam
- Default = MOCK (always works)
- Toggle MOCK off to use REAL model (if weights present)

### REAL model setup
- Place `weights/model.ts`
- Place `weights/labels.json`
- Files are ignored by git (do NOT commit)
- If REAL is enabled without weights, the UI will guide you back to MOCK

## Step 7: Colab training + export

Train a tiny baseline on the Kaggle WLASL processed dataset and export TorchScript for CPU inference.
Kaggle download requires a `kaggle.json` token and must be run in Colab.

Use the notebook:
- `colab/Step7_Colab_Train_Export.ipynb`

Outputs from Colab:
- `model.ts`
- `labels.json`

Place them locally:
- `weights/model.ts`
- `weights/labels.json`

Then REAL mode works without any code changes.
