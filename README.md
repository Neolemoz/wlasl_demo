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
