"use client";

import { useEffect, useMemo, useState } from "react";

type InferResult = {
  mode: string;
  input: string;
  pred_label: string;
  pred_prob: number;
  topk: { rank: number; label: string; prob: number }[];
  meta: {
    seed: number | null;
    temp: number | null;
    frames: number;
    fps: number | null;
    width: number | null;
    height: number | null;
    num_classes?: number;
    labels?: string[];
  };
};

const API_BASE = "http://127.0.0.1:8000";

class ApiError extends Error {
  error: string;
  hint?: string;
  code?: string;
  constructor(error: string, hint?: string, code?: string) {
    super(hint ? `${error} ${hint}` : error);
    this.error = error;
    this.hint = hint;
    this.code = code;
  }
}

async function postInfer(
  file: File,
  topk: number,
  mock: boolean,
  weights?: string,
  labels?: string
): Promise<InferResult> {
  const params = new URLSearchParams({ topk: String(topk), mock: String(mock) });
  if (!mock) {
    if (weights) params.set("weights", weights);
    if (labels) params.set("labels", labels);
  }
  const form = new FormData();
  form.append("file", file);
  const res = await fetch(`${API_BASE}/infer?${params.toString()}`, {
    method: "POST",
    body: form
  });
  let data: any = null;
  try {
    data = await res.json();
  } catch {
    data = null;
  }
  if (!res.ok) {
    const message = data?.error || "Request failed.";
    const hint = data?.hint;
    const code = data?.code;
    throw new ApiError(message, hint, code);
  }
  return data as InferResult;
}

function basename(path: string): string {
  const parts = path.split(/[/\\]/);
  return parts[parts.length - 1] || path;
}

export default function Page() {
  const [topk, setTopk] = useState(5);
  const [mock, setMock] = useState(true);
  const [weightsPath, setWeightsPath] = useState("weights/model.ts");
  const [labelsPath, setLabelsPath] = useState("weights/labels.json");
  const [status, setStatus] = useState("Idle");
  const [result, setResult] = useState<InferResult | null>(null);
  const [error, setError] = useState<string>("");
  const [realError, setRealError] = useState<{ error: string; hint?: string } | null>(null);
  const [apiOk, setApiOk] = useState<boolean | null>(null);

  const settingsText = useMemo(() => `mock=${mock} topk=${topk}`, [mock, topk]);
  const busy = status === "Recording" || status === "Uploading" || status === "Inferring";
  const metaClasses = result?.meta?.num_classes ?? result?.meta?.labels?.length;
  const metaFrames = result?.meta?.frames;

  useEffect(() => {
    let alive = true;
    fetch(`${API_BASE}/health`)
      .then((res) => res.json())
      .then((data) => {
        if (!alive) return;
        setApiOk(Boolean(data?.ok));
      })
      .catch(() => {
        if (!alive) return;
        setApiOk(false);
      });
    return () => {
      alive = false;
    };
  }, []);

  async function handleUpload(evt: React.FormEvent<HTMLFormElement>) {
    evt.preventDefault();
    setError("");
    setRealError(null);
    setResult(null);
    const input = evt.currentTarget.elements.namedItem("file") as HTMLInputElement | null;
    const file = input?.files?.[0];
    if (!file) {
      setError("Choose a video file first.");
      return;
    }
    try {
      setStatus("Uploading");
      const data = await postInfer(file, topk, mock, weightsPath, labelsPath);
      setStatus("Inferring");
      setResult(data);
      setStatus("Done");
    } catch (err) {
      setStatus("Error");
      if (err instanceof ApiError && !mock) {
        setRealError({ error: err.error, hint: err.hint });
      } else {
        setError((err as Error).message);
      }
    }
  }

  async function handleWebcam() {
    setError("");
    setRealError(null);
    setResult(null);
    if (!navigator.mediaDevices?.getUserMedia) {
      setError("Webcam not supported in this browser.");
      return;
    }
    try {
      setStatus("Recording");
      const stream = await navigator.mediaDevices.getUserMedia({ video: true, audio: false });
      const recorder = new MediaRecorder(stream);
      const chunks: BlobPart[] = [];
      recorder.ondataavailable = (evt) => {
        if (evt.data.size > 0) chunks.push(evt.data);
      };
      const stopPromise = new Promise<Blob>((resolve) => {
        recorder.onstop = () => {
          const blob = new Blob(chunks, { type: recorder.mimeType || "video/webm" });
          resolve(blob);
        };
      });
      recorder.start();
      await new Promise((r) => setTimeout(r, 2000));
      recorder.stop();
      stream.getTracks().forEach((t) => t.stop());
      const blob = await stopPromise;
      const ext = blob.type.includes("webm") ? "webm" : "mp4";
      const file = new File([blob], `webcam.${ext}`, { type: blob.type });
      setStatus("Uploading");
      const data = await postInfer(file, topk, mock, weightsPath, labelsPath);
      setStatus("Inferring");
      setResult(data);
      setStatus("Done");
    } catch (err) {
      setStatus("Error");
      if (err instanceof ApiError && !mock) {
        setRealError({ error: err.error, hint: err.hint });
      } else {
        setError((err as Error).message);
      }
    }
  }

  function handleReset() {
    setStatus("Idle");
    setResult(null);
    setError("");
    setRealError(null);
  }

  const top1 = result?.topk?.[0];

  return (
    <div className="page stack">
      <div className="card stack">
        <div className="row space">
          <h1>WLASL Demo</h1>
          <div className={`status ${apiOk ? "ok" : "bad"}`}>
            API: {apiOk ? "OK" : "DOWN"}
          </div>
        </div>
        {!apiOk && <div className="subtle">Start the API: ./scripts/run_server.sh</div>}
      </div>

      <div className="card stack">
        <div className="row space">
          <h2>Settings</h2>
          <div className="panel row" style={{ gap: 10, padding: "6px 10px" }}>
            <span className="badge">{mock ? "Model: MOCK" : "Model: REAL"}</span>
            {!mock && typeof metaClasses === "number" && (
              <span className="subtle">classes={metaClasses}</span>
            )}
            {!mock && typeof metaFrames === "number" && (
              <span className="subtle">frames={metaFrames}</span>
            )}
          </div>
        </div>
        <div className="row">
          <label className="label">Top‑K</label>
          <input
            className="input"
            type="number"
            min={1}
            max={10}
            value={topk}
            onChange={(e) => setTopk(Math.max(1, Math.min(10, Number(e.target.value))))}
          />
          <label className="label">
            <input
              type="checkbox"
              checked={mock}
              onChange={(e) => setMock(e.target.checked)}
            />{" "}
            Mock
          </label>
          <div className="label mono">{settingsText}</div>
        </div>
        <details>
          <summary className="label">Advanced (REAL)</summary>
          <div className="stack panel" style={{ marginTop: 8 }}>
            <label className="label">Weights path</label>
            <input
              className="input"
              type="text"
              value={weightsPath}
              onChange={(e) => setWeightsPath(e.target.value)}
              disabled={mock}
            />
            <label className="label">Labels path</label>
            <input
              className="input"
              type="text"
              value={labelsPath}
              onChange={(e) => setLabelsPath(e.target.value)}
              disabled={mock}
            />
            {mock && <div className="subtle">Disable Mock to use REAL paths.</div>}
          </div>
        </details>
      </div>

      <div className="card stack">
        <h2>Upload MP4/WebM</h2>
        <form className="row" onSubmit={handleUpload}>
          <input className="input" type="file" name="file" accept="video/mp4,video/webm" />
          <button className="button" type="submit" disabled={busy}>
            Upload & Infer
          </button>
        </form>
      </div>

      <div className="card stack">
        <h2>Record 2s Webcam</h2>
        <div className="row">
          <button className="button secondary" onClick={handleWebcam} disabled={busy}>
            Record & Infer
          </button>
          <button className="button ghost" onClick={handleReset} disabled={busy && status !== "Error"}>
            Reset
          </button>
        </div>
      </div>

      <div className="card stack">
        <h2>Status</h2>
        <div className="row">
          <div className={`status ${status === "Error" ? "bad" : "ok"}`}>{status}</div>
          {result && <div className="subtle">File: {basename(result.input)}</div>}
        </div>
        {realError && !mock && (
          <div className="panel">
            <div className="error">REAL model not ready</div>
            <div className="subtle">{realError.error}</div>
            {realError.hint && <div className="subtle">{realError.hint}</div>}
            <div style={{ marginTop: 10 }}>
              <button className="button ghost" onClick={() => setMock(true)}>
                Switch back to MOCK
              </button>
            </div>
          </div>
        )}
        {error && <div className="error">{error}</div>}
        {result && (
          <div className="stack">
            <div className="row">
              <span className="badge">Top‑1</span>
              <div className="highlight">
                {top1?.label} ({((top1?.prob ?? 0) * 100).toFixed(1)}%)
              </div>
            </div>
            <table className="table">
              <thead>
                <tr>
                  <th>Rank</th>
                  <th>Label</th>
                  <th>Prob</th>
                </tr>
              </thead>
              <tbody>
                {result.topk.map((item) => (
                  <tr key={item.rank}>
                    <td>{item.rank}</td>
                    <td>{item.label}</td>
                    <td>{(item.prob * 100).toFixed(1)}%</td>
                  </tr>
                ))}
              </tbody>
            </table>
            <div className="panel subtle">
              seed={result.meta.seed ?? "n/a"} temp={result.meta.temp ?? "n/a"} frames=
              {result.meta.frames} fps={result.meta.fps ?? "n/a"} size=
              {result.meta.width ?? "n/a"}x{result.meta.height ?? "n/a"}
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
