"use client";

import { useEffect, useMemo, useRef, useState } from "react";

type InferResult = {
  status: "ok" | "unknown";
  reason: "low_confidence" | "ambiguous" | null;
  top1: { label: string; score: number };
  topk: { label: string; score: number }[];
  confidence_threshold: number;
  margin_threshold: number;
  mode?: string;
  input?: string;
  meta?: {
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
  const [isRecording, setIsRecording] = useState(false);
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const recorderRef = useRef<MediaRecorder | null>(null);
  const chunksRef = useRef<BlobPart[]>([]);

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

  function stopStream() {
    if (streamRef.current) {
      streamRef.current.getTracks().forEach((t) => t.stop());
      streamRef.current = null;
    }
    if (videoRef.current) {
      videoRef.current.srcObject = null;
    }
  }

  function webcamErrorMessage(err: unknown): string {
    if (err instanceof DOMException) {
      if (err.name === "NotAllowedError" || err.name === "PermissionDeniedError") {
        return "Camera permission denied. Please allow camera access and try again.";
      }
      if (err.name === "NotFoundError" || err.name === "DevicesNotFoundError") {
        return "No camera found. Connect a camera and try again.";
      }
    }
    return (err as Error).message || "Unable to access webcam.";
  }

  async function ensureCameraReady() {
    if (streamRef.current && streamRef.current.getTracks().some((t) => t.readyState === "live")) {
      if (videoRef.current) videoRef.current.srcObject = streamRef.current;
      setStatus("Camera Ready");
      return streamRef.current;
    }
    if (!navigator.mediaDevices?.getUserMedia) {
      throw new Error("Webcam not supported in this browser.");
    }
    const stream = await navigator.mediaDevices.getUserMedia({ video: true, audio: false });
    streamRef.current = stream;
    if (videoRef.current) videoRef.current.srcObject = stream;
    setStatus("Camera Ready");
    return stream;
  }

  useEffect(() => {
    ensureCameraReady().catch((err) => {
      setStatus("Error");
      setError(webcamErrorMessage(err));
    });
    return () => {
      stopStream();
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

  async function handleStartRecording() {
    setError("");
    setRealError(null);
    setResult(null);
    try {
      const stream = await ensureCameraReady();
      if (typeof MediaRecorder === "undefined") {
        throw new Error("MediaRecorder not supported in this browser.");
      }
      chunksRef.current = [];
      const recorder = new MediaRecorder(stream);
      recorderRef.current = recorder;
      recorder.ondataavailable = (evt) => {
        if (evt.data.size > 0) chunksRef.current.push(evt.data);
      };
      recorder.onstart = () => {
        setIsRecording(true);
        setStatus("Recording");
      };
      recorder.start();
    } catch (err) {
      setStatus("Error");
      setError(webcamErrorMessage(err));
    }
  }

  async function handleStopRecording() {
    const recorder = recorderRef.current;
    if (!recorder || recorder.state !== "recording") return;
    setError("");
    setRealError(null);
    try {
      const stopPromise = new Promise<Blob>((resolve) => {
        recorder.onstop = () => {
          const blob = new Blob(chunksRef.current, { type: recorder.mimeType || "video/webm" });
          resolve(blob);
        };
      });
      recorder.stop();
      setIsRecording(false);
      setStatus("Inferring");
      const blob = await stopPromise;
      const file = new File([blob], "webcam.webm", { type: blob.type || "video/webm" });
      const data = await postInfer(file, topk, mock, weightsPath, labelsPath);
      setResult(data);
      setStatus("Done");
      recorderRef.current = null;
      chunksRef.current = [];
    } catch (err) {
      setIsRecording(false);
      setStatus("Error");
      if (err instanceof ApiError && !mock) {
        setRealError({ error: err.error, hint: err.hint });
      } else {
        setError(webcamErrorMessage(err));
      }
    }
  }

  function handleReset() {
    if (recorderRef.current && recorderRef.current.state === "recording") {
      recorderRef.current.stop();
    }
    setIsRecording(false);
    recorderRef.current = null;
    chunksRef.current = [];
    stopStream();
    setStatus("Idle");
    setResult(null);
    setError("");
    setRealError(null);
  }

  const top1 = result?.top1;

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
        <h2>Webcam</h2>
        <video
          ref={videoRef}
          autoPlay
          muted
          playsInline
          className="panel"
          style={{ width: "100%", maxHeight: 320, objectFit: "cover" }}
        />
        <div className="row">
          <button
            className="button secondary"
            onClick={handleStartRecording}
            disabled={busy || isRecording}
          >
            Start Recording
          </button>
          <button className="button secondary" onClick={handleStopRecording} disabled={!isRecording}>
            Stop Recording
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
          {result?.input && <div className="subtle">File: {basename(result.input)}</div>}
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
                {top1?.label} ({((top1?.score ?? 0) * 100).toFixed(1)}%)
              </div>
            </div>
            <div className="panel subtle">
              status={result.status} reason={result.reason ?? "n/a"} confidence_threshold=
              {result.confidence_threshold.toFixed(2)} margin_threshold=
              {result.margin_threshold.toFixed(2)}
            </div>
            <table className="table">
              <thead>
                <tr>
                  <th>Rank</th>
                  <th>Label</th>
                  <th>Score</th>
                </tr>
              </thead>
              <tbody>
                {result.topk.map((item, idx) => (
                  <tr key={`${item.label}-${idx}`}>
                    <td>{idx + 1}</td>
                    <td>{item.label}</td>
                    <td>{(item.score * 100).toFixed(1)}%</td>
                  </tr>
                ))}
              </tbody>
            </table>
            {result.meta && (
              <div className="panel subtle">
                seed={result.meta.seed ?? "n/a"} temp={result.meta.temp ?? "n/a"} frames=
                {result.meta.frames} fps={result.meta.fps ?? "n/a"} size=
                {result.meta.width ?? "n/a"}x{result.meta.height ?? "n/a"}
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
}
