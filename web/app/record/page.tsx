"use client";

import { useEffect, useRef, useState } from "react";

const API_URL = "http://localhost:8000/upload_clip";
const LABELS = ["go", "drink", "help", "before", "yes", "who", "computer", "walk", "orange", "many"];

function pickRecorderMimeType(): string | undefined {
  if (typeof MediaRecorder === "undefined" || typeof MediaRecorder.isTypeSupported !== "function") {
    return undefined;
  }
  const candidates = [
    "video/mp4;codecs=avc1.42E01E",
    "video/mp4",
    "video/webm;codecs=vp9",
    "video/webm;codecs=vp8",
    "video/webm",
  ];
  return candidates.find((mime) => MediaRecorder.isTypeSupported(mime));
}

export default function RecordPage() {
  const liveVideoRef = useRef<HTMLVideoElement | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const recorderRef = useRef<MediaRecorder | null>(null);
  const chunksRef = useRef<BlobPart[]>([]);
  const mimeTypeRef = useRef<string>("");
  const lastClipUrlRef = useRef<string>("");

  const [label, setLabel] = useState(LABELS[0]);
  const [status, setStatus] = useState("Idle");
  const [isRecording, setIsRecording] = useState(false);
  const [lastClipUrl, setLastClipUrl] = useState<string>("");

  useEffect(() => {
    let active = true;
    async function startCamera() {
      try {
        if (!navigator.mediaDevices?.getUserMedia) {
          setStatus("Error: getUserMedia is not supported in this browser.");
          return;
        }
        const stream = await navigator.mediaDevices.getUserMedia({
          video: {
            width: 640,
            height: 480,
            frameRate: 30,
          },
          audio: false,
        });
        if (!active) {
          stream.getTracks().forEach((track) => track.stop());
          return;
        }
        streamRef.current = stream;
        if (liveVideoRef.current) {
          liveVideoRef.current.srcObject = stream;
          liveVideoRef.current.onloadedmetadata = () => liveVideoRef.current?.play().catch(() => {});
        }
        setStatus("Camera ready");
      } catch (err) {
        const message = err instanceof Error ? err.message : "Unable to access webcam.";
        setStatus(`Error: ${message}`);
      }
    }
    startCamera();
    return () => {
      active = false;
      if (recorderRef.current && recorderRef.current.state === "recording") {
        recorderRef.current.stop();
      }
      if (streamRef.current) {
        streamRef.current.getTracks().forEach((track) => track.stop());
      }
      if (lastClipUrlRef.current) {
        URL.revokeObjectURL(lastClipUrlRef.current);
      }
    };
  }, []);

  function handleStart() {
    try {
      const stream = streamRef.current;
      if (!stream) {
        setStatus("Error: camera stream is not ready.");
        return;
      }
      if (typeof MediaRecorder === "undefined") {
        setStatus("Error: MediaRecorder is not supported in this browser.");
        return;
      }
      const mimeType = pickRecorderMimeType();
      const recorder = mimeType ? new MediaRecorder(stream, { mimeType }) : new MediaRecorder(stream);
      mimeTypeRef.current = mimeType || recorder.mimeType || "video/webm";
      chunksRef.current = [];
      recorder.ondataavailable = (event: BlobEvent) => {
        if (event.data.size > 0) {
          chunksRef.current.push(event.data);
        }
      };
      recorder.onstart = () => {
        setIsRecording(true);
        setStatus(`Recording (${mimeTypeRef.current || "default"})`);
      };
      recorder.onerror = () => {
        setStatus("Error: recording failed.");
        setIsRecording(false);
      };
      recorderRef.current = recorder;
      recorder.start();
    } catch (err) {
      const message = err instanceof Error ? err.message : "Unable to start recording.";
      setStatus(`Error: ${message}`);
      setIsRecording(false);
    }
  }

  async function handleStop() {
    const recorder = recorderRef.current;
    if (!recorder || recorder.state !== "recording") {
      setStatus("Error: no active recording.");
      return;
    }

    setStatus("Stopping...");
    const blob = await new Promise<Blob>((resolve, reject) => {
      recorder.onstop = () => {
        try {
          resolve(new Blob(chunksRef.current, { type: mimeTypeRef.current || "video/webm" }));
        } catch (err) {
          reject(err);
        }
      };
      recorder.onerror = () => reject(new Error("Recorder stopped with an error."));
      recorder.stop();
    }).catch((err) => {
      const message = err instanceof Error ? err.message : "Unknown stop error.";
      setStatus(`Error: ${message}`);
      return null;
    });

    setIsRecording(false);
    recorderRef.current = null;

    if (!blob) {
      return;
    }

    if (lastClipUrl) {
      URL.revokeObjectURL(lastClipUrl);
    }
    const clipUrl = URL.createObjectURL(blob);
    lastClipUrlRef.current = clipUrl;
    setLastClipUrl(clipUrl);

    try {
      setStatus("Uploading...");
      const isMp4 = (mimeTypeRef.current || "").toLowerCase().includes("mp4");
      const ext = isMp4 ? "mp4" : "webm";
      const file = new File([blob], `recorded.${ext}`, { type: mimeTypeRef.current || blob.type || `video/${ext}` });
      const form = new FormData();
      form.append("label", label);
      form.append("file", file);

      const res = await fetch(API_URL, {
        method: "POST",
        body: form,
      });
      const data = await res.json().catch(() => ({}));
      if (!res.ok) {
        const message = data?.error || "Upload failed.";
        const code = data?.code ? ` (${data.code})` : "";
        setStatus(`Error: ${message}${code}`);
        return;
      }
      setStatus(`Uploaded: ${data.path || "saved"}`);
    } catch (err) {
      const message = err instanceof Error ? err.message : "Upload request failed.";
      setStatus(`Error: ${message}`);
    } finally {
      chunksRef.current = [];
    }
  }

  return (
    <main style={{ maxWidth: 900, margin: "0 auto", padding: "1.25rem", display: "grid", gap: "1rem" }}>
      <a href="/">Back to Home</a>
      <h1>Record Labeled Clip</h1>

      <label>
        Label:{" "}
        <select value={label} onChange={(e) => setLabel(e.target.value)}>
          {LABELS.map((item) => (
            <option key={item} value={item}>
              {item}
            </option>
          ))}
        </select>
      </label>

      <div style={{ display: "flex", gap: "0.75rem" }}>
        <button onClick={handleStart} disabled={isRecording}>
          Start
        </button>
        <button onClick={handleStop} disabled={!isRecording}>
          Stop
        </button>
      </div>

      <p>Status: {status}</p>

      <section>
        <h2>Live Webcam</h2>
        <div
          style={{
            width: "100%",
            maxWidth: 640,
            margin: "0 auto",
          }}
        >
          <div
            style={{
              position: "relative",
              width: "100%",
              paddingTop: "75%",
              background: "#000",
              borderRadius: 12,
              overflow: "hidden",
            }}
          >
            <video
              ref={liveVideoRef}
              autoPlay
              muted
              playsInline
              style={{
                position: "absolute",
                inset: 0,
                width: "100%",
                height: "100%",
                objectFit: "contain",
              }}
            />
          </div>
        </div>
      </section>

      <section>
        <h2>Last Recorded Clip</h2>
        {lastClipUrl ? (
          <video controls playsInline src={lastClipUrl} style={{ width: "100%", maxWidth: 640, background: "#000" }} />
        ) : (
          <p>No clip recorded yet.</p>
        )}
      </section>
    </main>
  );
}
