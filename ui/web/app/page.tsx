"use client";

import { useCallback, useRef, useState } from "react";
import dynamic from "next/dynamic";
import FaceOverlayCanvas from "./components/FaceOverlayCanvas";
import FaceResultCard from "./components/FaceResultCard";
import { PredictResponse } from "./types";

// react-webcam uses browser APIs � load client-side only
const CameraCapture = dynamic(() => import("./components/CameraCapture"), {
  ssr: false,
});

type Tab = "upload" | "camera";
type Status = "idle" | "preview" | "analyzing" | "done" | "error";

const API = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

export default function Home() {
  const [tab, setTab] = useState<Tab>("upload");
  const [status, setStatus] = useState<Status>("idle");
  const [imageSrc, setImageSrc] = useState<string | null>(null);
  const [result, setResult] = useState<PredictResponse | null>(null);
  const [errorMsg, setErrorMsg] = useState<string>("");
  const fileInputRef = useRef<HTMLInputElement>(null);

  const reset = () => {
    setStatus("idle");
    setImageSrc(null);
    setResult(null);
    setErrorMsg("");
    if (fileInputRef.current) fileInputRef.current.value = "";
  };

  const loadImageFile = (file: File) => {
    const url = URL.createObjectURL(file);
    setImageSrc(url);
    setStatus("preview");
    setResult(null);
  };

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) loadImageFile(file);
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    const file = e.dataTransfer.files?.[0];
    if (file) loadImageFile(file);
  };

  const handleCameraCapture = (dataUrl: string) => {
    setImageSrc(dataUrl);
    setStatus("preview");
    setResult(null);
  };

  const analyze = useCallback(async () => {
    if (!imageSrc) return;
    setStatus("analyzing");
    setErrorMsg("");

    try {
      const res = await fetch(imageSrc);
      const blob = await res.blob();
      const form = new FormData();
      form.append("file", blob, "image.jpg");

      const apiRes = await fetch(`${API}/predict`, { method: "POST", body: form });
      if (!apiRes.ok) {
        const err = await apiRes.json().catch(() => ({ detail: apiRes.statusText }));
        throw new Error(err.detail ?? "Server error");
      }

      const data: PredictResponse = await apiRes.json();
      setResult(data);
      setStatus("done");
    } catch (err) {
      setErrorMsg(err instanceof Error ? err.message : "Unknown error");
      setStatus("error");
    }
  }, [imageSrc]);

  return (
    <main className="min-h-screen bg-[#0d0d0d] text-white">
      <div className="max-w-2xl mx-auto px-4 py-12 space-y-8">

        {/* Header */}
        <div className="text-center space-y-2">
          <h1 className="text-3xl font-bold tracking-tight">Age &amp; Gender Detection</h1>
          <p className="text-white/50 text-sm">
            Upload a photo or use your camera faces are detected and analysed automatically.
          </p>
        </div>

        {/* Tab switcher */}
        <div className="flex rounded-xl bg-white/5 p-1 gap-1">
          {(["upload", "camera"] as Tab[]).map((t) => (
            <button
              key={t}
              onClick={() => { setTab(t); reset(); }}
              className={`flex-1 py-2.5 rounded-lg text-sm font-medium transition-all ${
                tab === t
                  ? "bg-white text-black shadow"
                  : "text-white/50 hover:text-white"
              }`}
            >
              {t === "upload" ? " Upload Photo" : " Use Camera"}
            </button>
          ))}
        </div>

        {/* Input area */}
        <div>
          {tab === "upload" ? (
            status === "idle" ? (
              <div
                onDrop={handleDrop}
                onDragOver={(e) => e.preventDefault()}
                onClick={() => fileInputRef.current?.click()}
                className="border-2 border-dashed border-white/15 rounded-2xl p-12 flex flex-col items-center gap-3 cursor-pointer hover:border-white/30 hover:bg-white/[0.03] transition-all"
              >
                <div className="text-4xl">&#128444;&#65039;</div>
                <p className="text-white/60 text-sm text-center">
                  Drag &amp; drop an image here, or{" "}
                  <span className="text-white underline underline-offset-2">click to browse</span>
                </p>
                <p className="text-white/30 text-xs">JPEG, PNG, WebP supported</p>
                <input
                  ref={fileInputRef}
                  type="file"
                  accept="image/jpeg,image/png,image/webp"
                  className="hidden"
                  onChange={handleFileChange}
                />
              </div>
            ) : (
              // eslint-disable-next-line @next/next/no-img-element
              <img
                src={imageSrc!}
                alt="Preview"
                className="w-full rounded-2xl border border-white/10 object-contain max-h-80"
              />
            )
          ) : (
            status === "idle" ? (
              <CameraCapture onCapture={handleCameraCapture} />
            ) : (
              // eslint-disable-next-line @next/next/no-img-element
              <img
                src={imageSrc!}
                alt="Captured"
                className="w-full rounded-2xl border border-white/10 object-contain max-h-80"
              />
            )
          )}
        </div>

        {/* Action buttons */}
        {imageSrc && status !== "analyzing" && status !== "idle" && (
          <div className="flex gap-3">
            <button
              onClick={reset}
              className="flex-1 py-3 rounded-xl border border-white/15 text-white/60 text-sm font-medium hover:bg-white/5 hover:text-white transition-all"
            >
               Start Over
            </button>
            <button
              onClick={analyze}
              className="flex-1 py-3 rounded-xl bg-white text-black text-sm font-semibold hover:bg-white/90 active:scale-95 transition-all"
            >
              Analyze
            </button>
          </div>
        )}

        {/* Loading */}
        {status === "analyzing" && (
          <div className="flex flex-col items-center gap-3 py-6">
            <div className="w-8 h-8 border-2 border-white/20 border-t-white rounded-full animate-spin" />
            <p className="text-white/50 text-sm">Detecting faces &amp; predicting�</p>
          </div>
        )}

        {/* Error */}
        {status === "error" && (
          <div className="bg-red-500/10 border border-red-500/30 rounded-xl p-4 text-sm text-red-400">
            ? {errorMsg}
          </div>
        )}

        {/* Results */}
        {status === "done" && result && (
          <div className="space-y-5">
            {result.success && result.faces.length > 0 ? (
              <>
                <FaceOverlayCanvas imageSrc={imageSrc!} faces={result.faces} />
                <div className="space-y-3">
                  <h2 className="text-sm font-semibold text-white/40 uppercase tracking-wider">
                    {result.faces.length} face{result.faces.length > 1 ? "s" : ""} detected
                  </h2>
                  {result.faces.map((face, i) => (
                    <FaceResultCard key={i} face={face} index={i} />
                  ))}
                </div>
              </>
            ) : (
              <div className="bg-yellow-500/10 border border-yellow-500/30 rounded-xl p-4 text-sm text-yellow-400">
                ?? {result.error ?? "No faces detected in the image."}
              </div>
            )}
          </div>
        )}

        <p className="text-center text-xs text-white/20 pt-4">
          For research &amp; educational use only MobileNetV2 + UTKFace
        </p>
      </div>
    </main>
  );
}
