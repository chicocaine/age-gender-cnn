"use client";

import { useCallback, useRef } from "react";
import Webcam from "react-webcam";

interface Props {
  onCapture: (dataUrl: string) => void;
}

export default function CameraCapture({ onCapture }: Props) {
  const webcamRef = useRef<Webcam>(null);

  const capture = useCallback(() => {
    const dataUrl = webcamRef.current?.getScreenshot();
    if (dataUrl) onCapture(dataUrl);
  }, [onCapture]);

  return (
    <div className="flex flex-col items-center gap-4">
      <div className="relative rounded-2xl overflow-hidden border border-white/10 w-full max-w-md aspect-video bg-black">
        <Webcam
          ref={webcamRef}
          audio={false}
          screenshotFormat="image/jpeg"
          videoConstraints={{ facingMode: "user", aspectRatio: 16 / 9 }}
          className="w-full h-full object-cover"
        />
        {/* Corner guides */}
        <div className="absolute inset-0 pointer-events-none">
          {["top-3 left-3", "top-3 right-3", "bottom-3 left-3", "bottom-3 right-3"].map(
            (pos, i) => (
              <div key={i} className={`absolute ${pos} w-6 h-6 border-white/40 ${
                i === 0 ? "border-t-2 border-l-2 rounded-tl" :
                i === 1 ? "border-t-2 border-r-2 rounded-tr" :
                i === 2 ? "border-b-2 border-l-2 rounded-bl" :
                          "border-b-2 border-r-2 rounded-br"
              }`} />
            )
          )}
        </div>
      </div>

      <button
        onClick={capture}
        className="flex items-center gap-2 px-6 py-3 rounded-xl bg-white text-black font-semibold text-sm hover:bg-white/90 active:scale-95 transition-all"
      >
        <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <circle cx="12" cy="12" r="3" strokeWidth="2" />
          <path strokeWidth="2" d="M3 9a2 2 0 012-2h.93a2 2 0 001.664-.89l.812-1.22A2 2 0 0110.07 4h3.86a2 2 0 011.664.89l.812 1.22A2 2 0 0018.07 7H19a2 2 0 012 2v9a2 2 0 01-2 2H5a2 2 0 01-2-2V9z" />
        </svg>
        Capture Photo
      </button>
    </div>
  );
}
