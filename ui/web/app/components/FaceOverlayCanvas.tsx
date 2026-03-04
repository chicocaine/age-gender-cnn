"use client";

import { useCallback, useEffect, useRef } from "react";
import { FaceResult } from "../types";

interface Props {
  imageSrc: string; // data URL or object URL of the original image
  faces: FaceResult[];
}

export default function FaceOverlayCanvas({ imageSrc, faces }: Props) {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  const draw = useCallback(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const img = new window.Image();
    img.onload = () => {
      // Match canvas to natural image size (capped for display)
      const MAX = 560;
      const scale = Math.min(1, MAX / img.naturalWidth, MAX / img.naturalHeight);
      canvas.width = img.naturalWidth * scale;
      canvas.height = img.naturalHeight * scale;

      ctx.drawImage(img, 0, 0, canvas.width, canvas.height);

      faces.forEach((face, i) => {
        const [x, y, w, h] = face.bbox.map((v) => v * scale);
        const color = face.gender === "Female" ? "#f472b6" : "#60a5fa";

        // Box
        ctx.strokeStyle = color;
        ctx.lineWidth = 2.5;
        ctx.strokeRect(x, y, w, h);

        // Label background
        const label = `#${i + 1} ${face.gender}, ~${Math.round(face.age)}y`;
        ctx.font = "bold 13px sans-serif";
        const tw = ctx.measureText(label).width;
        ctx.fillStyle = color;
        ctx.fillRect(x, y - 22, tw + 10, 22);

        // Label text
        ctx.fillStyle = "#fff";
        ctx.fillText(label, x + 5, y - 6);
      });
    };
    img.src = imageSrc;
  }, [imageSrc, faces]);

  useEffect(() => {
    draw();
  }, [draw]);

  return (
    <canvas
      ref={canvasRef}
      className="rounded-xl border border-white/10 max-w-full mx-auto block"
    />
  );
}
