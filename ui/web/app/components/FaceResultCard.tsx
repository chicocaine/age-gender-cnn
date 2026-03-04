import { FaceResult } from "../types";

interface Props {
  face: FaceResult;
  index: number;
}

export default function FaceResultCard({ face, index }: Props) {
  const isFemale = face.gender === "Female";
  const genderConf = isFemale ? face.gender_prob : 1 - face.gender_prob;
  const accentColor = isFemale ? "from-pink-500 to-rose-400" : "from-blue-500 to-sky-400";
  const badgeColor = isFemale ? "bg-pink-500/20 text-pink-300 border-pink-500/30" : "bg-blue-500/20 text-blue-300 border-blue-500/30";

  return (
    <div className="bg-white/5 border border-white/10 rounded-2xl p-4 flex gap-4 items-start">
      {/* Cropped face */}
      <div className="shrink-0">
        {/* eslint-disable-next-line @next/next/no-img-element */}
        <img
          src={`data:image/jpeg;base64,${face.face_b64}`}
          alt={`Face ${index + 1}`}
          className="w-20 h-20 rounded-xl object-cover border border-white/10"
        />
      </div>

      {/* Predictions */}
      <div className="flex-1 min-w-0 space-y-3">
        <div className="flex items-center gap-2">
          <span className="text-xs text-white/40 font-medium uppercase tracking-wider">
            Face {index + 1}
          </span>
          <span className={`text-xs px-2 py-0.5 rounded-full border font-medium ${badgeColor}`}>
            {face.gender}
          </span>
        </div>

        {/* Age */}
        <div>
          <div className="flex justify-between text-sm mb-1">
            <span className="text-white/60">Predicted Age</span>
            <span className="text-white font-semibold">~{Math.round(face.age)} yrs</span>
          </div>
          {/* Age bar — maps 0–90 → 0–100% */}
          <div className="h-1.5 bg-white/10 rounded-full overflow-hidden">
            <div
              className={`h-full rounded-full bg-gradient-to-r ${accentColor}`}
              style={{ width: `${Math.min(100, (face.age / 90) * 100)}%` }}
            />
          </div>
        </div>

        {/* Gender confidence */}
        <div>
          <div className="flex justify-between text-sm mb-1">
            <span className="text-white/60">Gender Confidence</span>
            <span className="text-white font-semibold">{(genderConf * 100).toFixed(1)}%</span>
          </div>
          <div className="h-1.5 bg-white/10 rounded-full overflow-hidden">
            <div
              className={`h-full rounded-full bg-gradient-to-r ${accentColor}`}
              style={{ width: `${genderConf * 100}%` }}
            />
          </div>
        </div>
      </div>
    </div>
  );
}
