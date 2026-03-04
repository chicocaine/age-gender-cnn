export interface FaceResult {
  age: number;
  gender: "Male" | "Female";
  gender_prob: number; // sigmoid output — probability of Female (0–1)
  bbox: [number, number, number, number]; // [x, y, w, h] in original image px
  face_b64: string; // base64-encoded JPEG of the 224×224 crop
}

export interface PredictResponse {
  success: boolean;
  faces: FaceResult[];
  error: string | null;
}
