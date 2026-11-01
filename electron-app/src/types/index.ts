export interface SessionMeta {
  session_id: string;
  nifti_path: string;
  source: string;
  shape: number[];
  bvals: number[];
  bvec_count: number;
  stem: string;
  out_prefix: string;
}

export interface ModelInfo {
  number: number;
  label: string;
  key: string;
  outputs: string;
  group: string;
  formula?: string | null;
  principle?: string | null;
}

export interface RejectedModel {
  label: string;
  reason: string;
  /** Registry key when the model can be force-fitted despite unmet requirements; null/undefined = never forceable. */
  key?: string | null;
  group?: string;
  formula?: string | null;
  principle?: string | null;
}

export interface ModelsResponse {
  applicable: ModelInfo[];
  rejected: RejectedModel[];
  uniq_b: number[];
}

export interface FitQuality {
  overall: "good" | "fair" | "poor" | "unknown";
  score: number;
  norm_base: number;
  frac_unexplained: number;
  threshold: number;
  residual: { mean: number; median: number; p95: number };
  worst_slices: { slice: number; mean_relative: number }[];
  notes: string[];
  roi?: { voxels: number; median: number; p95: number; frac_unexplained: number };
}

export interface JobStatus {
  status: "queued" | "running" | "done" | "error";
  progress_pct: number;
  current_slice: number;
  total_slices: number;
  error?: string;
  outputs?: Record<string, string>;
  roi_voxels?: number | null;
  quality?: FitQuality;
}

export interface FitResponse {
  job_id: string;
}

export interface ExtrapolateResponse {
  path: string;
  param: string;
}

export interface RoiState {
  slice: number;
  cx: number;
  cy: number;
  a: number;
  b: number;
  mode: "extrude" | "single";
}

export interface RoiResponse {
  ok: boolean;
  voxels: number;
  mode: string;
  slice: number;
}

export interface RoiCalcModel {
  key: string;
  label: string;
  formula: string;
  requirement: string;
  params: string[];
}

export interface RoiCalcParam {
  name: string;
  value: number;
  unit: string;
}

export interface RoiCalcResponse {
  model_key: string;
  label: string;
  formula: string;
  params: RoiCalcParam[];
  residual: number;
}
