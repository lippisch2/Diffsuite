import type {
  SessionMeta,
  ModelsResponse,
  JobStatus,
  FitResponse,
  ExtrapolateResponse,
  RoiResponse,
  RoiCalcModel,
  RoiCalcResponse,
} from "../types";

declare global {
  interface Window {
    diffsuite: {
      openFileDialog: () => Promise<string | null>;
      openExternalViewer: (path: string) => Promise<{ ok: boolean; error?: string }>;
      apiBase: string;
    };
  }
}

const BASE = () =>
  typeof window !== "undefined" && window.diffsuite
    ? window.diffsuite.apiBase
    : "http://127.0.0.1:18923";

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${BASE()}${path}`, {
    ...init,
    headers: {
      "Content-Type": "application/json",
      ...(init?.headers || {}),
    },
  });
  if (!res.ok) {
    const body = await res.text();
    throw new Error(`API error ${res.status}: ${body}`);
  }
  return res.json() as Promise<T>;
}

export const api = {
  async loadByPath(inputPath: string, bvals?: string, denoise?: boolean): Promise<SessionMeta> {
    return request<SessionMeta>("/api/load", {
      method: "POST",
      body: JSON.stringify({ path: inputPath, bvals, denoise }),
    });
  },

  async loadByFile(file: File, bvals?: string, denoise?: boolean): Promise<SessionMeta> {
    const form = new FormData();
    form.append("file", file);
    if (bvals) form.append("bvals", bvals);
    if (denoise) form.append("denoise", "true");
    const res = await fetch(`${BASE()}/api/load/upload`, {
      method: "POST",
      body: form,
    });
    if (!res.ok) throw new Error(`Upload failed: ${res.statusText}`);
    return res.json() as Promise<SessionMeta>;
  },

  async getModels(bvals: number[]): Promise<ModelsResponse> {
    return request<ModelsResponse>("/api/models", {
      method: "POST",
      body: JSON.stringify({ bvals }),
    });
  },

  async startFit(sessionId: string, model: string, workers: number, selectedBvals?: number[], options?: Record<string, unknown>): Promise<FitResponse> {
    return request<FitResponse>("/api/fit", {
      method: "POST",
      body: JSON.stringify({ session_id: sessionId, model, workers, selected_bvals: selectedBvals, options }),
    });
  },

  async getJobStatus(jobId: string): Promise<JobStatus> {
    return request<JobStatus>(`/api/fit/${jobId}`);
  },

  async extrapolate(
    sessionId: string,
    modelKey: string,
    bTarget: number
  ): Promise<ExtrapolateResponse> {
    return request<ExtrapolateResponse>("/api/extrapolate", {
      method: "POST",
      body: JSON.stringify({
        session_id: sessionId,
        model_key: modelKey,
        b_target: bTarget,
      }),
    });
  },

  niftiUrl(absPath: string): string {
    return `${BASE()}/api/nifti?path=${encodeURIComponent(absPath)}`;
  },

  referenceUrl(sessionId: string): string {
    return `${BASE()}/api/reference/${sessionId}`;
  },

  roiMaskUrl(sessionId: string): string {
    return `${BASE()}/api/roi/mask/${sessionId}`;
  },

  async setRoi(
    sessionId: string,
    roi: { slice: number; cx: number; cy: number; a: number; b: number; mode: string }
  ): Promise<RoiResponse> {
    return request<RoiResponse>("/api/roi", {
      method: "POST",
      body: JSON.stringify({ session_id: sessionId, ...roi }),
    });
  },

  async clearRoi(sessionId: string): Promise<{ ok: boolean }> {
    return request<{ ok: boolean }>("/api/roi/clear", {
      method: "POST",
      body: JSON.stringify({ session_id: sessionId }),
    });
  },

  async getRoiCalcModels(): Promise<{ models: RoiCalcModel[] }> {
    return request<{ models: RoiCalcModel[] }>("/api/roi-calc/models");
  },

  async calcRoi(
    modelKey: string,
    bvals: number[],
    signals: number[]
  ): Promise<RoiCalcResponse> {
    return request<RoiCalcResponse>("/api/roi-calc", {
      method: "POST",
      body: JSON.stringify({ model_key: modelKey, bvals, signals }),
    });
  },
};

export {};
