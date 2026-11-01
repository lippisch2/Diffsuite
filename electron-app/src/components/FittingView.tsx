import { useState, useEffect } from "react";
import { api } from "../api/client";
import type { JobStatus, RoiState } from "../types";
import { NiftiViewer } from "./NiftiViewer";

interface Props {
  sessionId: string;
  modelKey: string;
  forcedReason?: string | null;
  uniqBvals?: number[];
  onComplete: (job: JobStatus) => void;
}

export function FittingView({ sessionId, modelKey, forcedReason, uniqBvals, onComplete }: Props) {
  const [workers, setWorkers] = useState(1);
  const [registerDdvd, setRegisterDdvd] = useState(false);
  const [resampleDdvd, setResampleDdvd] = useState(true);
  const [job, setJob] = useState<JobStatus | null>(null);
  const [running, setRunning] = useState(false);
  const [selectedBvals, setSelectedBvals] = useState<Set<number>>(
    () => new Set(uniqBvals ?? [])
  );

  // ROI state
  const [useRoi, setUseRoi] = useState(false);
  const [roiMode, setRoiMode] = useState<"extrude" | "single">("extrude");
  const [roi, setRoi] = useState<RoiState | null>(null);
  const [roiApplied, setRoiApplied] = useState(false);
  const [roiMsg, setRoiMsg] = useState<string | null>(null);

  async function start() {
    setRunning(true);
    try {
      const bvalsArg = modelKey === "adc" && uniqBvals && uniqBvals.length > selectedBvals.size
        ? Array.from(selectedBvals)
        : undefined;
      const { job_id } = await api.startFit(
        sessionId,
        modelKey,
        workers,
        bvalsArg,
        modelKey === "ddvd" ? { register: registerDdvd, resample: resampleDdvd } : undefined
      );
      poll(job_id);
    } catch (e: any) {
      setRunning(false);
      alert(`Fit failed: ${e.message}`);
    }
  }

  function poll(jobId: string) {
    const interval = setInterval(async () => {
      try {
        const status = await api.getJobStatus(jobId);
        setJob(status);
        if (status.status === "done" || status.status === "error") {
          clearInterval(interval);
          setRunning(false);
          if (status.status === "done") onComplete(status);
        }
      } catch {
        clearInterval(interval);
        setRunning(false);
      }
    }, 500);
  }

  function handleRoiDrawn(r: { slice: number; cx: number; cy: number; a: number; b: number }) {
    setRoi({ ...r, mode: roiMode });
    setRoiApplied(false);
    setRoiMsg(null);
  }

  async function applyRoi() {
    if (!roi) return;
    try {
      const resp = await api.setRoi(sessionId, { ...roi, mode: roiMode });
      setRoiApplied(true);
      setRoiMsg(`ROI set: ${resp.voxels} voxels (${resp.mode}, slice ${resp.slice + 1})`);
    } catch (e: any) {
      alert(`ROI failed: ${e.message}`);
    }
  }

  async function clearRoi() {
    try {
      await api.clearRoi(sessionId);
    } catch {
      /* ignore */
    }
    setRoi(null);
    setRoiApplied(false);
    setRoiMsg(null);
  }

  return (
    <div className="card">
      <h2>4. Run Fitting</h2>
      <p className="muted">
        Model: <code>{modelKey}</code>
      </p>

      {forcedReason && (
        <div className="warning-banner" role="alert">
          <strong>Forced fit.</strong> Model requirements not met ({forcedReason}).
          Results may be unreliable, degenerate, or blank where the model is
          under-determined.
        </div>
      )}

      <div className="row">
        <label>Workers (parallel processes):</label>
        <input
          type="range"
          min={1}
          max={8}
          value={workers}
          onChange={(e) => setWorkers(parseInt(e.target.value))}
        />
        <span>{workers}</span>
      </div>

      {modelKey === "ddvd" && (
        <div style={{ marginTop: 12 }}>
          <div className="row">
            <label title="Aligned each low-b volume to b0 (SimpleITK rigid registration) before differencing, to remove motion artifacts between the two acquisitions.">
              <input
                type="checkbox"
                checked={registerDdvd}
                onChange={(e) => setRegisterDdvd(e.target.checked)}
              />{" "}
              Register b0 ↔ low-b (motion correction)
            </label>
          </div>
          {registerDdvd && (
            <div className="row" style={{ marginTop: 6, paddingLeft: 24 }}>
              <label title="When off, each registered low-b volume is kept on its native grid instead of being resampled onto the b0 grid.">
                <input
                  type="checkbox"
                  checked={resampleDdvd}
                  onChange={(e) => setResampleDdvd(e.target.checked)}
                />{" "}
                Resample onto b0 grid
              </label>
            </div>
          )}
        </div>
      )}

      {modelKey === "adc" && uniqBvals && uniqBvals.length > 0 && (
        <div style={{ marginTop: 12 }}>
          <p className="muted" style={{ marginBottom: 6 }}>
            b-values used for ADC fitting (click to toggle):
          </p>
          <div className="row" style={{ gap: 6, flexWrap: "wrap" }}>
            {uniqBvals.map((b) => (
              <button
                key={b}
                onClick={() => {
                  setSelectedBvals((prev) => {
                    const next = new Set(prev);
                    if (next.has(b)) next.delete(b);
                    else next.add(b);
                    return next;
                  });
                }}
                style={{
                  padding: "4px 12px",
                  borderRadius: 12,
                  border: "1px solid var(--panel-3)",
                  background: selectedBvals.has(b) ? "var(--accent)" : "var(--panel)",
                  color: selectedBvals.has(b) ? "#fff" : "var(--text)",
                  cursor: "pointer",
                  fontSize: 13,
                }}
              >
                b={b}
              </button>
            ))}
          </div>
          {selectedBvals.size < 2 && (
            <p className="muted" style={{ color: "var(--red)", marginTop: 4 }}>
              Select at least 2 b-values.
            </p>
          )}
        </div>
      )}

      <div className="row" style={{ marginTop: 12 }}>
        <label>
          <input
            type="checkbox"
            checked={useRoi}
            onChange={(e) => {
              setUseRoi(e.target.checked);
              if (!e.target.checked) clearRoi();
            }}
          />{" "}
          Limit to ROI
        </label>
      </div>

      {useRoi && (
        <div style={{ marginTop: 12, borderTop: "1px solid var(--panel-2)", paddingTop: 12 }}>
          <p className="muted">
            Drag on the b0 reference to draw an ellipse, then drag its body to
            move or the yellow handle to resize. Click <strong>Apply ROI</strong> when satisfied.
          </p>
          <div className="row" style={{ marginBottom: 8 }}>
            <label>
              <input
                type="checkbox"
                checked={roiMode === "single"}
                onChange={(e) => setRoiMode(e.target.checked ? "single" : "extrude")}
              />{" "}
              Single slice only (else extrude across all slices)
            </label>
          </div>

          <NiftiViewer
            niftiPath={api.referenceUrl(sessionId)}
            title="b0 reference — draw ROI"
            onRoiDrawn={handleRoiDrawn}
          />

          <div className="row" style={{ marginTop: 8 }}>
            <button onClick={applyRoi} disabled={!roi}>
              Apply ROI
            </button>
            <button onClick={clearRoi} style={{ background: "var(--panel-2)" }}>
              Clear ROI
            </button>
            {roiApplied && roiMsg && (
              <span className="muted" style={{ color: "var(--green)" }}>
                {roiMsg}
              </span>
            )}
          </div>
        </div>
      )}

      <button onClick={start} disabled={running} style={{ marginTop: 12 }}>
        {running ? "Running…" : "Run Fit"}
      </button>

      {job && (
        <>
          <div className="progress">
            <div
              className="progress-bar"
              style={{ width: `${job.progress_pct}%` }}
            />
          </div>
          <p className="muted">
            Slice {job.current_slice} / {job.total_slices} —{" "}
            {job.progress_pct.toFixed(0)}%
          </p>
          {job.roi_voxels != null && (
            <p className="muted" style={{ color: "var(--green)" }}>
              Fitted on ROI: {job.roi_voxels} voxels
            </p>
          )}
          {job.status === "error" && (
            <p style={{ color: "var(--red)" }}>Error: {job.error}</p>
          )}
        </>
      )}
    </div>
  );
}
