import { useState, useEffect } from "react";
import { api } from "../api/client";
import type { JobStatus, SessionMeta } from "../types";
import { NiftiViewer } from "./NiftiViewer";

interface Props {
  sessionId: string;
  meta: SessionMeta;
  job: JobStatus;
}

interface ExtrapResult {
  b: number;
  path: string;
}

// Short description of each output map, keyed by the param names emitted by the
// Python fit functions (see src/diffsuite/models/*_volume out_names).
const PARAM_LEGEND: Record<string, string> = {
  s0: "S0 — baseline signal (fit intercept at b=0); reflects proton density/T2 weighting of the source DWI.",
  adc: "ADC — apparent diffusion coefficient, slope of the monoexponential decay S(b)=S0·exp(−b·ADC). Higher = less restricted diffusion. Units depend on the b-value scale (mm²/s if b is in s/mm²).",
  eadc: "eADC — exponential apparent diffusion coefficient, fraction of signal remaining at the highest b-value: eADC=exp(−ADC·b_max). Dimensionless (0–1); contrast is opposite to ADC (higher = more restricted diffusion).",
  d: "D — tissue diffusion coefficient (slow IVIM component, or DKI Gaussian diffusion). Units follow the b-value scale (mm²/s if b in s/mm²).",
  dstar: "D* — pseudo-diffusion coefficient of the microcirculation (fast IVIM component). Noisy by nature; interpret with caution.",
  f: "f — perfusion fraction: proportion of signal from flowing blood in capillaries (dimensionless, 0–1).",
  k: "K — kurtosis excess: deviation from Gaussian diffusion. K=0 would be pure Gaussian; higher K = more restricted/non-Gaussian diffusion.",
  ddc: "DDC — distributed diffusion coefficient of the stretched-exponential model S(b)=S0·exp(−(b·DDC)^α). Comparable to an ADC.",
  alpha: "α — stretching exponent of the DDC model (0–1): intravoxel heterogeneity of diffusion. α=1 → monoexponential; lower α = more heterogeneous.",
  beta: "β — fractional order / spatial heterogeneity index (CTRW/FROC): 0<β≤1; β=1 → Gaussian monoexponential decay.",
  mu: "μ — spatial constant (FROC microstructural scale). Helps separate bulk diffusion from geometric/trap effects; kept as a free fit parameter.",
  skew: "Skew — asymmetry of the intravoxel diffusion-distribution spectrum (DKI fast+ extension).",
  fused_skew_k: "Fused map — skew-weighted combination highlighting regions where non-Gaussian effects dominate.",
  synthetic: "Synthetic image — model-predicted signal at the chosen target b-value.",
  residual: "Residual — root-mean-square error of the voxel fit; high values flag unreliable fits (motion, noise, wrong model).",
};

function qualityBadge(v: string) {
  const colors: Record<string, string> = {
    good: "var(--green)",
    fair: "var(--yellow)",
    poor: "var(--red)",
    unknown: "var(--text-dim)",
  };
  return (
    <span
      style={{
        color: colors[v] ?? "var(--text-dim)",
        fontWeight: 600,
        textTransform: "uppercase",
        fontSize: "0.7em",
        marginLeft: 6,
      }}
    >
      ({v})
    </span>
  );
}

export function ResultsPanel({ sessionId, meta, job }: Props) {
  const [viewerPath, setViewerPath] = useState<string | null>(null);
  const [viewerTitle, setViewerTitle] = useState<string>("");
  const [extrapModel, setExtrapModel] = useState<string>("");
  const [roiWindowing, setRoiWindowing] = useState(false);
  // Target b-values the user has toggled on for extrapolation.
  const [targets, setTargets] = useState<number[]>([]);
  const [customB, setCustomB] = useState<string>("");
  const [results, setResults] = useState<ExtrapResult[]>([]);
  const [busy, setBusy] = useState(false);

  const outputs = job.outputs ?? {};

  // Derive available model registry keys from the output file paths. Filenames use a
  // filesystem-safe prefix (e.g. "dki2d_fast_plus"), which must be mapped back to the
  // registry key (e.g. "dki2d_fast+") that the API expects for /api/extrapolate.
  // Longest match wins so "dki2d_fast_plus" is preferred over its substring "dki2d_fast".
  const FILE_PREFIX_TO_KEY: Record<string, string> = {
    dki2d_fast_plus: "dki2d_fast+",
  };
  const modelKeys = Array.from(
    new Set(
      Object.values(outputs).map((p) => {
        const base = p.split(/[\\/]/).pop() ?? "";
        const candidates = [
          "dki2d_fast_plus", "dki2d_fast", "dki2d", "ddc",
          "ivim_seg", "ivim_full", "dki_ivim", "adc", "eadc",
          "dki_ais_u", "dki_ais_c", "dki_ais_sc", "akc_adci", "ddvd", "ctrw", "froc", "extrap",
        ].filter((k) => base.includes(`_${k}_`));
        candidates.sort((a, b) => b.length - a.length);
        const filePrefix = candidates[0] ?? "";
        return FILE_PREFIX_TO_KEY[filePrefix] ?? filePrefix;
      })
    )
  ).filter(Boolean);

  // Pre-select the first available model key for extrapolation once outputs exist.
  useEffect(() => {
    if (!extrapModel && modelKeys.length > 0) setExtrapModel(modelKeys[0]);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [outputs]);

  // Suggested target b-values: the existing b-values plus common higher extras.
  const suggested = Array.from(
    new Set([...meta.bvals.map((b) => Math.round(b)), 3000])
  ).sort((a, b) => a - b);

  function toggleTarget(b: number) {
    setTargets((prev) =>
      prev.includes(b) ? prev.filter((x) => x !== b) : [...prev, b]
    );
  }

  function addCustom() {
    const v = parseFloat(customB);
    if (!isNaN(v) && v > 0 && !targets.includes(v)) {
      setTargets((prev) => [...prev, v].sort((a, b) => a - b));
      setCustomB("");
    }
  }

  function openInViewer(path: string, title: string) {
    setViewerPath(path);
    setViewerTitle(title);
  }

  async function runExtrapolate() {
    if (!extrapModel) {
      alert("Select a fitted model first");
      return;
    }
    if (targets.length === 0) {
      alert("Select at least one target b-value to extrapolate to.");
      return;
    }
    setBusy(true);
    const collected: ExtrapResult[] = [];
    try {
      for (const b of targets) {
        const resp = await api.extrapolate(sessionId, extrapModel, b);
        collected.push({ b, path: resp.path });
      }
      setResults(collected);
      if (collected.length > 0) {
        openInViewer(collected[0].path, `Synthetic b=${collected[0].b} (${extrapModel})`);
      }
    } catch (e: any) {
      alert(`Extrapolation failed: ${e.message}`);
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="card">
      <h2>4. Results</h2>
      <p className="muted">Output maps saved next to the input file.</p>

      <table>
        <thead>
          <tr>
            <th>Parameter Map</th>
            <th>File</th>
            <th></th>
          </tr>
        </thead>
        <tbody>
          {Object.entries(outputs).map(([param, path]) => (
            <tr key={param}>
              <td>{param}</td>
              <td className="muted" style={{ maxWidth: 300, overflow: "hidden", textOverflow: "ellipsis" }}>
                {path}
              </td>
              <td>
                <button onClick={() => openInViewer(path, param)}>View</button>
              </td>
            </tr>
          ))}
        </tbody>
      </table>

      {Object.keys(outputs).some((p) => PARAM_LEGEND[p]) && (
        <div className="map-legend">
          <h3>Map legend</h3>
          <ul>
            {Object.keys(outputs)
              .filter((p) => PARAM_LEGEND[p])
              .map((p) => (
                <li key={p}>
                  <code>{p}</code> — {PARAM_LEGEND[p]}
                </li>
              ))}
          </ul>
        </div>
      )}

      {job.quality && job.quality.overall !== "unknown" && (
        <div className={`qa-panel qa-${job.quality.overall}`} style={{ marginTop: 14 }}>
          <div className="row" style={{ alignItems: "center", justifyContent: "space-between" }}>
            <h3 style={{ margin: 0 }}>Quality assessment {qualityBadge(job.quality.overall)}</h3>
            <span className="muted">
              score {job.quality.score.toFixed(2)} / 1.00
            </span>
          </div>
          <div className="qa-stats row" style={{ gap: 18, flexWrap: "wrap", marginTop: 8 }}>
            <span>Median rel. residual: <code>{job.quality.residual.median.toFixed(3)}</code></span>
            <span>95th pct.: <code>{job.quality.residual.p95.toFixed(3)}</code></span>
            <span>
              Unexplained voxels:{" "}
              <code>{(job.quality.frac_unexplained * 100).toFixed(1)}%</code>
              <span className="muted"> (&gt;{job.quality.threshold})</span>
            </span>
            {job.quality.roi && (
              <span>
                ROI: <code>{job.quality.roi.voxels}</code> vox, med{" "}
                <code>{job.quality.roi.median.toFixed(3)}</code>, unexplained{" "}
                <code>{(job.quality.roi.frac_unexplained * 100).toFixed(1)}%</code>
              </span>
            )}
          </div>
          {job.quality.notes.length > 0 && (
            <ul className="qa-notes" style={{ margin: "8px 0 0", paddingLeft: 18 }}>
              {job.quality.notes.map((n, i) => (
                <li key={i}>{n}</li>
              ))}
            </ul>
          )}
          {job.quality.worst_slices.length > 0 && (
            <div className="row" style={{ marginTop: 8, gap: 6, flexWrap: "wrap", alignItems: "center" }}>
              <span className="muted">Worst slices:</span>
              {job.quality.worst_slices.map((w) => (
                <button
                  key={w.slice}
                  className="secondary"
                  onClick={() => openInViewer(outputs["residual"], `residual slice ${w.slice}`)}
                  title={`Mean relative residual ${w.mean_relative.toFixed(3)}`}
                >
                  {w.slice}
                </button>
              ))}
            </div>
          )}
        </div>
      )}

      <div className="row" style={{ marginTop: 16, alignItems: "flex-start", flexDirection: "column" }}>
        <div className="row" style={{ width: "100%" }}>
          <label>Extrapolate from:</label>
          <select value={extrapModel} onChange={(e) => setExtrapModel(e.target.value)}>
            <option value="">— select —</option>
            {modelKeys.map((m) => (
              <option key={m} value={m}>{m}</option>
            ))}
          </select>
        </div>

        <div style={{ marginTop: 8, width: "100%" }}>
          <label className="muted">Target b-value(s) to extrapolate to:</label>
          <div style={{ display: "flex", flexWrap: "wrap", gap: 6, marginTop: 6 }}>
            {suggested.map((b) => (
              <button
                key={b}
                onClick={() => toggleTarget(b)}
                style={{
                  background: targets.includes(b) ? "var(--accent)" : "var(--panel-2)",
                  color: targets.includes(b) ? "#fff" : "var(--text)",
                }}
              >
                {b}
              </button>
            ))}
          </div>
          <div className="row" style={{ marginTop: 8 }}>
            <input
              type="number"
              value={customB}
              onChange={(e) => setCustomB(e.target.value)}
              placeholder="custom b"
              style={{ width: 100 }}
            />
            <button onClick={addCustom}>Add</button>
            {targets.length > 0 && (
              <span className="muted" style={{ marginLeft: 8 }}>
                selected: {targets.join(", ")}
              </span>
            )}
          </div>
        </div>

        <button onClick={runExtrapolate} disabled={busy} style={{ marginTop: 10 }}>
          {busy ? "Extrapolating…" : "Extrapolate"}
        </button>
      </div>

      {results.length > 1 && (
        <div className="row" style={{ marginTop: 10, flexWrap: "wrap", gap: 6 }}>
          {results.map((r) => (
            <button key={r.b} onClick={() => openInViewer(r.path, `Synthetic b=${r.b} (${extrapModel})`)}>
              View b={r.b}
            </button>
          ))}
        </div>
      )}

      {viewerPath && (
        <div style={{ marginTop: 20 }}>
          <h3 style={{ marginBottom: 8 }}>{viewerTitle}</h3>
          <div className="row" style={{ marginBottom: 8 }}>
            <label>
              <input
                type="checkbox"
                checked={roiWindowing}
                onChange={(e) => setRoiWindowing(e.target.checked)}
              />{" "}
              Window colormap by ROI
            </label>
            {roiWindowing && (
              <span className="muted" style={{ marginLeft: 8 }}>
                (uses the ROI drawn during fitting)
              </span>
            )}
          </div>
          <NiftiViewer
            niftiPath={viewerPath}
            title={viewerTitle}
            roiPath={roiWindowing ? api.roiMaskUrl(sessionId) : undefined}
            roiWindowing={roiWindowing}
            enableHistogramSelection
            onOpenExternal={() => window.diffsuite.openExternalViewer(viewerPath)}
          />
        </div>
      )}
    </div>
  );
}
