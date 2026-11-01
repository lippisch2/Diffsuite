import { useEffect, useState } from "react";
import { api } from "../api/client";
import type { ModelsResponse, ModelInfo, RejectedModel, SessionMeta } from "../types";

interface Props {
  models: ModelsResponse;
  meta: SessionMeta;
  selected: string | null;
  onSelect: (key: string, forcedReason?: string) => void;
  onBvalsChange?: (next: ModelsResponse, bvals: number[]) => void;
}

export function ModelSelector({ models, meta, selected, onSelect, onBvalsChange }: Props) {
  // Show the manual b-value override only when no b-values were detected
  // (e.g. no .bval sidecar / DICOM without b-info).
  const showOverride = meta.bvals.length === 0;
  const [bvalText, setBvalText] = useState("");
  const [status, setStatus] = useState<string | null>(null);
  // Force-fit toggle: resets to OFF whenever a new session (file) is loaded.
  const [allowForce, setAllowForce] = useState(false);
  const [confirmModel, setConfirmModel] = useState<RejectedModel | null>(null);

  useEffect(() => {
    setAllowForce(false);
    setConfirmModel(null);
  }, [meta.session_id]);

  async function applyOverride() {
    const vals = bvalText
      .split(/\s+/)
      .map((x) => parseFloat(x))
      .filter((x) => !isNaN(x));
    if (vals.length === 0) {
      setStatus("Enter at least one b-value.");
      return;
    }
    try {
      const resp = await api.getModels(vals);
      setStatus(`Loaded ${vals.length} b-values, ${resp.applicable.length} models available`);
      onBvalsChange?.(resp, vals);
    } catch (e: any) {
      setStatus(`Error: ${e.message}`);
    }
  }

  function canForce(r: RejectedModel): boolean {
    return allowForce && !!r.key;
  }

  return (
    <div className="card">
      <h2>2. Select Model</h2>
      <p className="muted">
        Only models compatible with your b-value configuration are enabled.
      </p>

      <div className="row" style={{ marginBottom: 12 }}>
        <span className="muted">b-values from <code>{meta.stem}.nii.gz</code>:</span>
        <code>{meta.bvals.length ? meta.bvals.join(", ") : "none detected"}</code>
      </div>

      {showOverride && (
        <div className="row" style={{ marginBottom: 12, borderTop: "1px solid var(--panel-2)", paddingTop: 12 }}>
          <p className="muted">
            No b-values were detected. Enter them manually (space-separated),
            then click Apply.
          </p>
          <input
            type="text"
            value={bvalText}
            onChange={(e) => setBvalText(e.target.value)}
            style={{ width: "100%" }}
            placeholder="e.g. 0 50 800 1400"
          />
          <button onClick={applyOverride}>Apply</button>
          {status && (
            <span className="muted" style={{ marginLeft: 8 }}>{status}</span>
          )}
        </div>
      )}

      <label className="row force-toggle" title="Run models whose b-value requirements are not met">
        <input
          type="checkbox"
          checked={allowForce}
          onChange={(e) => {
            setAllowForce(e.target.checked);
            setConfirmModel(null);
          }}
        />
        <span>Allow forcing blocked models</span>
        <span className="muted">(advanced — results may be unreliable)</span>
      </label>

      <div className="model-list" style={{ marginTop: 12 }}>
        {(() => {
          // Group applicable models by family
          const groups = new Map<string, ModelInfo[]>();
          for (const m of models.applicable) {
            const g = m.group || "Other";
            if (!groups.has(g)) groups.set(g, []);
            groups.get(g)!.push(m);
          }
          // Group rejected models by family
          const rejGroups = new Map<string, RejectedModel[]>();
          for (const r of models.rejected) {
            const g = r.group || "Other";
            if (!rejGroups.has(g)) rejGroups.set(g, []);
            rejGroups.get(g)!.push(r);
          }
          // Render order: families present in applicable first, then remaining rejected families
          const familyOrder = ["ADC", "Kurtosis (DKI)", "IVIM", "DDVD", "DDC", "Experimental", "Extrapolation", "Other"];
          const allFamilies = [
            ...familyOrder.filter((f) => groups.has(f) || rejGroups.has(f)),
            ...Array.from(new Set([...groups.keys(), ...rejGroups.keys()])).filter(
              (f) => !familyOrder.includes(f)
            ),
          ];
          return allFamilies.map((family) => {
            const famModels = groups.get(family) || [];
            const famRej = rejGroups.get(family) || [];
            if (famModels.length === 0 && famRej.length === 0) return null;
            return (
              <div key={family} className="model-family">
                <div className="model-family-header">{family}</div>
                {famModels.map((m) => (
                  <div
                    key={m.key}
                    className={
                      "model-row" +
                      (selected === m.key ? " selected" : "")
                    }
                    onClick={() => onSelect(m.key)}
                  >
                    <div className="model-row-top">
                      <span className="model-row-num">{m.number}</span>
                      <span className="model-row-name">{m.label}</span>
                      <span className="model-row-badge">
                        <span className="badge ok">available</span>
                      </span>
                      <span className="model-row-outputs">{m.outputs}</span>
                    </div>
                    {(m.formula || m.principle) && (
                      <div className="model-row-formula">
                        {m.formula && <code>{m.formula}</code>}
                        {m.principle && <div className="principle">{m.principle}</div>}
                      </div>
                    )}
                  </div>
                ))}
                {famRej.map((r) =>
                  canForce(r) ? (
                    <div
                      key={r.label}
                      className={"model-row forced" + (selected === r.key ? " selected" : "")}
                      onClick={() => setConfirmModel(r)}
                    >
                      <div className="model-row-top">
                        <span className="model-row-num">—</span>
                        <span className="model-row-name">{r.label}</span>
                        <span className="model-row-badge">
                          <span className="badge warn">force</span>
                        </span>
                        <span className="model-row-outputs">{r.reason}</span>
                      </div>
                      {(r.formula || r.principle) && (
                        <div className="model-row-formula">
                          {r.formula && <code>{r.formula}</code>}
                          {r.principle && <div className="principle">{r.principle}</div>}
                        </div>
                      )}
                    </div>
                  ) : (
                    <div key={r.label} className="model-row disabled">
                      <div className="model-row-top">
                        <span className="model-row-num">—</span>
                        <span className="model-row-name">{r.label}</span>
                        <span className="model-row-badge">
                          <span className="badge warn">blocked</span>
                        </span>
                        <span className="model-row-outputs">{r.reason}</span>
                      </div>
                      {(r.formula || r.principle) && (
                        <div className="model-row-formula">
                          {r.formula && <code>{r.formula}</code>}
                          {r.principle && <div className="principle">{r.principle}</div>}
                        </div>
                      )}
                    </div>
                  )
                )}
              </div>
            );
          });
        })()}
      </div>

      {confirmModel && (
        <div className="force-confirm" role="alertdialog">
          <p>
            Force fit <strong>“{confirmModel.label}”</strong>? Requirements not met
            (<em>{confirmModel.reason}</em>). The fit will run, but results may be
            unreliable, degenerate, or blank where the model is under-determined.
          </p>
          <div className="row">
            <button
              className="danger"
              onClick={() => {
                const key = confirmModel.key!;
                setConfirmModel(null);
                onSelect(key, confirmModel.reason);
              }}
            >
              Force fit anyway
            </button>
            <button className="secondary" onClick={() => setConfirmModel(null)}>
              Cancel
            </button>
          </div>
        </div>
      )}
    </div>
  );
}
