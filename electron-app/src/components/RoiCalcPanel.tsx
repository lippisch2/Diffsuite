import { useEffect, useState } from "react";
import { api } from "../api/client";
import type { RoiCalcModel, RoiCalcResponse } from "../types";

interface BRow {
  id: number;
  b: string;
  signal: string;
}

let rowId = 0;

function parseNums(s: string): number[] {
  return s
    .split(/[\s,;]+/)
    .map((x) => x.trim())
    .filter(Boolean)
    .map((x) => parseFloat(x))
    .filter((n) => Number.isFinite(n));
}

// Paste-accepting helper: fills signals starting at index `from` with pasted numbers.
// If the pasted list is longer than the remaining rows, new rows are appended.
function patchSignals(rows: BRow[], from: number, nums: number[]): BRow[] {
  const copy = rows.slice();
  nums.forEach((v, k) => {
    const idx = from + k;
    if (idx < copy.length) copy[idx] = { ...copy[idx], signal: String(v) };
    else copy.push({ id: ++rowId, b: "", signal: String(v) });
  });
  return copy;
}

export function RoiCalcPanel() {
  const [models, setModels] = useState<RoiCalcModel[]>([]);
  const [modelKey, setModelKey] = useState<string>("adc");
  const [rows, setRows] = useState<BRow[]>([]);
  const [result, setResult] = useState<RoiCalcResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  useEffect(() => {
    api.getRoiCalcModels().then((r) => {
      setModels(r.models);
      if (r.models.length) setModelKey(r.models[0].key);
    });
  }, []);

  const model = models.find((m) => m.key === modelKey);

  function updateRow(id: number, patch: Partial<BRow>) {
    setRows((rs) => rs.map((r) => (r.id === id ? { ...r, ...patch } : r)));
  }

  function addRow() {
    setRows((rs) => [...rs, { id: ++rowId, b: "", signal: "" }]);
  }

  function removeRow(id: number) {
    setRows((rs) => rs.filter((r) => r.id !== id));
  }

  async function calculate() {
    const bvals = rows.map((r) => parseFloat(r.b)).filter((n) => Number.isFinite(n));
    const signals = rows.map((r) => parseFloat(r.signal)).filter((n) => Number.isFinite(n));
    if (bvals.length < 2 || signals.length < 2) {
      setError("Enter at least 2 (b, signal) rows.");
      setResult(null);
      return;
    }
    setBusy(true);
    setError(null);
    try {
      const res = await api.calcRoi(modelKey, bvals, signals);
      setResult(res);
    } catch (e: any) {
      setError(e.message);
      setResult(null);
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="card" style={{ marginTop: 16 }}>
      <h2>Input ROI values — quick semi-auto fit</h2>
      <p className="muted" style={{ marginBottom: 12 }}>
        Made a region-of-interest (ROI) measurement on the b-value maps in another
        program? Enter each b-value and the ROI mean signal below, pick a model,
        and calculate the fitted parameters — no DWI file needed.
      </p>

      <label className="row" style={{ gap: 10, alignItems: "center" }}>
        <span>Model:</span>
        <select
          value={modelKey}
          onChange={(e) => {
            setModelKey(e.target.value);
            setResult(null);
          }}
          style={{ minWidth: 220 }}
        >
          {models.map((m) => (
            <option key={m.key} value={m.key}>
              {m.label}
            </option>
          ))}
        </select>
      </label>

      {model && (
        <div className="roi-model-hint">
          <div className="formula-cell">
            <code>{model.formula}</code>
            <div className="muted">Requires: {model.requirement}</div>
            <div className="muted">Outputs: {model.params.join(", ")}</div>
          </div>
        </div>
      )}

      <div className="roi-rows">
        <div className="roi-row roi-row-head">
          <span>#</span>
          <span>b-value (s/mm²)</span>
          <span>ROI mean signal</span>
          <span />
        </div>
        {rows.map((r, i) => (
          <div className="roi-row" key={r.id}>
            <span className="muted">{i + 1}</span>
            <input
              type="number"
              step="any"
              placeholder="e.g. 0"
              value={r.b}
              onChange={(e) => updateRow(r.id, { b: e.target.value })}
            />
            <input
              type="number"
              step="any"
              placeholder="e.g. 800"
              value={r.signal}
              onChange={(e) => updateRow(r.id, { signal: e.target.value })}
              onPaste={(e) => {
                const nums = parseNums(e.clipboardData.getData("text"));
                if (nums.length) {
                  e.preventDefault();
                  setRows((rs) => patchSignals(rs, i, nums));
                }
              }}
            />
            <button className="secondary roi-remove" onClick={() => removeRow(r.id)}>
              ✕
            </button>
          </div>
        ))}
        {rows.length === 0 && (
          <p className="muted" style={{ fontStyle: "italic" }}>
            No rows yet — add a b-value below.
          </p>
        )}
      </div>

      <div className="row" style={{ marginTop: 12 }}>
        <button className="secondary" onClick={addRow}>
          + Add b-value
        </button>
        <span className="muted">
          Tip: paste a column of signals (or "b signal" pairs) into the last cell.
        </span>
      </div>

      <button onClick={calculate} disabled={busy} style={{ marginTop: 8 }}>
        {busy ? "Calculating…" : "Calculate"}
      </button>

      {error && (
        <p style={{ color: "var(--red)", marginTop: 12 }}>Error: {error}</p>
      )}

      {result && (
        <div className="roi-result" style={{ marginTop: 16 }}>
          <strong>{result.label}</strong>
          <span className="muted" style={{ display: "block", marginBottom: 8 }}>
            {result.formula}
          </span>
          <table>
            <thead>
              <tr>
                <th>Parameter</th>
                <th>Value</th>
                <th>Unit</th>
              </tr>
            </thead>
            <tbody>
              {result.params.map((p) => (
                <tr key={p.name}>
                  <td>{p.name}</td>
                  <td>{p.value}</td>
                  <td className="muted">{p.unit}</td>
                </tr>
              ))}
            </tbody>
          </table>
          <span className="muted">Fit residual (RMSE): {result.residual}</span>
        </div>
      )}
    </div>
  );
}
