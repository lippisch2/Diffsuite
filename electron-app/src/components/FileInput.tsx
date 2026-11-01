import { useState } from "react";
import { api } from "../api/client";
import type { SessionMeta } from "../types";
import { RoiCalcPanel } from "./RoiCalcPanel";
import { NiftiViewer } from "./NiftiViewer";

interface Props {
  onLoaded: (meta: SessionMeta) => void;
}

export function FileInput({ onLoaded }: Props) {
  const [loading, setLoading] = useState(false);
  const [denoise, setDenoise] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [dragOver, setDragOver] = useState(false);
  // A pre-existing fitted map opened directly into the analysis viewer.
  const [mapPath, setMapPath] = useState<string | null>(null);
  const [mapDragOver, setMapDragOver] = useState(false);
  const [mapError, setMapError] = useState<string | null>(null);

  async function handlePath(p: string) {
    setLoading(true);
    setError(null);
    try {
      const meta = await api.loadByPath(p, undefined, denoise);
      onLoaded(meta);
    } catch (e: any) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  }

  async function handleFile(f: File) {
    setLoading(true);
    setError(null);
    try {
      const meta = await api.loadByFile(f, undefined, denoise);
      onLoaded(meta);
    } catch (e: any) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  }

  function handleMapPath(p: string) {
    setMapPath(p);
    setMapError(null);
  }

  function handleMapFile(f: File) {
    const name = f.name.toLowerCase();
    if (!name.endsWith(".nii") && !name.endsWith(".nii.gz") && !name.endsWith(".gz")) {
      setMapError("Please open a NIfTI map (.nii / .nii.gz).");
      return;
    }
    // In Electron, dragged files carry a non-standard `path` with the absolute
    // filesystem path, which /api/nifti serves for the viewer.
    const fp = (f as File & { path?: string }).path;
    if (fp) {
      handleMapPath(fp);
    } else {
      setMapError("Could not resolve the file path — use the browse box instead.");
    }
  }

  return (
    <>
    <div className="card">
      <h2>1. Input Data</h2>
      <p className="muted">
        Load a 4D DWI NIfTI volume (.nii / .nii.gz) or a DICOM folder/file.
      </p>

      <div
        className="dropzone"
        style={dragOver ? { borderColor: "var(--accent)" } : undefined}
        onClick={async () => {
          const p = await window.diffsuite.openFileDialog();
          if (p) handlePath(p);
        }}
        onDragOver={(e) => {
          e.preventDefault();
          setDragOver(true);
        }}
        onDragLeave={() => setDragOver(false)}
        onDrop={(e) => {
          e.preventDefault();
          setDragOver(false);
          const f = e.dataTransfer.files[0];
          if (f) handleFile(f);
        }}
      >
        {loading ? (
          <p>{denoise ? "Preprocessing with QSIprep\u2026" : "Loading\u2026"}</p>
        ) : (
          <p>
            Click to browse or drag &amp; drop a file
            <br />
            <span className="muted">(NIfTI or DICOM)</span>
          </p>
        )}
      </div>

      <label className="row" style={{ marginTop: 12, gap: 8, alignItems: "center" }}>
        <input
          type="checkbox"
          checked={denoise}
          onChange={(e) => setDenoise(e.target.checked)}
          disabled={loading}
        />
        <span>Denoise &amp; motion correct (QSIprep)</span>
      </label>
      <p className="muted" style={{ marginTop: 4, marginLeft: 24 }}>
        MP-PCA denoising + eddy motion correction via container (podman).
        First run will pull the QSIprep image (~5 GB).
      </p>

      {error && (
        <p style={{ color: "var(--red)", marginTop: 12 }}>Error: {error}</p>
      )}
    </div>

    <div className="card" style={{ marginTop: 16 }}>
      <h2>Fitted map</h2>
      <p className="muted">
        Open an existing fitted map (.nii / .nii.gz) to inspect it with the
        viewer and run a histogram study — no (re)fit needed.
      </p>
      <div
        className="dropzone"
        style={mapDragOver ? { borderColor: "var(--accent)" } : undefined}
        onClick={async () => {
          const p = await window.diffsuite.openFileDialog();
          if (p) handleMapPath(p);
        }}
        onDragOver={(e) => {
          e.preventDefault();
          setMapDragOver(true);
        }}
        onDragLeave={() => setMapDragOver(false)}
        onDrop={(e) => {
          e.preventDefault();
          setMapDragOver(false);
          const f = e.dataTransfer.files[0];
          if (f) handleMapFile(f);
        }}
      >
        <p>
          Click to browse or drag &amp; drop a fitted map
          <br />
          <span className="muted">(e.g. dwi_dki2d_d.nii.gz)</span>
        </p>
      </div>
      {mapError && <p style={{ color: "var(--red)", marginTop: 12 }}>{mapError}</p>}
      {mapPath && (
        <div style={{ marginTop: 16 }}>
          <p className="muted" style={{ fontSize: 12, wordBreak: "break-all" }}>
            {mapPath}
          </p>
          <button
            className="secondary"
            style={{ marginBottom: 12 }}
            onClick={() => setMapPath(null)}
          >
            Close map
          </button>
          <NiftiViewer niftiPath={mapPath} title="Fitted map" enableHistogramSelection />
        </div>
      )}
    </div>
    <RoiCalcPanel />
    </>
  );
}
