import { useEffect, useRef, useState, useCallback } from "react";
import * as nifti from "nifti-reader-js";
import { api } from "../api/client";
import { runHistogramStudy, type HistogramStudy } from "../lib/histogram";

interface Props {
  /** Absolute path or full URL to NIfTI file (served by FastAPI) */
  niftiPath: string;
  /** Optional param name for display */
  title?: string;
  /** Optional callback to open externally */
  onOpenExternal?: () => void;
  /** When set, an interactive ellipse ROI is editable and reported via this callback */
  onRoiDrawn?: (roi: { slice: number; cx: number; cy: number; a: number; b: number }) => void;
  /** Optional URL/path to a binary ROI mask NIfTI. When provided and roiWindowing
   *  is enabled, the colormap window is computed from the ROI foreground only. */
  roiPath?: string;
  /** If true, compute the auto window from the ROI mask (passed via roiPath) instead of the whole volume. */
  roiWindowing?: boolean;
  /** When set, enables multi-slice voxel selection (lasso / paint brush) and a
   *  histogram study over the selected voxels. */
  enableHistogramSelection?: boolean;
}

type Colormap = "gray" | "turbo" | "jet";

// Precomputed colormaps (256 entries, [r,g,b])
const COLORMAPS: Record<Colormap, [number, number, number][]> = {
  gray: Array.from({ length: 256 }, (_, i) => [i, i, i] as [number, number, number]),
  turbo: (() => {
    const stops: [number, number, number][] = [
      [48, 18, 59], [62, 73, 207], [38, 130, 238], [30, 194, 214],
      [95, 226, 150], [180, 232, 75], [245, 210, 40], [250, 150, 30],
      [230, 85, 20], [200, 30, 20],
    ];
    const out: [number, number, number][] = [];
    for (let i = 0; i < 256; i++) {
      const t = (i / 255) * (stops.length - 1);
      const lo = Math.floor(t);
      const hi = Math.min(lo + 1, stops.length - 1);
      const f = t - lo;
      out.push([
        Math.round(stops[lo][0] + (stops[hi][0] - stops[lo][0]) * f),
        Math.round(stops[lo][1] + (stops[hi][1] - stops[lo][1]) * f),
        Math.round(stops[lo][2] + (stops[hi][2] - stops[lo][2]) * f),
      ]);
    }
    return out;
  })(),
  jet: (() => {
    const stops: [number, number, number][] = [
      [0, 0, 128], [0, 0, 255], [0, 128, 255], [0, 255, 255],
      [128, 255, 128], [255, 255, 0], [255, 128, 0], [255, 0, 0], [128, 0, 0],
    ];
    const out: [number, number, number][] = [];
    for (let i = 0; i < 256; i++) {
      const t = (i / 255) * (stops.length - 1);
      const lo = Math.floor(t);
      const hi = Math.min(lo + 1, stops.length - 1);
      const f = t - lo;
      out.push([
        Math.round(stops[lo][0] + (stops[hi][0] - stops[lo][0]) * f),
        Math.round(stops[lo][1] + (stops[hi][1] - stops[lo][1]) * f),
        Math.round(stops[lo][2] + (stops[hi][2] - stops[lo][2]) * f),
      ]);
    }
    return out;
  })(),
};

interface Ellipse { cx: number; cy: number; a: number; b: number; }
interface RoiStats { n: number; mean: number; std: number; min: number; max: number; }

type DragMode = "draw" | "move" | "resize" | null;
type SelectionTool = "lasso" | "brush" | null;

export function NiftiViewer({ niftiPath, title, onOpenExternal, onRoiDrawn, roiPath, roiWindowing, enableHistogramSelection }: Props) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const wrapRef = useRef<HTMLDivElement>(null);
  const histRef = useRef<HTMLCanvasElement>(null);
  const studyHistRef = useRef<HTMLCanvasElement>(null);
  const [data, setData] = useState<Float32Array | null>(null);
  const [shape, setShape] = useState<[number, number, number] | null>(null);
  const [slice, setSlice] = useState(0);
  const [cmap, setCmap] = useState<Colormap>("turbo");
  const [vmin, setVmin] = useState(0);
  const [vmax, setVmax] = useState(1);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);
  const [roiMask, setRoiMask] = useState<Float32Array | null>(null);

  // Multi-slice voxel selection (3D mask) + histogram study
  const [selTool, setSelTool] = useState<SelectionTool>(null);
  const [brushRadius, setBrushRadius] = useState(5);
  const [selectionMask, setSelectionMask] = useState<Uint8Array | null>(null);
  const selectionRef = useRef<Uint8Array | null>(null);
  const lassoPointsRef = useRef<{ x: number; y: number }[]>([]);
  const [draftTick, setDraftTick] = useState(0);
  const sliceRef = useRef(0);
  const [brushDown, setBrushDown] = useState(false);
  const [selectedCount, setSelectedCount] = useState(0);
  const [study, setStudy] = useState<HistogramStudy | null>(null);
  const [studyErr, setStudyErr] = useState<string | null>(null);
  const [studyOpen, setStudyOpen] = useState(false);

  // Interactive ellipse ROI (voxel coords)
  const [ellipse, setEllipse] = useState<Ellipse | null>(null);
  const [roiStats, setRoiStats] = useState<RoiStats | null>(null);
  const [cursor, setCursor] = useState<string>("crosshair");
  const dragRef = useRef<{
    mode: DragMode;
    offX: number;
    offY: number;
  } | null>(null);

  const editable = !!onRoiDrawn;

  // Compute the auto-window. If roiWindowing is enabled and a ROI mask is
  // available, window over the ROI foreground only; otherwise fall back to the
  // full volume (using non-zero foreground when the map is mostly background).
  function computeWindow(arr: Float32Array, mask: Float32Array | null): [number, number] {
    let vals: number[];
    if (roiWindowing && mask) {
      const m = mask;
      vals = [];
      for (let i = 0; i < arr.length; i++) {
        if (m[i] > 0.5 && isFinite(arr[i]) && arr[i] !== 0) vals.push(arr[i]);
      }
    } else {
      const finite = arr.filter((v) => isFinite(v) && v !== 0);
      vals = finite.length > 0 && finite.length < arr.length * 0.5
        ? finite
        : arr.filter((v) => isFinite(v));
    }
    if (vals.length === 0) vals = arr.filter((v) => isFinite(v));
    const s = vals.slice().sort((a, b) => a - b);
    const p2 = s[Math.floor(s.length * 0.02)] ?? 0;
    const p98 = s[Math.floor(s.length * 0.98)] ?? 1;
    return [p2, p98 === p2 ? p2 + 1 : p98];
  }

  // Load NIfTI via nifti-reader-js
  useEffect(() => {
    setLoading(true);
    setError(null);
    const url = niftiPath.startsWith("http") ? niftiPath : api.niftiUrl(niftiPath);
    fetch(url)
      .then((r) => r.arrayBuffer())
      .then((buf) => {
        if (!nifti.isNIFTI(buf) && !nifti.isCompressed(buf)) {
          throw new Error("Not a valid NIfTI file");
        }
        const raw = nifti.isCompressed(buf) ? nifti.decompress(buf) : buf;
        const header = nifti.readHeader(raw);
        if (!header) throw new Error("Could not read NIfTI header");
        const image = nifti.readImage(header, raw) as ArrayBuffer;
        const nx = header.dims[1];
        const ny = header.dims[2];
        const nz = header.dims[3];
        let arr: Float32Array;
        if (header.datatypeCode === nifti.NIFTI1.TYPE_FLOAT32) arr = new Float32Array(image);
        else if (header.datatypeCode === nifti.NIFTI1.TYPE_INT16) arr = new Float32Array(new Int16Array(image));
        else if (header.datatypeCode === nifti.NIFTI1.TYPE_UINT8) arr = new Float32Array(new Uint8Array(image));
        else arr = new Float32Array(image);
        // Apply the NIfTI scaling/offset (scl_slope/scl_inter) so that derived
        // maps written by Slicer etc. (which store scaled integers with a header
        // scale+intercept) read back with physically correct values. Per the
        // NIfTI-1 spec a slope of 0 is treated as 1.
        const slope = Number.isFinite(header.scl_slope) && header.scl_slope !== 0 ? header.scl_slope : 1;
        const inter = Number.isFinite(header.scl_inter) ? header.scl_inter : 0;
        if (slope !== 1 || inter !== 0) {
          for (let i = 0; i < arr.length; i++) arr[i] = arr[i] * slope + inter;
        }
        setData(arr);
        setShape([nx, ny, nz]);
        setSlice(Math.floor(nz / 2));
        setVmin(0);
        setVmax(1);
        setSelectionMask(new Uint8Array(nx * ny * nz));
        selectionRef.current = new Uint8Array(nx * ny * nz);
        setSelectedCount(0);
        setStudy(null);
        setStudyOpen(false);
        setSelTool(null);
        setLoading(false);
      })
      .catch((e) => {
        setError(e.message);
        setLoading(false);
      });
  }, [niftiPath]);

  // Load optional ROI mask for windowing
  useEffect(() => {
    if (!roiPath) {
      setRoiMask(null);
      return;
    }
    const url = roiPath.startsWith("http") ? roiPath : api.niftiUrl(roiPath);
    fetch(url)
      .then((r) => r.arrayBuffer())
      .then((buf) => {
        const raw = nifti.isCompressed(buf) ? nifti.decompress(buf) : buf;
        const header = nifti.readHeader(raw);
        if (!header) return;
        const image = nifti.readImage(header, raw) as ArrayBuffer;
        let m: Float32Array;
        if (header.datatypeCode === nifti.NIFTI1.TYPE_FLOAT32) m = new Float32Array(image);
        else if (header.datatypeCode === nifti.NIFTI1.TYPE_INT16) m = new Float32Array(new Int16Array(image));
        else if (header.datatypeCode === nifti.NIFTI1.TYPE_UINT8) m = new Float32Array(new Uint8Array(image));
        else m = new Float32Array(image);
        const mslope = Number.isFinite(header.scl_slope) && header.scl_slope !== 0 ? header.scl_slope : 1;
        const minter = Number.isFinite(header.scl_inter) ? header.scl_inter : 0;
        if (mslope !== 1 || minter !== 0) {
          for (let i = 0; i < m.length; i++) m[i] = m[i] * mslope + minter;
        }
        setRoiMask(m);
      })
      .catch(() => setRoiMask(null));
  }, [roiPath]);

  // Recompute the auto-window whenever the data or ROI windowing settings change
  useEffect(() => {
    if (!data) return;
    const [lo, hi] = computeWindow(data, roiMask);
    setVmin(lo);
    setVmax(hi);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [data, roiMask, roiWindowing]);

  // Coordinate helpers (canvas pixel <-> voxel)
  function clientToCanvas(e: React.MouseEvent): [number, number] {
    const c = canvasRef.current!;
    const rect = c.getBoundingClientRect();
    const px = ((e.clientX - rect.left) / rect.width) * c.width;
    const py = ((e.clientY - rect.top) / rect.height) * c.height;
    return [px, py];
  }
  function canvasToVoxel(px: number, py: number): [number, number] {
    const c = canvasRef.current!;
    const [nx, ny] = [shape?.[0] ?? 1, shape?.[1] ?? 1];
    return [(px / c.width) * nx, (py / c.height) * ny];
  }
  function voxelToCanvas(vx: number, vy: number): [number, number] {
    const c = canvasRef.current!;
    const [nx, ny] = [shape?.[0] ?? 1, shape?.[1] ?? 1];
    return [(vx / nx) * c.width, (vy / ny) * c.height];
  }

  function pointInEllipse(vx: number, vy: number, e: Ellipse): boolean {
    if (e.a <= 0 || e.b <= 0) return false;
    const dx = (vx - e.cx) / e.a;
    const dy = (vy - e.cy) / e.b;
    return dx * dx + dy * dy <= 1;
  }
  function nearHandle(vx: number, vy: number, e: Ellipse): boolean {
    // handle at (cx + a, cy); threshold in canvas pixels -> voxel units
    const c = canvasRef.current!;
    const [nx] = [shape?.[0] ?? 1];
    const threshVox = (8 / c.width) * nx; // ~8px
    const dx = vx - (e.cx + e.a);
    const dy = vy - e.cy;
    return Math.sqrt(dx * dx + dy * dy) <= Math.max(threshVox, e.a * 0.15 + 0.5);
  }

  // ── Voxel selection helpers (lasso / brush) ─────────────────────────────────
  function mutSelection(): Uint8Array {
    if (!selectionRef.current) selectionRef.current = new Uint8Array((shape?.[0] ?? 0) * (shape?.[1] ?? 0) * (shape?.[2] ?? 0));
    return selectionRef.current;
  }
  function syncSelectionMask() {
    setSelectionMask(selectionRef.current ? selectionRef.current.slice() : null);
    let count = 0;
    if (selectionRef.current) {
      for (const v of selectionRef.current) if (v) count++;
    }
    setSelectedCount(count);
    if (count === 0) {
      setStudy(null);
      setStudyOpen(false);
    }
  }
  function clearSelection() {
    const sel = mutSelection();
    sel.fill(0);
    syncSelectionMask();
  }
  function pointInPolygon(pts: { x: number; y: number }[], vx: number, vy: number): boolean {
    let inside = false;
    for (let i = 0, j = pts.length - 1; i < pts.length; j = i++) {
      const xi = pts[i].x, yi = pts[i].y;
      const xj = pts[j].x, yj = pts[j].y;
      if (yi > vy !== yj > vy && vx < ((xj - xi) * (vy - yi)) / (yj - yi) + xi) {
        inside = !inside;
      }
    }
    return inside;
  }
  function fillPolygonOnSlice(pts: { x: number; y: number }[]) {
    if (!shape || pts.length < 3) return;
    const [nx, ny] = shape;
    const sel = mutSelection();
    const z = sliceRef.current;
    const minX = Math.max(0, Math.floor(Math.min(...pts.map((p) => p.x))));
    const maxX = Math.min(nx - 1, Math.ceil(Math.max(...pts.map((p) => p.x))));
    const minY = Math.max(0, Math.floor(Math.min(...pts.map((p) => p.y))));
    const maxY = Math.min(ny - 1, Math.ceil(Math.max(...pts.map((p) => p.y))));
    for (let y = minY; y <= maxY; y++) {
      for (let x = minX; x <= maxX; x++) {
        if (pointInPolygon(pts, x + 0.5, y + 0.5)) {
          sel[(z * ny + y) * nx + x] = 1;
        }
      }
    }
    syncSelectionMask();
  }
  function stampBrush(vx: number, vy: number) {
    if (!shape) return;
    const [nx, ny] = shape;
    const sel = mutSelection();
    const z = sliceRef.current;
    const r = brushRadius;
    const x0 = Math.max(0, Math.floor(vx - r));
    const x1 = Math.min(nx - 1, Math.ceil(vx + r));
    const y0 = Math.max(0, Math.floor(vy - r));
    const y1 = Math.min(ny - 1, Math.ceil(vy + r));
    for (let y = y0; y <= y1; y++) {
      for (let x = x0; x <= x1; x++) {
        const dx = x - vx;
        const dy = y - vy;
        if (dx * dx + dy * dy <= r * r) {
          sel[(z * ny + y) * nx + x] = 1;
        }
      }
    }
  }

  function renderSlice() {
    if (!data || !shape || !canvasRef.current) return;
    const [nx, ny, nz] = shape;
    const c = canvasRef.current;
    const ctx = c.getContext("2d")!;
    const w = c.width;
    const h = c.height;
    const img = ctx.createImageData(w, h);
    const cm = COLORMAPS[cmap];
    const range = vmax - vmin || 1;
    const z = slice;
    for (let y = 0; y < h; y++) {
      for (let x = 0; x < w; x++) {
        const dx = Math.floor((x / w) * nx);
        const dy = Math.floor((y / h) * ny);
        const idx = (z * ny + dy) * nx + dx;
        const val = data[idx] ?? 0;
        let t = (val - vmin) / range;
        t = Math.max(0, Math.min(1, t));
        const ci = Math.round(t * 255);
        const [r, g, b] = cm[ci];
        const o = (y * w + x) * 4;
        img.data[o] = r; img.data[o + 1] = g; img.data[o + 2] = b; img.data[o + 3] = 255;
      }
    }
    ctx.putImageData(img, 0, 0);

    // ── Selection overlay (selected voxels on the current slice) ──────────────
    if (enableHistogramSelection && selectionRef.current) {
      const sel = selectionRef.current;
      const z = slice;
      ctx.save();
      for (let y = 0; y < ny; y++) {
        for (let x = 0; x < nx; x++) {
          const idx = (z * ny + y) * nx + x;
          if (sel[idx]) {
            // tint a 1-voxel region in accent (semi-transparent)
            ctx.fillStyle = "rgba(0,168,166,0.45)";
            ctx.fillRect(
              (x / nx) * w,
              (y / ny) * h,
              Math.max(1, w / nx),
              Math.max(1, h / ny)
            );
          }
        }
      }
      ctx.restore();
    }

    // ── Lasso in-progress polyline ─────────────────────────────────────────────
    if (enableHistogramSelection && selTool === "lasso" && lassoPointsRef.current.length > 1) {
      const pts = lassoPointsRef.current;
      ctx.strokeStyle = "#00a8a6";
      ctx.lineWidth = 2;
      ctx.beginPath();
      pts.forEach((p, i) => {
        const [px, py] = voxelToCanvas(p.x, p.y);
        if (i === 0) ctx.moveTo(px, py);
        else ctx.lineTo(px, py);
      });
      ctx.stroke();
    }

    if (ellipse) {
      const [pcx, pcy] = voxelToCanvas(ellipse.cx, ellipse.cy);
      const [pax, pby] = voxelToCanvas(ellipse.cx + ellipse.a, ellipse.cy + ellipse.b);
      const ra = Math.abs(pax - pcx);
      const rb = Math.abs(pby - pcy);
      // ellipse outline
      ctx.strokeStyle = "#ffcc00";
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.ellipse(pcx, pcy, ra, rb, 0, 0, Math.PI * 2);
      ctx.stroke();
      // resize handle
      ctx.fillStyle = "#ffcc00";
      ctx.beginPath();
      ctx.arc(pcx + ra, pcy, 5, 0, Math.PI * 2);
      ctx.fill();
    }
  }

  useEffect(() => {
    renderSlice();
  }, [data, shape, slice, cmap, vmin, vmax, ellipse, selectionMask, selTool, selectedCount, draftTick]);

  // Keep the current slice available to (potentially stale) mouse handlers.
  useEffect(() => {
    sliceRef.current = slice;
  }, [slice]);

  // Recompute ROI stats when ellipse/slice/window changes
  useEffect(() => {
    if (!data || !shape || !ellipse) {
      setRoiStats(null);
      return;
    }
    const [nx, ny] = shape;
    const z = slice;
    const mask = new Uint8Array(nx * ny);
    for (let y = 0; y < ny; y++)
      for (let x = 0; x < nx; x++) {
        const dx = (x - ellipse.cx) / (ellipse.a || 1);
        const dy = (y - ellipse.cy) / (ellipse.b || 1);
        if (dx * dx + dy * dy <= 1) mask[y * nx + x] = 1;
      }
    const vals: number[] = [];
    for (let i = 0; i < mask.length; i++) {
      if (mask[i]) {
        const v = data[(z * ny + Math.floor(i / nx)) * nx + (i % nx)];
        if (isFinite(v)) vals.push(v);
      }
    }
    const n = vals.length;
    const mean = n ? vals.reduce((a, c) => a + c, 0) / n : 0;
    const std = n > 1 ? Math.sqrt(vals.reduce((a, c) => a + (c - mean) ** 2, 0) / (n - 1)) : 0;
    setRoiStats({ n, mean, std, min: n ? Math.min(...vals) : 0, max: n ? Math.max(...vals) : 0 });
    if (editable && ellipse.a > 0 && ellipse.b > 0) {
      onRoiDrawn!({ slice: z, cx: ellipse.cx, cy: ellipse.cy, a: ellipse.a, b: ellipse.b });
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [ellipse, slice, vmin, vmax]);

  function drawHistogram() {
    if (!histRef.current) return;
    const ctx = histRef.current.getContext("2d")!;
    const w = histRef.current.width;
    const h = histRef.current.height;
    ctx.clearRect(0, 0, w, h);
    ctx!.fillStyle = "#000";
    ctx!.fillRect(0, 0, w, h);
    if (!roiStats || !data || !shape) {
      ctx!.fillStyle = "#888";
      ctx!.font = "11px monospace";
      ctx!.fillText("No ROI", 10, h / 2);
      return;
    }
    const [nx, ny] = shape;
    const z = slice;
    const bins = new Array(64).fill(0);
    const range = vmax - vmin || 1;
    for (let y = 0; y < ny; y++)
      for (let x = 0; x < nx; x++) {
        const dx = (x - ellipse!.cx) / (ellipse!.a || 1);
        const dy = (y - ellipse!.cy) / (ellipse!.b || 1);
        if (dx * dx + dy * dy <= 1) {
          const v = data[(z * ny + y) * nx + x];
          let t = (v - vmin) / range;
          t = Math.max(0, Math.min(0.999, t));
          bins[Math.floor(t * 64)]++;
        }
      }
    const maxBin = Math.max(...bins, 1);
    ctx!.fillStyle = "#ffcc00";
    bins.forEach((cnt, i) => {
      const bh = (cnt / maxBin) * (h - 4);
      ctx!.fillRect((i / 64) * w, h - bh, w / 64, bh);
    });
  }

  useEffect(() => {
    drawHistogram();
  }, [roiStats, slice, vmin, vmax, ellipse]);

  // ── Interactive ellipse editing + voxel selection ────────────────────────────
  const selectActive = enableHistogramSelection && selTool !== null;

  // Lasso drawing state: starts on a click (mousedown) ON the image, drawn by
// drag, closes ONLY on double-click.
  const lassoActiveRef = useRef(false);
  const lassoDownRef = useRef(false);

  const onMouseDown = useCallback(
    (e: React.MouseEvent) => {
      if (!canvasRef.current) return;
      const [px, py] = clientToCanvas(e);
      const [vx, vy] = canvasToVoxel(px, py);

      // Voxel selection tools take priority
      if (selectActive) {
        if (selTool === "brush") {
          setBrushDown(true);
          stampBrush(vx, vy);
          syncSelectionMask();
          setCursor("none");
        } else if (selTool === "lasso") {
          // Start (or continue) the lasso polygon on a click ON the image.
          if (!lassoActiveRef.current) {
            lassoActiveRef.current = true;
            lassoDownRef.current = true;
            lassoPointsRef.current = [{ x: vx, y: vy }];
          } else {
            lassoDownRef.current = true;
            lassoPointsRef.current.push({ x: vx, y: vy });
          }
          setDraftTick((t) => t + 1);
          setCursor("crosshair");
        }
        return;
      }

      if (!editable) return;
      if (ellipse && nearHandle(vx, vy, ellipse)) {
        dragRef.current = { mode: "resize", offX: 0, offY: 0 };
        setCursor("nwse-resize");
      } else if (ellipse && pointInEllipse(vx, vy, ellipse)) {
        dragRef.current = { mode: "move", offX: vx - ellipse.cx, offY: vy - ellipse.cy };
        setCursor("move");
      } else {
        // start drawing a new ellipse
        dragRef.current = { mode: "draw", offX: 0, offY: 0 };
        setEllipse({ cx: vx, cy: vy, a: 0, b: 0 });
        setCursor("crosshair");
      }
    },
    [selectActive, selTool, editable, ellipse, shape, brushRadius]
  );

  const onMouseMove = useCallback(
    (e: React.MouseEvent) => {
      if (!canvasRef.current) return;
      const [px, py] = clientToCanvas(e);
      const [vx, vy] = canvasToVoxel(px, py);

      if (selectActive) {
        if (selTool === "brush" && brushDown) {
          stampBrush(vx, vy);
          syncSelectionMask();
          setCursor("none");
        } else if (selTool === "lasso" && lassoActiveRef.current && lassoDownRef.current) {
          lassoPointsRef.current.push({ x: vx, y: vy });
          setDraftTick((t) => t + 1);
          setCursor("crosshair");
        }
        return;
      }

      if (!editable) return;
      const drag = dragRef.current;
      if (!drag) {
        // hover cursor feedback
        if (ellipse && nearHandle(vx, vy, ellipse)) setCursor("nwse-resize");
        else if (ellipse && pointInEllipse(vx, vy, ellipse)) setCursor("move");
        else setCursor("crosshair");
        return;
      }
      if (drag.mode === "draw") {
        setEllipse((prev) => prev && { cx: prev.cx, cy: prev.cy, a: Math.abs(vx - prev.cx), b: Math.abs(vy - prev.cy) });
      } else if (drag.mode === "move" && ellipse) {
        setEllipse({ ...ellipse, cx: vx - drag.offX, cy: vy - drag.offY });
      } else if (drag.mode === "resize" && ellipse) {
        setEllipse({ ...ellipse, a: Math.abs(vx - ellipse.cx), b: Math.abs(vy - ellipse.cy) });
      }
    },
    [selectActive, selTool, brushDown, editable, ellipse, shape, brushRadius]
  );

  const endDrag = useCallback(() => {
    dragRef.current = null;
    setBrushDown(false);
    // Releasing the mouse does NOT close a lasso; only a double-click does.
    lassoDownRef.current = false;
    setCursor("crosshair");
  }, []);

  // Double-click closes the lasso and fills the current slice.
  const handleLassoDblClick = useCallback(
    (e: React.MouseEvent) => {
      if (!selectActive || selTool !== "lasso" || !lassoActiveRef.current) return;
      e.preventDefault();
      // The final click of the double-click added a closing vertex; drop it.
      const pts = lassoPointsRef.current.slice(0, -1);
      if (pts.length >= 3) {
        fillPolygonOnSlice(pts);
      }
      lassoActiveRef.current = false;
      lassoDownRef.current = false;
      lassoPointsRef.current = [];
      setDraftTick((t) => t + 1);
      setCursor("crosshair");
    },
    [selectActive, selTool]
  );

  // Switching tool / clearing resets any in-progress lasso.
  useEffect(() => {
    lassoActiveRef.current = false;
    lassoDownRef.current = false;
    lassoPointsRef.current = [];
  }, [selTool]);

  // Wheel over the viewer must switch slice and NOT scroll the page. React's
  // onWheel is attached passively at the root, so preventDefault would be a
  // no-op — attach a native non-passive listener instead.
  useEffect(() => {
    const el = wrapRef.current;
    if (!el) return;
    const onWheel = (e: WheelEvent) => {
      e.preventDefault();
      if (!shape) return;
      const delta = e.deltaY > 0 ? 1 : -1;
      lassoPointsRef.current = [];
      setSlice((s) => Math.max(0, Math.min(shape[2] - 1, s + delta)));
    };
    el.addEventListener("wheel", onWheel, { passive: false });
    return () => el.removeEventListener("wheel", onWheel);
  }, [shape]);

  function runStudy() {
    setStudyErr(null);
    if (!data || !selectionRef.current) return;
    const sel = selectionRef.current;
    const collected: number[] = [];
    let finiteCount = 0;
    for (let i = 0; i < sel.length; i++) {
      if (sel[i]) {
        const v = data[i];
        if (Number.isFinite(v)) finiteCount++;
        if (Number.isFinite(v) && v > 0) collected.push(v);
      }
    }
    const raw = new Float32Array(collected);
    if (raw.length === 0) {
      setStudyErr(
        selectionRef.current.reduce((a, b) => a + b, 0) === 0
          ? "No voxels are selected. Draw a lasso (double-click to close) or paint with the brush first."
          : `No valid voxel values in the selection (${finiteCount} finite, all \u2264 0).`
      );
      setStudy(null);
      setStudyOpen(true);
      return;
    }
    try {
      const res = runHistogramStudy(raw);
      setStudy(res);
      setStudyErr(null);
      setStudyOpen(true);
    } catch (err: any) {
      setStudy(null);
      setStudyErr("Histogram computation failed: " + (err?.message ?? err));
      setStudyOpen(true);
    }
  }

  // Draw the study histogram + Gaussian fit to the study canvas.
  useEffect(() => {
    const ctx = studyHistRef.current?.getContext("2d");
    const canvas = studyHistRef.current;
    if (!ctx || !canvas || !study) return;
    const w = canvas.width;
    const h = canvas.height;
    ctx.clearRect(0, 0, w, h);
    ctx.fillStyle = "#0e1117";
    ctx.fillRect(0, 0, w, h);

    const { counts, binEdges, binWidth } = study.histogram;
    if (counts.length === 0) {
      ctx.fillStyle = "#888";
      ctx.font = "12px monospace";
      ctx.fillText("No data", 10, h / 2);
      return;
    }
    const lo = binEdges[0];
    const hi = binEdges[binEdges.length - 1];
    const range = hi - lo || 1;
    const padL = 46, padR = 10, padT = 10, padB = 28;
    const plotW = w - padL - padR;
    const plotH = h - padT - padB;
    const maxCount = Math.max(...(counts as unknown as number[]), 1);

    // bars
    counts.forEach((cnt, i) => {
      const bx = padL + ((binEdges[i] - lo) / range) * plotW;
      const bw = Math.max(1, ((binEdges[i + 1] - binEdges[i]) / range) * plotW);
      const bh = (cnt / maxCount) * plotH;
      ctx.fillStyle = "#1c7f7d";
      ctx.fillRect(bx, padT + plotH - bh, bw, bh);
    });

    // Gaussian fit curve
    ctx.strokeStyle = "#00a8a6";
    ctx.lineWidth = 2;
    ctx.beginPath();
    let started = false;
    for (let i = 0; i < study.fitX.length; i++) {
      const x = study.fitX[i];
      const fx = padL + ((x - lo) / range) * plotW;
      const fy = padT + plotH - (study.fitCounts[i] / maxCount) * plotH;
      if (!started) { ctx.moveTo(fx, fy); started = true; }
      else ctx.lineTo(fx, fy);
    }
    ctx.stroke();

    // axes + labels
    ctx.strokeStyle = "#3a4048";
    ctx.beginPath();
    ctx.moveTo(padL, padT); ctx.lineTo(padL, padT + plotH);
    ctx.lineTo(padL + plotW, padT + plotH);
    ctx.stroke();
    ctx.fillStyle = "#9aa3ae";
    ctx.font = "11px monospace";
    ctx.textAlign = "center";
    ctx.fillText("bin width = " + binWidth.toExponential(2), padL + plotW / 2, h - 8);
    ctx.textAlign = "left";
  }, [study, studyOpen]);

  if (loading) return <p className="muted">Loading NIfTI…</p>;
  if (error) return <p style={{ color: "var(--red)" }}>Error: {error}</p>;

  return (
    <>
    <div className="viewer-grid">
      <div className="viewer-canvas-wrap" ref={wrapRef}>
        <canvas
          ref={canvasRef}
          width={512}
          height={512}
          className="viewer-canvas"
          style={{ cursor: selectActive ? cursor : editable ? cursor : "default", userSelect: "none" }}
          onMouseDown={selectActive || editable ? onMouseDown : undefined}
          onMouseMove={selectActive || editable ? onMouseMove : undefined}
          onMouseUp={selectActive || editable ? endDrag : undefined}
          onMouseLeave={selectActive || editable ? endDrag : undefined}
          onDoubleClick={selectActive ? handleLassoDblClick : undefined}
        />
      </div>
      <div className="viewer-side">
        {title && <strong>{title}</strong>}
        {enableHistogramSelection && (
          <div className="sel-toolbar">
            <div className="row" style={{ gap: 4, flexWrap: "wrap" }}>
              <button
                className={selTool === "lasso" ? "sel-tool active" : "sel-tool"}
                onClick={() => { setSelTool(selTool === "lasso" ? null : "lasso"); setStudyOpen(false); }}
              >
                Lasso
              </button>
              <button
                className={selTool === "brush" ? "sel-tool active" : "sel-tool"}
                onClick={() => { setSelTool(selTool === "brush" ? null : "brush"); setStudyOpen(false); }}
              >
                Brush
              </button>
              <button className="sel-tool" onClick={clearSelection} disabled={selectedCount === 0}>
                Clear
              </button>
            </div>
            <div className="row" style={{ gap: 8, alignItems: "center", marginTop: 6 }}>
              <label className="muted" style={{ fontSize: 11 }}>Brush r</label>
              <input
                type="range" min={2} max={25} value={brushRadius}
                onChange={(e) => setBrushRadius(parseInt(e.target.value))}
                style={{ width: 90 }}
              />
              <span className="muted" style={{ fontSize: 11 }}>{brushRadius}</span>
            </div>
            {selectedCount > 0 && (
              <p className="muted" style={{ fontSize: 11, marginTop: 6 }}>
                <span style={{ color: "var(--accent)", fontWeight: 600 }}>{selectedCount}</span> voxels selected
                across {shape?.[2] ?? 0} slices.
              </p>
            )}
            {selTool === "lasso" && (
              <p className="muted" style={{ fontSize: 11, marginTop: 6, lineHeight: 1.4 }}>
                Click on the image to start, drag to trace the boundary, and{" "}
                <strong>double-click</strong> to close &amp; fill.
              </p>
            )}
            {selTool === "brush" && (
              <p className="muted" style={{ fontSize: 11, marginTop: 6, lineHeight: 1.4 }}>
                Drag with the left button to paint voxels onto the current slice.
              </p>
            )}
            <button
              className="sel-run"
              onClick={runStudy}
              disabled={selectedCount === 0}
              style={{ marginTop: 8, width: "100%" }}
            >
              Run Histogram Study
            </button>
          </div>
        )}
        <div>
          <label className="muted">Slice</label>
          <input
            type="range"
            min={0}
            max={(shape?.[2] ?? 1) - 1}
            value={slice}
            onChange={(e) => setSlice(parseInt(e.target.value))}
            style={{ width: "100%" }}
          />
          <span className="muted">
            {slice + 1} / {shape?.[2]}
          </span>
        </div>
        <div>
          <label className="muted">Colormap</label>
          <select value={cmap} onChange={(e) => setCmap(e.target.value as Colormap)} style={{ width: "100%" }}>
            <option value="gray">Gray</option>
            <option value="turbo">Turbo</option>
            <option value="jet">Jet</option>
          </select>
        </div>
        <div>
          <label className="muted">Min</label>
          <input type="number" value={vmin} onChange={(e) => setVmin(parseFloat(e.target.value))} style={{ width: "100%" }} />
        </div>
        <div>
          <label className="muted">Max</label>
          <input type="number" value={vmax} onChange={(e) => setVmax(parseFloat(e.target.value))} style={{ width: "100%" }} />
        </div>
        {editable && (
          <p className="muted" style={{ fontSize: 12, lineHeight: 1.4 }}>
            Drag on the image to draw an ellipse. Drag its body to move, or the
            yellow handle to resize.
          </p>
        )}
        <div className="viewer-stats">
          {roiStats && (
            <div style={{ color: "#ffcc00", marginBottom: 6 }}>
              ROI: n={roiStats.n} mean={roiStats.mean.toFixed(4)} sd={roiStats.std.toFixed(4)}
              <br />
              min={roiStats.min.toFixed(4)} max={roiStats.max.toFixed(4)}
            </div>
          )}
        </div>
        <canvas ref={histRef} width={200} height={120} className="viewer-hist" />
        {onOpenExternal && (
          <button onClick={onOpenExternal} style={{ background: "var(--panel-2)" }}>
            Open Externally
          </button>
        )}
      </div>
    </div>

      {studyOpen && (study || studyErr) && (
        <div className="hist-study">
          <div className="row" style={{ justifyContent: "space-between", alignItems: "center" }}>
            <h3 style={{ margin: 0 }}>Histogram study</h3>
            <button className="secondary" onClick={() => setStudyOpen(false)}>Close</button>
          </div>
          {studyErr && <p style={{ color: "var(--red)" }}>{studyErr}</p>}
          {study && (
            <>
          <canvas ref={studyHistRef} width={520} height={280} className="hist-study-canvas" />
          <div className="hist-stats-table">
            <table>
              <thead>
                <tr>
                  <th>Metric</th>
                  <th>Value</th>
                  <th>Metric</th>
                  <th>Value</th>
                </tr>
              </thead>
              <tbody>
                <tr>
                  <td>N voxels</td><td>{study.stats.n}</td>
                  <td>Mean</td><td>{study.stats.mean.toPrecision(4)}</td>
                </tr>
                <tr>
                  <td>Median (P50)</td><td>{study.stats.median.toPrecision(4)}</td>
                  <td>Std</td><td>{study.stats.std.toPrecision(4)}</td>
                </tr>
                <tr>
                  <td>P10</td><td>{study.stats.p10.toPrecision(4)}</td>
                  <td>P75</td><td>{study.stats.p75.toPrecision(4)}</td>
                </tr>
                <tr>
                  <td>P90</td><td>{study.stats.p90.toPrecision(4)}</td>
                  <td>Min / Max</td><td>{study.stats.min.toPrecision(4)} / {study.stats.max.toPrecision(4)}</td>
                </tr>
                <tr>
                  <td>Skewness</td><td>{study.stats.skewness.toPrecision(4)}</td>
                  <td>Kurtosis</td>
                  <td>
                    {study.stats.excessKurtosis.toPrecision(4)}
                    <span className="muted" style={{ fontSize: 10 }}> (excess)</span>
                  </td>
                </tr>
              </tbody>
            </table>
            <p className="muted" style={{ fontSize: 11, marginTop: 6 }}>
              Gaussian fit: mean μ = {study.fit.mu.toPrecision(4)}, σ = {study.fit.sigma.toPrecision(4)}.
              {Math.abs(study.stats.skewness) > 1.0 && <> Notable skewness ({study.stats.skewness.toFixed(2)}): an asymmetric distribution.</>}
              {Math.abs(study.stats.excessKurtosis) > 2.0 && <> Heavy/light tails (excess kurtosis {study.stats.excessKurtosis.toFixed(2)}).</>}
            </p>
          </div>
          </>
          )}
        </div>
      )}
    </>
  );
}
