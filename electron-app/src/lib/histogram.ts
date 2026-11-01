export interface HistogramResult {
  counts: Float64Array;
  binEdges: Float64Array;
  binCenters: Float64Array;
  binWidth: number;
}

export interface GaussianFit {
  mu: number;
  sigma: number;
  /** pdf(x) for the fitted normal distribution */
  pdf: (x: number) => number;
}

export interface HistStats {
  n: number;
  mean: number;
  median: number;
  p10: number;
  p50: number;
  p75: number;
  p90: number;
  min: number;
  max: number;
  std: number;
  skewness: number;
  excessKurtosis: number;
}

/**
 * Clean raw voxel values: drop NaN/Inf and nonpositive values (noise floor),
 * mirroring the reference ADC-histogram "prepare_adc_values" behaviour.
 */
export function prepareValues(raw: Float32Array, minValue = 0): number[] {
  const out: number[] = [];
  for (let i = 0; i < raw.length; i++) {
    const v = raw[i];
    if (Number.isFinite(v) && v > minValue) out.push(v);
  }
  return out;
}

/**
 * Fixed / auto bin-width histogram. Uses ~sqrt(N) bins (Sturges-like) which is
 * robust for typical ROI-selected voxel counts.
 */
export function computeHistogram(values: number[], binCard?: number): HistogramResult {
  if (values.length === 0) {
    return { counts: new Float64Array(0), binEdges: new Float64Array(0), binCenters: new Float64Array(0), binWidth: 0 };
  }
  let vmin = Infinity;
  let vmax = -Infinity;
  for (const v of values) {
    if (v < vmin) vmin = v;
    if (v > vmax) vmax = v;
  }
  const nBins = binCard ?? Math.max(1, Math.round(Math.sqrt(values.length)));
  if (vmax === vmin) vmax = vmin + 1;
  const binWidth = (vmax - vmin) / nBins;
  const counts = new Float64Array(nBins);
  for (const v of values) {
    let b = Math.floor((v - vmin) / binWidth);
    if (b >= nBins) b = nBins - 1;
    if (b < 0) b = 0;
    counts[b]++;
  }
  const binEdges = new Float64Array(nBins + 1);
  const binCenters = new Float64Array(nBins);
  for (let i = 0; i <= nBins; i++) binEdges[i] = vmin + i * binWidth;
  for (let i = 0; i < nBins; i++) binCenters[i] = (binEdges[i] + binEdges[i + 1]) / 2;
  return { counts, binEdges, binCenters, binWidth };
}

/** Fit a Gaussian (normal) to raw voxel values: mean + sample std (ddof=1). */
export function fitGaussian(values: number[]): GaussianFit {
  const n = values.length;
  let mean = 0;
  for (const v of values) mean += v;
  mean /= n;
  let varSum = 0;
  for (const v of values) varSum += (v - mean) ** 2;
  const sigma = n > 1 ? Math.sqrt(varSum / (n - 1)) : 0;
  const inv = (2 * Math.PI);
  const pdf = (x: number) => {
    if (sigma === 0) return x === mean ? Infinity : 0;
    return Math.exp(-0.5 * ((x - mean) / sigma) ** 2) / (sigma * Math.sqrt(inv));
  };
  return { mu: mean, sigma, pdf };
}

/** Convert probability density to expected bin counts: counts(x) = N * binWidth * pdf(x). */
export function scalePdfToCounts(pdfValues: number[], n: number, binWidth: number): number[] {
  return pdfValues.map((p) => n * binWidth * p);
}

/** Sample statistics: mean, median, percentiles, std, skewness, excess kurtosis. */
export function computeStatistics(values: number[]): HistStats {
  const n = values.length;
  if (n === 0) {
    return { n, mean: 0, median: 0, p10: 0, p50: 0, p75: 0, p90: 0, min: 0, max: 0, std: 0, skewness: 0, excessKurtosis: 0 };
  }
  const sorted = values.slice().sort((a, b) => a - b);
  let sum = 0;
  for (const v of values) sum += v;
  const mean = sum / n;
  let varSum = 0;
  for (const v of values) varSum += (v - mean) ** 2;
  const std = n > 1 ? Math.sqrt(varSum / (n - 1)) : 0;

  const percentile = (p: number) => {
    const pos = (p / 100) * (n - 1);
    const lo = Math.floor(pos);
    const hi = Math.min(lo + 1, n - 1);
    const frac = pos - lo;
    return sorted[lo] + (sorted[hi] - sorted[lo]) * frac;
  };

// Bias-corrected (bias=False) skewness/excess kurtosis, matching scipy:
//   g1 = √(n(n-1))/(n-2) · M3 / M2^(3/2)
//   G2 = 1/((n-2)(n-3)) · ( (n²−1)·M4/M2² − 3(n−1)² )
// where Mk = Σ(x−x̄)^k / n are population-style moments.
  let s2 = 0, s3 = 0, s4 = 0;
  for (const v of values) {
    const d = v - mean;
    s2 += d * d;
    s3 += d * d * d;
    s4 += d * d * d * d;
  }
  const M2 = s2 / n;
  const M3 = s3 / n;
  const M4 = s4 / n;
  const skewness =
    n > 2 ? (Math.sqrt(n * (n - 1)) / (n - 2)) * (M3 / Math.pow(M2, 1.5)) : 0;
  const excessKurtosis =
    n > 3
      ? (1 / ((n - 2) * (n - 3))) * ((n * n - 1) * (M4 / (M2 * M2 || 1)) - 3 * Math.pow(n - 1, 2))
      : 0;

  return {
    n,
    mean,
    median: percentile(50),
    p10: percentile(10),
    p50: percentile(50),
    p75: percentile(75),
    p90: percentile(90),
    min: sorted[0],
    max: sorted[n - 1],
    std,
    skewness,
    excessKurtosis,
  };
}

export interface HistogramStudy {
  histogram: HistogramResult;
  fit: GaussianFit;
  stats: HistStats;
  /** x points for drawing the fit curve, in display units */
  fitX: number[];
  fitCounts: number[];
}

/** Run a full histogram study on the cleaned voxel values. */
export function runHistogramStudy(raw: Float32Array, minValue = 0, binCard?: number): HistogramStudy {
  const values = prepareValues(raw, minValue);
  const histogram = computeHistogram(values, binCard);
  const fit = fitGaussian(values);
  const stats = computeStatistics(values);
  const fitX: number[] = [];
  const fitCounts: number[] = [];
  if (histogram.binEdges.length >= 2) {
    const lo = histogram.binEdges[0];
    const hi = histogram.binEdges[histogram.binEdges.length - 1];
    const steps = 300;
    for (let i = 0; i <= steps; i++) {
      const x = lo + ((hi - lo) * i) / steps;
      fitX.push(x);
      fitCounts.push(scalePdfToCounts([fit.pdf(x)], values.length, histogram.binWidth)[0]);
    }
  }
  return { histogram, fit, stats, fitX, fitCounts };
}
