"use client";

import { useMemo, useState } from "react";
import dynamic from "next/dynamic";

import {
  kaplan_loss_ND,
  training_flops,
  flops_to_pf_days,
  tokens_per_parameter,
  human_num,
} from "@/lib/scaling_laws";

const Plot = dynamic(() => import("react-plotly.js"), { ssr: false });

const COLORS = [
  "#60a5fa",
  "#34d399",
  "#fbbf24",
  "#fb7185",
  "#a78bfa",
  "#22d3ee",
  "#f97316",
  "#84cc16",
  "#e879f9",
  "#2dd4bf",
  "#c084fc",
  "#facc15",
];

const X_METRIC_OPTIONS = [
  { value: "N", label: "Model size N" },
  { value: "D", label: "Dataset size D" },
  { value: "pf_days", label: "Compute, PF-days" },
  { value: "D_over_N", label: "Data/model ratio D/N" },
  { value: "coverage", label: "Training coverage" },
  { value: "steps", label: "Optimizer steps" },
];

const CALIBRATION_OPTIONS = [
  { value: "raw", label: "Raw Kaplan L(N,D)" },
  { value: "additive", label: "Fit additive offset: Kaplan + b" },
  { value: "affine", label: "Fit affine map: a·Kaplan + b" },
];

const DEFAULT_GROUP_COLUMNS = "num_train_tokens";

function clamp(x, lo, hi) {
  return Math.max(lo, Math.min(hi, x));
}

function safeDivide(a, b) {
  return Number.isFinite(a) && Number.isFinite(b) && b !== 0 ? a / b : NaN;
}

function stableId(text, index = 0) {
  return `${String(text ?? "curve")
    .trim()
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, "_")
    .replace(/^_+|_+$/g, "") || "curve"}_${index}`;
}

function sanitizeFilenamePart(value) {
  return String(value ?? "")
    .trim()
    .replace(/[^a-zA-Z0-9._-]+/g, "_")
    .replace(/^_+|_+$/g, "") || "kaplan_chart";
}

function formatSci(x, digits = 3) {
  if (!Number.isFinite(x)) return "—";
  return Number(x).toExponential(digits);
}

function formatLoss(x) {
  if (!Number.isFinite(x)) return "—";
  return Number(x).toFixed(5);
}

function formatPercent(x, digits = 1) {
  if (!Number.isFinite(x)) return "—";
  return `${x >= 0 ? "+" : ""}${x.toFixed(digits)}%`;
}

function formatTableValue(key, value) {
  if (value === null || value === undefined || Number.isNaN(value)) return "—";

  if (typeof value === "number") {
    const lower = key.toLowerCase();
    if (lower.includes("loss")) return formatLoss(value);
    if (lower.includes("pct") || lower.includes("percent") || lower.includes("error")) return formatPercent(value);
    if (lower.includes("ratio") || lower.includes("gain")) return Number.isFinite(value) ? value.toFixed(2) : "—";
    if (lower.includes("slope")) return Number.isFinite(value) ? value.toFixed(4) : "—";
    if (lower.includes("pf")) return Number.isFinite(value) ? value.toFixed(4) : "—";
    if (lower.includes("lr")) return formatSci(value, 3);
    return human_num(value);
  }

  return value;
}

function parseSmartNumber(raw) {
  if (typeof raw === "number") return raw;
  if (raw === null || raw === undefined) return NaN;

  let text = String(raw).trim();
  if (!text) return NaN;

  text = text
    .replace(/,/g, "")
    .replace(/_/g, "")
    .replace(/\s+/g, "")
    .toLowerCase();

  text = text.replace(
    /(tokens?|parameters?|params?|sequences?|seqs?|steps?|gpus?|gpu|pf-days?|pflops?|layers?|heads?|width|seconds?|sec|s)$/i,
    ""
  );

  const match = text.match(/^([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:e[+-]?\d+)?)([a-z]*)$/);
  if (!match) return NaN;

  const value = Number(match[1]);
  if (!Number.isFinite(value)) return NaN;

  const suffix = match[2];
  const multipliers = {
    "": 1,
    k: 1e3,
    thousand: 1e3,
    m: 1e6,
    million: 1e6,
    b: 1e9,
    bn: 1e9,
    billion: 1e9,
    t: 1e12,
    trillion: 1e12,
    p: 1e15,
  };

  return suffix in multipliers ? value * multipliers[suffix] : NaN;
}

function numberFromAny(...values) {
  for (const value of values) {
    const parsed = parseSmartNumber(value);
    if (Number.isFinite(parsed)) return parsed;
  }
  return NaN;
}

function rowValue(row, columnName) {
  if (!row || !columnName) return undefined;
  if (Object.prototype.hasOwnProperty.call(row, columnName)) return row[columnName];

  const normalizedTarget = String(columnName).trim().toLowerCase();
  const actualKey = Object.keys(row).find((key) => key.trim().toLowerCase() === normalizedTarget);
  return actualKey ? row[actualKey] : undefined;
}

function parseGroupColumns(groupColumnsText) {
  return String(groupColumnsText ?? "")
    .split(",")
    .map((column) => column.trim())
    .filter(Boolean);
}

function niceGroupColumnName(column) {
  const map = {
    num_train_tokens: "D",
    D_tokens: "D",
    num_params: "N",
    N_params: "N",
    learning_rate: "lr",
    min_lr: "min_lr",
    lr_scheduler: "scheduler",
    batch_size: "batch",
    world_size: "GPUs",
    n_layer: "L",
    n_head: "H",
    n_embd: "d_model",
  };

  return map[column] ?? column;
}

function formatGroupValue(column, value) {
  const parsed = parseSmartNumber(value);
  const lower = String(column).toLowerCase();

  if (!Number.isFinite(parsed)) return String(value ?? "").trim() || "missing";
  if (lower.includes("lr")) return formatSci(parsed, 2);
  if (lower.includes("tokens") || lower.includes("params") || lower.includes("size") || lower.includes("compute")) return human_num(parsed);
  return String(value ?? "").trim() || human_num(parsed);
}

function parseCsvText(text) {
  const rows = [];
  let row = [];
  let cell = "";
  let inQuotes = false;

  const pushCell = () => {
    row.push(cell);
    cell = "";
  };

  const pushRow = () => {
    if (row.some((value) => String(value).trim() !== "")) rows.push(row);
    row = [];
  };

  const source = String(text ?? "").replace(/^\ufeff/, "");

  for (let i = 0; i < source.length; i++) {
    const char = source[i];
    const next = source[i + 1];

    if (char === '"') {
      if (inQuotes && next === '"') {
        cell += '"';
        i += 1;
      } else {
        inQuotes = !inQuotes;
      }
      continue;
    }

    if (char === "," && !inQuotes) {
      pushCell();
      continue;
    }

    if ((char === "\n" || char === "\r") && !inQuotes) {
      pushCell();
      pushRow();
      if (char === "\r" && next === "\n") i += 1;
      continue;
    }

    cell += char;
  }

  pushCell();
  pushRow();

  if (rows.length < 2) {
    throw new Error("CSV import needs a header row and at least one data row.");
  }

  const headers = rows[0].map((header) => String(header).trim());
  const nonEmptyHeaders = headers.filter(Boolean);

  if (nonEmptyHeaders.length === 0) {
    throw new Error("CSV header row is empty.");
  }

  return rows.slice(1).map((values) => {
    const out = {};
    headers.forEach((header, index) => {
      if (!header) return;
      out[header] = values[index] === undefined ? "" : String(values[index]).trim();
    });
    return out;
  });
}

function inferTokensPerUpdate(row) {
  const direct = numberFromAny(rowValue(row, "tokens_per_update"));
  if (Number.isFinite(direct)) return direct;

  const batch = numberFromAny(rowValue(row, "batch_size"), rowValue(row, "batch_size_sequences"));
  const gradAccum = numberFromAny(rowValue(row, "gradient_accumulation_steps"), rowValue(row, "grad_accum"));
  const blockSize = numberFromAny(rowValue(row, "block_size"), rowValue(row, "max_sequence_length"), rowValue(row, "seq_len"));
  const worldSize = numberFromAny(rowValue(row, "world_size"), rowValue(row, "num_gpus"), 1);

  if ([batch, gradAccum, blockSize, worldSize].every(Number.isFinite)) {
    return batch * gradAccum * blockSize * worldSize;
  }

  return NaN;
}

function normalizeRow(rawRow, index = 0) {
  const N = numberFromAny(
    rowValue(rawRow, "N"),
    rowValue(rawRow, "N_params"),
    rowValue(rawRow, "num_params"),
    rowValue(rawRow, "non_embedding_params"),
    rowValue(rawRow, "estimated_non_embedding_params")
  );

  const D = numberFromAny(
    rowValue(rawRow, "D"),
    rowValue(rawRow, "D_tokens"),
    rowValue(rawRow, "num_train_tokens"),
    rowValue(rawRow, "dataset_tokens"),
    rowValue(rawRow, "tokens")
  );

  const measuredLoss = numberFromAny(
    rowValue(rawRow, "loss"),
    rowValue(rawRow, "val_loss"),
    rowValue(rawRow, "min_val_loss"),
    rowValue(rawRow, "min_saved_val_loss"),
    rowValue(rawRow, "validation_loss")
  );

  const trainLoss = numberFromAny(
    rowValue(rawRow, "train_loss"),
    rowValue(rawRow, "min_train_loss"),
    rowValue(rawRow, "min_saved_train_loss")
  );

  const rawKaplanLoss = Number.isFinite(N) && Number.isFinite(D) ? kaplan_loss_ND(N, D) : NaN;
  const flops = numberFromAny(rowValue(rawRow, "flops"), rowValue(rawRow, "training_flops"), rowValue(rawRow, "compute_flops"), rowValue(rawRow, "compute"));
  const pfDaysFromFlops = Number.isFinite(flops) && flops > 1e12 ? flops_to_pf_days(flops) : NaN;
  const pfDaysFromND = Number.isFinite(N) && Number.isFinite(D) ? flops_to_pf_days(training_flops(N, D)) : NaN;

  const steps = numberFromAny(
    rowValue(rawRow, "steps"),
    rowValue(rawRow, "optimizer_steps"),
    rowValue(rawRow, "max_iters"),
    rowValue(rawRow, "iters_saved"),
    rowValue(rawRow, "iters_trained")
  );

  const tokensPerUpdate = inferTokensPerUpdate(rawRow);
  const tokensSeen = Number.isFinite(steps) && Number.isFinite(tokensPerUpdate) ? steps * tokensPerUpdate : NaN;
  const coverage = numberFromAny(rowValue(rawRow, "coverage"), safeDivide(tokensSeen, D));

  return {
    index,
    name: rowValue(rawRow, "model_name") || rowValue(rawRow, "name") || `row ${index + 1}`,
    N,
    D,
    D_over_N: Number.isFinite(N) && Number.isFinite(D) ? tokens_per_parameter(D, N) : NaN,
    loss: measuredLoss,
    train_loss: trainLoss,
    train_val_gap: Number.isFinite(measuredLoss) && Number.isFinite(trainLoss) ? measuredLoss - trainLoss : NaN,
    kaplan_loss_raw: rawKaplanLoss,
    pf_days: Number.isFinite(pfDaysFromFlops) ? pfDaysFromFlops : pfDaysFromND,
    steps,
    tokens_per_update: tokensPerUpdate,
    tokens_seen: tokensSeen,
    coverage,
    learning_rate: numberFromAny(rowValue(rawRow, "learning_rate"), rowValue(rawRow, "lr")),
    min_lr: numberFromAny(rowValue(rawRow, "min_lr"), rowValue(rawRow, "lr_min")),
    lr_decay_iters: numberFromAny(rowValue(rawRow, "lr_decay_iters")),
    n_layer: numberFromAny(rowValue(rawRow, "n_layer"), rowValue(rawRow, "L")),
    n_head: numberFromAny(rowValue(rawRow, "n_head"), rowValue(rawRow, "H")),
    n_embd: numberFromAny(rowValue(rawRow, "n_embd"), rowValue(rawRow, "d_model"), rowValue(rawRow, "dModel")),
    raw: rawRow,
  };
}

function cleanCurve(name, rows, curveIndex = 0) {
  const points = rows
    .map((row, index) => normalizeRow(row, index))
    .filter((point) => Number.isFinite(point.N) && Number.isFinite(point.D) && Number.isFinite(point.loss) && Number.isFinite(point.kaplan_loss_raw))
    .sort((a, b) => a.N - b.N || a.D - b.D || a.index - b.index);

  return {
    id: stableId(name, curveIndex),
    name,
    color: COLORS[curveIndex % COLORS.length],
    hidden: false,
    points,
  };
}

function csvRowsToCurves(rows, groupColumnsText = DEFAULT_GROUP_COLUMNS) {
  const groupColumns = parseGroupColumns(groupColumnsText);

  if (groupColumns.length === 0) {
    return [cleanCurve("Pasted CSV", rows, 0)].filter((curve) => curve.points.length > 0);
  }

  const groups = new Map();

  for (const row of rows) {
    const groupName = groupColumns
      .map((column) => `${niceGroupColumnName(column)}=${formatGroupValue(column, rowValue(row, column))}`)
      .join(", ");

    if (!groups.has(groupName)) groups.set(groupName, []);
    groups.get(groupName).push(row);
  }

  return Array.from(groups.entries())
    .map(([name, groupRows], index) => cleanCurve(name, groupRows, index))
    .filter((curve) => curve.points.length > 0);
}

function mean(values) {
  const clean = values.filter(Number.isFinite);
  if (clean.length === 0) return NaN;
  return clean.reduce((sum, x) => sum + x, 0) / clean.length;
}

function median(values) {
  const clean = values.filter(Number.isFinite).slice().sort((a, b) => a - b);
  if (clean.length === 0) return NaN;
  const mid = Math.floor(clean.length / 2);
  return clean.length % 2 ? clean[mid] : 0.5 * (clean[mid - 1] + clean[mid]);
}

function rmse(values) {
  const clean = values.filter(Number.isFinite);
  if (clean.length === 0) return NaN;
  return Math.sqrt(mean(clean.map((x) => x * x)));
}

function linearFit(xs, ys) {
  const pairs = xs
    .map((x, i) => ({ x, y: ys[i] }))
    .filter((p) => Number.isFinite(p.x) && Number.isFinite(p.y));

  if (pairs.length < 2) return { slope: NaN, intercept: NaN, r2: NaN };

  const xbar = mean(pairs.map((p) => p.x));
  const ybar = mean(pairs.map((p) => p.y));
  const sxx = pairs.reduce((sum, p) => sum + (p.x - xbar) ** 2, 0);
  const sxy = pairs.reduce((sum, p) => sum + (p.x - xbar) * (p.y - ybar), 0);

  if (sxx === 0) return { slope: NaN, intercept: NaN, r2: NaN };

  const slope = sxy / sxx;
  const intercept = ybar - slope * xbar;
  const sst = pairs.reduce((sum, p) => sum + (p.y - ybar) ** 2, 0);
  const sse = pairs.reduce((sum, p) => sum + (p.y - (slope * p.x + intercept)) ** 2, 0);
  const r2 = sst > 0 ? 1 - sse / sst : NaN;

  return { slope, intercept, r2 };
}

function fitCalibration(points, mode) {
  const pairs = points
    .filter((p) => Number.isFinite(p.kaplan_loss_raw) && Number.isFinite(p.loss))
    .map((p) => ({ x: p.kaplan_loss_raw, y: p.loss }));

  if (pairs.length === 0 || mode === "raw") {
    return { mode: "raw", a: 1, b: 0, label: "raw Kaplan" };
  }

  if (mode === "additive") {
    const b = mean(pairs.map((p) => p.y - p.x));
    return { mode, a: 1, b: Number.isFinite(b) ? b : 0, label: `Kaplan ${b >= 0 ? "+" : ""}${formatLoss(b)}` };
  }

  const xbar = mean(pairs.map((p) => p.x));
  const ybar = mean(pairs.map((p) => p.y));
  const sxx = pairs.reduce((sum, p) => sum + (p.x - xbar) ** 2, 0);
  const sxy = pairs.reduce((sum, p) => sum + (p.x - xbar) * (p.y - ybar), 0);

  if (!Number.isFinite(sxx) || sxx === 0) {
    const b = mean(pairs.map((p) => p.y - p.x));
    return { mode: "additive", a: 1, b: Number.isFinite(b) ? b : 0, label: "additive fallback" };
  }

  const a = sxy / sxx;
  const b = ybar - a * xbar;
  return { mode, a, b, label: `${a.toFixed(3)}·Kaplan ${b >= 0 ? "+" : ""}${formatLoss(b)}` };
}

function applyCalibration(curves, calibration) {
  return curves.map((curve) => ({
    ...curve,
    points: curve.points.map((point) => {
      const kaplanLoss = calibration.a * point.kaplan_loss_raw + calibration.b;
      const residual = point.loss - kaplanLoss;
      const residualPct = safeDivide(100 * residual, Math.abs(kaplanLoss));

      return {
        ...point,
        kaplan_loss: kaplanLoss,
        kaplan_residual: residual,
        kaplan_residual_pct: residualPct,
        abs_residual_pct: Math.abs(residualPct),
      };
    }),
  }));
}

function getXValue(point, xMetric) {
  return point[xMetric];
}

function classifyPoint(point) {
  const ratio = point.D_over_N;
  if (!Number.isFinite(ratio)) return "invalid";
  if (ratio < 8) return "data-limited";
  if (ratio < 15) return "slightly data-limited";
  if (ratio <= 30) return "near D/N≈20";
  if (ratio <= 80) return "model-limited";
  return "very model-limited";
}

function makeCurveDiagnostics(curves, xMetric) {
  return curves.map((curve) => {
    const points = curve.points
      .filter((p) => Number.isFinite(getXValue(p, xMetric)) && getXValue(p, xMetric) > 0)
      .slice()
      .sort((a, b) => getXValue(a, xMetric) - getXValue(b, xMetric));

    const residuals = points.map((p) => p.kaplan_residual);
    const residualPcts = points.map((p) => p.kaplan_residual_pct);
    const xlogs = points.map((p) => Math.log10(getXValue(p, xMetric)));
    const measuredFit = linearFit(xlogs, points.map((p) => p.loss));
    const kaplanFit = linearFit(xlogs, points.map((p) => p.kaplan_loss));
    const residualFit = linearFit(xlogs, residualPcts);
    const first = points[0];
    const last = points[points.length - 1];
    const observedDrop = first && last ? first.loss - last.loss : NaN;
    const expectedDrop = first && last ? first.kaplan_loss - last.kaplan_loss : NaN;
    const gainRatio = safeDivide(observedDrop, expectedDrop);
    const slopeRatio = safeDivide(measuredFit.slope, kaplanFit.slope);
    const best = points.reduce((acc, p) => (!acc || p.loss < acc.loss ? p : acc), null);

    let verdict = "needs more points";
    if (points.length >= 3) {
      const medAbs = median(residualPcts.map(Math.abs));
      if (medAbs < 8 && gainRatio > 0.6 && gainRatio < 1.6) verdict = "Kaplan-like";
      else if (Number.isFinite(gainRatio) && gainRatio < 0.45) verdict = "too flat";
      else if (Number.isFinite(residualFit.slope) && residualFit.slope > 12) verdict = "large-x runs lag";
      else if (Number.isFinite(residualFit.slope) && residualFit.slope < -12) verdict = "large-x runs beat Kaplan";
      else verdict = "offset / shape mismatch";
    }

    return {
      curve: curve.name,
      points: points.length,
      rmse_loss: rmse(residuals),
      median_abs_pct_error: median(residualPcts.map(Math.abs)),
      mean_pct_error: mean(residualPcts),
      measured_slope: measuredFit.slope,
      kaplan_slope: kaplanFit.slope,
      slope_ratio: slopeRatio,
      gain_ratio: gainRatio,
      residual_pct_slope: residualFit.slope,
      best_loss: best?.loss ?? NaN,
      best_N: best?.N ?? NaN,
      best_D: best?.D ?? NaN,
      best_D_over_N: best?.D_over_N ?? NaN,
      verdict,
    };
  });
}

function makeGlobalStats(points) {
  const residuals = points.map((p) => p.kaplan_residual);
  const pct = points.map((p) => p.kaplan_residual_pct);
  const gaps = points.map((p) => p.train_val_gap);
  const best = points.reduce((acc, p) => (!acc || p.loss < acc.loss ? p : acc), null);
  const worstResidual = points.reduce((acc, p) => (!acc || Math.abs(p.kaplan_residual_pct) > Math.abs(acc.kaplan_residual_pct) ? p : acc), null);

  return {
    count: points.length,
    rmseLoss: rmse(residuals),
    medianAbsPct: median(pct.map(Math.abs)),
    meanPct: mean(pct),
    medianGap: median(gaps),
    best,
    worstResidual,
  };
}

function recommendationPriority(text, severity = "info") {
  return { text, severity };
}

function makeRecommendations(points, diagnostics) {
  const recs = [];
  const stats = makeGlobalStats(points);
  const ratios = points.map((p) => p.D_over_N).filter(Number.isFinite);
  const coverages = points.map((p) => p.coverage).filter(Number.isFinite);
  const gaps = points.map((p) => p.train_val_gap).filter(Number.isFinite);
  const tooFlat = diagnostics.filter((row) => row.verdict === "too flat");
  const lagging = diagnostics.filter((row) => row.verdict === "large-x runs lag");
  const modelLimitedFrac = ratios.length ? ratios.filter((r) => r > 80).length / ratios.length : 0;
  const dataLimitedFrac = ratios.length ? ratios.filter((r) => r < 15).length / ratios.length : 0;

  if (points.length < 12) {
    recs.push(recommendationPriority("Add more grid points before trusting the scaling diagnosis. A Kaplan-style comparison is much more stable with at least 3 dataset sizes × 4 model sizes.", "warning"));
  }

  if (Number.isFinite(stats.medianAbsPct) && stats.medianAbsPct > 25) {
    recs.push(recommendationPriority("The calibrated Kaplan residuals are large. First check loss-definition mismatch, tokenizer/vocab differences, validation-set consistency, and whether num_params should be non-embedding parameters.", "warning"));
  }

  if (tooFlat.length > 0) {
    recs.push(recommendationPriority(`Some curves are flatter than Kaplan expects (${tooFlat.map((r) => r.curve).slice(0, 3).join("; ")}). This usually points to undertraining, suboptimal LR schedule, too-narrow N range, or runs not reaching their best checkpoint.`, "warning"));
  }

  if (lagging.length > 0) {
    recs.push(recommendationPriority("Residuals get worse for larger x values. For loss-vs-N sweeps, check larger-model LR, warmup, grad accumulation, batch-size scaling, and architecture bugs that only appear at scale.", "warning"));
  }

  if (dataLimitedFrac > 0.25) {
    recs.push(recommendationPriority("A large fraction of points have D/N below about 15. To look more Kaplan-like, increase dataset size for the large models rather than only scaling N.", "info"));
  }

  if (modelLimitedFrac > 0.25) {
    recs.push(recommendationPriority("A large fraction of points have D/N above about 80. You are likely data-rich relative to model size; add larger N at fixed D to see the expected loss-vs-N bend.", "info"));
  }

  if (coverages.length > 0 && median(coverages) < 0.85) {
    recs.push(recommendationPriority("Median training coverage is below one epoch. If num_train_tokens is a dataset-size column, your runs may be undertrained relative to the D value used in Kaplan L(N,D).", "warning"));
  }

  if (gaps.length > 0 && median(gaps) > 0.08) {
    recs.push(recommendationPriority("The median train/val gap is fairly large. That looks more like overfitting or validation-distribution mismatch than clean Kaplan scaling.", "warning"));
  }

  if (recs.length === 0) {
    recs.push(recommendationPriority("Your curves are reasonably Kaplan-like after calibration. The next improvement is to widen the grid in both N and D so the shape is not determined by a small local region.", "good"));
  }

  return recs.slice(0, 6);
}

function makeNextRunRows(points) {
  const candidates = points
    .filter((p) => Number.isFinite(p.N) && Number.isFinite(p.D) && Number.isFinite(p.D_over_N))
    .slice()
    .sort((a, b) => {
      const aBadness = Math.abs(Math.log((a.D_over_N || NaN) / 20)) + 0.01 * Math.abs(a.kaplan_residual_pct || 0);
      const bBadness = Math.abs(Math.log((b.D_over_N || NaN) / 20)) + 0.01 * Math.abs(b.kaplan_residual_pct || 0);
      return bBadness - aBadness;
    });

  const rows = [];
  const seen = new Set();

  for (const point of candidates) {
    const ratio = point.D_over_N;
    let action = "tune / repeat";
    let suggestedN = point.N;
    let suggestedD = point.D;
    let reason = "Near D/N≈20 but residual is high; repeat with LR/seed/checkpoint sweep.";

    if (ratio < 15) {
      action = "add data";
      suggestedD = 20 * point.N;
      reason = "Data-limited relative to the D/N≈20 reference.";
    } else if (ratio > 50) {
      action = "add model";
      suggestedN = point.D / 20;
      reason = "Data-rich relative to the D/N≈20 reference.";
    }

    suggestedN = Math.max(1, suggestedN);
    suggestedD = Math.max(1, suggestedD);
    const key = `${Math.round(suggestedN)}_${Math.round(suggestedD)}_${action}`;
    if (seen.has(key)) continue;
    seen.add(key);

    rows.push({
      action,
      current_N: point.N,
      current_D: point.D,
      current_D_over_N: ratio,
      suggested_N: suggestedN,
      suggested_D: suggestedD,
      suggested_D_over_N: safeDivide(suggestedD, suggestedN),
      suggested_pf_days: flops_to_pf_days(training_flops(suggestedN, suggestedD)),
      reason,
    });

    if (rows.length >= 8) break;
  }

  return rows;
}

function makeDemoCurves() {
  const datasetSizes = [8.5e9, 17e9, 35e9];
  const modelSizes = [2e7, 3.9e7, 7e7, 1.08e8, 2.09e8];

  return datasetSizes.map((D, curveIndex) => {
    const points = modelSizes.map((N, pointIndex) => {
      const rawKaplan = kaplan_loss_ND(N, D);
      const systematicOffset = 0.075;
      const flattening = curveIndex === 0 ? 0.018 * pointIndex : 0.006 * Math.sin(pointIndex + curveIndex);
      const loss = rawKaplan + systematicOffset + flattening;

      return {
        index: pointIndex,
        name: `demo_${curveIndex}_${pointIndex}`,
        N,
        D,
        D_over_N: tokens_per_parameter(D, N),
        loss,
        train_loss: loss - 0.03,
        train_val_gap: 0.03,
        kaplan_loss_raw: rawKaplan,
        pf_days: flops_to_pf_days(training_flops(N, D)),
        steps: D / (32 * 96 * 1024),
        coverage: 1,
        learning_rate: 3e-4,
      };
    });

    return {
      id: `demo_${curveIndex}`,
      name: `D=${human_num(D)}`,
      color: COLORS[curveIndex % COLORS.length],
      hidden: false,
      points,
    };
  });
}

function HeaderName({ name }) {
  const map = {
    curve: "Curve",
    points: "Points",
    rmse_loss: "RMSE loss",
    median_abs_pct_error: "Median |% error|",
    mean_pct_error: "Mean % error",
    measured_slope: "Measured slope",
    kaplan_slope: "Kaplan slope",
    slope_ratio: "Slope ratio",
    gain_ratio: "Gain ratio",
    residual_pct_slope: "Residual % slope",
    best_loss: "Best loss",
    best_N: "Best N",
    best_D: "Best D",
    best_D_over_N: "Best D/N",
    verdict: "Verdict",
    action: "Action",
    current_N: "Current N",
    current_D: "Current D",
    current_D_over_N: "Current D/N",
    suggested_N: "Suggested N",
    suggested_D: "Suggested D",
    suggested_D_over_N: "Suggested D/N",
    suggested_pf_days: "Suggested PF-days",
    reason: "Reason",
  };

  return map[name] ?? String(name).replaceAll("_", " ").replace(/\b\w/g, (c) => c.toUpperCase());
}

function StatCard({ label, value, sub }) {
  return (
    <div className="rounded-xl border border-zinc-800 bg-zinc-950/80 px-3 py-2">
      <div className="flex items-baseline justify-between gap-3">
        <div className="min-w-0 truncate text-xs uppercase tracking-wide text-zinc-500">{label}</div>
        <div className="shrink-0 text-lg font-semibold text-zinc-100">{value}</div>
      </div>
      {sub && <div className="mt-0.5 truncate text-xs text-zinc-500">{sub}</div>}
    </div>
  );
}

function DataTable({ title, rows }) {
  const keys = Object.keys(rows[0] ?? {});

  return (
    <div className="rounded-2xl border border-zinc-800 bg-zinc-950/80 p-3">
      <h2 className="mb-2 text-base font-semibold text-zinc-100">{title}</h2>
      <div className="overflow-x-auto">
        <table className="w-full text-left text-xs">
          <thead className="border-b border-zinc-800 uppercase tracking-wide text-zinc-500">
            <tr>
              {keys.map((key) => (
                <th key={key} className="whitespace-nowrap px-2 py-1.5"><HeaderName name={key} /></th>
              ))}
            </tr>
          </thead>
          <tbody>
            {rows.length === 0 ? (
              <tr><td className="px-2 py-4 text-zinc-500">No rows yet.</td></tr>
            ) : rows.map((row, i) => (
              <tr key={i} className="border-b border-zinc-900 last:border-0">
                {keys.map((key) => (
                  <td key={key} className="max-w-[420px] whitespace-nowrap px-2 py-1.5 text-zinc-300">
                    {formatTableValue(key, row[key])}
                  </td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}

function ChartShell({ title, subtitle, children, footer, heightClass = "h-80" }) {
  return (
    <div className="rounded-2xl border border-zinc-800 bg-zinc-950/80 p-3">
      <div className="mb-2">
        <h2 className="text-base font-semibold text-zinc-100">{title}</h2>
        {subtitle && <p className="mt-1 text-xs leading-relaxed text-zinc-500">{subtitle}</p>}
      </div>
      <div className={heightClass}>{children}</div>
      {footer && <div className="mt-2 text-xs text-zinc-500">{footer}</div>}
    </div>
  );
}

function SelectControl({ label, value, setValue, options, help }) {
  return (
    <label className="block rounded-xl border border-zinc-800 bg-zinc-950/70 p-3">
      <div className="mb-2 text-sm font-medium text-zinc-100">{label}</div>
      <select
        value={value}
        onChange={(e) => setValue(e.target.value)}
        className="w-full rounded-lg border border-zinc-800 bg-zinc-900 px-3 py-2 text-sm text-blue-200 outline-none transition focus:border-blue-500 focus:ring-1 focus:ring-blue-500"
      >
        {options.map((option) => <option key={option.value} value={option.value}>{option.label}</option>)}
      </select>
      {help && <div className="mt-2 text-xs leading-relaxed text-zinc-500">{help}</div>}
    </label>
  );
}

function TextControl({ label, value, setValue, help, placeholder = "" }) {
  return (
    <label className="block rounded-xl border border-zinc-800 bg-zinc-950/70 p-3">
      <div className="mb-2 text-sm font-medium text-zinc-100">{label}</div>
      <input
        type="text"
        value={value}
        placeholder={placeholder}
        onChange={(event) => setValue(event.target.value)}
        className="w-full rounded-lg border border-zinc-800 bg-zinc-900 px-3 py-2 text-sm text-blue-200 outline-none transition focus:border-blue-500 focus:ring-1 focus:ring-blue-500"
        spellCheck={false}
      />
      {help && <div className="mt-2 text-xs leading-relaxed text-zinc-500">{help}</div>}
    </label>
  );
}

function CurveListPanel({ curves, setCurves }) {
  const toggleCurve = (id) => {
    setCurves((prev) => prev.map((curve) => curve.id === id ? { ...curve, hidden: !curve.hidden } : curve));
  };

  const removeCurve = (id) => {
    setCurves((prev) => prev.filter((curve) => curve.id !== id));
  };

  return (
    <div className="rounded-2xl border border-zinc-800 bg-zinc-950/80 p-3">
      <h2 className="text-sm font-semibold uppercase tracking-wide text-zinc-400">Active curves</h2>
      <div className="mt-2 space-y-2">
        {curves.length === 0 && <div className="rounded-xl border border-zinc-800 bg-zinc-950 p-3 text-xs text-zinc-500">No curves imported yet.</div>}
        {curves.map((curve, index) => (
          <div key={curve.id} className="rounded-xl border border-zinc-800 bg-zinc-950/70 p-3">
            <div className="flex items-start justify-between gap-2">
              <div className="min-w-0">
                <div className="flex items-center gap-2">
                  <span className="h-2.5 w-2.5 shrink-0 rounded-full" style={{ background: curve.color ?? COLORS[index % COLORS.length] }} />
                  <div className="truncate text-sm font-medium text-zinc-100">{curve.name}</div>
                </div>
                <div className="mt-1 text-xs text-zinc-500">{curve.points.length} usable point(s)</div>
              </div>
              <div className="flex shrink-0 gap-1">
                <button type="button" onClick={() => toggleCurve(curve.id)} className="rounded-lg border border-zinc-700 bg-zinc-900 px-2 py-1 text-[11px] text-zinc-300 hover:border-blue-500/60">
                  {curve.hidden ? "Show" : "Hide"}
                </button>
                <button type="button" onClick={() => removeCurve(curve.id)} className="rounded-lg border border-zinc-700 bg-zinc-900 px-2 py-1 text-[11px] text-zinc-300 hover:border-red-500/60">
                  Remove
                </button>
              </div>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}

function plotLayout({ xTitle, yTitle, logX = false, logY = false, shapes = [], legendY = 1.15 }) {
  return {
    autosize: true,
    paper_bgcolor: "rgba(0,0,0,0)",
    plot_bgcolor: "rgba(0,0,0,0)",
    font: { color: "#d4d4d8", size: 12 },
    margin: { l: 60, r: 24, t: 8, b: 54 },
    hovermode: "closest",
    dragmode: "zoom",
    legend: { orientation: "h", x: 0, y: legendY, font: { size: 11, color: "#d4d4d8" } },
    xaxis: {
      title: { text: xTitle, font: { color: "#a1a1aa", size: 12 } },
      type: logX ? "log" : "linear",
      gridcolor: "rgba(255,255,255,0.08)",
      zerolinecolor: "rgba(255,255,255,0.14)",
      linecolor: "#3f3f46",
      tickfont: { color: "#a1a1aa", size: 11 },
      exponentformat: "power",
      showspikes: true,
      spikemode: "across",
      spikedash: "dot",
      spikecolor: "rgba(255,255,255,0.28)",
    },
    yaxis: {
      title: { text: yTitle, font: { color: "#a1a1aa", size: 12 } },
      type: logY ? "log" : "linear",
      gridcolor: "rgba(255,255,255,0.08)",
      zerolinecolor: "rgba(255,255,255,0.14)",
      linecolor: "#3f3f46",
      tickfont: { color: "#a1a1aa", size: 11 },
      exponentformat: "power",
      showspikes: true,
      spikemode: "across",
      spikedash: "dot",
      spikecolor: "rgba(255,255,255,0.28)",
    },
    shapes,
  };
}

function plotConfig(filename) {
  return {
    responsive: true,
    scrollZoom: true,
    displaylogo: false,
    toImageButtonOptions: {
      format: "png",
      filename: sanitizeFilenamePart(filename),
      height: 900,
      width: 1400,
      scale: 2,
    },
  };
}

function PlotCard({ title, subtitle, traces, layout, config }) {
  return (
    <ChartShell title={title} subtitle={subtitle} footer="Drag to zoom. Mouse wheel zoom, pan, autoscale, reset axes, and PNG export are enabled.">
      <Plot
        data={traces}
        layout={layout}
        config={config ?? plotConfig(title)}
        useResizeHandler
        className="h-full w-full"
        style={{ width: "100%", height: "100%" }}
      />
    </ChartShell>
  );
}

function KaplanOverlayChart({ curves, xMetric, logX }) {
  const xLabel = X_METRIC_OPTIONS.find((option) => option.value === xMetric)?.label ?? xMetric;
  const traces = curves.flatMap((curve, index) => {
    const points = curve.points.filter((p) => Number.isFinite(getXValue(p, xMetric)) && getXValue(p, xMetric) > 0);
    const commonText = points.map((p) => [
      `curve: ${curve.name}`,
      `model: ${p.name}`,
      `N: ${human_num(p.N)}`,
      `D: ${human_num(p.D)}`,
      `D/N: ${human_num(p.D_over_N)}`,
      `measured: ${formatLoss(p.loss)}`,
      `Kaplan: ${formatLoss(p.kaplan_loss)}`,
      `residual: ${formatPercent(p.kaplan_residual_pct)}`,
      `coverage: ${Number.isFinite(p.coverage) ? p.coverage.toFixed(2) : "—"}`,
    ].join("<br>"));

    return [
      {
        type: "scatter",
        mode: "lines+markers",
        name: `${curve.name} measured`,
        x: points.map((p) => getXValue(p, xMetric)),
        y: points.map((p) => p.loss),
        text: commonText,
        hovertemplate: "%{text}<extra></extra>",
        line: { color: curve.color ?? COLORS[index % COLORS.length], width: 2.5 },
        marker: { color: curve.color ?? COLORS[index % COLORS.length], size: 7, line: { color: "rgba(255,255,255,0.35)", width: 1 } },
      },
      {
        type: "scatter",
        mode: "lines+markers",
        name: `${curve.name} Kaplan`,
        x: points.map((p) => getXValue(p, xMetric)),
        y: points.map((p) => p.kaplan_loss),
        text: commonText,
        hovertemplate: "%{text}<extra></extra>",
        line: { color: curve.color ?? COLORS[index % COLORS.length], width: 2, dash: "dash" },
        marker: { color: curve.color ?? COLORS[index % COLORS.length], size: 5, symbol: "diamond" },
      },
    ];
  });

  return (
    <PlotCard
      title="Measured curves vs Kaplan expectation"
      subtitle="Solid lines are your imported validation losses. Dashed lines are Kaplan L(N,D), optionally calibrated by the selected global offset/map."
      traces={traces}
      layout={plotLayout({ xTitle: xLabel, yTitle: "Validation loss", logX })}
    />
  );
}

function ResidualChart({ curves, xMetric, logX }) {
  const xLabel = X_METRIC_OPTIONS.find((option) => option.value === xMetric)?.label ?? xMetric;
  const traces = curves.map((curve, index) => {
    const points = curve.points.filter((p) => Number.isFinite(getXValue(p, xMetric)) && getXValue(p, xMetric) > 0 && Number.isFinite(p.kaplan_residual_pct));
    return {
      type: "scatter",
      mode: "lines+markers",
      name: curve.name,
      x: points.map((p) => getXValue(p, xMetric)),
      y: points.map((p) => p.kaplan_residual_pct),
      text: points.map((p) => [
        `curve: ${curve.name}`,
        `model: ${p.name}`,
        `measured: ${formatLoss(p.loss)}`,
        `Kaplan: ${formatLoss(p.kaplan_loss)}`,
        `residual: ${formatLoss(p.kaplan_residual)}`,
        `percent: ${formatPercent(p.kaplan_residual_pct)}`,
      ].join("<br>")),
      hovertemplate: "%{text}<extra></extra>",
      line: { color: curve.color ?? COLORS[index % COLORS.length], width: 2.4 },
      marker: { color: curve.color ?? COLORS[index % COLORS.length], size: 7, line: { color: "rgba(255,255,255,0.35)", width: 1 } },
    };
  });

  return (
    <PlotCard
      title="Kaplan residuals"
      subtitle="Positive means your run is worse than Kaplan predicts after calibration; negative means better than predicted. Slope in this plot is more important than absolute offset."
      traces={traces}
      layout={plotLayout({
        xTitle: xLabel,
        yTitle: "Measured − Kaplan (%)",
        logX,
        shapes: [{ type: "line", xref: "paper", x0: 0, x1: 1, yref: "y", y0: 0, y1: 0, line: { color: "#fbbf24", width: 2, dash: "dash" } }],
      })}
    />
  );
}

function ParityChart({ curves }) {
  const all = curves.flatMap((curve) => curve.points.map((point) => ({ ...point, curve: curve.name, color: curve.color })));
  const values = all.flatMap((p) => [p.kaplan_loss, p.loss]).filter(Number.isFinite);
  const lo = values.length ? Math.min(...values) : 0;
  const hi = values.length ? Math.max(...values) : 1;
  const pad = (hi - lo) * 0.08 || 0.05;

  const traces = curves.map((curve, index) => ({
    type: "scatter",
    mode: "markers",
    name: curve.name,
    x: curve.points.map((p) => p.kaplan_loss),
    y: curve.points.map((p) => p.loss),
    text: curve.points.map((p) => [
      `curve: ${curve.name}`,
      `model: ${p.name}`,
      `N: ${human_num(p.N)}`,
      `D: ${human_num(p.D)}`,
      `measured: ${formatLoss(p.loss)}`,
      `Kaplan: ${formatLoss(p.kaplan_loss)}`,
      `residual: ${formatPercent(p.kaplan_residual_pct)}`,
    ].join("<br>")),
    hovertemplate: "%{text}<extra></extra>",
    marker: { color: curve.color ?? COLORS[index % COLORS.length], size: 8, line: { color: "rgba(255,255,255,0.35)", width: 1 } },
  }));

  traces.push({
    type: "scatter",
    mode: "lines",
    name: "perfect agreement",
    x: [lo - pad, hi + pad],
    y: [lo - pad, hi + pad],
    hoverinfo: "skip",
    line: { color: "#fbbf24", width: 2, dash: "dash" },
  });

  return (
    <PlotCard
      title="Parity: measured vs Kaplan"
      subtitle="Points on the dashed line match Kaplan after calibration. Curved or fan-shaped structure means your scaling shape differs from Kaplan."
      traces={traces}
      layout={{
        ...plotLayout({ xTitle: "Kaplan predicted loss", yTitle: "Measured validation loss" }),
        xaxis: { ...plotLayout({ xTitle: "Kaplan predicted loss", yTitle: "Measured validation loss" }).xaxis, range: [lo - pad, hi + pad] },
        yaxis: { ...plotLayout({ xTitle: "Kaplan predicted loss", yTitle: "Measured validation loss" }).yaxis, range: [lo - pad, hi + pad], scaleanchor: "x", scaleratio: 1 },
      }}
    />
  );
}

function DNMapChart({ curves }) {
  const all = curves.flatMap((curve) => curve.points.map((point) => ({ ...point, curve: curve.name, color: curve.color })));
  const ns = all.map((p) => p.N).filter((x) => Number.isFinite(x) && x > 0);
  const loN = ns.length ? Math.min(...ns) * 0.8 : 1e6;
  const hiN = ns.length ? Math.max(...ns) * 1.25 : 1e9;
  const lineNs = [loN, hiN];

  const traces = curves.map((curve, index) => ({
    type: "scatter",
    mode: "markers",
    name: curve.name,
    x: curve.points.map((p) => p.N),
    y: curve.points.map((p) => p.D),
    text: curve.points.map((p) => [
      `curve: ${curve.name}`,
      `model: ${p.name}`,
      `N: ${human_num(p.N)}`,
      `D: ${human_num(p.D)}`,
      `D/N: ${human_num(p.D_over_N)}`,
      `regime: ${classifyPoint(p)}`,
    ].join("<br>")),
    hovertemplate: "%{text}<extra></extra>",
    marker: { color: curve.color ?? COLORS[index % COLORS.length], size: 8, line: { color: "rgba(255,255,255,0.35)", width: 1 } },
  }));

  for (const [ratio, dash, label] of [[8, "dot", "D/N=8"], [20, "dash", "D/N=20"], [80, "dot", "D/N=80"]]) {
    traces.push({
      type: "scatter",
      mode: "lines",
      name: label,
      x: lineNs,
      y: lineNs.map((n) => ratio * n),
      hoverinfo: "skip",
      line: { color: ratio === 20 ? "#fbbf24" : "#a1a1aa", width: ratio === 20 ? 2.4 : 1.6, dash },
    });
  }

  return (
    <PlotCard
      title="D/N coverage map"
      subtitle="This shows whether your grid actually covers the Kaplan-style model-size/data-size tradeoff instead of sitting in only one regime."
      traces={traces}
      layout={plotLayout({ xTitle: "Model size N", yTitle: "Dataset size D", logX: true, logY: true })}
    />
  );
}

function GainRatioChart({ rows }) {
  const traces = [{
    type: "bar",
    name: "observed / expected gain",
    x: rows.map((row) => row.curve),
    y: rows.map((row) => row.gain_ratio),
    text: rows.map((row) => row.verdict),
    hovertemplate: "curve: %{x}<br>gain ratio: %{y:.3f}<br>%{text}<extra></extra>",
    marker: { color: rows.map((row) => row.gain_ratio >= 0.6 && row.gain_ratio <= 1.6 ? "#34d399" : "#fb7185") },
  }];

  return (
    <PlotCard
      title="Shape score: observed gain / Kaplan gain"
      subtitle="Near 1 means the curve improves by about as much as Kaplan expects over the plotted x-range. Much below 1 means the curve is too flat."
      traces={traces}
      layout={plotLayout({
        xTitle: "Curve",
        yTitle: "Gain ratio",
        shapes: [{ type: "line", xref: "paper", x0: 0, x1: 1, yref: "y", y0: 1, y1: 1, line: { color: "#fbbf24", width: 2, dash: "dash" } }],
        legendY: 1.05,
      })}
    />
  );
}

function RecommendationPanel({ recommendations }) {
  const classes = {
    good: "border-emerald-500/30 bg-emerald-500/10 text-emerald-100",
    info: "border-blue-500/30 bg-blue-500/10 text-blue-100",
    warning: "border-amber-500/30 bg-amber-500/10 text-amber-100",
  };

  return (
    <div className="rounded-2xl border border-zinc-800 bg-zinc-950/80 p-3">
      <h2 className="text-base font-semibold text-zinc-100">How to make this more Kaplan-like</h2>
      <div className="mt-3 space-y-2">
        {recommendations.map((rec, index) => (
          <div key={index} className={`rounded-xl border px-3 py-2 text-sm leading-relaxed ${classes[rec.severity] ?? classes.info}`}>
            {rec.text}
          </div>
        ))}
      </div>
    </div>
  );
}

function ImportPanel({ textValue, setTextValue, groupColumns, setGroupColumns, onImport, importStatus }) {
  const readFile = (file) => {
    if (!file) return;
    const reader = new FileReader();
    reader.onload = () => {
      const nextText = String(reader.result ?? "");
      setTextValue(nextText);
      onImport(nextText);
    };
    reader.readAsText(file);
  };

  return (
    <div className="rounded-2xl border border-zinc-800 bg-zinc-950/80 p-3">
      <div className="mb-2">
        <h2 className="text-sm font-semibold uppercase tracking-wide text-zinc-400">Import run dataframe</h2>
        <p className="mt-1 text-xs leading-relaxed text-zinc-500">
          Paste <code className="text-zinc-300">df.to_csv(index=False)</code>. Required columns are <code className="text-zinc-300">num_train_tokens</code>, <code className="text-zinc-300">num_params</code>, and <code className="text-zinc-300">min_saved_val_loss</code>.
        </p>
      </div>

      <TextControl
        label="Curve group columns"
        value={groupColumns}
        setValue={setGroupColumns}
        placeholder="num_train_tokens"
        help="Use num_train_tokens for Kaplan-style loss-vs-N curves. Add learning_rate if you want separate curves for LR sweeps. Leave blank for one curve."
      />

      <textarea
        value={textValue}
        onChange={(event) => setTextValue(event.target.value)}
        placeholder={"model_name,num_train_tokens,n_layer,n_head,n_embd,num_params,iters_saved,min_saved_train_loss,min_saved_val_loss,coverage,tokens_per_update\nmodel_a,8500000000,4,4,256,20000000,12000,1.82,1.89,1.0,3145728"}
        className="mt-3 h-56 w-full resize-y rounded-xl border border-zinc-800 bg-zinc-950 p-3 text-xs leading-relaxed text-zinc-300 outline-none transition placeholder:text-zinc-700 focus:border-blue-500 focus:ring-1 focus:ring-blue-500"
        spellCheck={false}
      />

      <div className="mt-2 grid grid-cols-1 gap-2">
        <input
          type="file"
          accept="text/csv,.csv,text/plain,.txt"
          onChange={(event) => readFile(event.target.files?.[0])}
          className="block min-w-0 text-xs text-zinc-500 file:mr-3 file:rounded-lg file:border file:border-zinc-700 file:bg-zinc-900 file:px-3 file:py-1.5 file:text-xs file:font-semibold file:text-zinc-200 hover:file:border-blue-500/60"
        />

        <button
          type="button"
          onClick={() => onImport(textValue)}
          disabled={!textValue.trim()}
          className="rounded-xl border border-blue-500/40 bg-blue-500/15 px-4 py-2 text-sm font-semibold text-blue-100 transition hover:border-blue-400 hover:bg-blue-500/25 disabled:cursor-not-allowed disabled:opacity-50"
        >
          Replace with pasted CSV
        </button>
      </div>

      {importStatus && (
        <div className={`mt-2 rounded-xl border px-3 py-2 text-xs leading-relaxed ${importStatus.type === "success" ? "border-emerald-500/30 bg-emerald-500/10 text-emerald-200" : "border-amber-500/30 bg-amber-500/10 text-amber-200"}`}>
          {importStatus.text}
        </div>
      )}
    </div>
  );
}

function CsvFormatPanel() {
  return (
    <details className="rounded-2xl border border-zinc-800 bg-zinc-950/80 p-3">
      <summary className="cursor-pointer text-sm font-semibold uppercase tracking-wide text-zinc-400">What this page computes</summary>
      <div className="mt-2 space-y-2 text-xs leading-relaxed text-zinc-500">
        <p>
          For each row, the page computes Kaplan's expected loss from <code className="text-zinc-300">kaplan_loss_ND(num_params, num_train_tokens)</code>, estimates compute with <code className="text-zinc-300">C≈6ND</code>, and compares your measured validation loss against that prediction.
        </p>
        <p>
          The most important plots are the residual plot and the gain-ratio plot. Absolute offsets can come from your tokenizer, data distribution, or loss definition; shape mismatch is the stronger signal that the sweep is not behaving like a Kaplan scaling curve.
        </p>
      </div>
    </details>
  );
}

export default function KaplanComparisonPage() {
  const [textValue, setTextValue] = useState("");
  const [groupColumns, setGroupColumns] = useState(DEFAULT_GROUP_COLUMNS);
  const [curves, setCurves] = useState(() => makeDemoCurves());
  const [xMetric, setXMetric] = useState("N");
  const [calibrationMode, setCalibrationMode] = useState("additive");
  const [useLogX, setUseLogX] = useState(true);
  const [importStatus, setImportStatus] = useState(null);

  const handleImport = (sourceText) => {
    try {
      const rows = parseCsvText(sourceText);
      const nextCurves = csvRowsToCurves(rows, groupColumns);
      const pointCount = nextCurves.reduce((sum, curve) => sum + curve.points.length, 0);

      if (nextCurves.length === 0 || pointCount === 0) {
        throw new Error("Import produced zero usable points. Check num_train_tokens, num_params, and min_saved_val_loss columns.");
      }

      setCurves(nextCurves);
      setImportStatus({ type: "success", text: `Imported ${nextCurves.length} curve(s), ${pointCount} usable point(s).` });
    } catch (error) {
      setImportStatus({ type: "error", text: error?.message ?? "CSV import failed." });
    }
  };

  const visibleRawCurves = useMemo(() => curves.filter((curve) => !curve.hidden && curve.points.length > 0), [curves]);
  const allRawPoints = useMemo(() => visibleRawCurves.flatMap((curve) => curve.points), [visibleRawCurves]);
  const calibration = useMemo(() => fitCalibration(allRawPoints, calibrationMode), [allRawPoints, calibrationMode]);
  const visibleCurves = useMemo(() => applyCalibration(visibleRawCurves, calibration), [visibleRawCurves, calibration]);
  const allPoints = useMemo(() => visibleCurves.flatMap((curve) => curve.points.map((point) => ({ ...point, curve: curve.name, curveColor: curve.color }))), [visibleCurves]);
  const globalStats = useMemo(() => makeGlobalStats(allPoints), [allPoints]);
  const diagnosticRows = useMemo(() => makeCurveDiagnostics(visibleCurves, xMetric), [visibleCurves, xMetric]);
  const recommendations = useMemo(() => makeRecommendations(allPoints, diagnosticRows), [allPoints, diagnosticRows]);
  const nextRunRows = useMemo(() => makeNextRunRows(allPoints), [allPoints]);

  const kaplanLikeCount = diagnosticRows.filter((row) => row.verdict === "Kaplan-like").length;
  const medianDN = median(allPoints.map((p) => p.D_over_N));

  return (
    <div className="min-h-screen bg-zinc-950 text-zinc-100">
      <div className="min-h-screen lg:flex lg:items-stretch">
        <aside className="border-b border-zinc-800 bg-zinc-950/95 p-5 lg:sticky lg:top-0 lg:h-screen lg:w-[440px] lg:shrink-0 lg:overflow-y-auto lg:border-b-0 lg:border-r">
          <div className="mb-6">
            <h1 className="text-2xl font-semibold tracking-tight">Kaplan comparison</h1>
            <p className="mt-2 text-sm leading-relaxed text-zinc-500">
              Paste your run dataframe, compare measured losses to Kaplan's expected curve shape, and get next-run suggestions.
            </p>
          </div>

          <div className="space-y-6">
            <ImportPanel
              textValue={textValue}
              setTextValue={setTextValue}
              groupColumns={groupColumns}
              setGroupColumns={setGroupColumns}
              onImport={handleImport}
              importStatus={importStatus}
            />

            <section className="space-y-3">
              <h2 className="text-sm font-semibold uppercase tracking-wide text-zinc-400">Diagnostics</h2>
              <SelectControl
                label="X-axis / sweep direction"
                value={xMetric}
                setValue={setXMetric}
                options={X_METRIC_OPTIONS}
                help="For classic Kaplan loss-vs-model-size curves, use N and group by num_train_tokens."
              />
              <SelectControl
                label="Kaplan calibration"
                value={calibrationMode}
                setValue={setCalibrationMode}
                options={CALIBRATION_OPTIONS}
                help="Additive or affine calibration removes global loss-definition offsets so the page can judge shape instead of absolute scale."
              />
              <label className="flex cursor-pointer items-start justify-between gap-4 rounded-xl border border-zinc-800 bg-zinc-950/70 p-4">
                <div>
                  <div className="text-sm font-medium text-zinc-100">Use logarithmic x-axis</div>
                  <div className="mt-1 text-xs text-zinc-500">Recommended for N, D, PF-days, and D/N.</div>
                </div>
                <input type="checkbox" checked={useLogX} onChange={(e) => setUseLogX(e.target.checked)} className="mt-1 h-4 w-4 accent-blue-500" />
              </label>
            </section>

            <CurveListPanel curves={curves} setCurves={setCurves} />

            <div className="grid grid-cols-2 gap-2">
              <button type="button" onClick={() => setCurves(makeDemoCurves())} className="rounded-xl border border-zinc-700 bg-zinc-900 px-4 py-2 text-sm font-semibold text-zinc-200 transition hover:border-blue-500/60 hover:text-blue-100">
                Load demo
              </button>
              <button type="button" onClick={() => setCurves([])} className="rounded-xl border border-zinc-700 bg-zinc-900 px-4 py-2 text-sm font-semibold text-zinc-200 transition hover:border-red-500/60 hover:text-red-100">
                Clear
              </button>
            </div>

            <CsvFormatPanel />
          </div>
        </aside>

        <main className="min-w-0 flex-1 space-y-3 p-4 lg:p-5">
          <div className="flex flex-col gap-2 sm:flex-row sm:items-end sm:justify-between">
            <div>
              <h1 className="text-3xl font-semibold tracking-tight">Kaplan-style scaling-law diagnostics</h1>
              <p className="mt-2 text-sm text-zinc-500">
                {visibleCurves.length} active curve(s), {human_num(allPoints.length)} usable point(s). Calibration: <span className="text-blue-300">{calibration.label}</span>.
              </p>
            </div>

            {globalStats.best && (
              <div className="rounded-xl border border-blue-500/30 bg-blue-500/10 px-4 py-2 text-sm text-blue-100">
                Best loss: {formatLoss(globalStats.best.loss)} · D/N={human_num(globalStats.best.D_over_N)}
              </div>
            )}
          </div>

          <section className="grid grid-cols-1 gap-2 md:grid-cols-2 xl:grid-cols-4">
            <StatCard label="Median |Kaplan error|" value={formatPercent(globalStats.medianAbsPct, 1).replace("+", "")} sub="after selected calibration" />
            <StatCard label="Residual RMSE" value={formatLoss(globalStats.rmseLoss)} sub="measured − Kaplan" />
            <StatCard label="Kaplan-like curves" value={`${kaplanLikeCount}/${diagnosticRows.length}`} sub="based on shape + residual" />
            <StatCard label="Median D/N" value={human_num(medianDN)} sub="D/N≈20 is a useful reference" />
          </section>

          {visibleCurves.length === 0 ? (
            <div className="rounded-2xl border border-zinc-800 bg-zinc-950/80 p-8 text-center text-sm text-zinc-500">
              Paste df.to_csv(index=False) output to begin.
            </div>
          ) : (
            <>
              <section className="grid grid-cols-1 gap-3 xl:grid-cols-2">
                <KaplanOverlayChart curves={visibleCurves} xMetric={xMetric} logX={useLogX} />
                <ResidualChart curves={visibleCurves} xMetric={xMetric} logX={useLogX} />
              </section>

              <section className="grid grid-cols-1 gap-3 xl:grid-cols-2">
                <ParityChart curves={visibleCurves} />
                <DNMapChart curves={visibleCurves} />
              </section>

              <section className="grid grid-cols-1 gap-3 xl:grid-cols-2">
                <GainRatioChart rows={diagnosticRows} />
                <RecommendationPanel recommendations={recommendations} />
              </section>

              <section className="grid grid-cols-1 gap-3">
                <DataTable title="Curve diagnostics" rows={diagnosticRows} />
                <DataTable title="Suggested next runs" rows={nextRunRows} />
              </section>

              <div className="rounded-2xl border border-zinc-800 bg-zinc-950/80 p-3 text-xs leading-relaxed text-zinc-500">
                <div className="font-semibold uppercase tracking-wide text-zinc-400">Interpretation notes</div>
                <p className="mt-1">
                  Treat the raw Kaplan prediction as a reference curve, not as an absolute truth for your particleGPT data distribution. The additive/affine calibration is intentionally global: if calibration fixes the vertical offset but the residual still slopes with N, D, or compute, then the shape of your sweep differs from Kaplan.
                </p>
                <p className="mt-1">
                  The gain ratio is the simplest shape diagnostic: it compares the measured loss improvement across a curve to the Kaplan-predicted improvement across the same points. Values much below one mean your curve is flatter than expected; values near one mean the sweep shape is Kaplan-like over that range.
                </p>
              </div>
            </>
          )}
        </main>
      </div>
    </div>
  );
}
