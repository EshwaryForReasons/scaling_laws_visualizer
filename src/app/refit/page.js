"use client";

import { useMemo, useState } from "react";
import dynamic from "next/dynamic";

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

const DEFAULT_GROUP_COLUMNS = "num_train_tokens";
const FLOPS_PER_TOKEN_FACTOR = 6;

const KAPLAN_REFERENCE = {
  label: "Kaplan et al. 2020",
  equation: "L(N,D) = [(N_c/N)^(alpha_N/alpha_D) + D_c/D]^alpha_D",
  alpha_N: 0.076,
  alpha_D: 0.095,
  N_c: 8.8e13,
  D_c: 5.4e13,
  alpha_C: 0.057,
  C_c_pf_days: 1.6e7,
  alpha_C_min: 0.050,
  C_c_min_pf_days: 3.1e8,
};

const LOSS_WEIGHTING_OPTIONS = [
  { value: "absolute", label: "Absolute loss residuals" },
  { value: "relative", label: "Relative / percent residuals" },
  { value: "log", label: "Log-loss residuals" },
];

const FIT_MODEL_OPTIONS = [
  { value: "full", label: "Full Kaplan: E + A N^-α + B D^-β" },
  { value: "noE", label: "No irreducible term: A N^-α + B D^-β" },
];

function clamp(x, lo, hi) {
  return Math.max(lo, Math.min(hi, x));
}

function safeDivide(a, b) {
  return Number.isFinite(a) && Number.isFinite(b) && b !== 0 ? a / b : NaN;
}

function sanitizeFilenamePart(value) {
  return String(value ?? "")
    .trim()
    .replace(/[^a-zA-Z0-9._-]+/g, "_")
    .replace(/^_+|_+$/g, "") || "kaplan_refit_chart";
}

function stableId(text, index = 0) {
  return `${String(text ?? "curve")
    .trim()
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, "_")
    .replace(/^_+|_+$/g, "") || "curve"}_${index}`;
}

function log10(x) {
  return Math.log(x) / Math.LN10;
}

function logspace(lo, hi, n) {
  if (!Number.isFinite(lo) || !Number.isFinite(hi) || lo <= 0 || hi <= 0 || n <= 1) return [];
  const a = log10(lo);
  const b = log10(hi);
  return Array.from({ length: n }, (_, i) => 10 ** (a + (b - a) * i / (n - 1)));
}

function human_num(x, digits = 3) {
  if (!Number.isFinite(x)) return "—";
  const sign = x < 0 ? "-" : "";
  const abs = Math.abs(x);
  if (abs === 0) return "0";
  const units = [
    [1e15, "P"],
    [1e12, "T"],
    [1e9, "B"],
    [1e6, "M"],
    [1e3, "k"],
  ];
  for (const [scale, suffix] of units) {
    if (abs >= scale) {
      const value = abs / scale;
      const fixed = value >= 100 ? 0 : value >= 10 ? 1 : digits - 1;
      return `${sign}${value.toFixed(fixed).replace(/\.0+$/, "")}${suffix}`;
    }
  }
  if (abs >= 1) return `${sign}${abs.toFixed(abs >= 100 ? 0 : abs >= 10 ? 1 : 2).replace(/\.0+$/, "")}`;
  return `${sign}${abs.toExponential(2)}`;
}

function formatSci(x, digits = 3) {
  if (!Number.isFinite(x)) return "—";
  return Number(x).toExponential(digits);
}

function formatLoss(x) {
  if (!Number.isFinite(x)) return "—";
  return Number(x).toFixed(5);
}

function formatSmall(x) {
  if (!Number.isFinite(x)) return "—";
  if (Math.abs(x) >= 1e3 || Math.abs(x) < 1e-3) return formatSci(x, 4);
  return Number(x).toFixed(6).replace(/0+$/, "").replace(/\.$/, "");
}

function formatPercent(x, digits = 1) {
  if (!Number.isFinite(x)) return "—";
  return `${x >= 0 ? "+" : ""}${x.toFixed(digits)}%`;
}

function formatTableValue(key, value) {
  if (value === null || value === undefined || Number.isNaN(value)) return "—";

  if (typeof value === "number") {
    const lower = key.toLowerCase();
    if (lower.includes("loss") || lower === "e") return formatLoss(value);
    if (lower.includes("pct") || lower.includes("percent") || lower.includes("mape")) return formatPercent(value);
    if (lower.includes("r2")) return Number.isFinite(value) ? value.toFixed(4) : "—";
    if (["alpha", "beta", "exponent"].some((s) => lower.includes(s))) return Number.isFinite(value) ? value.toFixed(5) : "—";
    if (["a", "b", "nc", "dc"].includes(lower) || lower.includes("coefficient")) return formatSmall(value);
    if (lower.includes("ratio") || lower.includes("d_over_n")) return Number.isFinite(value) ? human_num(value) : "—";
    if (lower.includes("pf")) return Number.isFinite(value) ? value.toFixed(4) : "—";
    if (lower.includes("flop")) return formatSci(value, 3);
    return human_num(value);
  }

  return String(value);
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
    /(tokens?|parameters?|params?|sequences?|seqs?|steps?|gpus?|gpu|pf-days?|pfday|pflops?|flops?|layers?|heads?|width|seconds?|sec|iters?|s)$/i,
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
    mm: 1e6,
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

function trainingFlops(N, D) {
  return FLOPS_PER_TOKEN_FACTOR * N * D;
}

function flopsToPfDays(flops) {
  return safeDivide(flops, 1e15 * 86400);
}

function pfDaysToFlops(pfDays) {
  return pfDays * 1e15 * 86400;
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

  const explicitFlops = numberFromAny(rowValue(rawRow, "flops"), rowValue(rawRow, "training_flops"), rowValue(rawRow, "compute_flops"));
  const explicitPfDays = numberFromAny(rowValue(rawRow, "pf_days"), rowValue(rawRow, "compute_pf_days"));
  const flops = Number.isFinite(explicitFlops) ? explicitFlops : (Number.isFinite(N) && Number.isFinite(D) ? trainingFlops(N, D) : NaN);
  const pfDays = Number.isFinite(explicitPfDays) ? explicitPfDays : flopsToPfDays(flops);

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
    D_over_N: Number.isFinite(N) && Number.isFinite(D) ? safeDivide(D, N) : NaN,
    loss: measuredLoss,
    train_loss: trainLoss,
    train_val_gap: Number.isFinite(measuredLoss) && Number.isFinite(trainLoss) ? measuredLoss - trainLoss : NaN,
    flops,
    pf_days: pfDays,
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
    .filter((point) => Number.isFinite(point.N) && point.N > 0 && Number.isFinite(point.D) && point.D > 0 && Number.isFinite(point.loss))
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

function weightedError(pred, actual, weighting) {
  if (!Number.isFinite(pred) || !Number.isFinite(actual)) return NaN;
  if (weighting === "relative") return safeDivide(pred - actual, Math.max(Math.abs(actual), 1e-12));
  if (weighting === "log") return Number.isFinite(pred) && pred > 0 && actual > 0 ? Math.log(pred) - Math.log(actual) : NaN;
  return pred - actual;
}

function predictLoss(model, N, D) {
  if (!model || !Number.isFinite(N) || !Number.isFinite(D) || N <= 0 || D <= 0) return NaN;
  return model.E + model.A * N ** (-model.alpha) + model.B * D ** (-model.beta);
}

function coefficientToCriticalScale(coefficient, exponent) {
  if (!Number.isFinite(coefficient) || !Number.isFinite(exponent) || coefficient <= 0 || exponent <= 0) return NaN;
  return coefficient ** (1 / exponent);
}

function kaplanReferenceLossND(N, D) {
  if (!Number.isFinite(N) || !Number.isFinite(D) || N <= 0 || D <= 0) return NaN;
  const r = KAPLAN_REFERENCE;
  return ((r.N_c / N) ** (r.alpha_N / r.alpha_D) + (r.D_c / D)) ** r.alpha_D;
}

function percentDifference(value, reference) {
  return safeDivide(100 * (value - reference), Math.abs(reference));
}

function compareNote(value, reference, units = "") {
  if (!Number.isFinite(value) || !Number.isFinite(reference) || reference === 0) return "Not directly comparable";
  const ratio = value / reference;
  if (ratio > 1.5) return `Higher than Kaplan${units ? ` ${units}` : ""}`;
  if (ratio < 0.67) return `Lower than Kaplan${units ? ` ${units}` : ""}`;
  return `Same order as Kaplan${units ? ` ${units}` : ""}`;
}

function formatReferenceConstant(name, value) {
  if (!Number.isFinite(value)) return "—";
  const lower = String(name).toLowerCase();
  if (lower.includes("alpha") || lower.includes("exponent") || lower.includes("ratio")) return value.toFixed(5);
  if (lower.includes("loss") || lower === "e") return formatLoss(value);
  if (lower.includes("pf-days")) return human_num(value);
  return human_num(value);
}

function solveNonNegative2(x1s, x2s, ys, weights = []) {
  let s11 = 0;
  let s22 = 0;
  let s12 = 0;
  let t1 = 0;
  let t2 = 0;

  for (let i = 0; i < ys.length; i++) {
    const x1 = x1s[i];
    const x2 = x2s[i];
    const y = ys[i];
    const w = Number.isFinite(weights[i]) && weights[i] > 0 ? weights[i] : 1;
    if (![x1, x2, y].every(Number.isFinite)) continue;
    s11 += w * x1 * x1;
    s22 += w * x2 * x2;
    s12 += w * x1 * x2;
    t1 += w * x1 * y;
    t2 += w * x2 * y;
  }

  const candidates = [];
  const pushCandidate = (A, B) => {
    if (!Number.isFinite(A) || !Number.isFinite(B)) return;
    A = Math.max(0, A);
    B = Math.max(0, B);
    let sse = 0;
    for (let i = 0; i < ys.length; i++) {
      const w = Number.isFinite(weights[i]) && weights[i] > 0 ? weights[i] : 1;
      const err = A * x1s[i] + B * x2s[i] - ys[i];
      if (Number.isFinite(err)) sse += w * err * err;
    }
    candidates.push({ A, B, sse });
  };

  const det = s11 * s22 - s12 * s12;
  if (Math.abs(det) > 1e-300) {
    pushCandidate((t1 * s22 - t2 * s12) / det, (s11 * t2 - s12 * t1) / det);
  }

  pushCandidate(s11 > 0 ? t1 / s11 : 0, 0);
  pushCandidate(0, s22 > 0 ? t2 / s22 : 0);
  pushCandidate(0, 0);

  return candidates.sort((a, b) => a.sse - b.sse)[0] ?? { A: NaN, B: NaN, sse: Infinity };
}

function fitCoefficientsForExponents(points, E, alpha, beta, weighting) {
  const x1s = [];
  const x2s = [];
  const ys = [];
  const weights = [];

  for (const p of points) {
    const y = p.loss - E;
    if (!Number.isFinite(y) || y <= 0) continue;
    const x1 = p.N ** (-alpha);
    const x2 = p.D ** (-beta);
    if (![x1, x2].every(Number.isFinite)) continue;
    x1s.push(x1);
    x2s.push(x2);
    ys.push(y);

    if (weighting === "relative" || weighting === "log") weights.push(1 / Math.max(p.loss * p.loss, 1e-12));
    else weights.push(1);
  }

  return solveNonNegative2(x1s, x2s, ys, weights);
}

function scoreModel(points, model, weighting = "absolute") {
  const errors = [];
  const rawResiduals = [];
  const pctResiduals = [];

  for (const p of points) {
    const pred = predictLoss(model, p.N, p.D);
    const weighted = weightedError(pred, p.loss, weighting);
    if (Number.isFinite(weighted)) errors.push(weighted);
    if (Number.isFinite(pred)) {
      rawResiduals.push(p.loss - pred);
      pctResiduals.push(safeDivide(100 * (p.loss - pred), Math.abs(pred)));
    }
  }

  const ybar = mean(points.map((p) => p.loss));
  const sst = points.reduce((sum, p) => sum + (p.loss - ybar) ** 2, 0);
  const sse = rawResiduals.reduce((sum, e) => sum + e * e, 0);

  return {
    objective: errors.reduce((sum, e) => sum + e * e, 0) / Math.max(errors.length, 1),
    rmse: rmse(rawResiduals),
    mae: mean(rawResiduals.map(Math.abs)),
    median_abs_pct: median(pctResiduals.map(Math.abs)),
    mean_pct: mean(pctResiduals),
    max_abs_pct: (() => {
      const vals = pctResiduals.map((x) => Math.abs(x)).filter(Number.isFinite);
      return vals.length ? Math.max(...vals) : NaN;
    })(),
    r2: sst > 0 ? 1 - sse / sst : NaN,
  };
}

function makeModelFromEBasis(points, E, alpha, beta, weighting) {
  const coeffs = fitCoefficientsForExponents(points, E, alpha, beta, weighting);
  const model = {
    E,
    A: coeffs.A,
    alpha,
    B: coeffs.B,
    beta,
  };
  const score = scoreModel(points, model, weighting);
  return {
    ...model,
    objective: score.objective,
    score,
    N_c: coefficientToCriticalScale(model.A, model.alpha),
    D_c: coefficientToCriticalScale(model.B, model.beta),
  };
}

function estimateParameterWarnings(points) {
  const uniqueN = new Set(points.map((p) => Math.round(log10(p.N) * 1000) / 1000));
  const uniqueD = new Set(points.map((p) => Math.round(log10(p.D) * 1000) / 1000));
  const warnings = [];

  if (points.length < 8) warnings.push("Few points. Treat fitted exponents as directional, not final constants.");
  if (uniqueN.size < 3) warnings.push("Need at least three distinct model sizes for a stable N exponent.");
  if (uniqueD.size < 3) warnings.push("Need at least three distinct dataset sizes for a stable D exponent.");
  if (points.length < 6) warnings.push("The five-parameter fit is underconstrained unless you fix E or add more runs.");

  return warnings;
}

function fitKaplanLikeModel(points, options = {}) {
  const weighting = options.weighting ?? "absolute";
  const fitModel = options.fitModel ?? "full";
  const clean = points.filter((p) => Number.isFinite(p.N) && p.N > 0 && Number.isFinite(p.D) && p.D > 0 && Number.isFinite(p.loss));

  if (clean.length < 4) {
    return {
      ok: false,
      reason: "Need at least four usable points with N, D, and validation loss. Five or more is strongly preferred.",
      warnings: estimateParameterWarnings(clean),
    };
  }

  const losses = clean.map((p) => p.loss).filter(Number.isFinite);
  const minLoss = Math.min(...losses);
  const maxLoss = Math.max(...losses);
  const span = Math.max(maxLoss - minLoss, Math.max(0.05 * Math.abs(minLoss), 1e-4));

  const eHigh = fitModel === "noE" ? 0 : Math.max(0, minLoss - Math.max(span * 1e-3, 1e-8));
  const eLow = 0;
  const eGrid = fitModel === "noE" ? [0] : [0, 0.15, 0.3, 0.5, 0.7, 0.85, 0.95].map((f) => eLow + f * (eHigh - eLow));
  const alphaGrid = logspace(0.015, 0.7, 24);
  const betaGrid = logspace(0.015, 0.7, 24);

  let best = null;
  for (const E of eGrid) {
    for (const alpha of alphaGrid) {
      for (const beta of betaGrid) {
        const candidate = makeModelFromEBasis(clean, E, alpha, beta, weighting);
        if (!Number.isFinite(candidate.objective)) continue;
        if (!best || candidate.objective < best.objective) best = candidate;
      }
    }
  }

  if (!best) {
    return {
      ok: false,
      reason: "Fit failed. Check that losses are positive and that N/D are numeric.",
      warnings: estimateParameterWarnings(clean),
    };
  }

  let state = {
    E: best.E,
    logAlpha: Math.log(best.alpha),
    logBeta: Math.log(best.beta),
  };
  let step = {
    E: Math.max((eHigh - eLow) * 0.15, 1e-4),
    logAlpha: 0.35,
    logBeta: 0.35,
  };

  const evalState = (s) => {
    const E = fitModel === "noE" ? 0 : clamp(s.E, eLow, eHigh);
    const alpha = clamp(Math.exp(s.logAlpha), 0.005, 1.2);
    const beta = clamp(Math.exp(s.logBeta), 0.005, 1.2);
    return makeModelFromEBasis(clean, E, alpha, beta, weighting);
  };

  let current = evalState(state);

  for (let iter = 0; iter < 120; iter++) {
    let improved = false;
    const trials = [];

    if (fitModel !== "noE") {
      trials.push({ ...state, E: state.E + step.E });
      trials.push({ ...state, E: state.E - step.E });
    }
    trials.push({ ...state, logAlpha: state.logAlpha + step.logAlpha });
    trials.push({ ...state, logAlpha: state.logAlpha - step.logAlpha });
    trials.push({ ...state, logBeta: state.logBeta + step.logBeta });
    trials.push({ ...state, logBeta: state.logBeta - step.logBeta });

    for (const trial of trials) {
      const candidate = evalState(trial);
      if (Number.isFinite(candidate.objective) && candidate.objective < current.objective) {
        current = candidate;
        state = {
          E: candidate.E,
          logAlpha: Math.log(candidate.alpha),
          logBeta: Math.log(candidate.beta),
        };
        improved = true;
      }
    }

    if (!improved) {
      step.E *= 0.62;
      step.logAlpha *= 0.68;
      step.logBeta *= 0.68;
    }

    if (step.E < 1e-8 && step.logAlpha < 1e-4 && step.logBeta < 1e-4) break;
  }

  const decoratedPoints = clean.map((p) => {
    const pred = predictLoss(current, p.N, p.D);
    const residual = p.loss - pred;
    return {
      ...p,
      fit_loss: pred,
      fit_residual: residual,
      fit_residual_pct: safeDivide(100 * residual, Math.abs(pred)),
      model_term: current.A * p.N ** (-current.alpha),
      data_term: current.B * p.D ** (-current.beta),
    };
  });

  const observedCompute = decoratedPoints.map((p) => p.flops).filter((x) => Number.isFinite(x) && x > 0);
  const computeMin = observedCompute.length ? Math.min(...observedCompute) : trainingFlops(Math.min(...clean.map((p) => p.N)), Math.min(...clean.map((p) => p.D)));
  const computeMax = observedCompute.length ? Math.max(...observedCompute) : trainingFlops(Math.max(...clean.map((p) => p.N)), Math.max(...clean.map((p) => p.D)));

  return {
    ok: true,
    model: current,
    points: decoratedPoints,
    score: scoreModel(clean, current, weighting),
    warnings: estimateParameterWarnings(clean),
    computeMin,
    computeMax,
    fitModel,
    weighting,
  };
}

function optimumForCompute(model, flops) {
  if (!model || !Number.isFinite(flops) || flops <= 0) return null;
  const C = flops;
  const { A, alpha, B, beta } = model;

  if (![A, alpha, B, beta].every(Number.isFinite) || A <= 0 || B <= 0 || alpha <= 0 || beta <= 0) return null;

  const numerator = alpha * A;
  const denominator = beta * B;
  const Nstar = (numerator / denominator * (C / FLOPS_PER_TOKEN_FACTOR) ** beta) ** (1 / (alpha + beta));
  const Dstar = C / (FLOPS_PER_TOKEN_FACTOR * Nstar);
  const loss = predictLoss(model, Nstar, Dstar);

  return {
    flops: C,
    pf_days: flopsToPfDays(C),
    N: Nstar,
    D: Dstar,
    D_over_N: safeDivide(Dstar, Nstar),
    loss,
    model_term: model.A * Nstar ** (-model.alpha),
    data_term: model.B * Dstar ** (-model.beta),
  };
}

function makeComputeFrontier(model, computeMin, computeMax, extrapolationDecades) {
  const lo = Math.max(computeMin / (10 ** extrapolationDecades), 1);
  const hi = Math.max(computeMax * (10 ** extrapolationDecades), lo * 10);
  return logspace(lo, hi, 100).map((C) => optimumForCompute(model, C)).filter(Boolean);
}

function makeTargetComputeRows(model, computeMax) {
  const multipliers = [0.5, 1, 2, 5, 10, 30, 100];
  return multipliers.map((m) => {
    const opt = optimumForCompute(model, computeMax * m);
    return {
      compute_multiplier: `${m}× max observed`,
      pf_days: opt?.pf_days ?? NaN,
      optimal_N: opt?.N ?? NaN,
      optimal_D: opt?.D ?? NaN,
      optimal_D_over_N: opt?.D_over_N ?? NaN,
      predicted_loss: opt?.loss ?? NaN,
      model_term: opt?.model_term ?? NaN,
      data_term: opt?.data_term ?? NaN,
    };
  });
}

function makeConstantRows(fit) {
  if (!fit?.ok) return [];
  const m = fit.model;
  const alphaRatio = safeDivide(m.beta, m.alpha);
  const frontierNExponent = safeDivide(m.beta, m.alpha + m.beta);
  const frontierDExponent = safeDivide(m.alpha, m.alpha + m.beta);
  const lossComputeExponent = safeDivide(m.alpha * m.beta, m.alpha + m.beta);

  return [
    { constant: "E", value: m.E, meaning: "Irreducible / asymptotic loss floor in fitted loss units" },
    { constant: "A", value: m.A, meaning: "Model-size coefficient multiplying N^-α" },
    { constant: "alpha", value: m.alpha, meaning: "Model-size scaling exponent" },
    { constant: "B", value: m.B, meaning: "Dataset-size coefficient multiplying D^-β" },
    { constant: "beta", value: m.beta, meaning: "Dataset-size scaling exponent" },
    { constant: "N_c = A^(1/alpha)", value: m.N_c, meaning: "Kaplan-style critical model scale if written as (N_c/N)^α" },
    { constant: "D_c = B^(1/beta)", value: m.D_c, meaning: "Kaplan-style critical dataset scale if written as (D_c/D)^β" },
    { constant: "beta / alpha", value: alphaRatio, meaning: "Controls how optimal D/N changes with compute" },
    { constant: "N* compute exponent", value: frontierNExponent, meaning: "Along optimal frontier, N* ∝ C^(β/(α+β))" },
    { constant: "D* compute exponent", value: frontierDExponent, meaning: "Along optimal frontier, D* ∝ C^(α/(α+β))" },
    { constant: "loss-compute exponent", value: lossComputeExponent, meaning: "Along optimal frontier, excess loss roughly scales as C^(-αβ/(α+β))" },
  ];
}

function makeKaplanComparisonRows(fit) {
  if (!fit?.ok) return [];

  const m = fit.model;
  const r = KAPLAN_REFERENCE;
  const fittedRatio = safeDivide(m.beta, m.alpha);
  const referenceRatio = safeDivide(r.alpha_D, r.alpha_N);
  const fittedNFrontier = safeDivide(m.beta, m.alpha + m.beta);
  const fittedDFrontier = safeDivide(m.alpha, m.alpha + m.beta);
  const fittedLossCompute = safeDivide(m.alpha * m.beta, m.alpha + m.beta);
  const refNFrontier = safeDivide(r.alpha_D, r.alpha_N + r.alpha_D);
  const refDFrontier = safeDivide(r.alpha_N, r.alpha_N + r.alpha_D);
  const refLossCompute = safeDivide(r.alpha_N * r.alpha_D, r.alpha_N + r.alpha_D);

  const rows = [
    {
      constant: "alpha_N / alpha",
      fitted_value: m.alpha,
      kaplan_2020: r.alpha_N,
      meaning: "Model-size exponent; larger means loss improves faster with N.",
    },
    {
      constant: "alpha_D / beta",
      fitted_value: m.beta,
      kaplan_2020: r.alpha_D,
      meaning: "Dataset-size exponent; larger means loss improves faster with D.",
    },
    {
      constant: "N_c",
      fitted_value: m.N_c,
      kaplan_2020: r.N_c,
      meaning: "Critical model scale. Kaplan used non-embedding parameters.",
    },
    {
      constant: "D_c",
      fitted_value: m.D_c,
      kaplan_2020: r.D_c,
      meaning: "Critical dataset scale in training tokens.",
    },
    {
      constant: "alpha_D / alpha_N",
      fitted_value: fittedRatio,
      kaplan_2020: referenceRatio,
      meaning: "Relative strength of data scaling versus model scaling.",
    },
    {
      constant: "N*(C) exponent from L(N,D)",
      fitted_value: fittedNFrontier,
      kaplan_2020: refNFrontier,
      meaning: "Assuming C≈6ND and optimizing this N,D surface: N* ∝ C^x.",
    },
    {
      constant: "D*(C) exponent from L(N,D)",
      fitted_value: fittedDFrontier,
      kaplan_2020: refDFrontier,
      meaning: "Assuming C≈6ND and optimizing this N,D surface: D* ∝ C^x.",
    },
    {
      constant: "loss-compute exponent from L(N,D)",
      fitted_value: fittedLossCompute,
      kaplan_2020: refLossCompute,
      meaning: "Excess loss exponent implied by the fitted N,D surface only.",
    },
    {
      constant: "Kaplan reported alpha_C",
      fitted_value: NaN,
      kaplan_2020: r.alpha_C,
      meaning: "Kaplan's separate naive compute-scaling exponent, not directly fit by this page.",
    },
    {
      constant: "Kaplan reported alpha_C_min",
      fitted_value: NaN,
      kaplan_2020: r.alpha_C_min,
      meaning: "Kaplan's separate minimum-compute exponent, not directly fit by this page.",
    },
  ];

  return rows.map((row) => {
    const fitValue = row.fitted_value;
    const refValue = row.kaplan_2020;
    return {
      constant: row.constant,
      fitted_value: formatReferenceConstant(row.constant, fitValue),
      kaplan_2020: formatReferenceConstant(row.constant, refValue),
      fit_over_kaplan: safeDivide(fitValue, refValue),
      pct_difference: percentDifference(fitValue, refValue),
      interpretation: Number.isFinite(fitValue) ? compareNote(fitValue, refValue) : "Reference only",
      meaning: row.meaning,
    };
  });
}

function makeMetricRows(fit) {
  if (!fit?.ok) return [];
  return [
    { metric: "usable_points", value: fit.points.length, note: "Rows with valid N, D, and validation loss" },
    { metric: "rmse_loss", value: fit.score.rmse, note: "RMSE of measured − fitted loss" },
    { metric: "mae_loss", value: fit.score.mae, note: "Mean absolute residual" },
    { metric: "median_abs_pct", value: fit.score.median_abs_pct, note: "Median absolute percent residual" },
    { metric: "mean_pct", value: fit.score.mean_pct, note: "Mean percent residual; sign indicates global bias" },
    { metric: "max_abs_pct", value: fit.score.max_abs_pct, note: "Worst absolute percent residual" },
    { metric: "r2", value: fit.score.r2, note: "Variance explained in raw loss space" },
  ];
}

function makePredictionGridCsv(model, bounds, n = 26) {
  if (!model || !bounds) return "";
  const Ns = logspace(bounds.Nmin, bounds.Nmax, n);
  const Ds = logspace(bounds.Dmin, bounds.Dmax, n);
  const lines = ["N,D,D_over_N,predicted_loss,model_term,data_term,flops,pf_days"];
  for (const N of Ns) {
    for (const D of Ds) {
      const flops = trainingFlops(N, D);
      const modelTerm = model.A * N ** (-model.alpha);
      const dataTerm = model.B * D ** (-model.beta);
      lines.push([
        N,
        D,
        safeDivide(D, N),
        predictLoss(model, N, D),
        modelTerm,
        dataTerm,
        flops,
        flopsToPfDays(flops),
      ].map((x) => Number.isFinite(x) ? x : "").join(","));
    }
  }
  return lines.join("\n");
}

function downloadText(filename, text) {
  const blob = new Blob([text], { type: "text/plain;charset=utf-8" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  URL.revokeObjectURL(url);
}

function makeDemoCurves() {
  const trueModel = {
    E: 1.18,
    A: 3.0,
    alpha: 0.08,
    B: 2.8,
    beta: 0.08,
  };
  const datasetSizes = [8.5e9, 17e9, 35e9];
  const modelSizes = [2e7, 3.9e7, 7e7, 1.08e8, 2.09e8, 4.2e8];

  return datasetSizes.map((D, curveIndex) => {
    const points = modelSizes.map((N, pointIndex) => {
      const base = predictLoss(trueModel, N, D);
      const systematic = 0.012 * Math.sin(pointIndex * 1.7 + curveIndex * 0.6) + 0.006 * curveIndex;
      const loss = base + systematic;
      const flops = trainingFlops(N, D);
      return {
        index: pointIndex,
        name: `demo_${curveIndex}_${pointIndex}`,
        N,
        D,
        D_over_N: safeDivide(D, N),
        loss,
        train_loss: loss - 0.035,
        train_val_gap: 0.035,
        flops,
        pf_days: flopsToPfDays(flops),
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
    metric: "Metric",
    value: "Value",
    note: "Note",
    constant: "Constant",
    meaning: "Meaning",
    compute_multiplier: "Compute",
    pf_days: "PF-days",
    optimal_N: "Optimal N",
    optimal_D: "Optimal D",
    optimal_D_over_N: "Optimal D/N",
    predicted_loss: "Predicted loss",
    model_term: "Model term",
    data_term: "Data term",
    name: "Run",
    curve: "Curve",
    N: "N",
    D: "D",
    D_over_N: "D/N",
    loss: "Measured loss",
    fit_loss: "Fitted loss",
    fit_residual: "Residual",
    fit_residual_pct: "Residual %",
    pf_days_observed: "Observed PF-days",
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

function DataTable({ title, rows, maxHeight = "max-h-[420px]" }) {
  const keys = Object.keys(rows[0] ?? {});

  return (
    <div className="rounded-2xl border border-zinc-800 bg-zinc-950/80 p-3">
      <h2 className="mb-2 text-base font-semibold text-zinc-100">{title}</h2>
      <div className={`${maxHeight} overflow-auto`}>
        <table className="w-full text-left text-xs">
          <thead className="sticky top-0 z-10 border-b border-zinc-800 bg-zinc-950 uppercase tracking-wide text-zinc-500">
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
                  <td key={key} className="max-w-[520px] whitespace-nowrap px-2 py-1.5 text-zinc-300">
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

function NumberControl({ label, value, setValue, help, min = 0, max = 5, step = 0.25 }) {
  return (
    <label className="block rounded-xl border border-zinc-800 bg-zinc-950/70 p-3">
      <div className="mb-2 flex items-center justify-between gap-3">
        <div className="text-sm font-medium text-zinc-100">{label}</div>
        <div className="text-sm font-semibold text-blue-200">{value}</div>
      </div>
      <input
        type="range"
        min={min}
        max={max}
        step={step}
        value={value}
        onChange={(event) => setValue(Number(event.target.value))}
        className="w-full accent-blue-500"
      />
      {help && <div className="mt-2 text-xs leading-relaxed text-zinc-500">{help}</div>}
    </label>
  );
}

function plotLayout({ xTitle, yTitle, logX = false, logY = false, shapes = [], legendY = 1.14, heightPad = 54 }) {
  return {
    autosize: true,
    paper_bgcolor: "rgba(0,0,0,0)",
    plot_bgcolor: "rgba(0,0,0,0)",
    font: { color: "#d4d4d8", size: 12 },
    margin: { l: 62, r: 24, t: 8, b: heightPad },
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

function PlotCard({ title, subtitle, traces, layout, config, heightClass }) {
  return (
    <ChartShell title={title} subtitle={subtitle} footer="Drag to zoom. Mouse wheel zoom, pan, autoscale, reset axes, and PNG export are enabled." heightClass={heightClass}>
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
        help="Use num_train_tokens for loss-vs-N curves. Leave blank for one combined curve. Add learning_rate or architecture columns to split curves."
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

function EquationPanel() {
  return (
    <details className="rounded-2xl border border-zinc-800 bg-zinc-950/80 p-3" open>
      <summary className="cursor-pointer text-sm font-semibold uppercase tracking-wide text-zinc-400">What this page fits</summary>
      <div className="mt-2 space-y-2 text-xs leading-relaxed text-zinc-500">
        <p>
          Main fit: <code className="text-zinc-300">L(N,D) = E + A·N^(-α) + B·D^(-β)</code>. It also reports <code className="text-zinc-300">N_c=A^(1/α)</code> and <code className="text-zinc-300">D_c=B^(1/β)</code>, so the fit can be read as <code className="text-zinc-300">E + (N_c/N)^α + (D_c/D)^β</code>.
        </p>
        <p>
          Compute is estimated as <code className="text-zinc-300">C≈6ND</code> training FLOPs. The optimal frontier is derived analytically from that fit and gives <code className="text-zinc-300">N*(C)</code>, <code className="text-zinc-300">D*(C)</code>, predicted loss, and D/N.
        </p>
        <p>
          The Kaplan comparison table uses Kaplan et al.'s reference values <code className="text-zinc-300">α_N=0.076</code>, <code className="text-zinc-300">α_D=0.095</code>, <code className="text-zinc-300">N_c=8.8e13</code>, and <code className="text-zinc-300">D_c=5.4e13</code>. Kaplan's original combined law is <code className="text-zinc-300">[(N_c/N)^(α_N/α_D)+D_c/D]^α_D</code>, so the table compares exponents and critical scales rather than pretending every coefficient is exactly identical to the additive form.
        </p>
      </div>
    </details>
  );
}

function WarningPanel({ fit }) {
  if (!fit || !fit.warnings || fit.warnings.length === 0) return null;
  return (
    <div className="rounded-2xl border border-amber-500/30 bg-amber-500/10 p-3 text-sm leading-relaxed text-amber-100">
      <div className="font-semibold">Fit warnings</div>
      <ul className="mt-1 list-disc space-y-1 pl-5">
        {fit.warnings.map((warning, i) => <li key={i}>{warning}</li>)}
      </ul>
    </div>
  );
}

function makeBounds(points, extrapolationDecades) {
  const Ns = points.map((p) => p.N).filter((x) => Number.isFinite(x) && x > 0);
  const Ds = points.map((p) => p.D).filter((x) => Number.isFinite(x) && x > 0);
  if (Ns.length === 0 || Ds.length === 0) return null;
  const factor = 10 ** extrapolationDecades;
  return {
    Nmin: Math.min(...Ns) / factor,
    Nmax: Math.max(...Ns) * factor,
    Dmin: Math.min(...Ds) / factor,
    Dmax: Math.max(...Ds) * factor,
  };
}

function LossVsNChart({ fit, curves, extrapolationDecades }) {
  const bounds = makeBounds(fit.points, extrapolationDecades);
  const traces = [];

  curves.forEach((curve, index) => {
    const color = curve.color ?? COLORS[index % COLORS.length];
    const points = curve.points
      .map((p) => fit.points.find((fp) => fp.curve === curve.name && fp.index === p.index && fp.name === p.name && fp.N === p.N && fp.D === p.D) ?? p)
      .filter((p) => Number.isFinite(p.N) && Number.isFinite(p.loss))
      .sort((a, b) => a.N - b.N);

    traces.push({
      type: "scatter",
      mode: "markers+lines",
      name: `${curve.name} measured`,
      x: points.map((p) => p.N),
      y: points.map((p) => p.loss),
      text: points.map((p) => [
        `curve: ${curve.name}`,
        `run: ${p.name}`,
        `N: ${human_num(p.N)}`,
        `D: ${human_num(p.D)}`,
        `loss: ${formatLoss(p.loss)}`,
        `fit: ${formatLoss(p.fit_loss)}`,
        `residual: ${formatPercent(p.fit_residual_pct)}`,
      ].join("<br>")),
      hovertemplate: "%{text}<extra></extra>",
      line: { color, width: 2.2 },
      marker: { color, size: 7, line: { color: "rgba(255,255,255,0.35)", width: 1 } },
    });

    const representativeD = median(points.map((p) => p.D));
    if (bounds && Number.isFinite(representativeD)) {
      const Ns = logspace(bounds.Nmin, bounds.Nmax, 140);
      traces.push({
        type: "scatter",
        mode: "lines",
        name: `${curve.name} fitted extrap.`,
        x: Ns,
        y: Ns.map((N) => predictLoss(fit.model, N, representativeD)),
        hovertemplate: `N=%{x:.3e}<br>D=${human_num(representativeD)}<br>fit=%{y:.5f}<extra></extra>`,
        line: { color, width: 2, dash: "dash" },
      });
    }
  });

  return (
    <PlotCard
      title="Loss vs model size with fitted extrapolation"
      subtitle="Solid lines are imported rows. Dashed lines are the refit Kaplan-style model evaluated at each curve's representative dataset size."
      traces={traces}
      layout={plotLayout({ xTitle: "Model size N", yTitle: "Validation loss", logX: true })}
    />
  );
}

function LossVsDChart({ fit, extrapolationDecades }) {
  const bounds = makeBounds(fit.points, extrapolationDecades);
  const groups = new Map();
  fit.points.forEach((p) => {
    const key = `N=${human_num(p.N)}`;
    if (!groups.has(key)) groups.set(key, []);
    groups.get(key).push(p);
  });

  const entries = Array.from(groups.entries()).slice(0, 12);
  const traces = [];

  entries.forEach(([name, points], index) => {
    const color = COLORS[index % COLORS.length];
    const sorted = points.slice().sort((a, b) => a.D - b.D);
    traces.push({
      type: "scatter",
      mode: "markers+lines",
      name: `${name} measured`,
      x: sorted.map((p) => p.D),
      y: sorted.map((p) => p.loss),
      text: sorted.map((p) => [
        `run: ${p.name}`,
        `N: ${human_num(p.N)}`,
        `D: ${human_num(p.D)}`,
        `loss: ${formatLoss(p.loss)}`,
        `fit: ${formatLoss(p.fit_loss)}`,
        `residual: ${formatPercent(p.fit_residual_pct)}`,
      ].join("<br>")),
      hovertemplate: "%{text}<extra></extra>",
      line: { color, width: 2.2 },
      marker: { color, size: 7, line: { color: "rgba(255,255,255,0.35)", width: 1 } },
    });

    const representativeN = median(sorted.map((p) => p.N));
    if (bounds && Number.isFinite(representativeN)) {
      const Ds = logspace(bounds.Dmin, bounds.Dmax, 140);
      traces.push({
        type: "scatter",
        mode: "lines",
        name: `${name} fitted extrap.`,
        x: Ds,
        y: Ds.map((D) => predictLoss(fit.model, representativeN, D)),
        hovertemplate: `D=%{x:.3e}<br>N=${human_num(representativeN)}<br>fit=%{y:.5f}<extra></extra>`,
        line: { color, width: 2, dash: "dash" },
      });
    }
  });

  return (
    <PlotCard
      title="Loss vs dataset size with fitted extrapolation"
      subtitle="This view is most useful when you have repeated architecture sizes across multiple datasets."
      traces={traces}
      layout={plotLayout({ xTitle: "Dataset size D", yTitle: "Validation loss", logX: true })}
    />
  );
}

function ParityChart({ fit }) {
  const values = fit.points.flatMap((p) => [p.loss, p.fit_loss]).filter(Number.isFinite);
  const lo = values.length ? Math.min(...values) : 0;
  const hi = values.length ? Math.max(...values) : 1;
  const pad = (hi - lo) * 0.08 || 0.05;

  const traces = [
    {
      type: "scatter",
      mode: "markers",
      name: "runs",
      x: fit.points.map((p) => p.fit_loss),
      y: fit.points.map((p) => p.loss),
      text: fit.points.map((p) => [
        `run: ${p.name}`,
        `N: ${human_num(p.N)}`,
        `D: ${human_num(p.D)}`,
        `measured: ${formatLoss(p.loss)}`,
        `fit: ${formatLoss(p.fit_loss)}`,
        `residual: ${formatPercent(p.fit_residual_pct)}`,
      ].join("<br>")),
      hovertemplate: "%{text}<extra></extra>",
      marker: {
        color: fit.points.map((p) => p.fit_residual_pct),
        colorscale: "RdBu",
        reversescale: true,
        colorbar: { title: { text: "resid %", font: { color: "#a1a1aa" } }, tickfont: { color: "#a1a1aa" } },
        size: 8,
        line: { color: "rgba(255,255,255,0.35)", width: 1 },
      },
    },
    {
      type: "scatter",
      mode: "lines",
      name: "perfect fit",
      x: [lo - pad, hi + pad],
      y: [lo - pad, hi + pad],
      hoverinfo: "skip",
      line: { color: "#fbbf24", width: 2, dash: "dash" },
    },
  ];

  const base = plotLayout({ xTitle: "Fitted loss", yTitle: "Measured validation loss" });
  return (
    <PlotCard
      title="Parity: measured vs refit scaling law"
      subtitle="Good fits land close to the dashed line. Structured curvature means the fitted equation is missing something."
      traces={traces}
      layout={{
        ...base,
        xaxis: { ...base.xaxis, range: [lo - pad, hi + pad] },
        yaxis: { ...base.yaxis, range: [lo - pad, hi + pad], scaleanchor: "x", scaleratio: 1 },
      }}
    />
  );
}

function ResidualChart({ fit, xMetric = "N" }) {
  const labels = {
    N: "Model size N",
    D: "Dataset size D",
    D_over_N: "Data/model ratio D/N",
    pf_days: "Compute, PF-days",
    coverage: "Training coverage",
  };
  const sorted = fit.points.filter((p) => Number.isFinite(p[xMetric]) && p[xMetric] > 0).sort((a, b) => a[xMetric] - b[xMetric]);
  const traces = [{
    type: "scatter",
    mode: "markers+lines",
    name: "residual %",
    x: sorted.map((p) => p[xMetric]),
    y: sorted.map((p) => p.fit_residual_pct),
    text: sorted.map((p) => [
      `run: ${p.name}`,
      `N: ${human_num(p.N)}`,
      `D: ${human_num(p.D)}`,
      `loss: ${formatLoss(p.loss)}`,
      `fit: ${formatLoss(p.fit_loss)}`,
      `residual: ${formatPercent(p.fit_residual_pct)}`,
    ].join("<br>")),
    hovertemplate: "%{text}<extra></extra>",
    line: { color: "#60a5fa", width: 2.4 },
    marker: { color: "#60a5fa", size: 7, line: { color: "rgba(255,255,255,0.35)", width: 1 } },
  }];

  return (
    <PlotCard
      title="Residual structure"
      subtitle="Random scatter is good. Sloped residuals usually mean the exponent, compute accounting, or a sweep dimension is off."
      traces={traces}
      layout={plotLayout({
        xTitle: labels[xMetric] ?? xMetric,
        yTitle: "Measured − fit (%)",
        logX: !["coverage"].includes(xMetric),
        shapes: [{ type: "line", xref: "paper", x0: 0, x1: 1, yref: "y", y0: 0, y1: 0, line: { color: "#fbbf24", width: 2, dash: "dash" } }],
      })}
    />
  );
}

function LossSurfaceChart({ fit, extrapolationDecades }) {
  const bounds = makeBounds(fit.points, extrapolationDecades);
  if (!bounds) return null;

  const logNs = Array.from({ length: 45 }, (_, i) => log10(bounds.Nmin) + (log10(bounds.Nmax) - log10(bounds.Nmin)) * i / 44);
  const logDs = Array.from({ length: 45 }, (_, i) => log10(bounds.Dmin) + (log10(bounds.Dmax) - log10(bounds.Dmin)) * i / 44);
  const z = logDs.map((ld) => logNs.map((ln) => predictLoss(fit.model, 10 ** ln, 10 ** ld)));

  const traces = [
    {
      type: "contour",
      name: "fitted loss surface",
      x: logNs,
      y: logDs,
      z,
      colorscale: "Viridis",
      contours: { coloring: "heatmap", showlabels: true, labelfont: { color: "#e4e4e7", size: 10 } },
      colorbar: { title: { text: "loss", font: { color: "#a1a1aa" } }, tickfont: { color: "#a1a1aa" } },
      hovertemplate: "log10 N=%{x:.3f}<br>log10 D=%{y:.3f}<br>loss=%{z:.5f}<extra></extra>",
    },
    {
      type: "scatter",
      mode: "markers",
      name: "observed runs",
      x: fit.points.map((p) => log10(p.N)),
      y: fit.points.map((p) => log10(p.D)),
      text: fit.points.map((p) => [
        `run: ${p.name}`,
        `N: ${human_num(p.N)}`,
        `D: ${human_num(p.D)}`,
        `measured: ${formatLoss(p.loss)}`,
        `fit: ${formatLoss(p.fit_loss)}`,
        `residual: ${formatPercent(p.fit_residual_pct)}`,
      ].join("<br>")),
      hovertemplate: "%{text}<extra></extra>",
      marker: { color: "#fbbf24", size: 8, line: { color: "rgba(0,0,0,0.7)", width: 1.5 } },
    },
  ];

  return (
    <PlotCard
      title="Extrapolated N-D loss surface"
      subtitle="Contour plot of the fitted Kaplan-style law. Axes are log10(N) and log10(D), so the plot stays readable over many decades."
      traces={traces}
      layout={plotLayout({ xTitle: "log10 model size N", yTitle: "log10 dataset size D", legendY: 1.08 })}
      heightClass="h-[420px]"
    />
  );
}

function ComputeFrontierChart({ fit, extrapolationDecades }) {
  const frontier = makeComputeFrontier(fit.model, fit.computeMin, fit.computeMax, extrapolationDecades);
  const observed = fit.points.slice().sort((a, b) => a.pf_days - b.pf_days);

  const traces = [
    {
      type: "scatter",
      mode: "lines",
      name: "optimal frontier from fit",
      x: frontier.map((p) => p.pf_days),
      y: frontier.map((p) => p.loss),
      text: frontier.map((p) => [
        `PF-days: ${p.pf_days.toFixed(4)}`,
        `N*: ${human_num(p.N)}`,
        `D*: ${human_num(p.D)}`,
        `D/N: ${human_num(p.D_over_N)}`,
        `loss: ${formatLoss(p.loss)}`,
      ].join("<br>")),
      hovertemplate: "%{text}<extra></extra>",
      line: { color: "#34d399", width: 3 },
    },
    {
      type: "scatter",
      mode: "markers",
      name: "observed runs",
      x: observed.map((p) => p.pf_days),
      y: observed.map((p) => p.loss),
      text: observed.map((p) => [
        `run: ${p.name}`,
        `PF-days: ${formatTableValue("pf_days", p.pf_days)}`,
        `N: ${human_num(p.N)}`,
        `D: ${human_num(p.D)}`,
        `D/N: ${human_num(p.D_over_N)}`,
        `loss: ${formatLoss(p.loss)}`,
      ].join("<br>")),
      hovertemplate: "%{text}<extra></extra>",
      marker: { color: "#fbbf24", size: 8, line: { color: "rgba(255,255,255,0.35)", width: 1 } },
    },
  ];

  return (
    <PlotCard
      title="Optimal compute frontier"
      subtitle="For each compute budget C≈6ND, the green curve shows the fitted optimum N and D. Observed runs are shown for comparison."
      traces={traces}
      layout={plotLayout({ xTitle: "Compute C, PF-days", yTitle: "Validation loss", logX: true })}
    />
  );
}

function OptimalAllocationChart({ fit, extrapolationDecades }) {
  const frontier = makeComputeFrontier(fit.model, fit.computeMin, fit.computeMax, extrapolationDecades);
  const traces = [
    {
      type: "scatter",
      mode: "lines",
      name: "N*(C)",
      x: frontier.map((p) => p.pf_days),
      y: frontier.map((p) => p.N),
      hovertemplate: "PF-days=%{x:.4f}<br>N*=%{y:.3e}<extra></extra>",
      line: { color: "#60a5fa", width: 2.8 },
    },
    {
      type: "scatter",
      mode: "lines",
      name: "D*(C)",
      x: frontier.map((p) => p.pf_days),
      y: frontier.map((p) => p.D),
      hovertemplate: "PF-days=%{x:.4f}<br>D*=%{y:.3e}<extra></extra>",
      line: { color: "#fbbf24", width: 2.8 },
    },
    {
      type: "scatter",
      mode: "lines",
      name: "D*/N*",
      x: frontier.map((p) => p.pf_days),
      y: frontier.map((p) => p.D_over_N),
      yaxis: "y2",
      hovertemplate: "PF-days=%{x:.4f}<br>D/N=%{y:.3e}<extra></extra>",
      line: { color: "#fb7185", width: 2.6, dash: "dash" },
    },
  ];

  const base = plotLayout({ xTitle: "Compute C, PF-days", yTitle: "N*(C) and D*(C)", logX: true, logY: true });
  return (
    <PlotCard
      title="Optimal allocation implied by refit"
      subtitle="The secondary red dashed curve is optimal D/N. If it is far from your sweep, your runs are off-frontier according to the fit."
      traces={traces}
      layout={{
        ...base,
        yaxis2: {
          title: { text: "D*/N*", font: { color: "#a1a1aa", size: 12 } },
          type: "log",
          overlaying: "y",
          side: "right",
          gridcolor: "rgba(255,255,255,0)",
          tickfont: { color: "#a1a1aa", size: 11 },
        },
      }}
    />
  );
}

function TermBalanceChart({ fit }) {
  const sorted = fit.points.slice().sort((a, b) => a.D_over_N - b.D_over_N);
  const traces = [
    {
      type: "scatter",
      mode: "markers",
      name: "run term balance",
      x: sorted.map((p) => p.D_over_N),
      y: sorted.map((p) => safeDivide(p.data_term, p.model_term)),
      text: sorted.map((p) => [
        `run: ${p.name}`,
        `D/N: ${human_num(p.D_over_N)}`,
        `data/model term: ${formatSmall(safeDivide(p.data_term, p.model_term))}`,
        `model term: ${formatLoss(p.model_term)}`,
        `data term: ${formatLoss(p.data_term)}`,
      ].join("<br>")),
      hovertemplate: "%{text}<extra></extra>",
      marker: { color: "#a78bfa", size: 8, line: { color: "rgba(255,255,255,0.35)", width: 1 } },
    },
  ];

  return (
    <PlotCard
      title="Model term vs data term balance"
      subtitle="Values above 1 mean the D term dominates the fitted excess loss. Values below 1 mean the N term dominates."
      traces={traces}
      layout={plotLayout({
        xTitle: "D/N",
        yTitle: "data term / model term",
        logX: true,
        logY: true,
        shapes: [{ type: "line", xref: "paper", x0: 0, x1: 1, yref: "y", y0: 1, y1: 1, line: { color: "#fbbf24", width: 2, dash: "dash" } }],
      })}
    />
  );
}

function ExportPanel({ fit, bounds }) {
  if (!fit?.ok) return null;

  const constantsJson = JSON.stringify({
    equation: "L(N,D) = E + A*N^(-alpha) + B*D^(-beta)",
    flops: "C = 6*N*D",
    fit_model: fit.fitModel,
    weighting: fit.weighting,
    constants: fit.model,
    kaplan_reference: KAPLAN_REFERENCE,
    kaplan_comparison_rows: makeKaplanComparisonRows(fit),
    score: fit.score,
    warnings: fit.warnings,
  }, null, 2);

  const predictionCsv = makePredictionGridCsv(fit.model, bounds, 30);

  return (
    <div className="rounded-2xl border border-zinc-800 bg-zinc-950/80 p-3">
      <h2 className="text-base font-semibold text-zinc-100">Export fitted results</h2>
      <div className="mt-3 grid grid-cols-1 gap-2 sm:grid-cols-2">
        <button
          type="button"
          onClick={() => downloadText("kaplan_refit_constants.json", constantsJson)}
          className="rounded-xl border border-blue-500/40 bg-blue-500/15 px-4 py-2 text-sm font-semibold text-blue-100 transition hover:border-blue-400 hover:bg-blue-500/25"
        >
          Download constants JSON
        </button>
        <button
          type="button"
          onClick={() => downloadText("kaplan_refit_prediction_grid.csv", predictionCsv)}
          className="rounded-xl border border-emerald-500/40 bg-emerald-500/15 px-4 py-2 text-sm font-semibold text-emerald-100 transition hover:border-emerald-400 hover:bg-emerald-500/25"
        >
          Download prediction grid CSV
        </button>
      </div>
    </div>
  );
}

function RecommendationPanel({ fit }) {
  if (!fit?.ok) return null;
  const recs = [];
  const ratios = fit.points.map((p) => p.D_over_N).filter(Number.isFinite);
  const residualSlopeN = linearFit(fit.points.map((p) => log10(p.N)), fit.points.map((p) => p.fit_residual_pct));
  const residualSlopeD = linearFit(fit.points.map((p) => log10(p.D)), fit.points.map((p) => p.fit_residual_pct));
  const medRatio = median(ratios);

  if (fit.score.median_abs_pct > 10) {
    recs.push({ severity: "warning", text: "Median fitted residual is above 10%. Check whether num_params should be non-embedding parameters, whether all rows use the same validation set, and whether losses are directly comparable." });
  }
  if (Number.isFinite(residualSlopeN.slope) && Math.abs(residualSlopeN.slope) > 8) {
    recs.push({ severity: "warning", text: "Residuals still slope with model size. That usually means the N exponent is not stable yet, the largest models are undertrained, or architecture/hyperparameter changes are mixed into the fit." });
  }
  if (Number.isFinite(residualSlopeD.slope) && Math.abs(residualSlopeD.slope) > 8) {
    recs.push({ severity: "warning", text: "Residuals still slope with dataset size. That usually means the D exponent needs more repeated-N data, or the datasets are not distributionally identical." });
  }
  if (Number.isFinite(medRatio) && medRatio < 10) {
    recs.push({ severity: "info", text: "Your median D/N is very low. According to this fit, many points are likely data-limited; add larger D for the larger models before concluding the model-size law is flat." });
  } else if (Number.isFinite(medRatio) && medRatio > 100) {
    recs.push({ severity: "info", text: "Your median D/N is very high. According to this fit, many points are likely model-limited; add larger N at fixed D to reveal the model-size curve." });
  }
  if (fit.warnings.length > 0) {
    recs.push({ severity: "warning", text: "The fitted constants are only as good as the grid geometry. Add repeated model sizes across datasets and repeated datasets across model sizes to reduce degeneracy." });
  }
  if (recs.length === 0) {
    recs.push({ severity: "good", text: "The fitted scaling law is internally consistent. Next, validate it by running one or two off-grid points near the predicted optimal frontier." });
  }

  const classes = {
    good: "border-emerald-500/30 bg-emerald-500/10 text-emerald-100",
    info: "border-blue-500/30 bg-blue-500/10 text-blue-100",
    warning: "border-amber-500/30 bg-amber-500/10 text-amber-100",
  };

  return (
    <div className="rounded-2xl border border-zinc-800 bg-zinc-950/80 p-3">
      <h2 className="text-base font-semibold text-zinc-100">Fit interpretation</h2>
      <div className="mt-3 space-y-2">
        {recs.slice(0, 6).map((rec, index) => (
          <div key={index} className={`rounded-xl border px-3 py-2 text-sm leading-relaxed ${classes[rec.severity] ?? classes.info}`}>
            {rec.text}
          </div>
        ))}
      </div>
    </div>
  );
}

export default function KaplanRefitPage() {
  const [textValue, setTextValue] = useState("");
  const [groupColumns, setGroupColumns] = useState(DEFAULT_GROUP_COLUMNS);
  const [curves, setCurves] = useState(() => makeDemoCurves());
  const [lossWeighting, setLossWeighting] = useState("absolute");
  const [fitModel, setFitModel] = useState("full");
  const [residualX, setResidualX] = useState("N");
  const [extrapolationDecades, setExtrapolationDecades] = useState(1.5);
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

  const visibleCurves = useMemo(() => curves.filter((curve) => !curve.hidden && curve.points.length > 0), [curves]);
  const allPoints = useMemo(() => visibleCurves.flatMap((curve) => curve.points.map((point) => ({ ...point, curve: curve.name, curveColor: curve.color }))), [visibleCurves]);
  const fit = useMemo(() => fitKaplanLikeModel(allPoints, { weighting: lossWeighting, fitModel }), [allPoints, lossWeighting, fitModel]);
  const bounds = useMemo(() => fit?.ok ? makeBounds(fit.points, extrapolationDecades) : null, [fit, extrapolationDecades]);
  const constantRows = useMemo(() => makeConstantRows(fit), [fit]);
  const kaplanComparisonRows = useMemo(() => makeKaplanComparisonRows(fit), [fit]);
  const metricRows = useMemo(() => makeMetricRows(fit), [fit]);
  const targetComputeRows = useMemo(() => fit?.ok ? makeTargetComputeRows(fit.model, fit.computeMax) : [], [fit]);
  const bestObserved = useMemo(() => fit?.ok ? fit.points.reduce((acc, p) => (!acc || p.loss < acc.loss ? p : acc), null) : null, [fit]);
  const maxObservedComputeOpt = useMemo(() => fit?.ok ? optimumForCompute(fit.model, fit.computeMax) : null, [fit]);

  return (
    <div className="min-h-screen bg-zinc-950 text-zinc-100">
      <div className="min-h-screen lg:flex lg:items-stretch">
        <aside className="border-b border-zinc-800 bg-zinc-950/95 p-5 lg:sticky lg:top-0 lg:h-screen lg:w-[450px] lg:shrink-0 lg:overflow-y-auto lg:border-b-0 lg:border-r">
          <div className="mb-6">
            <h1 className="text-2xl font-semibold tracking-tight">Kaplan refit</h1>
            <p className="mt-2 text-sm leading-relaxed text-zinc-500">
              Paste your run dataframe, refit a Kaplan-style scaling law to your measured losses, inspect constants, and extrapolate the implied optimal frontier.
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
              <h2 className="text-sm font-semibold uppercase tracking-wide text-zinc-400">Fit settings</h2>
              <SelectControl
                label="Scaling law form"
                value={fitModel}
                setValue={setFitModel}
                options={FIT_MODEL_OPTIONS}
                help="Use the full model when you have enough points. Use no-E when the fit is underconstrained or E becomes unstable."
              />
              <SelectControl
                label="Fit objective"
                value={lossWeighting}
                setValue={setLossWeighting}
                options={LOSS_WEIGHTING_OPTIONS}
                help="Absolute is easiest to interpret. Relative/log is useful when losses span a wide range."
              />
              <SelectControl
                label="Residual x-axis"
                value={residualX}
                setValue={setResidualX}
                options={[
                  { value: "N", label: "Model size N" },
                  { value: "D", label: "Dataset size D" },
                  { value: "D_over_N", label: "D/N" },
                  { value: "pf_days", label: "Compute, PF-days" },
                  { value: "coverage", label: "Training coverage" },
                ]}
                help="Use this to diagnose whether residuals are structured along N, D, compute, or training coverage."
              />
              <NumberControl
                label="Extrapolation range, decades"
                value={extrapolationDecades}
                setValue={setExtrapolationDecades}
                min={0}
                max={3}
                step={0.25}
                help="Controls how far plots extend beyond your min/max observed N, D, and compute."
              />
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

            <EquationPanel />
          </div>
        </aside>

        <main className="min-w-0 flex-1 space-y-3 p-4 lg:p-5">
          <div className="flex flex-col gap-2 sm:flex-row sm:items-end sm:justify-between">
            <div>
              <h1 className="text-3xl font-semibold tracking-tight">Kaplan-style scaling-law refit</h1>
              <p className="mt-2 text-sm text-zinc-500">
                {visibleCurves.length} active curve(s), {human_num(allPoints.length)} usable point(s). Fit: <span className="text-blue-300">L = E + A·N^-α + B·D^-β</span>.
              </p>
            </div>

            {fit?.ok && bestObserved && (
              <div className="rounded-xl border border-blue-500/30 bg-blue-500/10 px-4 py-2 text-sm text-blue-100">
                Best observed: {formatLoss(bestObserved.loss)} · N={human_num(bestObserved.N)} · D={human_num(bestObserved.D)}
              </div>
            )}
          </div>

          {!fit?.ok ? (
            <div className="rounded-2xl border border-amber-500/30 bg-amber-500/10 p-6 text-sm leading-relaxed text-amber-100">
              {fit?.reason ?? "No fit available."}
            </div>
          ) : (
            <>
              <section className="grid grid-cols-1 gap-2 md:grid-cols-2 xl:grid-cols-4">
                <StatCard label="α / β" value={`${fit.model.alpha.toFixed(4)} / ${fit.model.beta.toFixed(4)}`} sub="model-size / data-size exponents" />
                <StatCard label="Median |residual|" value={formatPercent(fit.score.median_abs_pct, 1).replace("+", "")} sub="measured − fitted" />
                <StatCard label="R²" value={Number.isFinite(fit.score.r2) ? fit.score.r2.toFixed(4) : "—"} sub="raw loss space" />
                <StatCard label="Optimum @ max C" value={maxObservedComputeOpt ? `N=${human_num(maxObservedComputeOpt.N)}` : "—"} sub={maxObservedComputeOpt ? `D=${human_num(maxObservedComputeOpt.D)}, D/N=${human_num(maxObservedComputeOpt.D_over_N)}` : "—"} />
              </section>

              <WarningPanel fit={fit} />

              <section className="grid grid-cols-1 gap-3 xl:grid-cols-2">
                <LossVsNChart fit={fit} curves={visibleCurves} extrapolationDecades={extrapolationDecades} />
                <LossVsDChart fit={fit} extrapolationDecades={extrapolationDecades} />
              </section>

              <section className="grid grid-cols-1 gap-3 xl:grid-cols-2">
                <ParityChart fit={fit} />
                <ResidualChart fit={fit} xMetric={residualX} />
              </section>

              <section className="grid grid-cols-1 gap-3 xl:grid-cols-2">
                <ComputeFrontierChart fit={fit} extrapolationDecades={extrapolationDecades} />
                <OptimalAllocationChart fit={fit} extrapolationDecades={extrapolationDecades} />
              </section>

              <section className="grid grid-cols-1 gap-3 xl:grid-cols-2">
                <LossSurfaceChart fit={fit} extrapolationDecades={extrapolationDecades} />
                <TermBalanceChart fit={fit} />
              </section>

              <section className="grid grid-cols-1 gap-3 xl:grid-cols-2">
                <DataTable title="Fitted constants" rows={constantRows} />
                <DataTable title="Fit vs Kaplan 2020 constants" rows={kaplanComparisonRows} />
              </section>

              <section className="grid grid-cols-1 gap-3 xl:grid-cols-2">
                <DataTable title="Goodness of fit" rows={metricRows} />
                <DataTable title="Optimal future compute targets" rows={targetComputeRows} />
              </section>

              <section className="grid grid-cols-1 gap-3">
                <RecommendationPanel fit={fit} />
              </section>

              <section className="grid grid-cols-1 gap-3">
                <DataTable
                  title="Per-run fitted values and residuals"
                  rows={fit.points.map((p) => ({
                    curve: p.curve,
                    name: p.name,
                    N: p.N,
                    D: p.D,
                    D_over_N: p.D_over_N,
                    loss: p.loss,
                    fit_loss: p.fit_loss,
                    fit_residual: p.fit_residual,
                    fit_residual_pct: p.fit_residual_pct,
                    model_term: p.model_term,
                    data_term: p.data_term,
                    pf_days_observed: p.pf_days,
                  }))}
                />
                <ExportPanel fit={fit} bounds={bounds} />
              </section>

              <div className="rounded-2xl border border-zinc-800 bg-zinc-950/80 p-3 text-xs leading-relaxed text-zinc-500">
                <div className="font-semibold uppercase tracking-wide text-zinc-400">Important interpretation note</div>
                <p className="mt-1">
                  This page fits your measured particleGPT losses directly; it does not assume Kaplan's original constants. The fitted constants are meaningful only if rows use a consistent validation set, comparable loss definition, comparable tokenizer/vocab, and a sweep grid with independent variation in N and D.
                </p>
                <p className="mt-1">
                  The optimal frontier is an extrapolation of the fitted law. Use it to choose one or two validation runs near the predicted N*, D* allocation before committing a large sweep.
                </p>
              </div>
            </>
          )}
        </main>
      </div>
    </div>
  );
}
