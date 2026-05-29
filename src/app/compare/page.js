"use client";

import { useEffect, useMemo, useRef, useState } from "react";
import dynamic from "next/dynamic";
import { MathJax, MathJaxContext } from "better-react-mathjax";

import {
  gpt_like_nonembedding_params,
  kaplan_loss_ND,
  training_flops,
  flops_to_pf_days,
  tokens_per_parameter,
  human_num,
} from "@/lib/scaling_laws";

const MATHJAX_CONFIG = {
  tex: {
    inlineMath: [["\\(", "\\)"], ["$", "$"]],
    displayMath: [["\\[", "\\]"]],
  },
};

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
];

const X_METRIC_OPTIONS = [
  { value: "N", label: "Model size N" },
  { value: "D", label: "Dataset size D" },
  { value: "pf_days", label: "Compute, PF-days" },
  { value: "D_over_N", label: "Data/model ratio D/N" },
  { value: "steps", label: "Optimizer steps" },
];

const LOSS_SOURCE_OPTIONS = [
  { value: "measured", label: "Measured/imported loss" },
  { value: "predicted", label: "Kaplan predicted loss" },
  { value: "both", label: "Measured + predicted overlay" },
];

const SWEEP_SPECS = {
  D_tokens: { label: "Dataset size D", mathLabel: "D", scale: "log", step: null },
  N_params: { label: "Non-embedding parameters N", mathLabel: "N", scale: "log", step: null },
  n_layer: { label: "Layers L", mathLabel: "L", scale: "linear", step: 1 },
  n_embd: { label: "Width d_model", mathLabel: "d_model", scale: "linear", step: 64 },
  n_head: { label: "Attention heads H", mathLabel: "H", scale: "linear", step: 1 },
  batch_size_sequences: { label: "Micro-batch B_seq", mathLabel: "B_seq", scale: "log", step: null },
};

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
    .replace(/^_+|_+$/g, "") || "chart";
}

function formatSci(x, digits = 3) {
  if (!Number.isFinite(x)) return "—";
  return Number(x).toExponential(digits);
}

function formatLoss(x) {
  if (!Number.isFinite(x)) return "—";
  return Number(x).toFixed(5);
}

function formatTableValue(key, value) {
  if (value === null || value === undefined || Number.isNaN(value)) return "—";

  if (typeof value === "number") {
    const lower = key.toLowerCase();
    if (lower.includes("loss")) return formatLoss(value);
    if (lower.includes("delta")) return `${value >= 0 ? "+" : ""}${value.toFixed(2)}%`;
    if (lower.includes("pf")) return value.toFixed(4);
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
    /(tokens?|parameters?|params?|sequences?|seqs?|steps?|gpus?|gpu|pf-days?|pflops?|layers?|heads?|width)$/i,
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

function firstDefined(...values) {
  return values.find((value) => value !== undefined && value !== null);
}

function isPlainObject(value) {
  return value !== null && typeof value === "object" && !Array.isArray(value);
}

function humanizeIdentifier(x) {
  return String(x)
    .replaceAll("_", " ")
    .replace(/\b\w/g, (c) => c.toUpperCase());
}

function stripMathDelimiters(value) {
  return String(value)
    .replace(/^\\\(/, "")
    .replace(/\\\)$/, "")
    .replace(/^\$/, "")
    .replace(/\$$/, "");
}

function MathText({ children, className = "" }) {
  const [mounted, setMounted] = useState(false);

  useEffect(() => {
    setMounted(true);
  }, []);

  if (!mounted) {
    return (
      <span className={className} suppressHydrationWarning>
        {stripMathDelimiters(children)}
      </span>
    );
  }

  return (
    <MathJax dynamic inline className={className}>
      {children}
    </MathJax>
  );
}

function classifyRegime(D, N) {
  const ratio = safeDivide(D, N);

  if (!Number.isFinite(ratio)) return "invalid";
  if (ratio < 8) return "model-rich / data-limited";
  if (ratio < 15) return "slightly data-limited";
  if (ratio <= 30) return "near D/N ≈ 20";
  if (ratio <= 80) return "data-rich / model-limited";
  return "very data-rich";
}

function HeaderName({ name }) {
  const map = {
    rank: "Rank",
    curve: "Curve",
    points: "Points",
    best_loss: "Best loss",
    best_predicted_loss: "Best predicted",
    best_x: "Best x",
    best_N: <><MathText>{"\\(N\\)"}</MathText></>,
    best_D: <><MathText>{"\\(D\\)"}</MathText></>,
    best_pf_days: "PF-days",
    D_over_N: <><MathText>{"\\(D/N\\)"}</MathText></>,
    delta_vs_baseline: "Δ vs baseline",
    regime: "Regime",
  };

  return map[name] ?? humanizeIdentifier(name);
}

function StatCard({ label, value, sub }) {
  return (
    <div className="rounded-xl border border-zinc-800 bg-zinc-950/80 px-3 py-2">
      <div className="flex items-baseline justify-between gap-3">
        <div className="min-w-0 truncate text-xs uppercase tracking-wide text-zinc-500">
          {label}
        </div>
        <div className="shrink-0 text-lg font-semibold text-zinc-100">
          {value}
        </div>
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
                <th key={key} className="whitespace-nowrap px-2 py-1.5">
                  <HeaderName name={key} />
                </th>
              ))}
            </tr>
          </thead>

          <tbody>
            {rows.length === 0 ? (
              <tr>
                <td className="px-2 py-4 text-zinc-500">No rows yet.</td>
              </tr>
            ) : (
              rows.map((row, i) => (
                <tr key={i} className="border-b border-zinc-900 last:border-0">
                  {keys.map((key) => (
                    <td key={key} className="whitespace-nowrap px-2 py-1.5 text-zinc-300">
                      {formatTableValue(key, row[key])}
                    </td>
                  ))}
                </tr>
              ))
            )}
          </tbody>
        </table>
      </div>
    </div>
  );
}

function ChartShell({ title, subtitle, children, footer, heightClass = "h-80" }) {
  return (
    <div className="rounded-2xl border border-zinc-800 bg-zinc-950/80 p-3">
      <div className="mb-2 flex items-start justify-between gap-4">
        <div>
          <h2 className="text-base font-semibold text-zinc-100">{title}</h2>
          {subtitle && <p className="mt-1 text-xs text-zinc-500">{subtitle}</p>}
        </div>
      </div>

      <div className={heightClass}>{children}</div>
      {footer && <div className="mt-2 text-xs text-zinc-500">{footer}</div>}
    </div>
  );
}

function BaseTooltip({ labelFormatter, formatter }) {
  return (
    <Tooltip
      contentStyle={{
        background: "#09090b",
        border: "1px solid #27272a",
        borderRadius: "12px",
      }}
      labelStyle={{ color: "#d4d4d8" }}
      itemStyle={{ color: "#d4d4d8" }}
      labelFormatter={labelFormatter}
      formatter={formatter}
    />
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
        {options.map((option) => (
          <option key={option.value} value={option.value}>
            {option.label}
          </option>
        ))}
      </select>
      {help && <div className="mt-2 text-xs text-zinc-500">{help}</div>}
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

function ToggleControl({ label, checked, setChecked, help }) {
  return (
    <label className="flex cursor-pointer items-start justify-between gap-4 rounded-xl border border-zinc-800 bg-zinc-950/70 p-4">
      <div>
        <div className="text-sm font-medium text-zinc-100">{label}</div>
        {help && <div className="mt-1 text-xs text-zinc-500">{help}</div>}
      </div>

      <input
        type="checkbox"
        checked={checked}
        onChange={(e) => setChecked(e.target.checked)}
        className="mt-1 h-4 w-4 accent-blue-500"
      />
    </label>
  );
}

function normalizeSweepValue(variable, value) {
  const spec = SWEEP_SPECS[variable];
  if (!spec) return value;

  if (spec.scale === "log") return Math.max(value, 1e-300);

  const step = spec.step ?? 1;
  const rounded = Math.round(value / step) * step;

  if (["n_layer", "n_head", "n_embd", "batch_size_sequences"].includes(variable)) {
    return Math.max(1, Math.round(rounded));
  }

  return rounded;
}

function makeSweepValuesFromAxis(axis, fallbackPoints = 32) {
  if (!axis?.variable) return [];

  const count = clamp(Math.round(numberFromAny(axis.points, fallbackPoints)), 2, 256);
  const min = numberFromAny(axis.min, axis.actual_min, axis.value_min);
  const max = numberFromAny(axis.max, axis.actual_max, axis.value_max);

  if (!Number.isFinite(min) || !Number.isFinite(max)) return [];

  const lo = Math.min(min, max);
  const hi = Math.max(min, max);
  const spec = SWEEP_SPECS[axis.variable];
  const values = [];

  for (let i = 0; i < count; i++) {
    const f = count === 1 ? 0 : i / (count - 1);
    const value = spec?.scale === "log"
      ? lo * (hi / lo) ** f
      : lo + (hi - lo) * f;
    values.push(normalizeSweepValue(axis.variable, value));
  }

  const seen = new Set();
  return values.filter((value) => {
    const key = String(value);
    if (seen.has(key)) return false;
    seen.add(key);
    return true;
  });
}

function applySweepVariableToConfig(cfg, variable, value) {
  if (variable === "D_tokens") cfg.D_tokens = value;
  if (variable === "N_params") cfg.N_params = value;
  if (variable === "n_layer") cfg.n_layer = Math.max(1, Math.round(value));
  if (variable === "n_embd") cfg.n_embd = Math.max(1, Math.round(value));
  if (variable === "n_head") cfg.n_head = Math.max(1, Math.round(value));
  if (variable === "batch_size_sequences") cfg.batch_size_sequences = Math.max(1, Math.round(value));
  return cfg;
}

function makeBaseConfigFromDesignerProtocol(protocol) {
  const constants = protocol.constants ?? {};
  const inputPanel = protocol.input_panel ?? {};
  const architecture = inputPanel.architecture ?? {};
  const datasetTraining = inputPanel.dataset_and_training ?? {};

  const nLayer = Math.round(numberFromAny(architecture.n_layer, constants.n_layer, 8));
  const nHead = Math.round(numberFromAny(architecture.n_head, constants.n_head, 8));
  const nEmbd = Math.round(numberFromAny(architecture.n_embd, constants.n_embd, 512));
  const mlpRatio = numberFromAny(architecture.mlp_ratio, constants.mlp_ratio, 4);
  const architectureN = gpt_like_nonembedding_params(nLayer, nEmbd, mlpRatio, false);

  return {
    n_layer: nLayer,
    n_head: nHead,
    n_embd: nEmbd,
    mlp_ratio: mlpRatio,
    use_N_override: Boolean(firstDefined(architecture.use_N_override, constants.use_N_override, false)),
    N_params: numberFromAny(
      architecture.N_params_override,
      constants.N_params_override,
      architecture.estimated_architecture_N_params,
      architectureN
    ),
    D_tokens: numberFromAny(datasetTraining.D_tokens, constants.D_tokens, 35e9),
    seq_len: Math.round(numberFromAny(datasetTraining.seq_len, constants.seq_len, 1024)),
    batch_size_sequences: Math.round(numberFromAny(datasetTraining.batch_size_sequences, constants.batch_size_sequences, 32)),
    grad_accum: Math.round(numberFromAny(datasetTraining.grad_accum, constants.grad_accum, 96)),
    num_gpus: Math.round(numberFromAny(datasetTraining.num_gpus, constants.num_gpus, 4)),
    batch_size_is_per_gpu: Boolean(firstDefined(datasetTraining.batch_size_is_per_gpu, constants.batch_size_is_per_gpu, false)),
  };
}

function evaluateDesignerPoint({ cfg, activeVariables, x, y = null }) {
  const active = new Set(activeVariables);
  const architectureDrivenN = gpt_like_nonembedding_params(
    cfg.n_layer,
    cfg.n_embd,
    cfg.mlp_ratio,
    false
  );

  const N = active.has("N_params")
    ? cfg.N_params
    : cfg.use_N_override && !active.has("n_layer") && !active.has("n_embd")
      ? cfg.N_params
      : architectureDrivenN;

  const D = cfg.D_tokens;
  const globalBatchSequences = cfg.batch_size_is_per_gpu
    ? cfg.batch_size_sequences * cfg.grad_accum * cfg.num_gpus
    : cfg.batch_size_sequences * cfg.grad_accum;
  const globalBatchTokens = globalBatchSequences * cfg.seq_len;
  const predictedLoss = kaplan_loss_ND(N, D);
  const pfDays = flops_to_pf_days(training_flops(N, D));
  const DOverN = tokens_per_parameter(D, N);

  return {
    x,
    y,
    N,
    D,
    D_over_N: DOverN,
    loss: predictedLoss,
    predicted_loss: predictedLoss,
    measured_loss: NaN,
    pf_days: pfDays,
    steps: safeDivide(D, globalBatchTokens),
    L: cfg.n_layer,
    H: cfg.n_head,
    dModel: cfg.n_embd,
    regime: classifyRegime(D, N),
  };
}

function designerProtocolToCurves(protocol, labelHint = "Designer sweep") {
  const sweep = protocol.sweep ?? protocol.input_panel?.sweep ?? {};
  const xAxis = sweep.axes?.x ?? sweep.x_axis ?? protocol.input_panel?.sweep?.x_axis;
  const yAxis = sweep.axes?.y ?? sweep.y_axis ?? protocol.input_panel?.sweep?.y_axis;

  if (!xAxis?.variable) {
    throw new Error("Designer protocol is missing sweep axis information.");
  }

  const mode = sweep.mode === "2d" && yAxis?.variable ? "2d" : "1d";
  const baseCfg = makeBaseConfigFromDesignerProtocol(protocol);
  const xValues = makeSweepValuesFromAxis(xAxis, sweep.x_points ?? xAxis.points ?? 32);

  if (mode === "1d") {
    const activeVariables = [xAxis.variable];
    const points = xValues.map((x) => {
      const cfg = { ...baseCfg };
      applySweepVariableToConfig(cfg, xAxis.variable, x);
      return evaluateDesignerPoint({ cfg, activeVariables, x });
    });

    return [
      {
        id: stableId(labelHint, 0),
        name: labelHint,
        color: COLORS[0],
        source: "designer protocol",
        points,
      },
    ];
  }

  const yValues = makeSweepValuesFromAxis(yAxis, sweep.y_points ?? yAxis.points ?? 12);
  const activeVariables = [xAxis.variable, yAxis.variable];

  return yValues.map((y, index) => {
    const points = xValues.map((x) => {
      const cfg = { ...baseCfg };
      applySweepVariableToConfig(cfg, xAxis.variable, x);
      applySweepVariableToConfig(cfg, yAxis.variable, y);
      return evaluateDesignerPoint({ cfg, activeVariables, x, y });
    });

    const yLabel = `${yAxis.variable}=${human_num(y)}`;

    return {
      id: stableId(`${labelHint}_${yLabel}`, index),
      name: `${labelHint}: ${yLabel}`,
      color: COLORS[index % COLORS.length],
      source: "designer 2D protocol",
      points,
    };
  });
}

function normalizePoint(rawPoint, index = 0) {
  const metadata = rawPoint?.designer_metadata ?? {};
  const trainingConfig = rawPoint?.training_config ?? {};

  const N = numberFromAny(
    rawPoint.N,
    rawPoint.N_params,
    rawPoint.num_params,
    rawPoint.non_embedding_params,
    rawPoint.estimated_non_embedding_params,
    metadata.estimated_non_embedding_params,
    metadata.N_params
  );

  const D = numberFromAny(
    rawPoint.D,
    rawPoint.D_tokens,
    rawPoint.num_train_tokens,
    rawPoint.tokens,
    rawPoint.dataset_tokens,
    metadata.D_tokens
  );

  const importedLoss = numberFromAny(
    rawPoint.loss,
    rawPoint.val_loss,
    rawPoint.min_val_loss,
    rawPoint.min_saved_val_loss,
    rawPoint.validation_loss,
    rawPoint.y,
    metadata.measured_loss
  );

  const predictedLoss = Number.isFinite(N) && Number.isFinite(D)
    ? kaplan_loss_ND(N, D)
    : numberFromAny(rawPoint.predicted_loss, metadata.predicted_kaplan_loss);

  const loss = Number.isFinite(importedLoss) ? importedLoss : predictedLoss;

  const flops = numberFromAny(rawPoint.flops, rawPoint.training_flops, rawPoint.compute_flops, rawPoint.compute);
  const pfDays = numberFromAny(
    rawPoint.pf_days,
    rawPoint.compute_pf_days,
    rawPoint.training_compute_pf_days,
    rawPoint.Cmin,
    metadata.training_compute_pf_days,
    Number.isFinite(flops) ? flops_to_pf_days(flops) : NaN,
    Number.isFinite(N) && Number.isFinite(D) ? flops_to_pf_days(training_flops(N, D)) : NaN
  );

  const DOverN = numberFromAny(
    rawPoint.D_over_N,
    rawPoint.tokens_per_parameter,
    metadata.D_over_N,
    Number.isFinite(D) && Number.isFinite(N) ? tokens_per_parameter(D, N) : NaN
  );

  return {
    index,
    x: numberFromAny(rawPoint.x, rawPoint.variable, D, N, index + 1),
    N,
    D,
    D_over_N: DOverN,
    loss,
    measured_loss: importedLoss,
    predicted_loss: predictedLoss,
    pf_days: pfDays,
    steps: numberFromAny(rawPoint.steps, rawPoint.optimizer_steps, rawPoint.max_iters, rawPoint.iters_saved, rawPoint.iters_trained, metadata.optimizer_steps_from_D),
    L: numberFromAny(rawPoint.L, rawPoint.n_layer, trainingConfig.n_layer),
    H: numberFromAny(rawPoint.H, rawPoint.n_head, trainingConfig.n_head),
    dModel: numberFromAny(rawPoint.dModel, rawPoint.n_embd, trainingConfig.n_embd),
    regime: rawPoint.regime ?? metadata.regime ?? classifyRegime(D, N),
    raw: rawPoint,
  };
}

function cleanCurve(rawCurve, curveIndex = 0) {
  const rawPoints = rawCurve.points ?? rawCurve.rows ?? rawCurve.data ?? rawCurve.runs ?? [];
  const curveName = rawCurve.name ?? rawCurve.label ?? rawCurve.title ?? `Curve ${curveIndex + 1}`;

  const points = rawPoints
    .map((point, index) => normalizePoint(point, index))
    .filter((point) => Number.isFinite(point.N) && Number.isFinite(point.D) && Number.isFinite(point.loss))
    .sort((a, b) => {
      const ax = Number.isFinite(a.x) ? a.x : a.index;
      const bx = Number.isFinite(b.x) ? b.x : b.index;
      return ax - bx;
    });

  return {
    id: rawCurve.id ?? stableId(curveName, curveIndex),
    name: curveName,
    color: rawCurve.color ?? COLORS[curveIndex % COLORS.length],
    source: rawCurve.source ?? "curve bundle",
    hidden: Boolean(rawCurve.hidden),
    points,
  };
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
    // Ignore completely empty trailing rows.
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

function rowValue(row, columnName) {
  if (!row || !columnName) return undefined;
  if (Object.prototype.hasOwnProperty.call(row, columnName)) return row[columnName];

  const normalizedTarget = String(columnName).trim().toLowerCase();
  const actualKey = Object.keys(row).find((key) => key.trim().toLowerCase() === normalizedTarget);
  return actualKey ? row[actualKey] : undefined;
}

function formatGroupValue(column, value) {
  const parsed = parseSmartNumber(value);
  const lower = String(column).toLowerCase();

  if (!Number.isFinite(parsed)) return String(value ?? "").trim() || "missing";
  if (lower.includes("lr") || lower.includes("learning_rate")) return formatSci(parsed, 2);
  if (lower.includes("tokens") || lower.includes("params") || lower.includes("size") || lower.includes("compute")) return human_num(parsed);
  return String(value ?? "").trim() || human_num(parsed);
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
  };

  return map[column] ?? column;
}

function parseGroupColumns(groupColumnsText) {
  return String(groupColumnsText ?? "")
    .split(",")
    .map((column) => column.trim())
    .filter(Boolean);
}

function csvRowsToCurves(rows, options = {}) {
  const groupColumns = parseGroupColumns(options.groupColumns);

  if (rows.length === 0) return [];

  if (groupColumns.length === 0) {
    return [cleanCurve({
      name: options.labelHint ?? "Pasted CSV",
      source: "pasted CSV",
      points: rows,
    }, 0)];
  }

  const groups = new Map();

  for (const row of rows) {
    const groupParts = groupColumns.map((column) => {
      const value = rowValue(row, column);
      return `${niceGroupColumnName(column)}=${formatGroupValue(column, value)}`;
    });

    const groupName = groupParts.join(", ") || options.labelHint || "Pasted CSV";

    if (!groups.has(groupName)) groups.set(groupName, []);
    groups.get(groupName).push(row);
  }

  return Array.from(groups.entries()).map(([name, points], index) =>
    cleanCurve({ name, points, source: "pasted CSV" }, index)
  );
}

function normalizeImportedText(text, options = {}) {
  const trimmed = String(text ?? "").trim();
  if (!trimmed) throw new Error("Nothing to import.");

  if (trimmed.startsWith("{") || trimmed.startsWith("[")) {
    return normalizeImportedJson(JSON.parse(trimmed), options.labelHint ?? "Pasted JSON");
  }

  const rows = parseCsvText(trimmed);
  return csvRowsToCurves(rows, options);
}

function flatRowsToCurves(rows) {
  const groups = new Map();

  for (const row of rows) {
    const name = row.curve ?? row.curve_name ?? row.group ?? row.run_group ?? row.name ?? "Imported curve";
    if (!groups.has(name)) groups.set(name, []);
    groups.get(name).push(row);
  }

  return Array.from(groups.entries()).map(([name, points], index) =>
    cleanCurve({ name, points, source: "flat row import" }, index)
  );
}

function normalizeImportedJson(value, labelHint = "Imported") {
  if (Array.isArray(value)) return flatRowsToCurves(value);

  if (!isPlainObject(value)) {
    throw new Error("Top-level JSON must be an object or array.");
  }

  if (value.protocol === "particleGPT.designer_sweep" || value.route === "/designer" || value.sweep?.x_variable) {
    return designerProtocolToCurves(value, labelHint);
  }

  if (Array.isArray(value.curves)) {
    return value.curves.map((curve, index) => cleanCurve(curve, index));
  }

  const rows = value.points ?? value.rows ?? value.data ?? value.runs;
  if (Array.isArray(rows)) {
    return [cleanCurve({ name: value.name ?? labelHint, points: rows, source: "single curve import" }, 0)];
  }

  throw new Error("Could not find curves, points, rows, data, runs, or a designer sweep protocol.");
}

function makeDemoCurves() {
  const datasetSizes = [8.5e9, 17e9, 35e9];
  const modelSizes = [2e7, 3.9e7, 7e7, 1.08e8, 2.09e8];

  return datasetSizes.map((D, curveIndex) => {
    const points = modelSizes.map((N, pointIndex) => {
      const predicted = kaplan_loss_ND(N, D);
      const smallOffset = 0.025 * Math.sin(curveIndex * 1.7 + pointIndex * 0.9);
      const loss = predicted + smallOffset;

      return {
        N,
        D,
        D_over_N: tokens_per_parameter(D, N),
        loss,
        measured_loss: loss,
        predicted_loss: predicted,
        pf_days: flops_to_pf_days(training_flops(N, D)),
        steps: D / (32 * 96 * 1024),
        regime: classifyRegime(D, N),
      };
    });

    return {
      id: `demo_${curveIndex}`,
      name: `${human_num(D)} token curve`,
      color: COLORS[curveIndex],
      source: "demo",
      hidden: false,
      points,
    };
  });
}

function finitePositiveDomain(curves, key, pad = 1.08) {
  const values = curves
    .flatMap((curve) => curve.points.map((point) => point[key]))
    .filter((value) => Number.isFinite(value) && value > 0);

  if (values.length === 0) return [1, 10];

  const lo = Math.min(...values);
  const hi = Math.max(...values);

  if (Math.abs(Math.log10(hi) - Math.log10(lo)) < 1e-9) return [lo / pad, hi * pad];
  return [lo / pad, hi * pad];
}

function finiteYDomain(curves, key, padFraction = 0.08) {
  const values = curves
    .flatMap((curve) => curve.points.map((point) => point[key]))
    .filter((value) => Number.isFinite(value));

  if (values.length === 0) return ["dataMin", "dataMax"];

  const lo = Math.min(...values);
  const hi = Math.max(...values);

  if (Math.abs(hi - lo) < 1e-12) {
    const pad = Math.max(Math.abs(lo) * 0.05, 0.05);
    return [lo - pad, hi + pad];
  }

  const pad = (hi - lo) * padFraction;
  return [lo - pad, hi + pad];
}

function makeLogAxisTicks(domain, maxTicks = 7) {
  if (!Array.isArray(domain) || domain.length !== 2) return undefined;

  const lo = Math.max(Math.min(domain[0], domain[1]), 1e-300);
  const hi = Math.max(Math.max(domain[0], domain[1]), lo * 1.0001);
  const expLo = Math.floor(Math.log10(lo));
  const expHi = Math.ceil(Math.log10(hi));
  const ticks = [];

  for (let exp = expLo; exp <= expHi; exp++) {
    const decade = 10 ** exp;
    for (const mantissa of [1, 2, 5]) {
      const tick = mantissa * decade;
      if (tick >= lo / 1.000001 && tick <= hi * 1.000001) ticks.push(tick);
    }
  }

  if (ticks.length <= maxTicks) return ticks;

  const out = [];
  const seen = new Set();
  for (let i = 0; i < maxTicks; i++) {
    const index = Math.round((i * (ticks.length - 1)) / Math.max(maxTicks - 1, 1));
    if (!seen.has(index)) {
      out.push(ticks[index]);
      seen.add(index);
    }
  }

  return out;
}

function makeChartData(curves, xMetric, yMetric) {
  const rowMap = new Map();
  const series = [];

  curves.forEach((curve, curveIndex) => {
    const key = `curve_${curveIndex}`;
    series.push({
      key,
      name: curve.name,
      color: curve.color ?? COLORS[curveIndex % COLORS.length],
    });

    curve.points.forEach((point) => {
      const x = point[xMetric];
      const y = point[yMetric];
      if (!Number.isFinite(x) || !Number.isFinite(y)) return;

      const xKey = Number(x).toPrecision(12);
      const existing = rowMap.get(xKey) ?? { x };
      existing[key] = y;
      rowMap.set(xKey, existing);
    });
  });

  return {
    data: Array.from(rowMap.values()).sort((a, b) => a.x - b.x),
    series,
  };
}

function makeCurveStats(curves, xMetric, baselineCurveId) {
  const baselineCurve = curves.find((curve) => curve.id === baselineCurveId) ?? curves[0] ?? null;
  const baselineBest = baselineCurve?.points.reduce((best, point) => {
    if (!best || point.loss < best.loss) return point;
    return best;
  }, null);

  return curves.map((curve) => {
    const best = curve.points.reduce((currentBest, point) => {
      if (!currentBest || point.loss < currentBest.loss) return point;
      return currentBest;
    }, null);

    const delta = best && baselineBest
      ? 100 * (best.loss - baselineBest.loss) / Math.abs(baselineBest.loss)
      : NaN;

    return {
      curve: curve.name,
      points: curve.points.length,
      best_loss: best?.loss ?? NaN,
      best_predicted_loss: best?.predicted_loss ?? NaN,
      best_x: best?.[xMetric] ?? NaN,
      best_N: best?.N ?? NaN,
      best_D: best?.D ?? NaN,
      best_pf_days: best?.pf_days ?? NaN,
      D_over_N: best?.D_over_N ?? NaN,
      delta_vs_baseline: delta,
      regime: best?.regime ?? "—",
    };
  });
}

function makeParetoFrontier(points) {
  const clean = points
    .filter((point) => Number.isFinite(point.pf_days) && point.pf_days > 0 && Number.isFinite(point.loss))
    .sort((a, b) => a.pf_days - b.pf_days);

  const frontier = [];
  let bestLoss = Infinity;

  for (const point of clean) {
    if (point.loss < bestLoss) {
      frontier.push(point);
      bestLoss = point.loss;
    }
  }

  return frontier;
}

function MultiCurveChart({
  title,
  subtitle,
  curves,
  xMetric,
  yMetric,
  yLabel,
  logX = true,
  logY = false,
  referenceY = null,
}) {
  const xLabel = X_METRIC_OPTIONS.find((option) => option.value === xMetric)?.label ?? xMetric;

  const traces = useMemo(() => {
    return curves.map((curve, index) => {
      const cleanPoints = curve.points.filter((point) => (
        Number.isFinite(point?.[xMetric]) &&
        Number.isFinite(point?.[yMetric]) &&
        (!logX || point[xMetric] > 0) &&
        (!logY || point[yMetric] > 0)
      ));

      return {
        type: "scatter",
        mode: "lines+markers",
        name: curve.name,
        x: cleanPoints.map((point) => point[xMetric]),
        y: cleanPoints.map((point) => point[yMetric]),
        text: cleanPoints.map((point) => [
          `curve: ${curve.name}`,
          `N: ${human_num(point.N)}`,
          `D: ${human_num(point.D)}`,
          `loss: ${formatLoss(point.loss)}`,
          `D/N: ${human_num(point.D_over_N)}`,
          `PF-days: ${Number.isFinite(point.pf_days) ? point.pf_days.toFixed(4) : "—"}`,
          `regime: ${point.regime ?? "—"}`,
        ].join("<br>")),
        hovertemplate: "%{text}<extra></extra>",
        line: {
          color: curve.color ?? COLORS[index % COLORS.length],
          width: 2.4,
        },
        marker: {
          color: curve.color ?? COLORS[index % COLORS.length],
          size: 7,
          line: {
            color: "rgba(255,255,255,0.35)",
            width: 1,
          },
        },
      };
    });
  }, [curves, xMetric, yMetric, logX, logY]);

  const shapes = Number.isFinite(referenceY)
    ? [
        {
          type: "line",
          xref: "paper",
          x0: 0,
          x1: 1,
          yref: "y",
          y0: referenceY,
          y1: referenceY,
          line: {
            color: "#fbbf24",
            width: 2,
            dash: "dash",
          },
        },
      ]
    : [];

  const layout = {
    autosize: true,
    paper_bgcolor: "rgba(0,0,0,0)",
    plot_bgcolor: "rgba(0,0,0,0)",
    font: {
      color: "#d4d4d8",
      size: 12,
    },
    margin: {
      l: 58,
      r: 22,
      t: 8,
      b: 54,
    },
    hovermode: "closest",
    dragmode: "zoom",
    legend: {
      orientation: "h",
      x: 0,
      y: 1.14,
      font: { size: 11, color: "#d4d4d8" },
    },
    xaxis: {
      title: {
        text: xLabel,
        font: { color: "#a1a1aa", size: 12 },
      },
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
      title: {
        text: yLabel,
        font: { color: "#a1a1aa", size: 12 },
      },
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

  const config = {
    responsive: true,
    scrollZoom: true,
    displaylogo: false,
    modeBarButtonsToAdd: ["drawline", "eraseshape"],
    toImageButtonOptions: {
      format: "png",
      filename: sanitizeFilenamePart(title),
      height: 900,
      width: 1400,
      scale: 2,
    },
  };

  return (
    <ChartShell
      title={title}
      subtitle={subtitle}
      footer="Drag to box-zoom. Use the modebar to pan, zoom, autoscale, reset axes, or download a PNG. Mouse wheel zoom is enabled."
    >
      <Plot
        data={traces}
        layout={layout}
        config={config}
        useResizeHandler
        className="h-full w-full"
        style={{ width: "100%", height: "100%" }}
      />
    </ChartShell>
  );
}

function ImportPanel({ onAddCurves, onReplaceCurves }) {
  const [textValue, setTextValue] = useState("");
  const [csvGroupColumns, setCsvGroupColumns] = useState("num_train_tokens, learning_rate");
  const [status, setStatus] = useState(null);

  const handleCurves = (nextCurves, mode, sourceLabel = "input") => {
    if (nextCurves.length === 0) {
      throw new Error("Import produced zero usable curves. Check that the CSV has num_params, num_train_tokens, and min_saved_val_loss columns.");
    }

    if (mode === "replace") onReplaceCurves(nextCurves);
    else onAddCurves(nextCurves);

    const pointCount = nextCurves.reduce((sum, curve) => sum + curve.points.length, 0);

    setStatus({
      type: "success",
      text: `${mode === "replace" ? "Replaced with" : "Added"} ${nextCurves.length} curve(s), ${pointCount} point(s), from ${sourceLabel}.`,
    });
  };

  const applyText = (mode) => {
    try {
      const nextCurves = normalizeImportedText(textValue, {
        groupColumns: csvGroupColumns,
        labelHint: "Pasted data",
      });
      handleCurves(nextCurves, mode, textValue.trim().startsWith("{") || textValue.trim().startsWith("[") ? "pasted JSON" : "pasted CSV");
    } catch (error) {
      setStatus({ type: "error", text: error?.message ?? "Import failed." });
    }
  };

  const readFile = (file, mode = "add") => {
    if (!file) return;

    const reader = new FileReader();
    reader.onload = () => {
      const nextText = String(reader.result ?? "");
      setTextValue(nextText);

      try {
        const nextCurves = normalizeImportedText(nextText, {
          groupColumns: csvGroupColumns,
          labelHint: file.name.replace(/\.(json|csv|txt)$/i, ""),
        });
        handleCurves(nextCurves, mode, file.name);
      } catch (error) {
        setStatus({ type: "error", text: error?.message ?? `Could not import ${file.name}.` });
      }
    };

    reader.onerror = () => {
      setStatus({ type: "error", text: `Could not read ${file.name}.` });
    };

    reader.readAsText(file);
  };

  return (
    <div className="rounded-2xl border border-zinc-800 bg-zinc-950/80 p-3">
      <div className="mb-2">
        <h2 className="text-sm font-semibold uppercase tracking-wide text-zinc-400">
          Import experimental curves
        </h2>
        <p className="mt-1 text-xs leading-relaxed text-zinc-500">
          Paste the output of <code className="text-zinc-300">df.to_csv(index=False)</code>. JSON still works, but CSV is the main path.
        </p>
      </div>

      <TextControl
        label="CSV group columns"
        value={csvGroupColumns}
        setValue={setCsvGroupColumns}
        placeholder="num_train_tokens, learning_rate"
        help="Rows with the same values in these columns become one curve. Use num_train_tokens for loss-vs-N dataset curves. Leave blank to make one curve from all rows."
      />

      <textarea
        value={textValue}
        onChange={(event) => setTextValue(event.target.value)}
        placeholder={"model_name,vocab_size,max_sequence_length,num_train_tokens,...,num_params,min_saved_val_loss,...\nmodel_exp12_...,244,170,6800000,...,1639936,1.84559,..."}
        className="mt-3 h-56 w-full resize-y rounded-xl border border-zinc-800 bg-zinc-950 p-3 text-xs leading-relaxed text-zinc-300 outline-none transition placeholder:text-zinc-700 focus:border-blue-500 focus:ring-1 focus:ring-blue-500"
        spellCheck={false}
      />

      <div className="mt-2 grid grid-cols-1 gap-2">
        <input
          type="file"
          accept="application/json,.json,text/csv,.csv,text/plain,.txt"
          onChange={(event) => readFile(event.target.files?.[0], "add")}
          className="block min-w-0 text-xs text-zinc-500 file:mr-3 file:rounded-lg file:border file:border-zinc-700 file:bg-zinc-900 file:px-3 file:py-1.5 file:text-xs file:font-semibold file:text-zinc-200 hover:file:border-blue-500/60"
        />

        <div className="flex gap-2">
          <button
            type="button"
            onClick={() => applyText("add")}
            disabled={!textValue.trim()}
            className="flex-1 rounded-xl border border-blue-500/40 bg-blue-500/15 px-4 py-2 text-sm font-semibold text-blue-100 transition hover:border-blue-400 hover:bg-blue-500/25 disabled:cursor-not-allowed disabled:opacity-50"
          >
            Add data
          </button>

          <button
            type="button"
            onClick={() => applyText("replace")}
            disabled={!textValue.trim()}
            className="flex-1 rounded-xl border border-zinc-700 bg-zinc-900 px-4 py-2 text-sm font-semibold text-zinc-200 transition hover:border-blue-500/60 hover:text-blue-100 disabled:cursor-not-allowed disabled:opacity-50"
          >
            Replace
          </button>
        </div>
      </div>

      {status && (
        <div
          className={`mt-2 rounded-xl border px-3 py-2 text-xs leading-relaxed ${
            status.type === "success"
              ? "border-emerald-500/30 bg-emerald-500/10 text-emerald-200"
              : "border-amber-500/30 bg-amber-500/10 text-amber-200"
          }`}
        >
          {status.text}
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
      <div className="mb-2">
        <h2 className="text-sm font-semibold uppercase tracking-wide text-zinc-400">
          Active curves
        </h2>
        <p className="mt-1 text-xs text-zinc-500">
          Hide curves without deleting them, or remove noisy imports.
        </p>
      </div>

      <div className="space-y-2">
        {curves.length === 0 && (
          <div className="rounded-xl border border-zinc-800 bg-zinc-950 p-3 text-xs text-zinc-500">
            No curves imported yet.
          </div>
        )}

        {curves.map((curve, index) => (
          <div key={curve.id} className="rounded-xl border border-zinc-800 bg-zinc-950/70 p-3">
            <div className="flex items-start justify-between gap-2">
              <div className="min-w-0">
                <div className="flex items-center gap-2">
                  <span
                    className="h-2.5 w-2.5 shrink-0 rounded-full"
                    style={{ background: curve.color ?? COLORS[index % COLORS.length] }}
                  />
                  <div className="truncate text-sm font-medium text-zinc-100">
                    {curve.name}
                  </div>
                </div>
                <div className="mt-1 text-xs text-zinc-500">
                  {curve.points.length} points · {curve.source}
                </div>
              </div>

              <div className="flex shrink-0 gap-1">
                <button
                  type="button"
                  onClick={() => toggleCurve(curve.id)}
                  className="rounded-lg border border-zinc-700 bg-zinc-900 px-2 py-1 text-[11px] text-zinc-300 hover:border-blue-500/60"
                >
                  {curve.hidden ? "Show" : "Hide"}
                </button>
                <button
                  type="button"
                  onClick={() => removeCurve(curve.id)}
                  className="rounded-lg border border-zinc-700 bg-zinc-900 px-2 py-1 text-[11px] text-zinc-300 hover:border-red-500/60"
                >
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

function ExampleFormatPanel() {
  const csvExample = `model_name,num_train_tokens,n_layer,n_head,n_embd,num_params,iters_saved,min_saved_train_loss,min_saved_val_loss,world_size,tokens_per_update,compute
model_exp12_100k_tp1,6800000,2,4,256,1639936,19648,1.8195968,1.8455935,4,2088960,403856078174945280
model_exp12_100k_tp6,6800000,4,4,256,3214080,11552,1.8039595,1.8356986,4,2088960,465366628800921600
model_exp12_500k_tp1,34000000,2,4,256,1639936,74196,1.8184350,1.8242362,4,2088960,1525066448303554560`;

  return (
    <details className="rounded-2xl border border-zinc-800 bg-zinc-950/80 p-3">
      <summary className="cursor-pointer text-sm font-semibold uppercase tracking-wide text-zinc-400">
        CSV format
      </summary>

      <p className="mt-2 text-xs leading-relaxed text-zinc-500">
        In Python, use <code className="text-zinc-300">df.to_csv(index=False)</code>, then paste the result above.
        The important columns are <code className="text-zinc-300">num_train_tokens</code>, <code className="text-zinc-300">num_params</code>,
        and <code className="text-zinc-300">min_saved_val_loss</code>. Optional useful columns include <code className="text-zinc-300">compute</code>,
        <code className="text-zinc-300">iters_saved</code>, <code className="text-zinc-300">learning_rate</code>, and architecture fields.
      </p>

      <pre className="mt-3 max-h-80 overflow-auto rounded-xl border border-zinc-800 bg-zinc-950 p-3 text-xs leading-relaxed text-zinc-300">
        {csvExample}
      </pre>
    </details>
  );
}

export default function ScalingLawComparisonPage() {
  const layoutRef = useRef(null);
  const [sidebarWidth, setSidebarWidth] = useState(440);
  const [isResizingSidebar, setIsResizingSidebar] = useState(false);
  const [curves, setCurves] = useState(() => makeDemoCurves());
  const [xMetric, setXMetric] = useState("N");
  const [lossSource, setLossSource] = useState("measured");
  const [useLogX, setUseLogX] = useState(true);
  const [baselineCurveId, setBaselineCurveId] = useState("");

  useEffect(() => {
    if (!isResizingSidebar) return;

    const oldCursor = document.body.style.cursor;
    const oldUserSelect = document.body.style.userSelect;

    const onPointerMove = (event) => {
      const rect = layoutRef.current?.getBoundingClientRect();
      const left = rect?.left ?? 0;
      const layoutWidth = rect?.width ?? window.innerWidth;
      const maxSidebarWidth = Math.max(440, Math.min(820, layoutWidth - 420));
      const nextWidth = clamp(event.clientX - left, 440, maxSidebarWidth);
      setSidebarWidth(nextWidth);
    };

    const onPointerUp = () => setIsResizingSidebar(false);

    document.body.style.cursor = "col-resize";
    document.body.style.userSelect = "none";

    window.addEventListener("pointermove", onPointerMove);
    window.addEventListener("pointerup", onPointerUp);
    window.addEventListener("pointercancel", onPointerUp);

    return () => {
      document.body.style.cursor = oldCursor;
      document.body.style.userSelect = oldUserSelect;
      window.removeEventListener("pointermove", onPointerMove);
      window.removeEventListener("pointerup", onPointerUp);
      window.removeEventListener("pointercancel", onPointerUp);
    };
  }, [isResizingSidebar]);

  const visibleCurves = useMemo(
    () => curves.filter((curve) => !curve.hidden && curve.points.length > 0),
    [curves]
  );

  const plottedLossCurves = useMemo(() => {
    if (lossSource === "measured") return visibleCurves;

    const predicted = visibleCurves.map((curve) => ({
      ...curve,
      id: `${curve.id}_predicted`,
      name: lossSource === "both" ? `${curve.name} predicted` : curve.name,
      color: lossSource === "both" ? "#a1a1aa" : curve.color,
      points: curve.points
        .filter((point) => Number.isFinite(point.predicted_loss))
        .map((point) => ({ ...point, loss: point.predicted_loss })),
    }));

    if (lossSource === "predicted") return predicted;
    return [...visibleCurves, ...predicted];
  }, [visibleCurves, lossSource]);

  const baselineOptions = useMemo(() => {
    return [
      { value: "", label: "First visible curve" },
      ...visibleCurves.map((curve) => ({ value: curve.id, label: curve.name })),
    ];
  }, [visibleCurves]);

  const allPoints = useMemo(() => {
    return visibleCurves.flatMap((curve) => curve.points.map((point) => ({ ...point, curve: curve.name })));
  }, [visibleCurves]);

  const globalBest = useMemo(() => {
    return allPoints.reduce((best, point) => !best || point.loss < best.loss ? point : best, null);
  }, [allPoints]);

  const cheapest = useMemo(() => {
    return allPoints.reduce((best, point) => !best || point.pf_days < best.pf_days ? point : best, null);
  }, [allPoints]);

  const bestBalanced = useMemo(() => {
    return allPoints.reduce((best, point) => {
      const distance = Math.abs(Math.log((point.D_over_N || NaN) / 20));
      const bestDistance = Math.abs(Math.log((best?.D_over_N || NaN) / 20));
      return !best || distance < bestDistance ? point : best;
    }, null);
  }, [allPoints]);

  const statsRows = useMemo(() => {
    const baselineId = baselineCurveId || visibleCurves[0]?.id;
    return makeCurveStats(visibleCurves, xMetric, baselineId);
  }, [visibleCurves, xMetric, baselineCurveId]);

  const rankedRows = useMemo(() => {
    return statsRows
      .slice()
      .sort((a, b) => a.best_loss - b.best_loss)
      .map((row, index) => ({ rank: index + 1, ...row }));
  }, [statsRows]);

  const paretoCurves = useMemo(() => {
    return visibleCurves.map((curve) => ({ ...curve, points: makeParetoFrontier(curve.points) }));
  }, [visibleCurves]);

  const addCurves = (nextCurves) => {
    setCurves((prev) => {
      const usedIds = new Set(prev.map((curve) => curve.id));
      const offset = prev.length;
      const normalized = nextCurves.map((curve, index) => {
        let id = curve.id ?? stableId(curve.name, offset + index);
        if (usedIds.has(id)) id = `${id}_${offset + index}`;
        usedIds.add(id);
        return { ...curve, id, color: curve.color ?? COLORS[(offset + index) % COLORS.length] };
      });
      return [...prev, ...normalized];
    });
  };

  const replaceCurves = (nextCurves) => {
    setCurves(nextCurves.map((curve, index) => ({
      ...curve,
      id: curve.id ?? stableId(curve.name, index),
      color: curve.color ?? COLORS[index % COLORS.length],
    })));
    setBaselineCurveId("");
  };

  return (
    <MathJaxContext config={MATHJAX_CONFIG}>
      <div className="min-h-screen bg-zinc-950 text-zinc-100">
        <div ref={layoutRef} className="min-h-screen lg:flex lg:items-stretch">
          <aside
            className="border-b border-zinc-800 bg-zinc-950/95 p-5 lg:sticky lg:top-0 lg:h-screen lg:shrink-0 lg:overflow-y-auto lg:border-b-0 lg:border-r"
            style={{ width: `min(100%, ${sidebarWidth}px)`, flexBasis: `${sidebarWidth}px` }}
          >
            <div className="mb-6">
              <h1 className="text-2xl font-semibold tracking-tight">Curve comparison</h1>
              <p className="mt-2 text-sm text-zinc-500">
                Paste pandas CSV exports or sweep protocols, then compare loss, compute, and data/model balance.
              </p>
            </div>

            <div className="space-y-6">
              <ImportPanel onAddCurves={addCurves} onReplaceCurves={replaceCurves} />

              <section className="space-y-3">
                <h2 className="text-sm font-semibold uppercase tracking-wide text-zinc-400">Plot controls</h2>

                <SelectControl
                  label="X-axis"
                  value={xMetric}
                  setValue={setXMetric}
                  options={X_METRIC_OPTIONS}
                  help="Use N for architecture comparisons, D for dataset comparisons, and PF-days for compute-frontier comparisons."
                />

                <SelectControl
                  label="Loss source"
                  value={lossSource}
                  setValue={setLossSource}
                  options={LOSS_SOURCE_OPTIONS}
                  help="Measured loss is used when present. Predicted loss is computed from Kaplan L(N,D)."
                />

                <SelectControl
                  label="Baseline curve"
                  value={baselineCurveId}
                  setValue={setBaselineCurveId}
                  options={baselineOptions}
                  help="Used for the percentage-difference column in the summary table."
                />

                <ToggleControl
                  label="Use logarithmic x-axis"
                  checked={useLogX}
                  setChecked={setUseLogX}
                />
              </section>

              <CurveListPanel curves={curves} setCurves={setCurves} />

              <div className="grid grid-cols-2 gap-2">
                <button
                  type="button"
                  onClick={() => setCurves(makeDemoCurves())}
                  className="rounded-xl border border-zinc-700 bg-zinc-900 px-4 py-2 text-sm font-semibold text-zinc-200 transition hover:border-blue-500/60 hover:text-blue-100"
                >
                  Load demo
                </button>
                <button
                  type="button"
                  onClick={() => setCurves([])}
                  className="rounded-xl border border-zinc-700 bg-zinc-900 px-4 py-2 text-sm font-semibold text-zinc-200 transition hover:border-red-500/60 hover:text-red-100"
                >
                  Clear
                </button>
              </div>

              <ExampleFormatPanel />
            </div>
          </aside>

          <div
            role="separator"
            aria-orientation="vertical"
            aria-label="Resize input panel"
            title="Drag to resize input panel. Double-click to reset."
            onPointerDown={(event) => {
              event.preventDefault();
              setIsResizingSidebar(true);
            }}
            onDoubleClick={() => setSidebarWidth(440)}
            className="hidden w-1 shrink-0 cursor-col-resize bg-zinc-900 transition hover:bg-blue-500/40 active:bg-blue-500/60 lg:block"
            style={{ touchAction: "none" }}
          />

          <main className="min-w-0 flex-1 space-y-3 p-4 lg:p-5">
            <div className="flex flex-col gap-2 sm:flex-row sm:items-end sm:justify-between">
              <div>
                <h1 className="text-3xl font-semibold tracking-tight">Scaling curve comparison</h1>
                <p className="mt-2 text-sm text-zinc-500">
                  Comparing <span className="font-medium text-blue-300">{visibleCurves.length}</span> active curve(s) with{" "}
                  <span className="font-medium text-blue-300">{human_num(allPoints.length)}</span> total points.
                </p>
              </div>

              {globalBest && (
                <div className="rounded-xl border border-blue-500/30 bg-blue-500/10 px-4 py-2 text-sm text-blue-100">
                  Best loss: {formatLoss(globalBest.loss)} from {globalBest.curve}
                </div>
              )}
            </div>

            <section className="grid grid-cols-1 gap-2 md:grid-cols-2 xl:grid-cols-4">
              <StatCard label="Curves" value={human_num(visibleCurves.length)} sub={`${human_num(allPoints.length)} active points`} />
              <StatCard label="Best loss" value={globalBest ? formatLoss(globalBest.loss) : "—"} sub={globalBest ? globalBest.curve : "No data"} />
              <StatCard label="Cheapest point" value={cheapest ? cheapest.pf_days.toFixed(4) : "—"} sub={cheapest ? "PF-days" : "No data"} />
              <StatCard
                label={<><span>Closest to </span><MathText>{"\\(D/N\\approx20\\)"}</MathText></>}
                value={bestBalanced ? human_num(bestBalanced.D_over_N) : "—"}
                sub={bestBalanced ? bestBalanced.curve : "No data"}
              />
            </section>

            {visibleCurves.length === 0 ? (
              <div className="rounded-2xl border border-zinc-800 bg-zinc-950/80 p-8 text-center text-sm text-zinc-500">
                Paste df.to_csv(index=False) output or a designer sweep protocol to begin.
              </div>
            ) : (
              <>
                <section className="grid grid-cols-1 gap-3 xl:grid-cols-2">
                  <MultiCurveChart
                    title="Loss comparison"
                    subtitle="Overlay measured/imported curves and optional Kaplan predicted curves."
                    curves={plottedLossCurves}
                    xMetric={xMetric}
                    yMetric="loss"
                    yLabel="Loss"
                    logX={useLogX}
                  />

                  <MultiCurveChart
                    title="Compute comparison"
                    subtitle="Uses imported PF-days when present, otherwise estimates C ≈ 6ND."
                    curves={visibleCurves}
                    xMetric={xMetric}
                    yMetric="pf_days"
                    yLabel="PF-days"
                    logX={useLogX}
                    logY
                  />
                </section>

                <section className="grid grid-cols-1 gap-3 xl:grid-cols-2">
                  <MultiCurveChart
                    title="Data/model balance"
                    subtitle="The dashed reference is the rough D/N ≈ 20 rule of thumb."
                    curves={visibleCurves}
                    xMetric={xMetric}
                    yMetric="D_over_N"
                    yLabel="D/N"
                    referenceY={20}
                    logX={useLogX}
                    logY
                  />

                  <MultiCurveChart
                    title="Pareto frontier: loss vs compute"
                    subtitle="Only points that improve loss at increasing compute are kept for each curve."
                    curves={paretoCurves}
                    xMetric="pf_days"
                    yMetric="loss"
                    yLabel="Loss"
                    logX
                  />
                </section>

                <section className="grid grid-cols-1 gap-3 xl:grid-cols-2">
                  <DataTable title="Curve summary" rows={statsRows} />
                  <DataTable title="Ranked best points" rows={rankedRows} />
                </section>

                <div className="rounded-2xl border border-zinc-800 bg-zinc-950/80 p-3 text-xs leading-relaxed text-zinc-500">
                  <div className="font-semibold uppercase tracking-wide text-zinc-400">Notes</div>
                  <p className="mt-1">
                    This page is intended to compare completed or planned scaling-law curves. For measured runs, paste
                    <code className="text-zinc-300"> df.to_csv(index=False)</code> output with
                    <code className="text-zinc-300"> num_train_tokens</code>, <code className="text-zinc-300">num_params</code>, and
                    <code className="text-zinc-300"> min_saved_val_loss</code>. For planned runs, paste the exported designer sweep protocol.
                  </p>
                  <p className="mt-1">
                    When <code className="text-zinc-300">pf_days</code> is missing, compute is estimated from
                    <MathText>{"\\(C\\approx6ND\\)"}</MathText>. When measured loss is missing but <MathText>{"\\(N\\)"}</MathText> and{" "}
                    <MathText>{"\\(D\\)"}</MathText> are present, the page falls back to <MathText>{"\\(L(N,D)\\)"}</MathText>.
                  </p>
                </div>
              </>
            )}
          </main>
        </div>
      </div>
    </MathJaxContext>
  );
}
