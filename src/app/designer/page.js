"use client";

import { useEffect, useMemo, useRef, useState } from "react";
import dynamic from "next/dynamic";
import { MathJax, MathJaxContext } from "better-react-mathjax";

import {
  gpt_like_nonembedding_params,
  kaplan_loss_ND,
  training_flops,
  flops_to_pf_days,
  powerlines_Bopt_sequences,
  powerlines_Bcrit_sequences,
  recommended_batch_target,
  powerlines_tau_opt,
  tokens_per_parameter,
  human_num,
} from "@/lib/scaling_laws";

const Plot = dynamic(() => import("react-plotly.js"), { ssr: false });

const MATHJAX_CONFIG = {
  tex: {
    inlineMath: [["\\(", "\\)"], ["$", "$"]],
    displayMath: [["\\[", "\\]"]],
  },
};

const SWEEP_SPECS = {
  D_tokens: {
    label: "Dataset size D",
    mathLabel: "D",
    unit: "tokens",
    scale: "log",
    sliderMin: valueToLogSlider(1e5),
    sliderMax: valueToLogSlider(5e12),
    defaultMin: valueToLogSlider(1e8),
    defaultMax: valueToLogSlider(2e11),
    xKey: "D",
  },
  N_params: {
    label: "Non-embedding parameters N",
    mathLabel: "N",
    unit: "parameters",
    scale: "log",
    sliderMin: valueToLogSlider(1e5),
    sliderMax: valueToLogSlider(5e9),
    defaultMin: valueToLogSlider(1e6),
    defaultMax: valueToLogSlider(1e9),
    xKey: "N",
  },
  n_layer: {
    label: "Number of layers L",
    mathLabel: "L",
    unit: "layers",
    scale: "linear",
    sliderMin: 1,
    sliderMax: 192,
    defaultMin: 2,
    defaultMax: 48,
    step: 1,
    xKey: "L",
  },
  n_embd: {
    label: "Embedding dimension d_model",
    mathLabel: "d_model",
    unit: "width",
    scale: "linear",
    sliderMin: 64,
    sliderMax: 4096,
    defaultMin: 128,
    defaultMax: 2048,
    step: 64,
    xKey: "dModel",
  },
  n_head: {
    label: "Attention heads H",
    mathLabel: "H",
    unit: "heads",
    scale: "linear",
    sliderMin: 1,
    sliderMax: 128,
    defaultMin: 2,
    defaultMax: 32,
    step: 1,
    xKey: "H",
  },
  batch_size_sequences: {
    label: "Micro-batch size B_seq",
    mathLabel: "B_seq",
    unit: "sequences",
    scale: "log",
    sliderMin: valueToLogSlider(1),
    sliderMax: valueToLogSlider(8192),
    defaultMin: valueToLogSlider(4),
    defaultMax: valueToLogSlider(512),
    xKey: "BSeq",
  },
};

const SWEEP_MODE_OPTIONS = [
  { value: "1d", label: "1D sweep" },
  { value: "2d", label: "2D sweep" },
];

const KAPLAN_PLOT_SOURCE_OPTIONS = [
  { value: "matrix", label: "Full 2D matrix edge slices" },
  { value: "x_slice", label: "X-axis slice" },
  { value: "y_slice", label: "Y-axis slice" },
];


const MAX_ITERS_MODE_OPTIONS = [
  { value: "computed_from_tokens", label: "Compute from D / global batch" },
  { value: "fixed", label: "Use fixed max_iters" },
];

const LR_DECAY_MODE_OPTIONS = [
  { value: "fraction_of_max_iters", label: "Fraction of max_iters" },
  { value: "fixed", label: "Use fixed lr_decay_iters" },
];

function useDebouncedValue(value, delay = 80) {
  const [debounced, setDebounced] = useState(value);

  useEffect(() => {
    const id = setTimeout(() => setDebounced(value), delay);
    return () => clearTimeout(id);
  }, [value, delay]);

  return debounced;
}

function logSliderToValue(x) {
  return 10 ** x;
}

function valueToLogSlider(x) {
  return Math.log10(x);
}

function clamp(x, lo, hi) {
  return Math.max(lo, Math.min(hi, x));
}

function formatSci(x, digits = 3) {
  if (x === null || x === undefined || Number.isNaN(x)) return "—";
  return Number(x).toExponential(digits);
}

function safeDivide(a, b) {
  return Number.isFinite(a) && Number.isFinite(b) && b !== 0 ? a / b : NaN;
}

function parseSmartNumber(raw) {
  if (typeof raw === "number") return raw;

  let text = String(raw).trim();
  if (!text) return NaN;

  text = text
    .replace(/,/g, "")
    .replace(/_/g, "")
    .toLowerCase();

  text = text.replace(
    /\s*(tokens?|parameters?|params?|sequences?|seqs?|steps?|gpus?|gpu|pf-days?|pflops?|layers?|heads?|width)\s*$/i,
    ""
  );

  text = text.replace(/\s+/g, "");

  const match = text.match(
    /^([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:e[+-]?\d+)?)([a-z]*)$/
  );

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
    quadrillion: 1e15,
  };

  if (!(suffix in multipliers)) return NaN;

  return value * multipliers[suffix];
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

function QuantityName({ name }) {
  const map = {
    sweep_variable: "Sweep variable",
    sweep_min: "Sweep minimum",
    sweep_max: "Sweep maximum",
    sweep_points: "Number of sweep points",
    architecture: "Architecture",
    n_layer: (
      <>
        Layers <MathText>{"\\(L\\)"}</MathText>
      </>
    ),
    n_head: (
      <>
        Heads <MathText>{"\\(H\\)"}</MathText>
      </>
    ),
    n_embd: (
      <>
        Width <MathText>{"\\(d_{\\rm model}\\)"}</MathText>
      </>
    ),
    mlp_ratio: "MLP expansion ratio",
    D_tokens: (
      <>
        Dataset size <MathText>{"\\(D\\)"}</MathText>
      </>
    ),
    N_params: (
      <>
        Non-embedding params <MathText>{"\\(N\\)"}</MathText>
      </>
    ),
    tpp: (
      <>
        Tokens per parameter <MathText>{"\\(D/N\\)"}</MathText>
      </>
    ),
    loss: (
      <>
        Predicted loss <MathText>{"\\(L(N,D)\\)"}</MathText>
      </>
    ),
    compute: "Training compute",
    batch: (
      <>
        Global batch <MathText>{"\\(B\\)"}</MathText>
      </>
    ),
    steps: "Optimizer steps",
    tau: (
      <>
        Optimizer timescale <MathText>{"\\(\\tau_{\\rm opt}\\)"}</MathText>
      </>
    ),
    regime: "Regime",
  };

  return map[name] ?? humanizeIdentifier(name);
}

function HeaderName({ name }) {
  const map = {
    point: "Point",
    variable: "Variable",
    second_variable: "Second variable",
    x: "X value",
    y: "Y value",
    value: "Value",
    quantity: "Quantity",
    constant: "Constant",
    setting: "Setting",
    N: (
      <>
        <MathText>{"\\(N\\)"}</MathText>
      </>
    ),
    D: (
      <>
        <MathText>{"\\(D\\)"}</MathText>
      </>
    ),
    D_over_N: (
      <>
        <MathText>{"\\(D/N\\)"}</MathText>
      </>
    ),
    loss: "Predicted loss",
    pf_days: "PF-days",
    B_target: (
      <>
        <MathText>{"\\(B_{\\rm target}\\)"}</MathText>
      </>
    ),
    steps: "Steps",
    regime: "Regime",
  };

  return map[name] ?? humanizeIdentifier(name);
}

function renderTableValue(key, value) {
  if (value === null || value === undefined || Number.isNaN(value)) return "—";

  if (typeof value === "number") {
    const lowerKey = key.toLowerCase();
    if (lowerKey.includes("loss")) return value.toFixed(5);
    if (lowerKey.includes("pf")) return value.toFixed(4);
    if (lowerKey.includes("tau")) return value.toFixed(4);
    if (lowerKey.includes("lambda") || lowerKey.includes("lr")) return formatSci(value, 3);
    return human_num(value);
  }

  return value;
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

function sanitizePositiveLogRange(minLog, maxLog) {
  const min = Number.isFinite(minLog) ? minLog : 0;
  const max = Number.isFinite(maxLog) ? maxLog : min + 1;

  if (min <= max) return [min, max];
  return [max, min];
}

function normalizeSweepValue(spec, value) {
  if (spec.scale === "log") {
    return Math.max(logSliderToValue(value), 1e-300);
  }

  const step = spec.step ?? 1;
  const rounded = Math.round(value / step) * step;

  if (spec.xKey === "L" || spec.xKey === "H") return Math.max(1, Math.round(rounded));
  if (spec.xKey === "dModel") return Math.max(1, Math.round(rounded));

  return rounded;
}

function makeSweepValues(spec, minValue, maxValue, count) {
  const n = clamp(Math.round(count), 2, 256);
  const [lo, hi] = minValue <= maxValue ? [minValue, maxValue] : [maxValue, minValue];
  const raw = [];

  for (let i = 0; i < n; i++) {
    const f = n === 1 ? 0 : i / (n - 1);
    raw.push(lo + (hi - lo) * f);
  }

  const values = raw.map((v) => normalizeSweepValue(spec, v));
  const unique = [];
  const seen = new Set();

  for (const value of values) {
    const key = String(value);
    if (!seen.has(key)) {
      unique.push(value);
      seen.add(key);
    }
  }

  return unique;
}

function applySweepVariableToConfig(cfg, variable, value) {
  if (variable === "D_tokens") cfg.D_tokens = value;
  if (variable === "N_params") cfg.N_params = value;
  if (variable === "n_layer") cfg.n_layer = Math.max(1, Math.round(value));
  if (variable === "n_embd") cfg.n_embd = Math.max(1, Math.round(value));
  if (variable === "n_head") cfg.n_head = Math.max(1, Math.round(value));
  if (variable === "batch_size_sequences") {
    cfg.batch_size_sequences = Math.max(1, Math.round(value));
  }

  return cfg;
}

function evaluateSweepConfig({
  cfg,
  x,
  y = null,
  activeVariables = [],
  useNOverride,
  baseN,
  batchIsPerGpu,
  gradAccum,
  worldSize,
  seqLen,
}) {
  const architectureDrivenN = gpt_like_nonembedding_params(
    cfg.n_layer,
    cfg.n_embd,
    cfg.mlp_ratio,
    false
  );

  const active = new Set(activeVariables);
  const N = active.has("N_params")
    ? cfg.N_params
    : useNOverride && !active.has("n_layer") && !active.has("n_embd")
      ? baseN
      : architectureDrivenN;

  const D = cfg.D_tokens;
  const globalBSeq = batchIsPerGpu
    ? cfg.batch_size_sequences * gradAccum * worldSize
    : cfg.batch_size_sequences * gradAccum;
  const globalBTokens = globalBSeq * seqLen;
  const D_over_N = tokens_per_parameter(D, N);
  const loss = kaplan_loss_ND(N, D);
  const flops = training_flops(N, D);
  const pf_days = flops_to_pf_days(flops);
  const Bopt = powerlines_Bopt_sequences(D);
  const Bcrit = powerlines_Bcrit_sequences(D);
  const Btarget = recommended_batch_target(Bopt, Bcrit);
  const steps = safeDivide(D, globalBTokens);
  const tau = powerlines_tau_opt(D_over_N);
  const head_dim = safeDivide(cfg.n_embd, cfg.n_head);
  const balance_distance = Number.isFinite(D_over_N) && D_over_N > 0
    ? Math.abs(Math.log(D_over_N / 20))
    : NaN;

  return {
    x,
    y,
    variable: x,
    second_variable: y,
    L: cfg.n_layer,
    H: cfg.n_head,
    dModel: cfg.n_embd,
    head_dim,
    N,
    D,
    D_over_N,
    balance_distance,
    loss,
    flops,
    pf_days,
    Bopt,
    Btarget,
    Bcrit,
    Bactual: globalBSeq,
    steps,
    tau,
    regime: classifyRegime(D, N),
  };
}


function sanitizeFilenamePart(value) {
  return String(value ?? "")
    .trim()
    .replace(/[^a-zA-Z0-9._-]+/g, "_")
    .replace(/^_+|_+$/g, "") || "config";
}

function roundForJson(value, digits = 12) {
  if (!Number.isFinite(value)) return value;
  const scale = 10 ** digits;
  return Math.round(value * scale) / scale;
}


function isPlainObject(value) {
  return value !== null && typeof value === "object" && !Array.isArray(value);
}

function firstDefined(...values) {
  return values.find((value) => value !== undefined && value !== null);
}

function numberOrFallback(value, fallback) {
  const parsed = typeof value === "string" ? parseSmartNumber(value) : Number(value);
  return Number.isFinite(parsed) ? parsed : fallback;
}

function integerOrFallback(value, fallback, { min = -Infinity, max = Infinity } = {}) {
  const parsed = Math.round(numberOrFallback(value, fallback));
  if (!Number.isFinite(parsed)) return fallback;
  return clamp(parsed, min, max);
}

function booleanOrFallback(value, fallback) {
  if (typeof value === "boolean") return value;
  if (typeof value === "string") {
    const normalized = value.trim().toLowerCase();
    if (["true", "1", "yes", "y", "on"].includes(normalized)) return true;
    if (["false", "0", "no", "n", "off"].includes(normalized)) return false;
  }
  return fallback;
}

function optionOrFallback(value, options, fallback) {
  const allowed = new Set(options.map((option) => option.value));
  return allowed.has(value) ? value : fallback;
}

function sweepVariableOrFallback(value, fallback) {
  return Object.prototype.hasOwnProperty.call(SWEEP_SPECS, value) ? value : fallback;
}

function positiveActualToLogSlider(value, fallback) {
  const parsed = numberOrFallback(value, NaN);
  if (!Number.isFinite(parsed) || parsed <= 0) return fallback;
  return valueToLogSlider(parsed);
}

function actualValueToSweepSlider(spec, value, fallback) {
  const parsed = numberOrFallback(value, NaN);
  if (!Number.isFinite(parsed)) return fallback;
  if (spec.scale === "log") return positiveActualToLogSlider(parsed, fallback);
  return normalizeSweepValue(spec, parsed);
}

function readProtocolRangeForVariable({ protocol, variable, fallbackRange }) {
  const spec = SWEEP_SPECS[variable];
  if (!spec) return fallbackRange;

  const sweep = protocol?.sweep ?? {};
  const inputSweep = protocol?.input_panel?.sweep ?? {};
  const availableRanges = inputSweep.all_ranges ?? sweep.available_ranges ?? {};
  const xAxis = inputSweep.x_axis ?? sweep.axes?.x ?? null;
  const yAxis = inputSweep.y_axis ?? sweep.axes?.y ?? null;

  const candidates = [];

  if (availableRanges?.[variable]) candidates.push(availableRanges[variable]);
  if (xAxis?.variable === variable) candidates.push(xAxis);
  if (yAxis?.variable === variable) candidates.push(yAxis);

  if (sweep.x_variable === variable) {
    candidates.push({
      variable,
      min: sweep.x_min,
      max: sweep.x_max,
      slider_min: sweep.x_slider_min,
      slider_max: sweep.x_slider_max,
    });
  }

  if (sweep.y_variable === variable) {
    candidates.push({
      variable,
      min: sweep.y_min,
      max: sweep.y_max,
      slider_min: sweep.y_slider_min,
      slider_max: sweep.y_slider_max,
    });
  }

  for (const entry of candidates) {
    if (!entry || !isPlainObject(entry)) continue;

    const sliderMin = numberOrFallback(entry.slider_min, NaN);
    const sliderMax = numberOrFallback(entry.slider_max, NaN);
    if (Number.isFinite(sliderMin) && Number.isFinite(sliderMax)) {
      return { min: sliderMin, max: sliderMax };
    }

    const actualMin = firstDefined(entry.min, entry.actual_min, entry.value_min);
    const actualMax = firstDefined(entry.max, entry.actual_max, entry.value_max);
    const min = actualValueToSweepSlider(spec, actualMin, NaN);
    const max = actualValueToSweepSlider(spec, actualMax, NaN);

    if (Number.isFinite(min) && Number.isFinite(max)) {
      return { min, max };
    }
  }

  return fallbackRange;
}

function chooseValidKvHeads(nHead, preferredKvHeads) {
  const heads = Math.max(1, Math.round(nHead));
  const preferred = Math.max(1, Math.round(preferredKvHeads));

  for (let kv = Math.min(heads, preferred); kv >= 1; kv--) {
    if (heads % kv === 0) return kv;
  }

  return 1;
}

function getMicroBatchSizeFromRow({ row, batchIsPerGpu, gradAccum, worldSize }) {
  const denom = batchIsPerGpu
    ? Math.max(1, gradAccum * worldSize)
    : Math.max(1, gradAccum);

  return Math.max(1, Math.round(row.Bactual / denom));
}

function buildParticleGPTConfigFromRow({
  row,
  index,
  xVariable,
  yVariable,
  is2D,
  options,
}) {
  const microBatchSize = getMicroBatchSizeFromRow({
    row,
    batchIsPerGpu: options.batchIsPerGpu,
    gradAccum: options.gradAccum,
    worldSize: options.worldSize,
  });

  const computedMaxIters = Math.max(1, Math.ceil(row.steps));
  const fixedMaxIters = Math.max(1, Math.round(options.fixedMaxIters));
  const maxIters = options.maxItersMode === "fixed" ? fixedMaxIters : computedMaxIters;

  const fixedLrDecayIters = Math.max(1, Math.round(options.fixedLrDecayIters));
  const decayFraction = Math.max(0, Number(options.lrDecayFraction));
  const lrDecayIters = options.lrDecayMode === "fixed"
    ? fixedLrDecayIters
    : Math.max(1, Math.round(maxIters * decayFraction));

  const warmupIters = Math.max(0, Math.min(Math.round(options.warmupIters), Math.max(maxIters - 1, 0)));
  const nHead = Math.max(1, Math.round(row.H));
  const nKvHeads = chooseValidKvHeads(nHead, options.nKvHeads);

  const config = {
    preparation_name: options.preparationName,
    dataset: options.datasetName,
    data_mode: options.dataMode,
    training_config: {
      eval_interval: Math.max(1, Math.round(options.evalInterval)),
      eval_iters: Math.max(1, Math.round(options.evalIters)),
      log_interval: Math.max(1, Math.round(options.logInterval)),
      gradient_accumulation_steps: Math.max(1, Math.round(options.gradAccum)),
      batch_size: microBatchSize,
      context_events: Math.max(1, Math.round(options.contextEvents)),
      n_layer: Math.max(1, Math.round(row.L)),
      n_head: nHead,
      n_kv_heads: nKvHeads,
      n_embd: Math.max(1, Math.round(row.dModel)),
      dropout: roundForJson(Number(options.dropout), 8),
      learning_rate: roundForJson(Number(options.learningRate), 12),
      max_iters: maxIters,
      lr_decay_iters: lrDecayIters,
      min_lr: roundForJson(Number(options.minLr), 12),
      beta2: roundForJson(Number(options.beta2), 8),
      warmup_iters: warmupIters,
      max_num_failed_checkpoint_checks: Math.round(options.maxFailedCheckpointChecks),
      eval_every_epoch: Boolean(options.evalEveryEpoch),
    },
    sampling_config: {
      temperature: roundForJson(Number(options.temperature), 8),
      top_k: Math.max(1, Math.round(options.topK)),
      seed: Math.round(options.seed),
      device: options.device,
      compile: Boolean(options.compileSampling),
    },
  };

  if (options.includeMetadata) {
    const swept = is2D
      ? {
          x_variable: xVariable,
          x_value: row.x,
          y_variable: yVariable,
          y_value: row.y,
        }
      : {
          variable: xVariable,
          value: row.x,
        };

    const warnings = [];
    if (xVariable === "N_params" || yVariable === "N_params") {
      warnings.push("N_params is a planning-only sweep variable. particleGPT training_config uses architecture fields, so this file stores N_params only in designer_metadata.");
    }
    if (xVariable === "D_tokens" || yVariable === "D_tokens") {
      warnings.push("D_tokens is stored in designer_metadata. Set preparation_name/dataset to the matching prepared dataset for this run.");
    }
    if (nKvHeads !== Math.round(options.nKvHeads)) {
      warnings.push(`n_kv_heads was adjusted to ${nKvHeads} so it divides n_head=${nHead}.`);
    }

    config.designer_metadata = {
      generated_by: "/designer",
      run_index: index + 1,
      sweep_mode: is2D ? "2d" : "1d",
      swept,
      D_tokens: roundForJson(row.D, 6),
      estimated_non_embedding_params: roundForJson(row.N, 6),
      D_over_N: roundForJson(row.D_over_N, 8),
      predicted_kaplan_loss: roundForJson(row.loss, 10),
      training_compute_pf_days: roundForJson(row.pf_days, 10),
      global_batch_sequences: Math.round(row.Bactual),
      optimizer_steps_from_D: roundForJson(row.steps, 6),
      regime: row.regime,
      warnings,
    };
  }

  return config;
}

function makeParticleGPTConfigFiles({ rows, xVariable, yVariable, is2D, options }) {
  const prefix = sanitizeFilenamePart(options.filePrefix);
  const width = Math.max(4, String(rows.length).length);

  return rows.map((row, index) => {
    const config = buildParticleGPTConfigFromRow({
      row,
      index,
      xVariable,
      yVariable,
      is2D,
      options,
    });

    const name = `${prefix}_${String(index + 1).padStart(width, "0")}.json`;
    return {
      name,
      content: `${JSON.stringify(config, null, 4)}\n`,
      config,
      row,
    };
  });
}

function makeCrc32Table() {
  const table = new Uint32Array(256);

  for (let i = 0; i < 256; i++) {
    let c = i;
    for (let k = 0; k < 8; k++) {
      c = c & 1 ? 0xedb88320 ^ (c >>> 1) : c >>> 1;
    }
    table[i] = c >>> 0;
  }

  return table;
}

const CRC32_TABLE = makeCrc32Table();

function crc32(bytes) {
  let c = 0xffffffff;

  for (let i = 0; i < bytes.length; i++) {
    c = CRC32_TABLE[(c ^ bytes[i]) & 0xff] ^ (c >>> 8);
  }

  return (c ^ 0xffffffff) >>> 0;
}

function concatUint8Arrays(parts) {
  const totalLength = parts.reduce((sum, part) => sum + part.length, 0);
  const out = new Uint8Array(totalLength);
  let offset = 0;

  for (const part of parts) {
    out.set(part, offset);
    offset += part.length;
  }

  return out;
}

function createZipBlob(files) {
  const encoder = new TextEncoder();
  const localParts = [];
  const centralParts = [];
  let localOffset = 0;

  const now = new Date();
  const dosTime = ((now.getHours() & 0x1f) << 11) | ((now.getMinutes() & 0x3f) << 5) | (Math.floor(now.getSeconds() / 2) & 0x1f);
  const dosDate = (((now.getFullYear() - 1980) & 0x7f) << 9) | (((now.getMonth() + 1) & 0x0f) << 5) | (now.getDate() & 0x1f);

  for (const file of files) {
    const nameBytes = encoder.encode(file.name);
    const dataBytes = encoder.encode(file.content);
    const crc = crc32(dataBytes);
    const localHeader = new ArrayBuffer(30 + nameBytes.length);
    const localView = new DataView(localHeader);

    localView.setUint32(0, 0x04034b50, true);
    localView.setUint16(4, 20, true);
    localView.setUint16(6, 0, true);
    localView.setUint16(8, 0, true);
    localView.setUint16(10, dosTime, true);
    localView.setUint16(12, dosDate, true);
    localView.setUint32(14, crc, true);
    localView.setUint32(18, dataBytes.length, true);
    localView.setUint32(22, dataBytes.length, true);
    localView.setUint16(26, nameBytes.length, true);
    localView.setUint16(28, 0, true);
    new Uint8Array(localHeader, 30).set(nameBytes);

    localParts.push(new Uint8Array(localHeader), dataBytes);

    const centralHeader = new ArrayBuffer(46 + nameBytes.length);
    const centralView = new DataView(centralHeader);
    centralView.setUint32(0, 0x02014b50, true);
    centralView.setUint16(4, 20, true);
    centralView.setUint16(6, 20, true);
    centralView.setUint16(8, 0, true);
    centralView.setUint16(10, 0, true);
    centralView.setUint16(12, dosTime, true);
    centralView.setUint16(14, dosDate, true);
    centralView.setUint32(16, crc, true);
    centralView.setUint32(20, dataBytes.length, true);
    centralView.setUint32(24, dataBytes.length, true);
    centralView.setUint16(28, nameBytes.length, true);
    centralView.setUint16(30, 0, true);
    centralView.setUint16(32, 0, true);
    centralView.setUint16(34, 0, true);
    centralView.setUint16(36, 0, true);
    centralView.setUint32(38, 0, true);
    centralView.setUint32(42, localOffset, true);
    new Uint8Array(centralHeader, 46).set(nameBytes);
    centralParts.push(new Uint8Array(centralHeader));

    localOffset += localHeader.byteLength + dataBytes.length;
  }

  const centralOffset = localOffset;
  const centralDirectory = concatUint8Arrays(centralParts);
  const end = new ArrayBuffer(22);
  const endView = new DataView(end);
  endView.setUint32(0, 0x06054b50, true);
  endView.setUint16(4, 0, true);
  endView.setUint16(6, 0, true);
  endView.setUint16(8, files.length, true);
  endView.setUint16(10, files.length, true);
  endView.setUint32(12, centralDirectory.length, true);
  endView.setUint32(16, centralOffset, true);
  endView.setUint16(20, 0, true);

  return new Blob([...localParts, centralDirectory, new Uint8Array(end)], { type: "application/zip" });
}

function downloadBlob(blob, filename) {
  const url = URL.createObjectURL(blob);
  const anchor = document.createElement("a");
  anchor.href = url;
  anchor.download = filename;
  document.body.appendChild(anchor);
  anchor.click();
  anchor.remove();
  window.setTimeout(() => URL.revokeObjectURL(url), 1000);
}

function downloadParticleGPTConfigZip(files, filename) {
  if (!Array.isArray(files) || files.length === 0) return;
  const blob = createZipBlob(files);
  downloadBlob(blob, filename);
}

function downloadJsonValue(value, filename) {
  const blob = new Blob([`${JSON.stringify(value, null, 2)}\n`], {
    type: "application/json",
  });
  downloadBlob(blob, filename);
}

function getActualRangeForSweep(spec, range) {
  const fallback = {
    min: spec.defaultMin,
    max: spec.defaultMax,
  };

  const safeRange = range ?? fallback;
  const [rangeMin, rangeMax] = spec.scale === "log"
    ? sanitizePositiveLogRange(safeRange.min, safeRange.max)
    : safeRange.min <= safeRange.max
      ? [safeRange.min, safeRange.max]
      : [safeRange.max, safeRange.min];

  return [rangeMin, rangeMax];
}

const SliderControl = ({
  label,
  value,
  setValue,
  min,
  max,
  step = 1,
  format = (x) => x,
  help,
  disabled = false,
  inputValueToSliderValue = (x) => x,
  sanitizeValue = (x) => x,
}) => {
  const formattedValue = String(format(value));
  const sliderValue = clamp(value, min, max);

  const [draft, setDraft] = useState(formattedValue);
  const [isEditing, setIsEditing] = useState(false);

  useEffect(() => {
    if (!isEditing) setDraft(formattedValue);
  }, [formattedValue, isEditing]);

  const normalizeSliderValue = (nextValue) => {
    let next = Number(nextValue);

    if (!Number.isFinite(next)) return value;

    next = clamp(next, min, max);
    next = sanitizeValue(next);

    if (!Number.isFinite(next)) return value;

    return clamp(next, min, max);
  };

  const normalizeTextValue = (nextValue) => {
    let next = Number(nextValue);

    if (!Number.isFinite(next)) return value;

    next = sanitizeValue(next);

    if (!Number.isFinite(next)) return value;

    return next;
  };

  const applySliderValue = (nextValue) => {
    const next = normalizeSliderValue(nextValue);
    setValue(next);
    return next;
  };

  const applyTextValue = (nextValue) => {
    const next = normalizeTextValue(nextValue);
    setValue(next);
    return next;
  };

  const commitDraft = () => {
    const parsedInputValue = parseSmartNumber(draft);

    if (!Number.isFinite(parsedInputValue)) {
      setDraft(formattedValue);
      setIsEditing(false);
      return;
    }

    const nextValue = inputValueToSliderValue(parsedInputValue);

    if (!Number.isFinite(nextValue)) {
      setDraft(formattedValue);
      setIsEditing(false);
      return;
    }

    const next = applyTextValue(nextValue);
    setDraft(String(format(next)));
    setIsEditing(false);
  };

  const resetDraft = () => {
    setDraft(formattedValue);
    setIsEditing(false);
  };

  return (
    <div
      className={`flex items-center gap-x-3 gap-y-1 rounded-xl border border-zinc-800 bg-zinc-950/70 p-2 ${
        disabled ? "opacity-50" : ""
      }`}
    >
      <div className="w-25 pr-1 text-sm font-medium leading-tight text-zinc-100">
        {label}
      </div>

      <div className="flex-grow">
        <input
          type="range"
          min={min}
          max={max}
          step={step}
          value={sliderValue}
          disabled={disabled}
          onChange={(e) => {
            const next = applySliderValue(Number(e.target.value));
            setDraft(String(format(next)));
          }}
          className="block w-full min-w-0 accent-blue-500 disabled:cursor-not-allowed"
        />

        <div className="flex justify-between gap-2 text-[10px] text-zinc-600">
          <span className="min-w-0 truncate">{format(min)}</span>
          <span className="min-w-0 truncate text-right">{format(max)}</span>
        </div>
      </div>

      <div>
        <input
          type="text"
          value={draft}
          disabled={disabled}
          onFocus={(e) => {
            setIsEditing(true);
            e.currentTarget.select();
          }}
          onChange={(e) => setDraft(e.target.value)}
          onBlur={commitDraft}
          onKeyDown={(e) => {
            if (e.key === "Enter") {
              e.preventDefault();
              commitDraft();
            }

            if (e.key === "Escape") {
              e.preventDefault();
              resetDraft();
            }
          }}
          className="h-8 min-w-15 max-w-25 rounded-lg border border-zinc-800 bg-zinc-900 px-1.5 text-right text-xs text-blue-300 outline-none transition focus:border-blue-500 focus:ring-1 focus:ring-blue-500 disabled:cursor-not-allowed"
          spellCheck={false}
        />

        {help && (
          <div className="col-start-2 col-span-2 min-w-0 text-xs text-zinc-500">
            {help}
          </div>
        )}
      </div>
    </div>
  );
};

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
        onChange={(e) => setValue(e.target.value)}
        className="w-full rounded-lg border border-zinc-800 bg-zinc-900 px-3 py-2 text-sm text-blue-200 outline-none transition focus:border-blue-500 focus:ring-1 focus:ring-blue-500"
        spellCheck={false}
      />
      {help && <div className="mt-2 text-xs text-zinc-500">{help}</div>}
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

function FrozenControlNotice({ label, help }) {
  return (
    <div className="rounded-xl border border-blue-500/30 bg-blue-500/10 p-3">
      <div className="text-sm font-medium text-blue-100">{label}</div>
      <div className="mt-1 text-xs text-blue-200/70">{help}</div>
    </div>
  );
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
  return (
    <div className="rounded-2xl border border-zinc-800 bg-zinc-950/80 p-3">
      <h2 className="mb-2 text-base font-semibold text-zinc-100">{title}</h2>

      <div className="overflow-x-auto">
        <table className="w-full text-left text-xs">
          <thead className="border-b border-zinc-800 uppercase tracking-wide text-zinc-500">
            <tr>
              {Object.keys(rows[0] ?? {}).map((key) => (
                <th key={key} className="whitespace-nowrap px-2 py-1.5">
                  <HeaderName name={key} />
                </th>
              ))}
            </tr>
          </thead>

          <tbody>
            {rows.map((row, i) => (
              <tr key={i} className="border-b border-zinc-900 last:border-0">
                {Object.entries(row).map(([key, value]) => (
                  <td key={key} className="whitespace-nowrap px-2 py-1.5 text-zinc-300">
                    {renderTableValue(key, value)}
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

function ChartShell({ title, subtitle, children, footer, heightClass = "h-64" }) {
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

function BaseTooltip() {
  return null;
}


function logSpacedPoints(minValue, maxValue, count) {
  const n = clamp(Math.round(count), 2, 256);
  const lo = Math.max(minValue, 1e-300);
  const hi = Math.max(maxValue, lo * 1.0001);
  const out = [];

  for (let i = 0; i < n; i++) {
    const f = i / (n - 1);
    out.push(lo * (hi / lo) ** f);
  }

  return out;
}

const KAPLAN_COLORS = [
  "#fde725",
  "#bddf26",
  "#7ad151",
  "#22a884",
  "#2a788e",
  "#414487",
  "#440154",
  "#8b5cf6",
];

function kaplanSeriesKey(prefix, value) {
  return `${prefix}_${String(value).replace(/[^a-zA-Z0-9]/g, "_")}`;
}

function formatKaplanLegendValue(value, unit = "") {
  if (!Number.isFinite(value)) return "—";
  if (value >= 1e12) return `${Number(value / 1e12).toPrecision(2)}T${unit}`;
  if (value >= 1e9) return `${Number(value / 1e9).toPrecision(2)}B${unit}`;
  if (value >= 1e6) return `${Number(value / 1e6).toPrecision(3)}M${unit}`;
  if (value >= 1e3) return `${Number(value / 1e3).toPrecision(3)}K${unit}`;
  return `${Number(value).toPrecision(3)}${unit}`;
}

function actualSweepRangeForSpec(spec, range) {
  const fallback = {
    min: spec.defaultMin,
    max: spec.defaultMax,
  };

  const candidate = range ?? fallback;
  const minRaw = Number.isFinite(candidate.min) ? candidate.min : fallback.min;
  const maxRaw = Number.isFinite(candidate.max) ? candidate.max : fallback.max;

  if (spec.scale === "log") {
    const [lo, hi] = sanitizePositiveLogRange(minRaw, maxRaw);
    return [logSliderToValue(lo), logSliderToValue(hi)];
  }

  const lo = Math.min(minRaw, maxRaw);
  const hi = Math.max(minRaw, maxRaw);
  return [normalizeSweepValue(spec, lo), normalizeSweepValue(spec, hi)];
}

function finitePositiveDomainFromData(data, key, pad = 1.15) {
  const vals = data
    .map((row) => row?.[key])
    .filter((v) => Number.isFinite(v) && v > 0);

  if (vals.length === 0) return [1, 10];

  const lo = Math.min(...vals);
  const hi = Math.max(...vals);

  if (Math.abs(Math.log10(hi) - Math.log10(lo)) < 1e-9) {
    return [lo / pad, hi * pad];
  }

  return [lo / pad, hi * pad];
}

function finiteYDomainFromKeys(data, keys, padFraction = 0.08) {
  const vals = [];

  for (const row of data) {
    for (const key of keys) {
      const v = row?.[key];
      if (Number.isFinite(v)) vals.push(v);
    }
  }

  if (vals.length === 0) return ["dataMin", "dataMax"];

  const lo = Math.min(...vals);
  const hi = Math.max(...vals);

  if (Math.abs(hi - lo) < 1e-12) {
    const pad = Math.max(Math.abs(lo) * 0.05, 0.05);
    return [lo - pad, hi + pad];
  }

  const pad = (hi - lo) * padFraction;
  return [lo - pad, hi + pad];
}

function uniqueSortedPositive(values) {
  const unique = [];
  const seen = new Set();

  for (const value of values) {
    if (!Number.isFinite(value) || value <= 0) continue;
    const key = value.toPrecision(10);
    if (seen.has(key)) continue;
    seen.add(key);
    unique.push(value);
  }

  return unique.sort((a, b) => a - b);
}

function makeLogAxisTicks(domain, maxTicks = 6) {
  if (!Array.isArray(domain) || domain.length !== 2) return undefined;

  const rawLo = Number(domain[0]);
  const rawHi = Number(domain[1]);

  if (!Number.isFinite(rawLo) || !Number.isFinite(rawHi)) return undefined;

  const lo = Math.max(Math.min(rawLo, rawHi), 1e-300);
  const hi = Math.max(Math.max(rawLo, rawHi), lo * 1.0001);

  const expLo = Math.floor(Math.log10(lo));
  const expHi = Math.ceil(Math.log10(hi));
  const candidates = [];

  for (let exp = expLo; exp <= expHi; exp++) {
    const decade = 10 ** exp;
    for (const mantissa of [1, 2, 5]) {
      const tick = mantissa * decade;
      if (tick >= lo / 1.0000001 && tick <= hi * 1.0000001) {
        candidates.push(tick);
      }
    }
  }

  let ticks = uniqueSortedPositive(candidates);

  if (ticks.length < 2) {
    ticks = uniqueSortedPositive(logSpacedPoints(lo, hi, Math.min(maxTicks, 5)));
  }

  return downsampleEvenly(ticks, maxTicks);
}

function logSpacedComparisonValues(minValue, maxValue, count, currentValue = null) {
  const raw = logSpacedPoints(minValue, maxValue, Math.max(2, count));

  if (
    Number.isFinite(currentValue) &&
    currentValue > 0 &&
    currentValue >= minValue / 1.001 &&
    currentValue <= maxValue * 1.001
  ) {
    raw.push(currentValue);
  }

  return uniqueSortedPositive(raw);
}

function makeKaplanDesignerPanels({ sweepRanges, sweepPoints, baseN, baseD }) {
  const [dMin, dMax] = actualSweepRangeForSpec(
    SWEEP_SPECS.D_tokens,
    sweepRanges?.D_tokens
  );
  const [nMin, nMax] = actualSweepRangeForSpec(
    SWEEP_SPECS.N_params,
    sweepRanges?.N_params
  );

  const pointCount = clamp(Math.round(sweepPoints), 4, 48);
  const panelPointCount = Math.min(pointCount, 28);
  const multiLinePointCount = Math.min(pointCount, 24);

  const Dpoints = logSpacedPoints(dMin, dMax, panelPointCount);
  const Npoints = logSpacedPoints(nMin, nMax, panelPointCount);

  // Figure-1-style compute panel: pair log-spaced N and D values from
  // the user's configured N-range and D-range to form a diagonal run family.
  const computeData = Npoints
    .map((N, index) => {
      const f = panelPointCount <= 1 ? 0 : index / (panelPointCount - 1);
      const D = dMin * (dMax / dMin) ** f;
      const Cmin = flops_to_pf_days(training_flops(N, D));

      return {
        Cmin,
        N,
        D,
        loss: kaplan_loss_ND(N, D),
      };
    })
    .filter((row) => Number.isFinite(row.Cmin) && row.Cmin > 0 && Number.isFinite(row.loss));

  computeData.sort((a, b) => a.Cmin - b.Cmin);

  const datasetData = Dpoints
    .map((D) => ({
      D,
      N: baseN,
      loss: kaplan_loss_ND(baseN, D),
    }))
    .filter((row) => Number.isFinite(row.D) && row.D > 0 && Number.isFinite(row.loss));

  const paramsData = Npoints
    .map((N) => ({
      N,
      D: baseD,
      loss: kaplan_loss_ND(N, baseD),
    }))
    .filter((row) => Number.isFinite(row.N) && row.N > 0 && Number.isFinite(row.loss));

  const modelSizes = logSpacedComparisonValues(nMin, nMax, 6, baseN);
  const datasetSizes = logSpacedComparisonValues(dMin, dMax, 7, baseD);
  const figure4Dpoints = logSpacedPoints(dMin, dMax, multiLinePointCount);
  const figure9Npoints = logSpacedPoints(nMin, nMax, multiLinePointCount);

  const figure4Series = modelSizes.map((N, index) => {
    const isCurrent = Math.abs(Math.log(N / baseN)) < 1e-9;

    return {
      key: kaplanSeriesKey("N", N),
      name: `${formatKaplanLegendValue(N)}${isCurrent ? " current N" : ""}`,
      color: isCurrent ? "#60a5fa" : KAPLAN_COLORS[index % KAPLAN_COLORS.length],
      strokeWidth: isCurrent ? 2.6 : 1.8,
    };
  });

  const figure4Data = figure4Dpoints.map((D) => {
    const row = { D };

    for (const N of modelSizes) {
      row[kaplanSeriesKey("N", N)] = kaplan_loss_ND(N, D);
    }

    return row;
  });

  const figure9Series = datasetSizes.map((D, index) => {
    const isCurrent = Math.abs(Math.log(D / baseD)) < 1e-9;

    return {
      key: kaplanSeriesKey("D", D),
      name: `${formatKaplanLegendValue(D)}${isCurrent ? " current D" : ""}`,
      color: isCurrent
        ? "#60a5fa"
        : KAPLAN_COLORS[(KAPLAN_COLORS.length - 1 - index + KAPLAN_COLORS.length) % KAPLAN_COLORS.length],
      strokeWidth: isCurrent ? 2.6 : 1.8,
    };
  });

  const figure9Data = figure9Npoints.map((N) => {
    const row = { N };

    for (const D of datasetSizes) {
      row[kaplanSeriesKey("D", D)] = kaplan_loss_ND(N, D);
    }

    return row;
  });

  return {
    baseN,
    baseD,
    ranges: {
      D: [dMin, dMax],
      N: [nMin, nMax],
      compute: finitePositiveDomainFromData(computeData, "Cmin"),
    },
    figure1: {
      computeData,
      datasetData,
      paramsData,
      currentCompute: flops_to_pf_days(training_flops(baseN, baseD)),
    },
    figure4: {
      data: figure4Data,
      series: figure4Series,
      xDomain: [dMin / 1.08, dMax * 1.08],
      yDomain: finiteYDomainFromKeys(
        figure4Data,
        figure4Series.map((s) => s.key)
      ),
    },
    figure9: {
      data: figure9Data,
      series: figure9Series,
      xDomain: [nMin / 1.08, nMax * 1.08],
      yDomain: finiteYDomainFromKeys(
        figure9Data,
        figure9Series.map((s) => s.key)
      ),
    },
  };
}


function downsampleEvenly(values, maxCount) {
  if (!Array.isArray(values)) return [];
  if (values.length <= maxCount) return values;

  const out = [];
  const seen = new Set();

  for (let i = 0; i < maxCount; i++) {
    const index = Math.round((i * (values.length - 1)) / Math.max(maxCount - 1, 1));
    if (seen.has(index)) continue;
    seen.add(index);
    out.push(values[index]);
  }

  return out;
}

function finitePositiveDomainFromRows(rows, key, pad = 1.08) {
  return finitePositiveDomainFromData(rows ?? [], key, pad);
}

function stableValueKey(value) {
  if (!Number.isFinite(value)) return "nan";
  return Number(value).toPrecision(12);
}

function countUniquePositiveX(rows, xField) {
  const seen = new Set();

  for (const row of rows ?? []) {
    const x = row?.[xField];
    if (Number.isFinite(x) && x > 0) seen.add(stableValueKey(x));
  }

  return seen.size;
}

function makeSingleKaplanSeries(rows, xField, name = "generated run slice") {
  const pivot = new Map();

  for (const row of rows ?? []) {
    const x = row?.[xField];
    const loss = row?.loss;
    if (!Number.isFinite(x) || x <= 0 || !Number.isFinite(loss)) continue;

    const xKey = stableValueKey(x);
    const existing = pivot.get(xKey);

    if (!existing || loss < existing.single_curve) {
      pivot.set(xKey, {
        [xField]: x,
        single_curve: loss,
        N: row.N,
        D: row.D,
      });
    }
  }

  const data = Array.from(pivot.values()).sort((a, b) => a[xField] - b[xField]);

  return {
    data: downsampleEvenly(data, 128),
    series: [
      {
        key: "single_curve",
        name,
        color: "#60a5fa",
        strokeWidth: 2.4,
      },
    ],
  };
}

function pickKaplanGroups(groups, currentValue, maxSeries = 8, alwaysIncludeValue = null) {
  const viable = groups
    .filter((group) => countUniquePositiveX(group.rows, group.xField) >= 2)
    .sort((a, b) => a.value - b.value);

  const candidates = viable.length > 0 ? viable : groups.sort((a, b) => a.value - b.value);
  if (candidates.length <= maxSeries) return candidates;

  const picked = downsampleEvenly(candidates, maxSeries);

  const forceNearest = (targetValue) => {
    if (!Number.isFinite(targetValue) || targetValue <= 0) return;

    const nearest = candidates.reduce((best, group) => {
      if (!best) return group;
      return Math.abs(Math.log(group.value / targetValue)) < Math.abs(Math.log(best.value / targetValue))
        ? group
        : best;
    }, null);

    if (nearest && !picked.some((group) => group.key === nearest.key)) {
      picked[picked.length - 1] = nearest;
    }
  };

  forceNearest(currentValue);
  forceNearest(alwaysIncludeValue);

  return picked.sort((a, b) => a.value - b.value);
}

function groupRowsByPositiveField(rows, field, xField = null) {
  const groupMap = new Map();

  for (const row of rows ?? []) {
    const value = row?.[field];
    if (!Number.isFinite(value) || value <= 0) continue;

    const key = stableValueKey(value);

    if (!groupMap.has(key)) {
      groupMap.set(key, {
        key,
        value,
        xField,
        rows: [],
      });
    }

    groupMap.get(key).rows.push(row);
  }

  return Array.from(groupMap.values()).sort((a, b) => a.value - b.value);
}

function chooseExtremeKaplanSlice({ rows, xField, fixedField, prefer = "max" }) {
  const groups = groupRowsByPositiveField(rows, fixedField, xField);
  const viable = groups.filter((group) => countUniquePositiveX(group.rows, xField) >= 2);
  const candidates = viable.length > 0 ? viable : groups;

  if (candidates.length === 0) {
    return {
      rows: [],
      fixedValue: NaN,
      fixedField,
      xField,
      found: false,
    };
  }

  const chosen = prefer === "min" ? candidates[0] : candidates[candidates.length - 1];

  return {
    rows: chosen.rows,
    fixedValue: chosen.value,
    fixedField,
    xField,
    found: true,
  };
}

function makeCurveRowsFromSlice(slice, xField, maxPoints = 128) {
  const pivot = new Map();

  for (const row of slice?.rows ?? []) {
    const x = row?.[xField];
    const loss = row?.loss;
    if (!Number.isFinite(x) || x <= 0 || !Number.isFinite(loss)) continue;

    const xKey = stableValueKey(x);
    const existing = pivot.get(xKey);

    // If duplicate runs land on the same x coordinate, keep the better run.
    if (!existing || loss < existing.loss) {
      pivot.set(xKey, {
        ...row,
        [xField]: x,
        loss,
      });
    }
  }

  return downsampleEvenly(Array.from(pivot.values()).sort((a, b) => a[xField] - b[xField]), maxPoints);
}

function makeLogBinnedBestRows(rows, xField, yField = "loss", maxBins = 36) {
  const cleanRows = (rows ?? [])
    .filter((row) => Number.isFinite(row?.[xField]) && row[xField] > 0 && Number.isFinite(row?.[yField]))
    .sort((a, b) => a[xField] - b[xField]);

  if (cleanRows.length <= maxBins) return cleanRows;

  const lo = cleanRows[0][xField];
  const hi = cleanRows[cleanRows.length - 1][xField];
  const logLo = Math.log(lo);
  const logHi = Math.log(hi);
  const bins = new Map();

  for (const row of cleanRows) {
    const f = Math.abs(logHi - logLo) < 1e-12
      ? 0
      : (Math.log(row[xField]) - logLo) / (logHi - logLo);
    const index = clamp(Math.floor(f * maxBins), 0, maxBins - 1);
    const current = bins.get(index);

    if (!current || row[yField] < current[yField]) {
      bins.set(index, row);
    }
  }

  return Array.from(bins.values()).sort((a, b) => a[xField] - b[xField]);
}

function makeComputeFrontierRows(rows, maxPoints = 36) {
  return makeLogBinnedBestRows(rows, "pf_days", "loss", maxPoints).map((row) => ({
    ...row,
    Cmin: row.pf_days,
  }));
}

function makeGroupedKaplanSeries({
  rows,
  xField,
  groupField,
  prefix,
  currentGroupValue,
  groupLabel,
  maxSeries = 8,
  emphasizeValue = null,
}) {
  const cleanRows = (rows ?? []).filter((row) => {
    const x = row?.[xField];
    const g = row?.[groupField];
    return Number.isFinite(x) && x > 0 && Number.isFinite(g) && g > 0 && Number.isFinite(row?.loss);
  });

  const allGroups = groupRowsByPositiveField(cleanRows, groupField, xField);

  if (allGroups.length < 2 || countUniquePositiveX(cleanRows, xField) < 2) {
    return makeSingleKaplanSeries(cleanRows, xField, "generated run slice");
  }

  const selectedGroups = pickKaplanGroups(allGroups, currentGroupValue, maxSeries, emphasizeValue);
  const pivot = new Map();

  for (const group of selectedGroups) {
    const seriesKey = kaplanSeriesKey(prefix, group.value);

    for (const row of group.rows) {
      const x = row[xField];
      if (!Number.isFinite(x) || x <= 0) continue;

      const xKey = stableValueKey(x);
      const existing = pivot.get(xKey) ?? { [xField]: x };
      const current = existing[seriesKey];

      // If duplicate cells map to the same plotted point, keep the lower loss.
      existing[seriesKey] = Number.isFinite(current) ? Math.min(current, row.loss) : row.loss;
      pivot.set(xKey, existing);
    }
  }

  const data = Array.from(pivot.values()).sort((a, b) => a[xField] - b[xField]);
  const emphasizedKey = Number.isFinite(emphasizeValue) && emphasizeValue > 0
    ? selectedGroups.reduce((best, group) => {
        if (!best) return group;
        return Math.abs(Math.log(group.value / emphasizeValue)) < Math.abs(Math.log(best.value / emphasizeValue))
          ? group
          : best;
      }, null)?.key
    : null;
  const nearestCurrentKey = Number.isFinite(currentGroupValue) && currentGroupValue > 0
    ? selectedGroups.reduce((best, group) => {
        if (!best) return group;
        return Math.abs(Math.log(group.value / currentGroupValue)) < Math.abs(Math.log(best.value / currentGroupValue))
          ? group
          : best;
      }, null)?.key
    : null;

  const series = selectedGroups.map((group, index) => {
    const isEmphasized = group.key === emphasizedKey;
    const isCurrent = group.key === nearestCurrentKey;

    return {
      key: kaplanSeriesKey(prefix, group.value),
      name: `${groupLabel}=${formatKaplanLegendValue(group.value)}${isEmphasized ? " largest" : isCurrent ? " current" : ""}`,
      color: isEmphasized ? "#60a5fa" : KAPLAN_COLORS[index % KAPLAN_COLORS.length],
      strokeWidth: isEmphasized ? 2.8 : isCurrent ? 2.3 : 1.8,
    };
  });

  return { data, series };
}

function sortedKaplanProjection(rows, xField, maxPoints = 128) {
  const data = (rows ?? [])
    .filter((row) => Number.isFinite(row?.[xField]) && row[xField] > 0 && Number.isFinite(row?.loss))
    .sort((a, b) => a[xField] - b[xField])
    .map((row) => ({
      [xField]: row[xField],
      Cmin: row.pf_days,
      D: row.D,
      N: row.N,
      loss: row.loss,
    }));

  return downsampleEvenly(data, maxPoints);
}

function makeKaplanPanelsFromSweepRows({ rows, baseN, baseD, sourceDescription }) {
  const cleanRows = (rows ?? []).filter((row) => {
    return Number.isFinite(row?.loss) &&
      Number.isFinite(row?.N) && row.N > 0 &&
      Number.isFinite(row?.D) && row.D > 0 &&
      Number.isFinite(row?.pf_days) && row.pf_days > 0;
  });

  if (cleanRows.length === 0) {
    return makeKaplanDesignerPanels({
      sweepRanges: {},
      sweepPoints: 8,
      baseN,
      baseD,
    });
  }

  // Kaplan-style edge slices:
  //   L(D): hold model size fixed to the largest N available in the sweep.
  //   L(N): hold dataset size fixed to the largest D available in the sweep.
  // This avoids connecting unrelated points from a 2D matrix while still using
  // the full matrix to decide which edge curves are the most informative.
  const largestNSlice = chooseExtremeKaplanSlice({
    rows: cleanRows,
    xField: "D",
    fixedField: "N",
    prefer: "max",
  });
  const largestDSlice = chooseExtremeKaplanSlice({
    rows: cleanRows,
    xField: "N",
    fixedField: "D",
    prefer: "max",
  });

  const datasetSliceRows = makeCurveRowsFromSlice(largestNSlice, "D");
  const paramsSliceRows = makeCurveRowsFromSlice(largestDSlice, "N");
  const datasetData = datasetSliceRows.length >= 2
    ? datasetSliceRows
    : sortedKaplanProjection(cleanRows, "D");
  const paramsData = paramsSliceRows.length >= 2
    ? paramsSliceRows
    : sortedKaplanProjection(cleanRows, "N");
  const computeData = makeComputeFrontierRows(cleanRows, 36);

  const largestN = largestNSlice.found ? largestNSlice.fixedValue : Math.max(...cleanRows.map((row) => row.N));
  const largestD = largestDSlice.found ? largestDSlice.fixedValue : Math.max(...cleanRows.map((row) => row.D));

  const figure4 = makeGroupedKaplanSeries({
    rows: cleanRows,
    xField: "D",
    groupField: "N",
    prefix: "N",
    currentGroupValue: baseN,
    groupLabel: "N",
    emphasizeValue: largestN,
  });
  const figure9 = makeGroupedKaplanSeries({
    rows: cleanRows,
    xField: "N",
    groupField: "D",
    prefix: "D",
    currentGroupValue: baseD,
    groupLabel: "D",
    emphasizeValue: largestD,
  });

  return {
    baseN,
    baseD,
    sourceDescription: sourceDescription ?? "Generated from Kaplan-style edge slices of the active sweep data.",
    ranges: {
      D: finitePositiveDomainFromRows(datasetData, "D"),
      N: finitePositiveDomainFromRows(paramsData, "N"),
      compute: finitePositiveDomainFromRows(computeData, "Cmin"),
    },
    figure1: {
      computeData,
      datasetData,
      paramsData,
      currentCompute: flops_to_pf_days(training_flops(baseN, baseD)),
      computeSubtitle: "Lower-envelope over the generated matrix, binned logarithmically by compute.",
      datasetSubtitle: `Holds N fixed at the largest swept model, ${formatKaplanLegendValue(largestN)}, and scans D.`,
      paramsSubtitle: `Holds D fixed at the largest swept dataset, ${formatKaplanLegendValue(largestD)} tokens, and scans N.`,
    },
    figure4: {
      data: figure4.data,
      series: figure4.series,
      xDomain: finitePositiveDomainFromRows(figure4.data, "D"),
      yDomain: finiteYDomainFromKeys(
        figure4.data,
        figure4.series.map((s) => s.key)
      ),
      subtitle: `Kaplan-style fixed-N curves from your sweep matrix. The largest-N curve (${formatKaplanLegendValue(largestN)}) is highlighted, matching the L(D) edge used above.`,
    },
    figure9: {
      data: figure9.data,
      series: figure9.series,
      xDomain: finitePositiveDomainFromRows(figure9.data, "N"),
      yDomain: finiteYDomainFromKeys(
        figure9.data,
        figure9.series.map((s) => s.key)
      ),
      subtitle: `Kaplan-style fixed-D curves from your sweep matrix. The largest-D curve (${formatKaplanLegendValue(largestD)} tokens) is highlighted, matching the L(N) edge used above.`,
    },
  };
}

function plotlyLogRange(domain) {
  if (!Array.isArray(domain) || domain.length !== 2) return undefined;

  const rawLo = Number(domain[0]);
  const rawHi = Number(domain[1]);
  if (!Number.isFinite(rawLo) || !Number.isFinite(rawHi)) return undefined;

  const lo = Math.max(Math.min(rawLo, rawHi), 1e-300);
  const hi = Math.max(Math.max(rawLo, rawHi), lo * 1.0001);
  return [Math.log10(lo), Math.log10(hi)];
}

function plotlyLinearRange(domain) {
  if (!Array.isArray(domain) || domain.length !== 2) return undefined;

  const lo = Number(domain[0]);
  const hi = Number(domain[1]);
  if (!Number.isFinite(lo) || !Number.isFinite(hi)) return undefined;
  if (Math.abs(hi - lo) < 1e-12) {
    const pad = Math.max(Math.abs(lo) * 0.05, 0.05);
    return [lo - pad, hi + pad];
  }

  return [Math.min(lo, hi), Math.max(lo, hi)];
}

function plotlyAxisRange(domain, isLog) {
  return isLog ? plotlyLogRange(domain) : plotlyLinearRange(domain);
}

function finiteLinearDomainFromData(data, key, padFraction = 0.08) {
  const vals = (data ?? [])
    .map((row) => row?.[key])
    .filter((v) => Number.isFinite(v));

  if (vals.length === 0) return undefined;

  const lo = Math.min(...vals);
  const hi = Math.max(...vals);
  if (Math.abs(hi - lo) < 1e-12) {
    const pad = Math.max(Math.abs(lo) * 0.05, 0.05);
    return [lo - pad, hi + pad];
  }

  const pad = (hi - lo) * padFraction;
  return [lo - pad, hi + pad];
}

function nicePlotNumber(x, digits = 4) {
  if (!Number.isFinite(x)) return "—";
  if (Math.abs(x) >= 1e4 || Math.abs(x) < 1e-3) return Number(x).toExponential(3);
  return Number(x).toFixed(digits).replace(/0+$/, "").replace(/\.$/, "");
}

function plotLayout({
  xTitle,
  yTitle,
  logX = false,
  logY = false,
  xDomain,
  yDomain,
  shapes = [],
  legendY = 1.13,
  margin = { l: 64, r: 24, t: 12, b: 54 },
  hovermode = "closest",
  y2Title = null,
  logY2 = false,
}) {
  const layout = {
    autosize: true,
    paper_bgcolor: "rgba(0,0,0,0)",
    plot_bgcolor: "rgba(0,0,0,0)",
    font: { color: "#d4d4d8", size: 12 },
    margin,
    hovermode,
    dragmode: "zoom",
    uirevision: "designer-plotly-interactive",
    legend: {
      orientation: "h",
      x: 0,
      y: legendY,
      font: { size: 11, color: "#d4d4d8" },
      bgcolor: "rgba(9,9,11,0.35)",
    },
    xaxis: {
      title: { text: xTitle, font: { color: "#a1a1aa", size: 12 } },
      type: logX ? "log" : "linear",
      range: plotlyAxisRange(xDomain, logX),
      gridcolor: "rgba(255,255,255,0.08)",
      zerolinecolor: "rgba(255,255,255,0.14)",
      linecolor: "#3f3f46",
      tickfont: { color: "#a1a1aa", size: 11 },
      exponentformat: "power",
      showspikes: true,
      spikemode: "across",
      spikedash: "dot",
      spikecolor: "rgba(255,255,255,0.30)",
    },
    yaxis: {
      title: { text: yTitle, font: { color: "#a1a1aa", size: 12 } },
      type: logY ? "log" : "linear",
      range: plotlyAxisRange(yDomain, logY),
      gridcolor: "rgba(255,255,255,0.08)",
      zerolinecolor: "rgba(255,255,255,0.14)",
      linecolor: "#3f3f46",
      tickfont: { color: "#a1a1aa", size: 11 },
      exponentformat: "power",
      showspikes: true,
      spikemode: "across",
      spikedash: "dot",
      spikecolor: "rgba(255,255,255,0.30)",
    },
    shapes,
  };

  if (y2Title) {
    layout.yaxis2 = {
      title: { text: y2Title, font: { color: "#a1a1aa", size: 12 } },
      type: logY2 ? "log" : "linear",
      overlaying: "y",
      side: "right",
      gridcolor: "rgba(255,255,255,0)",
      zerolinecolor: "rgba(255,255,255,0.14)",
      linecolor: "#3f3f46",
      tickfont: { color: "#a1a1aa", size: 11 },
      exponentformat: "power",
      showspikes: true,
      spikemode: "across",
      spikedash: "dot",
      spikecolor: "rgba(255,255,255,0.30)",
    };
  }

  return layout;
}

function plotConfig(filename) {
  return {
    responsive: true,
    scrollZoom: true,
    displaylogo: false,
    displayModeBar: true,
    modeBarButtonsToRemove: ["lasso2d", "select2d"],
    toImageButtonOptions: {
      format: "png",
      filename: sanitizeFilenamePart(filename),
      height: 900,
      width: 1400,
      scale: 2,
    },
  };
}

function PlotCard({ title, subtitle, traces, layout, config, heightClass = "h-64", footer }) {
  return (
    <ChartShell
      title={title}
      subtitle={subtitle}
      footer={footer ?? "Drag to zoom. Mouse wheel zoom, pan, autoscale, reset axes, and PNG export are enabled."}
      heightClass={heightClass}
    >
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

function makeVerticalLineShape(x, color = "#fbbf24") {
  if (!Number.isFinite(x) || x <= 0) return null;
  return {
    type: "line",
    xref: "x",
    yref: "paper",
    x0: x,
    x1: x,
    y0: 0,
    y1: 1,
    line: { color, width: 2, dash: "dash" },
  };
}

function makeHorizontalLineShape(y, color = "#fbbf24") {
  if (!Number.isFinite(y)) return null;
  return {
    type: "line",
    xref: "paper",
    yref: "y",
    x0: 0,
    x1: 1,
    y0: y,
    y1: y,
    line: { color, width: 2, dash: "dash" },
  };
}

function numericLineTrace({
  data,
  xKey,
  yKey,
  name,
  color = "#60a5fa",
  dash = "solid",
  mode = "lines+markers",
  markerSize = 6,
  width = 2.2,
  text,
  yaxis,
  fill,
  fillcolor,
}) {
  const clean = (data ?? [])
    .filter((row) => Number.isFinite(row?.[xKey]) && Number.isFinite(row?.[yKey]))
    .sort((a, b) => a[xKey] - b[xKey]);

  return {
    type: "scatter",
    mode,
    name,
    x: clean.map((row) => row[xKey]),
    y: clean.map((row) => row[yKey]),
    text: text ? clean.map(text) : clean.map((row) => `${xKey}: ${human_num(row[xKey])}<br>${yKey}: ${nicePlotNumber(row[yKey])}`),
    hovertemplate: "%{text}<extra></extra>",
    line: { color, width, dash },
    marker: {
      color,
      size: markerSize,
      line: { color: "rgba(255,255,255,0.35)", width: 1 },
    },
    yaxis,
    fill,
    fillcolor,
  };
}

function axisLabelForSweepSpec(sweepSpec) {
  return sweepSpec?.label ?? "Sweep variable";
}

function KaplanFigure1Panel({ title, subtitle, data, xKey, xLabel, equation, xDomain, currentX }) {
  const lineTrace = numericLineTrace({
    data,
    xKey,
    yKey: "loss",
    name: "your config sweep",
    color: "#60a5fa",
    dash: "dash",
    width: 2.2,
    markerSize: 7,
    text: (row) => [
      `${xLabel}: ${human_num(row[xKey])}`,
      `N: ${human_num(row.N)}`,
      `D: ${human_num(row.D)}`,
      `loss: ${Number(row.loss).toFixed(5)}`,
    ].join("<br>"),
  });

  const shapes = [makeVerticalLineShape(currentX)].filter(Boolean);

  return (
    <div className="rounded-xl border border-zinc-800 bg-zinc-950/70 p-3">
      <div className="mb-2 min-h-14">
        <h3 className="text-sm font-semibold text-zinc-100">{title}</h3>
        <p className="mt-1 text-[11px] text-zinc-500">{subtitle}</p>
      </div>

      <div className="h-56">
        <Plot
          data={[lineTrace]}
          layout={plotLayout({
            xTitle: xLabel,
            yTitle: "Predicted loss",
            logX: true,
            xDomain,
            yDomain: finiteLinearDomainFromData(data, "loss"),
            shapes,
            legendY: 1.03,
            margin: { l: 58, r: 16, t: 4, b: 48 },
          })}
          config={plotConfig(`${title} Kaplan projection`)}
          useResizeHandler
          className="h-full w-full"
          style={{ width: "100%", height: "100%" }}
        />
      </div>

      <div className="mt-1 rounded-lg border border-zinc-800 bg-zinc-950 px-2 py-1 text-[11px] text-zinc-400">
        <MathText>{equation}</MathText>
      </div>
    </div>
  );
}

function KaplanFigure1Triptych({ panels }) {
  const computeData = panels.figure1.computeData;
  const datasetData = panels.figure1.datasetData;
  const paramsData = panels.figure1.paramsData;

  return (
    <div className="rounded-2xl border border-zinc-800 bg-zinc-950/80 p-3">
      <div className="mb-3 flex flex-col gap-1 sm:flex-row sm:items-end sm:justify-between">
        <div>
          <h2 className="text-base font-semibold text-zinc-100">
            Kaplan Figure 1 style: current-input power-law projections
          </h2>
          <p className="text-xs text-zinc-500">
            {panels.sourceDescription ?? "These are not hard-coded paper points. They are generated from your current inputs."}
          </p>
        </div>
        <div className="text-[11px] text-zinc-500">dots are log-spaced generated runs; dashed lines connect them</div>
      </div>

      <div className="grid grid-cols-1 gap-3 xl:grid-cols-3">
        <KaplanFigure1Panel
          title="Compute"
          subtitle={panels.figure1.computeSubtitle ?? "Paired log-spaced N and D from your configured ranges"}
          data={computeData}
          xKey="Cmin"
          xLabel="Compute, PF-days"
          xDomain={panels.ranges.compute}
          currentX={panels.figure1.currentCompute}
          equation={"\\(C\\approx6ND,\\quad L=L(N,D)\\)"}
        />
        <KaplanFigure1Panel
          title="Dataset size"
          subtitle={panels.figure1.datasetSubtitle ?? `Holds N fixed at ${formatKaplanLegendValue(panels.baseN)} and sweeps your D range`}
          data={datasetData}
          xKey="D"
          xLabel="Dataset size D"
          xDomain={[panels.ranges.D[0] / 1.08, panels.ranges.D[1] * 1.08]}
          currentX={panels.baseD}
          equation={"\\(L=L(N_{\\rm fixed},D)\\)"}
        />
        <KaplanFigure1Panel
          title="Parameters"
          subtitle={panels.figure1.paramsSubtitle ?? `Holds D fixed at ${formatKaplanLegendValue(panels.baseD)} tokens and sweeps your N range`}
          data={paramsData}
          xKey="N"
          xLabel="Parameters N"
          xDomain={[panels.ranges.N[0] / 1.08, panels.ranges.N[1] * 1.08]}
          currentX={panels.baseN}
          equation={"\\(L=L(N,D_{\\rm fixed})\\)"}
        />
      </div>
    </div>
  );
}

function KaplanMultiSeriesChart({
  title,
  subtitle,
  data,
  xKey,
  xLabel,
  yLabel,
  series,
  xDomain,
  yDomain,
  referenceX,
}) {
  const traces = (series ?? []).map((s, index) =>
    numericLineTrace({
      data,
      xKey,
      yKey: s.key,
      name: s.name,
      color: s.color ?? KAPLAN_COLORS[index % KAPLAN_COLORS.length],
      dash: "dash",
      width: s.strokeWidth ?? 1.8,
      markerSize: 6,
      text: (row) => [
        `${xLabel}: ${human_num(row[xKey])}`,
        `${s.name}: ${Number(row[s.key]).toFixed(5)}`,
      ].join("<br>"),
    })
  );

  const shapes = [makeVerticalLineShape(referenceX)].filter(Boolean);

  return (
    <PlotCard
      title={title}
      subtitle={subtitle}
      heightClass="h-80"
      traces={traces}
      layout={plotLayout({
        xTitle: xLabel,
        yTitle: yLabel,
        logX: true,
        xDomain,
        yDomain,
        shapes,
      })}
    />
  );
}

function KaplanFigure4LeftChart({ panels }) {
  return (
    <KaplanMultiSeriesChart
      title="Kaplan Figure 4 left style: loss vs dataset size"
      subtitle={panels.figure4.subtitle ?? "Each dashed line holds N fixed. The N values come from your configured parameter-count range; the highlighted line is your current N when it lies inside that range."}
      data={panels.figure4.data}
      xKey="D"
      xLabel="Tokens in dataset D"
      yLabel="Predicted loss"
      series={panels.figure4.series}
      xDomain={panels.figure4.xDomain}
      yDomain={panels.figure4.yDomain}
      referenceX={panels.baseD}
    />
  );
}

function KaplanFigure9LeftChart({ panels }) {
  return (
    <KaplanMultiSeriesChart
      title="Kaplan Figure 9 left style: loss vs model size"
      subtitle={panels.figure9.subtitle ?? "Each dashed line holds D fixed. The D values come from your configured dataset-size range; the highlighted line is your current D when it lies inside that range."}
      data={panels.figure9.data}
      xKey="N"
      xLabel="Parameters N"
      yLabel="Predicted loss"
      series={panels.figure9.series}
      xDomain={panels.figure9.xDomain}
      yDomain={panels.figure9.yDomain}
      referenceX={panels.baseN}
    />
  );
}

function KaplanPaperPanels({ panels }) {
  return (
    <section className="space-y-3">
      <KaplanFigure1Triptych panels={panels} />
      <section className="grid grid-cols-1 gap-3 xl:grid-cols-2">
        <KaplanFigure4LeftChart panels={panels} />
        <KaplanFigure9LeftChart panels={panels} />
      </section>
    </section>
  );
}

function clamp01(x) {
  return Math.max(0, Math.min(1, x));
}

function heatColor(t, invert = false) {
  const u = invert ? 1 - clamp01(t) : clamp01(t);
  const hue = 215 - 170 * u;
  const lightness = 18 + 38 * u;
  return `hsl(${hue} 82% ${lightness}%)`;
}

function HeatmapChart({
  title,
  subtitle,
  grid,
  xSpec,
  ySpec,
  valueKey,
  valueLabel,
  formatValue = human_num,
  lowerIsBetter = true,
  logColor = false,
}) {
  const rows = grid?.rows ?? [];
  const xValues = grid?.xValues ?? [];
  const yValues = grid?.yValues ?? [];

  if (xValues.length === 0 || yValues.length === 0) {
    return (
      <ChartShell title={title} subtitle={subtitle} heightClass="min-h-64">
        <div className="flex h-full items-center justify-center rounded-xl border border-zinc-800 bg-zinc-950 text-sm text-zinc-500">
          No 2D sweep data available.
        </div>
      </ChartShell>
    );
  }

  const rowByKey = new Map(rows.map((row) => [`${row.x}__${row.y}`, row]));
  const rawValues = rows
    .map((row) => row?.[valueKey])
    .filter((v) => Number.isFinite(v) && (!logColor || v > 0));
  const z = yValues.map((y) =>
    xValues.map((x) => {
      const raw = rowByKey.get(`${x}__${y}`)?.[valueKey];
      if (!Number.isFinite(raw)) return null;
      return logColor ? Math.log10(raw) : raw;
    })
  );

  const customdata = yValues.map((y) =>
    xValues.map((x) => {
      const row = rowByKey.get(`${x}__${y}`);
      return [
        formatValue(row?.[valueKey]),
        human_num(row?.N),
        human_num(row?.D),
        row?.regime ?? "—",
      ];
    })
  );

  const trace = {
    type: "heatmap",
    name: valueLabel,
    x: xValues,
    y: yValues,
    z,
    customdata,
    colorscale: lowerIsBetter
      ? [[0, "#facc15"], [0.5, "#2563eb"], [1, "#0f172a"]]
      : [[0, "#0f172a"], [0.5, "#2563eb"], [1, "#facc15"]],
    colorbar: {
      title: { text: logColor ? `log10 ${valueLabel}` : valueLabel, font: { color: "#a1a1aa" } },
      tickfont: { color: "#a1a1aa" },
    },
    hovertemplate:
      `${xSpec.label}: %{x}<br>` +
      `${ySpec.label}: %{y}<br>` +
      `${valueLabel}: %{customdata[0]}<br>` +
      `N: %{customdata[1]}<br>` +
      `D: %{customdata[2]}<br>` +
      `Regime: %{customdata[3]}<extra></extra>`,
  };

  return (
    <PlotCard
      title={title}
      subtitle={subtitle}
      heightClass="min-h-80"
      traces={[trace]}
      layout={plotLayout({
        xTitle: xSpec.label,
        yTitle: ySpec.label,
        logX: xSpec.scale === "log",
        logY: ySpec.scale === "log",
        xDomain: xSpec.scale === "log" ? finitePositiveDomainFromData(rows, "x", 1.02) : plotlyLinearRange([Math.min(...xValues), Math.max(...xValues)]),
        yDomain: ySpec.scale === "log" ? finitePositiveDomainFromData(rows, "y", 1.02) : plotlyLinearRange([Math.min(...yValues), Math.max(...yValues)]),
        legendY: 1.02,
        margin: { l: 62, r: 24, t: 10, b: 58 },
      })}
      footer={
        <div className="flex flex-wrap items-center justify-between gap-2">
          <span>x: {xSpec.label}; y: {ySpec.label}</span>
          <span>
            {valueLabel}: {formatValue(rawValues.length ? Math.min(...rawValues) : NaN)} → {formatValue(rawValues.length ? Math.max(...rawValues) : NaN)}
          </span>
          <span>Drag to zoom, shift-drag or modebar to pan, double-click to reset.</span>
        </div>
      }
    />
  );
}

function TwoDSweepHeatmaps({ grid, xSpec, ySpec }) {
  return (
    <section className="space-y-3">
      <div className="rounded-2xl border border-zinc-800 bg-zinc-950/80 p-3">
        <div className="flex flex-col gap-1 sm:flex-row sm:items-end sm:justify-between">
          <div>
            <h2 className="text-base font-semibold text-zinc-100">2D sweep surface</h2>
            <p className="text-xs text-zinc-500">
              Each square is one generated run. Existing line charts below use the midpoint slice of the y-axis variable.
            </p>
          </div>
          <div className="text-[11px] text-zinc-500">
            {grid.xValues.length} × {grid.yValues.length} = {grid.rows.length} candidate runs
          </div>
        </div>
      </div>

      <section className="grid grid-cols-1 gap-3 xl:grid-cols-3">
        <HeatmapChart
          title="Predicted loss heatmap"
          subtitle="Lower is better. Uses Kaplan L(N,D) evaluated at every 2D grid point."
          grid={grid}
          xSpec={xSpec}
          ySpec={ySpec}
          valueKey="loss"
          valueLabel="loss"
          formatValue={(v) => Number.isFinite(v) ? Number(v).toFixed(5) : "—"}
          lowerIsBetter
        />
        <HeatmapChart
          title="Compute heatmap"
          subtitle="Training compute estimate C ≈ 6ND, converted to PF-days."
          grid={grid}
          xSpec={xSpec}
          ySpec={ySpec}
          valueKey="pf_days"
          valueLabel="PF-days"
          formatValue={(v) => Number.isFinite(v) ? `${Number(v).toFixed(4)}` : "—"}
          lowerIsBetter
          logColor
        />
        <HeatmapChart
          title="D/N balance heatmap"
          subtitle="Distance from D/N ≈ 20. Darker cells are closer to the compute-optimal rule of thumb."
          grid={grid}
          xSpec={xSpec}
          ySpec={ySpec}
          valueKey="balance_distance"
          valueLabel="|log((D/N)/20)|"
          formatValue={(v) => Number.isFinite(v) ? Number(v).toFixed(3) : "—"}
          lowerIsBetter
        />
      </section>
    </section>
  );
}

function ExpectedLossChart({ data, sweepSpec }) {
  const isLog = sweepSpec.scale === "log";
  const xDomain = isLog ? finitePositiveDomainFromData(data, "x", 1.02) : finiteLinearDomainFromData(data, "x");

  return (
    <PlotCard
      title="Expected Kaplan loss over the sweep"
      subtitle="Uses L(N,D), holding non-swept quantities fixed."
      traces={[
        numericLineTrace({
          data,
          xKey: "x",
          yKey: "loss",
          name: "Kaplan loss",
          color: "#60a5fa",
          mode: "lines+markers",
          markerSize: 6,
          text: (row) => [
            `${axisLabelForSweepSpec(sweepSpec)}: ${human_num(row.x)}`,
            `N: ${human_num(row.N)}`,
            `D: ${human_num(row.D)}`,
            `loss: ${Number(row.loss).toFixed(5)}`,
          ].join("<br>"),
        }),
      ]}
      layout={plotLayout({
        xTitle: axisLabelForSweepSpec(sweepSpec),
        yTitle: "loss",
        logX: isLog,
        xDomain,
        yDomain: finiteLinearDomainFromData(data, "loss"),
      })}
    />
  );
}

function ComputeChart({ data, sweepSpec }) {
  const isLog = sweepSpec.scale === "log";
  const xDomain = isLog ? finitePositiveDomainFromData(data, "x", 1.02) : finiteLinearDomainFromData(data, "x");

  return (
    <PlotCard
      title="Training compute over the sweep"
      subtitle="Uses the usual dense-transformer estimate C ≈ 6ND."
      traces={[
        numericLineTrace({
          data,
          xKey: "x",
          yKey: "pf_days",
          name: "PF-days",
          color: "#34d399",
          mode: "lines+markers",
          markerSize: 6,
          text: (row) => [
            `${axisLabelForSweepSpec(sweepSpec)}: ${human_num(row.x)}`,
            `N: ${human_num(row.N)}`,
            `D: ${human_num(row.D)}`,
            `PF-days: ${Number(row.pf_days).toFixed(4)}`,
          ].join("<br>"),
        }),
      ]}
      layout={plotLayout({
        xTitle: axisLabelForSweepSpec(sweepSpec),
        yTitle: "PF-days",
        logX: isLog,
        xDomain,
        yDomain: finiteLinearDomainFromData(data, "pf_days"),
      })}
    />
  );
}

function RatioChart({ data, sweepSpec }) {
  const isLog = sweepSpec.scale === "log";
  const xDomain = isLog ? finitePositiveDomainFromData(data, "x", 1.02) : finiteLinearDomainFromData(data, "x");
  const shapes = [makeHorizontalLineShape(20)].filter(Boolean);

  return (
    <PlotCard
      title="Data/model balance over the sweep"
      subtitle="The horizontal reference is the rough Chinchilla / Power-Lines-style D/N ≈ 20 heuristic."
      traces={[
        numericLineTrace({
          data,
          xKey: "x",
          yKey: "D_over_N",
          name: "D/N",
          color: "#60a5fa",
          mode: "lines+markers",
          markerSize: 6,
          text: (row) => [
            `${axisLabelForSweepSpec(sweepSpec)}: ${human_num(row.x)}`,
            `D/N: ${human_num(row.D_over_N)}`,
            `N: ${human_num(row.N)}`,
            `D: ${human_num(row.D)}`,
          ].join("<br>"),
        }),
      ]}
      layout={plotLayout({
        xTitle: axisLabelForSweepSpec(sweepSpec),
        yTitle: "D/N",
        logX: isLog,
        logY: true,
        xDomain,
        yDomain: finitePositiveDomainFromData(data.concat([{ D_over_N: 20 }]), "D_over_N", 1.2),
        shapes,
      })}
    />
  );
}

function BatchEnvelopeChart({ data, sweepSpec }) {
  const isLog = sweepSpec.scale === "log";
  const xDomain = isLog ? finitePositiveDomainFromData(data, "x", 1.02) : finiteLinearDomainFromData(data, "x");
  const domainRows = data.flatMap((row) => [
    { y: row.Bopt },
    { y: row.Btarget },
    { y: row.Bcrit },
    { y: row.Bactual },
  ]);

  const traces = [
    numericLineTrace({
      data,
      xKey: "x",
      yKey: "Bcrit",
      name: "Bcrit",
      color: "#818cf8",
      mode: "lines",
      width: 2,
      fill: "tozeroy",
      fillcolor: "rgba(129,140,248,0.12)",
      text: (row) => `${axisLabelForSweepSpec(sweepSpec)}: ${human_num(row.x)}<br>Bcrit: ${human_num(row.Bcrit)}`,
    }),
    numericLineTrace({
      data,
      xKey: "x",
      yKey: "Bopt",
      name: "Bopt",
      color: "#60a5fa",
      mode: "lines+markers",
      markerSize: 5,
      text: (row) => `${axisLabelForSweepSpec(sweepSpec)}: ${human_num(row.x)}<br>Bopt: ${human_num(row.Bopt)}`,
    }),
    numericLineTrace({
      data,
      xKey: "x",
      yKey: "Btarget",
      name: "Btarget",
      color: "#fbbf24",
      dash: "dash",
      mode: "lines+markers",
      markerSize: 5,
      text: (row) => `${axisLabelForSweepSpec(sweepSpec)}: ${human_num(row.x)}<br>Btarget: ${human_num(row.Btarget)}`,
    }),
    numericLineTrace({
      data,
      xKey: "x",
      yKey: "Bactual",
      name: "Bactual",
      color: "#fb7185",
      dash: "dot",
      mode: "lines+markers",
      markerSize: 5,
      text: (row) => `${axisLabelForSweepSpec(sweepSpec)}: ${human_num(row.x)}<br>Bactual: ${human_num(row.Bactual)}`,
    }),
  ];

  return (
    <PlotCard
      title="Power Lines batch envelope over the sweep"
      subtitle="Compares your actual global batch to Bopt, geometric target, and Bcrit."
      traces={traces}
      layout={plotLayout({
        xTitle: axisLabelForSweepSpec(sweepSpec),
        yTitle: "sequences",
        logX: isLog,
        logY: true,
        xDomain,
        yDomain: finitePositiveDomainFromData(domainRows, "y", 1.2),
      })}
      footer={
        <span>
          <MathText>{"\\(B_{\\rm target}=\\sqrt{B_{\\rm opt}B_{\\rm crit}}\\)"}</MathText>{" "}
          is a practical middle choice. Drag to zoom, shift-drag/modebar to pan, double-click to reset.
        </span>
      }
    />
  );
}

function ArchitectureChart({ data, sweepSpec }) {
  const isLog = sweepSpec.scale === "log";
  const xDomain = isLog ? finitePositiveDomainFromData(data, "x", 1.02) : finiteLinearDomainFromData(data, "x");

  const traces = [
    numericLineTrace({
      data,
      xKey: "x",
      yKey: "N",
      name: "N params",
      color: "#60a5fa",
      mode: "lines+markers",
      markerSize: 6,
      text: (row) => `${axisLabelForSweepSpec(sweepSpec)}: ${human_num(row.x)}<br>N: ${human_num(row.N)}`,
    }),
    numericLineTrace({
      data,
      xKey: "x",
      yKey: "head_dim",
      name: "head dim",
      color: "#fbbf24",
      dash: "dash",
      mode: "lines+markers",
      markerSize: 6,
      yaxis: "y2",
      text: (row) => `${axisLabelForSweepSpec(sweepSpec)}: ${human_num(row.x)}<br>head dim: ${nicePlotNumber(row.head_dim)}`,
    }),
  ];

  return (
    <PlotCard
      title="Implied architecture and model size"
      subtitle="Useful when the sweep variable is a structural hyperparameter."
      traces={traces}
      layout={plotLayout({
        xTitle: axisLabelForSweepSpec(sweepSpec),
        yTitle: "N params",
        y2Title: "head dim",
        logX: isLog,
        logY: true,
        xDomain,
        yDomain: finitePositiveDomainFromData(data, "N", 1.2),
        legendY: 1.15,
      })}
    />
  );
}

function RegimeBar({ summary }) {
  const ratio = summary.D_over_N;
  const logRatio = Math.log10(Math.max(ratio, 1e-12));
  const lo = Math.log10(1);
  const hi = Math.log10(200);
  const pct = clamp(((logRatio - lo) / (hi - lo)) * 100, 0, 100);

  return (
    <div className="rounded-2xl border border-zinc-800 bg-zinc-950/80 p-3">
      <div className="mb-3 flex flex-col gap-1 sm:flex-row sm:items-end sm:justify-between">
        <div>
          <h2 className="text-base font-semibold text-zinc-100">Current midpoint regime</h2>
          <p className="text-xs text-zinc-500">
            The marker uses the middle sweep point as a representative design.
          </p>
        </div>
        <div className="text-sm font-semibold text-blue-200">
          {classifyRegime(summary.D, summary.N)}
        </div>
      </div>

      <div className="relative h-10 rounded-full bg-zinc-900">
        <div className="absolute inset-y-0 left-[0%] w-[33%] rounded-l-full bg-amber-500/15" />
        <div className="absolute inset-y-0 left-[33%] w-[20%] bg-blue-500/15" />
        <div className="absolute inset-y-0 left-[53%] w-[47%] rounded-r-full bg-emerald-500/15" />
        <div
          className="absolute top-1/2 h-8 w-8 -translate-x-1/2 -translate-y-1/2 rounded-full border-2 border-blue-300 bg-zinc-950 shadow-lg shadow-blue-500/20"
          style={{ left: `${pct}%` }}
          title={`D/N = ${human_num(ratio)}`}
        />
        <div className="absolute left-[43%] top-1/2 h-8 w-1 -translate-y-1/2 rounded-full bg-amber-300" />
      </div>

      <div className="mt-2 grid grid-cols-3 gap-2 text-[11px] text-zinc-500">
        <div>data-limited</div>
        <div className="text-center">near 20 tokens / parameter</div>
        <div className="text-right">model-limited</div>
      </div>
    </div>
  );
}

function JsonBlock({ value, filename = "designer_sweep_protocol.json" }) {
  return (
    <div className="rounded-2xl border border-zinc-800 bg-zinc-950/80 p-3">
      <div className="mb-2 flex flex-col gap-2 sm:flex-row sm:items-start sm:justify-between">
        <div>
          <h2 className="text-base font-semibold text-zinc-100">Exportable sweep protocol</h2>
          <p className="mt-1 text-xs text-zinc-500">
            Save this JSON and pass it to the companion Python script to generate the same particleGPT sweep configs outside the browser.
          </p>
        </div>
        <button
          type="button"
          onClick={() => downloadJsonValue(value, filename)}
          className="rounded-xl border border-zinc-700 bg-zinc-900 px-3 py-1.5 text-xs font-semibold text-zinc-200 transition hover:border-blue-500/60 hover:text-blue-100"
        >
          Download JSON
        </button>
      </div>
      <pre className="max-h-80 overflow-auto rounded-xl border border-zinc-800 bg-zinc-950 p-3 text-xs leading-relaxed text-zinc-300">
        {JSON.stringify(value, null, 2)}
      </pre>
    </div>
  );
}



function SweepProtocolImportPanel({ onImport }) {
  const [textValue, setTextValue] = useState("");
  const [status, setStatus] = useState(null);

  const applyText = () => {
    try {
      const parsed = JSON.parse(textValue);
      const message = onImport(parsed);
      setStatus({ type: "success", text: message ?? "Imported sweep protocol." });
    } catch (error) {
      setStatus({
        type: "error",
        text: error?.message ? `Import failed: ${error.message}` : "Import failed.",
      });
    }
  };

  const readFile = (file) => {
    if (!file) return;
    const reader = new FileReader();

    reader.onload = () => {
      const nextText = String(reader.result ?? "");
      setTextValue(nextText);
      try {
        const parsed = JSON.parse(nextText);
        const message = onImport(parsed);
        setStatus({ type: "success", text: message ?? `Imported ${file.name}.` });
      } catch (error) {
        setStatus({
          type: "error",
          text: error?.message ? `Import failed: ${error.message}` : "Import failed.",
        });
      }
    };

    reader.onerror = () => {
      setStatus({ type: "error", text: `Could not read ${file.name}.` });
    };

    reader.readAsText(file);
  };

  return (
    <div className="rounded-2xl border border-zinc-800 bg-zinc-950/80 p-3">
      <div className="mb-2 flex items-start justify-between gap-3">
        <div>
          <h2 className="text-sm font-semibold uppercase tracking-wide text-zinc-400">
            Import / resume
          </h2>
          <p className="mt-1 text-xs leading-relaxed text-zinc-500">
            Paste or upload an exported sweep protocol JSON to restore the input panel.
          </p>
        </div>
      </div>

      <textarea
        value={textValue}
        onChange={(event) => setTextValue(event.target.value)}
        placeholder="Paste designer_sweep_protocol.json here..."
        className="h-36 w-full resize-y rounded-xl border border-zinc-800 bg-zinc-950 p-3 text-xs leading-relaxed text-zinc-300 outline-none transition placeholder:text-zinc-700 focus:border-blue-500 focus:ring-1 focus:ring-blue-500"
        spellCheck={false}
      />

      <div className="mt-2 flex flex-col gap-2 sm:flex-row sm:items-center sm:justify-between">
        <input
          type="file"
          accept="application/json,.json"
          onChange={(event) => readFile(event.target.files?.[0])}
          className="block min-w-0 text-xs text-zinc-500 file:mr-3 file:rounded-lg file:border file:border-zinc-700 file:bg-zinc-900 file:px-3 file:py-1.5 file:text-xs file:font-semibold file:text-zinc-200 hover:file:border-blue-500/60"
        />

        <button
          type="button"
          onClick={applyText}
          disabled={!textValue.trim()}
          className="rounded-xl border border-blue-500/40 bg-blue-500/15 px-4 py-2 text-sm font-semibold text-blue-100 transition hover:border-blue-400 hover:bg-blue-500/25 disabled:cursor-not-allowed disabled:opacity-50"
        >
          Apply JSON
        </button>
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

function ConfigZipPanel({ files, zipFilename, summaryRows }) {
  const preview = files?.[0]?.config ?? null;

  return (
    <div className="rounded-2xl border border-zinc-800 bg-zinc-950/80 p-3">
      <div className="flex flex-col gap-3 sm:flex-row sm:items-start sm:justify-between">
        <div>
          <h2 className="text-base font-semibold text-zinc-100">particleGPT config ZIP</h2>
          <p className="mt-1 text-xs text-zinc-500">
            Generates one JSON file per active sweep point using the current constants and sweep matrix.
          </p>
        </div>
        <button
          type="button"
          onClick={() => downloadParticleGPTConfigZip(files, zipFilename)}
          disabled={!files || files.length === 0}
          className="rounded-xl border border-blue-500/40 bg-blue-500/15 px-4 py-2 text-sm font-semibold text-blue-100 transition hover:border-blue-400 hover:bg-blue-500/25 disabled:cursor-not-allowed disabled:opacity-50"
        >
          Download ZIP
        </button>
      </div>

      <div className="mt-3 grid grid-cols-1 gap-2 md:grid-cols-3">
        {summaryRows.map((row) => (
          <div key={row.label} className="rounded-xl border border-zinc-800 bg-zinc-950 px-3 py-2">
            <div className="text-[10px] uppercase tracking-wide text-zinc-500">{row.label}</div>
            <div className="mt-1 truncate text-sm font-semibold text-zinc-200">{row.value}</div>
          </div>
        ))}
      </div>

      {preview && (
        <details className="mt-3 rounded-xl border border-zinc-800 bg-zinc-950/70 p-3">
          <summary className="cursor-pointer text-sm font-medium text-zinc-200">
            Preview first generated config
          </summary>
          <pre className="mt-3 max-h-80 overflow-auto rounded-xl border border-zinc-800 bg-zinc-950 p-3 text-xs leading-relaxed text-zinc-300">
            {JSON.stringify(preview, null, 4)}
          </pre>
        </details>
      )}
    </div>
  );
}

function ControlSection({ title, children }) {
  return (
    <section className="space-y-3">
      <h2 className="text-sm font-semibold uppercase tracking-wide text-zinc-400">
        {title}
      </h2>
      {children}
    </section>
  );
}

export default function ScalingLawDesignerPage() {
  const layoutRef = useRef(null);
  const [sidebarWidth, setSidebarWidth] = useState(460);
  const [isResizingSidebar, setIsResizingSidebar] = useState(false);

  const [sweepMode, setSweepMode] = useState("1d");
  const [sweepVariable, setSweepVariable] = useState("D_tokens");
  const [secondSweepVariable, setSecondSweepVariable] = useState("N_params");
  const [kaplanPlotSource, setKaplanPlotSource] = useState("matrix");
  const [sweepRanges, setSweepRanges] = useState(() => {
    return Object.fromEntries(
      Object.entries(SWEEP_SPECS).map(([key, spec]) => [
        key,
        {
          min: spec.defaultMin,
          max: spec.defaultMax,
        },
      ])
    );
  });
  const [sweepPoints, setSweepPoints] = useState(56);
  const [secondSweepPoints, setSecondSweepPoints] = useState(14);

  // Architecture constants.
  const [nLayer, setNLayer] = useState(8);
  const [nHead, setNHead] = useState(8);
  const [nEmbd, setNEmbd] = useState(512);
  const [mlpRatio, setMlpRatio] = useState(4);
  const [useNOverride, setUseNOverride] = useState(false);
  const [logNOverride, setLogNOverride] = useState(valueToLogSlider(20e6));

  // Dataset and training constants.
  const [logD, setLogD] = useState(valueToLogSlider(35e9));
  const [seqLen, setSeqLen] = useState(1024);
  const [logBatchSeq, setLogBatchSeq] = useState(valueToLogSlider(32));
  const [gradAccum, setGradAccum] = useState(96);
  const [worldSize, setWorldSize] = useState(4);
  const [batchIsPerGpu, setBatchIsPerGpu] = useState(false);

  // Optimizer constants.
  const [logPeakLr, setLogPeakLr] = useState(valueToLogSlider(3e-4));

  // particleGPT config export settings.
  const [configFilePrefix, setConfigFilePrefix] = useState("model_designer_sweep");
  const [preparationName, setPreparationName] = useState("preparation_upgrade_1_benchmarks_1");
  const [datasetName, setDatasetName] = useState("rdataset_full_50M.csv");
  const [dataMode, setDataMode] = useState("generic");
  const [contextEvents, setContextEvents] = useState(1);
  const [nKvHeads, setNKvHeads] = useState(2);
  const [dropout, setDropout] = useState(0.1);
  const [evalInterval, setEvalInterval] = useState(100);
  const [evalIters, setEvalIters] = useState(200);
  const [logInterval, setLogInterval] = useState(10);
  const [maxItersMode, setMaxItersMode] = useState("computed_from_tokens");
  const [fixedMaxIters, setFixedMaxIters] = useState(1000);
  const [lrDecayMode, setLrDecayMode] = useState("fraction_of_max_iters");
  const [fixedLrDecayIters, setFixedLrDecayIters] = useState(400);
  const [lrDecayFraction, setLrDecayFraction] = useState(0.4);
  const [logMinLr, setLogMinLr] = useState(valueToLogSlider(5e-5));
  const [beta2, setBeta2] = useState(0.98);
  const [warmupIters, setWarmupIters] = useState(27);
  const [maxFailedCheckpointChecks, setMaxFailedCheckpointChecks] = useState(-1);
  const [evalEveryEpoch, setEvalEveryEpoch] = useState(true);
  const [temperature, setTemperature] = useState(0.8);
  const [topK, setTopK] = useState(200);
  const [samplingSeed, setSamplingSeed] = useState(1337);
  const [samplingDevice, setSamplingDevice] = useState("cuda");
  const [compileSampling, setCompileSampling] = useState(true);
  const [includeDesignerMetadata, setIncludeDesignerMetadata] = useState(true);

  const d_sweepMode = useDebouncedValue(sweepMode);
  const d_sweepVariable = useDebouncedValue(sweepVariable);
  const d_secondSweepVariableRaw = useDebouncedValue(secondSweepVariable);
  const d_kaplanPlotSource = useDebouncedValue(kaplanPlotSource);
  const d_sweepRanges = useDebouncedValue(sweepRanges);
  const d_sweepPoints = useDebouncedValue(sweepPoints);
  const d_secondSweepPoints = useDebouncedValue(secondSweepPoints);
  const d_nLayer = useDebouncedValue(nLayer);
  const d_nHead = useDebouncedValue(nHead);
  const d_nEmbd = useDebouncedValue(nEmbd);
  const d_mlpRatio = useDebouncedValue(mlpRatio);
  const d_logNOverride = useDebouncedValue(logNOverride);
  const d_logD = useDebouncedValue(logD);
  const d_seqLen = useDebouncedValue(seqLen);
  const d_logBatchSeq = useDebouncedValue(logBatchSeq);
  const d_gradAccum = useDebouncedValue(gradAccum);
  const d_worldSize = useDebouncedValue(worldSize);
  const d_logPeakLr = useDebouncedValue(logPeakLr);
  const d_configFilePrefix = useDebouncedValue(configFilePrefix);
  const d_preparationName = useDebouncedValue(preparationName);
  const d_datasetName = useDebouncedValue(datasetName);
  const d_dataMode = useDebouncedValue(dataMode);
  const d_contextEvents = useDebouncedValue(contextEvents);
  const d_nKvHeads = useDebouncedValue(nKvHeads);
  const d_dropout = useDebouncedValue(dropout);
  const d_evalInterval = useDebouncedValue(evalInterval);
  const d_evalIters = useDebouncedValue(evalIters);
  const d_logInterval = useDebouncedValue(logInterval);
  const d_maxItersMode = useDebouncedValue(maxItersMode);
  const d_fixedMaxIters = useDebouncedValue(fixedMaxIters);
  const d_lrDecayMode = useDebouncedValue(lrDecayMode);
  const d_fixedLrDecayIters = useDebouncedValue(fixedLrDecayIters);
  const d_lrDecayFraction = useDebouncedValue(lrDecayFraction);
  const d_logMinLr = useDebouncedValue(logMinLr);
  const d_beta2 = useDebouncedValue(beta2);
  const d_warmupIters = useDebouncedValue(warmupIters);
  const d_maxFailedCheckpointChecks = useDebouncedValue(maxFailedCheckpointChecks);
  const d_temperature = useDebouncedValue(temperature);
  const d_topK = useDebouncedValue(topK);
  const d_samplingSeed = useDebouncedValue(samplingSeed);
  const d_samplingDevice = useDebouncedValue(samplingDevice);

  const sweepSpec = SWEEP_SPECS[sweepVariable];
  const secondSweepOptions = Object.entries(SWEEP_SPECS)
    .filter(([value]) => value !== sweepVariable)
    .map(([value, spec]) => ({ value, label: spec.label }));
  const effectiveSecondSweepVariable = secondSweepVariable === sweepVariable
    ? secondSweepOptions[0]?.value ?? "N_params"
    : secondSweepVariable;
  const secondSweepSpec = SWEEP_SPECS[effectiveSecondSweepVariable];
  const d_effectiveSecondSweepVariable = d_secondSweepVariableRaw === d_sweepVariable
    ? Object.keys(SWEEP_SPECS).find((key) => key !== d_sweepVariable)
    : d_secondSweepVariableRaw;
  const d_secondSweepSpec = SWEEP_SPECS[d_effectiveSecondSweepVariable];
  const d_is2DSweep = d_sweepMode === "2d" &&
    d_effectiveSecondSweepVariable !== d_sweepVariable &&
    Boolean(d_secondSweepSpec);
  const d_sweepSpec = SWEEP_SPECS[d_sweepVariable];
  const currentRange = sweepRanges[sweepVariable] ?? {
    min: sweepSpec.defaultMin,
    max: sweepSpec.defaultMax,
  };
  const currentSecondRange = sweepRanges[effectiveSecondSweepVariable] ?? {
    min: secondSweepSpec.defaultMin,
    max: secondSweepSpec.defaultMax,
  };
  const sweptVariableSet = new Set(
    sweepMode === "2d"
      ? [sweepVariable, effectiveSecondSweepVariable]
      : [sweepVariable]
  );

  useEffect(() => {
    if (!isResizingSidebar) return;

    const oldCursor = document.body.style.cursor;
    const oldUserSelect = document.body.style.userSelect;

    const onPointerMove = (event) => {
      const rect = layoutRef.current?.getBoundingClientRect();
      const left = rect?.left ?? 0;
      const layoutWidth = rect?.width ?? window.innerWidth;
      const maxSidebarWidth = Math.max(460, Math.min(860, layoutWidth - 420));
      const nextWidth = clamp(event.clientX - left, 460, maxSidebarWidth);
      setSidebarWidth(nextWidth);
    };

    const onPointerUp = () => {
      setIsResizingSidebar(false);
    };

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

  const setRangeValueFor = (variable, which, value) => {
    setSweepRanges((prev) => ({
      ...prev,
      [variable]: {
        ...(prev[variable] ?? {
          min: SWEEP_SPECS[variable].defaultMin,
          max: SWEEP_SPECS[variable].defaultMax,
        }),
        [which]: value,
      },
    }));
  };

  const setCurrentRangeValue = (which, value) => {
    setRangeValueFor(sweepVariable, which, value);
  };

  const setSecondRangeValue = (which, value) => {
    setRangeValueFor(effectiveSecondSweepVariable, which, value);
  };


  const applyImportedSweepProtocol = (protocol) => {
    if (!isPlainObject(protocol)) {
      throw new Error("Top-level JSON must be an object.");
    }

    const sweep = protocol.sweep ?? protocol.input_panel?.sweep ?? {};
    const inputPanel = protocol.input_panel ?? {};
    const constants = protocol.constants ?? {};
    const architecture = inputPanel.architecture ?? {};
    const datasetTraining = inputPanel.dataset_and_training ?? {};
    const optimizer = inputPanel.optimizer ?? {};
    const bookkeeping = inputPanel.bookkeeping ?? {};
    const sampling = inputPanel.sampling ?? {};
    const exportSettings = protocol.particleGPT_config_export ?? inputPanel.particleGPT_config_export ?? {};
    const trainingDefaults = exportSettings.training_config_defaults ?? {};
    const samplingDefaults = exportSettings.sampling_config_defaults ?? {};

    const importedMode = firstDefined(sweep.mode, inputPanel.sweep?.mode);
    const nextMode = ["1d", "2d"].includes(importedMode) ? importedMode : "1d";
    const nextXVariable = sweepVariableOrFallback(
      firstDefined(sweep.x_variable, sweep.x_axis?.variable, inputPanel.sweep?.x_axis?.variable),
      "D_tokens"
    );
    const fallbackY = Object.keys(SWEEP_SPECS).find((key) => key !== nextXVariable) ?? "N_params";
    let nextYVariable = sweepVariableOrFallback(
      firstDefined(sweep.y_variable, sweep.y_axis?.variable, inputPanel.sweep?.y_axis?.variable),
      fallbackY
    );

    if (nextYVariable === nextXVariable) nextYVariable = fallbackY;

    setSweepMode(nextMode);
    setSweepVariable(nextXVariable);
    setSecondSweepVariable(nextYVariable);
    setKaplanPlotSource(optionOrFallback(
      firstDefined(sweep.kaplan_plot_source, inputPanel.sweep?.kaplan_plot_source),
      KAPLAN_PLOT_SOURCE_OPTIONS,
      "matrix"
    ));
    setSweepPoints(integerOrFallback(firstDefined(sweep.x_points, sweep.x_axis?.points, inputPanel.sweep?.x_axis?.points), sweepPoints, { min: 2, max: 256 }));
    setSecondSweepPoints(integerOrFallback(firstDefined(sweep.y_points, sweep.y_axis?.points, inputPanel.sweep?.y_axis?.points), secondSweepPoints, { min: 2, max: 256 }));

    const defaultRanges = Object.fromEntries(
      Object.entries(SWEEP_SPECS).map(([key, spec]) => [
        key,
        { min: spec.defaultMin, max: spec.defaultMax },
      ])
    );

    setSweepRanges(() => {
      return Object.fromEntries(
        Object.entries(SWEEP_SPECS).map(([variable, spec]) => {
          const fallbackRange = sweepRanges?.[variable] ?? defaultRanges[variable];
          const parsedRange = readProtocolRangeForVariable({
            protocol,
            variable,
            fallbackRange,
          });

          return [
            variable,
            {
              min: numberOrFallback(parsedRange.min, spec.defaultMin),
              max: numberOrFallback(parsedRange.max, spec.defaultMax),
            },
          ];
        })
      );
    });

    const nextNLayer = integerOrFallback(firstDefined(architecture.n_layer, constants.n_layer, trainingDefaults.n_layer), nLayer, { min: 1, max: 100000 });
    const nextNHead = integerOrFallback(firstDefined(architecture.n_head, constants.n_head, trainingDefaults.n_head), nHead, { min: 1, max: 100000 });
    const nextNEmbd = integerOrFallback(firstDefined(architecture.n_embd, constants.n_embd, trainingDefaults.n_embd), nEmbd, { min: 1, max: 1000000 });
    const nextMlpRatio = numberOrFallback(firstDefined(architecture.mlp_ratio, constants.mlp_ratio), mlpRatio);
    const nextUseNOverride = booleanOrFallback(firstDefined(architecture.use_N_override, constants.use_N_override), useNOverride);
    const nextNOverride = firstDefined(architecture.N_params_override, constants.N_params_override);

    setNLayer(nextNLayer);
    setNHead(nextNHead);
    setNEmbd(nextNEmbd);
    setMlpRatio(Number.isFinite(nextMlpRatio) ? nextMlpRatio : mlpRatio);
    setUseNOverride(nextUseNOverride);
    if (nextNOverride !== undefined && nextNOverride !== null) {
      setLogNOverride(positiveActualToLogSlider(nextNOverride, logNOverride));
    }

    const nextD = firstDefined(datasetTraining.D_tokens, constants.D_tokens);
    if (nextD !== undefined && nextD !== null) {
      setLogD(positiveActualToLogSlider(nextD, logD));
    }

    setSeqLen(integerOrFallback(firstDefined(datasetTraining.seq_len, constants.seq_len), seqLen, { min: 1, max: 1000000 }));
    const nextBatchSeq = firstDefined(datasetTraining.batch_size_sequences, constants.batch_size_sequences, trainingDefaults.batch_size);
    if (nextBatchSeq !== undefined && nextBatchSeq !== null) {
      setLogBatchSeq(positiveActualToLogSlider(nextBatchSeq, logBatchSeq));
    }
    setGradAccum(integerOrFallback(firstDefined(datasetTraining.grad_accum, constants.grad_accum, trainingDefaults.gradient_accumulation_steps), gradAccum, { min: 1, max: 1000000 }));
    setWorldSize(integerOrFallback(firstDefined(datasetTraining.num_gpus, constants.num_gpus), worldSize, { min: 1, max: 1000000 }));
    setBatchIsPerGpu(booleanOrFallback(firstDefined(datasetTraining.batch_size_is_per_gpu, constants.batch_size_is_per_gpu), batchIsPerGpu));
    setContextEvents(integerOrFallback(firstDefined(datasetTraining.context_events, trainingDefaults.context_events), contextEvents, { min: 1, max: 1000000 }));
    setDropout(numberOrFallback(firstDefined(datasetTraining.dropout, trainingDefaults.dropout), dropout));

    const nextPeakLr = firstDefined(optimizer.peak_lr, constants.peak_lr, trainingDefaults.learning_rate);
    if (nextPeakLr !== undefined && nextPeakLr !== null) {
      setLogPeakLr(positiveActualToLogSlider(nextPeakLr, logPeakLr));
    }

    const nextMinLr = firstDefined(optimizer.min_lr, constants.min_lr, trainingDefaults.min_lr);
    if (nextMinLr !== undefined && nextMinLr !== null) {
      setLogMinLr(positiveActualToLogSlider(nextMinLr, logMinLr));
    }

    setBeta2(numberOrFallback(firstDefined(optimizer.beta2, trainingDefaults.beta2), beta2));
    setWarmupIters(integerOrFallback(firstDefined(optimizer.warmup_iters, trainingDefaults.warmup_iters), warmupIters, { min: 0, max: 1000000 }));
    setMaxItersMode(optionOrFallback(firstDefined(optimizer.max_iters_mode, exportSettings.max_iters_mode), MAX_ITERS_MODE_OPTIONS, maxItersMode));
    setFixedMaxIters(integerOrFallback(firstDefined(optimizer.fixed_max_iters, exportSettings.fixed_max_iters, trainingDefaults.max_iters), fixedMaxIters, { min: 1, max: 1000000000 }));
    setLrDecayMode(optionOrFallback(firstDefined(optimizer.lr_decay_mode, exportSettings.lr_decay_mode), LR_DECAY_MODE_OPTIONS, lrDecayMode));
    setFixedLrDecayIters(integerOrFallback(firstDefined(optimizer.fixed_lr_decay_iters, exportSettings.fixed_lr_decay_iters, trainingDefaults.lr_decay_iters), fixedLrDecayIters, { min: 1, max: 1000000000 }));
    setLrDecayFraction(numberOrFallback(firstDefined(optimizer.lr_decay_fraction, exportSettings.lr_decay_fraction), lrDecayFraction));

    setEvalInterval(integerOrFallback(firstDefined(bookkeeping.eval_interval, trainingDefaults.eval_interval), evalInterval, { min: 1, max: 1000000 }));
    setEvalIters(integerOrFallback(firstDefined(bookkeeping.eval_iters, trainingDefaults.eval_iters), evalIters, { min: 1, max: 1000000 }));
    setLogInterval(integerOrFallback(firstDefined(bookkeeping.log_interval, trainingDefaults.log_interval), logInterval, { min: 1, max: 1000000 }));
    setMaxFailedCheckpointChecks(integerOrFallback(firstDefined(bookkeeping.max_num_failed_checkpoint_checks, trainingDefaults.max_num_failed_checkpoint_checks), maxFailedCheckpointChecks, { min: -1, max: 1000000 }));
    setEvalEveryEpoch(booleanOrFallback(firstDefined(bookkeeping.eval_every_epoch, trainingDefaults.eval_every_epoch), evalEveryEpoch));

    setConfigFilePrefix(String(firstDefined(exportSettings.file_prefix, configFilePrefix)));
    setPreparationName(String(firstDefined(exportSettings.preparation_name, preparationName)));
    setDatasetName(String(firstDefined(exportSettings.dataset, datasetName)));
    setDataMode(String(firstDefined(exportSettings.data_mode, dataMode)));
    setNKvHeads(integerOrFallback(firstDefined(architecture.n_kv_heads, trainingDefaults.n_kv_heads), nKvHeads, { min: 1, max: 100000 }));
    setTemperature(numberOrFallback(firstDefined(sampling.temperature, samplingDefaults.temperature), temperature));
    setTopK(integerOrFallback(firstDefined(sampling.top_k, samplingDefaults.top_k), topK, { min: 1, max: 1000000 }));
    setSamplingSeed(integerOrFallback(firstDefined(sampling.seed, samplingDefaults.seed), samplingSeed, { min: 0, max: 2147483647 }));
    setSamplingDevice(String(firstDefined(sampling.device, samplingDefaults.device, samplingDevice)));
    setCompileSampling(booleanOrFallback(firstDefined(sampling.compile, samplingDefaults.compile), compileSampling));
    setIncludeDesignerMetadata(booleanOrFallback(exportSettings.include_designer_metadata, includeDesignerMetadata));

    return `Imported ${nextMode.toUpperCase()} sweep protocol with ${nextMode === "2d" ? `${nextXVariable} × ${nextYVariable}` : nextXVariable}.`;
  };

  const DConstant = logSliderToValue(d_logD);
  const batchSeqConstant = Math.round(logSliderToValue(d_logBatchSeq));
  const NOverride = logSliderToValue(d_logNOverride);
  const peakLr = logSliderToValue(d_logPeakLr);
  const minLr = logSliderToValue(d_logMinLr);

  const architectureN = useMemo(() => {
    return gpt_like_nonembedding_params(
      d_nLayer,
      d_nEmbd,
      d_mlpRatio,
      false
    );
  }, [d_nLayer, d_nEmbd, d_mlpRatio]);

  const baseN = useNOverride ? NOverride : architectureN;

  const baseConfig = useMemo(() => ({
    n_layer: d_nLayer,
    n_head: d_nHead,
    n_embd: d_nEmbd,
    mlp_ratio: d_mlpRatio,
    D_tokens: DConstant,
    N_params: baseN,
    batch_size_sequences: batchSeqConstant,
  }), [d_nLayer, d_nHead, d_nEmbd, d_mlpRatio, DConstant, baseN, batchSeqConstant]);

  const sweepData = useMemo(() => {
    const spec = d_sweepSpec;
    const range = d_sweepRanges[d_sweepVariable] ?? {
      min: spec.defaultMin,
      max: spec.defaultMax,
    };
    const [rangeMin, rangeMax] = getActualRangeForSweep(spec, range);
    const values = makeSweepValues(spec, rangeMin, rangeMax, d_sweepPoints);

    let secondaryMidpoint = null;
    let activeVariables = [d_sweepVariable];

    if (d_is2DSweep && d_secondSweepSpec) {
      const secondRange = d_sweepRanges[d_effectiveSecondSweepVariable] ?? {
        min: d_secondSweepSpec.defaultMin,
        max: d_secondSweepSpec.defaultMax,
      };
      const [secondMin, secondMax] = getActualRangeForSweep(d_secondSweepSpec, secondRange);
      const secondValues = makeSweepValues(d_secondSweepSpec, secondMin, secondMax, d_secondSweepPoints);
      secondaryMidpoint = secondValues[Math.floor(secondValues.length / 2)] ?? null;
      activeVariables = [d_sweepVariable, d_effectiveSecondSweepVariable];
    }

    return values.map((x) => {
      const cfg = { ...baseConfig };
      applySweepVariableToConfig(cfg, d_sweepVariable, x);

      if (d_is2DSweep && secondaryMidpoint !== null) {
        applySweepVariableToConfig(cfg, d_effectiveSecondSweepVariable, secondaryMidpoint);
      }

      return evaluateSweepConfig({
        cfg,
        x,
        y: secondaryMidpoint,
        activeVariables,
        useNOverride,
        baseN,
        batchIsPerGpu,
        gradAccum: d_gradAccum,
        worldSize: d_worldSize,
        seqLen: d_seqLen,
      });
    });
  }, [
    d_sweepSpec,
    d_sweepRanges,
    d_sweepVariable,
    d_sweepPoints,
    d_is2DSweep,
    d_secondSweepSpec,
    d_effectiveSecondSweepVariable,
    d_secondSweepPoints,
    baseConfig,
    useNOverride,
    baseN,
    batchIsPerGpu,
    d_gradAccum,
    d_worldSize,
    d_seqLen,
  ]);


  const ySweepData = useMemo(() => {
    if (!d_is2DSweep || !d_secondSweepSpec) return [];

    const xRange = d_sweepRanges[d_sweepVariable] ?? {
      min: d_sweepSpec.defaultMin,
      max: d_sweepSpec.defaultMax,
    };
    const yRange = d_sweepRanges[d_effectiveSecondSweepVariable] ?? {
      min: d_secondSweepSpec.defaultMin,
      max: d_secondSweepSpec.defaultMax,
    };

    const [xMin, xMax] = getActualRangeForSweep(d_sweepSpec, xRange);
    const [yMin, yMax] = getActualRangeForSweep(d_secondSweepSpec, yRange);
    const xValues = makeSweepValues(d_sweepSpec, xMin, xMax, d_sweepPoints);
    const yValues = makeSweepValues(d_secondSweepSpec, yMin, yMax, d_secondSweepPoints);
    const xMidpoint = xValues[Math.floor(xValues.length / 2)] ?? null;
    const activeVariables = [d_sweepVariable, d_effectiveSecondSweepVariable];

    return yValues.map((y) => {
      const cfg = { ...baseConfig };

      if (xMidpoint !== null) {
        applySweepVariableToConfig(cfg, d_sweepVariable, xMidpoint);
      }

      applySweepVariableToConfig(cfg, d_effectiveSecondSweepVariable, y);

      return evaluateSweepConfig({
        cfg,
        x: y,
        y: xMidpoint,
        activeVariables,
        useNOverride,
        baseN,
        batchIsPerGpu,
        gradAccum: d_gradAccum,
        worldSize: d_worldSize,
        seqLen: d_seqLen,
      });
    });
  }, [
    d_is2DSweep,
    d_secondSweepSpec,
    d_sweepRanges,
    d_sweepVariable,
    d_sweepSpec,
    d_effectiveSecondSweepVariable,
    d_sweepPoints,
    d_secondSweepPoints,
    baseConfig,
    useNOverride,
    baseN,
    batchIsPerGpu,
    d_gradAccum,
    d_worldSize,
    d_seqLen,
  ]);

  const sweep2DGrid = useMemo(() => {
    if (!d_is2DSweep || !d_secondSweepSpec) {
      return { rows: [], xValues: [], yValues: [] };
    }

    const xRange = d_sweepRanges[d_sweepVariable] ?? {
      min: d_sweepSpec.defaultMin,
      max: d_sweepSpec.defaultMax,
    };
    const yRange = d_sweepRanges[d_effectiveSecondSweepVariable] ?? {
      min: d_secondSweepSpec.defaultMin,
      max: d_secondSweepSpec.defaultMax,
    };

    const [xMin, xMax] = getActualRangeForSweep(d_sweepSpec, xRange);
    const [yMin, yMax] = getActualRangeForSweep(d_secondSweepSpec, yRange);
    const xValues = makeSweepValues(d_sweepSpec, xMin, xMax, d_sweepPoints);
    const yValues = makeSweepValues(d_secondSweepSpec, yMin, yMax, d_secondSweepPoints);
    const activeVariables = [d_sweepVariable, d_effectiveSecondSweepVariable];
    const rows = [];

    for (const y of yValues) {
      for (const x of xValues) {
        const cfg = { ...baseConfig };
        applySweepVariableToConfig(cfg, d_sweepVariable, x);
        applySweepVariableToConfig(cfg, d_effectiveSecondSweepVariable, y);
        rows.push(evaluateSweepConfig({
          cfg,
          x,
          y,
          activeVariables,
          useNOverride,
          baseN,
          batchIsPerGpu,
          gradAccum: d_gradAccum,
          worldSize: d_worldSize,
          seqLen: d_seqLen,
        }));
      }
    }

    return { rows, xValues, yValues };
  }, [
    d_is2DSweep,
    d_secondSweepSpec,
    d_sweepRanges,
    d_sweepVariable,
    d_sweepSpec,
    d_effectiveSecondSweepVariable,
    d_sweepPoints,
    d_secondSweepPoints,
    baseConfig,
    useNOverride,
    baseN,
    batchIsPerGpu,
    d_gradAccum,
    d_worldSize,
    d_seqLen,
  ]);

  const kaplanPanelData = useMemo(() => {
    if (!d_is2DSweep) {
      return makeKaplanPanelsFromSweepRows({
        rows: sweepData,
        baseN,
        baseD: DConstant,
        sourceDescription: "Kaplan-style panels are generated from the active 1D sweep. Points are your proposed runs, not hard-coded paper values.",
      });
    }

    if (d_kaplanPlotSource === "x_slice") {
      return makeKaplanPanelsFromSweepRows({
        rows: sweepData,
        baseN,
        baseD: DConstant,
        sourceDescription: `Kaplan-style panels use the X-axis sweep (${d_sweepSpec.label}) with ${d_secondSweepSpec.label} held at its midpoint.`,
      });
    }

    if (d_kaplanPlotSource === "y_slice") {
      return makeKaplanPanelsFromSweepRows({
        rows: ySweepData,
        baseN,
        baseD: DConstant,
        sourceDescription: `Kaplan-style panels use the Y-axis sweep (${d_secondSweepSpec.label}) with ${d_sweepSpec.label} held at its midpoint.`,
      });
    }

    return makeKaplanPanelsFromSweepRows({
      rows: sweep2DGrid.rows,
      baseN,
      baseD: DConstant,
      sourceDescription: `Kaplan-style panels use the full 2D matrix (${sweep2DGrid.rows.length} generated runs) but plot Kaplan-style edge slices: L(D) at the largest available N and L(N) at the largest available D.`,
    });
  }, [
    d_is2DSweep,
    d_kaplanPlotSource,
    sweepData,
    ySweepData,
    sweep2DGrid.rows,
    baseN,
    DConstant,
    d_sweepSpec,
    d_secondSweepSpec,
  ]);

  const midpoint = sweepData[Math.floor(sweepData.length / 2)] ?? {};
  const bestBalanced = useMemo(() => {
    return sweepData.reduce((best, row) => {
      if (!best) return row;
      return Math.abs(Math.log(row.D_over_N / 20)) < Math.abs(Math.log(best.D_over_N / 20))
        ? row
        : best;
    }, null);
  }, [sweepData]);

  const minLoss = useMemo(() => {
    return sweepData.reduce((best, row) => {
      if (!best) return row;
      return row.loss < best.loss ? row : best;
    }, null);
  }, [sweepData]);

  const lowestCompute = useMemo(() => {
    return sweepData.reduce((best, row) => {
      if (!best) return row;
      return row.pf_days < best.pf_days ? row : best;
    }, null);
  }, [sweepData]);

  const sampledRunRows = useMemo(() => {
    if (sweepData.length === 0) return [];

    const targetRows = Math.min(10, sweepData.length);
    const rows = [];
    const seen = new Set();

    for (let i = 0; i < targetRows; i++) {
      const index = Math.round((i * (sweepData.length - 1)) / Math.max(targetRows - 1, 1));
      if (seen.has(index)) continue;
      seen.add(index);
      const row = sweepData[index];
      rows.push({
        point: index + 1,
        variable: row.x,
        N: row.N,
        D: row.D,
        D_over_N: row.D_over_N,
        loss: row.loss,
        pf_days: row.pf_days,
        B_target: row.Btarget,
        steps: row.steps,
        regime: row.regime,
      });
    }

    return rows;
  }, [sweepData]);

  const sampled2DRunRows = useMemo(() => {
    if (!d_is2DSweep || sweep2DGrid.rows.length === 0) return [];

    const targetRows = Math.min(12, sweep2DGrid.rows.length);
    const rows = [];
    const seen = new Set();

    for (let i = 0; i < targetRows; i++) {
      const index = Math.round((i * (sweep2DGrid.rows.length - 1)) / Math.max(targetRows - 1, 1));
      if (seen.has(index)) continue;
      seen.add(index);
      const row = sweep2DGrid.rows[index];
      rows.push({
        point: index + 1,
        variable: row.x,
        second_variable: row.y,
        N: row.N,
        D: row.D,
        D_over_N: row.D_over_N,
        loss: row.loss,
        pf_days: row.pf_days,
        B_target: row.Btarget,
        steps: row.steps,
        regime: row.regime,
      });
    }

    return rows;
  }, [d_is2DSweep, sweep2DGrid.rows]);

  const recommendationRows = useMemo(() => {
    const rows = [];

    if (midpoint?.x) {
      rows.push({
        quantity: "midpoint design",
        value: `${d_sweepSpec.label} = ${human_num(midpoint.x)}`,
      });
      rows.push({ quantity: "midpoint regime", value: midpoint.regime });
    }

    if (bestBalanced) {
      rows.push({
        quantity: "closest to D/N ≈ 20",
        value: `${human_num(bestBalanced.x)}; D/N = ${human_num(bestBalanced.D_over_N)}`,
      });
    }

    if (minLoss) {
      rows.push({
        quantity: "lowest predicted loss",
        value: `${human_num(minLoss.x)}; loss = ${minLoss.loss.toFixed(5)}`,
      });
    }

    if (lowestCompute) {
      rows.push({
        quantity: "lowest compute",
        value: `${human_num(lowestCompute.x)}; ${lowestCompute.pf_days.toFixed(4)} PF-days`,
      });
    }

    if (d_is2DSweep && sweep2DGrid.rows.length > 0) {
      const best2DLoss = sweep2DGrid.rows.reduce((best, row) => !best || row.loss < best.loss ? row : best, null);
      const best2DBalance = sweep2DGrid.rows.reduce((best, row) => !best || row.balance_distance < best.balance_distance ? row : best, null);
      const best2DCompute = sweep2DGrid.rows.reduce((best, row) => !best || row.pf_days < best.pf_days ? row : best, null);

      if (best2DLoss) {
        rows.push({
          quantity: "2D lowest loss",
          value: `${d_sweepSpec.mathLabel}=${human_num(best2DLoss.x)}, ${d_secondSweepSpec.mathLabel}=${human_num(best2DLoss.y)}; loss=${best2DLoss.loss.toFixed(5)}`,
        });
      }

      if (best2DBalance) {
        rows.push({
          quantity: "2D closest to D/N ≈ 20",
          value: `${d_sweepSpec.mathLabel}=${human_num(best2DBalance.x)}, ${d_secondSweepSpec.mathLabel}=${human_num(best2DBalance.y)}; D/N=${human_num(best2DBalance.D_over_N)}`,
        });
      }

      if (best2DCompute) {
        rows.push({
          quantity: "2D lowest compute",
          value: `${d_sweepSpec.mathLabel}=${human_num(best2DCompute.x)}, ${d_secondSweepSpec.mathLabel}=${human_num(best2DCompute.y)}; ${best2DCompute.pf_days.toFixed(4)} PF-days`,
        });
      }
    }

    rows.push({
      quantity: "peak learning rate",
      value: formatSci(peakLr, 2),
    });

    return rows;
  }, [
    midpoint,
    bestBalanced,
    minLoss,
    lowestCompute,
    d_sweepSpec,
    d_secondSweepSpec,
    d_is2DSweep,
    sweep2DGrid.rows,
    peakLr,
  ]);

  const exportConfig = useMemo(() => {
    const serializeSweepRange = (variable, points = null) => {
      const spec = SWEEP_SPECS[variable];
      if (!spec) return null;

      const range = d_sweepRanges[variable] ?? {
        min: spec.defaultMin,
        max: spec.defaultMax,
      };

      const [sliderMin, sliderMax] = getActualRangeForSweep(spec, range);
      const minActual = spec.scale === "log" ? logSliderToValue(sliderMin) : sliderMin;
      const maxActual = spec.scale === "log" ? logSliderToValue(sliderMax) : sliderMax;

      return {
        variable,
        label: spec.label,
        math_label: spec.mathLabel,
        unit: spec.unit,
        scale: spec.scale,
        min: minActual,
        max: maxActual,
        points,
        slider_min: sliderMin,
        slider_max: sliderMax,
        step: spec.step ?? null,
      };
    };

    const xAxis = serializeSweepRange(d_sweepVariable, d_sweepPoints);
    const yAxis = d_is2DSweep
      ? serializeSweepRange(d_effectiveSecondSweepVariable, d_secondSweepPoints)
      : null;

    const allSweepRanges = Object.fromEntries(
      Object.keys(SWEEP_SPECS).map((variable) => [
        variable,
        serializeSweepRange(variable, variable === d_sweepVariable
          ? d_sweepPoints
          : variable === d_effectiveSecondSweepVariable && d_is2DSweep
            ? d_secondSweepPoints
            : null),
      ])
    );

    const architectureSettings = {
      n_layer: d_nLayer,
      n_head: d_nHead,
      n_kv_heads: d_nKvHeads,
      n_embd: d_nEmbd,
      mlp_ratio: d_mlpRatio,
      use_N_override: useNOverride,
      N_params_override: useNOverride ? NOverride : null,
      estimated_architecture_N_params: architectureN,
    };

    const datasetTrainingSettings = {
      D_tokens: DConstant,
      seq_len: d_seqLen,
      batch_size_sequences: batchSeqConstant,
      grad_accum: d_gradAccum,
      num_gpus: d_worldSize,
      batch_size_is_per_gpu: batchIsPerGpu,
      context_events: d_contextEvents,
      dropout: d_dropout,
    };

    const optimizerSettings = {
      peak_lr: peakLr,
      min_lr: minLr,
      beta2: d_beta2,
      warmup_iters: d_warmupIters,
      max_iters_mode: d_maxItersMode,
      fixed_max_iters: d_fixedMaxIters,
      lr_decay_mode: d_lrDecayMode,
      fixed_lr_decay_iters: d_fixedLrDecayIters,
      lr_decay_fraction: d_lrDecayFraction,
    };

    const bookkeepingSettings = {
      eval_interval: d_evalInterval,
      eval_iters: d_evalIters,
      log_interval: d_logInterval,
      max_num_failed_checkpoint_checks: d_maxFailedCheckpointChecks,
      eval_every_epoch: evalEveryEpoch,
    };

    const samplingSettings = {
      temperature: d_temperature,
      top_k: d_topK,
      seed: d_samplingSeed,
      device: d_samplingDevice,
      compile: compileSampling,
    };

    const particleGPTExportSettings = {
      file_prefix: d_configFilePrefix,
      preparation_name: d_preparationName,
      dataset: d_datasetName,
      data_mode: d_dataMode,
      max_iters_mode: d_maxItersMode,
      fixed_max_iters: d_fixedMaxIters,
      lr_decay_mode: d_lrDecayMode,
      fixed_lr_decay_iters: d_fixedLrDecayIters,
      lr_decay_fraction: d_lrDecayFraction,
      include_designer_metadata: includeDesignerMetadata,
      training_config_defaults: {
        eval_interval: d_evalInterval,
        eval_iters: d_evalIters,
        log_interval: d_logInterval,
        gradient_accumulation_steps: d_gradAccum,
        batch_size: batchSeqConstant,
        context_events: d_contextEvents,
        n_layer: d_nLayer,
        n_head: d_nHead,
        n_kv_heads: d_nKvHeads,
        n_embd: d_nEmbd,
        dropout: d_dropout,
        learning_rate: peakLr,
        max_iters: d_fixedMaxIters,
        lr_decay_iters: d_fixedLrDecayIters,
        min_lr: minLr,
        beta2: d_beta2,
        warmup_iters: d_warmupIters,
        max_num_failed_checkpoint_checks: d_maxFailedCheckpointChecks,
        eval_every_epoch: evalEveryEpoch,
      },
      sampling_config_defaults: samplingSettings,
    };

    return {
      protocol: "particleGPT.designer_sweep",
      protocol_version: 1,
      route: "/designer",
      sweep: {
        mode: d_is2DSweep ? "2d" : "1d",
        x_variable: d_sweepVariable,
        x_label: d_sweepSpec.label,
        x_scale: d_sweepSpec.scale,
        x_min: xAxis?.min ?? null,
        x_max: xAxis?.max ?? null,
        x_points: d_sweepPoints,
        y_variable: d_is2DSweep ? d_effectiveSecondSweepVariable : null,
        y_label: d_is2DSweep ? d_secondSweepSpec.label : null,
        y_scale: d_is2DSweep ? d_secondSweepSpec.scale : null,
        y_min: d_is2DSweep ? yAxis?.min ?? null : null,
        y_max: d_is2DSweep ? yAxis?.max ?? null : null,
        y_points: d_is2DSweep ? d_secondSweepPoints : null,
        kaplan_plot_source: d_is2DSweep ? d_kaplanPlotSource : "1d_active_sweep",
        axes: {
          x: xAxis,
          y: yAxis,
        },
        available_ranges: allSweepRanges,
      },
      constants: {
        n_layer: d_nLayer,
        n_head: d_nHead,
        n_embd: d_nEmbd,
        mlp_ratio: d_mlpRatio,
        use_N_override: useNOverride,
        N_params_override: useNOverride ? NOverride : null,
        D_tokens: DConstant,
        seq_len: d_seqLen,
        batch_size_sequences: batchSeqConstant,
        grad_accum: d_gradAccum,
        num_gpus: d_worldSize,
        batch_size_is_per_gpu: batchIsPerGpu,
        peak_lr: peakLr,
        min_lr: minLr,
      },
      input_panel: {
        sweep: {
          mode: d_is2DSweep ? "2d" : "1d",
          x_axis: xAxis,
          y_axis: yAxis,
          kaplan_plot_source: d_is2DSweep ? d_kaplanPlotSource : "1d_active_sweep",
          all_ranges: allSweepRanges,
        },
        architecture: architectureSettings,
        dataset_and_training: datasetTrainingSettings,
        optimizer: optimizerSettings,
        bookkeeping: bookkeepingSettings,
        sampling: samplingSettings,
        particleGPT_config_export: particleGPTExportSettings,
      },
      particleGPT_config_export: particleGPTExportSettings,
    };
  }, [
    d_sweepRanges,
    d_sweepVariable,
    d_sweepSpec,
    d_sweepPoints,
    d_is2DSweep,
    d_effectiveSecondSweepVariable,
    d_secondSweepSpec,
    d_secondSweepPoints,
    d_kaplanPlotSource,
    d_nLayer,
    d_nHead,
    d_nKvHeads,
    d_nEmbd,
    d_mlpRatio,
    useNOverride,
    NOverride,
    architectureN,
    DConstant,
    d_seqLen,
    batchSeqConstant,
    d_gradAccum,
    d_worldSize,
    batchIsPerGpu,
    d_contextEvents,
    d_dropout,
    peakLr,
    minLr,
    d_beta2,
    d_warmupIters,
    d_maxItersMode,
    d_fixedMaxIters,
    d_lrDecayMode,
    d_fixedLrDecayIters,
    d_lrDecayFraction,
    d_evalInterval,
    d_evalIters,
    d_logInterval,
    d_maxFailedCheckpointChecks,
    evalEveryEpoch,
    d_temperature,
    d_topK,
    d_samplingSeed,
    d_samplingDevice,
    compileSampling,
    d_configFilePrefix,
    d_preparationName,
    d_datasetName,
    d_dataMode,
    includeDesignerMetadata,
  ]);

  const particleGPTExportRows = d_is2DSweep ? sweep2DGrid.rows : sweepData;

  const particleGPTConfigOptions = useMemo(() => ({
    filePrefix: d_configFilePrefix,
    preparationName: d_preparationName,
    datasetName: d_datasetName,
    dataMode: d_dataMode,
    contextEvents: d_contextEvents,
    nKvHeads: d_nKvHeads,
    dropout: d_dropout,
    evalInterval: d_evalInterval,
    evalIters: d_evalIters,
    logInterval: d_logInterval,
    maxItersMode: d_maxItersMode,
    fixedMaxIters: d_fixedMaxIters,
    lrDecayMode: d_lrDecayMode,
    fixedLrDecayIters: d_fixedLrDecayIters,
    lrDecayFraction: d_lrDecayFraction,
    minLr,
    beta2: d_beta2,
    warmupIters: d_warmupIters,
    maxFailedCheckpointChecks: d_maxFailedCheckpointChecks,
    evalEveryEpoch,
    temperature: d_temperature,
    topK: d_topK,
    seed: d_samplingSeed,
    device: d_samplingDevice,
    compileSampling,
    includeMetadata: includeDesignerMetadata,
    learningRate: peakLr,
    gradAccum: d_gradAccum,
    worldSize: d_worldSize,
    batchIsPerGpu,
  }), [
    d_configFilePrefix,
    d_preparationName,
    d_datasetName,
    d_dataMode,
    d_contextEvents,
    d_nKvHeads,
    d_dropout,
    d_evalInterval,
    d_evalIters,
    d_logInterval,
    d_maxItersMode,
    d_fixedMaxIters,
    d_lrDecayMode,
    d_fixedLrDecayIters,
    d_lrDecayFraction,
    minLr,
    d_beta2,
    d_warmupIters,
    d_maxFailedCheckpointChecks,
    evalEveryEpoch,
    d_temperature,
    d_topK,
    d_samplingSeed,
    d_samplingDevice,
    compileSampling,
    includeDesignerMetadata,
    peakLr,
    d_gradAccum,
    d_worldSize,
    batchIsPerGpu,
  ]);

  const particleGPTConfigFiles = useMemo(() => {
    return makeParticleGPTConfigFiles({
      rows: particleGPTExportRows,
      xVariable: d_sweepVariable,
      yVariable: d_is2DSweep ? d_effectiveSecondSweepVariable : null,
      is2D: d_is2DSweep,
      options: particleGPTConfigOptions,
    });
  }, [
    particleGPTExportRows,
    d_sweepVariable,
    d_effectiveSecondSweepVariable,
    d_is2DSweep,
    particleGPTConfigOptions,
  ]);

  const particleGPTZipFilename = `${sanitizeFilenamePart(d_configFilePrefix)}_${d_is2DSweep ? "2d" : "1d"}_configs.zip`;

  const particleGPTZipSummaryRows = useMemo(() => ([
    { label: "Files", value: human_num(particleGPTConfigFiles.length) },
    { label: "Format", value: "particleGPT JSON" },
    { label: "max_iters", value: d_maxItersMode === "fixed" ? human_num(d_fixedMaxIters) : "D / global batch" },
    { label: "lr_decay_iters", value: d_lrDecayMode === "fixed" ? human_num(d_fixedLrDecayIters) : `${Number(d_lrDecayFraction).toFixed(2)} × max_iters` },
    { label: "learning_rate", value: formatSci(peakLr, 2) },
    { label: "min_lr", value: formatSci(minLr, 2) },
  ]), [
    particleGPTConfigFiles.length,
    d_maxItersMode,
    d_fixedMaxIters,
    d_lrDecayMode,
    d_fixedLrDecayIters,
    d_lrDecayFraction,
    peakLr,
    minLr,
  ]);

  const headDim = safeDivide(d_nEmbd, d_nHead);
  const validHeads = Number.isInteger(headDim);
  const globalBSeqConstant = batchIsPerGpu
    ? batchSeqConstant * d_gradAccum * d_worldSize
    : batchSeqConstant * d_gradAccum;

  const sweepOptions = Object.entries(SWEEP_SPECS).map(([value, spec]) => ({
    value,
    label: spec.label,
  }));

  const makeFormatSweepSlider = (spec) => (value) => {
    if (spec.scale === "log") return human_num(logSliderToValue(value));
    if ((spec.step ?? 1) >= 1) return `${Math.round(value)}`;
    return Number(value).toFixed(2);
  };

  const makeSweepInputToSlider = (spec) => (value) => {
    if (spec.scale === "log") return valueToLogSlider(Math.max(value, 1e-300));
    return value;
  };

  const formatSweepSlider = makeFormatSweepSlider(sweepSpec);
  const sweepInputToSlider = makeSweepInputToSlider(sweepSpec);
  const formatSecondSweepSlider = makeFormatSweepSlider(secondSweepSpec);
  const secondSweepInputToSlider = makeSweepInputToSlider(secondSweepSpec);

  return (
    <MathJaxContext config={MATHJAX_CONFIG}>
      <div className="min-h-screen bg-zinc-950 text-zinc-100">
        <div ref={layoutRef} className="min-h-screen lg:flex lg:items-stretch">
          <aside
            className="border-b border-zinc-800 bg-zinc-950/95 p-5 lg:sticky lg:top-0 lg:h-screen lg:shrink-0 lg:overflow-y-auto lg:border-b-0 lg:border-r"
            style={{
              width: `min(100%, ${sidebarWidth}px)`,
              flexBasis: `${sidebarWidth}px`,
            }}
          >
            <div className="mb-6">
              <h1 className="text-2xl font-semibold tracking-tight">
                Scaling law designer
              </h1>
              <p className="mt-2 text-sm text-zinc-500">
                Pick a 1D or 2D sweep, keep the rest fixed, and inspect the expected scaling curves live.
              </p>
            </div>

            <div className="space-y-6">
              <SweepProtocolImportPanel onImport={applyImportedSweepProtocol} />

              <ControlSection title="Sweep variables">
                <SelectControl
                  label="Sweep mode"
                  value={sweepMode}
                  setValue={setSweepMode}
                  options={SWEEP_MODE_OPTIONS}
                  help="1D keeps the original behavior. 2D generates a grid over an x-axis and y-axis variable."
                />

                <SelectControl
                  label={sweepMode === "2d" ? "X-axis variable" : "Variable to sweep"}
                  value={sweepVariable}
                  setValue={setSweepVariable}
                  options={sweepOptions}
                  help="All other fields below are treated as constants unless they are selected sweep variables."
                />

                <SliderControl
                  label="Sweep minimum"
                  value={currentRange.min}
                  setValue={(v) => setCurrentRangeValue("min", v)}
                  min={sweepSpec.sliderMin}
                  max={sweepSpec.sliderMax}
                  step={sweepSpec.scale === "log" ? 0.01 : sweepSpec.step ?? 1}
                  format={formatSweepSlider}
                  inputValueToSliderValue={sweepInputToSlider}
                  sanitizeValue={(x) => sweepSpec.scale === "linear" ? normalizeSweepValue(sweepSpec, x) : x}
                />

                <SliderControl
                  label="Sweep maximum"
                  value={currentRange.max}
                  setValue={(v) => setCurrentRangeValue("max", v)}
                  min={sweepSpec.sliderMin}
                  max={sweepSpec.sliderMax}
                  step={sweepSpec.scale === "log" ? 0.01 : sweepSpec.step ?? 1}
                  format={formatSweepSlider}
                  inputValueToSliderValue={sweepInputToSlider}
                  sanitizeValue={(x) => sweepSpec.scale === "linear" ? normalizeSweepValue(sweepSpec, x) : x}
                />

                <SliderControl
                  label={sweepMode === "2d" ? "X sweep points" : "Sweep points"}
                  value={sweepPoints}
                  setValue={setSweepPoints}
                  min={4}
                  max={sweepMode === "2d" ? 32 : 128}
                  step={1}
                  format={(x) => `${Math.round(x)}`}
                  sanitizeValue={(x) => Math.round(x)}
                />

                {sweepMode === "2d" && (
                  <>
                    <SelectControl
                      label="Y-axis variable"
                      value={effectiveSecondSweepVariable}
                      setValue={setSecondSweepVariable}
                      options={secondSweepOptions}
                      help="Choose a second variable for the grid. It cannot be the same as the x-axis variable."
                    />

                    <SliderControl
                      label="Y sweep minimum"
                      value={currentSecondRange.min}
                      setValue={(v) => setSecondRangeValue("min", v)}
                      min={secondSweepSpec.sliderMin}
                      max={secondSweepSpec.sliderMax}
                      step={secondSweepSpec.scale === "log" ? 0.01 : secondSweepSpec.step ?? 1}
                      format={formatSecondSweepSlider}
                      inputValueToSliderValue={secondSweepInputToSlider}
                      sanitizeValue={(x) => secondSweepSpec.scale === "linear" ? normalizeSweepValue(secondSweepSpec, x) : x}
                    />

                    <SliderControl
                      label="Y sweep maximum"
                      value={currentSecondRange.max}
                      setValue={(v) => setSecondRangeValue("max", v)}
                      min={secondSweepSpec.sliderMin}
                      max={secondSweepSpec.sliderMax}
                      step={secondSweepSpec.scale === "log" ? 0.01 : secondSweepSpec.step ?? 1}
                      format={formatSecondSweepSlider}
                      inputValueToSliderValue={secondSweepInputToSlider}
                      sanitizeValue={(x) => secondSweepSpec.scale === "linear" ? normalizeSweepValue(secondSweepSpec, x) : x}
                    />

                    <SliderControl
                      label="Y sweep points"
                      value={secondSweepPoints}
                      setValue={setSecondSweepPoints}
                      min={4}
                      max={32}
                      step={1}
                      format={(x) => `${Math.round(x)}`}
                      sanitizeValue={(x) => Math.round(x)}
                    />

                    <SelectControl
                      label="Kaplan-style plot source"
                      value={kaplanPlotSource}
                      setValue={setKaplanPlotSource}
                      options={KAPLAN_PLOT_SOURCE_OPTIONS}
                      help="Choose the data source for the Kaplan-style panels. Full matrix mode uses all grid points to select Kaplan-style edge slices: L(D) at largest N and L(N) at largest D."
                    />
                  </>
                )}
              </ControlSection>

              <ControlSection title="Architecture constants">
                {sweptVariableSet.has("n_layer") ? (
                  <FrozenControlNotice
                    label="Layers are being swept"
                    help="The constant layer value is ignored for this sweep."
                  />
                ) : (
                  <SliderControl
                    label="Number of layers"
                    value={nLayer}
                    setValue={setNLayer}
                    min={1}
                    max={192}
                    step={1}
                    format={(x) => `${Math.round(x)}`}
                    sanitizeValue={(x) => Math.round(x)}
                  />
                )}

                {sweptVariableSet.has("n_head") ? (
                  <FrozenControlNotice
                    label="Heads are being swept"
                    help="The constant head count is ignored for this sweep."
                  />
                ) : (
                  <SliderControl
                    label="Number of attention heads"
                    value={nHead}
                    setValue={setNHead}
                    min={1}
                    max={128}
                    step={1}
                    format={(x) => `${Math.round(x)}`}
                    sanitizeValue={(x) => Math.round(x)}
                    help={!validHeads ? "Warning: d_model is not divisible by H." : " "}
                  />
                )}

                {sweptVariableSet.has("n_embd") ? (
                  <FrozenControlNotice
                    label="Width is being swept"
                    help="The constant d_model value is ignored for this sweep."
                  />
                ) : (
                  <SliderControl
                    label="Embedding dimension"
                    value={nEmbd}
                    setValue={setNEmbd}
                    min={64}
                    max={4096}
                    step={64}
                    format={(x) => `${Math.round(x)}`}
                    sanitizeValue={(x) => Math.round(x)}
                  />
                )}

                <SliderControl
                  label="MLP expansion ratio"
                  value={mlpRatio}
                  setValue={setMlpRatio}
                  min={1}
                  max={8}
                  step={0.25}
                  format={(x) => `${Number(x).toFixed(2)}`}
                />

                {sweptVariableSet.has("N_params") ? (
                  <FrozenControlNotice
                    label="N is being swept directly"
                    help="The direct parameter-count sweep overrides the architecture-derived N curve."
                  />
                ) : (
                  <ToggleControl
                    label="Override the parameter count"
                    checked={useNOverride}
                    setChecked={setUseNOverride}
                    help="Use a direct non-embedding parameter count instead of the GPT-like estimate. Disabled automatically when sweeping L or d_model."
                  />
                )}

                {useNOverride && !["N_params", "n_layer", "n_embd"].some((v) => sweptVariableSet.has(v)) && (
                  <SliderControl
                    label="Non-embedding parameter override"
                    value={logNOverride}
                    setValue={setLogNOverride}
                    min={valueToLogSlider(1e5)}
                    max={valueToLogSlider(5e9)}
                    step={0.01}
                    format={(x) => human_num(logSliderToValue(x))}
                    inputValueToSliderValue={(x) => valueToLogSlider(x)}
                  />
                )}
              </ControlSection>

              <ControlSection title="Dataset and training constants">
                {sweptVariableSet.has("D_tokens") ? (
                  <FrozenControlNotice
                    label="Dataset size is being swept"
                    help="The constant D value is ignored for this sweep."
                  />
                ) : (
                  <SliderControl
                    label={
                      <>
                        Dataset size <MathText>{"\\(D\\)"}</MathText>
                      </>
                    }
                    value={logD}
                    setValue={setLogD}
                    min={valueToLogSlider(1e5)}
                    max={valueToLogSlider(5e12)}
                    step={0.01}
                    format={(x) => human_num(logSliderToValue(x))}
                    inputValueToSliderValue={(x) => valueToLogSlider(x)}
                  />
                )}

                <SliderControl
                  label="Sequence length"
                  value={seqLen}
                  setValue={setSeqLen}
                  min={128}
                  max={8192}
                  step={128}
                  format={(x) => `${Math.round(x)}`}
                  sanitizeValue={(x) => Math.round(x)}
                />

                {sweptVariableSet.has("batch_size_sequences") ? (
                  <FrozenControlNotice
                    label="Micro-batch size is being swept"
                    help="The constant B_seq value is ignored for this sweep."
                  />
                ) : (
                  <SliderControl
                    label={
                      <>
                        Batch size <MathText>{"\\(B_{\\rm seq}\\)"}</MathText>
                      </>
                    }
                    value={logBatchSeq}
                    setValue={setLogBatchSeq}
                    min={valueToLogSlider(1)}
                    max={valueToLogSlider(8192)}
                    step={0.01}
                    format={(x) => human_num(Math.round(logSliderToValue(x)))}
                    inputValueToSliderValue={(x) => valueToLogSlider(x)}
                  />
                )}

                <SliderControl
                  label="Gradient accumulation steps"
                  value={gradAccum}
                  setValue={setGradAccum}
                  min={1}
                  max={512}
                  step={1}
                  format={(x) => `${Math.round(x)}`}
                  sanitizeValue={(x) => Math.round(x)}
                />

                <SliderControl
                  label="World size / GPUs"
                  value={worldSize}
                  setValue={setWorldSize}
                  min={1}
                  max={64}
                  step={1}
                  format={(x) => `${Math.round(x)}`}
                  sanitizeValue={(x) => Math.round(x)}
                />

                <ToggleControl
                  label="Batch size is per GPU"
                  checked={batchIsPerGpu}
                  setChecked={setBatchIsPerGpu}
                  help={
                    <>
                      If enabled, global batch is{" "}
                      <MathText>{"\\(B_{\\rm global}=B_{\\rm local}GS\\)"}</MathText>.
                    </>
                  }
                />
              </ControlSection>

              <ControlSection title="Optimizer constant">
                <SliderControl
                  label={
                    <>
                      Peak learning rate <MathText>{"\\(\\eta\\)"}</MathText>
                    </>
                  }
                  value={logPeakLr}
                  setValue={setLogPeakLr}
                  min={valueToLogSlider(1e-8)}
                  max={valueToLogSlider(1e-2)}
                  step={0.01}
                  format={(x) => formatSci(logSliderToValue(x), 2)}
                  inputValueToSliderValue={(x) => valueToLogSlider(x)}
                />
              </ControlSection>

              <ControlSection title="particleGPT config ZIP">
                <TextControl
                  label="Config file prefix"
                  value={configFilePrefix}
                  setValue={setConfigFilePrefix}
                  help="Downloaded files use this prefix plus a zero-padded sweep index."
                />

                <TextControl
                  label="preparation_name"
                  value={preparationName}
                  setValue={setPreparationName}
                />

                <TextControl
                  label="dataset"
                  value={datasetName}
                  setValue={setDatasetName}
                  help="For D sweeps, change this to the prepared dataset corresponding to each run, or use the metadata as a manifest."
                />

                <TextControl
                  label="data_mode"
                  value={dataMode}
                  setValue={setDataMode}
                />

                <SliderControl
                  label="context_events"
                  value={contextEvents}
                  setValue={setContextEvents}
                  min={1}
                  max={16}
                  step={1}
                  format={(x) => `${Math.round(x)}`}
                  sanitizeValue={(x) => Math.round(x)}
                />

                <SliderControl
                  label="n_kv_heads"
                  value={nKvHeads}
                  setValue={setNKvHeads}
                  min={1}
                  max={64}
                  step={1}
                  format={(x) => `${Math.round(x)}`}
                  sanitizeValue={(x) => Math.round(x)}
                  help="Generated configs automatically lower this if needed so n_kv_heads divides n_head."
                />

                <SliderControl
                  label="dropout"
                  value={dropout}
                  setValue={setDropout}
                  min={0}
                  max={0.5}
                  step={0.01}
                  format={(x) => Number(x).toFixed(2)}
                />

                <SelectControl
                  label="max_iters"
                  value={maxItersMode}
                  setValue={setMaxItersMode}
                  options={MAX_ITERS_MODE_OPTIONS}
                  help="Computed mode sets max_iters = ceil(D / global_batch_tokens) for each generated run."
                />

                {maxItersMode === "fixed" && (
                  <SliderControl
                    label="fixed max_iters"
                    value={fixedMaxIters}
                    setValue={setFixedMaxIters}
                    min={1}
                    max={200000}
                    step={1}
                    format={(x) => `${Math.round(x)}`}
                    sanitizeValue={(x) => Math.round(x)}
                  />
                )}

                <SelectControl
                  label="lr_decay_iters"
                  value={lrDecayMode}
                  setValue={setLrDecayMode}
                  options={LR_DECAY_MODE_OPTIONS}
                />

                {lrDecayMode === "fixed" ? (
                  <SliderControl
                    label="fixed lr_decay_iters"
                    value={fixedLrDecayIters}
                    setValue={setFixedLrDecayIters}
                    min={1}
                    max={200000}
                    step={1}
                    format={(x) => `${Math.round(x)}`}
                    sanitizeValue={(x) => Math.round(x)}
                  />
                ) : (
                  <SliderControl
                    label="decay fraction"
                    value={lrDecayFraction}
                    setValue={setLrDecayFraction}
                    min={0.05}
                    max={1.5}
                    step={0.05}
                    format={(x) => Number(x).toFixed(2)}
                  />
                )}

                <SliderControl
                  label="min_lr"
                  value={logMinLr}
                  setValue={setLogMinLr}
                  min={valueToLogSlider(1e-8)}
                  max={valueToLogSlider(1e-2)}
                  step={0.01}
                  format={(x) => formatSci(logSliderToValue(x), 2)}
                  inputValueToSliderValue={(x) => valueToLogSlider(x)}
                />

                <SliderControl
                  label="beta2"
                  value={beta2}
                  setValue={setBeta2}
                  min={0.8}
                  max={0.9999}
                  step={0.0005}
                  format={(x) => Number(x).toFixed(4)}
                />

                <SliderControl
                  label="warmup_iters"
                  value={warmupIters}
                  setValue={setWarmupIters}
                  min={0}
                  max={5000}
                  step={1}
                  format={(x) => `${Math.round(x)}`}
                  sanitizeValue={(x) => Math.round(x)}
                />

                <SliderControl
                  label="eval_interval"
                  value={evalInterval}
                  setValue={setEvalInterval}
                  min={1}
                  max={5000}
                  step={1}
                  format={(x) => `${Math.round(x)}`}
                  sanitizeValue={(x) => Math.round(x)}
                />

                <SliderControl
                  label="eval_iters"
                  value={evalIters}
                  setValue={setEvalIters}
                  min={1}
                  max={2000}
                  step={1}
                  format={(x) => `${Math.round(x)}`}
                  sanitizeValue={(x) => Math.round(x)}
                />

                <SliderControl
                  label="log_interval"
                  value={logInterval}
                  setValue={setLogInterval}
                  min={1}
                  max={1000}
                  step={1}
                  format={(x) => `${Math.round(x)}`}
                  sanitizeValue={(x) => Math.round(x)}
                />

                <SliderControl
                  label="temperature"
                  value={temperature}
                  setValue={setTemperature}
                  min={0.05}
                  max={2}
                  step={0.05}
                  format={(x) => Number(x).toFixed(2)}
                />

                <SliderControl
                  label="top_k"
                  value={topK}
                  setValue={setTopK}
                  min={1}
                  max={1000}
                  step={1}
                  format={(x) => `${Math.round(x)}`}
                  sanitizeValue={(x) => Math.round(x)}
                />

                <SliderControl
                  label="sampling seed"
                  value={samplingSeed}
                  setValue={setSamplingSeed}
                  min={0}
                  max={999999}
                  step={1}
                  format={(x) => `${Math.round(x)}`}
                  sanitizeValue={(x) => Math.round(x)}
                />

                <TextControl
                  label="sampling device"
                  value={samplingDevice}
                  setValue={setSamplingDevice}
                />

                <SliderControl
                  label="failed checkpoint checks"
                  value={maxFailedCheckpointChecks}
                  setValue={setMaxFailedCheckpointChecks}
                  min={-1}
                  max={100}
                  step={1}
                  format={(x) => `${Math.round(x)}`}
                  sanitizeValue={(x) => Math.round(x)}
                />

                <ToggleControl
                  label="eval_every_epoch"
                  checked={evalEveryEpoch}
                  setChecked={setEvalEveryEpoch}
                />

                <ToggleControl
                  label="sampling compile"
                  checked={compileSampling}
                  setChecked={setCompileSampling}
                />

                <ToggleControl
                  label="include designer_metadata"
                  checked={includeDesignerMetadata}
                  setChecked={setIncludeDesignerMetadata}
                  help="Recommended. Stores D_tokens, estimated N, predicted loss, and sweep coordinates without changing training_config."
                />
              </ControlSection>
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
            onDoubleClick={() => setSidebarWidth(460)}
            className="hidden w-1 shrink-0 cursor-col-resize bg-zinc-900 transition hover:bg-blue-500/40 active:bg-blue-500/60 lg:block"
            style={{ touchAction: "none" }}
          />

          <main className="min-w-0 flex-1 space-y-3 p-4 lg:p-5">
            <div className="flex flex-col gap-2 sm:flex-row sm:items-end sm:justify-between">
              <div>
                <h1 className="text-3xl font-semibold tracking-tight">
                  Sweep designer
                </h1>
                <p className="mt-2 text-sm text-zinc-500">
                  <span>{d_is2DSweep ? "Sweeping grid over " : "Sweeping "}</span>
                  <span className="font-medium text-blue-300">{d_sweepSpec.label}</span>
                  {d_is2DSweep && (
                    <>
                      <span> × </span>
                      <span className="font-medium text-blue-300">{d_secondSweepSpec.label}</span>
                    </>
                  )}
                  <span> with all remaining quantities held fixed.</span>
                </p>
              </div>

              {!validHeads && !["n_head", "n_embd"].some((v) => sweptVariableSet.has(v)) && (
                <div className="rounded-xl border border-amber-500/30 bg-amber-500/10 px-4 py-2 text-sm text-amber-200">
                  The embedding dimension is not divisible by the number of heads.
                </div>
              )}
            </div>

            <section className="grid grid-cols-1 gap-2 md:grid-cols-2 xl:grid-cols-4">
              <StatCard
                label={<QuantityName name="sweep_variable" />}
                value={d_is2DSweep ? `${d_sweepSpec.mathLabel} × ${d_secondSweepSpec.mathLabel}` : d_sweepSpec.mathLabel}
                sub={d_is2DSweep ? `${sweep2DGrid.rows.length} generated grid points` : `${sweepData.length} generated points`}
              />
              <StatCard
                label={<QuantityName name="N_params" />}
                value={human_num(midpoint.N)}
                sub="midpoint design"
              />
              <StatCard
                label={<QuantityName name="D_tokens" />}
                value={human_num(midpoint.D)}
                sub={`D/N = ${human_num(midpoint.D_over_N)}`}
              />
              <StatCard
                label={<QuantityName name="batch" />}
                value={human_num(midpoint.Bactual)}
                sub={`${human_num(midpoint.steps)} optimizer steps at midpoint`}
              />
            </section>

            <RegimeBar summary={midpoint} />

            {d_is2DSweep && (
              <TwoDSweepHeatmaps
                grid={sweep2DGrid}
                xSpec={d_sweepSpec}
                ySpec={d_secondSweepSpec}
              />
            )}

            <KaplanPaperPanels panels={kaplanPanelData} />

            <section className="grid grid-cols-1 gap-3 xl:grid-cols-2">
              <ExpectedLossChart data={sweepData} sweepSpec={d_sweepSpec} />
              <ComputeChart data={sweepData} sweepSpec={d_sweepSpec} />
            </section>

            <section className="grid grid-cols-1 gap-3 xl:grid-cols-2">
              <RatioChart data={sweepData} sweepSpec={d_sweepSpec} />
              <ArchitectureChart data={sweepData} sweepSpec={d_sweepSpec} />
            </section>

            <BatchEnvelopeChart data={sweepData} sweepSpec={d_sweepSpec} />

            <section className="grid grid-cols-1 gap-3 xl:grid-cols-2">
              <DataTable title="Designer recommendations" rows={recommendationRows} />
              <ConfigZipPanel
                files={particleGPTConfigFiles}
                zipFilename={particleGPTZipFilename}
                summaryRows={particleGPTZipSummaryRows}
              />
            </section>

            <JsonBlock value={exportConfig} />

            {d_is2DSweep && (
              <DataTable title="Sampled 2D run matrix" rows={sampled2DRunRows} />
            )}

            <DataTable
              title={d_is2DSweep ? "Midpoint y-slice run matrix" : "Sampled run matrix"}
              rows={sampledRunRows}
            />

            <div className="rounded-2xl border border-zinc-800 bg-zinc-950/80 p-3 text-xs leading-relaxed text-zinc-500">
              <div className="font-semibold uppercase tracking-wide text-zinc-400">Notes</div>
              <p className="mt-1">
                This page is intended as a planning view. The curves are scaling-law expectations from the current configuration,
                not measurements from completed runs. The most useful first pass is usually to sweep one of <MathText>{"\\(D\\)"}</MathText>,{" "}
                <MathText>{"\\(N\\)"}</MathText>, or <MathText>{"\\(d_{\\rm model}\\)"}</MathText>, then choose a run matrix near the bend of
                the predicted loss/compute tradeoff and near <MathText>{"\\(D/N\\approx20\\)"}</MathText>.
              </p>
              <p className="mt-1">
                Constant global batch at the current settings is {human_num(globalBSeqConstant)} sequences. The particleGPT ZIP generator
                writes one JSON config per active sweep point; D_tokens and direct N_params sweeps are included in designer_metadata because
                particleGPT training_config does not directly expose those as architecture fields.
              </p>
            </div>
          </main>
        </div>
      </div>
    </MathJaxContext>
  );
}
