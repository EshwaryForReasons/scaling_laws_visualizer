"use client";

import { useEffect, useMemo, useRef, useState } from "react";
import { MathJax, MathJaxContext } from "better-react-mathjax";
import {
  LineChart,
  Line,
  AreaChart,
  Area,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
  CartesianGrid,
  ReferenceLine,
  Legend,
} from "recharts";

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
    const key = value.toPrecision(12);
    if (seen.has(key)) continue;
    seen.add(key);
    unique.push(value);
  }

  return unique.sort((a, b) => a - b);
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

function KaplanFigure1Panel({ title, subtitle, data, xKey, xLabel, equation, xDomain, currentX }) {
  return (
    <div className="rounded-xl border border-zinc-800 bg-zinc-950/70 p-3">
      <div className="mb-2 min-h-14">
        <h3 className="text-sm font-semibold text-zinc-100">{title}</h3>
        <p className="mt-1 text-[11px] text-zinc-500">{subtitle}</p>
      </div>

      <div className="h-56">
        <ResponsiveContainer width="100%" height="100%">
          <LineChart data={data} margin={{ top: 8, right: 10, bottom: 8, left: 6 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.08)" />
            <XAxis
              type="number"
              dataKey={xKey}
              scale="log"
              domain={xDomain}
              allowDataOverflow
              stroke="#71717a"
              tick={{ fontSize: 11 }}
              tickFormatter={(x) => human_num(x)}
              label={{
                value: xLabel,
                position: "insideBottom",
                offset: -2,
                fill: "#a1a1aa",
                fontSize: 11,
              }}
            />
            <YAxis
              type="number"
              domain={["dataMin", "dataMax"]}
              stroke="#71717a"
              tick={{ fontSize: 11 }}
              tickFormatter={(x) => Number(x).toFixed(3)}
              label={{
                value: "Predicted loss",
                angle: -90,
                position: "insideLeft",
                fill: "#a1a1aa",
                fontSize: 11,
              }}
            />
            <BaseTooltip
              labelFormatter={(x) => `${xLabel} = ${human_num(x)}`}
              formatter={(v, name) => [Number(v).toFixed(5), name]}
            />
            {Number.isFinite(currentX) && currentX > 0 && (
              <ReferenceLine
                x={currentX}
                stroke="#fbbf24"
                strokeDasharray="6 4"
                ifOverflow="extendDomain"
              />
            )}
            <Line
              type="linear"
              dataKey="loss"
              name="your config sweep"
              dot={{ r: 3, strokeWidth: 1, fill: "#60a5fa", stroke: "#bfdbfe" }}
              activeDot={{ r: 5 }}
              stroke="#60a5fa"
              strokeWidth={2}
              strokeDasharray="3 3"
              isAnimationActive={false}
            />
          </LineChart>
        </ResponsiveContainer>
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
            These are not hard-coded paper points. They are generated from your current D range, N range, and fixed constants.
          </p>
        </div>
        <div className="text-[11px] text-zinc-500">dots are log-spaced generated runs; dashed lines connect them</div>
      </div>

      <div className="grid grid-cols-1 gap-3 xl:grid-cols-3">
        <KaplanFigure1Panel
          title="Compute"
          subtitle="Paired log-spaced N and D from your configured ranges"
          data={computeData}
          xKey="Cmin"
          xLabel="Compute, PF-days"
          xDomain={panels.ranges.compute}
          currentX={panels.figure1.currentCompute}
          equation={"\\(C\\approx6ND,\\quad L=L(N,D)\\)"}
        />
        <KaplanFigure1Panel
          title="Dataset size"
          subtitle={`Holds N fixed at ${formatKaplanLegendValue(panels.baseN)} and sweeps your D range`}
          data={datasetData}
          xKey="D"
          xLabel="Dataset size D"
          xDomain={[panels.ranges.D[0] / 1.08, panels.ranges.D[1] * 1.08]}
          currentX={panels.baseD}
          equation={"\\(L=L(N_{\\rm fixed},D)\\)"}
        />
        <KaplanFigure1Panel
          title="Parameters"
          subtitle={`Holds D fixed at ${formatKaplanLegendValue(panels.baseD)} tokens and sweeps your N range`}
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
  return (
    <ChartShell title={title} subtitle={subtitle} heightClass="h-80">
      <ResponsiveContainer width="100%" height="100%">
        <LineChart data={data} margin={{ top: 8, right: 18, bottom: 8, left: 8 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.08)" />
          <XAxis
            type="number"
            dataKey={xKey}
            scale="log"
            domain={xDomain}
            allowDataOverflow
            stroke="#71717a"
            tick={{ fontSize: 12 }}
            tickFormatter={(x) => human_num(x)}
            label={{
              value: xLabel,
              position: "insideBottom",
              offset: -2,
              fill: "#a1a1aa",
              fontSize: 12,
            }}
          />
          <YAxis
            type="number"
            domain={yDomain ?? ["dataMin", "dataMax"]}
            stroke="#71717a"
            tick={{ fontSize: 12 }}
            tickFormatter={(x) => Number(x).toFixed(3)}
            label={{
              value: yLabel,
              angle: -90,
              position: "insideLeft",
              fill: "#a1a1aa",
              fontSize: 12,
            }}
          />
          <BaseTooltip
            labelFormatter={(x) => `${xLabel} = ${human_num(x)}`}
            formatter={(v, name) => [Number(v).toFixed(5), name]}
          />
          <Legend wrapperStyle={{ fontSize: 11 }} />
          {Number.isFinite(referenceX) && referenceX > 0 && (
            <ReferenceLine
              x={referenceX}
              stroke="#fbbf24"
              strokeDasharray="6 4"
              ifOverflow="extendDomain"
            />
          )}
          {series.map((s, index) => (
            <Line
              key={s.key}
              type="linear"
              dataKey={s.key}
              name={s.name}
              dot={{ r: 3, strokeWidth: 1, fill: s.color ?? KAPLAN_COLORS[index % KAPLAN_COLORS.length] }}
              activeDot={{ r: 5 }}
              stroke={s.color ?? KAPLAN_COLORS[index % KAPLAN_COLORS.length]}
              strokeWidth={s.strokeWidth ?? 1.8}
              strokeDasharray="2 2"
              isAnimationActive={false}
            />
          ))}
        </LineChart>
      </ResponsiveContainer>
    </ChartShell>
  );
}

function KaplanFigure4LeftChart({ panels }) {
  return (
    <KaplanMultiSeriesChart
      title="Kaplan Figure 4 left style: loss vs dataset size"
      subtitle="Each dashed line holds N fixed. The N values come from your configured parameter-count range; the highlighted line is your current N when it lies inside that range."
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
      subtitle="Each dashed line holds D fixed. The D values come from your configured dataset-size range; the highlighted line is your current D when it lies inside that range."
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

function ExpectedLossChart({ data, sweepSpec }) {
  const isLog = sweepSpec.scale === "log";

  return (
    <ChartShell
      title="Expected Kaplan loss over the sweep"
      subtitle="Uses L(N,D), holding non-swept quantities fixed."
    >
      <ResponsiveContainer width="100%" height="100%">
        <LineChart data={data}>
          <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.08)" />
          <XAxis
            type="number"
            dataKey="x"
            scale={isLog ? "log" : "linear"}
            domain={["dataMin", "dataMax"]}
            allowDataOverflow
            stroke="#71717a"
            tick={{ fontSize: 12 }}
            tickFormatter={(x) => human_num(x)}
          />
          <YAxis
            stroke="#71717a"
            tick={{ fontSize: 12 }}
            tickFormatter={(x) => Number(x).toFixed(3)}
            label={{
              value: "loss",
              angle: -90,
              position: "insideLeft",
              fill: "#a1a1aa",
              fontSize: 12,
            }}
          />
          <BaseTooltip
            labelFormatter={(x) => `${sweepSpec.label} = ${human_num(x)}`}
            formatter={(v) => Number(v).toFixed(5)}
          />
          <Line
            type="monotone"
            dataKey="loss"
            name="Kaplan loss"
            dot={false}
            stroke="#60a5fa"
            strokeWidth={2}
            isAnimationActive={false}
          />
        </LineChart>
      </ResponsiveContainer>
    </ChartShell>
  );
}

function ComputeChart({ data, sweepSpec }) {
  const isLog = sweepSpec.scale === "log";

  return (
    <ChartShell
      title="Training compute over the sweep"
      subtitle="Uses the usual dense-transformer estimate C ≈ 6ND."
    >
      <ResponsiveContainer width="100%" height="100%">
        <LineChart data={data}>
          <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.08)" />
          <XAxis
            type="number"
            dataKey="x"
            scale={isLog ? "log" : "linear"}
            domain={["dataMin", "dataMax"]}
            allowDataOverflow
            stroke="#71717a"
            tick={{ fontSize: 12 }}
            tickFormatter={(x) => human_num(x)}
          />
          <YAxis
            stroke="#71717a"
            tick={{ fontSize: 12 }}
            tickFormatter={(x) => human_num(x)}
            label={{
              value: "PF-days",
              angle: -90,
              position: "insideLeft",
              fill: "#a1a1aa",
              fontSize: 12,
            }}
          />
          <BaseTooltip
            labelFormatter={(x) => `${sweepSpec.label} = ${human_num(x)}`}
            formatter={(v) => `${Number(v).toFixed(4)} PF-days`}
          />
          <Line
            type="monotone"
            dataKey="pf_days"
            name="PF-days"
            dot={false}
            stroke="#34d399"
            strokeWidth={2}
            isAnimationActive={false}
          />
        </LineChart>
      </ResponsiveContainer>
    </ChartShell>
  );
}

function RatioChart({ data, sweepSpec }) {
  const isLog = sweepSpec.scale === "log";

  return (
    <ChartShell
      title="Data/model balance over the sweep"
      subtitle="The horizontal reference is the rough Chinchilla / Power-Lines-style D/N ≈ 20 heuristic."
    >
      <ResponsiveContainer width="100%" height="100%">
        <LineChart data={data}>
          <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.08)" />
          <XAxis
            type="number"
            dataKey="x"
            scale={isLog ? "log" : "linear"}
            domain={["dataMin", "dataMax"]}
            allowDataOverflow
            stroke="#71717a"
            tick={{ fontSize: 12 }}
            tickFormatter={(x) => human_num(x)}
          />
          <YAxis
            stroke="#71717a"
            tick={{ fontSize: 12 }}
            tickFormatter={(x) => human_num(x)}
            label={{
              value: "D/N",
              angle: -90,
              position: "insideLeft",
              fill: "#a1a1aa",
              fontSize: 12,
            }}
          />
          <BaseTooltip
            labelFormatter={(x) => `${sweepSpec.label} = ${human_num(x)}`}
            formatter={(v, name) => [human_num(v), name]}
          />
          <ReferenceLine y={20} stroke="#fbbf24" strokeDasharray="6 4" />
          <Line
            type="monotone"
            dataKey="D_over_N"
            name="D/N"
            dot={false}
            stroke="#60a5fa"
            strokeWidth={2}
            isAnimationActive={false}
          />
        </LineChart>
      </ResponsiveContainer>
    </ChartShell>
  );
}

function BatchEnvelopeChart({ data, sweepSpec }) {
  const isLog = sweepSpec.scale === "log";

  return (
    <ChartShell
      title="Power Lines batch envelope over the sweep"
      subtitle="Compares your actual global batch to Bopt, geometric target, and Bcrit."
      footer={
        <span>
          <MathText>{"\\(B_{\\rm target}=\\sqrt{B_{\\rm opt}B_{\\rm crit}}\\)"}</MathText>{" "}
          is a practical middle choice between compute-efficient and critical batch sizes.
        </span>
      }
    >
      <ResponsiveContainer width="100%" height="100%">
        <AreaChart data={data}>
          <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.08)" />
          <XAxis
            type="number"
            dataKey="x"
            scale={isLog ? "log" : "linear"}
            domain={["dataMin", "dataMax"]}
            allowDataOverflow
            stroke="#71717a"
            tick={{ fontSize: 12 }}
            tickFormatter={(x) => human_num(x)}
          />
          <YAxis
            stroke="#71717a"
            tick={{ fontSize: 12 }}
            tickFormatter={(x) => human_num(x)}
          />
          <BaseTooltip
            labelFormatter={(x) => `${sweepSpec.label} = ${human_num(x)}`}
            formatter={(v, name) => [human_num(v), name]}
          />
          <Legend wrapperStyle={{ fontSize: 12 }} />
          <Area
            type="monotone"
            dataKey="Bcrit"
            name="Bcrit"
            stroke="#818cf8"
            fill="#818cf8"
            fillOpacity={0.12}
            strokeWidth={2}
            isAnimationActive={false}
          />
          <Line
            type="monotone"
            dataKey="Bopt"
            name="Bopt"
            dot={false}
            stroke="#60a5fa"
            strokeWidth={2}
            isAnimationActive={false}
          />
          <Line
            type="monotone"
            dataKey="Btarget"
            name="Btarget"
            dot={false}
            stroke="#fbbf24"
            strokeWidth={2}
            strokeDasharray="6 4"
            isAnimationActive={false}
          />
          <Line
            type="monotone"
            dataKey="Bactual"
            name="Bactual"
            dot={false}
            stroke="#fb7185"
            strokeWidth={2}
            strokeDasharray="2 4"
            isAnimationActive={false}
          />
        </AreaChart>
      </ResponsiveContainer>
    </ChartShell>
  );
}

function ArchitectureChart({ data, sweepSpec }) {
  const isLog = sweepSpec.scale === "log";

  return (
    <ChartShell
      title="Implied architecture and model size"
      subtitle="Useful when the sweep variable is a structural hyperparameter."
    >
      <ResponsiveContainer width="100%" height="100%">
        <LineChart data={data}>
          <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.08)" />
          <XAxis
            type="number"
            dataKey="x"
            scale={isLog ? "log" : "linear"}
            domain={["dataMin", "dataMax"]}
            allowDataOverflow
            stroke="#71717a"
            tick={{ fontSize: 12 }}
            tickFormatter={(x) => human_num(x)}
          />
          <YAxis
            yAxisId="left"
            stroke="#71717a"
            tick={{ fontSize: 12 }}
            tickFormatter={(x) => human_num(x)}
          />
          <YAxis
            yAxisId="right"
            orientation="right"
            stroke="#71717a"
            tick={{ fontSize: 12 }}
            tickFormatter={(x) => human_num(x)}
          />
          <BaseTooltip
            labelFormatter={(x) => `${sweepSpec.label} = ${human_num(x)}`}
            formatter={(v, name) => [human_num(v), name]}
          />
          <Legend wrapperStyle={{ fontSize: 12 }} />
          <Line
            yAxisId="left"
            type="monotone"
            dataKey="N"
            name="N params"
            dot={false}
            stroke="#60a5fa"
            strokeWidth={2}
            isAnimationActive={false}
          />
          <Line
            yAxisId="right"
            type="monotone"
            dataKey="head_dim"
            name="head dim"
            dot={false}
            stroke="#fbbf24"
            strokeWidth={2}
            strokeDasharray="6 4"
            isAnimationActive={false}
          />
        </LineChart>
      </ResponsiveContainer>
    </ChartShell>
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

function JsonBlock({ value }) {
  return (
    <div className="rounded-2xl border border-zinc-800 bg-zinc-950/80 p-3">
      <h2 className="mb-2 text-base font-semibold text-zinc-100">Exportable sweep config</h2>
      <pre className="max-h-80 overflow-auto rounded-xl border border-zinc-800 bg-zinc-950 p-3 text-xs leading-relaxed text-zinc-300">
        {JSON.stringify(value, null, 2)}
      </pre>
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

  const [sweepVariable, setSweepVariable] = useState("D_tokens");
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

  const d_sweepVariable = useDebouncedValue(sweepVariable);
  const d_sweepRanges = useDebouncedValue(sweepRanges);
  const d_sweepPoints = useDebouncedValue(sweepPoints);
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

  const sweepSpec = SWEEP_SPECS[sweepVariable];
  const d_sweepSpec = SWEEP_SPECS[d_sweepVariable];
  const currentRange = sweepRanges[sweepVariable];

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

  const setCurrentRangeValue = (which, value) => {
    setSweepRanges((prev) => ({
      ...prev,
      [sweepVariable]: {
        ...prev[sweepVariable],
        [which]: value,
      },
    }));
  };

  const DConstant = logSliderToValue(d_logD);
  const batchSeqConstant = Math.round(logSliderToValue(d_logBatchSeq));
  const NOverride = logSliderToValue(d_logNOverride);
  const peakLr = logSliderToValue(d_logPeakLr);

  const architectureN = useMemo(() => {
    return gpt_like_nonembedding_params(
      d_nLayer,
      d_nEmbd,
      d_mlpRatio,
      false
    );
  }, [d_nLayer, d_nEmbd, d_mlpRatio]);

  const baseN = useNOverride ? NOverride : architectureN;

  const sweepData = useMemo(() => {
    const spec = d_sweepSpec;
    const range = d_sweepRanges[d_sweepVariable] ?? {
      min: spec.defaultMin,
      max: spec.defaultMax,
    };

    const [rangeMin, rangeMax] =
      spec.scale === "log"
        ? sanitizePositiveLogRange(range.min, range.max)
        : range.min <= range.max
          ? [range.min, range.max]
          : [range.max, range.min];

    const values = makeSweepValues(spec, rangeMin, rangeMax, d_sweepPoints);

    return values.map((x) => {
      const cfg = {
        n_layer: d_nLayer,
        n_head: d_nHead,
        n_embd: d_nEmbd,
        mlp_ratio: d_mlpRatio,
        D_tokens: DConstant,
        N_params: baseN,
        batch_size_sequences: batchSeqConstant,
      };

      if (d_sweepVariable === "D_tokens") cfg.D_tokens = x;
      if (d_sweepVariable === "N_params") cfg.N_params = x;
      if (d_sweepVariable === "n_layer") cfg.n_layer = x;
      if (d_sweepVariable === "n_embd") cfg.n_embd = x;
      if (d_sweepVariable === "n_head") cfg.n_head = x;
      if (d_sweepVariable === "batch_size_sequences") {
        cfg.batch_size_sequences = Math.max(1, Math.round(x));
      }

      const architectureDrivenN = gpt_like_nonembedding_params(
        cfg.n_layer,
        cfg.n_embd,
        cfg.mlp_ratio,
        false
      );

      const N = d_sweepVariable === "N_params"
        ? cfg.N_params
        : useNOverride && !["n_layer", "n_embd"].includes(d_sweepVariable)
          ? baseN
          : architectureDrivenN;

      const D = cfg.D_tokens;
      const globalBSeq = batchIsPerGpu
        ? cfg.batch_size_sequences * d_gradAccum * d_worldSize
        : cfg.batch_size_sequences * d_gradAccum;
      const globalBTokens = globalBSeq * d_seqLen;
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

      return {
        x,
        variable: x,
        L: cfg.n_layer,
        H: cfg.n_head,
        dModel: cfg.n_embd,
        head_dim,
        N,
        D,
        D_over_N,
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
    });
  }, [
    d_sweepSpec,
    d_sweepRanges,
    d_sweepVariable,
    d_sweepPoints,
    d_nLayer,
    d_nHead,
    d_nEmbd,
    d_mlpRatio,
    DConstant,
    baseN,
    batchSeqConstant,
    useNOverride,
    batchIsPerGpu,
    d_gradAccum,
    d_worldSize,
    d_seqLen,
  ]);

  const kaplanPanelData = useMemo(() => {
    return makeKaplanDesignerPanels({
      sweepRanges: d_sweepRanges,
      sweepPoints: d_sweepPoints,
      baseN,
      baseD: DConstant,
    });
  }, [d_sweepRanges, d_sweepPoints, baseN, DConstant]);

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

    rows.push({
      quantity: "peak learning rate",
      value: formatSci(peakLr, 2),
    });

    return rows;
  }, [midpoint, bestBalanced, minLoss, lowestCompute, d_sweepSpec, peakLr]);

  const exportConfig = useMemo(() => {
    const range = d_sweepRanges[d_sweepVariable] ?? {
      min: d_sweepSpec.defaultMin,
      max: d_sweepSpec.defaultMax,
    };

    const minActual = d_sweepSpec.scale === "log" ? logSliderToValue(range.min) : range.min;
    const maxActual = d_sweepSpec.scale === "log" ? logSliderToValue(range.max) : range.max;

    return {
      route: "/designer",
      sweep: {
        variable: d_sweepVariable,
        label: d_sweepSpec.label,
        min: minActual,
        max: maxActual,
        points: d_sweepPoints,
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
      },
    };
  }, [
    d_sweepRanges,
    d_sweepVariable,
    d_sweepSpec,
    d_sweepPoints,
    d_nLayer,
    d_nHead,
    d_nEmbd,
    d_mlpRatio,
    useNOverride,
    NOverride,
    DConstant,
    d_seqLen,
    batchSeqConstant,
    d_gradAccum,
    d_worldSize,
    batchIsPerGpu,
    peakLr,
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

  const formatSweepSlider = (value) => {
    if (sweepSpec.scale === "log") return human_num(logSliderToValue(value));
    if ((sweepSpec.step ?? 1) >= 1) return `${Math.round(value)}`;
    return Number(value).toFixed(2);
  };

  const sweepInputToSlider = (value) => {
    if (sweepSpec.scale === "log") return valueToLogSlider(Math.max(value, 1e-300));
    return value;
  };

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
                Pick one sweep variable, keep the rest fixed, and inspect the expected scaling curves live.
              </p>
            </div>

            <div className="space-y-6">
              <ControlSection title="Sweep variable">
                <SelectControl
                  label="Variable to sweep"
                  value={sweepVariable}
                  setValue={setSweepVariable}
                  options={sweepOptions}
                  help="All other fields below are treated as constants unless they are the selected sweep variable."
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
                  label="Sweep points"
                  value={sweepPoints}
                  setValue={setSweepPoints}
                  min={4}
                  max={128}
                  step={1}
                  format={(x) => `${Math.round(x)}`}
                  sanitizeValue={(x) => Math.round(x)}
                />
              </ControlSection>

              <ControlSection title="Architecture constants">
                {sweepVariable === "n_layer" ? (
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

                {sweepVariable === "n_head" ? (
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

                {sweepVariable === "n_embd" ? (
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

                {sweepVariable === "N_params" ? (
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

                {useNOverride && !["N_params", "n_layer", "n_embd"].includes(sweepVariable) && (
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
                {sweepVariable === "D_tokens" ? (
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

                {sweepVariable === "batch_size_sequences" ? (
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
                  <span>Sweeping </span>
                  <span className="font-medium text-blue-300">{d_sweepSpec.label}</span>
                  <span> with constants held fixed from the left panel.</span>
                </p>
              </div>

              {!validHeads && sweepVariable !== "n_head" && (
                <div className="rounded-xl border border-amber-500/30 bg-amber-500/10 px-4 py-2 text-sm text-amber-200">
                  The embedding dimension is not divisible by the number of heads.
                </div>
              )}
            </div>

            <section className="grid grid-cols-1 gap-2 md:grid-cols-2 xl:grid-cols-4">
              <StatCard
                label={<QuantityName name="sweep_variable" />}
                value={d_sweepSpec.mathLabel}
                sub={`${sweepData.length} generated points`}
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
              <JsonBlock value={exportConfig} />
            </section>

            <DataTable title="Sampled run matrix" rows={sampledRunRows} />

            <div className="rounded-2xl border border-zinc-800 bg-zinc-950/80 p-3 text-xs leading-relaxed text-zinc-500">
              <div className="font-semibold uppercase tracking-wide text-zinc-400">Notes</div>
              <p className="mt-1">
                This page is intended as a planning view. The curves are scaling-law expectations from the current configuration,
                not measurements from completed runs. The most useful first pass is usually to sweep one of <MathText>{"\\(D\\)"}</MathText>,{" "}
                <MathText>{"\\(N\\)"}</MathText>, or <MathText>{"\\(d_{\\rm model}\\)"}</MathText>, then choose a run matrix near the bend of
                the predicted loss/compute tradeoff and near <MathText>{"\\(D/N\\approx20\\)"}</MathText>.
              </p>
              <p className="mt-1">
                Constant global batch at the current settings is {human_num(globalBSeqConstant)} sequences. The exported JSON above is
                meant to be copied into a config generator or notebook so the visual sweep and actual training sweep stay synchronized.
              </p>
            </div>
          </main>
        </div>
      </div>
    </MathJaxContext>
  );
}
