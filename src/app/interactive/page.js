"use client";

import { useEffect, useMemo, useRef, useState } from "react";
import { MathJax, MathJaxContext } from "better-react-mathjax";
import dynamic from "next/dynamic";

import {
  single_setting_report,
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
  startup: {
    typeset: false,
  },
  tex: {
    inlineMath: [["\\(", "\\)"], ["$", "$"]],
    displayMath: [["\\[", "\\]"]],
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

function formatSci(x, digits = 3) {
  if (x === null || x === undefined || Number.isNaN(x)) return "—";
  return Number(x).toExponential(digits);
}

function clamp(x, lo, hi) {
  return Math.max(lo, Math.min(hi, x));
}

function parseSmartNumber(raw) {
  if (typeof raw === "number") return raw;

  let text = String(raw).trim();
  if (!text) return NaN;

  text = text
    .replace(/,/g, "")
    .replace(/_/g, "")
    .toLowerCase();

  // Accept values like "17M tokens", "2B params", or "512 sequences".
  text = text.replace(
    /\s*(tokens?|parameters?|params?|sequences?|seqs?|steps?|gpus?|gpu|pf-days?|pflops?)\s*$/i,
    ""
  );

  // Accept both "17 million" and "17million".
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
    D_tokens: (
      <>
        Dataset size <MathText>{"\\(D\\)"}</MathText>
      </>
    ),
    N_params_non_embedding: (
      <>
        Non-embedding parameters <MathText>{"\\(N\\)"}</MathText>
      </>
    ),
    architecture: "Architecture",
    "TPP = D/N": (
      <>
        Tokens per parameter <MathText>{"\\(D/N\\)"}</MathText>
      </>
    ),
    "Kaplan L(N,D) predicted loss": (
      <>
        Kaplan predicted loss <MathText>{"\\(L(N,D)\\)"}</MathText>
      </>
    ),
    "Kaplan balanced D for this N": (
      <>
        Kaplan balanced dataset size for <MathText>{"\\(N\\)"}</MathText>
      </>
    ),
    "Kaplan balanced N for this D": (
      <>
        Kaplan balanced parameter count for <MathText>{"\\(D\\)"}</MathText>
      </>
    ),
    "Power Lines / Chinchilla compute-optimal N ≈ D/20": (
      <>
        Compute-optimal parameter count <MathText>{"\\(N \\approx D/20\\)"}</MathText>
      </>
    ),
    "Power Lines tau_opt": (
      <>
        Power Lines optimizer timescale <MathText>{"\\(\\tau_{\\rm opt}\\)"}</MathText>
      </>
    ),
    "Training FLOPs ≈ 6ND": (
      <>
        Training FLOPs <MathText>{"\\(C \\approx 6ND\\)"}</MathText>
      </>
    ),
    "Training compute PF-days": "Training compute in PF-days",
    "Actual global B_seq": (
      <>
        Actual global batch size <MathText>{"\\(B\\)"}</MathText>, sequences
      </>
    ),
    "Actual global B_tokens": (
      <>
        Actual global batch size <MathText>{"\\(B\\)"}</MathText>, tokens
      </>
    ),
    "Actual optimizer steps": "Actual optimizer steps",
    "Kaplan Bcrit from L(N,D), tokens": (
      <>
        Kaplan critical batch size <MathText>{"\\(B_{\\rm crit}(L)\\)"}</MathText>
      </>
    ),
    "Power Lines Bopt, sequences": (
      <>
        Power Lines optimal batch size <MathText>{"\\(B_{\\rm opt}\\)"}</MathText>
      </>
    ),
    "Power Lines Bcrit, sequences, approx Dmin=D": (
      <>
        Power Lines critical batch size <MathText>{"\\(B_{\\rm crit}\\)"}</MathText>
      </>
    ),
    "Power Lines effective Dmin for actual B": (
      <>
        Effective minimum dataset size <MathText>{"\\(D_{\\min}\\)"}</MathText>
      </>
    ),
    "Power Lines Bcrit at effective Dmin": (
      <>
        Critical batch size at <MathText>{"\\(D_{\\min}\\)"}</MathText>
      </>
    ),
    "Actual extra-data factor vs Dmin": (
      <>
        Extra-data factor versus <MathText>{"\\(D_{\\min}\\)"}</MathText>
      </>
    ),
    "Peak LR used for lambda": (
      <>
        Peak learning rate used for <MathText>{"\\(\\lambda\\)"}</MathText>
      </>
    ),

    batch_size_sequences: (
      <>
        Batch size <MathText>{"\\(B\\)"}</MathText>, sequences
      </>
    ),
    batch_size_tokens: (
      <>
        Batch size <MathText>{"\\(B\\)"}</MathText>, tokens
      </>
    ),
    tau: (
      <>
        Optimizer timescale <MathText>{"\\(\\tau\\)"}</MathText>
      </>
    ),
    lambda_at_Bopt: (
      <>
        Weight decay <MathText>{"\\(\\lambda\\)"}</MathText> near{" "}
        <MathText>{"\\(B_{\\rm opt}\\)"}</MathText>
      </>
    ),
    N_params_for_D: (
      <>
        Parameter count <MathText>{"\\(N\\)"}</MathText> for this dataset
      </>
    ),
    D_tokens_for_N: (
      <>
        Dataset size <MathText>{"\\(D\\)"}</MathText> for this model
      </>
    ),
  };

  return map[name] ?? humanizeIdentifier(name);
}

function BatchChoiceName({ value }) {
  const map = {
    "min / Bopt": (
      <>
        Minimum / <MathText>{"\\(B_{\\rm opt}\\)"}</MathText>
      </>
    ),
    "target / sqrt(Bopt*Bcrit)": (
      <>
        Target / <MathText>{"\\(\\sqrt{B_{\\rm opt}B_{\\rm crit}}\\)"}</MathText>
      </>
    ),
    "max / Bcrit": (
      <>
        Maximum / <MathText>{"\\(B_{\\rm crit}\\)"}</MathText>
      </>
    ),
    actual: "Actual",
  };

  return map[value] ?? value;
}

function HeaderName({ name }) {
  const map = {
    quantity: "Quantity",
    value: "Value",
    parameter: "Parameter",
    minimum: "Minimum",
    recommended_center: "Recommended center",
    maximum_critical: "Maximum / critical",
    actual: "Actual",
    status: "Status",
    batch_choice: "Batch choice",
    B_seq: (
      <>
        <MathText>{"\\(B\\)"}</MathText> sequences
      </>
    ),
    B_tokens: (
      <>
        <MathText>{"\\(B\\)"}</MathText> tokens
      </>
    ),
    steps_for_D: (
      <>
        Steps for <MathText>{"\\(D\\)"}</MathText>
      </>
    ),
    lambda_opt_at_peak_lr: (
      <>
        <MathText>{"\\(\\lambda_{\\rm opt}\\)"}</MathText> at peak learning rate
      </>
    ),
  };

  return map[name] ?? humanizeIdentifier(name);
}

function StatusText({ status }) {
  const map = {
    OK: "Inside range",
    "below Bopt": (
      <>
        Below <MathText>{"\\(B_{\\rm opt}\\)"}</MathText>
      </>
    ),
    "above Bcrit": (
      <>
        Above <MathText>{"\\(B_{\\rm crit}\\)"}</MathText>
      </>
    ),
    "sweep bracket, not a hard law": "Sweep bracket, not a hard law",
    "scales linearly with chosen B and inversely with LR": (
      <>
        Scales linearly with <MathText>{"\\(B\\)"}</MathText> and inversely with the learning rate
      </>
    ),
    "Kaplan balance bracket; compare also D/20 rule": (
      <>
        Kaplan balance bracket; compare with <MathText>{"\\(D/20\\)"}</MathText>
      </>
    ),
    "Kaplan balance bracket; >center means overtrained/data-rich": (
      <>Kaplan balance bracket; above the center means overtrained / data-rich</>
    ),
  };

  return map[status] ?? status;
}

function ResultInfoButton({ status }) {
  if (!status) return null;

  return (
    <span className="group relative inline-flex align-middle">
      <span className="ml-1 inline-flex h-4 w-4 cursor-help items-center justify-center rounded-full border border-zinc-700 bg-zinc-900 text-[10px] font-semibold text-zinc-500 transition group-hover:border-blue-500/60 group-hover:text-blue-300">
        i
      </span>
      <span className="pointer-events-none absolute left-1/2 top-6 z-40 hidden w-72 -translate-x-1/2 rounded-xl border border-zinc-700 bg-zinc-950 p-3 text-xs normal-case leading-snug text-zinc-300 shadow-xl shadow-black/40 group-hover:block">
        <StatusText status={status} />
      </span>
    </span>
  );
}

function renderTableValue(key, value) {
  if (value === null || value === undefined || Number.isNaN(value)) return "—";

  if (typeof value === "number") {
    return key.toLowerCase().includes("lambda")
      ? formatSci(value, 3)
      : human_num(value);
  }

  return value;
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

    // Text input values are intentionally allowed outside the slider range.
    // The slider thumb stays pinned to min/max while the true state uses this value.
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
    <div className="grid grid-cols-[10rem_minmax(0,1fr)_auto] items-center gap-x-3 gap-y-1 rounded-xl border border-zinc-800 bg-zinc-950/70 p-2">
      <div className="min-w-0 pr-1 text-sm font-medium leading-tight text-zinc-100">
        {label}
      </div>

      <div className="min-w-0 w-full">
        <input
          type="range"
          min={min}
          max={max}
          step={step}
          value={sliderValue}
          onChange={(e) => {
            const next = applySliderValue(Number(e.target.value));
            setDraft(String(format(next)));
          }}
          className="block w-full min-w-0 accent-blue-500"
        />

        <div className="flex justify-between gap-2 text-[10px] text-zinc-600">
          <span className="min-w-0 truncate">{format(min)}</span>
          <span className="min-w-0 truncate text-right">{format(max)}</span>
        </div>
      </div>

      <input
        type="text"
        value={draft}
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
        className="h-8 w-25 min-w-25 max-w-25 rounded-lg border border-zinc-800 bg-zinc-900 px-1.5 text-right text-xs text-blue-300 outline-none transition focus:border-blue-500 focus:ring-1 focus:ring-blue-500"
        spellCheck={false}
      />

      {help && (
        <div className="col-start-2 col-span-2 min-w-0 text-xs text-zinc-500">
          {help}
        </div>
      )}
    </div>
  );
};

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

function RecommendedRangesChart({ ranges }) {
  const safeNum = (x) =>
    typeof x === "number" && Number.isFinite(x) && x > 0 ? x : null;

  const log10 = (x) => Math.log10(Math.max(x, 1e-300));

  const pct = (x, lo, hi) => {
    const v = safeNum(x);
    if (v === null) return null;

    const L = log10(lo);
    const H = log10(hi);

    if (Math.abs(H - L) < 1e-12) return 50;

    return Math.max(0, Math.min(100, 100 * (log10(v) - L) / (H - L)));
  };

  const format = (x) => {
    if (x === null || x === undefined || Number.isNaN(x)) return "—";
    return human_num(x);
  };

  return (
    <div className="rounded-2xl border border-zinc-800 bg-zinc-950/80 p-3">
      <div className="mb-3 flex items-center justify-between gap-4">
        <h2 className="text-base font-semibold text-zinc-100">
          Recommended ranges
        </h2>

        <div className="flex items-center gap-3 text-[11px] text-zinc-400">
          <span className="inline-flex items-center gap-1">
            <span className="h-2 w-5 rounded-full bg-blue-500/30" />
            Range
          </span>
          <span className="inline-flex items-center gap-1">
            <span className="h-2 w-1.5 rounded-full bg-blue-400" />
            Recommended
          </span>
          <span className="inline-flex items-center gap-1">
            <span className="h-3 w-3 rounded-full border-2 border-amber-300" />
            Actual
          </span>
        </div>
      </div>

      <div className="space-y-2">
        {ranges.map((row) => {
          const min = safeNum(row.minimum);
          const rec = safeNum(row.recommended_center);
          const max = safeNum(row.maximum_critical);
          const actual = safeNum(row.actual);

          const vals = [min, rec, max, actual].filter(Boolean);
          if (vals.length < 2) return null;

          const rawLo = Math.min(...vals);
          const rawHi = Math.max(...vals);
          const domainLo = rawLo / 1.35;
          const domainHi = rawHi * 1.35;

          const minPct = pct(min, domainLo, domainHi);
          const recPct = pct(rec, domainLo, domainHi);
          const maxPct = pct(max, domainLo, domainHi);
          const actualPct = pct(actual, domainLo, domainHi);

          const bandLeft = Math.min(minPct ?? 0, maxPct ?? 100);
          const bandWidth = Math.abs((maxPct ?? 100) - (minPct ?? 0));

          const actualStatus =
            actual === null
              ? "no actual"
              : min !== null && actual < min
                ? "below range"
                : max !== null && actual > max
                  ? "above range"
                  : "inside range";

          const statusClass =
            actualStatus === "inside range"
              ? "text-emerald-300"
              : actualStatus === "no actual"
                ? "text-zinc-500"
                : "text-amber-300";

          return (
            <div
              key={row.parameter}
              className="grid grid-cols-[minmax(10rem,14rem)_minmax(10rem,1fr)_minmax(13rem,18rem)] items-center gap-3 rounded-xl border border-zinc-800 bg-zinc-950/70 px-3 py-2"
            >
              <div className="min-w-0">
                <div className="flex min-w-0 items-center text-sm font-medium text-zinc-100">
                  <span className="min-w-0 truncate">
                    <QuantityName name={row.parameter} />
                  </span>
                  <ResultInfoButton status={row.status} />
                </div>
              </div>

              <div className="relative h-8 min-w-0">
                <div className="absolute left-0 right-0 top-3.5 h-2 rounded-full bg-zinc-800" />

                <div
                  className="absolute top-3.5 h-2 rounded-full bg-blue-500/30"
                  style={{
                    left: `${bandLeft}%`,
                    width: `${bandWidth}%`,
                  }}
                />

                {recPct !== null && (
                  <div
                    className="absolute top-2 h-5 w-1.5 -translate-x-1/2 rounded-full bg-blue-400"
                    style={{ left: `${recPct}%` }}
                    title={`recommended: ${format(rec)}`}
                  />
                )}

                {actualPct !== null && (
                  <div
                    className="absolute top-1.5 h-6 w-6 -translate-x-1/2 rounded-full border-2 border-amber-300 bg-zinc-950 shadow-lg shadow-amber-500/20"
                    style={{ left: `${actualPct}%` }}
                    title={`actual: ${format(actual)}`}
                  />
                )}
              </div>

              <div className="grid min-w-0 grid-cols-2 gap-x-3 gap-y-0.5 text-[11px] leading-tight">
                <div className="min-w-0 truncate">
                  <span className="text-zinc-500">Min </span>
                  <span className="text-zinc-300">{format(min)}</span>
                </div>
                <div className="min-w-0 truncate text-right">
                  <span className="text-zinc-500">Max </span>
                  <span className="text-zinc-300">{format(max)}</span>
                </div>
                <div className="min-w-0 truncate">
                  <span className="text-blue-300">Rec </span>
                  <span className="text-blue-200">{format(rec)}</span>
                </div>
                <div className="min-w-0 truncate text-right">
                  <span className="text-amber-300">Actual </span>
                  <span className="text-amber-200">{format(actual)}</span>
                </div>
                <div className={`col-span-2 min-w-0 truncate text-right ${statusClass}`}>
                  {humanizeIdentifier(actualStatus)}
                </div>
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}

function sanitizeFilenamePart(value) {
  return String(value ?? "")
    .trim()
    .replace(/[^a-zA-Z0-9._-]+/g, "_")
    .replace(/^_+|_+$/g, "") || "scaling_law_chart";
}

function plainText(value) {
  if (typeof value === "string") return value;
  return "interactive scaling-law chart";
}

function plotLayout({
  xTitle,
  yTitle,
  logX = false,
  logY = false,
  shapes = [],
  legendY = 1.14,
  margin = { l: 62, r: 24, t: 8, b: 54 },
}) {
  return {
    autosize: true,
    paper_bgcolor: "rgba(0,0,0,0)",
    plot_bgcolor: "rgba(0,0,0,0)",
    font: { color: "#d4d4d8", size: 12 },
    margin,
    hovermode: "closest",
    dragmode: "zoom",
    legend: {
      orientation: "h",
      x: 0,
      y: legendY,
      font: { size: 11, color: "#d4d4d8" },
    },
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
    displayModeBar: "hover",
    toImageButtonOptions: {
      format: "png",
      filename: sanitizeFilenamePart(filename),
      height: 900,
      width: 1400,
      scale: 2,
    },
    modeBarButtonsToRemove: ["lasso2d", "select2d"],
  };
}

function PlotCard({ title, subtitle, traces, layout, config, heightClass = "h-56", footer }) {
  const fileTitle = plainText(title);

  return (
    <div className="rounded-2xl border border-zinc-800 bg-zinc-950/80 p-3">
      <div className="mb-2">
        <h2 className="text-base font-semibold text-zinc-100">{title}</h2>
        {subtitle && <p className="mt-1 text-xs leading-relaxed text-zinc-500">{subtitle}</p>}
      </div>

      <div className={heightClass}>
        <Plot
          data={traces}
          layout={layout}
          config={config ?? plotConfig(fileTitle)}
          useResizeHandler
          className="h-full w-full"
          style={{ width: "100%", height: "100%" }}
        />
      </div>

      <div className="mt-2 text-xs text-zinc-500">
        {footer ?? "Drag to zoom. Use the modebar to pan, autoscale, reset axes, and export PNG. Mouse wheel zoom is enabled."}
      </div>
    </div>
  );
}

function SmallChart({ title, data, xKey, yKey, yLabel, tooltipFormatter, logX = true, logY = false }) {
  const clean = (data ?? []).filter(
    (row) => Number.isFinite(row?.[xKey]) && row[xKey] > 0 && Number.isFinite(row?.[yKey])
  );

  const traces = [
    {
      type: "scatter",
      mode: "lines",
      name: yLabel,
      x: clean.map((row) => row[xKey]),
      y: clean.map((row) => row[yKey]),
      text: clean.map((row) => {
        const yText = tooltipFormatter ? tooltipFormatter(row[yKey]) : human_num(row[yKey]);
        return [
          `${xKey}: ${human_num(row[xKey])}`,
          `${yLabel}: ${yText}`,
        ].join("<br>");
      }),
      hovertemplate: "%{text}<extra></extra>",
      line: { color: "#60a5fa", width: 2.6 },
    },
  ];

  return (
    <PlotCard
      title={title}
      traces={traces}
      layout={plotLayout({
        xTitle: xKey,
        yTitle: yLabel,
        logX,
        logY,
      })}
      config={plotConfig(`${plainText(title)}_${xKey}_${yKey}`)}
    />
  );
}

function BatchRangeChart({ data }) {
  const clean = (data ?? []).filter((row) => Number.isFinite(row?.D) && row.D > 0);

  const bopt = clean.map((row) => row.Bopt);
  const bcrit = clean.map((row) => row.Bcrit);
  const btarget = clean.map((row) => row.Btarget);
  const bactual = clean.map((row) => row.Bactual);
  const ds = clean.map((row) => row.D);

  const traces = [
    {
      type: "scatter",
      mode: "lines",
      name: "Bcrit",
      x: ds,
      y: bcrit,
      hovertemplate: "D=%{x:.3e}<br>Bcrit=%{y:.3e}<extra></extra>",
      line: { color: "#60a5fa", width: 2.6 },
    },
    {
      type: "scatter",
      mode: "lines",
      name: "Btarget",
      x: ds,
      y: btarget,
      hovertemplate: "D=%{x:.3e}<br>Btarget=%{y:.3e}<extra></extra>",
      line: { color: "#34d399", width: 2.6, dash: "dash" },
    },
    {
      type: "scatter",
      mode: "lines",
      name: "Bopt",
      x: ds,
      y: bopt,
      hovertemplate: "D=%{x:.3e}<br>Bopt=%{y:.3e}<extra></extra>",
      line: { color: "#fbbf24", width: 2.6 },
    },
    {
      type: "scatter",
      mode: "lines",
      name: "Bactual",
      x: ds,
      y: bactual,
      hovertemplate: "D=%{x:.3e}<br>Bactual=%{y:.3e}<extra></extra>",
      line: { color: "#fb7185", width: 2.4, dash: "dot" },
    },
  ];

  return (
    <PlotCard
      title="Power Lines batch range versus dataset size"
      subtitle="The four traces are now Plotly traces, so you can zoom, pan, autoscale, reset axes, and export the chart."
      traces={traces}
      layout={plotLayout({
        xTitle: "Dataset size D, tokens",
        yTitle: "Batch size B, sequences",
        logX: true,
        logY: true,
        legendY: 1.16,
      })}
      config={plotConfig("power_lines_batch_range")}
      footer={
        <span>
          <MathText>{"\\(B_{\\rm opt}\\)"}</MathText> = compute-efficient ·{" "}
          <MathText>{"\\(B_{\\rm target}=\\sqrt{B_{\\rm opt}B_{\\rm crit}}\\)"}</MathText> ·{" "}
          <MathText>{"\\(B_{\\rm crit}\\)"}</MathText> = critical batch ·{" "}
          <MathText>{"\\(B_{\\rm actual}\\)"}</MathText> = your global batch
        </span>
      }
    />
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

export default function ScalingLawInteractiveMenu() {
  const layoutRef = useRef(null);
  const [sidebarWidth, setSidebarWidth] = useState(430);
  const [isResizingSidebar, setIsResizingSidebar] = useState(false);

  // Main architecture
  const [nLayer, setNLayer] = useState(8);
  const [nHead, setNHead] = useState(8);
  const [nEmbd, setNEmbd] = useState(512);
  const [mlpRatio, setMlpRatio] = useState(4);

  // Dataset/training
  const [logD, setLogD] = useState(valueToLogSlider(35e9));
  const [seqLen, setSeqLen] = useState(1024);
  const [logBatchSeq, setLogBatchSeq] = useState(valueToLogSlider(32));
  const [gradAccum, setGradAccum] = useState(96);
  const [worldSize, setWorldSize] = useState(4);
  const [batchIsPerGpu, setBatchIsPerGpu] = useState(false);

  // Optimizer
  const [logPeakLr, setLogPeakLr] = useState(valueToLogSlider(3e-4));
  const [useMup, setUseMup] = useState(false);
  const [proxyWidth, setProxyWidth] = useState(128);
  const [logProxyLr, setLogProxyLr] = useState(valueToLogSlider(3e-4));

  // Optional direct N override
  const [useNOverride, setUseNOverride] = useState(false);
  const [logNOverride, setLogNOverride] = useState(valueToLogSlider(20e6));

  // Debounce expensive outputs
  const d_nLayer = useDebouncedValue(nLayer);
  const d_nHead = useDebouncedValue(nHead);
  const d_nEmbd = useDebouncedValue(nEmbd);
  const d_mlpRatio = useDebouncedValue(mlpRatio);
  const d_logD = useDebouncedValue(logD);
  const d_seqLen = useDebouncedValue(seqLen);
  const d_logBatchSeq = useDebouncedValue(logBatchSeq);
  const d_gradAccum = useDebouncedValue(gradAccum);
  const d_worldSize = useDebouncedValue(worldSize);
  const d_logPeakLr = useDebouncedValue(logPeakLr);
  const d_proxyWidth = useDebouncedValue(proxyWidth);
  const d_logProxyLr = useDebouncedValue(logProxyLr);
  const d_logNOverride = useDebouncedValue(logNOverride);

  useEffect(() => {
    if (!isResizingSidebar) return;

    const oldCursor = document.body.style.cursor;
    const oldUserSelect = document.body.style.userSelect;

    const onPointerMove = (event) => {
      const rect = layoutRef.current?.getBoundingClientRect();
      const left = rect?.left ?? 0;
      const layoutWidth = rect?.width ?? window.innerWidth;

      // Keep enough room for the output panel, and avoid absurdly wide sidebars.
      const maxSidebarWidth = Math.max(430, Math.min(820, layoutWidth - 420));
      const nextWidth = clamp(event.clientX - left, 430, maxSidebarWidth);
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

  const D_tokens = logSliderToValue(d_logD);
  const batchSizeSequences = Math.round(logSliderToValue(d_logBatchSeq));
  const peakLr = logSliderToValue(d_logPeakLr);
  const proxyBaseLr = logSliderToValue(d_logProxyLr);
  const NOverride = logSliderToValue(d_logNOverride);

  const architectureN = useMemo(() => {
    return gpt_like_nonembedding_params(
      d_nLayer,
      d_nEmbd,
      d_mlpRatio,
      false
    );
  }, [d_nLayer, d_nEmbd, d_mlpRatio]);

  const report = useMemo(() => {
    return single_setting_report({
      D_tokens,
      N_params: useNOverride ? NOverride : null,
      n_layer: d_nLayer,
      n_embd: d_nEmbd,
      n_head: d_nHead,
      mlp_ratio: d_mlpRatio,
      seq_len: d_seqLen,
      batch_size_sequences: batchSizeSequences,
      grad_accum: d_gradAccum,
      num_gpus: d_worldSize,
      batch_size_is_per_gpu: batchIsPerGpu,
      peak_lr: peakLr,
      proxy_width: useMup ? d_proxyWidth : null,
      proxy_base_lr: useMup ? proxyBaseLr : null,
    });
  }, [
    D_tokens,
    useNOverride,
    NOverride,
    d_nLayer,
    d_nEmbd,
    d_nHead,
    d_mlpRatio,
    d_seqLen,
    batchSizeSequences,
    d_gradAccum,
    d_worldSize,
    batchIsPerGpu,
    peakLr,
    useMup,
    d_proxyWidth,
    proxyBaseLr,
  ]);

  const raw = report.raw ?? {};
  const N = raw.N_params_non_embedding ?? architectureN;
  const tpp = raw.tpp ?? tokens_per_parameter(D_tokens, N);
  const tau = raw.tau ?? powerlines_tau_opt(tpp);

  const globalBSeq = batchIsPerGpu
    ? batchSizeSequences * d_gradAccum * d_worldSize
    : batchSizeSequences * d_gradAccum;

  const globalBTokens = globalBSeq * d_seqLen;
  const optimizerSteps = D_tokens / globalBTokens;
  const headDim = d_nEmbd / d_nHead;
  const validHeads = Number.isInteger(headDim);

  const lossVsNData = useMemo(() => {
    const points = [];
    const lo = 1e5;
    const hi = 1e10;
    const count = 56;

    for (let i = 0; i < count; i++) {
      const f = i / (count - 1);
      const Np = lo * (hi / lo) ** f;
      points.push({
        N: Np,
        loss: kaplan_loss_ND(Np, D_tokens),
      });
    }

    return points;
  }, [D_tokens]);

  const computeVsDData = useMemo(() => {
    const points = [];
    const lo = 1e8;
    const hi = 2e11;
    const count = 56;

    for (let i = 0; i < count; i++) {
      const f = i / (count - 1);
      const Dp = lo * (hi / lo) ** f;

      points.push({
        D: Dp,
        pfDays: flops_to_pf_days(training_flops(N, Dp)),
      });
    }

    return points;
  }, [N]);

  const batchRangeData = useMemo(() => {
    const points = [];
    const lo = 1e8;
    const hi = 2e11;
    const count = 56;

    for (let i = 0; i < count; i++) {
      const f = i / (count - 1);
      const Dp = lo * (hi / lo) ** f;
      const Bopt = powerlines_Bopt_sequences(Dp);
      const Bcrit = powerlines_Bcrit_sequences(Dp);

      points.push({
        D: Dp,
        Bopt,
        Btarget: recommended_batch_target(Bopt, Bcrit),
        Bcrit,
        Bactual: globalBSeq,
      });
    }

    return points;
  }, [globalBSeq]);

  const usefulSummaryRows = useMemo(() => {
    const rows = report.summary ?? [];

    return rows.map((row) => ({
      quantity: <QuantityName name={row.quantity} />,
      value: row.formatted ?? human_num(row.value),
    }));
  }, [report.summary]);

  const lambdaRows = useMemo(() => {
    const rows = report.lambdaTable ?? report.lambda_table ?? [];

    return rows.map((row) => ({
      batch_choice: <BatchChoiceName value={row.batch_choice} />,
      B_seq: row.B_seq,
      B_tokens: row.B_tokens,
      steps_for_D: row.steps_for_D,
      lambda_opt_at_peak_lr: row.lambda_opt_at_peak_lr,
    }));
  }, [report.lambdaTable, report.lambda_table]);

  return (
    <MathJaxContext config={MATHJAX_CONFIG}>
      <div className="min-h-screen bg-zinc-950 text-zinc-100">
        <div
          ref={layoutRef}
          className="min-h-screen lg:flex lg:items-stretch"
        >
          {/* LEFT INPUT PANEL */}
          <aside
            className="border-b border-zinc-800 bg-zinc-950/95 p-5 lg:sticky lg:top-0 lg:h-screen lg:shrink-0 lg:overflow-y-auto lg:border-b-0 lg:border-r"
            style={{
              width: `min(100%, ${sidebarWidth}px)`,
              flexBasis: `${sidebarWidth}px`,
            }}
          >

            <div className="mb-6">
              <h1 className="text-2xl font-semibold tracking-tight">
                Scaling law controls
              </h1>
              <p className="mt-2 text-sm text-zinc-500">
                Move the sliders on the left. The outputs update on the right.
              </p>
            </div>

            <div className="space-y-6">
              <section className="space-y-3">
                <h2 className="text-sm font-semibold uppercase tracking-wide text-zinc-400">
                  Architecture
                </h2>

                <SliderControl
                  label="Number of layers"
                  value={nLayer}
                  setValue={setNLayer}
                  min={1}
                  max={96}
                  step={1}
                  format={(x) => `${Math.round(x)}`}
                  sanitizeValue={(x) => Math.round(x)}
                />

                <SliderControl
                  label="Number of attention heads"
                  value={nHead}
                  setValue={setNHead}
                  min={1}
                  max={64}
                  step={1}
                  format={(x) => `${Math.round(x)}`}
                  sanitizeValue={(x) => Math.round(x)}
                  help={
                    !validHeads
                      ? "Warning: the embedding dimension is not divisible by the number of attention heads."
                      : " "
                  }
                />

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

                <SliderControl
                  label="MLP expansion ratio"
                  value={mlpRatio}
                  setValue={setMlpRatio}
                  min={1}
                  max={8}
                  step={0.25}
                  format={(x) => `${x.toFixed(2)}`}
                />

                <ToggleControl
                  label="Override the parameter count"
                  checked={useNOverride}
                  setChecked={setUseNOverride}
                  help="Use a direct non-embedding parameter count instead of the GPT-like estimate."
                />

                {useNOverride && (
                  <SliderControl
                    label="Non-embedding parameter override"
                    value={logNOverride}
                    setValue={setLogNOverride}
                    min={valueToLogSlider(1e5)}
                    max={valueToLogSlider(2e9)}
                    step={0.01}
                    format={(x) => human_num(logSliderToValue(x))}
                    inputValueToSliderValue={(x) => valueToLogSlider(x)}
                  />
                )}
              </section>

              <section className="space-y-3">
                <h2 className="text-sm font-semibold uppercase tracking-wide text-zinc-400">
                  Dataset and training
                </h2>

                <SliderControl
                  label={
                    <>
                      Dataset size <MathText>{"\\(D\\)"}</MathText>
                    </>
                  }
                  value={logD}
                  setValue={setLogD}
                  min={valueToLogSlider(1e5)}
                  max={valueToLogSlider(2e11)}
                  step={0.01}
                  format={(x) => human_num(logSliderToValue(x))}
                  inputValueToSliderValue={(x) => valueToLogSlider(x)}
                />

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

                <SliderControl
                  label={
                    <>
                      Batch size <MathText>{"\\(B\\)"}</MathText>, sequences
                    </>
                  }
                  value={logBatchSeq}
                  setValue={setLogBatchSeq}
                  min={valueToLogSlider(1)}
                  max={valueToLogSlider(4096)}
                  step={0.01}
                  format={(x) => human_num(Math.round(logSliderToValue(x)))}
                  inputValueToSliderValue={(x) => valueToLogSlider(x)}
                />

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
                      If enabled, global batch size is{" "}
                      <MathText>{"\\(B_{\\rm global}=B_{\\rm local}GS\\)"}</MathText>.
                    </>
                  }
                />
              </section>

              <section className="space-y-3">
                <h2 className="text-sm font-semibold uppercase tracking-wide text-zinc-400">
                  Optimizer
                </h2>

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

                <ToggleControl
                  label="Use µP learning-rate scaling"
                  checked={useMup}
                  setChecked={setUseMup}
                  help={
                    <MathText>
                      {"\\(\\eta_{\\rm target}=(d_{\\rm proxy}/d_{\\rm target})\\eta_{\\rm proxy}\\)"}
                    </MathText>
                  }
                />

                {useMup && (
                  <>
                    <SliderControl
                      label="Proxy width"
                      value={proxyWidth}
                      setValue={setProxyWidth}
                      min={32}
                      max={2048}
                      step={32}
                      format={(x) => `${Math.round(x)}`}
                      sanitizeValue={(x) => Math.round(x)}
                    />

                    <SliderControl
                      label={
                        <>
                          Proxy base learning rate <MathText>{"\\(\\eta_{\\rm proxy}\\)"}</MathText>
                        </>
                      }
                      value={logProxyLr}
                      setValue={setLogProxyLr}
                      min={valueToLogSlider(1e-8)}
                      max={valueToLogSlider(1e-2)}
                      step={0.01}
                      format={(x) => formatSci(logSliderToValue(x), 2)}
                      inputValueToSliderValue={(x) => valueToLogSlider(x)}
                    />
                  </>
                )}
              </section>
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
            onDoubleClick={() => setSidebarWidth(430)}
            className="hidden w-1 shrink-0 cursor-col-resize bg-zinc-900 transition hover:bg-blue-500/40 active:bg-blue-500/60 lg:block"
            style={{ touchAction: "none" }}
          />

          {/* RIGHT OUTPUT PANEL */}
          <main className="min-w-0 flex-1 space-y-3 p-4 lg:p-5">
            <div className="flex flex-col gap-2 sm:flex-row sm:items-end sm:justify-between">
              <div>
                <h1 className="text-3xl font-semibold tracking-tight">
                  Scaling law dashboard
                </h1>
                <p className="mt-2 text-sm text-zinc-500">
                  <MathText>
                    {`\\(L=${d_nLayer},\\ H=${d_nHead},\\ d_{\\rm model}=${d_nEmbd},\\ d_{\\rm head}=${
                      validHeads ? headDim.toFixed(0) : headDim.toFixed(2)
                    }\\)`}
                  </MathText>
                </p>
              </div>

              {!validHeads && (
                <div className="rounded-xl border border-amber-500/30 bg-amber-500/10 px-4 py-2 text-sm text-amber-200">
                  The embedding dimension is not divisible by the number of attention heads.
                </div>
              )}
            </div>

            <section className="grid grid-cols-1 gap-2 md:grid-cols-2 xl:grid-cols-4">
              <StatCard
                label={
                  <>
                    Non-embedding parameters <MathText>{"\\(N\\)"}</MathText>
                  </>
                }
                value={human_num(N)}
                sub={useNOverride ? "Manual override" : "GPT-like estimate"}
              />

              <StatCard
                label={
                  <>
                    Dataset size <MathText>{"\\(D\\)"}</MathText>
                  </>
                }
                value={human_num(D_tokens)}
                sub={
                  <MathText>{`\\(D/N=${tpp.toFixed(2)}\\) tokens per parameter`}</MathText>
                }
              />

              <StatCard
                label={
                  <>
                    Global batch size <MathText>{"\\(B\\)"}</MathText>
                  </>
                }
                value={human_num(globalBSeq)}
                sub={`${human_num(globalBTokens)} tokens per step`}
              />

              <StatCard
                label="Optimizer steps"
                value={human_num(optimizerSteps)}
                sub={<MathText>{`\\(\\tau_{\\rm opt}\\approx ${tau.toFixed(4)}\\)`}</MathText>}
              />
            </section>

            <RecommendedRangesChart ranges={report.ranges ?? []} />

            <section className="grid grid-cols-1 gap-3 xl:grid-cols-2">
              <SmallChart
                title={
                  <>
                    Kaplan predicted loss versus model size <MathText>{"\\(N\\)"}</MathText>
                  </>
                }
                data={lossVsNData}
                xKey="N"
                yKey="loss"
                yLabel="Predicted loss"
                tooltipFormatter={(v) => Number(v).toFixed(5)}
                logX
              />

              <SmallChart
                title={
                  <>
                    Training compute versus dataset size <MathText>{"\\(D\\)"}</MathText>
                  </>
                }
                data={computeVsDData}
                xKey="D"
                yKey="pfDays"
                yLabel="PF-days"
                tooltipFormatter={(v) => `${Number(v).toFixed(4)} PF-days`}
                logX
                logY
              />
            </section>

            <BatchRangeChart data={batchRangeData} />

            <section className="grid grid-cols-1 gap-3 xl:grid-cols-2">
              <DataTable title="Summary" rows={usefulSummaryRows} />
              <DataTable title="Weight decay recommendations" rows={lambdaRows} />
            </section>
          </main>
        </div>
      </div>
    </MathJaxContext>
  );
}
