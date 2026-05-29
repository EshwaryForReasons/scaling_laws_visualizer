// lib/scalingLaws.js
// Browser-safe / Next.js-safe JavaScript port of the attached Python scaling-law utilities.
// Import this from a client component that owns your sliders.

// =====================
// Constants from papers
// =====================

export const KAPLAN = Object.freeze({
    // Summary one-factor power laws from Kaplan et al. Section 1.2
    alpha_N_summary: 0.076,
    N_c_summary: 8.8e13,
    alpha_D_summary: 0.095,
    D_c_summary: 5.4e13,
    alpha_Cmin: 0.050,
    Cmin_c_pf_days: 3.1e8,

    // Joint N,D fit from Kaplan Table 2 / Eq. 1.5 / Eq. 4.1
    alpha_N: 0.076,
    alpha_D: 0.103,
    N_c: 6.4e13,
    D_c: 1.8e13,

    // Training-step / critical batch parameters from Kaplan Eq. 1.4 and Eq. 1.6
    B_star_tokens: 2e8,
    alpha_B: 0.21,
    S_c: 2.1e3,
    alpha_S: 0.76,

    flops_per_pf_day: 8.64e19,
});

export const PL = Object.freeze({
    // Figure 1 fits, v2 paper.
    c_tau: 1.084,
    m_tau: -0.527,
    c_bopt: 0.0306,
    m_bopt: 0.383,
    c_bcrit: 0.0471,
    m_bcrit: 0.462,

    // Reported uncertainty-ish exponent percentiles in text; useful for sweep brackets.
    m_bopt_p10: 0.367,
    m_bopt_p90: 0.391,
    m_bcrit_p10: 0.491,
    m_bcrit_p90: 0.526,

    // Paper settings. Used only for warnings/reporting; lambda uses provided seq_len by default.
    paper_seq_len: 2048,
    compute_optimal_tpp_rule: 20.0,
    studied_tpp_min: 20.0,
    studied_tpp_max: 1280.0,
});

// =====================
// Formatting and small utilities
// =====================

export function isNil(x) {
    return x === null || x === undefined;
}

export function isFiniteNumber(x) {
    return typeof x === "number" && Number.isFinite(x);
}

export function formatSig(x, precision = 3) {
    if (!isFiniteNumber(x)) return "nan";
    return Number(x).toPrecision(precision).replace(/\.0+(?=e|$)/i, "").replace(/(\.\d*?)0+(?=e|$)/i, "$1");
}

export function formatFixed(x, digits = 4) {
    if (!isFiniteNumber(x)) return "nan";
    return Number(x).toFixed(digits).replace(/\.0+$/, "").replace(/(\.\d*?)0+$/, "$1");
}

export function formatScientific(x, digits = 3) {
    if (!isFiniteNumber(x)) return "nan";
    return Number(x).toExponential(digits);
}

export function human_num(x, precision = 3) {
    if (isNil(x) || Number.isNaN(Number(x))) return "nan";

    const value = Number(x);
    const sign = value < 0 ? "-" : "";
    const absValue = Math.abs(value);
    const units = [
        [1e15, "P"],
        [1e12, "T"],
        [1e9, "B"],
        [1e6, "M"],
        [1e3, "K"],
    ];

    for (const [scale, suffix] of units) {
        if (absValue >= scale) return `${sign}${formatSig(absValue / scale, precision)}${suffix}`;
    }

    return `${sign}${formatSig(absValue, precision)}`;
}

export function finiteOrNaN(x) {
    return Number.isFinite(Number(x)) ? Number(x) : Number.NaN;
}

// =====================
// Architecture helpers
// =====================

export function kaplan_nonembedding_params(n_layer, d_model) {
    // Kaplan approximation: N ≈ 12 * n_layer * d_model^2 for standard GPT-like blocks.
    return 12.0 * Number(n_layer) * Number(d_model) ** 2;
}

export function gpt_like_nonembedding_params(
    n_layer,
    d_model,
    mlp_ratio = 4.0,
    include_bias = false,
) {
    /*
    Slightly more explicit GPT-like non-embedding parameter estimate.
  
    Per layer, ignoring norms/biases:
      attention qkv + proj: 4*d^2
      MLP up + down: 2*d*(mlp_ratio*d)
  
    For mlp_ratio=4, this is 12*d^2 per layer, matching Kaplan's approximation.
    */
    const L = Number(n_layer);
    const d = Number(d_model);
    const hidden = Math.trunc(Number(mlp_ratio) * d);

    const perLayer = 4 * d ** 2 + 2 * d * hidden;
    let total = L * perLayer;

    if (include_bias) {
        // Very rough: qkv/proj/MLP biases plus two layernorm scales per block.
        total += L * (4 * d + hidden + d + 2 * d);
    }

    return total;
}

export function embedding_params(
    vocab_size,
    d_model,
    seq_len = 0,
    include_positional = true,
) {
    const tok = Number(vocab_size) * Number(d_model);
    const pos = include_positional ? Number(seq_len) * Number(d_model) : 0;
    return Math.trunc(tok + pos);
}

export function resolve_N_params(
    N_params,
    n_layer,
    d_model,
    mlp_ratio = 4.0,
) {
    if (!isNil(N_params) && Number.isFinite(Number(N_params))) return Number(N_params);
    return gpt_like_nonembedding_params(n_layer, d_model, mlp_ratio);
}

export function global_batch_sequences(
    batch_size_sequences,
    grad_accum,
    num_gpus = 1,
    batch_size_is_per_gpu = false,
) {
    const b = Number(batch_size_sequences);
    const ga = Number(grad_accum);
    const g = Number(num_gpus);
    return Math.trunc(batch_size_is_per_gpu ? b * ga * g : b * ga);
}

// =====================
// Kaplan scaling-law functions
// =====================

export function kaplan_loss_N(N, k = KAPLAN) {
    // Kaplan one-factor infinite-data model-size-limited loss.
    return (k.N_c_summary / Number(N)) ** k.alpha_N_summary;
}

export function kaplan_loss_D(D_tokens, k = KAPLAN) {
    // Kaplan one-factor data-limited loss.
    return (k.D_c_summary / Number(D_tokens)) ** k.alpha_D_summary;
}

export function kaplan_loss_ND(N, D_tokens, k = KAPLAN) {
    // Kaplan joint model/data early-stopped loss, Eq. 1.5 / 4.1.
    return (
        (k.N_c / Number(N)) ** (k.alpha_N / k.alpha_D) +
        k.D_c / Number(D_tokens)
    ) ** k.alpha_D;
}

export function kaplan_balance_D_for_N(N, k = KAPLAN) {
    // Dataset size where Kaplan's model-size and data-size terms are equal.
    // This gives the D ∝ N^(alpha_N/alpha_D) balance implied by Eq. 1.5.
    return k.D_c * (Number(N) / k.N_c) ** (k.alpha_N / k.alpha_D);
}

export function kaplan_balance_N_for_D(D_tokens, k = KAPLAN) {
    // Inverse of kaplan_balance_D_for_N.
    return k.N_c * (Number(D_tokens) / k.D_c) ** (k.alpha_D / k.alpha_N);
}

export function kaplan_Bcrit_tokens_from_loss(loss, k = KAPLAN) {
    // Kaplan critical batch size in tokens: Bcrit(L) = B_star / L^(1/alpha_B).
    return k.B_star_tokens / Number(loss) ** (1.0 / k.alpha_B);
}

export function training_flops(N, D_tokens) {
    // One-pass / total-token training compute approximation C ≈ 6 N D FLOPs.
    return 6.0 * Number(N) * Number(D_tokens);
}

export function flops_to_pf_days(flops, k = KAPLAN) {
    return Number(flops) / k.flops_per_pf_day;
}

export function kaplan_early_stop_lower_bound_steps(N, D_tokens, k = KAPLAN) {
    /*
    Rough Kaplan lower-bound estimate from Eq. 5.7:
        S_stop >= S_c / (L(N,D) - L(N,∞))^(1/alpha_S)
    Uses joint L(N,D) and infinite-D L(N,∞).
    */
    const L_nd = kaplan_loss_ND(N, D_tokens, k);
    const L_n_inf = (k.N_c / Number(N)) ** k.alpha_N;
    const delta = Math.max(L_nd - L_n_inf, 1e-12);
    return k.S_c / delta ** (1.0 / k.alpha_S);
}

// =====================
// Power Lines functions
// =====================

export function tokens_per_parameter(D_tokens, N) {
    return Number(D_tokens) / Number(N);
}

export function powerlines_tau_opt(tpp, pl = PL) {
    return pl.c_tau * Number(tpp) ** pl.m_tau;
}

export function powerlines_Bopt_sequences(D_tokens, pl = PL) {
    // Power Lines Fig. 1 fit: Bopt = 0.0306 * D^0.383. B is reported in sequences.
    return pl.c_bopt * Number(D_tokens) ** pl.m_bopt;
}

export function powerlines_Bcrit_sequences(Dmin_tokens, pl = PL) {
    // Power Lines Fig. 1 / Eq. 7 fit: Bcrit = 0.0471 * Dmin^0.462. B is reported in sequences.
    return pl.c_bcrit * Number(Dmin_tokens) ** pl.m_bcrit;
}

export function powerlines_required_tokens_for_batch(Dmin_tokens, B_seq, pl = PL) {
    // Eq. 6: D = Dmin * (1 + B/Bcrit(Dmin)).
    const dmin = Number(Dmin_tokens);
    return dmin * (1.0 + Number(B_seq) / powerlines_Bcrit_sequences(dmin, pl));
}

export function infer_Dmin_from_planned_tokens(
    D_actual_tokens,
    B_seq,
    pl = PL,
    { rtol = 1e-8, max_iter = 200 } = {},
) {
    /*
    Solve D_actual = Dmin * (1 + B/Bcrit(Dmin)) for Dmin by bisection.
    Useful when you know your actual planned tokens and batch size and want an effective Dmin.
    */
    let lo = 1.0;
    let hi = Number(D_actual_tokens);

    for (let i = 0; i < max_iter; i += 1) {
        const mid = 0.5 * (lo + hi);
        const val = powerlines_required_tokens_for_batch(mid, B_seq, pl);

        if (val > Number(D_actual_tokens)) hi = mid;
        else lo = mid;

        if (Math.abs(hi - lo) / Math.max(mid, 1.0) < rtol) break;
    }

    return 0.5 * (lo + hi);
}

export function powerlines_lambda_opt(
    B_seq,
    D_tokens,
    N,
    peak_lr,
    seq_len,
    pl = PL,
) {
    /*
    Eq. 2 + Eq. 3 + Eq. 4:
        tau = B / (eta * lambda * D)
        lambda_opt = B / (eta * D_seq * tau_opt)
  
    Here B and D must be in the same unit for step counting; use sequences:
        D_seq = D_tokens / seq_len.
    TPP is still D_tokens / N.
    */
    const eta = Number(peak_lr);
    if (eta <= 0) return Number.NaN;

    const D_seq = Number(D_tokens) / Number(seq_len);
    const tau = powerlines_tau_opt(tokens_per_parameter(D_tokens, N), pl);
    return Number(B_seq) / (eta * D_seq * tau);
}

export function recommended_batch_target(Bopt_seq, Bcrit_seq, mode = "geomean") {
    /*
    Pick a target between Bopt and Bcrit.
      - geomean: balanced sweep center in log-space
      - opt: most compute-efficient
      - crit: most time-efficient before strong diminishing returns
    */
    if (mode === "opt") return Number(Bopt_seq);
    if (mode === "crit") return Number(Bcrit_seq);
    return Math.sqrt(Number(Bopt_seq) * Number(Bcrit_seq));
}

export function mup_scaled_lr(proxy_base_lr, proxy_width, target_width) {
    // Power Lines µP rule: eta_target = (d_proxy / d_target) * eta_proxy.
    if (isNil(proxy_base_lr) || isNil(proxy_width)) return null;
    return (Number(proxy_width) / Number(target_width)) * Number(proxy_base_lr);
}

// =====================
// Single-setting report
// =====================

function addFormattedCopies(rows, columnsToHumanFormat, scientificColumns = []) {
    return rows.map((row) => {
        const out = { ...row };

        for (const col of columnsToHumanFormat) {
            if (Object.prototype.hasOwnProperty.call(row, col)) {
                out[`${col}_fmt`] = isFiniteNumber(row[col]) ? human_num(row[col]) : "nan";
            }
        }

        for (const col of scientificColumns) {
            if (Object.prototype.hasOwnProperty.call(row, col)) {
                out[`${col}_fmt`] = isFiniteNumber(row[col]) ? formatScientific(row[col], 3) : "nan";
            }
        }

        return out;
    });
}

export function single_setting_report({
    D_tokens,
    N_params = null,
    n_layer = 8,
    n_embd = 512,
    n_head = 8,
    mlp_ratio = 4.0,
    seq_len = 1024,
    batch_size_sequences = 32,
    grad_accum = 96,
    num_gpus = 4,
    batch_size_is_per_gpu = false,
    peak_lr = 3e-4,
    proxy_width = null,
    proxy_base_lr = null,
} = {}) {
    const D = Number(D_tokens);
    if (!Number.isFinite(D) || D <= 0) {
        throw new Error("single_setting_report requires a positive finite D_tokens value.");
    }

    const N = resolve_N_params(N_params, n_layer, n_embd, mlp_ratio);
    const B_actual_seq = global_batch_sequences(batch_size_sequences, grad_accum, num_gpus, batch_size_is_per_gpu);
    const B_actual_tokens = B_actual_seq * Number(seq_len);
    const steps_actual = D / B_actual_tokens;
    const tpp = tokens_per_parameter(D, N);

    // Kaplan outputs
    const L_nd = kaplan_loss_ND(N, D);
    const D_bal = kaplan_balance_D_for_N(N);
    const N_bal = kaplan_balance_N_for_D(D);
    const Bcrit_kap_tokens = kaplan_Bcrit_tokens_from_loss(L_nd);
    const C_flops = training_flops(N, D);

    // Power Lines outputs
    const Bopt_seq = powerlines_Bopt_sequences(D);
    // Approximate by using planned D as Dmin. Exact effective-Dmin is computed too.
    const Bcrit_seq_approx = powerlines_Bcrit_sequences(D);
    const B_target_seq = recommended_batch_target(Bopt_seq, Bcrit_seq_approx);
    const Dmin_eff = infer_Dmin_from_planned_tokens(D, B_actual_seq);
    const Bcrit_seq_eff = powerlines_Bcrit_sequences(Dmin_eff);
    const extra_data_factor_actual = D / Dmin_eff;
    const tau = powerlines_tau_opt(tpp);

    const lr_mup = mup_scaled_lr(proxy_base_lr, proxy_width, n_embd);
    const eta_for_lambda = lr_mup === null ? Number(peak_lr) : lr_mup;

    const lambdaTable = addFormattedCopies(
        [
            ["min / Bopt", Bopt_seq],
            ["target / sqrt(Bopt*Bcrit)", B_target_seq],
            ["max / Bcrit", Bcrit_seq_approx],
            ["actual", B_actual_seq],
        ].map(([label, bseq]) => ({
            batch_choice: label,
            B_seq: bseq,
            B_tokens: bseq * Number(seq_len),
            steps_for_D: D / (bseq * Number(seq_len)),
            lambda_opt_at_peak_lr: powerlines_lambda_opt(bseq, D, N, eta_for_lambda, seq_len),
        })),
        ["B_seq", "B_tokens", "steps_for_D"],
        ["lambda_opt_at_peak_lr"],
    );

    const batchStatus =
        Bopt_seq <= B_actual_seq && B_actual_seq <= Bcrit_seq_approx
            ? "OK"
            : B_actual_seq < Bopt_seq
                ? "below Bopt"
                : "above Bcrit";

    const summary = [
        { quantity: "D_tokens", value: D, formatted: human_num(D) },
        { quantity: "N_params_non_embedding", value: N, formatted: human_num(N) },
        { quantity: "architecture", value: Number.NaN, formatted: `L=${n_layer}, d=${n_embd}, heads=${n_head}` },
        { quantity: "TPP = D/N", value: tpp, formatted: formatSig(tpp, 3) },
        { quantity: "Kaplan L(N,D) predicted loss", value: L_nd, formatted: formatSig(L_nd, 4) },
        { quantity: "Kaplan balanced D for this N", value: D_bal, formatted: human_num(D_bal) },
        { quantity: "Kaplan balanced N for this D", value: N_bal, formatted: human_num(N_bal) },
        {
            quantity: "Power Lines / Chinchilla compute-optimal N ≈ D/20",
            value: D / PL.compute_optimal_tpp_rule,
            formatted: human_num(D / PL.compute_optimal_tpp_rule),
        },
        { quantity: "Power Lines tau_opt", value: tau, formatted: formatSig(tau, 4) },
        { quantity: "Training FLOPs ≈ 6ND", value: C_flops, formatted: human_num(C_flops) },
        {
            quantity: "Training compute PF-days",
            value: flops_to_pf_days(C_flops),
            formatted: formatSig(flops_to_pf_days(C_flops), 4),
        },
        { quantity: "Actual global B_seq", value: B_actual_seq, formatted: human_num(B_actual_seq) },
        { quantity: "Actual global B_tokens", value: B_actual_tokens, formatted: human_num(B_actual_tokens) },
        { quantity: "Actual optimizer steps", value: steps_actual, formatted: human_num(steps_actual) },
        { quantity: "Kaplan Bcrit from L(N,D), tokens", value: Bcrit_kap_tokens, formatted: human_num(Bcrit_kap_tokens) },
        { quantity: "Power Lines Bopt, sequences", value: Bopt_seq, formatted: human_num(Bopt_seq) },
        {
            quantity: "Power Lines Bcrit, sequences, approx Dmin=D",
            value: Bcrit_seq_approx,
            formatted: human_num(Bcrit_seq_approx),
        },
        { quantity: "Power Lines effective Dmin for actual B", value: Dmin_eff, formatted: human_num(Dmin_eff) },
        { quantity: "Power Lines Bcrit at effective Dmin", value: Bcrit_seq_eff, formatted: human_num(Bcrit_seq_eff) },
        {
            quantity: "Actual extra-data factor vs Dmin",
            value: extra_data_factor_actual,
            formatted: `${formatSig(extra_data_factor_actual, 3)}x`,
        },
        { quantity: "Peak LR used for lambda", value: eta_for_lambda, formatted: formatSig(eta_for_lambda, 3) },
    ];

    const ranges = addFormattedCopies(
        [
            {
                parameter: "batch_size_sequences",
                minimum: Bopt_seq,
                recommended_center: B_target_seq,
                maximum_critical: Bcrit_seq_approx,
                actual: B_actual_seq,
                status: batchStatus,
            },
            {
                parameter: "batch_size_tokens",
                minimum: Bopt_seq * Number(seq_len),
                recommended_center: B_target_seq * Number(seq_len),
                maximum_critical: Bcrit_seq_approx * Number(seq_len),
                actual: B_actual_tokens,
                status: batchStatus,
            },
            {
                parameter: "tau",
                minimum: tau / 2,
                recommended_center: tau,
                maximum_critical: tau * 2,
                actual: Number.NaN,
                status: "sweep bracket, not a hard law",
            },
            {
                parameter: "lambda_at_Bopt",
                minimum: powerlines_lambda_opt(Bopt_seq / 2, D, N, eta_for_lambda, seq_len),
                recommended_center: powerlines_lambda_opt(Bopt_seq, D, N, eta_for_lambda, seq_len),
                maximum_critical: powerlines_lambda_opt(2 * Bopt_seq, D, N, eta_for_lambda, seq_len),
                actual: Number.NaN,
                status: "scales linearly with chosen B and inversely with LR",
            },
            {
                parameter: "N_params_for_D",
                minimum: N_bal / 2,
                recommended_center: N_bal,
                maximum_critical: N_bal * 2,
                actual: N,
                status: "Kaplan balance bracket; compare also D/20 rule",
            },
            {
                parameter: "D_tokens_for_N",
                minimum: D_bal / 2,
                recommended_center: D_bal,
                maximum_critical: D_bal * 2,
                actual: D,
                status: "Kaplan balance bracket; >center means overtrained/data-rich",
            },
        ],
        ["minimum", "recommended_center", "maximum_critical", "actual"],
    );

    return {
        summary,
        ranges,
        lambdaTable,
        raw: {
            D_tokens: D,
            N_params_non_embedding: N,
            B_actual_seq,
            B_actual_tokens,
            steps_actual,
            tpp,
            L_nd,
            D_bal,
            N_bal,
            Bcrit_kap_tokens,
            C_flops,
            Bopt_seq,
            Bcrit_seq_approx,
            B_target_seq,
            Dmin_eff,
            Bcrit_seq_eff,
            extra_data_factor_actual,
            tau,
            eta_for_lambda,
            lr_mup,
        },
    };
}

// Optional convenience wrapper with a shorter, UI-friendly name.
export const getScalingLawReport = single_setting_report;

// Optional helper for building log-scale sliders.
export function logSliderValue(log10Value) {
    return 10 ** Number(log10Value);
}

export function toLog10SliderValue(value) {
    return Math.log10(Number(value));
}
