# Kaplan-Style Comparison Diagnostics

This document summarizes the ten main comparisons performed by the Kaplan-style scaling-law comparison page.

## 1. Measured Loss vs. Kaplan Expected Loss

For every imported CSV row, the page extracts:

```text
N = num_params
D = num_train_tokens
measured loss = min_saved_val_loss
```

It then computes:

```text
Kaplan loss = kaplan_loss_ND(N, D)
```

The page overlays your measured validation loss against Kaplan's predicted loss at the same model size `N` and dataset size `D`.

This answers:

> For this exact model size and dataset size, is my validation loss above, below, or shaped similarly to Kaplan's expectation?

The page supports three comparison modes:

```text
Raw Kaplan L(N, D)
Kaplan + b
a·Kaplan + b
```

The calibrated modes are useful because your particleGPT loss scale probably will not exactly match OpenAI language-model loss. They help separate absolute loss-scale mismatch from shape mismatch.

## 2. Kaplan Residuals

The page computes:

```text
residual = measured_loss - calibrated_kaplan_loss
residual_pct = 100 * residual / |calibrated_kaplan_loss|
```

Interpretation:

```text
positive residual  => your run is worse than Kaplan predicts
negative residual  => your run is better than Kaplan predicts
flat residual      => same shape as Kaplan, but possibly shifted
sloped residual    => your scaling shape differs from Kaplan
```

This is one of the most important diagnostics. If the residuals are mostly flat, then your curve may have a Kaplan-like shape even if the absolute loss values differ. If the residuals slope strongly with `N`, `D`, or compute, your scaling behavior is not matching Kaplan.

## 3. Parity Plot

The parity plot uses:

```text
x = Kaplan predicted loss
y = measured validation loss
```

The dashed diagonal corresponds to perfect agreement.

Interpretation:

```text
points near diagonal          => close agreement with Kaplan
line parallel to diagonal     => Kaplan-like shape with a constant offset
curved or fanned-out points   => different scaling behavior
separation by dataset size    => dataset scaling differs from Kaplan
```

This plot is useful because it directly shows whether Kaplan's predicted ordering of runs matches your measured ordering.

## 4. D/N Regime Comparison

The page computes:

```text
D/N = num_train_tokens / num_params
```

It then classifies each point roughly as:

```text
D/N < 8      data-limited
8–15         slightly data-limited
15–30        near D/N ≈ 20
30–80        model-limited / data-rich
>80          very model-limited / very data-rich
```

This checks whether your sweep actually covers the useful Kaplan-style tradeoff region.

For example:

```text
low D/N   => model is large relative to dataset; add data
high D/N  => dataset is large relative to model; add model size
D/N ≈ 20  => near the rough Chinchilla-like balance heuristic
```

This does not prove optimality, but it gives a quick regime diagnosis.

## 5. Slope Comparison

For each curve, the page fits simple trends versus the selected x-axis, usually `N`, `D`, or compute.

Conceptually, it compares:

```text
measured_slope = slope of measured loss vs log10(x)
kaplan_slope   = slope of Kaplan loss vs log10(x)
slope_ratio    = measured_slope / kaplan_slope
```

This answers:

> As I scale model size, dataset size, or compute, does my loss improve as steeply as Kaplan expects?

Interpretation:

```text
slope_ratio ≈ 1       measured curve improves like Kaplan
slope_ratio << 1      measured curve is too flat
slope_ratio > 1       measured curve improves faster than Kaplan
slope_ratio < 0       measured loss worsens as x increases
```

This is a shape comparison, not just a value comparison.

## 6. Gain Ratio

The gain ratio compares how much your curve improves across its range to how much Kaplan expected it to improve.

The page computes:

```text
observed_gain = first_measured_loss - last_measured_loss
expected_gain = first_kaplan_loss - last_kaplan_loss

gain_ratio = observed_gain / expected_gain
```

Interpretation:

```text
gain_ratio ≈ 1       your curve improves about as much as Kaplan expects
gain_ratio << 1      your curve is too flat
gain_ratio > 1       your curve improves faster than Kaplan expects
negative             loss got worse as x increased
```

This is probably the single most compact diagnostic for whether a curve is Kaplan-like.

## 7. Train/Validation Gap

If the CSV includes `min_saved_train_loss`, the page computes:

```text
train_val_gap = min_saved_val_loss - min_saved_train_loss
```

Interpretation:

```text
small gap       => train and validation are close
large gap       => possible overfitting or validation mismatch
growing gap     => scaling may be limited by generalization, not capacity
negative gap    => suspicious; check logging or data split
```

This helps distinguish a true scaling-law issue from an overfitting or dataset-split issue.

For example, if larger models have much lower training loss but validation loss barely improves, the issue may not be Kaplan-style scaling. It may be overfitting, dataset mismatch, insufficient regularization, or validation-set noise.

## 8. Coverage / Undertraining Check

If the CSV has enough information, the page estimates how much of the dataset was actually seen by the model.

A typical estimate is:

```text
tokens_per_update = batch_size * gradient_accumulation_steps * block_size * world_size
tokens_seen = iters_saved * tokens_per_update
coverage = tokens_seen / num_train_tokens
```

Interpretation:

```text
coverage < 1      less than one epoch; likely undertrained
coverage ≈ 1      approximately one pass through the data
coverage > 1      multiple passes through the data
```

This is important because Kaplan's `D` means training tokens used, not merely available dataset size. If `num_train_tokens` is the size of the dataset but the model only saw a fraction of it, then the comparison to Kaplan can be misleading.

## 9. Curve Verdicts

Each curve gets a rough verdict based on residuals, residual slope, and gain ratio.

Possible verdicts include:

```text
Kaplan-like
too flat
large-x runs lag
large-x runs beat Kaplan
offset / shape mismatch
needs more points
```

The goal is to summarize the plots into a fast interpretation.

Examples:

```text
Kaplan-like
```

means the curve has roughly the expected shape after calibration.

```text
too flat
```

means increasing `N`, `D`, or compute is not improving loss as much as Kaplan would predict.

```text
large-x runs lag
```

means the larger or more expensive runs are worse than expected, which can indicate undertraining, bad learning-rate schedules, optimization instability, or data bottlenecks.

## 10. Suggested Next Runs

The page proposes next runs using a rough `D/N ≈ 20` target.

The logic is approximately:

```text
if D/N < 15:
    suggest adding data, D ≈ 20N

if D/N > 50:
    suggest adding model size, N ≈ D/20

otherwise:
    suggest tuning or repeating suspicious points
```

So the suggestions are based on whether each run looks data-limited, model-limited, or near-balanced.

Examples:

```text
low D/N:
    add more training tokens for the same model size

high D/N:
    add larger models for the same dataset size

near D/N ≈ 20 but poor residuals:
    repeat runs, tune LR schedule, check validation set, check parameter counts, or train longer
```

The purpose is not to give a perfect optimal sweep automatically. It is to turn the Kaplan comparison into practical next actions.

## Summary

For each experimental point, compare measured validation loss to Kaplan's expected loss at the same `N` and `D`, then check whether the shape, slope, gain, residuals, `D/N` regime, and training coverage look Kaplan-like.

## Disclaimer

This document was generated with ChatGPT. The code was provided and it was asked to generate this documentation. I have read through it and it is accurate.
