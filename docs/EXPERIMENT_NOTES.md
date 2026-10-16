# Original experiment record

The following notes describe the supplied runs. See PROJECT_REVIEW.md for the independent audit and current limitations. Scientific results are preserved, not replaced with estimates. The smoothing comparison is historical and non-adaptive.

# ECG Research Project

**Robust ECG Arrhythmia Classification Under Adversarial Perturbations**

A research prototype for studying how a small 1D CNN trained on MIT-BIH heartbeats responds to gradient-based adversarial perturbations, and whether adversarial training can make it more robust without giving up too much clean performance. This is a signal-analysis and model-robustness project, not a diagnostic tool — see [Ethical / medical disclaimer](#16-ethical--medical-disclaimer).

---

## 1. Project overview

ECG Research Project trains a 1D convolutional network to classify single heartbeats from the MIT-BIH Arrhythmia Database as **Normal** or **Arrhythmia**, then studies how that classifier behaves under two gradient-based adversarial attacks (FGSM, PGD), and whether adversarial training closes the resulting robustness gap. Everything downstream — the CLI scripts and the dashboard — reads from the same artifact files, so every number in this document and in the app is the output of an actual run, not a hand-typed estimate. A secondary experiment ([§14](#14-multi-class-aami-experiment-secondary)) repeats the clean/adversarial evaluation on a finer-grained 4-class AAMI task instead of the binary one.

The system has two operating modes:

- **`demo`** — a small synthetic ECG fixture generated on the fly (`src/data/synthetic.py`), for exercising the full pipeline in seconds with no download. Numbers from this mode are pipeline sanity checks, not benchmark results.
- **`research`** — the real MIT-BIH Arrhythmia Database. All results reported below are from this mode.

## 2. Research question

> How vulnerable is an ECG arrhythmia classifier to small, constrained adversarial perturbations, and can adversarial training improve robustness without significantly degrading performance on clean ECG signals?

## 3. Why ECG classification robustness matters

Automated ECG classifiers are gradient-based models like any other image or audio classifier, and gradient-based models are known to be vulnerable to small, deliberately-constructed input perturbations. A perturbation that's minor by construction (bounded in amplitude, measured in the same units the signal is normalized to) can still flip a prediction. Studying *how much* perturbation is needed, *how* the attack navigates the loss landscape (single-step vs. iterative), and *whether* a standard defense recovers the correct prediction is a well-defined, measurable research question — independent of whether or how such a classifier would ever be deployed. That deployment question is out of scope here; see the disclaimer below.

## 4. Dataset

[MIT-BIH Arrhythmia Database](https://physionet.org/content/mitdb/1.0.0/) — 48 half-hour two-lead ambulatory ECG recordings at 360 Hz, with expert beat-by-beat annotations, accessed via [`wfdb`](https://github.com/MIT-LCP/wfdb-python).

The repository does not commit the database. `scripts/prepare_data.py --mode research` downloads all 48 records (~100 MB) via `wfdb.dl_database` into `data/raw/mitdb/` (gitignored). `scripts/prepare_data.py --mode demo` instead generates a small synthetic fixture into `data/demo/` (committed, since it's tiny and contains no real patient data).

Records **102, 104, 107, 217** are excluded from the primary experiment — they're predominantly paced rhythms, and following AAMI/de Chazal convention, paced-rhythm detection is a different problem from arrhythmia classification and would confound an inter-patient split.

## 5. ECG preprocessing

```
Record load (wfdb.rdrecord + rdann)
   ↓
Signal validation (fs == 360 Hz, length sane, no NaN/Inf)
   ↓
Baseline wander removal — two-stage median filter (200ms then 600ms window)
   ↓
Noise filtering — off by default (see below)
   ↓
R-peak / beat localization — expert annotations (real data) or ground-truth
   peak positions (synthetic demo data); a from-scratch simplified
   Pan-Tompkins detector exists independently for validation, not segmentation
   ↓
Heartbeat segmentation — fixed window: 90 samples before R, 160 after (250
   total, ≈0.69s @ 360Hz)
   ↓
Beat normalization — per-beat z-score (mean 0, std 1), clipped to ±6σ
   ↓
Label mapping — AAMI EC57 grouping, binary task
   ↓
Train / val / test split (record-wise)
   ↓
1D CNN
```

Each step exists for a specific, checkable reason:

- **Baseline wander removal** (`src/preprocessing/filtering.py`): a median filter with a ~200ms window is wide enough to suppress the QRS complex and P-wave; running a second, longer (~600ms) filter over that result further suppresses the T-wave, leaving an estimate of the slow drift alone. Subtracting it removes wander without distorting QRS morphology the way a naive high-pass filter can.
- **Noise filtering is off by default.** MIT-BIH is a curated, expert-verified benchmark; blindly low-pass filtering it isn't justified by anything in the data. An optional 45Hz low-pass exists behind `configs/dataset.yaml`'s `denoise.enabled` flag for a documented sensitivity check, not as a default step.
- **R-peak localization** uses the database's own expert annotations for the real pipeline — that's what they're for, and re-deriving them for segmentation would just add noise. A simplified Pan-Tompkins-style detector (`src/preprocessing/qrs_detect.py`: bandpass → derivative → square → moving-window integration → peak picking) exists separately, used for the synthetic demo path (which has no expert annotations) and as an independent annotation-vs-detector agreement check on the Signal Analysis dashboard page.
- **Segmentation window** (90 pre / 160 post) is a fixed config value (`configs/dataset.yaml`), not a magic number in code — beats whose window would run past a record's start/end are dropped rather than zero-padded, since padding would bake an artificial discontinuity into a handful of beats.
- **Per-beat z-score normalization** is also the reference scale the adversarial epsilon budget is defined in (see [Perturbation constraints](#10-adversarial-attack-methodology)).

## 6. Data-splitting strategy

**Record-wise, not beat-wise.** This original protocol is not fully subject-disjoint: validation record 201 and test record 202 belong to the same subject. The saved results must be interpreted with that model-selection leakage caveat. Splitting individual heartbeats randomly would put near-duplicate beats from the same patient/recording on both sides of the split — the model would partly memorize patient-specific morphology rather than learn generalizable arrhythmia patterns, inflating test metrics in a way that wouldn't hold on an unseen patient. This is a well-documented pitfall in MIT-BIH literature; inter-patient evaluation is known to be substantially harder (and more honest) than intra-patient/random splits.

The base partition is the **de Chazal et al. (2004) DS1/DS2 inter-patient split**, the standard reference partition for MIT-BIH beat classification:

- **DS1** (used for train + validation, 22 records): 101, 106, 108, 109, 112, 114, 115, 116, 118, 119, 122, 124, 201, 203, 205, 207, 208, 209, 215, 220, 223, 230
- **DS2** (test, held out entirely, 22 records): 100, 103, 105, 111, 113, 117, 121, 123, 200, 202, 210, 212, 213, 214, 219, 221, 222, 228, 231, 232, 233, 234

Within DS1, records **201, 203, 220, 230** are held out as validation (`configs/dataset.yaml`, `split.val_records`); the rest of DS1 is training. This list was verified against the actually-downloaded database rather than assumed correct from the literature reference alone. The identical split (and cache infrastructure) is reused for the multi-class experiment in §14 — only the label mapping differs.

**Measured class distribution** (binary task: Normal vs. Arrhythmia = S+V+F, Q/paced excluded):

| Split | Beats | Normal | Arrhythmia | Arrhythmia % |
|---|---|---|---|---|
| Train (18 records) | 41,754 | 37,478 | 4,276 | 10.2% |
| Validation (4 records) | 9,240 | 8,370 | 870 | 9.4% |
| Test / DS2 (22 records) | 49,685 | 44,240 | 5,445 | 11.0% |

Arrhythmia prevalence varies a lot by record (e.g. record 208 is 46.3% arrhythmia; records 115, 122, and 230 are close to 0%) — this is a real property of the database, not an artifact of the split, and it's part of why the 4-record validation set is a noisier estimator of generalization than the much larger DS2 test set. Record 203 in particular is a commonly-cited noisy/atypical recording in MIT-BIH literature; it sits in the validation set here, not the test set, so it doesn't affect the headline test numbers.

## 7. Class labeling and imbalance

AAMI EC57 groups the raw MIT-BIH annotation symbols into 5 superclasses (`configs/dataset.yaml`, `label_map`): **N** (normal + bundle-branch-block variants), **S** (supraventricular ectopic), **V** (ventricular ectopic), **F** (fusion), **Q** (paced/unknown). The primary binary task is **N vs. {S, V, F}**, with **Q excluded** — paced/unknown beats are a different, confounding problem, not "arrhythmia" in the sense this project studies.

Class weighting (`pos_weight` in `BCEWithLogitsLoss`, computed from the actual training split's class ratio, not hardcoded) is used instead of oversampling by default. Accuracy is reported but never treated as the headline metric — recall, precision, F1, and PR-AUC are emphasized throughout, since a classifier that simply predicts "Normal" for everything would already score >88% accuracy on this data while catching zero arrhythmias.

## 8. Model architecture

A small 1D CNN (`src/models/cnn1d.py`), **11,265 parameters**:

```
Input: (batch, 1, 250)
Conv1d(1→16, k=7)  → BatchNorm1d → ReLU → MaxPool1d(2)   → (16, 125)
Conv1d(16→32, k=5) → BatchNorm1d → ReLU → MaxPool1d(2)   → (32, 62)
Conv1d(32→64, k=3) → BatchNorm1d → ReLU                  → (64, 62)
GlobalAveragePool1d                                        → (64,)
Dense(64→32) → ReLU → Dropout(0.3)
Dense(32→1)  → logit
```

The third conv block skips pooling so global average pooling still has 62 timesteps of context to average over, rather than collapsing to a single point beforehand. Output is a raw logit; `BCEWithLogitsLoss` handles the sigmoid. The final layer's width is a `num_classes` parameter (default 1) — the multi-class experiment in §14 reuses this exact architecture with `num_classes=4` and no other change.

**Why 1D CNN and not 2D:** the signal is a single-channel time series, not an image — a 1D convolution's receptive field directly corresponds to a time window, which is the natural inductive bias for QRS/T-wave morphology. A 2D CNN would require reshaping the signal into an image-like grid (e.g. a spectrogram) and would learn a different, less direct representation.

**Secondary baseline** (`src/models/baseline_features.py`, `scripts/run_benchmark.py`): LogisticRegression and RandomForest (scikit-learn) on 10 hand-engineered per-beat features — amplitude statistics, skew/kurtosis, a crude QRS-width proxy, and Welch band power in three frequency ranges. This exists to answer one question: *does the CNN's learned waveform representation actually beat simple statistical/spectral features?* (see [Experimental results](#13-experimental-results) — yes, substantially).

## 9. Training procedure

`scripts/train.py` — a deterministic, config-driven training loop (`configs/training.yaml`):

- Seed 42 (`src/utils/seed.py`, applied to `random`/`numpy`/`torch`)
- Adam, lr=5e-4, weight_decay=1e-5
- `ReduceLROnPlateau` on validation F1 (patience 4, factor 0.5)
- Early stopping on validation F1 (patience 15), max 50 epochs
- Checkpointing: the best-validation-F1 epoch's weights are saved, along with the model config and a calibrated decision threshold (see below) — not necessarily the final epoch's weights
- Full per-epoch history (loss, all metrics) saved to `artifacts/models/{tag}_history.json`

**Decision threshold calibration.** With `pos_weight` reweighting the loss for class imbalance, the raw sigmoid output is no longer centered around 0.5 — a fixed 0.5 cutoff caused validation F1 to swing wildly epoch to epoch (0.14–0.48) even though ROC-AUC/PR-AUC stayed stable, because small shifts in the model's calibration moved thousands of borderline negatives across the fixed threshold. `src/evaluation/metrics.py:best_f1_threshold` calibrates the operating threshold on validation data (via `precision_recall_curve`, not a brute-force per-value sweep) each epoch; the threshold from the best epoch is frozen into the checkpoint and reused unchanged on test and adversarial data — it is never re-calibrated against data it's being evaluated on.

Training is a separate, explicit step from the dashboard — `python scripts/train.py` — the dashboard only ever loads an existing checkpoint.

## 10. Adversarial attack methodology

**Perturbation budget.** Beats are z-score normalized (std ≈ 1 by construction); epsilon is expressed in that same scale, so it's interpretable as "a fraction of one signal standard deviation" rather than an arbitrary number. Checked against the real processed test set before use: mean absolute amplitude is ≈0.58, median per-beat R-peak (max) amplitude is ≈5.0, with the 10th/90th percentile spanning roughly 3.5–6.0 (the normalization clip bound). The sweep **{0, 0.02, 0.05, 0.1, 0.15, 0.25, 0.4, 0.6}** spans from a small fraction of even the quiet baseline segments to a change large enough to be clearly visible away from the R-peak.

### FGSM (`src/attacks/fgsm.py`)

Fast Gradient Sign Method (Goodfellow et al., 2015) — a single step in the sign of the loss gradient with respect to the input:

```
x_adv = clip(x + epsilon · sign(∇ₓ Loss(model(x), y)), clip_min, clip_max)
```

Untargeted by default (maximizes loss w.r.t. the true label); a `targeted` flag is supported, though for this binary task targeted-toward-the-other-class and untargeted-away-from-the-true-class are mathematically identical (`BCEWithLogitsLoss`'s gradient sign doesn't depend on which of the two always-positive quantities, `sigmoid(logits)` or its complement, is differentiated through) — targeted attacks only add information once there's more than two classes to choose a target from (as in §14's multi-class model).

### PGD (`src/attacks/pgd.py`)

Projected Gradient Descent (Madry et al., 2018) — iterated FGSM-style steps, with the accumulated perturbation re-projected onto the epsilon L∞ ball around the original input after every step:

```
for i in range(iterations):
    step = x_adv + alpha · sign(∇ₓ Loss(model(x_adv), y))
    delta = clip(step - x_original, -epsilon, epsilon)   # project into the L∞ ball
    x_adv = clip(x_original + delta, clip_min, clip_max)
```

Default: 10 iterations, step size `alpha = 0.25 · epsilon`, random start within the epsilon ball. A single PGD iteration with `alpha == epsilon` and no random start is, by construction, exactly one FGSM step (`tests/test_pgd.py::test_pgd_single_step_matches_fgsm`).

Both attacks take `loss_fn` as a parameter rather than hardcoding `BCEWithLogitsLoss`, specifically so they work unchanged against the multi-class model with `CrossEntropyLoss` — verified directly, not just assumed (`test_fgsm.py::test_fgsm_works_against_a_multiclass_model_with_cross_entropy`, `test_pgd.py::test_pgd_works_against_a_multiclass_model_with_cross_entropy`).

**Attack success rate** (untargeted definition, used throughout — never mixed with a targeted definition): the percentage of samples the model classified correctly *before* the attack whose predicted class changed *after* it. Samples already misclassified are excluded from the denominator, since there's nothing for the attack to flip.

## 11. Adversarial training

Primary defense (`src/defenses/adversarial_training.py`): each training batch is a configurable clean/adversarial mix (default 50/50), with adversarial examples generated on the fly against the model's *current* weights every step, using PGD (a stronger, more standard adversary for this than FGSM — Madry et al.). The resulting **defended model is evaluated against both FGSM and PGD** at test time, which doubles as a check on whether training against one attack generalizes to the other.

Strict separation is maintained throughout: adversarial examples for training come only from the training split; the clean test set and adversarial test set are both built from DS2, which the defended model never saw during training in any form.

**Secondary defense — input smoothing.** `src/defenses/input_smoothing.py` applies a small moving-average filter to the input at inference time, no retraining involved (`scripts/run_benchmark.py --secondary-defense`, `configs/experiments.yaml`'s `defense_secondary`). Applied to the *standard* (non-adversarially-trained) model against PGD:

| epsilon | No defense | + input smoothing |
|---|---|---|
| 0.10 | 78.7% | 83.8% |
| 0.25 | 59.7% | 83.4% |
| 0.40 | **3.9%** | **87.3%** |

These are **non-adaptive transfer results**: the attack was crafted against the unsmoothed model, then smoothing was applied afterward. They do not establish white-box robustness of the smoothing pipeline or justify ranking it above adversarial training. The revised evaluator differentiates through smoothing and includes epsilon=0; a full research rerun remains pending.

## 12. Evaluation metrics

`src/evaluation/metrics.py` / `src/evaluation/robustness.py`, computed once and consumed by both this document's numbers and the dashboard — no metric is computed two different ways in two different places:

- **Clean:** accuracy, precision, recall, F1, PR-AUC, ROC-AUC, confusion matrix
- **Adversarial:** robust accuracy (accuracy on adversarial examples), adversarial F1/precision/recall, attack success rate, plus per-attack perturbation statistics (L∞, L2, RMS, SNR in dB)
- **Multi-class variant** (§14): accuracy, macro-F1, weighted-F1, and a per-class precision/recall/F1/support breakdown (`compute_multiclass_metrics`) — there's no single positive class to define precision/recall/AUC against once there are more than two mutually-exclusive labels.

## 13. Experimental results

The static tables below summarize `artifacts/experiments/*/clean_metrics.json` and `robustness_curve.csv` — generated by `scripts/run_benchmark.py --mode research`, not automatically refreshed when experiments are rerun.

### Experiment 1 — clean performance (DS2 test set, 49,685 beats)

| Model | Accuracy | Precision | Recall | F1 | PR-AUC | ROC-AUC |
|---|---|---|---|---|---|---|
| **1D CNN (standard)** | 93.3% | 74.6% | 58.2% | **65.4%** | 69.8% | 88.6% |
| 1D CNN (defended) | 93.5% | 89.5% | 45.8% | 60.6% | 65.0% | 88.5% |
| Logistic Regression (10 features) | — | — | — | 23.9% | — | — |
| Random Forest (10 features) | — | — | — | 29.9% | — | — |

The CNN's learned waveform representation clearly outperforms both hand-engineered-feature baselines — more than double the F1 of the stronger (Random Forest) baseline — answering this project's secondary research question directly.

### Experiments 2/3 — FGSM vs. PGD (standard model, DS2 test set)

| epsilon | FGSM robust acc. | FGSM attack success | PGD robust acc. | PGD attack success |
|---|---|---|---|---|
| 0.02 | 84.9% | 9.0% | 79.0% | 15.2% |
| 0.05 | 88.6% | 6.9% | 77.3% | 17.1% |
| 0.10 | 89.0% | 6.8% | 78.7% | 15.6% |
| 0.15 | 89.0% | 6.8% | 78.5% | 15.9% |
| 0.25 | 89.0% | 6.8% | 59.5% | 36.2% |
| 0.40 | 89.0% | 6.8% | **3.9%** | **95.9%** |
| 0.60 | 89.0% | 6.8% | **0.0%** | **100%** |

**PGD is dramatically stronger than FGSM at the same epsilon** — by epsilon 0.4, PGD reaches 95.9% attack success (nearly every correctly-classified beat gets flipped) while FGSM plateaus around 6.8–9%, confirming that a single gradient-sign step is a much weaker adversary than an iterative one navigating the same constrained region. Notably, adversarial F1 (the model's ability to still catch the minority arrhythmia class) collapses toward 0 much faster than either accuracy or attack-success-rate would suggest on their own — even a modest perturbation (epsilon=0.02) is enough to suppress most positive predictions, which is exactly why F1/recall are reported alongside attack success rate rather than either one alone. SNR at epsilon=0.1 is still ≈20–24dB (`robustness_curve.csv`) — a small perturbation in a precise, measured sense, not an assumption.

### Experiments 4/5 — adversarial training (standard vs. defended CNN)

The defended CNN was trained with PGD-based adversarial training at epsilon=0.1, 50/50 clean/adversarial batch mix (`configs/experiments.yaml`, `adversarial_training`), then evaluated — like the standard model — against **both** FGSM and PGD across the full epsilon sweep, to check whether training against one attack generalizes to the other.

**PGD robust accuracy, standard vs. defended:**

| epsilon | Standard | Defended | Standard attack success | Defended attack success |
|---|---|---|---|---|
| 0.02 | 79.0% | 93.1% | 15.2% | 0.4% |
| 0.05 | 77.3% | 92.1% | 17.1% | 1.5% |
| 0.10 | 78.7% | 89.9% | 15.6% | 3.8% |
| 0.15 | 78.5% | 86.0% | 15.9% | 8.0% |
| 0.25 | 59.5% | 74.3% | 36.2% | 20.5% |
| 0.40 | **3.9%** | **56.6%** | **95.9%** | **39.4%** |
| 0.60 | **0.0%** | **37.1%** | **100%** | **60.3%** |

This is the central result. At small epsilons the standard model was already fairly PGD-resistant in absolute accuracy terms (though still measurably weaker than the defended model), but at epsilon ≥ 0.4 — where the standard model **completely collapses** (0–3.9% robust accuracy, 96–100% attack success) — the defended model **retains 37–57% robust accuracy** and cuts attack success roughly in half or better. Training against PGD generalizes to FGSM too: the defended model's FGSM attack success stays under 23% across the whole sweep (`artifacts/experiments/defended_cnn/robustness_curve.csv`), vs. the standard model, which was already fairly FGSM-resistant in absolute terms but still measurably weaker than the defended model at every epsilon.

**Clean-performance tradeoff** (see the Experiment 1 table above): clean F1 drops from 65.4% to 60.6% (−4.8pp, ≈−7.3% relative) — not free, as expected, but the shift isn't a uniform decline: precision *improves* substantially (74.6%→89.5%) while recall drops (58.2%→45.8%). Adversarial training made this model more conservative — more confident when it does predict Arrhythmia, but missing more of the minority class. Accuracy and ROC-AUC are essentially unchanged (93.3%→93.5%, 88.6%→88.5%), so the tradeoff shows up specifically in the precision/recall balance, not overall discriminative ability.

**Answering the research question:** yes — on this model and this task, adversarial training substantially improves robustness against both attacks, especially at larger perturbation budgets where the undefended model fails completely, at a real but modest clean-F1 cost concentrated in reduced recall rather than a broad performance collapse.

## 14. Multi-class AAMI experiment (secondary)

A secondary experiment on the same data, splits, and model architecture, replacing the binary Normal-vs-Arrhythmia task with the full 4-class AAMI grouping (N/S/V/F; Q still excluded, for the same reason as the binary task — see §7). Kept deliberately lean relative to the primary binary experiment: its own training script (`scripts/train_multiclass.py`), config (`configs/training_multiclass.yaml`), and evaluation script (`scripts/run_benchmark_multiclass.py`), reusing the same preprocessing, record-wise splitting, model architecture, and attack code — but not integrated into the dashboard, and without its own adversarial-training run (see [Future work](#23-future-work)).

**What differs from the binary setup:** `CrossEntropyLoss` with inverse-frequency class weights (`compute_class_weights`, the multi-class analog of `pos_weight`) instead of `BCEWithLogitsLoss`; argmax over 4 logits instead of a calibrated decision threshold (there's no single "positive class" or threshold once labels are more than two mutually-exclusive options); macro-F1 (the unweighted average of each class's F1) as the model-selection/early-stopping metric instead of binary F1, for the same reason binary F1 was preferred over accuracy — it doesn't let the majority class hide poor minority-class performance. The model itself is identical in every other respect: 11,364 parameters (vs. the binary model's 11,265 — the only difference is the output layer's width), same seed, same record-wise DS1/DS2 split.

**Measured class distribution:**

| Split | N | S | V | F | Total |
|---|---|---|---|---|---|
| Train | 37,478 (89.8%) | 720 (1.7%) | 3,145 (7.5%) | 411 (1.0%) | 41,754 |
| Validation | 8,370 (90.6%) | 224 (2.4%) | 643 (7.0%) | **3 (0.03%)** | 9,240 |
| Test / DS2 | 44,240 (89.0%) | 1,837 (3.7%) | 3,220 (6.5%) | 388 (0.8%) | 49,685 |

F has only 411 training examples and **3 in the entire validation set** — validation macro-F1 for this experiment is dominated by essentially random noise on the F component, a genuine limitation of splitting an already-rare AAMI class across a record-wise inter-patient partition (see [Limitations](#15-limitations)).

### Clean performance (DS2 test set, 49,685 beats)

| Metric | Value |
|---|---|
| Accuracy | 68.5% |
| Macro-F1 | 35.5% |
| Weighted-F1 | 75.8% |

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| N | 94.4% | 70.6% | 80.8% | 44,240 |
| S | 3.0% | 10.8% | 4.7% | 1,837 |
| V | 43.6% | 79.7% | 56.4% | 3,220 |
| F | 0.1% | 1.0% | 0.2% | 388 |

The model learns N and V reasonably — V's recall is actually *higher* than N's, at the cost of V's precision, since the class-weighted loss trades some N confidence for V sensitivity — but **essentially fails on S and F**. This isn't a training bug: it's the direct, expected consequence of S and F having far too few examples (720 and 411 respectively) for this small model to learn a reliable decision boundary, even with aggressive inverse-frequency class weighting. Notably, overall accuracy (68.5%) is *lower* than the ~89% a trivial "always predict N" classifier would score — the class weighting deliberately trades away that trivial accuracy in an attempt at minority-class sensitivity, and the trade only partly pays off (V) while mostly failing (S, F).

### Adversarial vulnerability (PGD, standard AAMI model, DS2 test set)

| epsilon | Robust accuracy | Robust macro-F1 | Attack success rate |
|---|---|---|---|
| 0.00 | 68.5% | 35.5% | 0.0% |
| 0.01 | 40.8% | 25.7% | 40.3% |
| 0.02 | 19.3% | 15.6% | 71.8% |
| 0.05 | 0.5% | 0.3% | 99.3% |
| 0.10 | 0.0% | 0.0% | 100% |
| 0.25 | 0.0% | 0.0% | 100% |
| 0.40 | 0.0% | 0.0% | 100% |

This isn't just a smaller-epsilon-number artifact: at epsilon=0.02, PGD's measured perturbation is essentially the same size for both models by SNR (36.1dB for the binary model vs. 35.3dB here, `robustness_curve.csv` in both experiment folders) — the same-magnitude perturbation, by the actual measurement in §10, produces 15.2% attack success against the binary model and 71.8% against this one.

**The multi-class model is dramatically more fragile than the binary one.** The binary standard model needed epsilon≈0.4 for PGD to reach ~96% attack success (§13); this 4-class model reaches 99.3% by epsilon=0.05 — roughly an order of magnitude smaller perturbation budget for a comparably complete collapse. Two plausible, non-exclusive explanations: (1) with 4 classes there are 3 possible "wrong" outcomes instead of 1, so there are more nearby decision boundaries for a single gradient step to cross; (2) the clean model is already considerably less confident/separated (68.5% accuracy vs. the binary model's 93.3%), and a model with weaker margins is generally easier to perturb across them. Whether adversarial training would close this much larger robustness gap as effectively as it did for the binary task (§13, Experiments 4/5) is an open question — not run for this secondary experiment, see [Future work](#23-future-work).

## 15. Limitations

- **Small validation set.** DS1's 4 held-out validation records (9,240 beats) are a much noisier estimator of generalization than the 22-record, 49,685-beat DS2 test set, and their arrhythmia prevalence is bimodal (two records >15%, two near 0%) rather than representative of DS1's average. Model selection (early stopping, threshold calibration) was done on this noisier set by design — deliberately not re-tuned further to avoid overfitting design choices to a small held-out set. This is much more severe for the multi-class experiment's F class, which has only 3 validation examples total.
- **Multi-class S/F classes lack sufficient training data.** 720 and 411 training examples respectively is not enough for this model to learn either class reliably (§14) — a finding about data scarcity, not an implementation defect.
- **CPU-only, single seed.** Results reflect one training run at one seed; no multi-seed variance estimate is reported.
- **Attacks assume white-box gradient access.** Both FGSM and PGD require access to the model's gradients — this studies a worst-case, not a realistic black-box attack surface.
- **Input smoothing's own clean-accuracy cost isn't measured.** The secondary defense comparison ([§11](#11-adversarial-training)) reports its adversarial-accuracy recovery but not how much smoothing degrades *clean* accuracy on the standard model — an incomplete tradeoff picture, unlike the primary defense's fully-reported tradeoff.
- **No adversarial training for the multi-class model.** §14 reports the multi-class model's clean performance and undefended vulnerability only; whether the same PGD-based adversarial training that helped the binary model would help here is untested.
- **Not a deployable or diagnostic system** — see below.

## 16. Ethical / medical disclaimer

**Research prototype. Results are intended for model evaluation and experimentation, not clinical diagnosis.** This project makes no claims of medical diagnosis, clinical validation, patient monitoring, regulatory approval, or clinical-grade performance, and is not a replacement for a physician or any diagnostic device. It is a study of a classifier's behavior under adversarial perturbation, using a public research database, for portfolio and educational purposes.

---

## 17. Installation

```bash
python -m venv .venv --system-site-packages
.venv\Scripts\python.exe -m pip install -e ".[dev]"
```

(`--system-site-packages` is optional — it just avoids re-downloading torch/numpy/scipy/pandas/scikit-learn/matplotlib if they're already installed system-wide. A plain `python -m venv .venv` works too, it just installs everything from scratch.)

## 18. Dataset setup

```bash
# fast, no internet required -- synthetic fixture for pipeline verification
.venv\Scripts\python.exe scripts\prepare_data.py --mode demo

# real MIT-BIH download (~100MB) -- required for the numbers in this README
.venv\Scripts\python.exe scripts\prepare_data.py --mode research
```

## 19. Training

```bash
.venv\Scripts\python.exe scripts\train.py --mode research --tag baseline_cnn
.venv\Scripts\python.exe scripts\train.py --mode research --tag defended_cnn --defense adversarial_training

# secondary multi-class experiment (see §14)
.venv\Scripts\python.exe scripts\train_multiclass.py --mode research --tag aami_cnn
```

## 20. Benchmarking

```bash
.venv\Scripts\python.exe scripts\run_benchmark.py --mode research --tag baseline_cnn
.venv\Scripts\python.exe scripts\run_benchmark.py --mode research --tag defended_cnn
.venv\Scripts\python.exe scripts\generate_attacks.py --mode research --tag baseline_cnn --attack pgd --epsilon 0.15

# secondary multi-class experiment
.venv\Scripts\python.exe scripts\run_benchmark_multiclass.py --mode research --tag aami_cnn
```

## 21. Running the dashboard

```bash
.venv\Scripts\python.exe -m streamlit run app\main.py
```

The dashboard only ever loads existing artifacts from `artifacts/` — it never trains or benchmarks anything itself beyond a single on-demand attack in the Adversarial Lab. (The multi-class experiment is script-only and isn't wired into the dashboard pages.)

## 22. Project structure

```
ecg-research-project/
├── app/                  Streamlit dashboard (main.py, pages/, components/, charts/, styles/)
├── src/
│   ├── data/             MIT-BIH + synthetic loaders, record-wise splitting, caching
│   ├── preprocessing/     filtering, QRS detection, segmentation, normalization
│   ├── models/            1D CNN, checkpoint loading, classical baseline features
│   ├── attacks/            FGSM, PGD
│   ├── defenses/           adversarial training, input smoothing
│   ├── evaluation/         metrics (binary + multi-class), robustness sweep, prediction helpers
│   ├── explainability/     saliency, integrated gradients
│   └── utils/              config loading, seeding, device selection
├── scripts/               prepare_data.py, train.py, generate_attacks.py, run_benchmark.py,
│                          train_multiclass.py, run_benchmark_multiclass.py (§14)
├── configs/               dataset.yaml, training.yaml, experiments.yaml, training_multiclass.yaml
├── tests/                 unit + regression tests (pytest)
├── artifacts/             trained models, experiment results (real, checked into git)
└── data/                  raw/, processed/ (gitignored) and demo/ (committed fixture)
```

## 23. Future work

- Adversarial training for the multi-class AAMI model, to see whether it closes the much larger robustness gap found in §14 as effectively as it did for the binary task.
- More training data (or targeted oversampling) for the AAMI S and F classes — their near-total lack of support (720 and 411 training examples) is the likely cause of their near-zero F1 in §14, not a modeling failure.
- Cross-epsilon generalization: does a model adversarially trained at one epsilon hold up across the whole sweep, or specifically around its training epsilon?
- Measure input smoothing's own clean-accuracy cost, to make the secondary-defense comparison in [§11](#11-adversarial-training) a complete tradeoff rather than a one-sided robustness number.
