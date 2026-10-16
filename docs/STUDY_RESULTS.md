# Study results — ECG Research Project V2

Study directory: `studies/v2_20261016`. All results below were computed from the source and saved checkpoints included in this release.

## Protocol and selection

The split excludes shared-subject record 201, retains 202 in test, and separates 18 training, 3 validation, and 22 test records. Training/validation/test contain 41,754 / 7,277 / 49,685 beats. An always-N prediction reaches 89.04% test accuracy. Expert annotations supply peak positions for extraction and past-RR features; this is an offline classifier experiment.

Four variants were trained with seeds 17, 42, and 73. Checkpoints and binary thresholds were selected by validation F1; the four-class head uses validation macro-F1. The selected seed is the one with the highest validation score, not the best test result. The architecture, split, seeds, selection rules, and attack settings were fixed for this rerun. The test database has been inspected during this project; these results are not an untouched external validation cohort.

## Across training seeds

Entries are mean ± sample standard deviation across three seeds, not confidence intervals or significance tests.

| Variant | F1 | Recall | Average precision |
|---|---|---|---|
| Waveform CNN | 61.79% ± 3.87 pp | 47.21% ± 4.57 pp | 70.43% ± 2.40 pp |
| Waveform + past RR | 64.20% ± 4.47 pp | 51.32% ± 10.09 pp | 78.40% ± 4.14 pp |
| Context + adversarial training | 53.21% ± 6.58 pp | 36.88% ± 6.66 pp | 79.27% ± 4.68 pp |

Context improves average ranking performance, but F1 varies substantially across seeds. The selected context model is not the best test-F1 context seed and does not beat the selected waveform-only model on F1. Do not substitute test-best seeds into the dashboard.

## Validation-selected models

| Variant | Selected run | F1 | Recall | Precision | AP | Threshold |
|---|---|---|---|---|---|---|
| Waveform CNN | v2_morphology_s17 | 65.99% | 52.19% | 89.71% | 72.65% | 0.669708 |
| Waveform + past RR | v2_context_s42 | 62.26% | 45.90% | 96.79% | 83.18% | 0.998720 |
| Context + adversarial training | v2_context_defended_s42 | 52.16% | 35.56% | 97.83% | 82.83% | 0.975649 |

The selected context model misses 2,946 of 5,445 arrhythmic test beats. Its high threshold and class-weighted training do not turn its sigmoid scores into calibrated probabilities. Report recall and precision alongside F1 and accuracy.

## Variation across subjects

2,000 bootstrap repetitions resample whole subjects with replacement. These 95% percentile intervals condition on the selected model and this database. They do not include every source of development uncertainty or establish clinical reliability.

| Variant | Metric | Point | 95% subject-bootstrap interval |
|---|---|---|---|
| Waveform CNN | f1 | 65.99% | 49.87%–79.11% |
| Waveform CNN | recall | 52.19% | 36.46%–69.19% |
| Waveform CNN | precision | 89.71% | 73.64%–96.57% |
| Waveform CNN | accuracy | 94.10% | 89.68%–97.24% |
| Waveform + past RR | f1 | 62.26% | 35.57%–82.02% |
| Waveform + past RR | recall | 45.90% | 22.18%–71.09% |
| Waveform + past RR | precision | 96.79% | 85.91%–99.69% |
| Waveform + past RR | accuracy | 93.90% | 88.09%–97.72% |
| Context + adversarial training | f1 | 52.16% | 30.41%–74.36% |
| Context + adversarial training | recall | 35.56% | 18.02%–59.83% |
| Context + adversarial training | precision | 97.83% | 89.79%–100.00% |
| Context + adversarial training | accuracy | 92.85% | 86.43%–97.42% |

## Full-test adaptive attacks

All rows below use 49,685 test beats. At epsilon 0.1, PGD uses 20 steps, step size 0.01, one clean start plus two random starts, and an additional 20-step BCE-objective pass. Each example retains its worst threshold-adjusted margin across candidates. The smoothed pipeline differentiates through a five-sample moving average and retains its original validation-selected threshold. Epsilon zero is each pipeline's own clean baseline.

| Pipeline | Epsilon | Accuracy under attack | F1 | Recall | Success on clean-correct beats |
|---|---|---|---|---|---|
| morphology | 0.0 | 94.10% | 65.99% | 52.19% | 0.00% |
| morphology | 0.1 | 46.95% | 0.14% | 0.35% | 50.11% |
| context | 0.0 | 93.90% | 62.26% | 45.90% | 0.00% |
| context | 0.1 | 43.86% | 0.00% | 0.00% | 53.29% |
| context_defended | 0.0 | 92.85% | 52.16% | 35.56% | 0.00% |
| context_defended | 0.1 | 90.20% | 24.39% | 14.42% | 2.85% |
| context_smoothed | 0.0 | 94.23% | 65.83% | 50.76% | 0.00% |
| context_smoothed | 0.1 | 76.73% | 8.85% | 10.30% | 18.57% |

This is a waveform-only L∞ threat model in normalized units. Expert peak positions and RR features stay fixed; waveform values remain within ±6. Timing attacks, peak-detection failures, and physiological plausibility are outside this evaluation. Finite attack results do not certify robustness.

## Separate strength audit

A fixed 64 beats per test record, 1,408 total, were selected without labels or predictions. This record-balanced population differs from the full beat-weighted test set. The audit also covers epsilon 0, 0.05, 0.1, 0.25, and 0.4. Earlier-budget candidates remain eligible at larger budgets. At epsilon 0.1, a 50-step, three-start attack with step size 0.004 is unioned with the 20-step candidates.

| Pipeline | Audit | Steps | Accuracy under attack | F1 | Recall |
|---|---|---|---|---|---|
| morphology | record_balanced_audit | 20 | 46.88% | 0.00% | 0.00% |
| morphology | strength_check | 50 | 43.54% | 0.00% | 0.00% |
| context | record_balanced_audit | 20 | 44.89% | 0.00% | 0.00% |
| context | strength_check | 50 | 36.51% | 0.00% | 0.00% |
| context_defended | record_balanced_audit | 20 | 90.62% | 22.35% | 13.01% |
| context_defended | strength_check | 50 | 90.62% | 22.35% | 13.01% |
| context_smoothed | record_balanced_audit | 20 | 78.55% | 7.93% | 8.90% |
| context_smoothed | strength_check | 50 | 78.12% | 7.78% | 8.90% |

If additional steps lower accuracy, the weaker run has not saturated. Even an unchanged result is not proof of global attack convergence. Full-test numbers above describe the stated 20-step suite, not the stronger 50-step audit. Complete curves and prediction arrays are saved beside each attack manifest.

## Four-class limitations

Selected checkpoint: `v2_aami_context_s17`. Macro-F1: 43.12%; three-seed mean: 44.60%.

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| N | 96.38% | 94.70% | 95.53% | 44240 |
| S | 51.43% | 2.94% | 5.56% | 1837 |
| V | 56.84% | 95.87% | 71.37% | 3220 |
| F | 0.00% | 0.00% | 0.00% | 388 |

Validation contains only one fusion beat. The selected model misses all fusion beats and almost all supraventricular beats on test. Aggregate accuracy and weighted F1 conceal those failures. This is an exploratory experiment, not a reliable four-class classifier.

Selected context model, lowest-recall records with at least 50 positive beats:

| Record | Arrhythmic beats | Detected | Recall |
|---|---|---|---|
| 222 | 209 | 1 | 0.48% |
| 234 | 53 | 3 | 5.66% |
| 232 | 1382 | 98 | 7.09% |
| 228 | 365 | 97 | 26.58% |
| 202 | 75 | 22 | 29.33% |

## CPU inference measurements

Hardware reported by runtime: AMD EPYC 9V74 80-Core Processor. Python 3.12.14; CPU PyTorch 2.14.1+cpu; two Torch threads. Each case uses 20 warm-up batches and 100 measured batches with no training/attack processes running.

| Variant | Batch size | Median ms/batch | p95 ms/batch | Median ms/beat |
|---|---|---|---|---|
| morphology | 1 | 0.130 | 0.363 | 0.1303 |
| morphology | 256 | 12.915 | 17.325 | 0.0504 |
| context | 1 | 0.146 | 0.280 | 0.1458 |
| context | 256 | 10.091 | 13.624 | 0.0394 |
| context_defended | 1 | 0.139 | 0.200 | 0.1389 |
| context_defended | 256 | 9.027 | 12.824 | 0.0353 |

Process peak RSS: 520.8 MiB, including imports and loaded data. Timings exclude preprocessing, network, cold startup, and UI rendering. They are measurements in this environment, not a production latency guarantee.

## What remains unproven

- External-database generalization and an automatic peak-detection classification pipeline.
- Reliable minority-class recognition, especially fusion and supraventricular beats.
- Score calibration and a clinically appropriate operating threshold.
- A built/running Docker image or public cloud deployment; only the local app and locked Python environment were executed here.
- Certified or end-to-end adversarial robustness.

Use a new development-only validation design and a predeclared external study for the next research iteration. Do not repeatedly tune this model on the current test results and relabel that as independent validation. Historical original-protocol metrics remain available but are not directly comparable to this changed protocol.
