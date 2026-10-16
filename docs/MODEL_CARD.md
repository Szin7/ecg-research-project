# Model card — ECG Research Project V2

## Intended use

Educational ECG research and engineering demonstrations. Not for diagnosis, triage, patient monitoring, or treatment decisions.

## Architecture and inputs

A compact three-block 1D CNN consumes 250 normalized waveform samples. Context models concatenate four bounded log past-RR features with the pooled waveform representation. The waveform model has 11,265 parameters; binary context models have 11,393; the four-class context model has 11,492. Expert annotations supply peak locations. Processing is offline.

Binary loss uses positive-class weighting; multiclass loss uses inverse-square-root-frequency weights. Adversarial training mixes clean and three-step PGD examples at epsilon 0.1. Each variant has three seeded training runs with up to 30 epochs and validation-based early stopping.

## Published models

| Variant | Selected run | F1 | Recall | Precision | AP | Threshold |
|---|---|---|---|---|---|---|
| Waveform CNN | v2_morphology_s17 | 65.99% | 52.19% | 89.71% | 72.65% | 0.669708 |
| Waveform + past RR | v2_context_s42 | 62.26% | 45.90% | 96.79% | 83.18% | 0.998720 |
| Context + adversarial training | v2_context_defended_s42 | 52.16% | 35.56% | 97.83% | 82.83% | 0.975649 |

The selected context model has 62.26% F1 and 45.90% recall. Its 95% subject-bootstrap F1 interval is 35.57%–82.02%. Weighted training and validation-selected thresholds do not calibrate model scores into probabilities.

The selected four-class model has 43.12% macro-F1. Its S recall is 2.94% and F recall is 0.00%. Only one fusion beat is available in validation. These limitations rule out reliable clinical use.

## Evaluation boundaries

Subjects are disjoint within MIT-BIH after excluding record 201. No external database was evaluated. Automatic peak-detection errors are not included in classifier metrics. The fixed-context waveform threat model does not cover timing attacks; empirical PGD results are not certificates. Subject bootstrap and three-seed variation describe different uncertainty sources.

See [the complete study](STUDY_RESULTS.md) for exact attack settings, stronger-step audit, per-record errors, and timings; [the data card](DATA_CARD.md) for source attribution and processing; and [deployment status](DEPLOYMENT.md) for what was actually executed.

## Provenance

Saved run manifests, configurations, code/data/checkpoint hashes, histories, predictions, and evaluation outputs accompany the release. Retain MIT-BIH/PhysioNet attribution and applicable repository history. Code appearance does not establish authorship; be prepared to explain and reproduce the implementation.
