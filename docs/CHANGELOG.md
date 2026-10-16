# Project rename

Renamed the dashboard, page titles, documentation, CLI descriptions, package metadata, and container configuration to ECG Research Project. Subtitle: ECG Classification & Adversarial Robustness. Historical experiment records and model/data files are unchanged. Prior test/browser validation applies to the pre-rename release; this naming update was checked with Python syntax parsing and archive comparisons. Docker and remote CI remain unverified.

# Research V2 update — 16 October 2026

## Changes

- Rebuilt the subject-disjoint protocol, excluding record 201 and preserving record 202 in test.
- Added causal past-RR features and waveform/context model comparisons across three seeds per variant.
- Reproduced all 12 training runs; retained their actual checkpoints, histories, configurations, and hashes.
- Added full-test adaptive PGD and a separate record-balanced strength audit with worst-candidate retention and fixed-context constraints.
- Added subject-bootstrap intervals, per-record errors, a model card, data card, and measured study report.
- Added an eighth Study evidence page, research preview data, reliable dataset switching, context attribution, and clearer caption contrast.
- Added one-to-one peak timing agreement, hashed CPU dependencies, a saved-study verifier, and container/CI configuration.
- Updated résumé and interview material to reflect measured work and explicitly unverified deployment claims.

## Verification

The fresh locked Python environment passes 105 tests. All eight pages and the principal interactions were checked in a browser against the compact package. The definitive release record is `artifacts/release_validation.json`; attack metrics and CPU measurements are in `docs/STUDY_RESULTS.md`.

Docker build/startup and the remote GitHub Actions workflow were not executed here. No external database or clinical validation was performed. Original models/results remain under their original tags, with their historical protocol limitations.

---

# Review update — 13 October 2026

## Delivered

- Seven-page Streamlit interface redesign with consistent navigation and dataset selection.
- Metrics ordered around F1, recall and precision; positive-class scores replace confidence claims.
- Clear original-protocol limitations and transfer/adaptive attack distinctions.
- AAMI per-class results, benchmark CSV download and run-summary CSV download.
- Bounded, cached inference for selecting attackable beats.
- Correct integrated-gradients loss scaling and device-aware interpolation.
- Adaptive smoothing evaluation including clean performance; historical comparison retained.
- CPU/GPU placement corrections in binary benchmark and attack-example paths.
- Fingerprinted data caches and atomic cache writes.
- Mode-aware CLI tags; demo defaults no longer overwrite research defaults.
- Restricted checkpoint loading and smoothing-kernel validation.
- A working data-preparation command that also constructs processed splits.
- Tested dependency constraints and a CPU GitHub Actions test workflow.
- Short README, original experiment notes, a prioritized review and résumé/interview notes.

## Verification completed

| Check | Result |
|---|---|
| Uploaded ZIP comparison | Both archives identical by SHA-256. |
| Original unit tests | 91 passed. |
| Revised unit/regression tests | 99 passed on CPU / Python 3.12.14. |
| Source compilation | app, src and scripts compiled successfully. |
| Independent binary inference | Standard and defended confusion matrices and thresholded metrics reproduced on 49,685 supplied cached research test beats. |
| Existing weights and research results | Byte-for-byte unchanged from the supplied archive. |
| Synthetic data preparation | Binary and AAMI train/validation/test caches rebuilt; original numerical arrays preserved. |
| Adaptive smoothing smoke experiment | Completed on the 80-beat synthetic demo, epsilon 0, 0.1 and 0.4; saved separately as secondary_defense_adaptive.csv. This is not research evidence. |
| Browser page checks | All seven pages loaded without app exceptions in research mode. |
| Browser interactions | Dataset switch to demo, arrhythmia filter, zero-budget attack, AAMI tab and sweep CSV download passed. |
| Responsive check | Fresh 390 × 844 viewport: sidebar initially collapsed, document width 390; desktop inspected at 1440 px. |

A demo-only chart-ID collision was discovered and fixed during interaction checks. No browser screenshots are included in the deliverable.

## Not completed or claimed

- No full retraining or corrected subject-disjoint experiment was performed.
- No new research-set adaptive defense score is claimed.
- CUDA/MPS execution was not hardware-tested.
- GitHub Actions workflow was authored; it has not run on a remote repository.
- No cloud deployment, Docker build, load test, clinical validation or accessibility conformance audit is claimed.
- Tested constraints are not a full transitive dependency lock.
- Existing single-seed, single-start attack artifacts remain historical results.

## Distribution

This is a compact source update with checkpoints, saved experiment results, synthetic records and processed demo beats. It omits raw MIT-BIH files, processed research arrays, VCS internals, workstation settings and build caches. Preserve your original upload. Copy existing research data into the revised tree or run the preparation command in the README.
