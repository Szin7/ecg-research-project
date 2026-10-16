> Historical review of the original submission. See STUDY_RESULTS.md and DEPLOYMENT.md for V2 results and current verification status.

# ECG Research Project — independent project review

Reviewed 13 October 2026. Both uploaded ZIP files had the same SHA-256 digest and identical content. Review included source inspection, the original 91-test suite, independent binary checkpoint inference against the supplied 49,685-beat test cache, and browser checks of the revised interface.

## Verdict

**Original project: 7/10 as a junior-to-intermediate ML engineering portfolio project.** This is a subjective review, not a certification or an average of arbitrary feature counts. It has substantially more substance than a notebook that reports accuracy alone. The weaknesses are in experimental validity, minority-class performance, and reproducibility, not lack of features.

| Area | Original score | Evidence |
|---|---:|---|
| Problem and portfolio value | 8.5/10 | ECG signal processing plus adversarial robustness gives a focused engineering story. |
| ML implementation | 7/10 | Compact CNN, loss weighting, validation threshold selection, attacks, classical baselines, real checkpoints. |
| Experimental validity | 5.5/10 | Validation/test subject overlap, single-seed results, non-adaptive smoothing claim, limited attack strength. |
| Software engineering | 7/10 | Modular code and 91 passing tests; cache, device, attribution and default-tag defects. |
| Original interface | 5.5/10 | Functional but text-heavy, low contrast, fragmented navigation, weak result hierarchy. |
| Reproducibility and delivery | 6/10 | Artifacts and configs present, but no CI in the upload, broad dependencies, no full run manifest or deployment evidence. |

The revised interface and implementation are stronger. I would not relabel the whole project 10/10 without rerunning the corrected research protocol and recording the results.

## What is already good

- Real signal preprocessing and beat segmentation, not a classifier wrapper around a CSV.
- A clear research question with FGSM, PGD and adversarial training.
- Record-wise splits, appropriate imbalance-aware metrics, and validation-only threshold selection are good foundations.
- Separate synthetic/demo and research artifacts exist.
- Classical feature baselines and four-class experiments make comparison possible.
- Useful tests cover numerical properties: perturbation bounds and the single-step FGSM/PGD relationship, for example.
- Both supplied binary models reproduce the stored confusion matrices and thresholded metrics on the supplied cached test set. AUC differences under this runtime are only at tiny floating-point/tie-order levels.

## Findings ranked by importance

### 1. Record-disjoint does not mean subject-disjoint here

`configs/dataset.yaml` puts record 201 in validation and record 202 in test. PhysioNet states that those recordings belong to the same person. The training records do not contain that person in this configuration, but validation is used for checkpoint and threshold selection, so test evaluation is not completely independent at the subject level.

**Next change:** keep both recordings in one partition or exclude 201 from model development while retaining the literature test set. Document the deviation and rerun all training, threshold selection and evaluation. Add a subject mapping and assert subject disjointness in train/validation/test. Do not merely rename the existing results as patient-independent.

The current update preserves the original experiment protocol and prominently labels this caveat. It does not manufacture new numbers.

Source: https://physionet.org/content/mitdb/1.0.0/

### 2. Accuracy overstates arrhythmia detection

| Measured result | Standard CNN | Adversarially trained CNN |
|---|---:|---:|
| Accuracy | 93.25% | 93.47% |
| Arrhythmia F1 | 65.40% | 60.59% |
| Arrhythmia recall | 58.20% | 45.80% |
| Precision | 74.63% | 89.45% |
| Missed arrhythmic beats | 2,276 / 5,445 | 2,951 / 5,445 |

The majority-class baseline is 89.04% accurate. A 93% accuracy résumé claim without recall/F1 would conceal a major limitation. The defended model improves precision but loses 12.40 percentage points of clean recall.

The saved four-class model has **35.51% macro-F1**. Its S-class F1 is 4.72% and F-class F1 is 0.18%. That is a useful failed/partial experiment to analyze, not a strong final multiclass classifier.

**Next experiments:** error analysis by subject and beat type; compare beat morphology alone with RR/context features; controlled ablations for loss weighting, sampling and receptive field. Choose improvements using validation, then evaluate the locked test set once. Do not promise a target score before measuring it.

### 3. The original smoothing result is non-adaptive

`scripts/run_benchmark.py` originally generated an attack against the plain model, then smoothed the attacked signal. That measures transfer through preprocessing, not the strength of an attack aware of the defense. Its 87.3% accuracy at epsilon 0.4 cannot establish that smoothing is a superior robust defense.

**Changed:** a differentiable `SmoothedModel` wrapper lets PGD attack the full pipeline. The evaluator includes epsilon zero to measure clean performance and records F1, recall, precision and attack success. It writes a separate file so the historical evidence is preserved.

**Still needed:** rerun the full research set, add PGD restarts and attack-strength sweeps, retain the strongest per-sample adversarial examples, and check additional attack objectives. Report attack-based estimates, not certificates.

Source: https://arxiv.org/abs/1902.06705

### 4. Engineering issues corrected in this update

| Original issue | Change |
|---|---|
| Binary benchmark moved inputs to the selected device but left the checkpoint on CPU. | Move the loaded model to the same device; fix GPU-to-NumPy conversion in example generation. CPU verified; accelerator hardware was unavailable. |
| Adversarial page inferred all 49,685 beats in one gradient-tracked forward pass. | Bounded inference batches, inference mode, and version-aware result caching. |
| Integrated gradients with a mean-reduced loss acquired an extra factor of 1/steps. | Correct loss scaling and trapezoidal path integration; add completeness/step-invariance regression test. |
| Interpolation coefficients created on CPU regardless of input device. | Match input device and dtype. |
| Cached beats ignored config and record-list changes. | Fingerprint configuration, label/preprocessing code, record list and source bytes; atomically replace caches. |
| Default demo training tags could overwrite canonical research checkpoints. | CLI derives and validates mode-specific tags, including the defended-model default. |
| Smoothing accepted even kernels that change signal length. | Require a positive odd kernel. |
| Checkpoints loaded with unrestricted pickle behavior. | Use `weights_only=True`; supplied checkpoints are compatible. |
| The UI called class-weighted sigmoid output “confidence.” | Show an uncalibrated arrhythmia score and its decision threshold. |
| Raw ECG graphs were labeled as z-score amplitude. | Label MIT-BIH physical signals as mV and synthetic signals separately. |
| README preprocessing command did not actually prepare cached beats. | `prepare_data` now prepares binary/AAMI splits after download/generation. |

### 5. Reproducibility is more than setting one seed

A seed-42 run cannot quantify training variability. The original attack scripts also did not record a complete environment/config/checkpoint manifest. Preserve these original results as historical runs.

**Next change:** immutable run directories containing config snapshots, seed, dataset fingerprint, checkpoint digest, dependency lock, device, timestamp, git revision and metrics. Train at least three seeds under a frozen protocol; report mean/spread and subject-level bootstrap intervals. A tested direct-dependency constraints file and CPU CI workflow are included now; neither is a full experiment registry or a deployed service.

### 6. Explanation and detector claims need precision

The original attribution page described prediction explanations while passing labels and differentiating a loss. The revised page explicitly explains the positive-class logit. It states that magnitude shading is normalized separately per chart and is not evidence of physiological causation.

Counting detected and annotated R-peaks is not a timing-agreement metric. Add one-to-one matching within a predeclared tolerance and report precision, recall and timing error. Also distinguish beat annotations from rhythm markers in that evaluation.

## Interface changes delivered

- Consistent navigation and page hierarchy across seven pages.
- Warm neutral background, dark ink text, serif headings and green/coral waveform colors.
- F1, recall and average precision take priority over accuracy.
- Waveform-led overview with a contextual reading of the results.
- Attack controls grouped into a form, an explicit attack seed, and a correctly applied class filter.
- Clear separation between transfer attacks and white-box benchmark results.
- Benchmark tabs for clean performance, attack sweeps and four-class results.
- Dataset selectors, artifact refresh on file changes, CSV exports, explicit missing-data states.
- Reduced-motion handling and visible keyboard-focus styling; narrow-screen checks are documented in the changelog.

## What would make this an excellent portfolio project?

Do these in order, rather than adding more unrelated features:

1. **Defensible evaluation.** Subject-disjoint protocol, recorded label/lead choices, fresh checkpoints, fresh results, no tuning against test data.
2. **Strong attack evaluation.** Adaptive defense attacks; multiple starts; iteration and step-size checks; worst-case per-sample aggregation; retain failed cases.
3. **Evidence of generalization.** Several seeds, subject-level uncertainty intervals, error analysis and one independently prepared external dataset if compatible with the task.
4. **Useful model improvement.** Demonstrate a measured gain from RR/context features or architecture/loss changes using controlled ablations. Report recall and false positives together.
5. **Engineering delivery.** Immutable experiment runs, a full lockfile, containerized demo, tested inference interface, health checks and measured CPU latency/memory. Choose a small useful deployment before introducing Kubernetes.
6. **A clear project story.** Short README, model/data cards, a 90-second walkthrough and a candid limitations section. The best interview evidence is being able to explain and modify the system live.

## Authorship and originality

The original archive contains a `.claude` configuration directory and a Claude co-author entry in Git metadata. Those are explicit tool records; prose style alone cannot establish authorship. The source distribution excludes workstation metadata and VCS internals as ordinary packaging hygiene. The uploaded originals are untouched and their history has not been rewritten.

No reliable guarantee of “zero AI traces” can be given. Code should be judged by its correctness, understandable decisions and demonstrated ownership. Avoid fake commit histories, invented performance claims, deliberately awkward comments or needless renaming. Retain required licenses/attribution and describe assistance truthfully if asked.
