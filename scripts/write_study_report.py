"""Generate the study report from completed saved results, without re-evaluation."""

import json
from pathlib import Path
import pandas as pd
from src.utils.paths import ARTIFACTS_DIR, PROJECT_ROOT


def table(headers, rows):
    return "\n".join(
        [
            "| " + " | ".join(headers) + " |",
            "|" + "|".join(["---"] * len(headers)) + "|",
            *["| " + " | ".join(map(str, row)) + " |" for row in rows],
        ]
    )


def percent(value):
    return f"{100 * value:.2f}%"


def build_report():
    active = json.loads((ARTIFACTS_DIR / "active_study.json").read_text())
    study = ARTIFACTS_DIR / active["study_dir"]
    seeds = pd.read_csv(study / "seed_results.csv")
    labels = {
        "morphology": "Waveform CNN",
        "context": "Waveform + past RR",
        "context_defended": "Context + adversarial training",
        "aami_context": "Four-class context CNN",
    }
    means = []
    for variant in ["morphology", "context", "context_defended"]:
        frame = seeds[seeds.variant == variant]
        means.append(
            [
                labels[variant],
                *[
                    f"{100 * frame[k].mean():.2f}% ± {100 * frame[k].std():.2f} pp"
                    for k in ["f1", "recall", "pr_auc"]
                ],
            ]
        )
    selected = []
    results = {}
    intervals = []
    for variant, run in active["selected_runs"].items():
        result = json.loads((study / f"{run}_clean.json").read_text())
        results[variant] = result
        if variant != "aami_context":
            m = result["cnn"]
            selected.append(
                [
                    labels[variant],
                    run,
                    percent(m["f1"]),
                    percent(m["recall"]),
                    percent(m["precision"]),
                    percent(m["pr_auc"]),
                    f"{result['threshold']:.6f}",
                ]
            )
            for metric, ci in result["subject_bootstrap_95"].items():
                intervals.append(
                    [
                        labels[variant],
                        metric,
                        percent(m[metric]),
                        f"{percent(ci['lower'])}–{percent(ci['upper'])}",
                    ]
                )
    attacks = []
    strength = []
    for pipeline in ["morphology", "context", "context_defended", "context_smoothed"]:
        folder = study / f"attacks_{pipeline}"
        manifest = json.loads((folder / "manifest.json").read_text())
        assert manifest["status"] == "completed", pipeline
        frame = pd.read_csv(folder / "metrics.csv")
        for row in frame[frame.scope == "full_test"].itertuples():
            attacks.append(
                [
                    pipeline,
                    row.epsilon,
                    percent(row.robust_accuracy),
                    percent(row.adversarial_f1),
                    percent(row.adversarial_recall),
                    percent(row.attack_success_rate),
                ]
            )
        for row in frame[
            (frame.scope != "full_test") & (frame.epsilon == 0.1)
        ].itertuples():
            strength.append(
                [
                    pipeline,
                    row.scope,
                    row.steps,
                    percent(row.robust_accuracy),
                    percent(row.adversarial_f1),
                    percent(row.adversarial_recall),
                ]
            )
    aam = results["aami_context"]["metrics"]
    aami_rows = [
        [label, *[percent(m[k]) for k in ["precision", "recall", "f1"]], m["support"]]
        for label, m in aam["per_class"].items()
    ]
    errors = pd.read_csv(study / f"{active['selected_runs']['context']}_errors.csv")
    error_rows = [
        [int(r.record), int(r.support), int(r.correct), percent(r.recall)]
        for r in errors[(errors["class"] == 1) & (errors.support >= 50)]
        .sort_values("recall")
        .head(5)
        .itertuples()
    ]
    timing = json.loads((study / "inference_benchmark.json").read_text())
    time_rows = [
        [
            r["variant"],
            r["batch_size"],
            f"{r['median_ms_per_batch']:.3f}",
            f"{r['p95_ms_per_batch']:.3f}",
            f"{r['median_ms_per_beat']:.4f}",
        ]
        for r in timing["results"]
    ]
    model = results["context"]
    cm = model["cnn"]
    ci = model["subject_bootstrap_95"]["f1"]
    content = f"""# Study results — ECG Research Project V2

Study directory: `{active["study_dir"]}`. All results below were computed from the source and saved checkpoints included in this release.

## Protocol and selection

The split excludes shared-subject record 201, retains 202 in test, and separates 18 training, 3 validation, and 22 test records. Training/validation/test contain 41,754 / 7,277 / 49,685 beats. An always-N prediction reaches 89.04% test accuracy. Expert annotations supply peak positions for extraction and past-RR features; this is an offline classifier experiment.

Four variants were trained with seeds 17, 42, and 73. Checkpoints and binary thresholds were selected by validation F1; the four-class head uses validation macro-F1. The selected seed is the one with the highest validation score, not the best test result. The architecture, split, seeds, selection rules, and attack settings were fixed for this rerun. The test database has been inspected during this project; these results are not an untouched external validation cohort.

## Across training seeds

Entries are mean ± sample standard deviation across three seeds, not confidence intervals or significance tests.

{table(["Variant", "F1", "Recall", "Average precision"], means)}

Context improves average ranking performance, but F1 varies substantially across seeds. The selected context model is not the best test-F1 context seed and does not beat the selected waveform-only model on F1. Do not substitute test-best seeds into the dashboard.

## Validation-selected models

{table(["Variant", "Selected run", "F1", "Recall", "Precision", "AP", "Threshold"], selected)}

The selected context model misses {model["cnn"]["confusion_matrix"][1][0]:,} of 5,445 arrhythmic test beats. Its high threshold and class-weighted training do not turn its sigmoid scores into calibrated probabilities. Report recall and precision alongside F1 and accuracy.

## Variation across subjects

2,000 bootstrap repetitions resample whole subjects with replacement. These 95% percentile intervals condition on the selected model and this database. They do not include every source of development uncertainty or establish clinical reliability.

{table(["Variant", "Metric", "Point", "95% subject-bootstrap interval"], intervals)}

## Full-test adaptive attacks

All rows below use 49,685 test beats. At epsilon 0.1, PGD uses 20 steps, step size 0.01, one clean start plus two random starts, and an additional 20-step BCE-objective pass. Each example retains its worst threshold-adjusted margin across candidates. The smoothed pipeline differentiates through a five-sample moving average and retains its original validation-selected threshold. Epsilon zero is each pipeline's own clean baseline.

{table(["Pipeline", "Epsilon", "Accuracy under attack", "F1", "Recall", "Success on clean-correct beats"], attacks)}

This is a waveform-only L∞ threat model in normalized units. Expert peak positions and RR features stay fixed; waveform values remain within ±6. Timing attacks, peak-detection failures, and physiological plausibility are outside this evaluation. Finite attack results do not certify robustness.

## Separate strength audit

A fixed 64 beats per test record, 1,408 total, were selected without labels or predictions. This record-balanced population differs from the full beat-weighted test set. The audit also covers epsilon 0, 0.05, 0.1, 0.25, and 0.4. Earlier-budget candidates remain eligible at larger budgets. At epsilon 0.1, a 50-step, three-start attack with step size 0.004 is unioned with the 20-step candidates.

{table(["Pipeline", "Audit", "Steps", "Accuracy under attack", "F1", "Recall"], strength)}

If additional steps lower accuracy, the weaker run has not saturated. Even an unchanged result is not proof of global attack convergence. Full-test numbers above describe the stated 20-step suite, not the stronger 50-step audit. Complete curves and prediction arrays are saved beside each attack manifest.

## Four-class limitations

Selected checkpoint: `{active["selected_runs"]["aami_context"]}`. Macro-F1: {percent(aam["macro_f1"])}; three-seed mean: {percent(seeds[seeds.variant == "aami_context"].macro_f1.mean())}.

{table(["Class", "Precision", "Recall", "F1", "Support"], aami_rows)}

Validation contains only one fusion beat. The selected model misses all fusion beats and almost all supraventricular beats on test. Aggregate accuracy and weighted F1 conceal those failures. This is an exploratory experiment, not a reliable four-class classifier.

Selected context model, lowest-recall records with at least 50 positive beats:

{table(["Record", "Arrhythmic beats", "Detected", "Recall"], error_rows)}

## CPU inference measurements

Hardware reported by runtime: {timing["environment"]["cpu"]}. Python {timing["environment"]["python"]}; CPU PyTorch {timing["environment"]["packages"]["torch"]}; two Torch threads. Each case uses 20 warm-up batches and 100 measured batches with no training/attack processes running.

{table(["Variant", "Batch size", "Median ms/batch", "p95 ms/batch", "Median ms/beat"], time_rows)}

Process peak RSS: {timing["process_peak_rss_mib_linux"]:.1f} MiB, including imports and loaded data. Timings exclude preprocessing, network, cold startup, and UI rendering. They are measurements in this environment, not a production latency guarantee.

## What remains unproven

- External-database generalization and an automatic peak-detection classification pipeline.
- Reliable minority-class recognition, especially fusion and supraventricular beats.
- Score calibration and a clinically appropriate operating threshold.
- A built/running Docker image or public cloud deployment; only the local app and locked Python environment were executed here.
- Certified or end-to-end adversarial robustness.

Use a new development-only validation design and a predeclared external study for the next research iteration. Do not repeatedly tune this model on the current test results and relabel that as independent validation. Historical original-protocol metrics remain available but are not directly comparable to this changed protocol.
"""
    (PROJECT_ROOT / "docs/STUDY_RESULTS.md").write_text(content)
    card = f"""# Model card — ECG Research Project V2

## Intended use

Educational ECG research and engineering demonstrations. Not for diagnosis, triage, patient monitoring, or treatment decisions.

## Architecture and inputs

A compact three-block 1D CNN consumes 250 normalized waveform samples. Context models concatenate four bounded log past-RR features with the pooled waveform representation. The waveform model has 11,265 parameters; binary context models have 11,393; the four-class context model has 11,492. Expert annotations supply peak locations. Processing is offline.

Binary loss uses positive-class weighting; multiclass loss uses inverse-square-root-frequency weights. Adversarial training mixes clean and three-step PGD examples at epsilon 0.1. Each variant has three seeded training runs with up to 30 epochs and validation-based early stopping.

## Published models

{table(["Variant", "Selected run", "F1", "Recall", "Precision", "AP", "Threshold"], selected)}

The selected context model has {percent(cm["f1"])} F1 and {percent(cm["recall"])} recall. Its 95% subject-bootstrap F1 interval is {percent(ci["lower"])}–{percent(ci["upper"])}. Weighted training and validation-selected thresholds do not calibrate model scores into probabilities.

The selected four-class model has {percent(aam["macro_f1"])} macro-F1. Its S recall is {percent(aam["per_class"]["S"]["recall"])} and F recall is {percent(aam["per_class"]["F"]["recall"])}. Only one fusion beat is available in validation. These limitations rule out reliable clinical use.

## Evaluation boundaries

Subjects are disjoint within MIT-BIH after excluding record 201. No external database was evaluated. Automatic peak-detection errors are not included in classifier metrics. The fixed-context waveform threat model does not cover timing attacks; empirical PGD results are not certificates. Subject bootstrap and three-seed variation describe different uncertainty sources.

See [the complete study](STUDY_RESULTS.md) for exact attack settings, stronger-step audit, per-record errors, and timings; [the data card](DATA_CARD.md) for source attribution and processing; and [deployment status](DEPLOYMENT.md) for what was actually executed.

## Provenance

Saved run manifests, configurations, code/data/checkpoint hashes, histories, predictions, and evaluation outputs accompany the release. Retain MIT-BIH/PhysioNet attribution and applicable repository history. Code appearance does not establish authorship; be prepared to explain and reproduce the implementation.
"""
    (PROJECT_ROOT / "docs/MODEL_CARD.md").write_text(card)
    print("Wrote study report and model card")


if __name__ == "__main__":
    build_report()
