# ECG Research Project

**ECG Classification & Adversarial Robustness**

Suggested GitHub repository name: `ecg-research-project`.

An ECG classification and adversarial-robustness workbench. Inspect annotated beats, compare waveform and rhythm-context models, and trace reported results to saved experiments.

**Research prototype, not a diagnostic device.** V2 corrects the original shared-subject split and reports three seeds for each of four model variants. Its value is transparent evaluation and engineering; minority-class recognition and external validity remain limitations.

## Run the app

Use Linux/WSL, Python 3.12, and CPU PyTorch for the fully pinned environment:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --require-hashes -r requirements-cpu.lock
python -m pip install --no-deps --no-build-isolation -e .
python -m streamlit run app/main.py
```

The compact package includes checkpoints, run histories, saved full-test results, a 1,408-beat research preview, and synthetic demo data. Interactive research pages use the preview until full data are prepared. Saved aggregate metrics retain their stated full-test population. Raw-signal analysis needs the original recordings or the synthetic demo.

Windows PowerShell alternative (provided but not executed in this Linux environment):

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install torch==2.14.1 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -c requirements-tested.txt -e ".[dev]"
python -m streamlit run app/main.py
```

The hashed lock targets Linux x86-64. The Windows alternative uses direct constraints rather than the complete lock.

## What is implemented

- Subject-disjoint preparation: 18 training, 3 validation, and 22 test records; shared-subject record 201 excluded.
- Waveform CNN, waveform-plus-past-RR CNN, adversarial training, and exploratory four-class N/S/V/F classification.
- Twelve seeded training runs, validation-only selection, and whole-subject bootstrap intervals.
- Adaptive PGD with multiple starts, worst-candidate retention, differentiable smoothing, and stronger-step checks.
- Eight dashboard pages for exploration, attacks, attribution, benchmarks, signal processing, run history, and study evidence.
- Hash-locked CPU dependencies, numerical tests, experiment checksums, and non-root container configuration.

## Measured clean performance

Same subject-disjoint test set, 49,685 beats. Mean ± sample standard deviation across three seeds:

| Model | Arrhythmia F1 | Average precision |
|---|---:|---:|
| Waveform CNN | 61.79% ± 3.87 pp | 70.43% ± 2.40 pp |
| Waveform + past RR | 64.20% ± 4.47 pp | 78.40% ± 4.14 pp |
| Context + adversarial training | 53.21% ± 6.58 pp | 79.27% ± 4.68 pp |

The validation-selected context checkpoint reaches 62.26% F1 and 45.90% recall. Its subject-bootstrap F1 interval is wide (35.57–82.02%). Four-class performance remains exploratory, with only one fusion beat in validation. These are dataset-specific results, not clinical guarantees.

See [measured study results](docs/STUDY_RESULTS.md) and the [model card](docs/MODEL_CARD.md). Historical results and model tags are preserved separately and must not be relabeled as V2 experiments.

## Reproduce the study

```bash
python -m scripts.prepare_protocol --download
for variant in morphology context context_defended aami_context; do
  for seed in 17 42 73; do
    python -m scripts.train_protocol --variant "$variant" --seed "$seed" --run-id "rerun_${variant}_s${seed}"
  done
done
python -m scripts.evaluate_protocol --name rerun_study --run-prefix rerun
python -m scripts.attack_protocol --variant morphology
python -m scripts.attack_protocol --variant context
python -m scripts.attack_protocol --variant context_defended
python -m scripts.attack_protocol --variant context --smoothing
python -m scripts.benchmark_inference
python -m scripts.verify_study
python -m pytest -q
```

Run/study directories refuse reuse. Evaluation updates the active-study pointer and `v2_*` dashboard aliases; original run records remain unchanged. The attack suite takes substantially longer than training. Run CPU timing measurements without competing workloads.

The waveform threat model keeps expert peak positions and past-RR features fixed. Epsilon is in normalized waveform units, not mV. These experiments do not evaluate an end-to-end detector, timing attacks, or certified robustness. Sigmoid scores are not calibrated probabilities.

## Documentation

- [Study results](docs/STUDY_RESULTS.md) and [model card](docs/MODEL_CARD.md)
- [Data and attribution](docs/DATA_CARD.md)
- [Deployment and verification](docs/DEPLOYMENT.md)
- [Résumé entry and interview preparation](docs/RESUME_AND_INTERVIEW.md)
- [Change history](docs/CHANGELOG.md)
- [Historical review](docs/PROJECT_REVIEW.md)

## References

- [MIT-BIH Arrhythmia Database v1.0.0, PhysioNet](https://physionet.org/content/mitdb/1.0.0/)
- [de Chazal et al., morphology and heartbeat interval features](https://doi.org/10.1109/TBME.2004.827359)
- [Madry et al., adversarial training](https://arxiv.org/abs/1706.06083)
- [Carlini et al., evaluating adversarial robustness](https://arxiv.org/abs/1902.06705)
- [Sundararajan et al., integrated gradients](https://arxiv.org/abs/1703.01365)
