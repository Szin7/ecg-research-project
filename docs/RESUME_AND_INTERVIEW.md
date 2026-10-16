# Résumé and interview preparation

**ECG Classification & Adversarial Robustness — Research Project**  
Python · PyTorch · WFDB · Streamlit · Docker

Suggested résumé entry, after you can explain and demonstrate the implementation:

- Built a subject-disjoint ECG pipeline covering 98,716 MIT-BIH beats; compared four model variants across 12 seeded runs with validation-only selection and whole-subject bootstrap intervals.
- Evaluated adaptive waveform attacks on 49,685 held-out beats, adding multiple PGD starts, worst-candidate retention, differentiable smoothing, and a separate 1,408-beat strength audit.
- Developed an eight-page research workbench with attribution, error analysis, reproducible experiment records, hashed CPU dependencies, automated tests, and container/CI configuration.

Use two or three bullets on the résumé itself. Do not claim cloud deployment, clinical performance, certified robustness, external validation, or sole authorship where those are not true. Container and remote CI execution remain unverified here. Preserve applicable attribution and repository history.

## A 90-second demonstration

1. Explain why the majority-class accuracy makes accuracy alone misleading.
2. Open a research-preview beat and identify annotation, score, and validation-selected threshold.
3. Apply PGD; describe epsilon in normalized waveform units and the fixed timing features.
4. Compare full-test results with the separate record-balanced audit.
5. Show all three seeds and a record where recall is poor.
6. Explain the next independent experiment and the current limitations.

## Questions to prepare

- Why do records 201 and 202 make record-only splitting insufficient?
- Why do expert peak annotations matter, and what changes with an automatic detector?
- Which features are causal? Why is the whole pipeline still offline?
- Why can the best validation seed differ from the best test seed?
- Why does weighted training produce scores that are not calibrated probabilities?
- What do epsilon, step size, number of steps, and random starts change in PGD?
- Why retain each example's worst candidate? Why differentiate through smoothing?
- Why can high accuracy coexist with poor arrhythmia recall?
- Why resample subjects rather than independent beats?
- What conclusions are limited by one fusion beat in validation?
- What do the latency measurements exclude?
- What is configured for deployment, and what has actually been executed?

Prepare concrete examples from the saved results. Reproducing and explaining one failure is more persuasive than hiding it. The next research step is a predeclared external evaluation or a new development-only validation scheme with better minority-class coverage; repeated tuning on the current test set would not remain independent validation.
