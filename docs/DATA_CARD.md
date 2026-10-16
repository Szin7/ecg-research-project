# Data card — subject-disjoint MIT-BIH V2

## Source and attribution

[MIT-BIH Arrhythmia Database v1.0.0](https://physionet.org/content/mitdb/1.0.0/), PhysioNet. Retain the source attribution and Open Data Commons Attribution License v1.0 when distributing derived data.

Citations requested by the source:

- Moody GB, Mark RG. The impact of the MIT-BIH Arrhythmia Database. IEEE Engineering in Medicine and Biology Magazine. 2001;20(3):45–50. https://doi.org/10.1109/51.932724
- Goldberger AL et al. PhysioBank, PhysioToolkit, and PhysioNet: Components of a New Research Resource for Complex Physiologic Signals. Circulation. 2000;101(23):e215–e220. https://doi.org/10.1161/01.CIR.101.23.e215

The project does not claim ownership of the recordings. The compact package contains a transformed subset; full source files can be downloaded directly from PhysioNet.

## Protocol

Record 201 shares a subject with test record 202 and is excluded entirely. Paced records 102, 104, 107, and 217 are excluded. Complete record lists are in `configs/protocol_v2.yaml`; runtime checks reject record and subject overlap.

| Split | Records | Beats | N | S | V | F |
|---|---:|---:|---:|---:|---:|---:|
| Training | 18 | 41,754 | 37,478 | 720 | 3,145 | 411 |
| Validation | 3 | 7,277 | 6,735 | 96 | 445 | 1 |
| Test | 22 | 49,685 | 44,240 | 1,837 | 3,220 | 388 |

Processing uses MLII where available, otherwise the first lead; median-filter baseline correction; and 250-sample expert-annotated beat windows, with 90 samples before the peak and 160 starting at it, at 360 Hz. Beats are normalized independently and clipped to ±6 standard deviations. Q-group and unsupported labels are omitted. N is an AAMI superclass, not a clinical diagnosis of health.

The context vector contains bounded log values for previous RR, the mean of up to ten preceding intervals, their ratio, and the ratio to the preceding interval. No future beat interval or class label is used to compute these values. Waveform extraction includes post-peak samples and baseline filtering is offline, so this is not a streaming system.

Source recordings and processed splits have SHA-256 digests in each run manifest and the saved study data manifest. Training reads only training and validation arrays.

## Preview and limitations

`data/preview/v2_test.npz` contains 64 fixed beats per test record, 1,408 in total, selected without consulting labels or predictions. It includes normalized waveforms, RR context, labels, record IDs, and annotation sample positions. It is also the robustness audit subset. Its equal weighting of records differs from full-test beat weighting; the two populations must not be interchanged.

Subject separation within this historical database is not external validation. Only one fusion beat is present in validation, severely limiting four-class selection. Expert peaks are supplied to the classifier; errors from an automatic peak detector are outside the reported classification metrics. Neither acquisition conditions nor population diversity establish clinical generalization.

Explorer indices refer to the currently loaded array. Saved attack examples and study prediction files retain full-test indices.
