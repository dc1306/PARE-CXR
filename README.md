# PARE-CXR

Code for **PARE: Pathology-Aware Representation Enhancement via Optimal Transport for Chest X-Ray Report Generation** (submitted to IEEE ISBI 2027).

PARE recovers findings that a vision-language model represents internally but omits from its report. For each pathology, a Gaussian optimal-transport map is fitted between the early-layer visual-token representations of studies the model reported correctly (true positives) and studies it missed (false negatives). At inference, late-layer probes gate which maps are applied. The displacement is injected during a second, steered pass, and new findings are appended to the unchanged baseline report. Model weights stay frozen.

## Data

All experiments use [MIMIC-CXR-JPG v2.0.0](https://physionet.org/content/mimic-cxr-jpg/2.0.0/) and the free-text reports of [MIMIC-CXR v2.0.0](https://physionet.org/content/mimic-cxr/2.0.0/). Both require credentialed PhysioNet access and a signed data use agreement. **This repository contains no data, reports, or derived files.** All manifests are rebuilt from the official files by the scripts.

## Setup

The results were produced in two environments, one per model. They differ in the Python version, and both use the same key package versions:

| Environment | Python | Used for |
|---|---|---|
| `pare-8b` | 3.10 | CheXagent-8B (`StanfordAIMI/CheXagent-8b`), scripts 01–03, 07, 09 |
| `pare-3b` | 3.11 | CheXagent-2-3B (`StanfordAIMI/CheXagent-2-3b`), scripts 04–06, 08, 09 |

```bash
conda create -n pare-8b python=3.10 -y && conda activate pare-8b
pip install -r requirements/requirements-8b.txt

conda create -n pare-3b python=3.11 -y && conda activate pare-3b
pip install -r requirements/requirements-3b.txt
```

The requirement files are exact snapshots (`pip freeze`) of these environments. The key pins, identical in both, are `torch==2.1.2`, `transformers==4.37.2`, `scikit-learn==1.2.2`, and [`f1chexbert==0.0.2`](https://pypi.org/project/f1chexbert/). The last one downloads the CheXbert checkpoint on first use.

Set the data and output locations:

```bash
export MIMIC_CXR_JPG=/path/to/mimic-cxr-jpg/2.0.0         # contains files/ and the unzipped mimic-cxr-2.0.0-*.csv
export MIMIC_CXR_REPORTS=/path/to/mimic-cxr-reports/files # contains p10/ ... p19/
export PARE_OUT=/path/to/outputs                          # default: ./outputs
```

Select a GPU with `CUDA_VISIBLE_DEVICES`. Every script runs on a single GPU.

## Pipeline

Run the scripts in order. Each one caches its outputs in a sub-directory of `$PARE_OUT` and resumes from checkpoints if interrupted.

| Step | Script | Env | What it does |
|---|---|---|---|
| 1 | `scripts/01_test_set_and_baseline_8b.py` | 8b | Builds the test manifest (3,269 studies, up to two images per study, frontal views first) and generates CheXagent-8B baseline reports. |
| 2 | `scripts/02_trainval_features_8b.py` | 8b | Builds the train (15,000 studies, seed 42) and validation (1,808) manifests, extracts layer-3 and layer-31 features, and generates baseline reports on the training studies. It also fits a preliminary five-pathology component set, which step 3 uses only to verify the training manifest. |
| 3 | `scripts/03_fit_components_8b.py` | 8b | Fits the probes (layer 31), Youden-J thresholds (validation set), and Gaussian OT maps (layer 3) for CheXagent-8B. |
| 4 | `scripts/04_baseline_3b.py` | 3b | Generates CheXagent-2-3B baseline reports on the test and training studies (one image per study). |
| 5 | `scripts/05_fit_components_3b.py` | 3b | Fits probes, thresholds, and OT maps for CheXagent-2-3B. |
| 6 | `scripts/06_lambda_sweep_3b.py` | 3b | Selects the 3B steering strength (λ = 0.13) on 200 reserved test studies. These studies are excluded from evaluation. |
| 7 | `scripts/07_steer_test_8b.py` | 8b | Steered generation on the test set, CheXagent-8B, λ = 1.0. |
| 8 | `scripts/08_steer_test_3b.py` | 3b | Steered generation on the test set, CheXagent-2-3B, λ = 0.13. |
| 9 | `scripts/09_evaluate.py --model {8b,3b}` | either | Evaluation on the 2,000-study cohort (see below). |

Steps 2–3 and 4–5 are the expensive ones (feature extraction and report generation on 15,000 training studies).

### Steering strength

For CheXagent-8B, λ = 1.0 is fixed (the full transport displacement) and is not tuned. For CheXagent-2-3B, λ = 0.13 is selected in step 6 from {0.05, 0.1, 0.13, 0.15, 0.2, 0.3, 0.5}. The selection uses the first 200 test studies with a non-empty findings section. Step 9 excludes them from evaluation.

### Evaluation (step 9)

- **Cohort.** Of the 2,255 test studies with a non-empty findings section, the 200 used in step 6 are excluded. A random subset of 2,000 of the remaining 2,055 is drawn once (seed 42) and saved to `$PARE_OUT/evaluation/cohort_2000.json`. Both models are evaluated on the same cohort.
- **Labels.** Reference and generated reports are labelled with CheXbert (`f1chexbert`, `rrg` mode: uncertain counts as positive).
- **Conditions.**
  - baseline
  - raw PARE (steered report)
  - merged: baseline plus appended steered sentences, using `pare.safe_merger.safe_merge_negation_aware`
  - probe+template: the same gate and conditions as the merger, but a fixed sentence is appended instead of a steered one
- **Metrics.** Steer-5 macro-F1 (Atelectasis, Cardiomegaly, Consolidation, Edema, Pleural Effusion), All-13 macro-F1, Micro-14 F1, and per-label F1, sensitivity, and specificity.
- **Statistics.** Paired study-level bootstrap with 5,000 resamples (seed 2027) and 95% percentile intervals. The script also reports probe AUROC, the probe's recall on baseline false negatives, and the baseline true positives lost by raw PARE.

Outputs: `$PARE_OUT/evaluation/{8b,3b}/summary.txt` and `eval_2000.json`.

The full results reported in the paper, including per-label F1 for all 14 labels and sensitivity/specificity for the steered pathologies, are in [`RESULTS.md`](RESULTS.md).

## Fitted components

The probes, thresholds, and transport maps are fitted on MIMIC-CXR and are derived from it, so they are not distributed here (PhysioNet data use agreement). Steps 1–5 regenerate them from the official data.

## Repository layout

```
RESULTS.md          full per-label results for both models
pare/
  paths.py          all data and output locations (environment variables)
  safe_merger.py    append-only report merger (safe_merge_negation_aware is the one used in the paper)
scripts/            pipeline steps 01–09
requirements/       pinned environments
```

## Citation

```bibtex
@inproceedings{choudhary2027pare,
  title     = {{PARE}: Pathology-Aware Representation Enhancement via Optimal Transport for Chest X-Ray Report Generation},
  author    = {Choudhary, Darsh and Paul, Angshuman},
  booktitle = {IEEE International Symposium on Biomedical Imaging (ISBI)},
  year      = {2027},
  note      = {Under review}
}
```

## License

MIT (see `LICENSE`). MIMIC-CXR data remain subject to the PhysioNet credentialed health data use agreement. The CheXagent and CheXbert weights are subject to their own licenses.
