# Detailed results

Full results for the paper, on the 2,000-study MIMIC-CXR test cohort (see the README, step 9).
All numbers are produced by `scripts/09_evaluate.py`. Labels come from CheXbert, with uncertain
mentions counted as positive.

Conditions: **Baseline** is the unsteered model. **Raw PARE** is the steered report alone.
**PARE** is the full method (baseline report plus appended steered sentences). **Probe+template**
is a classifier-style reference that appends a fixed sentence whenever the probe fires.
Steered pathologies are marked with \*.

## CheXagent-8B (λ = 1.0)

### Aggregate F1

| Metric | Baseline | Raw PARE | PARE | Probe+template |
|---|---|---|---|---|
| Steer-5 (macro) | .462 | .483 (+.022, 95% CI 0.7 to 3.7 pp) | .495 (+.034, 95% CI 2.4 to 4.5 pp) | .524 (+.062, 95% CI 4.8 to 7.7 pp) |
| All-13 (macro) | .327 | .332 (+.006, 95% CI −0.5 to 1.6 pp) | .343 (+.016, 95% CI 1.0 to 2.1 pp) | .353 (+.026, 95% CI 1.9 to 3.4 pp) |
| Micro-14 | .454 | .475 (+.020, 95% CI 1.0 to 3.0 pp) | .479 (+.024, 95% CI 1.9 to 3.0 pp) | .486 (+.032, 95% CI 2.5 to 3.9 pp) |

### Per-label F1

| Label | Baseline | Raw PARE | PARE | Δ (PARE − Baseline) | Probe+template |
|---|---|---|---|---|---|
| Enlarged Cardiomediastinum | .139 | .108 | .146 | +.007 | .139 |
| Cardiomegaly \* | .550 | .610 | .609 | +.059 | .630 |
| Lung Opacity | .454 | .403 | .455 | +.001 | .439 |
| Lung Lesion | .055 | .074 | .062 | +.008 | .055 |
| Edema \* | .547 | .547 | .562 | +.015 | .556 |
| Consolidation \* | .141 | .131 | .152 | +.011 | .208 |
| Pneumonia | .240 | .205 | .238 | −.002 | .240 |
| Atelectasis \* | .394 | .442 | .469 | +.075 | .542 |
| Pneumothorax | .274 | .298 | .272 | −.002 | .282 |
| Pleural Effusion \* | .676 | .688 | .685 | +.009 | .684 |
| Pleural Other | .000 | .000 | .000 | .000 | .033 |
| Fracture | .095 | .123 | .104 | +.008 | .095 |
| Support Devices | .685 | .694 | .698 | +.013 | .685 |
| No Finding | .241 | .241 | .232 | −.009 | .222 |

### Sensitivity and specificity (steered pathologies)

| Pathology | Baseline | Raw PARE | PARE | Probe+template |
|---|---|---|---|---|
| Atelectasis | 0.40 / 0.73 | 0.48 / 0.71 | 0.57 / 0.63 | 0.82 / 0.49 |
| Cardiomegaly | 0.54 / 0.76 | 0.68 / 0.68 | 0.68 / 0.68 | 0.78 / 0.60 |
| Consolidation | 0.16 / 0.92 | 0.14 / 0.92 | 0.19 / 0.91 | 0.48 / 0.76 |
| Edema | 0.71 / 0.74 | 0.80 / 0.66 | 0.83 / 0.67 | 0.88 / 0.62 |
| Pleural Effusion | 0.72 / 0.79 | 0.83 / 0.71 | 0.80 / 0.73 | 0.80 / 0.73 |

Each cell: sensitivity / specificity.

## CheXagent-3B (CheXagent-2-3B, λ = 0.13)

### Aggregate F1

| Metric | Baseline | Raw PARE | PARE | Probe+template |
|---|---|---|---|---|
| Steer-5 (macro) | .507 | .513 (+.005, 95% CI −0.6 to 1.7 pp) | .518 (+.010, 95% CI 0.4 to 1.8 pp) | .522 (+.014, 95% CI −0.0 to 2.9 pp) |
| All-13 (macro) | .413 | .419 (+.006, 95% CI −0.6 to 1.7 pp) | .417 (+.004, 95% CI 0.1 to 0.7 pp) | .418 (+.005, 95% CI −0.1 to 1.0 pp) |
| Micro-14 | .537 | .544 (+.007, 95% CI 0.2 to 1.2 pp) | .542 (+.005, 95% CI 0.2 to 0.7 pp) | .532 (−.005, 95% CI −1.1 to 0.1 pp) |

### Per-label F1

| Label | Baseline | Raw PARE | PARE | Δ (PARE − Baseline) | Probe+template |
|---|---|---|---|---|---|
| Enlarged Cardiomediastinum | .317 | .321 | .318 | +.001 | .318 |
| Cardiomegaly \* | .631 | .649 | .643 | +.012 | .640 |
| Lung Opacity | .484 | .491 | .484 | .000 | .472 |
| Lung Lesion | .233 | .229 | .233 | .000 | .233 |
| Edema \* | .555 | .563 | .563 | +.009 | .543 |
| Consolidation \* | .151 | .157 | .171 | +.020 | .196 |
| Pneumonia | .248 | .256 | .252 | +.004 | .247 |
| Atelectasis \* | .511 | .504 | .520 | +.009 | .544 |
| Pneumothorax | .427 | .467 | .421 | −.006 | .427 |
| Pleural Effusion \* | .690 | .693 | .691 | +.001 | .686 |
| Pleural Other | .067 | .034 | .067 | .000 | .067 |
| Fracture | .286 | .304 | .286 | .000 | .284 |
| Support Devices | .775 | .781 | .775 | .000 | .777 |
| No Finding | .290 | .287 | .286 | −.004 | .268 |

### Sensitivity and specificity (steered pathologies)

| Pathology | Baseline | Raw PARE | PARE | Probe+template |
|---|---|---|---|---|
| Atelectasis | 0.61 / 0.68 | 0.61 / 0.66 | 0.64 / 0.65 | 0.80 / 0.51 |
| Cardiomegaly | 0.74 / 0.64 | 0.78 / 0.64 | 0.77 / 0.63 | 0.83 / 0.55 |
| Consolidation | 0.11 / 0.97 | 0.12 / 0.97 | 0.14 / 0.97 | 0.47 / 0.75 |
| Edema | 0.69 / 0.76 | 0.73 / 0.74 | 0.74 / 0.74 | 0.88 / 0.59 |
| Pleural Effusion | 0.73 / 0.81 | 0.76 / 0.78 | 0.74 / 0.79 | 0.80 / 0.73 |

Each cell: sensitivity / specificity.
