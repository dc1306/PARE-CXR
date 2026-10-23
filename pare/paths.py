"""Central path configuration for all PARE scripts.

Set these environment variables before running anything (see README):

    MIMIC_CXR_JPG      root of MIMIC-CXR-JPG v2.0.0: contains files/ and the
                       (unzipped) mimic-cxr-2.0.0-{split,metadata,chexpert}.csv
    MIMIC_CXR_REPORTS  directory containing the report folders p10/ ... p19/
                       (from mimic-cxr-reports.zip of MIMIC-CXR v2.0.0)
    PARE_OUT           output directory (default: ./outputs)
"""
import os
from pathlib import Path


def _env(name: str) -> Path:
    value = os.environ.get(name)
    if not value:
        raise RuntimeError(f"Environment variable {name} is not set. See README, section 'Setup'.")
    return Path(value).expanduser()


MIMIC_JPG = _env("MIMIC_CXR_JPG")
MIMIC_REPORTS = _env("MIMIC_CXR_REPORTS")
OUT = Path(os.environ.get("PARE_OUT", "outputs")).expanduser().resolve()

MIMIC_SPLIT = MIMIC_JPG / "mimic-cxr-2.0.0-split.csv"
MIMIC_META = MIMIC_JPG / "mimic-cxr-2.0.0-metadata.csv"
CHEXPERT_CSV = MIMIC_JPG / "mimic-cxr-2.0.0-chexpert.csv"
IMAGE_DIRS = [str(MIMIC_JPG / "files")]
REPORTS_DIR = str(MIMIC_REPORTS)

# Output sub-directories, one per pipeline stage.
STAGES = (
    "test_set_and_baseline_8b",     # step 1: test manifest + CheXagent-8B baseline reports
    "trainval_features_8b",         # step 2: train/val manifests, 8B features and training reports
    "components_8b",                # step 3: 8B probes, thresholds and transport maps
    "baseline_and_components_3b",   # steps 4-5: 3B baseline reports and components
    "lambda_sweep_3b",              # step 6: 3B steering-strength selection
    "steered_8b",                   # step 7: 8B steered test reports (lambda = 1.0)
    "steered_3b",                   # step 8: 3B steered test reports (lambda = 0.13)
    "evaluation",                   # step 9: evaluation on the 2,000-study cohort
)


def out(stage: str) -> Path:
    if stage not in STAGES:
        raise ValueError(f"Unknown pipeline stage: {stage}")
    d = OUT / stage
    d.mkdir(parents=True, exist_ok=True)
    return d
