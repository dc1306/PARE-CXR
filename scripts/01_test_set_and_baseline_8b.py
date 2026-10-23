#!/usr/bin/env python3
"""Step 1: test set and CheXagent-8B baseline reports.

Builds the study-level test manifest from the official MIMIC-CXR test split
(3,269 studies) and generates unsteered CheXagent-8B reports for it.

Protocol:
  - Up to two images per study, frontal views first (PA, AP, then lateral).
  - The prompt contains the indication section of the reference report.
  - Greedy decoding (do_sample=False, num_beams=1, max_new_tokens=512).

Outputs ($PARE_OUT/test_set_and_baseline_8b/):
  study_test_manifest.json    test manifest, used by all later steps
  chexagent_8b_reports.json   baseline reports

Usage:
  python scripts/01_test_set_and_baseline_8b.py
"""

import sys as _sys; from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))  # repo root
from pare import paths
import os
import torch, json, time, re, gc, glob
import pandas as pd
import numpy as np
from collections import defaultdict
from PIL import Image

# Paths
MIMIC_SPLIT  = str(paths.MIMIC_SPLIT)
MIMIC_META   = str(paths.MIMIC_META)
REPORTS_DIR  = paths.REPORTS_DIR
IMAGE_DIRS   = paths.IMAGE_DIRS
EXP_DIR      = paths.out("test_set_and_baseline_8b")
EXP_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = "cuda:0"
DTYPE  = torch.float16
CHECKPOINT = "StanfordAIMI/CheXagent-8b"
MAX_IMAGES = 2   # up to two images per study
LOG_FILE = EXP_DIR / "chexagent_8b.log"

def log(msg):
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    with open(LOG_FILE, "a") as f:
        f.write(line + "\n")


# Step 1a: build the study-level test manifest
def parse_report(report_path):
    """Parse MIMIC-CXR report into sections."""
    with open(report_path, "r") as f:
        text = f.read()
    sections = {}
    patterns = [
        (r'(?:INDICATION|HISTORY|CLINICAL INFORMATION|CLINICAL HISTORY|REASON FOR EXAM(?:INATION)?)\s*:?\s*', 'indication'),
        (r'FINDINGS?\s*:?\s*', 'findings'),
        (r'IMPRESSION\s*:?\s*', 'impression'),
    ]
    boundaries = []
    for pattern, name in patterns:
        for m in re.finditer(pattern, text, re.IGNORECASE):
            boundaries.append((m.start(), m.end(), name))
    boundaries.sort(key=lambda x: x[0])
    for i, (start, content_start, name) in enumerate(boundaries):
        end = boundaries[i+1][0] if i+1 < len(boundaries) else len(text)
        content = re.sub(r'\s+', ' ', text[content_start:end]).strip()
        if content:
            sections[name] = content
    return sections


def build_study_manifest():
    """Build the study-level test manifest (up to two images per study)."""
    manifest_path = EXP_DIR / "study_test_manifest.json"
    # The manifest is always rebuilt from the official files
    log("Building study-level test manifest...")
    split_df = pd.read_csv(MIMIC_SPLIT)
    test_df = split_df[split_df['split'] == 'test']
    log(f"  Official test DICOMs: {len(test_df)}")

    # Find all images
    all_jpgs = {}
    for d in IMAGE_DIRS:
        if not os.path.exists(d): continue
        for p in glob.glob(f'{d}/**/*.jpg', recursive=True):
            all_jpgs[os.path.basename(p).replace('.jpg', '')] = p
    log(f"  Images on disk: {len(all_jpgs)}")

    # View position for sorting
    meta_df = pd.read_csv(MIMIC_META)
    view_map = dict(zip(meta_df['dicom_id'], meta_df['ViewPosition'].fillna('')))

    # Group by study
    studies = defaultdict(list)
    for _, row in test_df.iterrows():
        d = row['dicom_id']
        if d in all_jpgs:
            studies[(row['subject_id'], row['study_id'])].append({
                'dicom_id': d, 'image_path': all_jpgs[d],
                'view': view_map.get(d, ''),
            })
    log(f"  Test studies with images: {len(studies)}")

    manifest = []
    missing_reports = 0
    for (subject_id, study_id), images in studies.items():
        # Up to two images per study
        # Frontal views first: PA, AP, then lateral
        vp = {'PA': 0, 'AP': 1, 'LATERAL': 2, 'LL': 3}
        images.sort(key=lambda x: vp.get(x['view'].upper(), 4))
        selected = images[:MAX_IMAGES]

        p_dir = f"p{str(subject_id)[:2]}"
        rpath = os.path.join(REPORTS_DIR, p_dir, f"p{subject_id}", f"s{study_id}.txt")
        if not os.path.exists(rpath):
            missing_reports += 1
            log(f"  Missing report: p{subject_id}/s{study_id}")
            continue

        sections = parse_report(rpath)
        manifest.append({
            'subject_id': int(subject_id), 'study_id': int(study_id),
            'image_paths': [img['image_path'] for img in selected],
            'views': [img['view'] for img in selected],
            'n_images': len(selected),
            'section_indication': sections.get('indication', ''),
            'section_findings': sections.get('findings', ''),
        })

    # Every study must have a report
    assert missing_reports == 0, f"Error: {missing_reports} missing reports, test population changed"
    log(f"  Final: {len(manifest)} studies, 0 missing reports ")
    json.dump(manifest, open(manifest_path, "w"), indent=2)
    return manifest


# Step 1b: generate baseline reports with CheXagent-8B
def generate_reports(manifest):
    reports_path = EXP_DIR / "chexagent_8b_reports.json"
    checkpoint_path = EXP_DIR / "chexagent_8b_reports_checkpoint.json"

    if reports_path.exists():
        log(f"Loading completed reports: {reports_path}")
        return json.load(open(reports_path))

    # Resume from checkpoint
    results = []
    start_idx = 0
    if checkpoint_path.exists():
        results = json.load(open(checkpoint_path))
        start_idx = len(results)
        log(f"Resuming from checkpoint: {start_idx}/{len(manifest)}")

    log(f"\n{'='*60}")
    log("CheXagent-8B report generation")
    log("  do_sample=False, num_beams=1, max_new_tokens=512")
    log(f"  max_images={MAX_IMAGES}, decode=full(skip_special_tokens)")
    log(f"  Starting from index {start_idx}")
    log(f"{'='*60}")

    from transformers import AutoModelForCausalLM, AutoProcessor

    model = AutoModelForCausalLM.from_pretrained(
        CHECKPOINT, torch_dtype=DTYPE, trust_remote_code=True
    ).to(DEVICE)
    model.eval()
    proc = AutoProcessor.from_pretrained(CHECKPOINT, trust_remote_code=True)
    tok = proc.tokenizer
    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id

    errors = 0
    t0 = time.time()

    for i in range(start_idx, len(manifest)):
        sample = manifest[i]
        try:
            images = [Image.open(p).convert("RGB") for p in sample['image_paths']]

            # Prompt and chat template
            indication = sample['section_indication'] or "None provided"
            raw_prompt = f'Given the indication: "{indication}", write a structured findings section for the CXR.'
            prompt = f' USER: <s>{raw_prompt} ASSISTANT: <s>'

            inputs = proc(images=images, text=prompt, return_tensors='pt')
            for k, v in inputs.items():
                if isinstance(v, torch.Tensor):
                    inputs[k] = v.to(DEVICE) if v.dtype in (torch.long, torch.int) else v.to(DEVICE, dtype=DTYPE)

            with torch.no_grad():
                output_ids = model.generate(
                    **inputs,
                    do_sample=False,
                    num_beams=1,
                    max_new_tokens=512,
                    temperature=1.0,
                    top_p=1.0,
                    use_cache=True,
                    pad_token_id=tok.pad_token_id,
                )

            # Decode the full output; skip_special_tokens removes the prompt and control tokens.
            # Slicing by input length does not work for CheXagent-8B: generate() replaces input_ids
            # with visual embeddings, so the prompt length is not the offset of the generated text.
            text = tok.decode(output_ids[0], skip_special_tokens=True).strip()

            results.append({
                'study_id': sample['study_id'],
                'subject_id': sample['subject_id'],
                'section_findings': sample['section_findings'],
                'candidate_findings': text,
            })

        except Exception as e:
            errors += 1
            log(f"  Error study {sample['study_id']}: {e}")
            # Record the error instead of writing an empty report
            results.append({
                'study_id': sample['study_id'],
                'subject_id': sample['subject_id'],
                'section_findings': sample['section_findings'],
                'candidate_findings': f"__ERROR__: {str(e)}",
                '_error': True,
            })

        done = i + 1
        if done % 50 == 0:
            json.dump(results, open(checkpoint_path, "w"), indent=2, ensure_ascii=False)
        if done % 100 == 0:
            elapsed = time.time() - t0
            rate = (done - start_idx) / elapsed if elapsed > 0 else 0
            eta = (len(manifest) - done) / rate if rate > 0 else 0
            log(f"  {done}/{len(manifest)} ({rate:.1f}/s, ETA:{eta/60:.0f}m, err:{errors})")

    # Stop if any generation failed
    log(f"  Generation complete: {len(results)} reports, {errors} errors")
    assert errors == 0, f"Error: {errors} generation errors, cannot produce the baseline"

    json.dump(results, open(reports_path, "w"), indent=2, ensure_ascii=False)
    if checkpoint_path.exists():
        checkpoint_path.unlink()

    del model, proc; torch.cuda.empty_cache(); gc.collect()
    return results


# Step 1c: evaluate the baseline with CheXbert
def evaluate_reports(results):
    log(f"\n{'='*60}")
    log("F1CheXbert evaluation")
    log(f"{'='*60}")

    # Text cleaning before CheXbert labelling
    clean = lambda x: re.sub(r"\s+", " ", re.sub(r"\[.*?\]", "", x).replace("**", "")).strip().lower()

    candidates = [clean(s['candidate_findings']) for s in results]
    references = [clean(s['section_findings']) for s in results]

    # Filter empty references
    pairs = [(c, r) for c, r in zip(candidates, references) if r]
    cands = [p[0] for p in pairs]
    refs  = [p[1] for p in pairs]

    log(f"  Total: {len(results)}, with ref: {len(pairs)}, skipped: {len(results)-len(pairs)}")

    from f1chexbert import F1CheXbert
    scorer = F1CheXbert(device=DEVICE)

    # Convert NumPy types for JSON
    def to_json(obj):
        if isinstance(obj, dict): return {k: to_json(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)): return [to_json(v) for v in obj]
        if isinstance(obj, np.ndarray): return obj.tolist()
        if isinstance(obj, (np.floating, np.integer)): return float(obj)
        return obj

    # F1CheXbert scores
    log("  Running the F1CheXbert scorer...")
    official_scores = to_json(scorer(refs, cands))
    log("\n  F1CheXbert results")
    log(f"  {json.dumps(official_scores, indent=2)}")

    # Per-pathology TP, FP and FN
    LABELS = ["Enlarged Cardiomediastinum","Cardiomegaly","Lung Opacity",
        "Lung Lesion","Edema","Consolidation","Pneumonia","Atelectasis",
        "Pneumothorax","Pleural Effusion","Pleural Other","Fracture",
        "Support Devices","No Finding"]

    log("\n  Per-pathology diagnostic analysis...")
    ref_all, cand_all = [], []
    for i in range(len(refs)):
        r_lab = scorer.get_label(refs[i], mode='rrg')
        c_lab = scorer.get_label(cands[i], mode='rrg')
        ref_all.append([int(r_lab[j]) if j < len(r_lab) else 0 for j in range(14)])
        cand_all.append([int(c_lab[j]) if j < len(c_lab) else 0 for j in range(14)])
        if (i+1) % 500 == 0:
            log(f"    Labeled {i+1}/{len(refs)}")

    ref_arr = np.array(ref_all)
    cand_arr = np.array(cand_all)

    log(f"\n  {'Pathology':<35} {'TP':>5} {'FP':>5} {'FN':>5} {'F1':>7}")
    log(f"  {'-'*60}")
    per_f1 = {}
    for j, name in enumerate(LABELS):
        tp = int(((ref_arr[:,j]==1) & (cand_arr[:,j]==1)).sum())
        fp = int(((ref_arr[:,j]==0) & (cand_arr[:,j]==1)).sum())
        fn = int(((ref_arr[:,j]==1) & (cand_arr[:,j]==0)).sum())
        f1 = 2*tp/(2*tp+fp+fn) if (2*tp+fp+fn)>0 else 0.0
        per_f1[name] = f1
        log(f"  {name:<35} {tp:>5} {fp:>5} {fn:>5} {f1:>7.4f}")

    # F1 on the five steered pathologies
    TARG5 = ["Atelectasis","Cardiomegaly","Lung Opacity","Pleural Effusion","Edema"]
    target_f1 = np.mean([per_f1[l] for l in TARG5])
    log(f"\n  Target-pathology Macro-F1: {target_f1:.4f}")

    # No Finding analysis
    nf_tp = int(((ref_arr[:,13]==1) & (cand_arr[:,13]==1)).sum())
    nf_fp = int(((ref_arr[:,13]==0) & (cand_arr[:,13]==1)).sum())
    nf_fn = int(((ref_arr[:,13]==1) & (cand_arr[:,13]==0)).sum())
    nf_prec = nf_tp/(nf_tp+nf_fp) if (nf_tp+nf_fp)>0 else 0
    nf_rec  = nf_tp/(nf_tp+nf_fn) if (nf_tp+nf_fn)>0 else 0
    nf_f1   = 2*nf_tp/(2*nf_tp+nf_fp+nf_fn) if (2*nf_tp+nf_fp+nf_fn)>0 else 0
    log(f"  No Finding, P:{nf_prec:.3f} R:{nf_rec:.3f} F1:{nf_f1:.3f}")

    json.dump({
        'model': CHECKPOINT,
        'protocol': 'do_sample=False, num_beams=1, max_new_tokens=512, full decode',
        'image_selection': f'up to {MAX_IMAGES} images per study, frontal views first',
        'n_studies': len(results),
        'n_evaluated': len(pairs),
        'official_f1chexbert': official_scores,
        'per_pathology_f1': per_f1,
        'target_pathology_macro_f1': float(target_f1),
        'no_finding': {'precision': nf_prec, 'recall': nf_rec, 'f1': nf_f1},
    }, open(EXP_DIR / "chexagent_8b_eval.json", "w"), indent=2)

    del scorer; torch.cuda.empty_cache(); gc.collect()
    return official_scores


if __name__ == "__main__":
    log(f"\n{'='*60}")
    log("Test set and baseline reports: CheXagent-8B")
    log(f"{'='*60}")

    manifest = build_study_manifest()
    results = generate_reports(manifest)
    scores = evaluate_reports(results)

    log("\nDone, results in chexagent_8b_eval.json")
