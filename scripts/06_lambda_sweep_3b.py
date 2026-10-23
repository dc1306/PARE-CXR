#!/usr/bin/env python3
"""Step 6: select the steering strength lambda for CheXagent-3B.

Evaluates lambda in {0.05, 0.1, 0.13, 0.15, 0.2, 0.3, 0.5} on the first 200
test studies with a non-empty findings section and selects the value with the
highest merged Steer-5 F1 (merger without the negation check, safe_merge).
The selected value is lambda = 0.13. These 200 studies are reserved for this
selection and are excluded from the evaluation in step 9.

Output ($PARE_OUT/lambda_sweep_3b/):
  lambda_sweep_results.json

Usage:
  python scripts/06_lambda_sweep_3b.py
"""
import sys as _sys; from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))  # repo root
from pare import paths
import os

import torch, json, time, gc, warnings, pickle, re
import numpy as np
from PIL import Image
warnings.filterwarnings('ignore')

# Settings
TARGET_5 = ['Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema', 'Pleural Effusion']
LAMBDAS = [0.05, 0.1, 0.13, 0.15, 0.2, 0.3, 0.5]
N_SWEEP = 200
CANON = paths.out("test_set_and_baseline_8b")
COMP_DIR = paths.out("baseline_and_components_3b")
EXP_DIR = paths.out("lambda_sweep_3b")
DEVICE = "cuda:0"; DTYPE = torch.bfloat16

LABELS_14 = ["Enlarged Cardiomediastinum","Cardiomegaly","Lung Opacity","Lung Lesion",
    "Edema","Consolidation","Pneumonia","Atelectasis","Pneumothorax",
    "Pleural Effusion","Pleural Other","Fracture","Support Devices","No Finding"]
ALL_13 = [l for l in LABELS_14 if l != "No Finding"]

LOG_FILE = EXP_DIR / "lambda_sweep.log"
def log(msg):
    ts = time.strftime("%Y-%m-%d %H:%M:%S"); line = f"[{ts}] {msg}"
    print(line, flush=True)
    with open(LOG_FILE, "a") as f: f.write(line + "\n")

clean = lambda x: re.sub(r"\s+", " ", re.sub(r"\[.*?\]", "", x).replace("**", "")).strip()

def build_input_ids(tok, image_path, indication, device):
    prompt = (
        f'Given the indication: "{indication}", '
        "write a structured findings section for the CXR."
    )
    query = tok.from_list_format([
        {"image": image_path},
        {"text": prompt},
    ])
    conv = [
        {"from": "system", "value": "You are a helpful assistant."},
        {"from": "human", "value": query},
    ]
    input_ids = tok.apply_chat_template(conv, add_generation_prompt=True, return_tensors="pt")
    return input_ids.to(device)

def compute_delta(l_steer_tokens, targets, monge_maps, lam):
    n_vis, hidden = l_steer_tokens.shape
    total = torch.zeros(n_vis, hidden, dtype=torch.float32)
    for p in targets:
        mm = monge_maps[p]
        tokens = l_steer_tokens.float()
        x_sc = (tokens - torch.tensor(mm['sc_mean'])) / (torch.tensor(mm['sc_scale']) + 1e-8)
        x_pca = (x_sc - torch.tensor(mm['pca_mean'])) @ torch.tensor(mm['pca_comp']).T
        transported = torch.tensor(mm['mu_t']) + (x_pca - torch.tensor(mm['mu_s'])) @ torch.tensor(mm['A']).T
        delta = (transported - x_pca) @ torch.tensor(mm['pca_comp']) * (torch.tensor(mm['sc_scale']) + 1e-8)
        total += delta * lam
    return total

def main():
    log(f"\n{'='*60}")
    log("Lambda sweep, CheXagent-3B")
    log(f"  Targets: {TARGET_5}")
    log(f"  Lambdas: {LAMBDAS}")
    log(f"  N_sweep: {N_SWEEP}")
    log(f"{'='*60}")

    # Load components
    comp = pickle.load(open(COMP_DIR / 'components_3b.pkl', 'rb'))
    CHECKPOINT = comp['checkpoint']
    assert CHECKPOINT == "StanfordAIMI/CheXagent-2-3b"
    STEER_LAYER = comp['steer_layer']; PROBE_LAYER = comp['probe_layer']
    N_VIS = comp['n_vis']; IMG_TOKEN_ID = comp['img_token_id']
    HIDDEN_DIM = comp['hidden_dim']
    probes = comp['probes']; thresholds = comp['thresholds']
    monge_maps = comp['monge_maps']; sc_probe = comp['sc_probe']
    log(f"  Config: steer_L{STEER_LAYER}, probe_L{PROBE_LAYER}, N_VIS={N_VIS}, hidden={HIDDEN_DIM}")

    # Load manifest + baseline
    manifest = json.load(open(CANON / "study_test_manifest.json"))
    bl_saved = json.load(open(COMP_DIR / "baseline_reports.json"))
    baseline_reports = bl_saved['results']
    bl_by = {r['study_id']: r for r in baseline_reports}

    # Select subset with reference findings
    ref_studies = [s for s in manifest if s.get('section_findings', '').strip()]
    sweep_studies = ref_studies[:N_SWEEP]
    log(f"  Selected {len(sweep_studies)} studies with reference findings")

    # Load model
    from transformers import AutoModelForCausalLM, AutoTokenizer
    model = AutoModelForCausalLM.from_pretrained(CHECKPOINT, trust_remote_code=True).to(DTYPE).to(DEVICE)
    model.eval()
    tok = AutoTokenizer.from_pretrained(CHECKPOINT, trust_remote_code=True)
    if tok.pad_token_id is None: tok.pad_token_id = tok.eos_token_id
    dec_layers = model.model.layers
    log("Model loaded")

    results_by_lambda = {lam: [] for lam in LAMBDAS}
    baseline_texts = []
    reference_texts = []
    tmp_path = f"/tmp/_sweep_3b_{os.getpid()}.png"

    t0 = time.time()
    for i, sample in enumerate(sweep_studies):
        sid = sample['study_id']
        try:
            indication = sample.get('section_indication') or "None provided"
            Image.open(sample['image_paths'][0]).convert('RGB').save(tmp_path)
            input_ids = build_input_ids(tok, tmp_path, indication, DEVICE)

            img_mask = (input_ids[0] == IMG_TOKEN_ID)

            # Diagnostic pass (once per study)
            captured = {'steer': None, 'probe': None}
            def make_capture(name):
                def hook_fn(module, inp, output):
                    h = output[0] if isinstance(output, tuple) else output
                    if h.shape[1] > 1:
                        vis_h = h[0, img_mask, :].detach().cpu().float()
                        if name == 'steer':
                            captured[name] = vis_h
                        else:
                            captured[name] = vis_h.mean(dim=0, keepdim=True)
                    return output
                return hook_fn

            h_steer = dec_layers[STEER_LAYER].register_forward_hook(make_capture('steer'))
            h_probe = dec_layers[PROBE_LAYER].register_forward_hook(make_capture('probe'))
            try:
                with torch.no_grad(): _ = model(input_ids, output_hidden_states=False)
            finally: h_steer.remove(); h_probe.remove()

            assert captured['steer'] is not None and captured['probe'] is not None

            # Probe scores (once per study)
            probe_scaled = sc_probe.transform(captured['probe'].numpy())
            targets = []
            for p in TARGET_5:
                score = float(probes[p].predict_proba(probe_scaled)[0, 1])
                if score >= thresholds[p]:
                    targets.append(p)

            baseline_texts.append(clean(bl_by[sid].get('candidate_findings', '')).lower())
            reference_texts.append(clean(sample['section_findings']).lower())

            if not targets:
                bl_text = bl_by[sid].get('candidate_findings', '')
                for lam in LAMBDAS:
                    results_by_lambda[lam].append({
                        'study_id': sid, 'steered': False, 'text': bl_text,
                        'n_targets': 0, 'delta_norm': 0.0
                    })
            else:
                vis_positions = torch.where(img_mask)[0]
                for lam in LAMBDAS:
                    delta = compute_delta(captured['steer'], targets, monge_maps, lam)
                    delta_norm = float(delta.norm())

                    fired = [False]
                    def make_steer_hook(d, flag, vis_pos):
                        def hook_fn(module, inp, output):
                            if flag[0]: return output
                            h = output[0] if isinstance(output, tuple) else output
                            if h.shape[1] > 1:
                                flag[0] = True
                                hm = h.clone()
                                n_apply = min(len(vis_pos), d.shape[0])
                                hm[0, vis_pos[:n_apply], :] += d[:n_apply].to(device=h.device, dtype=h.dtype)
                                return (hm,) + output[1:] if isinstance(output, tuple) else hm
                            return output
                        return hook_fn

                    hook = dec_layers[STEER_LAYER].register_forward_hook(
                        make_steer_hook(delta, fired, vis_positions))
                    try:
                        with torch.no_grad():
                            output_ids = model.generate(input_ids, do_sample=False, num_beams=1,
                                max_new_tokens=512, use_cache=True, pad_token_id=tok.pad_token_id)
                    finally: hook.remove()

                    text = tok.decode(output_ids[0][input_ids.shape[1]:], skip_special_tokens=True).strip()
                    results_by_lambda[lam].append({
                        'study_id': sid, 'steered': True, 'text': text,
                        'n_targets': len(targets), 'delta_norm': delta_norm,
                        'targets': targets,
                    })

        except Exception as e:
            log(f"  Error in {sid}: {e}")
            bl_text = bl_by[sid].get('candidate_findings', '')
            baseline_texts.append(clean(bl_text).lower())
            reference_texts.append(clean(sample.get('section_findings', '')).lower())
            for lam in LAMBDAS:
                results_by_lambda[lam].append({
                    'study_id': sid, 'steered': False, 'text': bl_text,
                    'n_targets': 0, 'delta_norm': 0.0, '_error': str(e),
                })
        finally:
            if os.path.exists(tmp_path): os.remove(tmp_path)

        if (i+1) % 20 == 0:
            elapsed = time.time() - t0
            rate = (i+1) / elapsed
            eta = (len(sweep_studies) - i - 1) / rate
            log(f"  {i+1}/{len(sweep_studies)} ({rate:.3f}/s, ETA:{eta/60:.0f}m)")

    del model; torch.cuda.empty_cache(); gc.collect()

    # Save the generated reports before evaluation
    ckpt_data = {}
    for lam in LAMBDAS:
        ckpt_data[str(lam)] = results_by_lambda[lam]
    json.dump({'baseline_texts': baseline_texts, 'reference_texts': reference_texts,
               'results_by_lambda': ckpt_data},
              open(EXP_DIR / 'sweep_checkpoint.json', 'w'))
    log(f"Checkpoint saved: {EXP_DIR / 'sweep_checkpoint.json'}")
    log("Model unloaded, starting evaluation...")

    # Evaluate with CheXbert
    from f1chexbert import F1CheXbert
    scorer = F1CheXbert(device='cuda:0')

    def label_texts(texts):
        out = []
        for t in texts:
            tc = t.strip()
            if not tc: out.append([0]*14); continue
            raw = scorer.get_label(tc, mode='rrg')
            out.append([int(raw[j]) if j < len(raw) and raw[j] != '' else 0 for j in range(14)])
        return np.array(out)

    ref_labels = label_texts(reference_texts)
    bl_labels = label_texts(baseline_texts)

    def compute_metrics(pred, ref):
        s5_f1s = []; all_f1s = []
        for j, label in enumerate(LABELS_14):
            tp = int(((ref[:,j]==1)&(pred[:,j]==1)).sum())
            fp = int(((ref[:,j]==0)&(pred[:,j]==1)).sum())
            fn = int(((ref[:,j]==1)&(pred[:,j]==0)).sum())
            f1 = 2*tp/(2*tp+fp+fn) if (2*tp+fp+fn) > 0 else 0
            if label in TARGET_5: s5_f1s.append(f1)
            if label != "No Finding": all_f1s.append(f1)
        mtp = sum(int(((ref[:,j]==1)&(pred[:,j]==1)).sum()) for j in range(14))
        mfp = sum(int(((ref[:,j]==0)&(pred[:,j]==1)).sum()) for j in range(14))
        mfn = sum(int(((ref[:,j]==1)&(pred[:,j]==0)).sum()) for j in range(14))
        mi14 = 2*mtp/(2*mtp+mfp+mfn) if (2*mtp+mfp+mfn) > 0 else 0
        return np.mean(s5_f1s), np.mean(all_f1s), mi14

    bl_s5, bl_a13, bl_mi14 = compute_metrics(bl_labels, ref_labels)
    log(f"\n  Baseline: S5={bl_s5:.4f} A13={bl_a13:.4f} M14={bl_mi14:.4f}")

    from pare.safe_merger import safe_merge

    best_lam = None; best_mg_s5 = -1
    sweep_results = {}

    for lam in LAMBDAS:
        results = results_by_lambda[lam]
        pare_texts = [clean(r['text']).lower() for r in results]
        pa_labels = label_texts(pare_texts)
        pa_s5, pa_a13, pa_mi14 = compute_metrics(pa_labels, ref_labels)

        merged_texts = []
        for r in results:
            if r['steered']:
                fired = r.get('targets', TARGET_5)
                merged_text, merge_actions = safe_merge(
                    bl_by[r['study_id']].get('candidate_findings',''),
                    r['text'], fired, scorer)
                merged_texts.append(clean(merged_text).lower())
            else:
                merged_texts.append(clean(r['text']).lower())
        mg_labels = label_texts(merged_texts)
        mg_s5, mg_a13, mg_mi14 = compute_metrics(mg_labels, ref_labels)

        steered = [r for r in results if r['steered']]
        mean_norm = np.mean([r['delta_norm'] for r in steered]) if steered else 0

        log(f"  λ={lam:5.3f}: PA S5={pa_s5:.4f}({pa_s5-bl_s5:+.4f}) A13={pa_a13:.4f}({pa_a13-bl_a13:+.4f}) "
            f"| MG S5={mg_s5:.4f}({mg_s5-bl_s5:+.4f}) A13={mg_a13:.4f}({mg_a13-bl_a13:+.4f}) "
            f"| norm={mean_norm:.0f}")

        sweep_results[str(lam)] = {
            'lambda': lam, 'mean_delta_norm': mean_norm,
            'n_steered': len(steered), 'n_total': len(results),
            'raw': {'steer_5_f1': pa_s5, 'all_13_f1': pa_a13, 'micro_14': pa_mi14},
            'merged': {'steer_5_f1': mg_s5, 'all_13_f1': mg_a13, 'micro_14': mg_mi14},
            'delta_raw': {'s5': pa_s5-bl_s5, 'a13': pa_a13-bl_a13, 'mi14': pa_mi14-bl_mi14},
            'delta_merged': {'s5': mg_s5-bl_s5, 'a13': mg_a13-bl_a13, 'mi14': mg_mi14-bl_mi14},
        }

        if mg_s5 > best_mg_s5:
            best_mg_s5 = mg_s5; best_lam = lam

    log(f"\n  Best λ = {best_lam} (merged Steer-5 F1 = {best_mg_s5:.4f})")

    output = {
        'model': '3B', 'checkpoint': CHECKPOINT,
        'n_sweep': len(sweep_studies), 'targets': TARGET_5,
        'n_vis': N_VIS, 'hidden_dim': HIDDEN_DIM,
        'baseline': {'steer_5_f1': bl_s5, 'all_13_f1': bl_a13, 'micro_14': bl_mi14},
        'best_lambda': best_lam, 'lambdas': sweep_results,
    }
    json.dump(output, open(EXP_DIR / "lambda_sweep_results.json", 'w'), indent=2)
    log("\nLambda sweep complete, saved to lambda_sweep_results.json")

if __name__ == "__main__":
    main()
