#!/usr/bin/env python3
"""Step 8: steered generation on the test set, CheXagent-3B (lambda = 0.13).

Same procedure as step 7, using the components of step 5 and the baseline
reports of step 4. lambda = 0.13 was selected in step 6.

Output ($PARE_OUT/steered_3b/):
  test_reports.json

Usage:
  python scripts/08_steer_test_3b.py
"""
import sys as _sys; from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))  # repo root
from pare import paths
import os

import torch, json, time, pickle, warnings, gc
import numpy as np
from PIL import Image
from collections import Counter, defaultdict
warnings.filterwarnings('ignore')

# Settings
TARGET_5 = ['Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema', 'Pleural Effusion']
CANON = paths.out("test_set_and_baseline_8b")
COMP_DIR = paths.out("baseline_and_components_3b")
EXP_DIR  = paths.out("steered_3b")
DEVICE = "cuda:0"; DTYPE = torch.bfloat16

LOG_FILE = EXP_DIR / "test.log"
def log(msg):
    ts = time.strftime("%Y-%m-%d %H:%M:%S"); line = f"[{ts}] {msg}"
    print(line, flush=True)
    with open(LOG_FILE, "a") as f: f.write(line + "\n")

def build_input_ids(tok, image_path, indication, device):
    """Documented CheXagent-3B chat-template protocol."""
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
    input_ids = tok.apply_chat_template(
        conv, add_generation_prompt=True, return_tensors="pt",
    )
    return input_ids.to(device)

def compute_delta(l_steer_tokens, targets, monge_maps, lam):
    """Per-token Monge delta across all visual tokens, summed over pathologies."""
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
    log("Steered generation: CheXagent-3B")
    log(f"  Targets: {TARGET_5}")
    log("  Gate: probe score >= threshold")
    log(f"{'='*60}")

    # Load and validate the step-5 components
    comp = pickle.load(open(COMP_DIR / 'components_3b.pkl', 'rb'))
    CHECKPOINT = comp['checkpoint']
    assert CHECKPOINT == "StanfordAIMI/CheXagent-2-3b", f"Wrong checkpoint: {CHECKPOINT}"
    STEER_LAYER = comp['steer_layer']; PROBE_LAYER = comp['probe_layer']
    N_VIS = comp['n_vis']; BEST_LAMBDA = 0.13  # selected by scripts/06_lambda_sweep_3b.py on the 200 reserved test studies
    IMG_TOKEN_ID = comp['img_token_id']
    probes = comp['probes']; thresholds = comp['thresholds']
    monge_maps = comp['monge_maps']; sc_probe = comp['sc_probe']

    for p in TARGET_5:
        assert p in probes, f"Missing probe for {p}"
        assert p in monge_maps, f"Missing transport map for {p}"
        assert p in thresholds, f"Missing threshold for {p}"

    comp_hash_path = COMP_DIR / 'components_3b_hash.json'
    if comp_hash_path.exists():
        comp_hash = json.load(open(comp_hash_path))['comp_hash']
    else:
        comp_hash = 'no_sidecar'
    log(f"Components loaded, hash={comp_hash}")
    log(f"Config: probe=L{PROBE_LAYER}, steer=L{STEER_LAYER}, λ={BEST_LAMBDA}, N_VIS={N_VIS}")
    for p in TARGET_5:
        log(f"    {p:30s} τ={thresholds[p]:.4f}")

    # Load test manifest + 3B baseline
    manifest = json.load(open(CANON / "study_test_manifest.json"))
    bl_saved = json.load(open(COMP_DIR / "baseline_reports.json"))
    assert isinstance(bl_saved, dict) and '_metadata' in bl_saved
    baseline_reports = bl_saved['results']
    bl_by = {r['study_id']: r for r in baseline_reports}
    m_ids = {s['study_id'] for s in manifest}
    b_ids = set(bl_by.keys())
    assert m_ids == b_ids, "ID mismatch"
    log(f"Loaded: {len(manifest)} studies, baseline verified ")

    # Check cache
    results_path = EXP_DIR / "test_reports.json"
    if results_path.exists():
        cached = json.load(open(results_path))
        cached_prov = cached[0].get('_provenance', {}) if cached else {}
        n_errors = sum(1 for r in cached if '_error' in r)
        cached_ids = {r['study_id'] for r in cached}
        if (len(cached) == len(manifest) and n_errors == 0
            and cached_ids == m_ids
            and cached_prov.get('comp_hash') == comp_hash
            and cached_prov.get('experiment') == 'steer_3b'):
            log(f"Loading validated cached results: {len(cached)} studies ")
            return cached

    # Resume
    results = []; start_idx = 0
    ckpt_path = EXP_DIR / "test_checkpoint.json"
    if ckpt_path.exists():
        ckpt = json.load(open(ckpt_path))
        ckpt_prov = ckpt['results'][0].get('_provenance', {}) if ckpt['results'] else {}
        if ckpt_prov.get('experiment') == 'steer_3b':
            results = ckpt['results']; start_idx = len(results)
            log(f"Resuming from {start_idx}")
        else:
            log("Stale checkpoint, discarding")

    # Load model
    from transformers import AutoModelForCausalLM, AutoTokenizer
    model = AutoModelForCausalLM.from_pretrained(CHECKPOINT, trust_remote_code=True).to(DTYPE).to(DEVICE)
    model.eval()
    tok = AutoTokenizer.from_pretrained(CHECKPOINT, trust_remote_code=True)
    if tok.pad_token_id is None: tok.pad_token_id = tok.eos_token_id
    dec_layers = model.model.layers
    log(f"Model loaded: {CHECKPOINT}")
    log("  Generation: do_sample=False, num_beams=1, max_new_tokens=512")

    errors = 0; steered_count = 0; unsteered_count = 0; t0 = time.time()

    for i in range(start_idx, len(manifest)):
        sample = manifest[i]
        sid = sample['study_id']
        tmp_path = f"/tmp/_pare_steer_3b_{os.getpid()}.png"
        try:
            indication = sample.get('section_indication') or "None provided"
            Image.open(sample['image_paths'][0]).convert('RGB').save(tmp_path)
            input_ids = build_input_ids(tok, tmp_path, indication, DEVICE)

            img_mask = (input_ids[0] == IMG_TOKEN_ID)
            n_vis_actual = int(img_mask.sum())
            if abs(n_vis_actual - N_VIS) > 5:
                raise ValueError(f"N_VIS mismatch: got {n_vis_actual}, expected ~{N_VIS}")

            # Diagnostic pass: capture the steering and probe layers
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

            # Gate: steer pathology p if its probe score reaches the threshold
            probe_scaled = sc_probe.transform(captured['probe'].numpy())
            targets = []; probe_scores = {}
            for p in TARGET_5:
                score = float(probes[p].predict_proba(probe_scaled)[0, 1])
                probe_scores[p] = score
                if score >= thresholds[p]:
                    targets.append(p)

            if targets:
                steered_count += 1
                delta = compute_delta(captured['steer'], targets, monge_maps, BEST_LAMBDA)
                delta_norm = float(delta.norm())

                fired = [False]
                vis_positions = torch.where(img_mask)[0]

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
                assert fired[0], f"Steering hook did not fire for {sid}"
                text = tok.decode(output_ids[0][input_ids.shape[1]:], skip_special_tokens=True).strip()
                assert text, f"Empty steered generation for {sid}"
            else:
                unsteered_count += 1
                text = bl_by[sid].get('candidate_findings', '')
                delta_norm = 0.0

            results.append({
                'study_id': sid, 'subject_id': sample.get('subject_id'),
                'section_findings': sample.get('section_findings', ''),
                'candidate_findings': text, 'steered': bool(targets),
                'targets': targets, 'n_targets': len(targets),
                'probe_scores': probe_scores, 'delta_norm': delta_norm,
                '_provenance': {'comp_hash': comp_hash, 'lambda': BEST_LAMBDA,
                                'experiment': 'steer_3b', 'checkpoint': CHECKPOINT,
                                'targets': TARGET_5},
            })
        except Exception as e:
            errors += 1; log(f"  Error {sid}: {e}")
            results.append({
                'study_id': sid, 'subject_id': sample.get('subject_id'),
                'section_findings': sample.get('section_findings', ''),
                'candidate_findings': bl_by[sid].get('candidate_findings', ''),
                'steered': False, 'targets': [], 'n_targets': 0,
                '_error': str(e), 'probe_scores': {}, 'delta_norm': 0.0,
                '_provenance': {'comp_hash': comp_hash, 'lambda': BEST_LAMBDA,
                                'experiment': 'steer_3b', 'checkpoint': CHECKPOINT,
                                'targets': TARGET_5},
            })
        finally:
            if os.path.exists(tmp_path): os.remove(tmp_path)

        done = i + 1
        if done % 25 == 0:
            json.dump({'results': results}, open(ckpt_path, 'w'))
        if done % 100 == 0:
            elapsed = time.time() - t0; rate = (done - start_idx) / elapsed if elapsed > 0 else 0
            eta = (len(manifest) - done) / rate if rate > 0 else 0
            log(f"  {done}/{len(manifest)} ({rate:.1f}/s, ETA:{eta/60:.0f}m, "
                f"steered:{steered_count}, unsteered:{unsteered_count}, err:{errors})")

    assert errors == 0, f"Error: {errors} errors"
    json.dump(results, open(results_path, 'w'), indent=1, ensure_ascii=False)
    if ckpt_path.exists(): ckpt_path.unlink()

    # Summary
    log("\n  Steering summary:")
    log(f"    Steered: {steered_count}/{len(results)} ({100*steered_count/len(results):.1f}%)")
    tc = Counter()
    for r in results:
        for t in r.get('targets', []): tc[t] += 1
    for p in TARGET_5:
        log(f"    {p:30s}: {tc.get(p,0):>5} ({100*tc.get(p,0)/len(results):.1f}%)")

    norms = defaultdict(list)
    for r in results:
        if r['steered']: norms[r['n_targets']].append(r['delta_norm'])
    log("\n  Displacement norms:")
    for k in sorted(norms.keys()):
        v = norms[k]
        log(f"    {k} targets: n={len(v)}, mean={np.mean(v):.2f}, max={np.max(v):.2f}")

    del model; torch.cuda.empty_cache(); gc.collect()
    log(f"\nStep 8 complete: {len(results)} studies, {steered_count} steered, 0 errors")

if __name__ == "__main__":
    main()
