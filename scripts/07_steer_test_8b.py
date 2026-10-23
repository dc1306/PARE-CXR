#!/usr/bin/env python3
"""Step 7: steered generation on the test set, CheXagent-8B (lambda = 1.0).

Steers the five evaluated pathologies (Atelectasis, Cardiomegaly,
Consolidation, Edema, Pleural Effusion) using the components of step 3.
A diagnostic pass computes the probe scores; if no probe fires, the baseline
report of step 1 is reused. Otherwise the summed transport displacements are
added to the layer-3 visual tokens during the prefill of a second pass.

Output ($PARE_OUT/steered_8b/):
  test_reports.json

Usage:
  python scripts/07_steer_test_8b.py
"""
import sys as _sys; from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))  # repo root
from pare import paths
import torch, json, time, gc, warnings
import numpy as np
from PIL import Image
from collections import Counter, defaultdict
import pickle
warnings.filterwarnings('ignore')

# Settings
TARGET_5 = ['Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema', 'Pleural Effusion']
BASELINE_DIR = paths.out("test_set_and_baseline_8b")
COMP_DIR     = paths.out("components_8b")
EXP_DIR      = paths.out("steered_8b")
DEVICE = "cuda:0"; DTYPE = torch.float16; CHECKPOINT = "StanfordAIMI/CheXagent-8b"
N_VIS = 128; STEER_LAYER = 3; PROBE_LAYER = 31; BEST_LAMBDA = 1.0

LOG_FILE = EXP_DIR / "test.log"
def log(msg):
    ts = time.strftime("%Y-%m-%d %H:%M:%S"); line = f"[{ts}] {msg}"
    print(line, flush=True)
    with open(LOG_FILE, "a") as f: f.write(line + "\n")

def compute_delta(l3_tokens, targets, monge_maps, lam):
    """Sum of per-pathology Monge deltas."""
    total = torch.zeros(N_VIS, 4096, dtype=torch.float32)
    for p in targets:
        mm = monge_maps[p]
        tokens = l3_tokens[:N_VIS].float()
        x_sc = (tokens - torch.tensor(mm['sc_mean'])) / (torch.tensor(mm['sc_scale']) + 1e-8)
        x_pca = (x_sc - torch.tensor(mm['pca_mean'])) @ torch.tensor(mm['pca_comp']).T
        transported = torch.tensor(mm['mu_t']) + (x_pca - torch.tensor(mm['mu_s'])) @ torch.tensor(mm['A']).T
        delta = (transported - x_pca) @ torch.tensor(mm['pca_comp']) * (torch.tensor(mm['sc_scale']) + 1e-8)
        total += delta * lam
    return total

def main():
    log(f"\n{'='*60}")
    log("Steered generation: CheXagent-8B")
    log(f"  Targets: {TARGET_5}")
    log("  Gate: probe score >= threshold")
    log(f"{'='*60}")

    # Load manifest + baseline
    manifest = json.load(open(BASELINE_DIR / "study_test_manifest.json"))
    baseline_reports = json.load(open(BASELINE_DIR / "chexagent_8b_reports.json"))
    bl_by = {r['study_id']: r for r in baseline_reports}
    m_ids = {s['study_id'] for s in manifest}
    b_ids = {r['study_id'] for r in baseline_reports}
    assert m_ids == b_ids, "ID mismatch"
    log(f"Loaded: {len(manifest)} test studies, baseline verified ")

    # Load the step-3 components; steer only the five target pathologies
    comp = pickle.load(open(COMP_DIR / 'components_8b.pkl', 'rb'))
    assert comp['checkpoint'] == CHECKPOINT
    assert comp['steer_layer'] == STEER_LAYER
    assert comp['probe_layer'] == PROBE_LAYER
    assert comp['n_vis'] == N_VIS
    assert comp['pca_dims'] == 128
    assert comp['lambda'] == BEST_LAMBDA
    comp_hash = comp['train_manifest_hash']
    probes = comp['probes']; thresholds = comp['thresholds']
    monge_maps = comp['monge_maps']; sc31 = comp['sc31']

    # Validate all 5 targets available
    for p in TARGET_5:
        assert p in probes, f"Missing probe for {p}"
        assert p in monge_maps, f"Missing transport map for {p}"
        assert p in thresholds, f"Missing threshold for {p}"
    log(f"Components loaded, hash={comp_hash[:8]}")
    log("  All 5 target pathologies available")
    for p in TARGET_5:
        log(f"    {p:30s} τ={thresholds[p]:.4f}")

    # Check for valid cached results
    results_path = EXP_DIR / "test_reports.json"
    if results_path.exists():
        cached = json.load(open(results_path))
        cached_prov = cached[0].get('_provenance', {}) if cached else {}
        n_errors = sum(1 for r in cached if '_error' in r)
        cached_ids = {r['study_id'] for r in cached}
        if (len(cached) == len(manifest) and n_errors == 0
            and cached_ids == m_ids
            and cached_prov.get('comp_hash') == comp_hash
            and cached_prov.get('experiment') == 'steer_8b'):
            log(f"Loading validated cached results: {len(cached)} studies ")
            return cached
        else:
            log("Cached results invalid, regenerating")

    # Resume
    results = []; start_idx = 0
    ckpt_path = EXP_DIR / "test_checkpoint.json"
    if ckpt_path.exists():
        ckpt = json.load(open(ckpt_path))
        ckpt_prov = ckpt['results'][0].get('_provenance', {}) if ckpt['results'] else {}
        if ckpt_prov.get('experiment') == 'steer_8b':
            results = ckpt['results']; start_idx = len(results)
            log(f"Resuming from {start_idx}")
        else:
            log("Stale checkpoint, discarding")

    # Load model
    from transformers import AutoModelForCausalLM, AutoProcessor
    model = AutoModelForCausalLM.from_pretrained(CHECKPOINT, torch_dtype=DTYPE, trust_remote_code=True).to(DEVICE)
    model.eval()
    proc = AutoProcessor.from_pretrained(CHECKPOINT, trust_remote_code=True)
    tok = proc.tokenizer
    if tok.pad_token_id is None: tok.pad_token_id = tok.eos_token_id
    dec_layers = model.language_model.model.layers

    log("  Generation: do_sample=False, num_beams=1, max_new_tokens=512")
    log(f"  PARE: L{STEER_LAYER} steer, L{PROBE_LAYER} probe, λ={BEST_LAMBDA}")

    errors = 0; steered_count = 0; unsteered_count = 0; t0 = time.time()

    for i in range(start_idx, len(manifest)):
        sample = manifest[i]
        sid = sample['study_id']
        try:
            images = [Image.open(p).convert("RGB") for p in sample['image_paths']]
            indication = sample['section_indication'] or "None provided"
            prompt = f' USER: <s>Given the indication: "{indication}", write a structured findings section for the CXR. ASSISTANT: <s>'
            inputs = proc(images=images, text=prompt, return_tensors='pt')
            for k, v in inputs.items():
                if isinstance(v, torch.Tensor):
                    inputs[k] = v.to(DEVICE) if v.dtype in (torch.long, torch.int) else v.to(DEVICE, dtype=DTYPE)

            # Diagnostic pass: capture layer-3 visual tokens and layer-31 features
            captured = {'l3': None, 'l31': None}
            def make_capture(name, vis_only):
                def hook_fn(module, inp, output):
                    h = output[0] if isinstance(output, tuple) else output
                    if h.shape[1] > 1:
                        if vis_only: captured[name] = h[0, :N_VIS, :].detach().cpu().float()
                        else: captured[name] = h[0].detach().cpu().float().mean(dim=0, keepdim=True)
                    return output
                return hook_fn

            h3 = dec_layers[STEER_LAYER].register_forward_hook(make_capture('l3', True))
            h31 = dec_layers[PROBE_LAYER].register_forward_hook(make_capture('l31', False))
            try:
                with torch.no_grad(): _ = model(**{k: v for k, v in inputs.items()})
            finally: h3.remove(); h31.remove()

            assert captured['l3'] is not None and captured['l31'] is not None

            # Gate: steer pathology p if its probe score reaches the threshold
            l31_scaled = sc31.transform(captured['l31'].numpy())
            targets = []; probe_scores = {}
            for p in TARGET_5:
                score = float(probes[p].predict_proba(l31_scaled)[0, 1])
                probe_scores[p] = score
                if score >= thresholds[p]:
                    targets.append(p)

            if targets:
                steered_count += 1
                delta = compute_delta(captured['l3'], targets, monge_maps, BEST_LAMBDA)
                delta_norm = float(delta.norm())

                fired = [False]
                def make_steer_hook(d, flag):
                    def hook_fn(module, inp, output):
                        if flag[0]: return output
                        h = output[0] if isinstance(output, tuple) else output
                        if h.shape[1] > 1:
                            flag[0] = True
                            hm = h.clone()
                            hm[0, :N_VIS, :] += d.to(device=h.device, dtype=h.dtype)
                            return (hm,) + output[1:] if isinstance(output, tuple) else hm
                        return output
                    return hook_fn

                hook = dec_layers[STEER_LAYER].register_forward_hook(make_steer_hook(delta, fired))
                try:
                    with torch.no_grad():
                        output_ids = model.generate(**inputs, do_sample=False, num_beams=1,
                            max_new_tokens=512, temperature=1.0, top_p=1.0, use_cache=True,
                            pad_token_id=tok.pad_token_id)
                finally: hook.remove()
                assert fired[0], f"Steering hook did not fire for {sid}"
                text = tok.decode(output_ids[0], skip_special_tokens=True).strip()
            else:
                unsteered_count += 1
                text = bl_by[sid].get('candidate_findings', '')
                delta_norm = 0.0

            results.append({
                'study_id': sid, 'subject_id': sample['subject_id'],
                'section_findings': sample.get('section_findings', ''),
                'candidate_findings': text, 'steered': bool(targets),
                'targets': targets, 'n_targets': len(targets),
                'probe_scores': probe_scores, 'delta_norm': delta_norm,
                '_provenance': {'comp_hash': comp_hash, 'lambda': BEST_LAMBDA,
                                'experiment': 'steer_8b', 'targets': TARGET_5},
            })
        except Exception as e:
            errors += 1; log(f"  Error in {sid}: {e}")
            results.append({
                'study_id': sid, 'subject_id': sample['subject_id'],
                'section_findings': sample.get('section_findings', ''),
                'candidate_findings': bl_by[sid].get('candidate_findings', ''),
                'steered': False, 'targets': [], 'n_targets': 0,
                '_error': str(e), 'probe_scores': {}, 'delta_norm': 0.0,
                '_provenance': {'comp_hash': comp_hash, 'lambda': BEST_LAMBDA,
                                'experiment': 'steer_8b', 'targets': TARGET_5},
            })

        done = i + 1
        if done % 50 == 0:
            json.dump({'results': results}, open(ckpt_path, 'w'))
        if done % 100 == 0:
            elapsed = time.time() - t0; rate = (done - start_idx) / elapsed if elapsed > 0 else 0
            eta = (len(manifest) - done) / rate if rate > 0 else 0
            avg_t = np.mean([r['n_targets'] for r in results if r['steered']]) if steered_count > 0 else 0
            log(f"  {done}/{len(manifest)} ({rate:.1f}/s, ETA:{eta/60:.0f}m, "
                f"steered:{steered_count}, unsteered:{unsteered_count}, avg_t:{avg_t:.1f}, err:{errors})")

    assert errors == 0, f"Error: {errors} errors"
    json.dump(results, open(results_path, 'w'), indent=2, ensure_ascii=False)
    if ckpt_path.exists(): ckpt_path.unlink()

    # Summary
    log("\n  Steering summary:")
    log(f"    Steered: {steered_count}/{len(results)} ({100*steered_count/len(results):.1f}%)")
    log(f"    Unsteered: {unsteered_count}")
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

    del model, proc; torch.cuda.empty_cache(); gc.collect()
    log(f"\nStep 7 complete: {len(results)} studies, {steered_count} steered, 0 errors")
    return results

if __name__ == "__main__":
    main()
