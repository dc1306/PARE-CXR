#!/usr/bin/env python3
"""Step 5: fit the PARE components for CheXagent-3B.

Same procedure as step 3, with two model-specific differences:
  - CheXagent-3B has 996 SigLIP visual tokens, and all of them are steered at
    test time. To keep memory manageable, the scaler, PCA projection and
    covariance estimates are fitted on a fixed random subsample of studies and
    tokens.
  - Probe features are mean-pooled over the visual tokens only.
TP/FN groups are defined from the reference labels and the CheXagent-3B
baseline reports of step 4.

Output ($PARE_OUT/baseline_and_components_3b/):
  components_3b.pkl, components_3b_hash.json

Usage:
  python scripts/05_fit_components_3b.py
"""
import sys as _sys; from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))  # repo root
from pare import paths
import os

import torch, json, time, gc, warnings, pickle, hashlib
import pandas as pd, numpy as np
from PIL import Image
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.decomposition import PCA
from sklearn.covariance import LedoitWolf
from scipy.linalg import sqrtm, inv
warnings.filterwarnings('ignore')

# Paths
CHEXPERT_CSV = str(paths.CHEXPERT_CSV)
TRAINVAL_DIR = paths.out("trainval_features_8b")
EXP_DIR = paths.out("baseline_and_components_3b")
EXP_DIR.mkdir(parents=True, exist_ok=True)

# CheXagent-3B settings
CHECKPOINT = "StanfordAIMI/CheXagent-2-3b"
DEVICE = "cuda:0"; DTYPE = torch.bfloat16
IMG_TOKEN_ID = 50297
HIDDEN_DIM = 2560
PROBE_LAYER = 31; STEER_LAYER = 3
# All visual tokens are used, as for CheXagent-8B
PCA_DIMS = 128; BEST_LAMBDA = 1.0
MIN_STUDIES_FOR_MONGE = 50
MAX_TOKENS_FOR_FIT = 100000  # token cap for fitting the transport maps (memory)
PROTOCOL_VERSION = "chat_system_single_image"  # input protocol of step 4

PATHOLOGIES = [
    'Enlarged Cardiomediastinum', 'Cardiomegaly', 'Lung Opacity',
    'Lung Lesion', 'Edema', 'Consolidation', 'Pneumonia',
    'Atelectasis', 'Pneumothorax', 'Pleural Effusion',
    'Pleural Other', 'Fracture', 'Support Devices',
]
CHEXBERT_IDX = {p: i for i, p in enumerate(PATHOLOGIES)}
TARGET_5 = ['Atelectasis', 'Cardiomegaly', 'Lung Opacity', 'Pleural Effusion', 'Edema']

LOG_FILE = EXP_DIR / "train.log"
def log(msg):
    ts = time.strftime("%Y-%m-%d %H:%M:%S"); line = f"[{ts}] {msg}"
    print(line, flush=True)
    with open(LOG_FILE, "a") as f: f.write(line + "\n")

def chexpert_uones(x):
    if pd.isna(x): return 0
    return 1 if int(float(x)) in (1, -1) else 0

def study_id_hash(manifest):
    return hashlib.md5(json.dumps([s['study_id'] for s in manifest]).encode()).hexdigest()

def full_provenance_hash(manifest):
    """Hash including indication (part of the prompt) and image paths."""
    fp = [{'study_id': s['study_id'], 'subject_id': s['subject_id'],
           'image_paths': s['image_paths'],
           'indication': s.get('section_indication', '')} for s in manifest]
    return hashlib.md5(json.dumps(fp, sort_keys=True).encode()).hexdigest()


def build_input_ids(tok, image_path, indication, device):
    """Build the CheXagent-3B chat-template input.

    1. tok.from_list_format builds the query with image and text tokens.
    2. The query is wrapped in a system and human conversation.
    3. apply_chat_template tokenizes it once, with return_tensors.
    """
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
        conv,
        add_generation_prompt=True,
        return_tensors="pt",
    )
    return input_ids.to(device)


# Step 5a: verify the step-4 training reports
def verify_3b_baseline(train_manifest):
    log("Step 0: Verifying 3B train baseline reports...")
    bl_path = EXP_DIR / "train_baseline_reports.json"
    assert bl_path.exists(), \
        f"Error: {bl_path} not found. Run scripts/04_baseline_3b.py first."
    saved = json.load(open(bl_path))
    # Reject files without protocol metadata
    assert isinstance(saved, dict) and '_metadata' in saved, \
        f"Error: {bl_path} has no protocol metadata. Regenerate with scripts/04_baseline_3b.py."
    meta = saved['_metadata']
    assert meta.get('protocol_version') == PROTOCOL_VERSION, \
        f"Train baseline protocol mismatch: {meta.get('protocol_version')} != {PROTOCOL_VERSION}"
    assert meta.get('checkpoint') == CHECKPOINT, \
        f"Train baseline checkpoint mismatch: {meta.get('checkpoint')}"
    train_bl = saved['results']
    assert len(train_bl) == len(train_manifest), \
        f"Count mismatch: {len(train_bl)} vs {len(train_manifest)}"
    for i in range(len(train_manifest)):
        assert train_bl[i]['study_id'] == train_manifest[i]['study_id'], \
            f"ID mismatch at {i}"
    n_errors = sum(1 for r in train_bl if '_error' in r or not r.get('candidate_findings', '').strip())
    assert n_errors == 0, f"Error: {n_errors} bad train baseline reports"
    log(f"  {len(train_bl)} reports, 0 errors, protocol={PROTOCOL_VERSION}")
    return train_bl


# Step 5b: add reference labels
def enrich_manifests(train_manifest, val_manifest):
    log("\nStep 1: Enriching with CheXpert GT (U-Ones)...")
    cp = pd.read_csv(CHEXPERT_CSV)
    cp_key = {}
    for _, row in cp.iterrows():
        k = (int(row['subject_id']), int(row['study_id']))
        cp_key[k] = {p: chexpert_uones(row.get(p, 0)) for p in PATHOLOGIES}
    for name, manifest in [('train', train_manifest), ('val', val_manifest)]:
        missing = 0
        for s in manifest:
            key = (int(s['subject_id']), int(s['study_id']))
            if key in cp_key: s['gt_labels'] = cp_key[key]
            else: missing += 1
        assert missing == 0, f"Error: {missing} {name} studies have no GT"
        log(f"  {name}: {len(manifest)} studies, 0 missing GT ")
    return train_manifest, val_manifest


# Step 5c: label the training reports with CheXbert
def label_3b_train_reports(train_manifest, train_bl):
    labels_path = EXP_DIR / 'train_3b_chexbert_labels.json'
    # Hash the full report text
    report_hash = hashlib.md5(
        json.dumps([r.get('candidate_findings', '') for r in train_bl]).encode()
    ).hexdigest()
    manifest_hash = study_id_hash(train_manifest)

    if labels_path.exists():
        log("\nStep 2: Loading cached 3B CheXbert labels...")
        cached = json.load(open(labels_path))
        assert cached.get('_manifest_hash') == manifest_hash, "Manifest hash mismatch"
        assert cached.get('_report_hash') == report_hash, "Report hash mismatch, regenerate"
        labels = cached['labels']
        assert len(labels) == len(train_manifest)
        # Check that the study IDs match
        expected_sids = {str(s['study_id']) for s in train_manifest}
        assert set(labels.keys()) == expected_sids, "Study ID set mismatch in cached labels"
        log(f"  Cache verified: {len(labels)} labels ")
        return labels

    log("\nStep 2: CheXbert-labeling 3B train baseline reports...")
    from f1chexbert import F1CheXbert
    scorer = F1CheXbert(device=DEVICE)
    labels = {}
    for i, r in enumerate(train_bl):
        sid = str(train_manifest[i]['study_id'])
        text = r.get('candidate_findings', '')
        assert text.strip(), f"Empty report at {i} (study {sid})"
        raw = scorer.get_label(text, mode='rrg')
        labels[sid] = {
            p: int(raw[CHEXBERT_IDX[p]]) if CHEXBERT_IDX[p] < len(raw) and raw[CHEXBERT_IDX[p]] != '' else 0
            for p in PATHOLOGIES
        }
        if (i + 1) % 500 == 0: log(f"  Labeled {i+1}/{len(train_bl)}")
    json.dump({'_manifest_hash': manifest_hash, '_report_hash': report_hash, 'labels': labels},
              open(labels_path, 'w'), indent=2)
    del scorer; torch.cuda.empty_cache(); gc.collect()
    log(f"  Done: {len(labels)} labels saved")
    return labels


# Step 5d: extract features (steering features are stored on disk)
def extract_features(train_manifest, val_manifest):
    meta_path = EXP_DIR / "features_3b_meta.pkl"
    steer_path = EXP_DIR / "train_steer.npy"  # memory-mapped
    probe_path = EXP_DIR / "train_probe.npy"
    val_probe_path = EXP_DIR / "val_probe.npy"

    if meta_path.exists():
        log("\nStep 3: Loading cached features...")
        meta = pickle.load(open(meta_path, 'rb'))
        assert meta.get('probe_layer') == PROBE_LAYER, "probe_layer mismatch"
        assert meta.get('steer_layer') == STEER_LAYER, "steer_layer mismatch"
        assert meta.get('checkpoint') == CHECKPOINT, "checkpoint mismatch"
        assert meta.get('protocol_version') == PROTOCOL_VERSION, \
            f"Protocol version mismatch: cached={meta.get('protocol_version')}, " \
            f"current={PROTOCOL_VERSION}. Delete cache and re-extract."
        assert meta.get('train_manifest_hash') == study_id_hash(train_manifest), \
            "Train manifest hash mismatch"
        assert meta.get('val_manifest_hash') == study_id_hash(val_manifest), \
            "Val manifest hash mismatch"
        assert meta.get('train_sids') == [s['study_id'] for s in train_manifest], \
            "Train study ID ordering mismatch"
        assert meta.get('val_sids') == [s['study_id'] for s in val_manifest], \
            "Val study ID ordering mismatch"
        # Check the provenance hashes
        assert meta.get('train_provenance_hash') == full_provenance_hash(train_manifest), \
            "Train provenance hash mismatch (image paths / indications changed)"
        assert meta.get('val_provenance_hash') == full_provenance_hash(val_manifest), \
            "Val provenance hash mismatch"
        n_vis = meta['n_vis_per_study']
        # Open the steering features as a memory map
        train_l_steer = np.load(steer_path, mmap_mode='r')
        train_l_probe = np.load(probe_path)
        val_l_probe = np.load(val_probe_path)
        assert train_l_steer.shape == (len(train_manifest), n_vis, HIDDEN_DIM)
        log(f"  Features verified: steer={train_l_steer.shape} (mmap), "
            f"probe={train_l_probe.shape}, val={val_l_probe.shape}, protocol={PROTOCOL_VERSION}")
        meta['train_l_steer'] = train_l_steer
        meta['train_l_probe'] = train_l_probe
        meta['val_l_probe'] = val_l_probe
        return meta

    log(f"\nStep 3: Extracting features (probe=L{PROBE_LAYER}, steer=L{STEER_LAYER})...")
    log("  Using disk-backed storage to avoid ~140GB RAM spike")
    from transformers import AutoModelForCausalLM, AutoTokenizer
    model = AutoModelForCausalLM.from_pretrained(
        CHECKPOINT, trust_remote_code=True).to(DTYPE).to(DEVICE)
    model.eval()
    tok = AutoTokenizer.from_pretrained(CHECKPOINT, trust_remote_code=True)
    dec_layers = model.model.layers

    # Determine N_VIS from first study
    sample = train_manifest[0]
    tmp = f"/tmp/_pare3b_nvis_{os.getpid()}.png"
    Image.open(sample["image_paths"][0]).convert("RGB").save(tmp)
    test_ids = build_input_ids(tok, tmp, sample.get("section_indication") or "None", DEVICE)
    # Run one forward pass to check the token count
    with torch.no_grad():
        _ = model(test_ids, output_hidden_states=False)
    os.remove(tmp)
    n_vis = int((test_ids[0] == IMG_TOKEN_ID).sum())
    log(f"  Detected N_VIS = {n_vis} visual tokens")
    assert n_vis > 100, f"Unexpected N_VIS={n_vis}"

    def extract_one(sample_data):
        indication = sample_data.get("section_indication") or "None provided"
        tmp_path = f"/tmp/_pare3b_feat_{os.getpid()}.png"
        Image.open(sample_data["image_paths"][0]).convert("RGB").save(tmp_path)
        input_ids = build_input_ids(tok, tmp_path, indication, DEVICE)
        img_mask = (input_ids[0] == IMG_TOKEN_ID)
        captured = {}
        def make_hook(name):
            def hook_fn(module, inp, output):
                h = output[0] if isinstance(output, tuple) else output
                if img_mask.any():
                    vis_h = h[0, img_mask, :].detach().cpu().float()
                    if name == 'steer':
                        captured[name] = vis_h.numpy()
                    else:
                        captured[name] = vis_h.mean(dim=0).numpy()
                return output
            return hook_fn
        hooks = [
            dec_layers[STEER_LAYER].register_forward_hook(make_hook('steer')),
            dec_layers[PROBE_LAYER].register_forward_hook(make_hook('probe')),
        ]
        with torch.no_grad():
            _ = model(input_ids, output_hidden_states=False)
        for h in hooks: h.remove()
        if os.path.exists(tmp_path): os.remove(tmp_path)
        assert 'steer' in captured and 'probe' in captured
        assert captured['steer'].shape == (n_vis, HIDDEN_DIM)
        return captured

    # Training studies: write steering features to disk
    n_train = len(train_manifest)
    mmap_steer = np.lib.format.open_memmap(
        str(steer_path), mode='w+', dtype=np.float32, shape=(n_train, n_vis, HIDDEN_DIM))
    train_probes = []
    for i, sample_data in enumerate(train_manifest):
        feats = extract_one(sample_data)
        mmap_steer[i] = feats['steer']
        train_probes.append(feats['probe'])
        if (i+1) % 200 == 0:
            mmap_steer.flush()
            log(f"  train: {i+1}/{n_train}")
    mmap_steer.flush()
    train_l_probe = np.array(train_probes, dtype=np.float32)
    np.save(probe_path, train_l_probe)
    del train_probes; gc.collect()
    log(f"  Train done: steer={mmap_steer.shape} (disk), probe={train_l_probe.shape}")

    # Validation studies: probe features only
    val_probes = []
    for i, sample_data in enumerate(val_manifest):
        feats = extract_one(sample_data)
        val_probes.append(feats['probe'])
        if (i+1) % 200 == 0: log(f"  val: {i+1}/{len(val_manifest)}")
    val_l_probe = np.array(val_probes, dtype=np.float32)
    np.save(val_probe_path, val_l_probe)
    del val_probes; gc.collect()
    log(f"  Val done: probe={val_l_probe.shape}")

    meta = {
        'n_vis_per_study': n_vis,
        'probe_layer': PROBE_LAYER, 'steer_layer': STEER_LAYER,
        'checkpoint': CHECKPOINT, 'protocol_version': PROTOCOL_VERSION,
        'train_manifest_hash': study_id_hash(train_manifest),
        'val_manifest_hash': study_id_hash(val_manifest),
        'train_sids': [s['study_id'] for s in train_manifest],
        'val_sids': [s['study_id'] for s in val_manifest],
        'train_provenance_hash': full_provenance_hash(train_manifest),
        'val_provenance_hash': full_provenance_hash(val_manifest),
    }
    pickle.dump(meta, open(meta_path, 'wb'))
    del model, tok; torch.cuda.empty_cache(); gc.collect()
    log(f"  Features cached (protocol={PROTOCOL_VERSION})")
    # Reopen as read-only mmap for subsequent use
    meta['train_l_steer'] = np.load(steer_path, mmap_mode='r')
    meta['train_l_probe'] = train_l_probe
    meta['val_l_probe'] = val_l_probe
    return meta


# Step 5e: fit probes, thresholds and transport maps
def build_pare_components(train_manifest, train_l_steer, train_l_probe,
                          train_labels, val_manifest, val_l_probe, n_vis):
    log(f"\n{'='*60}")
    log(f"Step 4: Building the components, N_VIS={n_vis}")
    log(f"{'='*60}")

    # Check that train, validation and test are disjoint by study and by patient
    CANON = paths.out("test_set_and_baseline_8b")
    test_manifest = json.load(open(CANON / 'study_test_manifest.json'))
    train_sids = {s['study_id'] for s in train_manifest}
    val_sids = {s['study_id'] for s in val_manifest}
    test_sids = {s['study_id'] for s in test_manifest}
    assert len(train_sids & val_sids) == 0, "Train/val study_id overlap!"
    assert len(train_sids & test_sids) == 0, "Train/test study_id overlap!"
    assert len(val_sids & test_sids) == 0, "Val/test study_id overlap!"
    # Patient-level separation
    train_subj = {s.get('subject_id') for s in train_manifest} - {None}
    val_subj = {s.get('subject_id') for s in val_manifest} - {None}
    test_subj = {s.get('subject_id') for s in test_manifest} - {None}
    if train_subj and val_subj and test_subj:
        assert len(train_subj & val_subj) == 0, "Train/val subject_id overlap!"
        assert len(train_subj & test_subj) == 0, "Train/test subject_id overlap!"
        assert len(val_subj & test_subj) == 0, "Val/test subject_id overlap!"
        log(f"  3-way split verified: {len(train_sids)} train, {len(val_sids)} val, "
            f"{len(test_sids)} test (study + subject disjoint )")
    else:
        log("  3-way split verified (study_id only; subject_id not available in all manifests)")
    del test_manifest

    gt_train = {p: np.array([s['gt_labels'][p] for s in train_manifest]) for p in PATHOLOGIES}
    gt_val = {p: np.array([s['gt_labels'][p] for s in val_manifest]) for p in PATHOLOGIES}

    # TP/FN sets from reference labels and the CheXagent-3B baseline reports
    tp_idx = {p: [] for p in PATHOLOGIES}
    fn_idx = {p: [] for p in PATHOLOGIES}
    for i in range(len(train_manifest)):
        sid = str(train_manifest[i]['study_id'])
        cb = train_labels[sid]
        for p in PATHOLOGIES:
            gt = gt_train[p][i]; pred = cb[p]
            if gt == 1 and pred == 1: tp_idx[p].append(i)
            elif gt == 1 and pred == 0: fn_idx[p].append(i)

    log("\n  TP/FN (GT x 3B baseline):")
    for p in PATHOLOGIES:
        m = " (target)" if p in TARGET_5 else ""
        log(f"    {p:30s}: TP={len(tp_idx[p]):5d}, FN={len(fn_idx[p]):5d}{m}")

    # Probes (same settings as CheXagent-8B)
    sc_probe = StandardScaler()
    d_tr = sc_probe.fit_transform(train_l_probe)
    d_va = sc_probe.transform(val_l_probe)

    probes = {}; thresholds = {}; probe_aurocs = {}
    for p in PATHOLOGIES:
        n_pos_tr = int(gt_train[p].sum()); n_pos_va = int(gt_val[p].sum())
        n_neg_va = len(gt_val[p]) - n_pos_va
        if n_pos_tr < 20 or n_pos_va < 10 or n_neg_va < 10:
            log(f"    {p:30s}: probe skipped (tr_pos={n_pos_tr}, va_pos={n_pos_va}, va_neg={n_neg_va})")
            continue
        clf = LogisticRegression(penalty='l1', solver='liblinear', C=0.01,
            class_weight='balanced', max_iter=5000, random_state=42)
        clf.fit(d_tr, gt_train[p])
        val_scores = clf.predict_proba(d_va)[:, 1]
        val_auroc = roc_auc_score(gt_val[p], val_scores)
        fpr, tpr, th = roc_curve(gt_val[p], val_scores)
        best = np.argmax(tpr - fpr)
        t_val = float(th[best])
        assert np.isfinite(t_val), f"Non-finite threshold for {p}"
        thresholds[p] = t_val
        probes[p] = clf
        probe_aurocs[p] = float(val_auroc)
        m = " (target)" if p in TARGET_5 else ""
        log(f"    {p:30s}: AUROC={val_auroc:.4f}, thresh={t_val:.4f}{m}")

    json.dump(probe_aurocs, open(EXP_DIR / 'probe_aurocs.json', 'w'), indent=2)

    # Transport maps (same formula as CheXagent-8B)
    log(f"\n  Monge maps (FN->TP, min {MIN_STUDIES_FOR_MONGE} each)...")
    def compute_monge(X_s, X_t):
        mu_s = X_s.mean(0); mu_t = X_t.mean(0); d = X_s.shape[1]
        Sig_s = LedoitWolf().fit(X_s).covariance_ if X_s.shape[0] > d else np.cov(X_s.T) + 1e-3*np.eye(d)
        Sig_t = LedoitWolf().fit(X_t).covariance_ if X_t.shape[0] > d else np.cov(X_t.T) + 1e-3*np.eye(d)
        S_sqrt = np.real(sqrtm(Sig_s)); S_inv_sqrt = np.real(inv(S_sqrt))
        M = S_sqrt @ Sig_t @ S_sqrt
        A = S_inv_sqrt @ np.real(sqrtm(M)) @ S_inv_sqrt
        return mu_s.astype(np.float32), mu_t.astype(np.float32), ((A+A.T)/2).astype(np.float32)

    monge_maps = {}; skipped = []
    for p in PATHOLOGIES:
        if p not in probes:
            skipped.append((p, "no_probe")); continue
        tp_i = np.array(tp_idx[p]); fn_i = np.array(fn_idx[p])
        if len(tp_i) < MIN_STUDIES_FOR_MONGE or len(fn_i) < MIN_STUDIES_FOR_MONGE:
            log(f"    {p:30s}: map skipped (TP={len(tp_i)}, FN={len(fn_i)})")
            skipped.append((p, f"TP={len(tp_i)},FN={len(fn_i)}"))
            del probes[p]; del thresholds[p]
            continue

        # Fitting on all tokens of all studies would need over 100 GB of memory, so the scaler,
        # PCA and covariances are fitted on a fixed subsample. The resulting map is applied to
        # all visual tokens at inference.
        rng = np.random.RandomState(42)

        # Subsample studies first
        n_tp_total = len(tp_i) * n_vis; n_fn_total = len(fn_i) * n_vis
        total_tokens = n_tp_total + n_fn_total
        if total_tokens > MAX_TOKENS_FOR_FIT:
            # Proportional study-level subsample
            ratio = MAX_TOKENS_FOR_FIT / total_tokens
            n_tp_studies = max(MIN_STUDIES_FOR_MONGE, int(len(tp_i) * ratio))
            n_fn_studies = max(MIN_STUDIES_FOR_MONGE, int(len(fn_i) * ratio))
            tp_i_sample = tp_i[rng.choice(len(tp_i), min(n_tp_studies, len(tp_i)), replace=False)]
            fn_i_sample = fn_i[rng.choice(len(fn_i), min(n_fn_studies, len(fn_i)), replace=False)]
            log(f"    {p:30s}: Memory-bounded: {len(tp_i_sample)}/{len(tp_i)} TP, "
                f"{len(fn_i_sample)}/{len(fn_i)} FN studies sampled for fitting")
        else:
            tp_i_sample = tp_i; fn_i_sample = fn_i

        # Flatten to one row per token
        tp_tok = train_l_steer[tp_i_sample].reshape(-1, HIDDEN_DIM).astype(np.float32)
        fn_tok = train_l_steer[fn_i_sample].reshape(-1, HIDDEN_DIM).astype(np.float32)

        # Standardise on the pooled TP and FN tokens
        all_tok = np.vstack([tp_tok, fn_tok])
        sc = StandardScaler().fit(all_tok)
        tp_sc = sc.transform(tp_tok); fn_sc = sc.transform(fn_tok)
        del tp_tok, fn_tok, all_tok; gc.collect()

        # Subsample tokens for PCA fitting
        n_fit = min(MAX_TOKENS_FOR_FIT, len(tp_sc) + len(fn_sc))
        n_tp_s = int(n_fit * len(tp_sc) / (len(tp_sc) + len(fn_sc)))
        tp_sub = tp_sc[rng.choice(len(tp_sc), min(n_tp_s, len(tp_sc)), replace=False)]
        fn_sub = fn_sc[rng.choice(len(fn_sc), min(n_fit - n_tp_s, len(fn_sc)), replace=False)]
        pca = PCA(n_components=PCA_DIMS, random_state=42).fit(np.vstack([tp_sub, fn_sub]))
        del tp_sub, fn_sub

        # Gaussian OT map in PCA space
        mu_s, mu_t, A = compute_monge(pca.transform(fn_sc), pca.transform(tp_sc))
        del tp_sc, fn_sc; gc.collect()

        monge_maps[p] = {
            'A': A, 'mu_s': mu_s, 'mu_t': mu_t,
            'sc_mean': sc.mean_.astype(np.float32), 'sc_scale': sc.scale_.astype(np.float32),
            'pca_mean': pca.mean_.astype(np.float32), 'pca_comp': pca.components_.astype(np.float32),
        }
        m = " (target)" if p in TARGET_5 else ""
        log(f"    {p:30s}: TP={len(tp_i)}, FN={len(fn_i)}, fit_tok={n_tp_total+n_fn_total}, "
            f"PCA_var={pca.explained_variance_ratio_.sum():.4f}{m}")
        gc.collect()

    steerable = sorted(set(probes.keys()) & set(monge_maps.keys()))
    log(f"\n  Steerable: {len(steerable)}/{len(PATHOLOGIES)}")
    for p in steerable: log(f"    {p}")
    assert len(steerable) >= 5, f"Only {len(steerable)} steerable"
    missing_t5 = [p for p in TARGET_5 if p not in steerable]
    if missing_t5: log(f"  Target pathologies without a map: {missing_t5}")

    # Save with provenance metadata
    components = {
        'probes': probes, 'thresholds': thresholds, 'monge_maps': monge_maps,
        'sc_probe': sc_probe, 'pathologies': PATHOLOGIES, 'steerable_pathologies': steerable,
        'probe_aurocs': probe_aurocs, 'target_5': TARGET_5,
        'checkpoint': CHECKPOINT, 'steer_layer': STEER_LAYER, 'probe_layer': PROBE_LAYER,
        'n_vis': n_vis, 'pca_dims': PCA_DIMS, 'lambda': BEST_LAMBDA,
        'hidden_dim': HIDDEN_DIM, 'img_token_id': IMG_TOKEN_ID,
        'n_train': len(train_manifest),
        'train_manifest_hash': study_id_hash(train_manifest),
        'train_provenance_hash': full_provenance_hash(train_manifest),
        'protocol_version': PROTOCOL_VERSION,
        'min_studies_for_monge': MIN_STUDIES_FOR_MONGE,
        'max_tokens_for_fit': MAX_TOKENS_FOR_FIT,
        'tp_fn_counts': {p: {'TP': len(tp_idx[p]), 'FN': len(fn_idx[p])} for p in PATHOLOGIES},
    }
    comp_path = EXP_DIR / 'components_3b.pkl'
    pickle.dump(components, open(comp_path, 'wb'))
    # Store the file hash in a separate file
    comp_hash = hashlib.sha256(open(comp_path, 'rb').read()).hexdigest()[:16]
    sidecar = {'comp_hash': comp_hash, 'protocol_version': PROTOCOL_VERSION}
    json.dump(sidecar, open(EXP_DIR / 'components_3b_hash.json', 'w'), indent=2)
    log(f"\n  Saved components (file_hash={comp_hash}, sidecar written)")
    return components


if __name__ == "__main__":
    log(f"\n{'='*60}")
    log("Fitting PARE components: CheXagent-3B")
    log(f"{'='*60}")

    train_manifest = json.load(open(TRAINVAL_DIR / 'train_manifest.json'))
    val_manifest = json.load(open(TRAINVAL_DIR / 'val_manifest.json'))
    log(f"Manifests: train={len(train_manifest)}, val={len(val_manifest)}")

    # CheXagent-3B uses the first image of each study; check on a sample
    # that the manifests list frontal views first.
    def verify_frontal_first(manifest, name):
        n_checked = min(200, len(manifest))
        n_frontal = 0
        for s in manifest[:n_checked]:
            p0 = s['image_paths'][0].lower()
            # MIMIC-CXR frontal views are AP or PA; lateral contains 'lateral'
            if 'lateral' not in p0:
                n_frontal += 1
        pct = n_frontal / n_checked * 100
        log(f"  {name}: {n_frontal}/{n_checked} ({pct:.0f}%) first images are non-lateral")
        assert pct >= 95, \
            f"Error: {name} manifest does not list frontal views first ({pct:.0f}% non-lateral). "\
            f"Re-sort manifests or explicitly select frontal images."
    verify_frontal_first(train_manifest, "train")
    verify_frontal_first(val_manifest, "val")

    train_bl = verify_3b_baseline(train_manifest)
    train_manifest, val_manifest = enrich_manifests(train_manifest, val_manifest)
    train_labels = label_3b_train_reports(train_manifest, train_bl)

    cache = extract_features(train_manifest, val_manifest)

    components = build_pare_components(
        train_manifest, cache['train_l_steer'], cache['train_l_probe'],
        train_labels, val_manifest, cache['val_l_probe'], cache['n_vis_per_study'])

    log("\nStep 5 complete. Next: scripts/06_lambda_sweep_3b.py")
