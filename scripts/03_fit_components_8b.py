#!/usr/bin/env python3
"""Step 3: fit the PARE components for CheXagent-8B.

For each of the 13 CheXbert pathology labels (No Finding excluded):
  - an l1-regularised logistic-regression probe on standardised layer-31
    features (balanced class weights, C=0.01, liblinear), with its threshold
    chosen on the validation set by Youden's J statistic;
  - a Gaussian optimal-transport map between the layer-3 visual tokens of
    false-negative (source) and true-positive (target) training studies, in a
    128-dimensional PCA space with Ledoit-Wolf covariance estimates. A map is
    fitted only if both groups contain at least 50 studies.
The steering strength is fixed at lambda = 1.0 (not tuned).

Output ($PARE_OUT/components_8b/):
  components_8b.pkl

Usage:
  python scripts/03_fit_components_8b.py
"""
import sys as _sys; from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))  # repo root
from pare import paths
import torch, json, time, gc, warnings, pickle, hashlib
import pandas as pd, numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.decomposition import PCA
from sklearn.covariance import LedoitWolf
from scipy.linalg import sqrtm, inv
warnings.filterwarnings('ignore')

# Paths
CHEXPERT_CSV = str(paths.CHEXPERT_CSV)
TRAINVAL_DIR    = paths.out("trainval_features_8b")
BASELINE_DIR = paths.out("test_set_and_baseline_8b")
EXP_DIR      = paths.out("components_8b")
EXP_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = "cuda:0"; CHECKPOINT = "StanfordAIMI/CheXagent-8b"
N_VIS = 128; STEER_LAYER = 3; PROBE_LAYER = 31; PCA_DIMS = 128
BEST_LAMBDA = 1.0  # fixed, not tuned
MIN_STUDIES_FOR_MONGE = 50  # minimum number of TP and of FN studies per map

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
    x = int(float(x))
    return 1 if x in (1, -1) else 0

def study_id_hash(manifest):
    """Weak hash: ordered study_ids only."""
    return hashlib.md5(json.dumps([s['study_id'] for s in manifest]).encode()).hexdigest()

def full_provenance_hash(manifest):
    """Strong hash: study_id + subject_id + image_paths + indication for every entry."""
    fingerprint = []
    for s in manifest:
        fingerprint.append({
            'study_id': s['study_id'],
            'subject_id': s['subject_id'],
            'image_paths': s['image_paths'],
            'indication': s.get('section_indication', ''),
        })
    return hashlib.md5(json.dumps(fingerprint, sort_keys=True).encode()).hexdigest()


# Step 3a: verify the step-2 outputs
def verify_provenance():
    """Check that the step-2 features and manifests are aligned before using them."""
    log("Step 0: Verifying the step-2 outputs...")

    # The preliminary components of step 2 store the manifest hash used during
    # feature extraction. Verify it, then compute a stronger full-provenance hash.
    p5_comp = pickle.load(open(TRAINVAL_DIR / 'prelim_components_8b.pkl', 'rb'))
    p5_hash = p5_comp['train_manifest_hash']
    p5_n    = p5_comp['n_train']
    log(f"  Step-2 components: n_train={p5_n}, manifest_hash={p5_hash[:8]}")

    # Load the training manifest and verify that the hash matches
    train_manifest = json.load(open(TRAINVAL_DIR / 'train_manifest.json'))
    current_hash = study_id_hash(train_manifest)
    assert current_hash == p5_hash, \
        f"Error: training manifest hash mismatch. Components={p5_hash[:8]}, current={current_hash[:8]}. " \
        f"Features were generated for a different manifest ordering."
    log(f"  Manifest hash verified: {current_hash[:8]} == {p5_hash[:8]}")

    # Verify feature shapes match manifest length
    train_l3  = np.load(TRAINVAL_DIR / 'train_l3.npy', mmap_mode='r')
    train_l31 = np.load(TRAINVAL_DIR / 'train_l31.npy', mmap_mode='r')
    val_l31   = np.load(TRAINVAL_DIR / 'val_l31.npy', mmap_mode='r')
    val_manifest = json.load(open(TRAINVAL_DIR / 'val_manifest.json'))
    assert train_l3.shape[0] == len(train_manifest), \
        f"Error: train_l3 rows ({train_l3.shape[0]}) != manifest ({len(train_manifest)})"
    assert train_l31.shape[0] == len(train_manifest), \
        f"Error: train_l31 rows ({train_l31.shape[0]}) != manifest ({len(train_manifest)})"
    assert val_l31.shape[0] == len(val_manifest), \
        f"Error: val_l31 rows ({val_l31.shape[0]}) != val_manifest ({len(val_manifest)})"
    log(f"  Feature shapes aligned: train={len(train_manifest)}, val={len(val_manifest)}")

    # Compute and save the full provenance hash
    fp_hash = full_provenance_hash(train_manifest)
    log(f"  Full provenance hash (study+subject+images+indication): {fp_hash[:12]}")

    # Check that the training reports align with the manifest
    train_reports = json.load(open(TRAINVAL_DIR / 'train_reports.json'))
    assert len(train_reports) == len(train_manifest), \
        f"Error: reports ({len(train_reports)}) != manifest ({len(train_manifest)})"
    for i in range(len(train_manifest)):
        r_sid = train_reports[i].get('study_id')
        m_sid = train_manifest[i]['study_id']
        assert r_sid == m_sid, \
            f"Error: Report/manifest study_id mismatch at index {i}: report={r_sid}, manifest={m_sid}"
    log(f"  All {len(train_reports)} report↔manifest study_ids verified")

    # Stop if any training report failed to generate
    bad_reports = [i for i, r in enumerate(train_reports)
                   if not r.get('candidate_findings')
                   or r['candidate_findings'].startswith('__ERROR__')]
    assert len(bad_reports) == 0, \
        f"Error: {len(bad_reports)} training reports are errors/empty. " \
        f"These would be silently treated as all-negative labels. " \
        f"Indices: {bad_reports[:10]}"
    log(f"  All {len(train_reports)} training reports have valid text")

    return train_manifest, val_manifest


# Step 3b: add reference labels for the 13 pathologies
def enrich_manifests(train_manifest, val_manifest):
    log("\nStep 1: Enriching manifests with 13-pathology CheXpert GT (U-Ones)...")
    cp = pd.read_csv(CHEXPERT_CSV)
    cp_key = {}
    for _, row in cp.iterrows():
        k = (int(row['subject_id']), int(row['study_id']))
        cp_key[k] = {p: chexpert_uones(row.get(p, 0)) for p in PATHOLOGIES}

    for split_name, manifest in [('train', train_manifest), ('val', val_manifest)]:
        missing = 0
        for s in manifest:
            key = (int(s['subject_id']), int(s['study_id']))
            if key in cp_key:
                s['gt_labels'] = cp_key[key]
            else:
                missing += 1
        # Stop if any reference label is missing
        assert missing == 0, \
            f"Error: {missing} {split_name} studies have no CheXpert GT. " \
            f"Missing labels cannot be treated as negative."
        log(f"  {split_name}: {len(manifest)} studies, 0 missing GT ")
        for p in PATHOLOGIES:
            n_pos = sum(1 for s in manifest if s['gt_labels'].get(p, 0) == 1)
            log(f"    {p:30s}: {n_pos:5d} pos ({100*n_pos/len(manifest):.1f}%)")
        json.dump(manifest, open(EXP_DIR / f'{split_name}_manifest.json', 'w'), indent=2)

    return train_manifest, val_manifest


# Step 3c: label the training reports with CheXbert
def label_train_reports_13(train_manifest):
    labels_path = EXP_DIR / 'train_chexbert_labels_13.json'
    if labels_path.exists():
        log("\nStep 2: Loading cached 13-pathology CheXbert labels...")
        cached = json.load(open(labels_path))
        # Reuse cached labels only if they match this manifest and report text
        cached_manifest_hash = cached.get('_manifest_hash', '')
        cached_report_hash = cached.get('_report_hash', '')
        expected_manifest_hash = study_id_hash(train_manifest)
        # Compute report-text hash to detect re-generated baseline reports
        train_reports = json.load(open(TRAINVAL_DIR / 'train_reports.json'))
        expected_report_hash = hashlib.md5(
            json.dumps([r.get('candidate_findings', '')[:200] for r in train_reports]).encode()
        ).hexdigest()
        assert cached_manifest_hash == expected_manifest_hash, \
            f"Error: Cached labels manifest hash ({cached_manifest_hash[:8]}) != current ({expected_manifest_hash[:8]}). " \
            f"Delete {labels_path} and re-run."
        assert cached_report_hash == expected_report_hash, \
            f"Error: Cached labels report hash ({cached_report_hash[:8]}) != current ({expected_report_hash[:8]}). " \
            f"Report text changed since labels were generated. Delete {labels_path} and re-run."
        labels = cached['labels']
        assert len(labels) == len(train_manifest), \
            f"Cached label count ({len(labels)}) != manifest ({len(train_manifest)})"
        # Check that the study IDs match
        expected_sids = {str(s['study_id']) for s in train_manifest}
        cached_sids = set(labels.keys())
        assert cached_sids == expected_sids, \
            f"Error: Cached label study-IDs don't match manifest. " \
            f"Extra in cache: {cached_sids - expected_sids}, missing: {expected_sids - cached_sids}"
        log(f"  Cache verified: manifest_hash={cached_manifest_hash[:8]}, report_hash={cached_report_hash[:8]}, {len(labels)} labels ")
        return labels

    log("\nStep 2: CheXbert-labeling 15K train reports for 13 pathologies...")
    train_reports = json.load(open(TRAINVAL_DIR / 'train_reports.json'))

    from f1chexbert import F1CheXbert
    scorer = F1CheXbert(device=DEVICE)

    # Key labels by study_id
    labels = {}
    for i, r in enumerate(train_reports):
        sid = str(train_manifest[i]['study_id'])
        text = r.get('candidate_findings', '')
        # Failed reports are rejected in verify_provenance(); check again here
        assert text and not text.startswith('__ERROR__'), \
            f"Report {i} (study {sid}) is empty/error, should have been caught earlier"
        raw = scorer.get_label(text, mode='rrg')
        labels[sid] = {
            p: int(raw[CHEXBERT_IDX[p]]) if CHEXBERT_IDX[p] < len(raw) and raw[CHEXBERT_IDX[p]] != '' else 0
            for p in PATHOLOGIES
        }
        if (i + 1) % 500 == 0:
            log(f"  Labeled {i+1}/{len(train_reports)}")

    # Save with the manifest and report-text hashes
    report_hash = hashlib.md5(
        json.dumps([r.get('candidate_findings', '')[:200] for r in train_reports]).encode()
    ).hexdigest()
    json.dump({
        '_manifest_hash': study_id_hash(train_manifest),
        '_report_hash': report_hash,
        'labels': labels,
    }, open(labels_path, 'w'), indent=2)
    del scorer; torch.cuda.empty_cache(); gc.collect()
    log(f"  Done: {len(labels)} reports labeled")
    return labels


# Step 3d: fit probes, thresholds and transport maps
def build_pare_components(train_manifest, train_l3, train_l31, train_labels,
                          val_manifest, val_l31):
    log(f"\n{'='*60}")
    log("Step 3: Building the components")
    log(f"  Min studies for Monge map: {MIN_STUDIES_FOR_MONGE}")
    log(f"{'='*60}")

    gt_train = {p: np.array([s['gt_labels'][p] for s in train_manifest]) for p in PATHOLOGIES}
    gt_val   = {p: np.array([s['gt_labels'][p] for s in val_manifest])   for p in PATHOLOGIES}

    # TP/FN sets from CheXbert labels and reference labels
    tp_idx = {p: [] for p in PATHOLOGIES}
    fn_idx = {p: [] for p in PATHOLOGIES}
    for i in range(len(train_manifest)):
        sid = str(train_manifest[i]['study_id'])
        cb = train_labels[sid]
        for p in PATHOLOGIES:
            gt = gt_train[p][i]; pred = cb[p]
            if gt == 1 and pred == 1: tp_idx[p].append(i)
            elif gt == 1 and pred == 0: fn_idx[p].append(i)

    log("\n  TP/FN study counts:")
    for p in PATHOLOGIES:
        m = " (target)" if p in TARGET_5 else ""
        log(f"    {p:30s}: TP={len(tp_idx[p]):5d}, FN={len(fn_idx[p]):5d}{m}")

    # Probes on layer-31 features
    log("\n  Training L31 probes (train), Youden-J thresholds (val)...")
    sc31 = StandardScaler()
    d31_tr = sc31.fit_transform(train_l31.astype(np.float32))
    d31_va = sc31.transform(val_l31.astype(np.float32))

    probes = {}; thresholds = {}; probe_aurocs = {}
    for p in PATHOLOGIES:
        n_pos_tr = int(gt_train[p].sum()); n_pos_va = int(gt_val[p].sum())
        if n_pos_tr < 20 or n_pos_va < 10:
            log(f"    {p:30s}: probe skipped (train_pos={n_pos_tr}, val_pos={n_pos_va})")
            continue
        clf = LogisticRegression(penalty='l1', solver='liblinear', C=0.01,
            class_weight='balanced', max_iter=5000, random_state=42)
        clf.fit(d31_tr, gt_train[p])
        val_scores = clf.predict_proba(d31_va)[:, 1]
        val_auroc = roc_auc_score(gt_val[p], val_scores)
        fpr, tpr, th = roc_curve(gt_val[p], val_scores)
        best = np.argmax(tpr - fpr)
        thresholds[p] = float(th[best])
        probes[p] = clf
        probe_aurocs[p] = float(val_auroc)
        m = " (target)" if p in TARGET_5 else ""
        log(f"    {p:30s}: AUROC={val_auroc:.4f}, thresh={thresholds[p]:.4f}{m}")

    json.dump(probe_aurocs, open(EXP_DIR / 'probe_aurocs.json', 'w'), indent=2)

    # Transport maps on layer-3 visual tokens
    log(f"\n  Building Monge maps (FN->TP, min {MIN_STUDIES_FOR_MONGE} studies each)...")
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
            log(f"    {p:30s}: map skipped (TP={len(tp_i)}, FN={len(fn_i)}, minimum {MIN_STUDIES_FOR_MONGE})")
            skipped.append((p, f"TP={len(tp_i)},FN={len(fn_i)}"))
            del probes[p]; del thresholds[p]
            continue

        tp_tok = train_l3[tp_i].reshape(-1, 4096).astype(np.float32)
        fn_tok = train_l3[fn_i].reshape(-1, 4096).astype(np.float32)
        all_tok = np.vstack([tp_tok, fn_tok])
        sc = StandardScaler().fit(all_tok)
        tp_sc = sc.transform(tp_tok); fn_sc = sc.transform(fn_tok)
        rng = np.random.RandomState(42)
        n_fit = min(100000, len(tp_sc) + len(fn_sc))
        n_tp_s = int(n_fit * len(tp_sc) / (len(tp_sc) + len(fn_sc)))
        tp_sub = tp_sc[rng.choice(len(tp_sc), min(n_tp_s, len(tp_sc)), replace=False)]
        fn_sub = fn_sc[rng.choice(len(fn_sc), min(n_fit - n_tp_s, len(fn_sc)), replace=False)]
        pca = PCA(n_components=PCA_DIMS, random_state=42).fit(np.vstack([tp_sub, fn_sub]))
        mu_s, mu_t, A = compute_monge(pca.transform(fn_sc), pca.transform(tp_sc))
        monge_maps[p] = {
            'A': A, 'mu_s': mu_s, 'mu_t': mu_t,
            'sc_mean': sc.mean_.astype(np.float32), 'sc_scale': sc.scale_.astype(np.float32),
            'pca_mean': pca.mean_.astype(np.float32), 'pca_comp': pca.components_.astype(np.float32),
        }
        m = " (target)" if p in TARGET_5 else ""
        log(f"    {p:30s}: TP={len(tp_i)}, FN={len(fn_i)}, PCA_var={pca.explained_variance_ratio_.sum():.4f}{m}")
        del tp_tok, fn_tok, all_tok; gc.collect()

    if skipped:
        log(f"\n  Skipped {len(skipped)} pathologies:")
        for p, reason in skipped: log(f"    {p:30s}: {reason}")

    steerable = sorted(set(probes.keys()) & set(monge_maps.keys()))
    log(f"\n  Steerable: {len(steerable)}/{len(PATHOLOGIES)} pathologies")
    for p in steerable: log(f"    {p}")

    # Check that enough maps were fitted
    assert len(steerable) >= 5, \
        f"Error: Only {len(steerable)} pathologies are steerable. " \
        f"At least the five target pathologies are needed."
    # Warn if any of the five target pathologies has no map
    missing_t5 = [p for p in TARGET_5 if p not in steerable]
    if missing_t5:
        log(f"  Warning: target pathologies without a map: {missing_t5}")
    log(f"  {len(steerable)} of 13 pathologies are steerable")
    t5_in = [p for p in TARGET_5 if p in steerable]
    log(f"  Target pathologies with a map: {len(t5_in)}/5 ({t5_in})")
    non_t5 = [p for p in steerable if p not in TARGET_5]
    log(f"  Other pathologies with a map: {len(non_t5)} ({non_t5})")

    # Save with provenance metadata
    mhash = study_id_hash(train_manifest)
    fp_hash = full_provenance_hash(train_manifest)
    components = {
        'probes': probes, 'thresholds': thresholds, 'monge_maps': monge_maps,
        'sc31': sc31, 'pathologies': PATHOLOGIES, 'steerable_pathologies': steerable,
        'probe_aurocs': probe_aurocs, 'target_5': TARGET_5,
        'checkpoint': CHECKPOINT, 'steer_layer': STEER_LAYER, 'probe_layer': PROBE_LAYER,
        'n_vis': N_VIS, 'pca_dims': PCA_DIMS, 'lambda': BEST_LAMBDA,
        'n_train': len(train_manifest),
        'train_manifest_hash': mhash,
        'train_provenance_hash': fp_hash,
        'min_studies_for_monge': MIN_STUDIES_FOR_MONGE,
        'tp_fn_counts': {p: {'TP': len(tp_idx[p]), 'FN': len(fn_idx[p])} for p in PATHOLOGIES},
    }
    pickle.dump(components, open(EXP_DIR / 'components_8b.pkl', 'wb'))
    log(f"\n  Saved components (hash={mhash[:8]})")
    return probes, thresholds, monge_maps, sc31


if __name__ == "__main__":
    log(f"\n{'='*60}")
    log("Fitting PARE components: CheXagent-8B")
    log(f"{'='*60}")

    train_manifest, val_manifest = verify_provenance()
    train_manifest, val_manifest = enrich_manifests(train_manifest, val_manifest)

    log("\nStep 1b: Loading cached features...")
    train_l3  = np.load(TRAINVAL_DIR / 'train_l3.npy')
    train_l31 = np.load(TRAINVAL_DIR / 'train_l31.npy')
    val_l31   = np.load(TRAINVAL_DIR / 'val_l31.npy')
    log(f"  train_l3={train_l3.shape}, train_l31={train_l31.shape}, val_l31={val_l31.shape}")

    train_labels = label_train_reports_13(train_manifest)
    probes, thresholds, monge_maps, sc31 = build_pare_components(
        train_manifest, train_l3, train_l31, train_labels, val_manifest, val_l31)

    log("\nStep 3 complete. Next: scripts/07_steer_test_8b.py")
