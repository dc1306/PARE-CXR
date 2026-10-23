#!/usr/bin/env python3
"""
PARE evaluation on the 2,000-study test cohort (paper, Sec. 4).

No generation: uses the baseline and steered reports from scripts 01/04/07/08.
Runs on CheXbert only (one GPU, about 1-2 h per model).

Steps (all deterministic):
  0. Cohort   : 2,255 test studies with findings
                -> drop the first 200 (used by the lambda sweeps)
                -> random 2,000 of the remaining 2,055 (seed 42)
                Written once to <PARE_OUT>/evaluation/cohort_2000.json, then only re-read.
  2. Evaluate : on the 2,000 cohort, for
                baseline | raw PARE | merged (negation-aware, Sec. 3.5) |
                probe + template baseline
  3. Bootstrap: paired, study-level, 5,000 resamples (seed 2027),
                95% percentile CIs of the change vs. baseline.
  4. Probes   : AUROC and recall on baseline false negatives.
  5. True positives of the baseline lost by raw PARE.

Usage:
  CUDA_VISIBLE_DEVICES=<gpu> python scripts/09_evaluate.py --model 8b
  CUDA_VISIBLE_DEVICES=<gpu> python scripts/09_evaluate.py --model 3b
"""
import sys as _sys; from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))  # repo root
from pare import paths
from pare.safe_merger import safe_merge_negation_aware
import argparse, json, re, time
from collections import Counter

import numpy as np
from sklearn.metrics import roc_auc_score

LABELS_14 = ["Enlarged Cardiomediastinum", "Cardiomegaly", "Lung Opacity", "Lung Lesion",
             "Edema", "Consolidation", "Pneumonia", "Atelectasis", "Pneumothorax",
             "Pleural Effusion", "Pleural Other", "Fracture", "Support Devices", "No Finding"]
STEER_5 = ["Atelectasis", "Cardiomegaly", "Consolidation", "Edema", "Pleural Effusion"]
ALL_13 = [l for l in LABELS_14 if l != "No Finding"]
IDX = {l: i for i, l in enumerate(LABELS_14)}
S5_IDX = [IDX[l] for l in STEER_5]
A13_IDX = [IDX[l] for l in ALL_13]

N_SWEEP = 200            # first 200 test studies with findings, used in step 6
N_EVAL = 2000            # fixed evaluation cohort size
COHORT_SEED = 42
BOOT_N = 5000
BOOT_SEED = 2027

# Templates for the probe+template baseline. Each is checked to be CheXbert-positive.
TEMPLATES = {
    "Atelectasis": "There is atelectasis.",
    "Cardiomegaly": "The heart is enlarged.",
    "Consolidation": "There is consolidation.",
    "Edema": "There is pulmonary edema.",
    "Pleural Effusion": "There is a pleural effusion.",
}

# Text normalisation applied before CheXbert labelling
clean = lambda x: re.sub(r"\s+", " ", re.sub(r"\[.*?\]", "", x).replace("**", "")).strip()


def inputs(model):
    canon = paths.out("test_set_and_baseline_8b")
    if model == "8b":
        return dict(manifest=canon / "study_test_manifest.json",
                    baseline=canon / "chexagent_8b_reports.json",
                    pare_dir=paths.out("steered_8b"))      # lambda = 1.0
    return dict(manifest=canon / "study_test_manifest.json",
                baseline=paths.out("baseline_and_components_3b") / "baseline_reports.json",
                pare_dir=paths.out("steered_3b"))       # lambda = 0.13


def load_baseline(p):
    d = json.load(open(p))
    return d["results"] if isinstance(d, dict) else d


class Labeler:
    """CheXbert labels with caching. 'rrg': uncertain->1, negative/blank->0.
    'cls': raw classes ('' blank, 1 pos, 0 neg, -1 uncertain)."""
    def __init__(self, scorer):
        self.s = scorer; self.cache = {}

    def rrg(self, text):
        t = text.strip() if text else ""
        if not t:
            return [0] * 14
        k = ("rrg", t)
        if k not in self.cache:
            raw = self.s.get_label(t, mode="rrg")
            self.cache[k] = [int(raw[j]) if j < len(raw) and raw[j] != "" else 0 for j in range(14)]
        return self.cache[k]

    def cls(self, text):
        t = text.strip() if text else ""
        if not t:
            return [""] * 14
        k = ("cls", t)
        if k not in self.cache:
            self.cache[k] = list(self.s.get_label(t, mode="classification"))
        return self.cache[k]

    def eval_labels(self, texts):
        # clean().lower(), then CheXbert in rrg mode
        return np.array([self.rrg(clean(t).lower() if t else "") for t in texts], dtype=int)


def counts(pred, ref):
    tp = ((ref == 1) & (pred == 1)); fp = ((ref == 0) & (pred == 1)); fn = ((ref == 1) & (pred == 0))
    return tp.astype(np.int64), fp.astype(np.int64), fn.astype(np.int64)


def summary_from_sums(TP, FP, FN):
    den = 2 * TP + FP + FN
    f1 = np.where(den > 0, 2 * TP / np.maximum(den, 1), 0.0)
    s5 = f1[..., S5_IDX].mean(-1); a13 = f1[..., A13_IDX].mean(-1)
    mden = 2 * TP.sum(-1) + FP.sum(-1) + FN.sum(-1)
    mi = np.where(mden > 0, 2 * TP.sum(-1) / np.maximum(mden, 1), 0.0)
    return f1, s5, a13, mi


def full_metrics(pred, ref):
    tp, fp, fn = counts(pred, ref)
    TP, FP, FN = tp.sum(0), fp.sum(0), fn.sum(0)
    TN = ((ref == 0) & (pred == 0)).sum(0)
    f1, s5, a13, mi = summary_from_sums(TP, FP, FN)
    per = {}
    for j, l in enumerate(LABELS_14):
        per[l] = dict(TP=int(TP[j]), FP=int(FP[j]), FN=int(FN[j]), TN=int(TN[j]),
                      sens=float(TP[j] / (TP[j] + FN[j])) if TP[j] + FN[j] else 0.0,
                      spec=float(TN[j] / (TN[j] + FP[j])) if TN[j] + FP[j] else 0.0,
                      f1=float(f1[j]))
    return dict(per_label=per, steer_5=float(s5), all_13=float(a13), micro_14=float(mi), n=int(len(ref)))


def bootstrap(preds, ref, rng):
    """Paired study-level bootstrap of (condition - baseline)."""
    n = len(ref)
    cnt = {k: counts(v, ref) for k, v in preds.items()}
    out = {k: {"steer_5": [], "all_13": [], "micro_14": []} for k in preds if k != "baseline"}
    for _ in range(BOOT_N):
        idx = rng.integers(0, n, n)
        res = {}
        for k, (tp, fp, fn) in cnt.items():
            _, s5, a13, mi = summary_from_sums(tp[idx].sum(0), fp[idx].sum(0), fn[idx].sum(0))
            res[k] = (s5, a13, mi)
        for k in out:
            for m, i in (("steer_5", 0), ("all_13", 1), ("micro_14", 2)):
                out[k][m].append(res[k][i] - res["baseline"][i])
    ci = {}
    for k, d in out.items():
        ci[k] = {}
        for m, arr in d.items():
            arr = np.array(arr)
            lo, hi = np.percentile(arr, [2.5, 97.5])
            ci[k][m] = dict(lo=float(lo), hi=float(hi), excludes_zero=bool(lo > 0 or hi < 0))
    return ci


def make_or_load_cohort(manifest, out_dir):
    ref_all = [s for s in manifest if (s.get("section_findings") or "").strip()]
    ref_truthy = [s for s in manifest if s.get("section_findings")]
    assert [s["study_id"] for s in ref_all] == [s["study_id"] for s in ref_truthy], \
        "Reference-study filters disagree (whitespace-only findings)."
    assert len(ref_all) == 2255, f"Expected 2255 reference studies, got {len(ref_all)}."
    sweep_ids = [s["study_id"] for s in ref_all[:N_SWEEP]]
    remaining = [s["study_id"] for s in ref_all[N_SWEEP:]]
    assert len(remaining) == 2055

    path = out_dir / "cohort_2000.json"
    if path.exists():
        c = json.load(open(path))
        assert c["sweep_ids"] == sweep_ids and len(c["eval_ids"]) == N_EVAL, "Cohort file inconsistent."
        assert set(c["eval_ids"]) <= set(remaining) and not set(c["eval_ids"]) & set(sweep_ids)
        return c, ref_all
    rng = np.random.default_rng(COHORT_SEED)
    pick = np.sort(rng.choice(len(remaining), size=N_EVAL, replace=False))
    eval_ids = [remaining[i] for i in pick]
    c = dict(created=time.strftime("%Y-%m-%d %H:%M:%S"), seed=COHORT_SEED,
             rule="test studies with non-empty findings; first 200 (lambda sweep) excluded; "
                  "random 2000 of remaining 2055",
             n_reference=len(ref_all), sweep_ids=sweep_ids, eval_ids=eval_ids)
    json.dump(c, open(path, "w"), indent=1)
    return c, ref_all


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["8b", "3b"], required=True)
    args = ap.parse_args()
    P = inputs(args.model)
    out_root = paths.out("evaluation"); out_dir = out_root / args.model
    out_dir.mkdir(parents=True, exist_ok=True)

    from f1chexbert import F1CheXbert

    log_f = open(out_dir / "evaluate.log", "a")
    def log(m):
        line = f"[{time.strftime('%H:%M:%S')}] {m}"; print(line, flush=True); log_f.write(line + "\n"); log_f.flush()

    log(f"PARE evaluation: {args.model}")
    manifest = json.load(open(P["manifest"]))
    bl_by = {r["study_id"]: r for r in load_baseline(P["baseline"])}
    pa_list = json.load(open(P["pare_dir"] / "test_reports.json"))
    pa_by = {r["study_id"]: r for r in pa_list}
    assert set(bl_by) == set(pa_by) == {s["study_id"] for s in manifest}, "Study IDs do not match."
    assert not any("_error" in r for r in pa_list), "test_reports.json contains generation errors."
    lam = {r["_provenance"]["lambda"] for r in pa_list}
    log(f"Steered reports lambda: {lam}")
    assert lam == ({1.0} if args.model == "8b" else {0.13}), "Steered reports have an unexpected lambda."

    cohort, ref_all = make_or_load_cohort(manifest, out_root)
    log(f"Cohort: {len(cohort['eval_ids'])} studies (seed {cohort['seed']}), {len(cohort['sweep_ids'])} sweep studies excluded")

    scorer = F1CheXbert(device="cuda:0")
    L = Labeler(scorer)

    # Templates must be CheXbert-positive for their own pathology
    for p, t in TEMPLATES.items():
        assert L.rrg(clean(t).lower())[IDX[p]] == 1, f"Template for {p} not CheXbert-positive."

    def raw_text(sid):
        pr = pa_by[sid]
        return pr["candidate_findings"] if pr.get("steered") else bl_by[sid]["candidate_findings"]

    # Step 2: conditions on the 2,000-study cohort
    ids = cohort["eval_ids"]
    log(f"Step 2: merges on {len(ids)} cohort studies ...")
    merged_text, template_text = {}, {}
    act_merged = Counter()
    for n_done, sid in enumerate(ids, 1):
        b_raw = bl_by[sid].get("candidate_findings", "") or ""
        pr = pa_by[sid]
        targets = list(pr.get("targets", [])) if pr.get("steered") else []
        bl_cls = L.cls(b_raw)
        bl_rrg = L.rrg(b_raw)
        negated = {p for p in targets if bl_cls[IDX[p]] == 0}   # explicit CheXbert "negative"

        if targets:
            st_raw = pr.get("candidate_findings", "") or ""
            m, acts = safe_merge_negation_aware(b_raw, st_raw, targets, scorer)
            for a in acts.values(): act_merged[a] += 1
            merged_text[sid] = m
        else:
            merged_text[sid] = b_raw

        # Probe + template: same gate and same baseline conditions as the merger,
        # but appends a fixed sentence instead of a steered one.
        add = [TEMPLATES[p] for p in targets if bl_rrg[IDX[p]] == 0 and p not in negated]
        t = b_raw.rstrip()
        if add and t:
            if not t.endswith("."): t += "."
            t += " " + " ".join(add)
        template_text[sid] = t if t else b_raw
        if n_done % 250 == 0: log(f"  {n_done}/{len(ids)}")

    ref = L.eval_labels([pa_by[i].get("section_findings", "") or "" for i in ids])
    preds = {
        "baseline": L.eval_labels([bl_by[i]["candidate_findings"] for i in ids]),
        "raw": L.eval_labels([raw_text(i) for i in ids]),
        "merged": L.eval_labels([merged_text[i] for i in ids]),
        "probe_template": L.eval_labels([template_text[i] for i in ids]),
    }
    metrics = {k: full_metrics(v, ref) for k, v in preds.items()}

    # Step 3: bootstrap
    log("Step 3: bootstrap ...")
    ci = bootstrap(preds, ref, np.random.default_rng(BOOT_SEED))

    # Step 4: probes
    bl = preds["baseline"]; raw = preds["raw"]
    probe = {}
    for p in STEER_5:
        j = IDX[p]
        scores = np.array([pa_by[i]["probe_scores"][p] for i in ids])
        fired = np.array([p in pa_by[i].get("targets", []) for i in ids])
        fn_mask = (ref[:, j] == 1) & (bl[:, j] == 0)
        tp_mask = (ref[:, j] == 1) & (bl[:, j] == 1)
        probe[p] = dict(
            auroc=float(roc_auc_score(ref[:, j], scores)) if 0 < ref[:, j].sum() < len(ids) else None,
            n_fn=int(fn_mask.sum()), fn_flagged=int((fired & fn_mask).sum()),
            fn_recall=float((fired & fn_mask).sum() / fn_mask.sum()) if fn_mask.sum() else None,
            gate_rate=float(fired.mean()),
            n_baseline_tp=int(tp_mask.sum()), tp_lost_raw=int((tp_mask & (raw[:, j] == 0)).sum()),
            tp_lost_raw_frac=float((tp_mask & (raw[:, j] == 0)).sum() / tp_mask.sum()) if tp_mask.sum() else None,
        )
    steered = [pa_by[i] for i in ids if pa_by[i].get("steered")]
    steering = dict(frac_steered=len(steered) / len(ids),
                    mean_targets_when_steered=float(np.mean([r["n_targets"] for r in steered])) if steered else 0.0)

    result = dict(model=args.model, cohort_file=str(out_root / "cohort_2000.json"), n=len(ids),
                  metrics=metrics, bootstrap_ci=ci, probes=probe, steering=steering,
                  merger_actions=dict(act_merged), templates=TEMPLATES,
                  bootstrap=dict(n=BOOT_N, seed=BOOT_SEED, unit="study", ci="percentile 95%"))
    json.dump(result, open(out_dir / "eval_2000.json", "w"), indent=2)
    json.dump(merged_text, open(out_dir / "merged_reports_2000.json", "w"), indent=1)
    json.dump(template_text, open(out_dir / "template_reports_2000.json", "w"), indent=1)

    # Readable summary
    S = [f"PARE evaluation, {args.model}, n={len(ids)} (cohort seed {cohort['seed']})", ""]
    S.append(f"{'condition':16s} {'Steer-5':>8s} {'All-13':>8s} {'Micro-14':>9s}   CI(dS5)            CI(dA13)           CI(dMi14)")
    for k, m in metrics.items():
        row = f"{k:16s} {m['steer_5']:8.4f} {m['all_13']:8.4f} {m['micro_14']:9.4f}"
        if k in ci:
            row += "   " + "  ".join(f"[{ci[k][x]['lo']:+.4f},{ci[k][x]['hi']:+.4f}]{'*' if ci[k][x]['excludes_zero'] else ' '}"
                                     for x in ("steer_5", "all_13", "micro_14"))
        S.append(row)
    S += ["", f"{'pathology':28s}" + "".join(f"{k[:9]:>10s}" for k in metrics)]
    for l in LABELS_14:
        S.append(f"{l:28s}" + "".join(f"{metrics[k]['per_label'][l]['f1']:10.4f}" for k in metrics))
    S += ["", "Probes (steered pathologies):"]
    for p, d in probe.items():
        S.append(f"  {p:18s} AUROC={d['auroc']}  FN flagged {d['fn_flagged']}/{d['n_fn']} = {d['fn_recall']}  "
                 f"gate={d['gate_rate']:.3f}  raw TP lost {d['tp_lost_raw']}/{d['n_baseline_tp']}")
    S += ["", f"Steered: {steering['frac_steered']:.3f}, mean targets {steering['mean_targets_when_steered']:.2f}",
          f"Merger actions: {dict(act_merged)}"]
    open(out_dir / "summary.txt", "w").write("\n".join(S) + "\n")
    log("\n".join(S)); log("Done")


if __name__ == "__main__":
    main()
