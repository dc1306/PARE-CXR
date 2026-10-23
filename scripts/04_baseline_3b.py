#!/usr/bin/env python3
"""Step 4: CheXagent-3B baseline reports (test and training studies).

Generates unsteered CheXagent-2-3B reports for the test manifest (step 1) and
the 15,000 training studies (step 2). The model takes a single image: the first
image of each study after frontal-first ordering.

Protocol:
  - Chat template with system prompt "You are a helpful assistant.".
  - Greedy decoding (do_sample=False, num_beams=1, max_new_tokens=512).
  - Every output records the protocol, checkpoint and generation settings;
    cached outputs are reused only if these match exactly.

Outputs ($PARE_OUT/baseline_and_components_3b/):
  baseline_reports.json, train_baseline_reports.json

Usage:
  python scripts/04_baseline_3b.py
"""
import sys as _sys; from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))  # repo root
from pare import paths
import os

import torch, json, time, warnings, hashlib
from PIL import Image
warnings.filterwarnings("ignore")

def manifest_provenance_hash(manifest):
    """Hash study IDs + image paths + indications for artifact binding."""
    fp = [{'study_id': s['study_id'], 'image_paths': s.get('image_paths', []),
           'indication': s.get('section_indication', '')} for s in manifest]
    return hashlib.md5(json.dumps(fp, sort_keys=True).encode()).hexdigest()

CANON = paths.out("test_set_and_baseline_8b")
TRAINVAL_DIR = paths.out("trainval_features_8b")
EXP_DIR = paths.out("baseline_and_components_3b")
EXP_DIR.mkdir(exist_ok=True)

CHECKPOINT = "StanfordAIMI/CheXagent-2-3b"
DEVICE = "cuda:0"; DTYPE = torch.bfloat16
PROTOCOL_VERSION = "chat_system_single_image"
IMAGE_POLICY = "single_frontal"  # one image per study

# Greedy decoding
GEN_CONFIG = dict(
    do_sample=False, num_beams=1, max_new_tokens=512,
    temperature=1.0, top_p=1.0, use_cache=True,
)

LOG = EXP_DIR / "baseline.log"
def log(msg):
    ts = time.strftime("%Y-%m-%d %H:%M:%S"); line = f"[{ts}] {msg}"
    print(line, flush=True)
    with open(LOG, "a") as f: f.write(line + "\n")


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


def validate_completed(results, manifest):
    """Full validation of a completed result set."""
    if len(results) != len(manifest):
        return False, f"count {len(results)} != {len(manifest)}"
    r_sids = [r['study_id'] for r in results]
    m_sids = [s['study_id'] for s in manifest]
    if r_sids != m_sids:
        return False, "ID ordering mismatch"
    if len(set(r_sids)) != len(r_sids):
        return False, "duplicate IDs"
    for r in results:
        if '_error' in r:
            return False, f"contains error for {r['study_id']}"
        if not r.get('candidate_findings', '').strip():
            return False, f"empty report for {r['study_id']}"
    return True, "OK"


def validate_protocol(metadata, manifest=None):
    """Verify saved artifact was generated with the current frozen protocol."""
    if metadata.get('protocol_version') != PROTOCOL_VERSION:
        return False, f"protocol {metadata.get('protocol_version')} != {PROTOCOL_VERSION}"
    if metadata.get('checkpoint') != CHECKPOINT:
        return False, "checkpoint mismatch"
    if metadata.get('image_policy') != IMAGE_POLICY:
        return False, "image_policy mismatch"
    saved_gc = metadata.get('gen_config')
    if saved_gc and saved_gc != GEN_CONFIG:
        return False, f"gen_config mismatch: saved={saved_gc}"
    if manifest is not None:
        expected_hash = manifest_provenance_hash(manifest)
        if metadata.get('manifest_hash') and metadata['manifest_hash'] != expected_hash:
            return False, "manifest_hash mismatch"
    return True, "OK"


def generate_split(model, tok, manifest, out_path, ckpt_path, split_name):
    log(f"\n{'='*60}")
    log(f"Baseline generation, CheXagent-3B, {split_name}")
    log(f"  Studies: {len(manifest)}, Output: {out_path}")
    log(f"  Protocol: {PROTOCOL_VERSION}, Image: {IMAGE_POLICY}")
    log(f"  Gen config: {GEN_CONFIG}")
    log(f"{'='*60}")

    m_hash = manifest_provenance_hash(manifest)

    # Reuse existing outputs only if the protocol and manifest match
    if out_path.exists():
        saved = json.load(open(out_path))
        if not (isinstance(saved, dict) and '_metadata' in saved):
            log("  Existing file has no protocol metadata; regenerating.")
        else:
            metadata = saved['_metadata']
            results_list = saved['results']
            proto_ok, proto_reason = validate_protocol(metadata, manifest)
            if proto_ok:
                valid, reason = validate_completed(results_list, manifest)
                if valid:
                    log(f"  Already complete: {len(results_list)} reports, protocol+manifest verified ")
                    return results_list
                else:
                    log(f"  Existing file invalid ({reason}). Re-running...")
            else:
                log(f"  Existing file outdated ({proto_reason}). Regenerating...")

    # Resume from a validated checkpoint
    results = []; start_idx = 0
    if ckpt_path.exists():
        ckpt = json.load(open(ckpt_path))
        ckpt_results = ckpt.get("results", [])
        ckpt_meta = ckpt.get("_metadata", {})
        ckpt_proto = ckpt_meta.get("protocol_version")
        ckpt_mhash = ckpt_meta.get("manifest_hash")
        if ckpt_proto != PROTOCOL_VERSION or (ckpt_mhash and ckpt_mhash != m_hash):
            log("  Checkpoint protocol/manifest mismatch. Discarding.")
        else:
            has_errors = any('_error' in r or not r.get('candidate_findings', '').strip()
                             for r in ckpt_results)
            if has_errors:
                log("  Checkpoint contains errors/empty reports, discarding")
            else:
                for j in range(len(ckpt_results)):
                    if j >= len(manifest) or ckpt_results[j]['study_id'] != manifest[j]['study_id']:
                        log(f"  Checkpoint ID mismatch at {j}, discarding")
                        break
                else:
                    results = ckpt_results
                    start_idx = len(results)
                    log(f"  Resuming from {start_idx} (verified, protocol={PROTOCOL_VERSION})")

    errors = 0; t0 = time.time()
    for i in range(start_idx, len(manifest)):
        sample = manifest[i]
        sid = sample["study_id"]
        tmp_path = f"/tmp/_pare3b_bl_{os.getpid()}.png"
        try:
            indication = sample.get("section_indication") or "None provided"

            # Save the image as PNG, as expected by the CheXagent-3B processor
            # The temporary file must exist until generate() returns (images load lazily)
            Image.open(sample["image_paths"][0]).convert("RGB").save(tmp_path)
            input_ids = build_input_ids(tok, tmp_path, indication, DEVICE)
            input_len = input_ids.shape[1]

            with torch.no_grad():
                output_ids = model.generate(
                    input_ids, pad_token_id=tok.pad_token_id, **GEN_CONFIG)
            text = tok.decode(output_ids[0][input_len:], skip_special_tokens=True).strip()
            assert text, f"Empty generation for study {sid}"

            results.append({
                "study_id": sid, "subject_id": sample.get("subject_id"),
                "candidate_findings": text,
                "section_findings": sample.get("section_findings", ""),
                "section_indication": indication,
                "image_paths": sample.get("image_paths", []),
            })
        except Exception as e:
            errors += 1
            log(f"  Error {sid}: {e}")
            results.append({
                "study_id": sid, "subject_id": sample.get("subject_id"),
                "candidate_findings": "", "_error": str(e),
                "section_findings": sample.get("section_findings", ""),
                "image_paths": sample.get("image_paths", []),
            })
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
        done = i + 1
        if done % 25 == 0:
            json.dump({"results": results, "_metadata": {
                "protocol_version": PROTOCOL_VERSION, "checkpoint": CHECKPOINT,
                "image_policy": IMAGE_POLICY, "manifest_hash": m_hash,
            }}, open(ckpt_path, "w"))
        if done % 100 == 0:
            elapsed = time.time() - t0
            rate = (done - start_idx) / elapsed if elapsed > 0 else 0
            eta = (len(manifest) - done) / rate if rate > 0 else 0
            log(f"  {done}/{len(manifest)} ({rate:.2f}/s, ETA:{eta/60:.0f}m, errors={errors})")

    # Stop on any error
    assert errors == 0, \
        f"Error: {errors} generation errors in {split_name}. Fix and re-run."

    # Validate outputs
    valid, reason = validate_completed(results, manifest)
    assert valid, f"Error: Final validation failed: {reason}"

    # Save with protocol metadata
    output = {
        "results": results,
        "_metadata": {
            "protocol_version": PROTOCOL_VERSION,
            "checkpoint": CHECKPOINT,
            "image_policy": IMAGE_POLICY,
            "gen_config": GEN_CONFIG,
            "manifest_hash": m_hash,
            "n_studies": len(results),
            "split": split_name,
        }
    }
    json.dump(output, open(out_path, "w"), indent=1, ensure_ascii=False)
    if ckpt_path.exists(): ckpt_path.unlink()
    log(f"  {split_name} Complete: {len(results)} reports, 0 errors")
    return results


if __name__ == "__main__":
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model = AutoModelForCausalLM.from_pretrained(
        CHECKPOINT, trust_remote_code=True).to(DTYPE).to(DEVICE)
    model.eval()
    tok = AutoTokenizer.from_pretrained(CHECKPOINT, trust_remote_code=True)
    if tok.pad_token_id is None: tok.pad_token_id = tok.eos_token_id
    log(f"Model loaded: {CHECKPOINT}")
    log(f"Protocol: {PROTOCOL_VERSION}, Image: {IMAGE_POLICY}")
    log(f"Gen config: {GEN_CONFIG}")

    # Test set
    test_manifest = json.load(open(CANON / "study_test_manifest.json"))
    generate_split(model, tok, test_manifest,
                   EXP_DIR / "baseline_reports.json",
                   EXP_DIR / "baseline_checkpoint.json", "TEST")

    # Training set (needed for the TP/FN sets)
    train_manifest = json.load(open(TRAINVAL_DIR / "train_manifest.json"))
    generate_split(model, tok, train_manifest,
                   EXP_DIR / "train_baseline_reports.json",
                   EXP_DIR / "train_baseline_checkpoint.json", "TRAIN")

    log(f"\n{'='*60}")
    log("All baselines complete")
    log(f"{'='*60}")
