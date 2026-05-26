"""Post-hoc re-evaluation of exam predictions using KT baselines.

Reads an existing eval run's practice_trajectories.jsonl and exam_set.json,
then re-predicts exam performance using BKT / DKT (KC-level)
instead of KT2.

Trained model files are stored in a shared directory independent of any eval run:
    output/exam_eval/Eval-Shared/KT-Models/{dataset_name}/bkt_params.json
    output/exam_eval/Eval-Shared/KT-Models/{dataset_name}/dkt_model.pt
Override with --kt-models-root.  Dataset name is inferred from --dataset-dir.

All KT methods are trained once per dataset (not per module) using only
train-split students from all modules.

Eval results (exam_predictions_*.jsonl, module_summary_*.json) are saved to:
    <run-dir>/<module>/kt_baselines/
Override with --output-dir.

Usage examples:

  # BKT — first run: fits params, saves to shared KT-Models dir
  python reeval_with_kt_baselines.py \
      --run-dir output/exam_eval/Eval-Result/<run_folder> \
      --module Application_Module \
      --kt-method bkt

  # DKT — first run: trains model, saves checkpoint to shared KT-Models dir
  python reeval_with_kt_baselines.py \
      --run-dir output/exam_eval/Eval-Result/<run_folder> \
      --module Application_Module \
      --kt-method dkt \
      --device cuda \
      --dkt-epochs 50

  # All methods (BKT + DKT)
  python reeval_with_kt_baselines.py \
      --run-dir output/exam_eval/Eval-Result/<run_folder> \
      --module Application_Module \
      --kt-method all \
      --device cuda
"""
from __future__ import annotations

import argparse
import json
import random
import statistics
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Base directories derived from ROOT.
DEFAULT_DATASET_ROOT = ROOT / "data" / "dataset"
DEFAULT_KT_MODELS_ROOT = ROOT / "output" / "exam_eval" / "Eval-Shared" / "KT-Models"

from eval.kt_baselines.data_utils import (
    load_question_info,
    build_qid_to_kc,
    load_recordings,
    recordings_to_kc_sequences,
    build_kc_index,
    extract_burn_in,
    load_practice_traces,
    load_exam_set,
    collect_kc_observations,
    load_sids_from_graph_dirs,
)
from eval.kt_baselines.bkt import (
    BKTPredictor,
    fit_bkt_all_kcs,
    save_bkt_params,
    load_bkt_params,
)


# ------------------------------------------------------------------
# Core re-evaluation logic
# ------------------------------------------------------------------

def reeval_bkt(
    *,
    practice_traces: Dict[str, List[Tuple[str, int]]],
    burn_in_seqs: Dict[str, List[Tuple[str, int]]],
    exam_set: List[Dict[str, Any]],
    bkt_params_path: str | None,
    kc_observations: Dict[str, List[List[int]]] | None,
    output_dir: Path,
    model_dir: Path,
) -> Dict[str, Any]:
    """Re-evaluate exam predictions using BKT.

    Model files are saved to / loaded from `model_dir` (shared across runs).
    Eval results are written to `output_dir` (per-run).
    """
    default_params_path = model_dir / "bkt_params.json"

    # Load or fit BKT parameters
    resolved_params_path = bkt_params_path or (str(default_params_path) if default_params_path.is_file() else None)
    if resolved_params_path and Path(resolved_params_path).is_file():
        print(f"Loading BKT params from {resolved_params_path}")
        kc_params = load_bkt_params(resolved_params_path)
    else:
        if kc_observations is None:
            raise ValueError("Must provide either --bkt-params-path or dataset for fitting.")
        print(f"Fitting BKT on {len(kc_observations)} KCs ...")
        kc_params = fit_bkt_all_kcs(kc_observations)
        model_dir.mkdir(parents=True, exist_ok=True)
        save_bkt_params(kc_params, str(default_params_path))
        print(f"Saved BKT params to {default_params_path}")

    # Re-evaluate each student
    student_results: List[Dict[str, Any]] = []
    for sid, practice_seq in practice_traces.items():
        predictor = BKTPredictor(kc_params)

        # Process burn-in observations (from original dataset)
        burn_in = burn_in_seqs.get(sid, [])
        if burn_in:
            predictor.process_sequence(burn_in)

        # Predict BEFORE practice (initial)
        initial_preds = []
        for q in exam_set:
            p = predictor.predict(q["kc"])
            initial_preds.append({
                "question_id": q["question_id"],
                "kc": q["kc"],
                "difficulty": q.get("difficulty", "medium"),
                "p_correct": p,
            })

        # Process practice
        predictor.process_sequence(practice_seq)

        # Predict AFTER practice (final)
        final_preds = []
        for q in exam_set:
            p = predictor.predict(q["kc"])
            final_preds.append({
                "question_id": q["question_id"],
                "kc": q["kc"],
                "difficulty": q.get("difficulty", "medium"),
                "p_correct": p,
            })

        mean_initial = statistics.fmean(x["p_correct"] for x in initial_preds)
        mean_final = statistics.fmean(x["p_correct"] for x in final_preds)
        student_results.append({
            "student_id": sid,
            "mean_exam_p_correct_initial": mean_initial,
            "mean_exam_sampled_correct_initial": mean_initial,
            "mean_exam_p_correct": mean_final,
            "mean_exam_sampled_correct": mean_final,
            "mean_exam_p_correct_delta": mean_final - mean_initial,
            "mean_exam_sampled_correct_delta": mean_final - mean_initial,
            "exam_predictions_initial": initial_preds,
            "exam_predictions": final_preds,
        })

    return _build_summary("bkt", student_results, exam_set, output_dir)


def reeval_dkt(
    *,
    practice_traces: Dict[str, List[Tuple[str, int]]],
    burn_in_seqs: Dict[str, List[Tuple[str, int]]],
    exam_set: List[Dict[str, Any]],
    kc_sequences: Dict[str, List[Tuple[str, int]]],
    kc_to_idx: Dict[str, int],
    dkt_checkpoint: str | None,
    dkt_epochs: int,
    dkt_hidden_dim: int,
    dkt_lr: float,
    dkt_batch_size: int,
    dkt_max_seq_len: int,
    device: str,
    output_dir: Path,
    model_dir: Path,
    dkt_train_sids: set[str],
) -> Dict[str, Any]:
    """Re-evaluate exam predictions using DKT.

    Model checkpoint is saved to / loaded from `model_dir` (shared across runs).
    Eval results are written to `output_dir` (per-run).
    `dkt_train_sids`: if non-empty, only these students are used for training
    (i.e., train-split students from all modules). If empty, uses all recordings.
    """
    import torch
    from eval.kt_baselines.dkt import (
        DKTModel,
        DKTPredictor,
        train_dkt,
        save_dkt_checkpoint,
        load_dkt_checkpoint,
    )

    num_kcs = len(kc_to_idx)
    default_ckpt_path = model_dir / "dkt_model.pt"

    # Resolve checkpoint path: explicit arg > shared model dir > train from scratch
    resolved_ckpt = dkt_checkpoint or (str(default_ckpt_path) if default_ckpt_path.is_file() else None)

    if resolved_ckpt and Path(resolved_ckpt).is_file():
        print(f"Loading DKT checkpoint from {resolved_ckpt}")
        model, loaded_kc_to_idx = load_dkt_checkpoint(resolved_ckpt, device=device)
        kc_to_idx = loaded_kc_to_idx
        num_kcs = model.num_kcs
    else:
        # Prepare training data — only train-split students (or all if not specified)
        if dkt_train_sids:
            train_seqs_raw = {
                sid: seq for sid, seq in kc_sequences.items()
                if sid in dkt_train_sids
            }
        else:
            train_seqs_raw = dict(kc_sequences)
        print(f"Training DKT: {len(train_seqs_raw)} students, {num_kcs} KCs")

        # Convert to idx sequences, truncate to max_seq_len
        train_seqs_idx: List[List[Tuple[int, int]]] = []
        for sid, seq in train_seqs_raw.items():
            idx_seq = []
            for kc, correct in seq:
                kc_idx = kc_to_idx.get(kc)
                if kc_idx is not None:
                    idx_seq.append((kc_idx, correct))
            if len(idx_seq) >= 3:
                if dkt_max_seq_len > 0 and len(idx_seq) > dkt_max_seq_len:
                    idx_seq = idx_seq[:dkt_max_seq_len]
                train_seqs_idx.append(idx_seq)
        # Free raw data no longer needed
        del train_seqs_raw

        # Split train/val (90/10)
        rng = random.Random(42)
        rng.shuffle(train_seqs_idx)
        split_idx = int(len(train_seqs_idx) * 0.9)
        train_data = train_seqs_idx[:split_idx]
        val_data = train_seqs_idx[split_idx:]

        model = DKTModel(num_kcs=num_kcs, hidden_dim=dkt_hidden_dim)
        print(f"  Train: {len(train_data)} seqs, Val: {len(val_data)} seqs")
        history = train_dkt(
            model, train_data,
            num_kcs=num_kcs,
            epochs=dkt_epochs,
            batch_size=dkt_batch_size,
            lr=dkt_lr,
            device=device,
            val_sequences=val_data if val_data else None,
            patience=10,
        )

        model_dir.mkdir(parents=True, exist_ok=True)
        save_dkt_checkpoint(model, kc_to_idx, str(default_ckpt_path), history)
        print(f"Saved DKT checkpoint to {default_ckpt_path}")

    # Inference
    predictor = DKTPredictor(model, kc_to_idx, device=device)

    student_results: List[Dict[str, Any]] = []
    for sid, practice_seq in practice_traces.items():
        predictor.reset()

        # Process burn-in
        burn_in = burn_in_seqs.get(sid, [])
        if burn_in:
            predictor.process_sequence(burn_in)

        # Predict BEFORE practice
        initial_preds = []
        for q in exam_set:
            p = predictor.predict(q["kc"])
            initial_preds.append({
                "question_id": q["question_id"],
                "kc": q["kc"],
                "difficulty": q.get("difficulty", "medium"),
                "p_correct": p,
            })

        # Process practice
        predictor.process_sequence(practice_seq)

        # Predict AFTER practice
        final_preds = []
        for q in exam_set:
            p = predictor.predict(q["kc"])
            final_preds.append({
                "question_id": q["question_id"],
                "kc": q["kc"],
                "difficulty": q.get("difficulty", "medium"),
                "p_correct": p,
            })

        mean_initial = statistics.fmean(x["p_correct"] for x in initial_preds)
        mean_final = statistics.fmean(x["p_correct"] for x in final_preds)
        student_results.append({
            "student_id": sid,
            "mean_exam_p_correct_initial": mean_initial,
            "mean_exam_sampled_correct_initial": mean_initial,
            "mean_exam_p_correct": mean_final,
            "mean_exam_sampled_correct": mean_final,
            "mean_exam_p_correct_delta": mean_final - mean_initial,
            "mean_exam_sampled_correct_delta": mean_final - mean_initial,
            "exam_predictions_initial": initial_preds,
            "exam_predictions": final_preds,
        })

    return _build_summary("dkt", student_results, exam_set, output_dir)


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _build_summary(
    method: str,
    student_results: List[Dict[str, Any]],
    exam_set: List[Dict[str, Any]],
    output_dir: Path,
) -> Dict[str, Any]:
    """Write exam predictions JSONL and module summary JSON."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Write exam predictions
    exam_path = output_dir / f"exam_predictions_{method}.jsonl"
    with open(exam_path, "w", encoding="utf-8") as f:
        for rec in student_results:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    # Summary
    initials = [r["mean_exam_p_correct_initial"] for r in student_results]
    finals = [r["mean_exam_p_correct"] for r in student_results]
    deltas = [r["mean_exam_p_correct_delta"] for r in student_results]

    summary = {
        "kt_method": method,
        "num_students": len(student_results),
        "exam_set_size": len(exam_set),
        "avg_exam_score_initial": statistics.fmean(initials) if initials else 0.0,
        "avg_exam_score_final": statistics.fmean(finals) if finals else 0.0,
        "avg_exam_score_delta": statistics.fmean(deltas) if deltas else 0.0,
        "students": [
            {
                "student_id": r["student_id"],
                "mean_exam_p_correct_initial": r["mean_exam_p_correct_initial"],
                "mean_exam_p_correct": r["mean_exam_p_correct"],
                "mean_exam_p_correct_delta": r["mean_exam_p_correct_delta"],
            }
            for r in student_results
        ],
    }
    summary_path = output_dir / f"module_summary_{method}.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"\n[{method.upper()}] Results:")
    print(f"  Students: {len(student_results)}")
    print(f"  Avg initial: {summary['avg_exam_score_initial']:.4f}")
    print(f"  Avg final:   {summary['avg_exam_score_final']:.4f}")
    print(f"  Avg delta:   {summary['avg_exam_score_delta']:.4f}")
    print(f"  Saved to: {output_dir}")

    return summary


# ------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Post-hoc re-evaluation of exam predictions using BKT / DKT."
    )
    p.add_argument("--run-dir", required=True, help="Path to existing eval run directory.")
    p.add_argument(
        "--module",
        default="Application_Module",
        help=(
            "Module subfolder under <run-dir> — matches root_node: "
            "Application_Module / Computation_Module / Counting_Module (XES3G5M) or "
            "Wine_Knowledge / Circuit_Design / Education_Theory (MOOCRadar)."
        ),
    )
    p.add_argument(
        "--kt-method", required=True, choices=["bkt", "dkt", "all"],
        help="Which KT baseline to use for re-evaluation.",
    )

    # Dataset name — other paths default to <dataset_root>/<dataset>/...
    p.add_argument(
        "--dataset", default="XES3G5M",
        help="Dataset name (e.g. XES3G5M, MOOCRadar). Used to derive default paths for recordings, question_info, dataset-dir.",
    )

    # Dataset paths (override defaults derived from --dataset)
    p.add_argument(
        "--recordings-path", default="",
        help="Path to recordings.jsonl. Defaults to <dataset_root>/<dataset>/recordings.jsonl.",
    )
    p.add_argument(
        "--question-info-path", default="",
        help="Path to question_info.json. Defaults to <dataset_root>/<dataset>/question_info.json.",
    )
    p.add_argument("--burn-in-size", type=int, default=10, help="Number of burn-in observations.")

    # BKT-specific
    p.add_argument("--bkt-params-path", default="", help="Load pre-fitted BKT params from this file.")
    p.add_argument(
        "--bkt-max-seq-len", type=int, default=60,
        help="Truncate per-KC observation sequences for BKT fitting (0=no truncation). Default 60.",
    )

    # DKT-specific
    p.add_argument("--dkt-checkpoint", default="", help="Load pre-trained DKT model from this file.")
    p.add_argument("--dkt-epochs", type=int, default=50)
    p.add_argument("--dkt-hidden-dim", type=int, default=100)
    p.add_argument("--dkt-lr", type=float, default=0.001)
    p.add_argument("--dkt-batch-size", type=int, default=64)
    p.add_argument(
        "--dkt-max-seq-len", type=int, default=60,
        help="Truncate training sequences longer than this (0=no truncation). Default 60.",
    )

    p.add_argument("--device", default="cpu", help="Device for DKT (cpu or cuda).")

    # Paths for model storage and eval results
    p.add_argument(
        "--kt-models-root",
        default=str(DEFAULT_KT_MODELS_ROOT),
        help=(
            "Root directory for shared KT model files. "
            "Files saved to <root>/<dataset_name>/. "
            f"Default: {DEFAULT_KT_MODELS_ROOT}"
        ),
    )
    p.add_argument(
        "--dataset-dir", default="",
        help="Dataset root dir. Defaults to <dataset_root>/<dataset>/.",
    )
    p.add_argument(
        "--output-dir", default="",
        help="Directory for eval results (JSONL + summary). Defaults to <run-dir>/<module>/kt_baselines/",
    )

    return p.parse_args()


def main() -> None:
    args = parse_args()

    # Resolve paths from --dataset when not explicitly provided
    dataset_base = DEFAULT_DATASET_ROOT / args.dataset
    recordings_path = args.recordings_path or str(dataset_base / "recordings.jsonl")
    question_info_path = args.question_info_path or str(dataset_base / "question_info.json")
    dataset_dir_str = args.dataset_dir or str(dataset_base)

    run_dir = Path(args.run_dir).resolve()
    module_dir = run_dir / args.module

    # Load eval run data
    traj_path = module_dir / "practice_trajectories.jsonl"
    exam_set_path = module_dir / "exam_set.json"
    if not traj_path.is_file():
        raise FileNotFoundError(f"Missing: {traj_path}")
    if not exam_set_path.is_file():
        raise FileNotFoundError(f"Missing: {exam_set_path}")

    print(f"Dataset: {args.dataset} ({dataset_dir_str})")
    print("Loading eval run data ...")
    practice_traces = load_practice_traces(traj_path)
    exam_set = load_exam_set(exam_set_path)
    eval_student_ids = set(practice_traces.keys())
    print(f"  {len(practice_traces)} students, {len(exam_set)} exam questions")

    # Load dataset for fitting / training
    print("Loading dataset ...")
    question_info = load_question_info(question_info_path)
    qid_to_kc = build_qid_to_kc(question_info)
    recordings = load_recordings(recordings_path)
    kc_sequences = recordings_to_kc_sequences(recordings, qid_to_kc)
    del recordings  # free raw recordings (~500MB)
    kc_to_idx, idx_to_kc = build_kc_index(kc_sequences)
    print(f"  {len(kc_sequences)} students, {len(kc_to_idx)} KCs in dataset")

    # Extract burn-in sequences for eval students
    burn_in_seqs = extract_burn_in(
        {sid: kc_sequences[sid] for sid in eval_student_ids if sid in kc_sequences},
        args.burn_in_size,
    )
    print(f"  Burn-in: {len(burn_in_seqs)} students with burn-in data")

    # Directories — all KT methods share one model dir per dataset
    model_dir = Path(args.kt_models_root) / args.dataset
    output_dir = Path(args.output_dir) if args.output_dir else module_dir / "kt_baselines"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Collect train-split student IDs from all modules — KT methods only train on these
    dataset_dir = Path(dataset_dir_str)
    train_sids = load_sids_from_graph_dirs(dataset_dir, "train")
    test_sids = load_sids_from_graph_dirs(dataset_dir, "test")
    if train_sids:
        print(f"  KT training pool: {len(train_sids)} train students (all modules), "
              f"{len(test_sids)} test students excluded")
    else:
        print(f"  [warning] No students found in {dataset_dir}/*/train/; using all recordings")

    print(f"  Model dir (shared): {model_dir}")
    print(f"  Output dir (per-run): {output_dir}")

    if args.kt_method == "all":
        methods = ["bkt", "dkt"]
    else:
        methods = [args.kt_method]

    # Pre-compute KC observations for BKT (only if needed)
    train_kc_sequences = None
    if "bkt" in methods:
        train_kc_sequences = (
            {sid: seq for sid, seq in kc_sequences.items() if sid in train_sids}
            if train_sids else kc_sequences
        )

    for method in methods:
        print(f"\n{'='*60}")
        print(f"Running {method.upper()} re-evaluation ...")
        print(f"{'='*60}")

        if method == "bkt":
            kc_obs = collect_kc_observations(train_kc_sequences, max_seq_len=args.bkt_max_seq_len)
            reeval_bkt(
                practice_traces=practice_traces,
                burn_in_seqs=burn_in_seqs,
                exam_set=exam_set,
                bkt_params_path=args.bkt_params_path or None,
                kc_observations=kc_obs,
                output_dir=output_dir,
                model_dir=model_dir,
            )
        elif method == "dkt":
            reeval_dkt(
                practice_traces=practice_traces,
                burn_in_seqs=burn_in_seqs,
                exam_set=exam_set,
                kc_sequences=kc_sequences,
                kc_to_idx=kc_to_idx,
                dkt_checkpoint=args.dkt_checkpoint or None,
                dkt_epochs=args.dkt_epochs,
                dkt_hidden_dim=args.dkt_hidden_dim,
                dkt_lr=args.dkt_lr,
                dkt_batch_size=args.dkt_batch_size,
                dkt_max_seq_len=args.dkt_max_seq_len,
                device=args.device,
                output_dir=output_dir,
                model_dir=model_dir,
                dkt_train_sids=train_sids,
            )


if __name__ == "__main__":
    main()
