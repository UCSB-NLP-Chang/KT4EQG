"""Oracle-Gen-Verify practice evaluation pipeline.

Flow per practice round:
  1. Oracle scans all leaf KCs via V_edu -> selects best oracle_kc
  2. Build prompt with student state, inject forced prefix for oracle_kc
  3. Model generates question text (continuation after prefix)
  4. Parse JSON output; extract question_text (or use raw completion if parse fails)
  5. Verifier re-predicts KC by scoring question text against all leaf KCs
  6. Use verified KC for KT update (predict p_correct, sample response, update graph)
  7. After all practice rounds, evaluate on frozen exam set

Reuses shared exam sets and initial-eval caches from recommender_eval.py.

Usage:
  python eval/gen_practice_eval.py \
      --ckpt-path /path/to/checkpoint \
      --split test \
      --practice-rounds 10 \
      --exam-size 30 \
      --device cuda
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import random
import statistics
import sys
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from time import perf_counter
from typing import Any, Dict, List, Optional, Tuple

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from config.config import load_config
from eval.recommenders.ckpt_llm import resolve_ckpt_load_path
from eval.recommender_eval import (
    DIFFICULTIES,
    MODULE_STATE_DIRS,
    ROOT_TO_MODULE,
    QuestionSpec,
    _build_exam_combo_pool,
    _build_question_pool,
    _build_real_exam_index,
    _get_or_create_initial_eval_cache,
    _get_or_create_shared_exam_set,
    _initial_cache_path,
    _leaf_kcs,
    _load_question_info,
    _resolve_effective_exam_size,
    _safe_name_token,
    _shared_exam_path,
    _student_ids,
)
from kt.predict import predict_on_new_question
from kt.runtime import KTRuntime
from kt.update import (
    configure_temp_recordings_path,
    current_temp_recordings_path,
    one_step_update_single,
)
from value.value_fn import value_fn
from verl_bridge.chat_template_utils import apply_chat_template_compat
from verl_bridge.prompt_build import build_question_prompt


# ------------------------------------------------------------------
# Forced-prefix generation helpers
# ------------------------------------------------------------------

def build_forced_prefix(kc: str, medium_only: bool = True) -> str:
    """Build forced JSON prefix that fixes the KC in generation output.

    Replicates Stage1bDataset._build_forced_prefix from
    verl_bridge/stage1b_dataset.py.
    """
    if medium_only:
        return f'{{"knowledge_concept": "{kc}",'
    else:
        return f'{{"knowledge_concept": "{kc}","difficulty_level": "medium",'


def generate_with_prefix(
    llm,
    student_graph: Dict[str, Any],
    leaf_kcs: List[str],
    oracle_kc: str,
    *,
    student_id: Optional[str] = None,
    practice_size: Optional[int] = None,
    max_new_tokens: int = 256,
    temperature: float = 0.8,
    top_p: float = 0.9,
    medium_only: bool = True,
) -> Tuple[str, str, str]:
    """Generate a question with forced-prefix KC injection.

    Returns (completion, forced_prefix, full_output) where:
      - completion: only the model's generated continuation (after prefix)
      - forced_prefix: the injected prefix string
      - full_output: forced_prefix + completion (the complete JSON attempt)
    """
    system_prompt, user_prompt = build_question_prompt(
        student_graph, leaf_kcs,
        student_id=student_id, practice_size=practice_size,
    )
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    raw_prompt = apply_chat_template_compat(
        llm.tokenizer, messages, tokenize=False, add_generation_prompt=True,
    )

    forced_prefix = build_forced_prefix(oracle_kc, medium_only=medium_only)
    injected_prompt = raw_prompt + forced_prefix

    # Use BaseLLM.generate which strips input tokens and returns only new text
    completion = llm.generate(
        injected_prompt,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
    )
    full_output = forced_prefix + completion
    return completion, forced_prefix, full_output


def generate_free(
    llm,
    student_graph: Dict[str, Any],
    leaf_kcs: List[str],
    *,
    student_id: Optional[str] = None,
    practice_size: Optional[int] = None,
    max_new_tokens: int = 256,
    temperature: float = 0.8,
    top_p: float = 0.9,
) -> Tuple[str, str, str]:
    """Generate a question without forced prefix (model chooses KC freely).

    Returns (raw_output, "", raw_output) — same 3-tuple shape as
    generate_with_prefix for uniform handling downstream.
    """
    system_prompt, user_prompt = build_question_prompt(
        student_graph, leaf_kcs,
        student_id=student_id, practice_size=practice_size,
    )
    raw_output = llm.generate_chat(
        system_prompt, user_prompt,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
    )
    return raw_output, "", raw_output


# ------------------------------------------------------------------
# NLL computation
# ------------------------------------------------------------------

def _nll_for_text(model, text: str) -> float:
    """Mean negative log-likelihood per token for a plain text."""
    inputs = model.tokenizer(text, return_tensors="pt")
    inputs = {k: v.to(model.model.device) for k, v in inputs.items()}
    with torch.no_grad():
        out = model.model(**inputs, labels=inputs["input_ids"])
    return float(out.loss)


# ------------------------------------------------------------------
# JSON parsing
# ------------------------------------------------------------------

def _try_parse_question(full_output: str) -> Dict[str, Any] | None:
    """Try to parse the full output (prefix+completion) as JSON and extract question_text."""
    try:
        start = full_output.find("{")
        end = full_output.rfind("}")
        if start < 0 or end <= start:
            return None
        obj = json.loads(full_output[start : end + 1])
        if isinstance(obj, dict) and obj.get("question_text"):
            return obj
    except Exception:
        pass
    return None


# ------------------------------------------------------------------
# Verifier KC re-prediction (batch-efficient)
# ------------------------------------------------------------------

def verify_kc(
    verifier,
    text: str,
    leaf_kcs: List[str],
    difficulty: str = "medium",
) -> Tuple[str, Dict[str, float]]:
    """Score text against all leaf KCs and return the best-matching one.

    Returns (verified_kc, scores_dict) where scores_dict maps kc -> score.
    """
    scores: Dict[str, float] = {}
    for kc in leaf_kcs:
        try:
            scores[kc] = verifier.score_alignment(text, kc, difficulty=difficulty)
        except KeyError:
            # KC not in verifier vocab — skip
            scores[kc] = 0.0
    if not scores:
        raise RuntimeError("No leaf KC could be scored by verifier.")
    verified_kc = max(scores, key=scores.get)
    return verified_kc, scores


# ------------------------------------------------------------------
# Oracle KC selection (reuses value_fn scan from oracle_vedu.py)
# ------------------------------------------------------------------

def oracle_select_kc(
    rt: Any,
    student_id: str,
    kc_candidates: List[str],
    fixed_difficulty: str,
) -> Tuple[str, float, List[Tuple[str, float]]]:
    """Scan all KCs via value_fn and pick the best. Returns (best_kc, best_score, all_scores)."""
    scores: List[Tuple[str, float]] = []
    for kc in kc_candidates:
        question = {"kc": kc, "difficulty": fixed_difficulty, "question_text": ""}
        score = float(value_fn(question, rt, student_id))
        scores.append((kc, score))
    scores.sort(key=lambda x: x[1], reverse=True)
    best_kc, best_score = scores[0]
    return best_kc, best_score, scores


# ------------------------------------------------------------------
# Main practice loop
# ------------------------------------------------------------------

def _run_for_module(
    *,
    module_key: str,
    module_state_dir: str,
    dataset_dir: Path,
    split: str,
    burn_in_step: int,
    exam_size: int,
    practice_rounds: int,
    max_students: int,
    rng: random.Random,
    run_output_dir: Path,
    shared_exam_root: Path,
    shared_initial_root: Path,
    seed: int,
    practice_fixed_difficulty: str,
    practice_response_mode: str,
    exam_fixed_difficulty: str,
    exam_question_info: Dict[str, Any],
    # Generation-specific
    llm: Any,
    verifier: Any,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    medium_only: bool,
    free_gen: bool = False,
    ref_model: Any = None,
    compute_nll: bool = False,
) -> Dict[str, Any]:
    state_root = dataset_dir / module_state_dir / split
    if not state_root.is_dir():
        raise FileNotFoundError(f"State root not found: {state_root}")

    runtime_temp_dir = run_output_dir / "_runtime_cache" / module_key / "kt_temp_states"
    rt = KTRuntime(
        str(state_root),
        burn_in_size=burn_in_step,
        temp_dir=str(runtime_temp_dir),
    )
    sids = _student_ids(rt, max_students)
    if not sids:
        raise ValueError(f"No students found in {state_root}")

    rt.set_practice_size(sids[0], burn_in_step)
    sample_graph = rt.load_student_graph(sids[0], practice_size=burn_in_step, refresh=True)
    leaf_kcs = _leaf_kcs(sample_graph)
    full_pool = _build_question_pool(module_key, leaf_kcs)
    exam_combo_pool = _build_exam_combo_pool(module_key, leaf_kcs, exam_fixed_difficulty)
    real_exam_index = _build_real_exam_index(
        module_key=module_key,
        leaf_kcs=leaf_kcs,
        question_info=exam_question_info,
    )
    shared_exam_path = _shared_exam_path(
        shared_exam_root, module_key, exam_size, exam_fixed_difficulty,
    )
    exam_set = _get_or_create_shared_exam_set(
        shared_exam_path=shared_exam_path,
        rng=rng,
        combo_pool=exam_combo_pool,
        real_exam_index=real_exam_index,
        exam_size=exam_size,
    )
    medium_kc_candidates = [
        kc for kc in leaf_kcs
        if (kc, practice_fixed_difficulty) in {(q.kc, q.difficulty) for q in full_pool}
    ]
    if not medium_kc_candidates:
        raise ValueError(
            f"No practice candidates with difficulty={practice_fixed_difficulty} "
            f"for module={module_key}."
        )

    module_dir = run_output_dir / module_key
    module_dir.mkdir(parents=True, exist_ok=True)
    initial_cache_file = _initial_cache_path(
        shared_initial_root=shared_initial_root,
        module_key=module_key,
        burn_in_step=burn_in_step,
        exam_set=exam_set,
        sids=sids,
    )
    initial_by_student = _get_or_create_initial_eval_cache(
        path=initial_cache_file,
        rt=rt,
        sids=sids,
        burn_in_step=burn_in_step,
        exam_set=exam_set,
        seed=seed,
        module_key=module_key,
    )

    with open(module_dir / "exam_set.json", "w", encoding="utf-8") as f:
        json.dump([asdict(x) for x in exam_set], f, ensure_ascii=False, indent=2)

    trajectory_path = module_dir / "practice_trajectories.jsonl"
    exam_path = module_dir / "exam_predictions.jsonl"

    # Accumulators
    avg_exam_scores_initial: List[float] = []
    avg_exam_scores_final: List[float] = []
    coverage_ratios: List[float] = []
    coverage_counts: List[int] = []
    module_student_results: List[Dict[str, Any]] = []
    kc_match_counts: List[int] = []
    oracle_vedu_scores: List[float] = []
    parse_ok_counts: List[int] = []
    nll_values: List[float] = []
    csv_rows: List[Dict[str, Any]] = []

    with open(trajectory_path, "w", encoding="utf-8") as traj_f, \
         open(exam_path, "w", encoding="utf-8") as exam_f:

        for idx, sid in enumerate(sids, start=1):
            rt.clear_actual_temp_graphs()
            rt.set_practice_size(sid, burn_in_step)
            rt.load_student_graph(sid, practice_size=burn_in_step, refresh=True)
            initial_payload = initial_by_student.get(sid)
            if initial_payload is None:
                raise KeyError(f"Missing initial cache payload for student={sid}")
            initial_exam_records = initial_payload["exam_predictions_initial"]
            mean_exam_p_initial = float(initial_payload["mean_exam_p_correct_initial"])
            mean_exam_sampled_initial = float(initial_payload["mean_exam_sampled_correct_initial"])

            traces: List[Dict[str, Any]] = []
            student_kc_match = 0
            student_parse_ok = 0

            for rid in range(1, practice_rounds + 1):
                student_graph_now = rt.load_student_graph(sid, refresh=True)
                current_practice_size = rt.practice_size_for_student(sid)

                # Step 1: Oracle KC selection (skipped in free-gen mode)
                oracle_kc: Optional[str] = None
                oracle_score = 0.0
                oracle_elapsed = 0.0
                if not free_gen:
                    t0 = perf_counter()
                    oracle_kc, oracle_score, _all_oracle_scores = oracle_select_kc(
                        rt, sid, medium_kc_candidates, practice_fixed_difficulty,
                    )
                    oracle_elapsed = perf_counter() - t0
                    oracle_vedu_scores.append(oracle_score)

                # Step 2: Generate question
                t1 = perf_counter()
                if free_gen:
                    completion, forced_prefix, full_output = generate_free(
                        llm, student_graph_now, leaf_kcs,
                        student_id=sid,
                        practice_size=current_practice_size,
                        max_new_tokens=max_new_tokens,
                        temperature=temperature,
                        top_p=top_p,
                    )
                else:
                    completion, forced_prefix, full_output = generate_with_prefix(
                        llm, student_graph_now, leaf_kcs, oracle_kc,
                        student_id=sid,
                        practice_size=current_practice_size,
                        max_new_tokens=max_new_tokens,
                        temperature=temperature,
                        top_p=top_p,
                        medium_only=medium_only,
                    )
                gen_elapsed = perf_counter() - t1

                # Step 3: Parse and extract question text
                parsed = _try_parse_question(full_output)
                parse_ok = parsed is not None
                if parse_ok:
                    question_text = parsed["question_text"]
                    student_parse_ok += 1
                else:
                    # Use completion only (prefix stripped → no oracle KC leakage)
                    question_text = completion

                # Step 4: Verifier re-prediction
                t2 = perf_counter()
                verified_kc, verifier_scores = verify_kc(
                    verifier, question_text, leaf_kcs,
                    difficulty=practice_fixed_difficulty,
                )
                verify_elapsed = perf_counter() - t2

                kc_match = (oracle_kc == verified_kc) if oracle_kc is not None else None
                if kc_match:
                    student_kc_match += 1

                # Step 5: NLL computation (optional)
                nll = 0.0
                if compute_nll and ref_model is not None and question_text:
                    nll = _nll_for_text(ref_model, question_text)
                    nll_values.append(nll)

                # Step 6: KT update with verified KC
                if practice_response_mode == "always_correct":
                    p_correct = None
                    sampled = 1
                else:
                    pred_q = {
                        "kc": verified_kc,
                        "difficulty": practice_fixed_difficulty,
                        "question_text": question_text,
                    }
                    p_correct = float(predict_on_new_question(rt, sid, pred_q))
                    sampled = 1 if rng.random() < p_correct else 0

                obs = {
                    "question": f"GEN_{module_key}_{sid}_{rid}",
                    "kc": verified_kc,
                    "difficulty": practice_fixed_difficulty,
                    "response": sampled,
                    "source": "gen_practice_eval",
                }
                update_info = one_step_update_single(
                    rt, sid, [obs],
                    mode="actual",
                    tag=f"gen_eval_round_{rid}",
                    reset_temp=False,
                )

                # Top-5 verifier scores for debugging
                sorted_vscores = sorted(
                    verifier_scores.items(), key=lambda x: x[1], reverse=True,
                )[:5]

                traces.append({
                    "round_id": rid,
                    "kc": verified_kc,  # compat with reeval_with_kt_baselines
                    "oracle_kc": oracle_kc,
                    "oracle_vedu_score": oracle_score,
                    "verified_kc": verified_kc,
                    "kc_match": kc_match,
                    "generated_question_text": question_text,
                    "generation_parse_ok": parse_ok,
                    "nll": nll,
                    "p_correct": p_correct,
                    "sampled_correct": sampled,
                    "difficulty": practice_fixed_difficulty,
                    "verifier_scores_top5": [
                        {"kc": kc, "score": s} for kc, s in sorted_vscores
                    ],
                    "oracle_elapsed_sec": oracle_elapsed,
                    "gen_elapsed_sec": gen_elapsed,
                    "verify_elapsed_sec": verify_elapsed,
                    "updated_student_graph_path": (update_info or {}).get("student_graph"),
                    "updated_parameter_graph_path": (update_info or {}).get("parameter_graph"),
                })

                csv_rows.append({
                    "student_id": sid,
                    "practice_size": current_practice_size,
                    "kc": verified_kc,
                    "question_text": question_text,
                    "nll": nll,
                })

            kc_match_counts.append(student_kc_match)
            parse_ok_counts.append(student_parse_ok)

            # Exam evaluation (frozen, no updates)
            exam_records: List[Dict[str, Any]] = []
            for q in exam_set:
                pred_q = {"kc": q.kc, "difficulty": q.difficulty, "question_text": q.question_text}
                p = float(predict_on_new_question(rt, sid, pred_q))
                exam_records.append({
                    "question_id": q.question_id,
                    "kc": q.kc,
                    "difficulty": q.difficulty,
                    "p_correct": p,
                })

            mean_exam_p = statistics.fmean(x["p_correct"] for x in exam_records)
            mean_exam_sampled = mean_exam_p
            avg_exam_scores_initial.append(mean_exam_sampled_initial)
            avg_exam_scores_final.append(mean_exam_sampled)

            practiced_kcs = {t["verified_kc"] for t in traces}
            coverage_count = len(practiced_kcs)
            coverage_ratio = (coverage_count / len(leaf_kcs)) if leaf_kcs else 0.0
            coverage_counts.append(coverage_count)
            coverage_ratios.append(coverage_ratio)

            final_step = rt.practice_size_for_student(sid)
            final_state_path = rt.student_path(sid, final_step)

            traj_payload = {
                "student_id": sid,
                "module": module_key,
                "start_step": burn_in_step,
                "final_step": final_step,
                "practice_rounds": practice_rounds,
                "practice_trace": traces,
                "practice_unique_kc_count": coverage_count,
                "practice_kc_coverage_ratio": coverage_ratio,
                "final_state_graph": final_state_path,
            }
            exam_payload = {
                "student_id": sid,
                "module": module_key,
                "mean_exam_p_correct_initial": mean_exam_p_initial,
                "mean_exam_sampled_correct_initial": mean_exam_sampled_initial,
                "mean_exam_p_correct": mean_exam_p,
                "mean_exam_sampled_correct": mean_exam_sampled,
                "mean_exam_p_correct_delta": mean_exam_p - mean_exam_p_initial,
                "mean_exam_sampled_correct_delta": mean_exam_sampled - mean_exam_sampled_initial,
                "exam_predictions_initial": initial_exam_records,
                "exam_predictions": exam_records,
            }

            traj_f.write(json.dumps(traj_payload, ensure_ascii=False) + "\n")
            exam_f.write(json.dumps(exam_payload, ensure_ascii=False) + "\n")

            module_student_results.append({
                "student_id": sid,
                "mean_exam_p_correct_initial": mean_exam_p_initial,
                "mean_exam_sampled_correct_initial": mean_exam_sampled_initial,
                "mean_exam_p_correct": mean_exam_p,
                "mean_exam_sampled_correct": mean_exam_sampled,
                "mean_exam_p_correct_delta": mean_exam_p - mean_exam_p_initial,
                "mean_exam_sampled_correct_delta": mean_exam_sampled - mean_exam_sampled_initial,
                "practice_unique_kc_count": coverage_count,
                "practice_kc_coverage_ratio": coverage_ratio,
                "kc_match_count": student_kc_match,
                "kc_match_rate": student_kc_match / practice_rounds if practice_rounds > 0 else 0.0,
                "parse_ok_count": student_parse_ok,
                "parse_ok_rate": student_parse_ok / practice_rounds if practice_rounds > 0 else 0.0,
                "final_step": final_step,
            })

            if idx % 10 == 0 or idx == len(sids):
                print(f"[{module_key}] processed {idx}/{len(sids)} students")

    # Module summary
    total_rounds = sum(len(r.get("practice_trace", []) if isinstance(r, dict) else []) for r in [])  # not needed
    total_rounds_actual = len(sids) * practice_rounds
    total_kc_match = sum(kc_match_counts)
    total_parse_ok = sum(parse_ok_counts)

    def _mean_std(xs: List[float]):
        if not xs:
            return 0.0, 0.0
        m = statistics.fmean(xs)
        v = statistics.pstdev(xs) if len(xs) > 1 else 0.0
        return m, v

    nll_mean, nll_std = _mean_std(nll_values)

    summary: Dict[str, Any] = {
        "module": module_key,
        "pipeline": "free_gen_verify" if free_gen else "oracle_gen_verify",
        "free_gen": free_gen,
        "compute_nll": compute_nll,
        "num_students": len(sids),
        "num_leaf_kcs": len(leaf_kcs),
        "exam_set_size": len(exam_set),
        "exam_fixed_difficulty": exam_fixed_difficulty,
        "practice_rounds": practice_rounds,
        "practice_fixed_difficulty": practice_fixed_difficulty,
        "practice_response_mode": practice_response_mode,
        "initial_eval_cache_file": str(initial_cache_file),
        "avg_practice_unique_kc_count": statistics.fmean(coverage_counts) if coverage_counts else 0.0,
        "avg_practice_kc_coverage_ratio": statistics.fmean(coverage_ratios) if coverage_ratios else 0.0,
        "avg_exam_sampled_score_initial": statistics.fmean(avg_exam_scores_initial) if avg_exam_scores_initial else 0.0,
        "avg_exam_sampled_score_final": statistics.fmean(avg_exam_scores_final) if avg_exam_scores_final else 0.0,
        "avg_exam_sampled_score_delta": (
            statistics.fmean(avg_exam_scores_final) - statistics.fmean(avg_exam_scores_initial)
        ) if avg_exam_scores_initial and avg_exam_scores_final else 0.0,
        "avg_kc_match_rate": total_kc_match / total_rounds_actual if (total_rounds_actual > 0 and not free_gen) else None,
        "avg_parse_ok_rate": total_parse_ok / total_rounds_actual if total_rounds_actual > 0 else 0.0,
        "avg_oracle_vedu_score": statistics.fmean(oracle_vedu_scores) if oracle_vedu_scores else None,
        "avg_nll": nll_mean if compute_nll else None,
        "nll_std": nll_std if compute_nll else None,
        "students": module_student_results,
    }
    with open(module_dir / "module_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"\n[{module_key}] Summary ({'free-gen' if free_gen else 'oracle-gen'}):")
    print(f"  Students: {len(sids)}")
    print(f"  Avg exam initial: {summary['avg_exam_sampled_score_initial']:.4f}")
    print(f"  Avg exam final:   {summary['avg_exam_sampled_score_final']:.4f}")
    print(f"  Avg exam delta:   {summary['avg_exam_sampled_score_delta']:.4f}")
    if not free_gen:
        print(f"  KC match rate:    {summary['avg_kc_match_rate']:.4f}")
        print(f"  Avg oracle V_edu: {summary['avg_oracle_vedu_score']:.4f}")
    print(f"  Parse OK rate:    {summary['avg_parse_ok_rate']:.4f}")
    if compute_nll:
        print(f"  NLL mean/std:     {nll_mean:.4f} / {nll_std:.4f}")

    # Write CSV with per-round generation results
    csv_path = module_dir / "gen_outputs.csv"
    csv_fields = ["student_id", "practice_size", "kc", "question_text", "nll"]
    with open(csv_path, "w", newline="", encoding="utf-8") as csv_f:
        writer = csv.DictWriter(csv_f, fieldnames=csv_fields)
        writer.writeheader()
        writer.writerows(csv_rows)
    print(f"  CSV output: {csv_path} ({len(csv_rows)} rows)")

    return summary


# ------------------------------------------------------------------
# Top-level orchestrator
# ------------------------------------------------------------------

def run_gen_eval(
    *,
    ckpt_path: Optional[str],
    model_path: Optional[str] = None,
    dataset: str,
    root_node: str,
    split: str,
    exam_size: int,
    practice_rounds: int,
    burn_in_step: int,
    max_students: int,
    seed: int,
    output_root: Path,
    shared_exam_root: Path,
    shared_initial_root: Path,
    device: str,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    practice_fixed_difficulty: str,
    practice_response_mode: str,
    exam_fixed_difficulty: str,
    exam_question_info_path: str,
    verifier_ckpt_path: Optional[str],
    free_gen: bool = False,
    ref_model_path: Optional[str] = None,
    compute_nll: bool = False,
) -> Dict[str, Any]:
    cfg = load_config()
    dataset_dir = (ROOT / cfg.KT.dataset_dir / dataset).resolve()
    if not dataset_dir.is_dir():
        raise FileNotFoundError(f"Dataset folder not found: {dataset_dir}")
    exam_q_path = Path(exam_question_info_path)
    if not exam_q_path.is_absolute():
        exam_q_path = (ROOT / exam_q_path).resolve()
    if not exam_q_path.is_file():
        raise FileNotFoundError(f"Exam question_info not found: {exam_q_path}")
    exam_question_info = _load_question_info(exam_q_path)

    medium_only = bool(int(os.getenv("EQG_MEDIUM_ONLY", "1")))

    # Resolve and load generator model
    from models.base_llm import BaseLLM
    if ckpt_path:
        ckpt_load_path = resolve_ckpt_load_path(ckpt_path)
        print(f"Loading generator model from checkpoint: {ckpt_load_path}")
        llm = BaseLLM(model_name=ckpt_load_path, device=device, trainable=False)
    elif model_path:
        print(f"Loading generator model from HF: {model_path}")
        llm = BaseLLM(model_name=model_path, device=device, trainable=False)
        ckpt_load_path = model_path
    else:
        raise ValueError("Either --ckpt-path or --model-path must be provided.")

    # Load reference model for NLL
    ref_model = None
    if compute_nll and ref_model_path:
        print(f"Loading reference model for NLL: {ref_model_path}")
        ref_model = BaseLLM(model_name=ref_model_path, device=device, trainable=False)

    # Load verifier
    # VerifierScorer expects ckpt_path to be a file, not a directory.
    # If the user passes a directory, resolve to the checkpoint file inside it.
    resolved_verifier_ckpt = verifier_ckpt_path
    if resolved_verifier_ckpt and os.path.isdir(resolved_verifier_ckpt):
        candidate = os.path.join(resolved_verifier_ckpt, "simple_verifier.pt")
        if os.path.isfile(candidate):
            resolved_verifier_ckpt = candidate
        else:
            # Try any .pt file in the directory
            pt_files = [f for f in os.listdir(resolved_verifier_ckpt) if f.endswith(".pt")]
            if pt_files:
                resolved_verifier_ckpt = os.path.join(resolved_verifier_ckpt, pt_files[0])
    print(f"Loading verifier (ckpt={resolved_verifier_ckpt}) ...")
    from verifier.inference import VerifierScorer
    verifier = VerifierScorer(
        ckpt_path=resolved_verifier_ckpt,
        device=device,
    )

    rng = random.Random(seed)
    module_key, module_state_dir = ROOT_TO_MODULE.get(root_node), MODULE_STATE_DIRS.get(
        ROOT_TO_MODULE.get(root_node, ""), ""
    )
    if module_key is None:
        raise ValueError(f"Unsupported root_node={root_node}. Expected: {list(ROOT_TO_MODULE.keys())}")

    # Infer model name from ckpt/model path for output dir naming
    p = Path(ckpt_path or model_path).expanduser().resolve()
    if p.name == "huggingface" and p.parent.name == "actor":
        p = p.parent.parent
    elif p.name in {"huggingface", "actor"}:
        p = p.parent
    parts = []
    if p.parent.name:
        parts.append(_safe_name_token(p.parent.name))
    parts.append(_safe_name_token(p.name))
    gen_prefix = "gen_free_" if free_gen else "gen_verify_"
    effective_model_name = gen_prefix + "_".join(parts)
    effective_model_name = effective_model_name[:120]

    effective_exam_size = _resolve_effective_exam_size(
        shared_exam_root=shared_exam_root / dataset / split,
        requested_exam_size=exam_size,
        module_keys=[module_key],
        exam_fixed_difficulty=exam_fixed_difficulty,
    )
    exam_mode_suffix = (
        f"_examfix{exam_fixed_difficulty}" if exam_fixed_difficulty != "none" else ""
    )
    run_tag = f"{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}_p{os.getpid()}"
    run_output_dir = output_root / effective_model_name / (
        f"{dataset}_{split}_k{practice_rounds}_exam{effective_exam_size}"
        f"{exam_mode_suffix}_resp{practice_response_mode}_{run_tag}"
    )
    run_output_dir.mkdir(parents=True, exist_ok=True)
    run_temp_recordings_path = run_output_dir / "runtime_cache" / f"{module_key}_recordings_temp.json"
    previous_temp_recordings_path = current_temp_recordings_path()
    configured_temp_recordings_path = configure_temp_recordings_path(str(run_temp_recordings_path))
    shared_exam_root = shared_exam_root / dataset / split
    shared_initial_root = shared_initial_root / dataset / split

    manifest = {
        "pipeline": "free_gen_verify" if free_gen else "oracle_gen_verify",
        "free_gen": free_gen,
        "compute_nll": compute_nll,
        "model_name": effective_model_name,
        "ckpt_path": ckpt_path,
        "model_path": model_path,
        "ckpt_load_path": ckpt_load_path,
        "ref_model_path": ref_model_path if compute_nll else None,
        "dataset": dataset,
        "root_node": root_node,
        "module_key": module_key,
        "split": split,
        "exam_size": exam_size,
        "effective_exam_size_for_naming": effective_exam_size,
        "exam_fixed_difficulty": exam_fixed_difficulty,
        "practice_rounds": practice_rounds,
        "burn_in_step": burn_in_step,
        "max_students": max_students,
        "seed": seed,
        "device": device,
        "max_new_tokens": max_new_tokens,
        "temperature": temperature,
        "top_p": top_p,
        "medium_only": medium_only,
        "practice_fixed_difficulty": practice_fixed_difficulty,
        "practice_response_mode": practice_response_mode,
        "verifier_ckpt_path": verifier_ckpt_path,
        "exam_question_info_path": str(exam_q_path),
        "temp_recordings_path": str(configured_temp_recordings_path),
        "runtime_temp_root": str(run_output_dir / "_runtime_cache"),
        "run_output_dir": str(run_output_dir),
        "shared_exam_root": str(shared_exam_root),
        "shared_initial_root": str(shared_initial_root),
    }
    try:
        with open(run_output_dir / "run_config.json", "w", encoding="utf-8") as f:
            json.dump(manifest, f, ensure_ascii=False, indent=2)

        module_summaries: Dict[str, Any] = {}
        print(f"Running module={module_key} (root_node={root_node})")
        module_summaries[module_key] = _run_for_module(
            module_key=module_key,
            module_state_dir=module_state_dir,
            dataset_dir=dataset_dir,
            split=split,
            burn_in_step=burn_in_step,
            exam_size=exam_size,
            practice_rounds=practice_rounds,
            max_students=max_students,
            rng=rng,
            run_output_dir=run_output_dir,
            shared_exam_root=shared_exam_root,
            shared_initial_root=shared_initial_root,
            seed=seed,
            practice_fixed_difficulty=practice_fixed_difficulty,
            practice_response_mode=practice_response_mode,
            exam_fixed_difficulty=exam_fixed_difficulty,
            exam_question_info=exam_question_info,
            llm=llm,
            verifier=verifier,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            medium_only=medium_only,
            free_gen=free_gen,
            ref_model=ref_model,
            compute_nll=compute_nll,
        )

        global_summary = {
            "manifest": manifest,
            "module_summaries": module_summaries,
        }
        with open(run_output_dir / "summary.json", "w", encoding="utf-8") as f:
            json.dump(global_summary, f, ensure_ascii=False, indent=2)

        print(f"\nDone. Outputs at: {run_output_dir}")
        return global_summary
    finally:
        configure_temp_recordings_path(previous_temp_recordings_path)


# ------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Oracle-Gen-Verify practice evaluation pipeline",
    )
    parser.add_argument("--ckpt-path", default=None, help="Path to generator checkpoint (local)")
    parser.add_argument("--model-path", default=None, help="HuggingFace model name (alternative to --ckpt-path)")
    parser.add_argument("--dataset", default="", help="Defaults to config.KT.dataset")
    parser.add_argument("--root-node", default="", help="Defaults to config.KT.root_node")
    parser.add_argument("--split", default="test", choices=["train", "test"])
    parser.add_argument("--exam-size", type=int, default=30)
    parser.add_argument(
        "--exam-fixed-difficulty", default="none",
        choices=["none", "easy", "medium", "hard"],
    )
    parser.add_argument("--practice-rounds", type=int, default=10)
    parser.add_argument("--burn-in-step", type=int, default=10)
    parser.add_argument("--max-students", type=int, default=-1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument(
        "--practice-fixed-difficulty", default="medium",
        choices=["easy", "medium", "hard"],
    )
    parser.add_argument(
        "--practice-response-mode", default="sampled",
        choices=["sampled", "always_correct"],
    )
    parser.add_argument(
        "--output-root",
        default=str(ROOT / "output" / "exam_eval" / "Eval-Result"),
    )
    parser.add_argument(
        "--shared-exam-root",
        default=str(ROOT / "output" / "exam_eval" / "Eval-Shared" / "shared_exam_sets"),
    )
    parser.add_argument(
        "--shared-initial-root",
        default=str(ROOT / "output" / "exam_eval" / "Eval-Shared" / "shared_initial_evals"),
    )
    parser.add_argument("--exam-question-info-path", default="")
    parser.add_argument("--verifier-ckpt-path", default=None, help="Explicit verifier checkpoint path")
    parser.add_argument("--model-name", default="", help="Override model name in output dir")
    parser.add_argument("--free-gen", action="store_true", default=False,
                        help="Free generation mode: skip oracle KC selection, no forced prefix")
    parser.add_argument("--ref-model-path", default="Qwen/Qwen3-8B",
                        help="Reference model for NLL computation")
    parser.add_argument("--compute-nll", type=int, default=1,
                        help="Compute NLL against reference model (1=on, 0=off)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config()
    dataset = args.dataset or cfg.KT.dataset
    root_node = args.root_node or cfg.KT.root_node
    if args.exam_question_info_path:
        exam_question_info_path = args.exam_question_info_path
    else:
        dataset_root = (ROOT / cfg.KT.dataset_dir / dataset).resolve()
        true_q = dataset_root / "true_question_info.json"
        default_q = dataset_root / "question_info.json"
        exam_question_info_path = str(true_q if true_q.is_file() else default_q)
    run_gen_eval(
        ckpt_path=args.ckpt_path,
        model_path=args.model_path,
        dataset=dataset,
        root_node=root_node,
        split=args.split,
        exam_size=args.exam_size,
        practice_rounds=args.practice_rounds,
        burn_in_step=args.burn_in_step,
        max_students=args.max_students,
        seed=args.seed,
        output_root=Path(args.output_root).resolve(),
        shared_exam_root=Path(args.shared_exam_root).resolve(),
        shared_initial_root=Path(args.shared_initial_root).resolve(),
        device=args.device,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        practice_fixed_difficulty=args.practice_fixed_difficulty,
        practice_response_mode=args.practice_response_mode,
        exam_fixed_difficulty=args.exam_fixed_difficulty,
        exam_question_info_path=exam_question_info_path,
        verifier_ckpt_path=args.verifier_ckpt_path,
        free_gen=args.free_gen,
        ref_model_path=args.ref_model_path,
        compute_nll=bool(args.compute_nll),
    )


if __name__ == "__main__":
    main()
