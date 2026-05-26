"""KC-recommender evaluation (no LLM generation).

Benchmarks different practice-round KC selection strategies against the same exam set.
Question text is a fixed placeholder, so results isolate the effect of which KC is
picked, independent of question-generation quality.

Supported `--recommender-type`:
  * `random`           — uniform over leaf KCs
  * `qwen`             — off-the-shelf HF LLM picks a KC from the candidate list
  * `ckpt`             — local LLM checkpoint picks a KC (KC-only mode)
  * `oracle_vedu`      — oracle scan by V_edu
  * `lowest_posterior` — pick KC with lowest current mastery posterior

Also exports shared helpers (QuestionSpec, exam-set cache, initial-eval cache) that
are reused by the main generation-based pipeline in gen_practice_eval.py.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import statistics
import sys
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from config.config import load_config
from eval.recommenders.ckpt_llm import CkptLLMKCRecommender, resolve_ckpt_load_path
from eval.recommenders.lowest_posterior import LowestPosteriorRecommender
from eval.recommenders.oracle_vedu import OracleVEduRecommender
from eval.recommenders.untrained_llm import UntrainedLLMKCRecommender
from kt.predict import predict_on_new_question
from kt.runtime import KTRuntime
from kt.update import (
    one_step_update_single,
    configure_temp_recordings_path,
    current_temp_recordings_path,
)

DIFFICULTIES = ["easy", "medium", "hard"]
# module_key == root_node; kept as a map for the validation path (unknown root_node -> error).
ROOT_TO_MODULE = {
    # XES3G5M
    "Application_Module": "Application_Module",
    "Computation_Module": "Computation_Module",
    "Counting_Module": "Counting_Module",
    # MOOCRadar
    "Wine_Knowledge": "Wine_Knowledge",
    "Circuit_Design": "Circuit_Design",
    "Education_Theory": "Education_Theory",
}
# Maps module_key -> on-disk state subdirectory under data/dataset/<dataset>/.
MODULE_STATE_DIRS = {
    "Application_Module": "application_states",
    "Computation_Module": "computation_states",
    "Counting_Module": "counting_states",
    "Wine_Knowledge": "wine_states",
    "Circuit_Design": "circuit_states",
    "Education_Theory": "education_states",
}


@dataclass
class QuestionSpec:
    question_id: str
    module: str
    kc: str
    difficulty: str
    question_text: str


@dataclass
class RoundTrace:
    round_id: int
    question_id: str
    kc: str
    difficulty: str
    p_correct: float | None
    sampled_correct: int
    updated_student_graph_path: str | None
    updated_parameter_graph_path: str | None
    recommender: str | None = None
    recommender_raw: str | None = None
    recommender_meta: Dict[str, Any] | None = None


def _leaf_kcs(student_graph: Dict[str, Any]) -> List[str]:
    leaves: List[str] = []
    for name, node in student_graph.items():
        children = getattr(node, "children", None)
        if children is None and isinstance(node, dict):
            children = node.get("children", [])
        if children is not None and len(children) == 0:
            leaves.append(name)
    if not leaves:
        raise ValueError("No leaf KC found in student graph.")
    leaves.sort()
    return leaves


def _build_question_pool(module_name: str, leaf_kcs: List[str]) -> List[QuestionSpec]:
    pool: List[QuestionSpec] = []
    qidx = 1
    for kc in leaf_kcs:
        for diff in DIFFICULTIES:
            qid = f"EVAL_{module_name}_{qidx:05d}_{diff[0].upper()}"
            qtxt = f"[EVAL_PLACEHOLDER] module={module_name}, kc={kc}, difficulty={diff}"
            pool.append(
                QuestionSpec(
                    question_id=qid,
                    module=module_name,
                    kc=kc,
                    difficulty=diff,
                    question_text=qtxt,
                )
            )
            qidx += 1
    return pool


def _build_exam_combo_pool(
    module_name: str,
    leaf_kcs: List[str],
    exam_fixed_difficulty: str,
) -> List[QuestionSpec]:
    full_pool = _build_question_pool(module_name, leaf_kcs)
    if exam_fixed_difficulty == "none":
        return full_pool
    return [q for q in full_pool if q.difficulty == exam_fixed_difficulty]


def _sample_exam_set(rng: random.Random, pool: List[QuestionSpec], exam_size: int) -> List[QuestionSpec]:
    if exam_size <= 0:
        raise ValueError("exam_size must be > 0")
    if exam_size >= len(pool):
        return list(pool)
    return rng.sample(pool, exam_size)


def _load_question_info(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _build_real_exam_index(
    module_key: str,
    leaf_kcs: List[str],
    question_info: Dict[str, Any],
) -> Dict[tuple, List[QuestionSpec]]:
    leaf_set = set(leaf_kcs)
    index: Dict[tuple, List[QuestionSpec]] = {}
    for qid, q in question_info.items():
        kc = q.get("kc")
        diff = str(q.get("difficulty", "")).lower()
        if kc not in leaf_set or diff not in DIFFICULTIES:
            continue
        key = (kc, diff)
        index.setdefault(key, []).append(
            QuestionSpec(
                question_id=str(qid),
                module=module_key,
                kc=kc,
                difficulty=diff,
                question_text=str(q.get("content", "")),
            )
        )
    return index


def _sample_real_exam_set(
    rng: random.Random,
    combo_pool: List[QuestionSpec],
    real_exam_index: Dict[tuple, List[QuestionSpec]],
    exam_size: int,
) -> List[QuestionSpec]:
    # Fallback index: kc -> all questions with this kc (ignore difficulty).
    kc_only_index: Dict[str, List[QuestionSpec]] = {}
    for (kc, _), items in real_exam_index.items():
        kc_only_index.setdefault(kc, []).extend(items)

    # Pre-filter combo_pool to only KCs that have real questions available.
    available_pool = [
        c for c in combo_pool
        if real_exam_index.get((c.kc, c.difficulty)) or kc_only_index.get(c.kc)
    ]
    skipped_kcs = {c.kc for c in combo_pool} - {c.kc for c in available_pool}
    if skipped_kcs:
        print(
            f"[warning] {len(skipped_kcs)} KC(s) have no real questions and were excluded "
            f"from exam sampling: {sorted(skipped_kcs)[:10]}"
            + (f" ... and {len(skipped_kcs) - 10} more" if len(skipped_kcs) > 10 else "")
        )
    if len(available_pool) == 0:
        raise ValueError(
            "No KCs in the combo pool have matching real questions. "
            "Please verify exam question_info source."
        )
    if len(available_pool) < exam_size:
        print(
            f"[warning] available (kc, difficulty) combos ({len(available_pool)}) < "
            f"exam_size ({exam_size}); sampling with replacement to reach target."
        )

    # Sample from the filtered pool, with replacement if needed.
    sampled_combos = _sample_exam_set(rng, available_pool, exam_size)
    if len(sampled_combos) < exam_size:
        extra = rng.choices(available_pool, k=exam_size - len(sampled_combos))
        sampled_combos.extend(extra)

    out: List[QuestionSpec] = []
    relaxed_count = 0
    for combo in sampled_combos:
        key = (combo.kc, combo.difficulty)
        cands = real_exam_index.get(key, [])
        if not cands:
            cands = kc_only_index.get(combo.kc, [])
            relaxed_count += 1
        out.append(rng.choice(cands))
    if relaxed_count > 0:
        print(f"[info] exam sampling used kc-only fallback for {relaxed_count}/{exam_size} items.")
    return out


def _stable_hash(items: List[str]) -> str:
    joined = "||".join(items).encode("utf-8")
    return hashlib.sha1(joined).hexdigest()[:12]


def _load_exam_set(path: Path) -> List[QuestionSpec]:
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    return [QuestionSpec(**item) for item in raw]


def _shared_exam_path(
    shared_exam_root: Path,
    module_key: str,
    exam_size: int,
    exam_fixed_difficulty: str,
) -> Path:
    if exam_fixed_difficulty == "none":
        return shared_exam_root / module_key / f"exam_set_size_{exam_size}.json"
    return shared_exam_root / module_key / f"exam_set_size_{exam_size}_fix_{exam_fixed_difficulty}.json"


def _resolve_effective_exam_size(
    *,
    shared_exam_root: Path,
    requested_exam_size: int,
    module_keys: List[str],
    exam_fixed_difficulty: str,
) -> int:
    sizes: List[int] = []
    for module_key in module_keys:
        candidate = _shared_exam_path(
            shared_exam_root,
            module_key,
            requested_exam_size,
            exam_fixed_difficulty,
        )
        legacy_candidate = shared_exam_root / module_key / "exam_set.json"
        if candidate.is_file():
            with open(candidate, "r", encoding="utf-8") as f:
                raw = json.load(f)
            sizes.append(len(raw))
        elif exam_fixed_difficulty == "none" and legacy_candidate.is_file():
            with open(legacy_candidate, "r", encoding="utf-8") as f:
                raw = json.load(f)
            if len(raw) == requested_exam_size:
                sizes.append(len(raw))

    if not sizes:
        return requested_exam_size

    first_size = sizes[0]
    if any(s != first_size for s in sizes):
        print(
            f"[warning] inconsistent shared exam sizes across modules: {sizes}; "
            f"using first size={first_size} for run folder naming."
        )
    return first_size


def _get_or_create_shared_exam_set(
    *,
    shared_exam_path: Path,
    rng: random.Random,
    combo_pool: List[QuestionSpec],
    real_exam_index: Dict[tuple, List[QuestionSpec]],
    exam_size: int,
) -> List[QuestionSpec]:
    if shared_exam_path.is_file():
        print(f"Reuse existing exam set: {shared_exam_path}")
        exam_set = _load_exam_set(shared_exam_path)
        # Legacy placeholder exam sets (EVAL_*) should be replaced by real-question exam sets.
        has_placeholder = any(item.question_id.startswith("EVAL_") for item in exam_set)
        if has_placeholder:
            print(f"[info] Found placeholder exam set at {shared_exam_path}; rebuilding from real question_info.")
        else:
            if len(exam_set) != exam_size:
                print(
                    f"[warning] shared exam size={len(exam_set)} differs from requested exam_size={exam_size}; "
                    "using existing shared exam set."
                )
            return exam_set

    # Backward compatibility: legacy path without size in filename.
    legacy_path = shared_exam_path.parent / "exam_set.json"
    if legacy_path.is_file():
        legacy_exam = _load_exam_set(legacy_path)
        has_placeholder = any(item.question_id.startswith("EVAL_") for item in legacy_exam)
        if len(legacy_exam) == exam_size and not has_placeholder:
            print(f"Reuse legacy exam set with matching size: {legacy_path}")
            shared_exam_path.parent.mkdir(parents=True, exist_ok=True)
            with open(shared_exam_path, "w", encoding="utf-8") as f:
                json.dump([asdict(x) for x in legacy_exam], f, ensure_ascii=False, indent=2)
            return legacy_exam

    shared_exam_path.parent.mkdir(parents=True, exist_ok=True)
    exam_set = _sample_real_exam_set(
        rng=rng,
        combo_pool=combo_pool,
        real_exam_index=real_exam_index,
        exam_size=exam_size,
    )
    with open(shared_exam_path, "w", encoding="utf-8") as f:
        json.dump([asdict(x) for x in exam_set], f, ensure_ascii=False, indent=2)
    print(f"Created new exam set: {shared_exam_path}")
    return exam_set


def _practice_question_fixed_difficulty(
    rng: random.Random,
    medium_kc_candidates: List[str],
    practice_lookup: Dict[tuple, QuestionSpec],
    fixed_difficulty: str,
) -> QuestionSpec:
    kc = rng.choice(medium_kc_candidates)
    return practice_lookup[(kc, fixed_difficulty)]


def _initial_cache_path(
    *,
    shared_initial_root: Path,
    module_key: str,
    burn_in_step: int,
    exam_set: List[QuestionSpec],
    sids: List[str],
) -> Path:
    exam_sig = _stable_hash([q.question_id for q in exam_set])
    stu_sig = _stable_hash(sids)
    exam_size = len(exam_set)
    fname = f"initial_burn{burn_in_step}_{exam_sig}_stu{len(sids)}_{stu_sig}.json"
    return shared_initial_root / module_key / f"exam_size_{exam_size}" / fname


def _get_or_create_initial_eval_cache(
    *,
    path: Path,
    rt: KTRuntime,
    sids: List[str],
    burn_in_step: int,
    exam_set: List[QuestionSpec],
    seed: int,
    module_key: str,
) -> Dict[str, Dict[str, Any]]:
    if path.is_file():
        print(f"Reuse existing initial-eval cache: {path}")
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        by_student = payload.get("by_student", {})
        return {str(k): v for k, v in by_student.items()}

    path.parent.mkdir(parents=True, exist_ok=True)
    module_seed_offset = int(_stable_hash([module_key]), 16) % 100000
    initial_rng = random.Random(seed + module_seed_offset)
    by_student: Dict[str, Dict[str, Any]] = {}

    for sid in sids:
        rt.clear_actual_temp_graphs()
        rt.set_practice_size(sid, burn_in_step)
        rt.load_student_graph(sid, practice_size=burn_in_step, refresh=True)

        initial_exam_records: List[Dict[str, Any]] = []
        for q in exam_set:
            pred_q = {"kc": q.kc, "difficulty": q.difficulty, "question_text": q.question_text}
            p0 = float(predict_on_new_question(rt, sid, pred_q))
            initial_exam_records.append(
                {
                    "question_id": q.question_id,
                    "kc": q.kc,
                    "difficulty": q.difficulty,
                    "p_correct": p0,
                }
            )
        mean_exam_p_initial = statistics.fmean(x["p_correct"] for x in initial_exam_records)
        by_student[sid] = {
            "mean_exam_p_correct_initial": mean_exam_p_initial,
            # Backward-compatible field name; now identical to expected score.
            "mean_exam_sampled_correct_initial": mean_exam_p_initial,
            "exam_predictions_initial": initial_exam_records,
        }

    payload = {
        "module": module_key,
        "burn_in_step": burn_in_step,
        "exam_question_ids": [q.question_id for q in exam_set],
        "student_ids": sids,
        "by_student": by_student,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f"Created initial-eval cache: {path}")
    return by_student


def _student_ids(rt: KTRuntime, max_students: int) -> List[str]:
    sids = sorted(list(rt.iter_student_ids()))
    if max_students > 0:
        sids = sids[:max_students]
    return sids


def _resolve_module_from_root(root_node: str) -> tuple[str, str]:
    module_key = ROOT_TO_MODULE.get(root_node)
    if module_key is None:
        raise ValueError(
            f"Unsupported root_node={root_node}. Expected one of: {list(ROOT_TO_MODULE.keys())}"
        )
    state_dir = MODULE_STATE_DIRS[module_key]
    return module_key, state_dir


def _safe_name_token(text: str) -> str:
    out = "".join(ch if (ch.isalnum() or ch in {"_", "-"}) else "_" for ch in text.strip())
    out = out.strip("_")
    return out or "unknown"


def _infer_model_name(
    *,
    requested_model_name: str,
    recommender_type: str,
    recommender_ckpt_path: str,
) -> str:
    if requested_model_name and requested_model_name != "random_baseline":
        return requested_model_name
    if recommender_type != "ckpt":
        return requested_model_name or "random_baseline"
    if not recommender_ckpt_path:
        return "ckpt_recommender"

    p = Path(recommender_ckpt_path).expanduser().resolve()
    # Normalize common HF-export subpaths back to the real checkpoint folder.
    if p.name == "huggingface" and p.parent.name == "actor":
        p = p.parent.parent
    elif p.name in {"huggingface", "actor"}:
        p = p.parent

    parts: List[str] = []
    if p.parent.name:
        parts.append(_safe_name_token(p.parent.name))
    parts.append(_safe_name_token(p.name))
    inferred = "ckpt_" + "_".join(parts)
    return inferred[:120]


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
    recommender_type: str,
    recommender_model_name: str,
    recommender_ckpt_path: str,
    recommender_device: str,
    llm_decode_kwargs: Dict[str, Any],
    seed: int,
    practice_fixed_difficulty: str,
    practice_response_mode: str,
    exam_fixed_difficulty: str,
    exam_question_info: Dict[str, Any],
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
        shared_exam_root,
        module_key,
        exam_size,
        exam_fixed_difficulty,
    )
    exam_set = _get_or_create_shared_exam_set(
        shared_exam_path=shared_exam_path,
        rng=rng,
        combo_pool=exam_combo_pool,
        real_exam_index=real_exam_index,
        exam_size=exam_size,
    )
    # Keep practice candidates independent from exam sampling. In question generation
    # settings, same (kc, difficulty) can still correspond to different generated items.
    practice_pool = list(full_pool)
    practice_lookup = {(q.kc, q.difficulty): q for q in practice_pool}
    medium_kc_candidates = [
        kc for kc in leaf_kcs if (kc, practice_fixed_difficulty) in practice_lookup
    ]
    if not medium_kc_candidates:
        raise ValueError(
            f"No practice questions with fixed difficulty={practice_fixed_difficulty} "
            f"for module={module_key}. Consider reducing exam_size."
        )
    recommender = None
    if recommender_type == "qwen":
        recommender = UntrainedLLMKCRecommender(
            model_name=recommender_model_name,
            device=recommender_device,
        )
    elif recommender_type == "ckpt":
        recommender = CkptLLMKCRecommender(
            ckpt_path=recommender_ckpt_path,
            device=recommender_device,
        )
    elif recommender_type == "oracle_vedu":
        recommender = OracleVEduRecommender()
    elif recommender_type == "lowest_posterior":
        recommender = LowestPosteriorRecommender()

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

    module_student_results: List[Dict[str, Any]] = []
    avg_exam_scores_initial: List[float] = []
    avg_exam_scores_final: List[float] = []
    coverage_ratios: List[float] = []
    coverage_counts: List[int] = []
    oracle_elapsed_values: List[float] = []
    oracle_scan_counts: List[int] = []

    with open(trajectory_path, "w", encoding="utf-8") as traj_f, open(
        exam_path, "w", encoding="utf-8"
    ) as exam_f:
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

            traces: List[RoundTrace] = []
            for rid in range(1, practice_rounds + 1):
                student_graph_now = rt.load_student_graph(sid, refresh=True)
                rec_raw = None
                rec_meta = None
                if recommender_type in {"qwen", "ckpt"}:
                    assert recommender is not None
                    q, rec_raw = recommender.choose_question(
                        student_graph=student_graph_now,
                        leaf_kcs=leaf_kcs,
                        practice_lookup=practice_lookup,
                        medium_kc_candidates=medium_kc_candidates,
                        fixed_difficulty=practice_fixed_difficulty,
                        decode_kwargs=llm_decode_kwargs,
                        rng=rng,
                    )
                elif recommender_type == "oracle_vedu":
                    assert recommender is not None
                    q, rec_meta = recommender.choose_question(
                        rt=rt,
                        student_id=sid,
                        practice_lookup=practice_lookup,
                        kc_candidates=medium_kc_candidates,
                        fixed_difficulty=practice_fixed_difficulty,
                    )
                    if rec_meta:
                        elapsed = rec_meta.get("oracle_elapsed_sec")
                        scanned = rec_meta.get("oracle_num_kc_scanned")
                        if isinstance(elapsed, (int, float)):
                            oracle_elapsed_values.append(float(elapsed))
                        if isinstance(scanned, int):
                            oracle_scan_counts.append(scanned)
                elif recommender_type == "lowest_posterior":
                    assert recommender is not None
                    q, rec_meta = recommender.choose_question(
                        student_graph=student_graph_now,
                        practice_lookup=practice_lookup,
                        kc_candidates=medium_kc_candidates,
                        fixed_difficulty=practice_fixed_difficulty,
                    )
                else:
                    q = _practice_question_fixed_difficulty(
                        rng=rng,
                        medium_kc_candidates=medium_kc_candidates,
                        practice_lookup=practice_lookup,
                        fixed_difficulty=practice_fixed_difficulty,
                    )
                if practice_response_mode == "always_correct":
                    p = None
                    sampled = 1
                else:
                    pred_q = {"kc": q.kc, "difficulty": q.difficulty, "question_text": q.question_text}
                    p = float(predict_on_new_question(rt, sid, pred_q))
                    sampled = 1 if rng.random() < p else 0

                obs = {
                    "question": q.question_id,
                    "kc": q.kc,
                    "difficulty": q.difficulty,
                    "response": sampled,
                    "source": "recommender_eval",
                }
                update_info = one_step_update_single(
                    rt,
                    sid,
                    [obs],
                    mode="actual",
                    tag=f"eval_round_{rid}",
                    reset_temp=False,
                )

                traces.append(
                    RoundTrace(
                        round_id=rid,
                        question_id=q.question_id,
                        kc=q.kc,
                        difficulty=q.difficulty,
                        p_correct=p,
                        sampled_correct=sampled,
                        updated_student_graph_path=(update_info or {}).get("student_graph"),
                        updated_parameter_graph_path=(update_info or {}).get("parameter_graph"),
                        recommender=recommender_type,
                        recommender_raw=rec_raw,
                        recommender_meta=rec_meta,
                    )
                )

            # Fixed-state exam evaluation: no update during scoring.
            exam_records: List[Dict[str, Any]] = []
            for q in exam_set:
                pred_q = {"kc": q.kc, "difficulty": q.difficulty, "question_text": q.question_text}
                p = float(predict_on_new_question(rt, sid, pred_q))
                exam_records.append(
                    {
                        "question_id": q.question_id,
                        "kc": q.kc,
                        "difficulty": q.difficulty,
                        "p_correct": p,
                    }
                )

            mean_exam_p = statistics.fmean(x["p_correct"] for x in exam_records)
            # Backward-compatible field name; now identical to expected score.
            mean_exam_sampled = mean_exam_p
            avg_exam_scores_initial.append(mean_exam_sampled_initial)
            avg_exam_scores_final.append(mean_exam_sampled)
            practiced_kcs = {t.kc for t in traces if t.kc}
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
                "practice_trace": [asdict(t) for t in traces],
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

            module_student_results.append(
                {
                    "student_id": sid,
                    "mean_exam_p_correct_initial": mean_exam_p_initial,
                    "mean_exam_sampled_correct_initial": mean_exam_sampled_initial,
                    "mean_exam_p_correct": mean_exam_p,
                    "mean_exam_sampled_correct": mean_exam_sampled,
                    "mean_exam_p_correct_delta": mean_exam_p - mean_exam_p_initial,
                    "mean_exam_sampled_correct_delta": mean_exam_sampled - mean_exam_sampled_initial,
                    "practice_unique_kc_count": coverage_count,
                    "practice_kc_coverage_ratio": coverage_ratio,
                    "final_step": final_step,
                }
            )

            if idx % 20 == 0 or idx == len(sids):
                print(f"[{module_key}] processed {idx}/{len(sids)} students")

    if recommender_type == "oracle_vedu" and oracle_elapsed_values:
        sorted_elapsed = sorted(oracle_elapsed_values)
        p50 = sorted_elapsed[len(sorted_elapsed) // 2]
        p95 = sorted_elapsed[min(len(sorted_elapsed) - 1, int(len(sorted_elapsed) * 0.95))]
        avg_elapsed = statistics.fmean(sorted_elapsed)
        total_elapsed = sum(sorted_elapsed)
        avg_scanned = statistics.fmean(oracle_scan_counts) if oracle_scan_counts else 0.0
        print(
            f"[oracle_vedu][{module_key}] rounds={len(sorted_elapsed)} "
            f"avg_scan_kc={avg_scanned:.2f} avg_sec={avg_elapsed:.4f} "
            f"p50_sec={p50:.4f} p95_sec={p95:.4f} total_sec={total_elapsed:.2f}"
        )

    summary = {
        "module": module_key,
        "num_students": len(sids),
        "num_leaf_kcs": len(leaf_kcs),
        "question_pool_size": len(full_pool),
        "exam_set_size": len(exam_set),
        "exam_fixed_difficulty": exam_fixed_difficulty,
        "practice_pool_size": len(practice_pool),
        "practice_rounds": practice_rounds,
        "practice_fixed_difficulty": practice_fixed_difficulty,
        "practice_response_mode": practice_response_mode,
        "recommender_type": recommender_type,
        "initial_eval_cache_file": str(initial_cache_file),
        "avg_practice_unique_kc_count": statistics.fmean(coverage_counts) if coverage_counts else 0.0,
        "avg_practice_kc_coverage_ratio": statistics.fmean(coverage_ratios) if coverage_ratios else 0.0,
        "avg_exam_sampled_score_initial": statistics.fmean(avg_exam_scores_initial)
        if avg_exam_scores_initial
        else 0.0,
        "avg_exam_sampled_score_final": statistics.fmean(avg_exam_scores_final)
        if avg_exam_scores_final
        else 0.0,
        "avg_exam_sampled_score_delta": (
            statistics.fmean(avg_exam_scores_final) - statistics.fmean(avg_exam_scores_initial)
        )
        if avg_exam_scores_initial and avg_exam_scores_final
        else 0.0,
        "students": module_student_results,
    }
    with open(module_dir / "module_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    return summary


def run_recommender_eval(
    *,
    model_name: str,
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
    recommender_type: str,
    recommender_model_name: str,
    recommender_ckpt_path: str,
    recommender_device: str,
    llm_decode_kwargs: Dict[str, Any],
    practice_fixed_difficulty: str,
    practice_response_mode: str,
    exam_fixed_difficulty: str,
    exam_question_info_path: str,
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
    if recommender_type == "ckpt" and not recommender_ckpt_path.strip():
        raise ValueError("recommender_type=ckpt requires --recommender-ckpt-path")

    rng = random.Random(seed)
    if recommender_type == "qwen":
        active_model_name = recommender_model_name or "Qwen/Qwen2.5-1.5B-Instruct"
        active_ckpt_load_path = ""
    elif recommender_type == "ckpt":
        active_model_name = None
        active_ckpt_load_path = resolve_ckpt_load_path(recommender_ckpt_path)
    else:
        active_model_name = None
        active_ckpt_load_path = ""
    module_key, module_state_dir = _resolve_module_from_root(root_node)
    effective_model_name = _infer_model_name(
        requested_model_name=model_name,
        recommender_type=recommender_type,
        recommender_ckpt_path=recommender_ckpt_path,
    )

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
        "model_name": effective_model_name,
        "model_name_input": model_name,
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
        "recommender_type": recommender_type,
        "recommender_model_name": active_model_name,
        "recommender_ckpt_path": recommender_ckpt_path or None,
        "recommender_ckpt_load_path": active_ckpt_load_path or None,
        "recommender_device": recommender_device,
        "llm_decode_kwargs": llm_decode_kwargs,
        "practice_fixed_difficulty": practice_fixed_difficulty,
        "practice_response_mode": practice_response_mode,
        "exam_scoring_mode": "expected_only",
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
            recommender_type=recommender_type,
            recommender_model_name=active_model_name or "",
            recommender_ckpt_path=active_ckpt_load_path or "",
            recommender_device=recommender_device,
            llm_decode_kwargs=llm_decode_kwargs,
            seed=seed,
            practice_fixed_difficulty=practice_fixed_difficulty,
            practice_response_mode=practice_response_mode,
            exam_fixed_difficulty=exam_fixed_difficulty,
            exam_question_info=exam_question_info,
        )

        global_summary = {
            "manifest": manifest,
            "module_summaries": module_summaries,
        }
        with open(run_output_dir / "summary.json", "w", encoding="utf-8") as f:
            json.dump(global_summary, f, ensure_ascii=False, indent=2)

        print(f"Done. Outputs at: {run_output_dir}")
        return global_summary
    finally:
        configure_temp_recordings_path(previous_temp_recordings_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="KC-recommender evaluation: compare practice-round KC selection strategies (random / LLM / oracle / lowest-posterior) without LLM generation."
    )
    parser.add_argument("--model-name", default="random_baseline")
    parser.add_argument("--dataset", default="", help="Defaults to config.KT.dataset when empty")
    parser.add_argument("--root-node", default="", help="Defaults to config.KT.root_node when empty")
    parser.add_argument("--split", default="train", choices=["train", "test"])
    parser.add_argument("--exam-size", type=int, default=30)
    parser.add_argument(
        "--exam-fixed-difficulty",
        default="none",
        choices=["none", "easy", "medium", "hard"],
        help="If set, exam set samples KC only and fixes difficulty to this value.",
    )
    parser.add_argument("--practice-rounds", type=int, default=10)
    parser.add_argument("--burn-in-step", type=int, default=10)
    parser.add_argument("--max-students", type=int, default=-1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--recommender-type",
        default="random",
        choices=["random", "qwen", "ckpt", "oracle_vedu", "lowest_posterior"],
        help="How to pick next practice question",
    )
    parser.add_argument(
        "--recommender-model-name",
        default="",
        help="HuggingFace model name for qwen recommender",
    )
    parser.add_argument(
        "--recommender-ckpt-path",
        default="",
        help="Local checkpoint folder for ckpt recommender (supports actor/huggingface).",
    )
    parser.add_argument(
        "--recommender-device",
        default="cuda",
        help="Device for qwen/ckpt recommenders, e.g., cuda or cpu",
    )
    parser.add_argument("--llm-temperature", type=float, default=0.7)
    parser.add_argument("--llm-top-p", type=float, default=0.9)
    parser.add_argument("--llm-max-new-tokens", type=int, default=128)
    parser.add_argument(
        "--practice-fixed-difficulty",
        default="medium",
        choices=["easy", "medium", "hard"],
        help="Practice recommendation fixes difficulty and only recommends KC",
    )
    parser.add_argument(
        "--practice-response-mode",
        default="sampled",
        choices=["sampled", "always_correct"],
        help="How to assign correctness during practice updates",
    )
    parser.add_argument(
        "--output-root",
        default=str(ROOT / "output" / "exam_eval" / "Eval-Result"),
        help="Root folder for eval outputs",
    )
    parser.add_argument(
        "--shared-exam-root",
        default=str(ROOT / "output" / "exam_eval" / "Eval-Shared" / "shared_exam_sets"),
        help="Global shared exam-set folder (reused across baselines)",
    )
    parser.add_argument(
        "--shared-initial-root",
        default=str(ROOT / "output" / "exam_eval" / "Eval-Shared" / "shared_initial_evals"),
        help="Global shared initial-eval cache folder (reused across baselines)",
    )
    parser.add_argument(
        "--exam-question-info-path",
        default="",
        help="Question info used to build exam set (must contain real content/kc/difficulty).",
    )
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
    llm_decode_kwargs = {
        "temperature": args.llm_temperature,
        "top_p": args.llm_top_p,
        "max_new_tokens": args.llm_max_new_tokens,
    }
    run_recommender_eval(
        model_name=args.model_name,
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
        recommender_type=args.recommender_type,
        recommender_model_name=args.recommender_model_name,
        recommender_ckpt_path=args.recommender_ckpt_path,
        recommender_device=args.recommender_device,
        llm_decode_kwargs=llm_decode_kwargs,
        practice_fixed_difficulty=args.practice_fixed_difficulty,
        practice_response_mode=args.practice_response_mode,
        exam_fixed_difficulty=args.exam_fixed_difficulty,
        exam_question_info_path=exam_question_info_path,
    )


if __name__ == "__main__":
    main()
