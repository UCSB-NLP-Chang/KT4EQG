from __future__ import annotations

import json
import os
import sys
from collections import defaultdict
from copy import deepcopy
import re
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from kt.runtime import KTRuntime
from kt.KT2.KT.graph_update import update_e_step
from kt.KT2.EM.M_Step import m_step
from kt.KT2.module.calibration import calibrate_r_diff_closed_form
from kt.KT2.module.data_cache import (
    initialize_train_uids,
    initialize_data,
    load_data,
    load_full_recordings,
    get_emission_dict,
    get_train_uids,
    append_temp_records,
    reset_temp_recordings,
    clear_cached_data,
    set_temp_recordings_path as _set_temp_recordings_path,
    get_temp_recordings_path as _get_temp_recordings_path,
)
from config.config import load_config
cfg = load_config()
graph_dir = cfg.KT.graph_dir
dataset = cfg.KT.dataset
BURN_IN_SIZE = cfg.KT.burn_in_size
WINDOW_SIZE = getattr(cfg.KT, 'window_size', -1)  # Sliding window size; default -1 (use all data)
NUM_SAMPLE_STUDENTS = getattr(cfg.KT, 'num_sample_students', -1)  # Number of sampled students; default -1 (use all)

# One-step propagation using KT2 modules.

_HISTORY_ATTR = "_kt_question_history"
_MAPPING_ATTR = "_kt_kc_mapping"
_TRAIN_SIZE_ATTR = "_kt_train_size"
_EXTRA_DATA_ATTR = "_kt_extra_records"
_TRAIN_UIDS_ATTR = "_kt_train_uids"
_TEST_UIDS_ATTR = "_kt_test_uids"
_INITIAL_TRAIN_SIZE_ATTR = "_kt_initial_train_size"


def configure_temp_recordings_path(path: str | None) -> str:
    """Set process-local temp recordings path used by KT update pipeline."""
    return _set_temp_recordings_path(path)


def current_temp_recordings_path() -> str:
    return _get_temp_recordings_path()


def one_step_update_all(
    rt: KTRuntime,
    observations_by_student: Dict[str, List[Dict[str, Any]]],
    *,
    mode: str = "actual",
    tags: Optional[Mapping[str, str]] = None,
    temp_root: Optional[str] = None,
    reset_temp: bool = False,
    persist_temp_records: bool = True,
) -> Dict[str, Dict[str, Optional[str]]]:
    """Perform one-step updates for a subset of students while keeping KT2 semantics.

    Parameters
    ----------
    observations_by_student:
        Mapping from student identifiers to new observation dicts. Identifiers may
        include or omit the ``.json`` suffix.
    mode:
        ``"actual"`` (default) persists updates back to the canonical EM results.
        ``"assumption"`` writes results to a temp directory without mutating the
        runtime cache or on-disk state.
    tags:
        Optional mapping of student ids to explicit filename tags (only used when
        ``mode="assumption"``).
    temp_root:
        Optional override for the temporary directory root (defaults to
        ``rt.temp_dir()``).
    reset_temp:
        When ``True``, delete the recordings temp file before processing so the
        model reloads data from the original dataset.
    persist_temp_records:
        When ``True`` in ``mode="actual"``, append newly observed records into
        ``recordings_temp.json`` for cross-run persistence. Set to ``False`` for
        eval/simulation flows that should not read/write temp recordings.
    """
    if not observations_by_student:
        return {}

    if reset_temp:
        # Preserve caller-selected practice sizes across temp resets.
        # Without this, clear_actual_temp_graphs() resets to historical max steps.
        preserved_practice_sizes = dict(getattr(rt, "_practice_size_map", {}))
        reset_temp_recordings()
        clear_cached_data()
        rt.clear_actual_temp_graphs()
        for sid, size in preserved_practice_sizes.items():
            rt.set_practice_size(sid, int(size))
        for attr in (
            _HISTORY_ATTR,
            _TRAIN_SIZE_ATTR,
            _INITIAL_TRAIN_SIZE_ATTR,
            _EXTRA_DATA_ATTR,
            _TRAIN_UIDS_ATTR,
            _TEST_UIDS_ATTR,
        ):
            if hasattr(rt, attr):
                delattr(rt, attr)
    resolved_obs = _resolve_student_ids(rt, observations_by_student)
    if not resolved_obs:
        return {}

    rt._active_students = set(resolved_obs.keys())

    history = _ensure_history(rt)
    kc_mapping = _ensure_kc_mapping(rt)
    train_sizes = _ensure_train_sizes(rt)
    extra_map = _ensure_extra_records(rt)
    tag_map = tags or {}

    mode_norm = mode.lower()
    if mode_norm not in {"actual", "assumption"}:
        raise ValueError(f"Unsupported update mode: {mode}")
    is_actual = mode_norm == "actual"
    temp_root = temp_root or (rt.temp_dir() if not is_actual else None)

    results: Dict[str, Dict[str, Optional[str]]] = {}

    for sid, obs_list in resolved_obs.items():
        if not obs_list:
            continue

        dataset_id = _extract_dataset_id(rt, sid)
        practice_size = rt.practice_size_for_student(sid)
        train_sizes[dataset_id] = max(practice_size, train_sizes.get(dataset_id, BURN_IN_SIZE))
        current_param_graph = rt.param_graph if is_actual else deepcopy(rt.param_graph)
        r_diff_current = _resolve_rdiff(current_param_graph, rt.root_kc)

        base_graph = rt.load_student_graph(sid, practice_size=practice_size)
        target_graph = base_graph if is_actual else deepcopy(base_graph)
        student_history = history.setdefault(sid, defaultdict(list))
        working_history = (
            student_history if is_actual else _clone_history(student_history)
        )

        updated_kcs: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        normalized_records: List[Dict[str, Any]] = []

        for obs in obs_list:
            kc_raw = obs.get("kc")
            if not kc_raw:
                continue
            kc = _normalize_kc_name(kc_raw, kc_mapping)
            if kc not in target_graph:
                continue
            record = _normalize_observation(
                obs,
                kc,
                len(working_history.get(kc, [])),
            )
            working_history[kc].append(record)
            updated_kcs[kc].append(record)
            normalized_records.append(record)

        if not updated_kcs:
            continue

        # Apply the sliding window to the E-step: only pass in-window data, without modifying working_history itself
        if WINDOW_SIZE > 0:
            single_map = {
                kc: deepcopy(records[-WINDOW_SIZE:]) 
                for kc, records in working_history.items()
            }
        else:
            single_map = {kc: deepcopy(records) for kc, records in working_history.items()}
        get_emission_dict(reset=True)

        for kc in updated_kcs.keys():
            target_graph = update_e_step(
                target_graph,
                single_map,
                current_param_graph,
                r_diff_current,
                kc,
            )
            _sync_question_count(target_graph, kc, len(working_history[kc]))
        
        initial_size = _initial_train_size(rt, dataset_id)
        existing_extras = [dict(item) for item in extra_map.get(dataset_id, [])]

        updated_param_graph, new_r_diff, combined_extras = _run_parameter_update(
            rt=rt,
            sid=sid,
            dataset_id=dataset_id,
            target_graph=target_graph,
            new_records=normalized_records,
            param_graph=current_param_graph,
            is_actual=is_actual,
            kc_mapping=kc_mapping,
            train_sizes=train_sizes,
            initial_size=initial_size,
            existing_extras=existing_extras,
            window_size=WINDOW_SIZE,
            num_sample_students=NUM_SAMPLE_STUDENTS,
        )

        if updated_param_graph is None:
            continue

        new_size = practice_size + len(normalized_records)

        if is_actual:
            rt.ensure_actual_dirs()
            student_path = rt.save_student_graph(
                sid,
                target_graph,
                dest_dir=rt.actual_student_dir(),
                filename=rt.student_filename(sid, new_size),
                practice_size=new_size,
                cache=True,
            )
            param_path = rt.save_param_graph(
                graph=updated_param_graph,
                dest_dir=rt.actual_param_dir(),
                filename=rt._param_filename(sid, new_size),
                student_id=sid,
                practice_size=new_size,
                update_runtime=True,
            )
            rt.set_practice_size(sid, new_size)
            r_diff_current = new_r_diff or r_diff_current
            extra_map[dataset_id] = combined_extras
            if persist_temp_records and normalized_records:
                # Only save the new records to the temp file when in actual mode (i.e., student really answered the question)
                append_temp_records(dataset_id, normalized_records)
        else:
            tag_value = _resolve_assumption_tag(
                sid,
                updated_kcs,
                tag_map,
            )
            assert temp_root is not None
            student_temp_dir = os.path.join(temp_root, "students_graphs")
            parameter_temp_dir = os.path.join(temp_root, "parameter_graphs")
            student_filename = f"{sid}__{tag_value}_step_{new_size}.json"
            param_filename = f"parameter_graph_student_{sid}_step_{new_size}__{tag_value}.json"
            student_path = rt.save_student_graph(
                sid,
                target_graph,
                dest_dir=student_temp_dir,
                filename=student_filename,
                practice_size=new_size,
                cache=False,
                update_practice_map=False,
            )
            param_path = rt.save_param_graph(
                graph=updated_param_graph,
                dest_dir=parameter_temp_dir,
                filename=param_filename,
                student_id=sid,
                practice_size=new_size,
                update_runtime=False,
            )
            results.setdefault(sid, {})["tag"] = tag_value

        result_entry = results.setdefault(sid, {})
        result_entry["student_graph"] = student_path
        result_entry["parameter_graph"] = param_path

    rt._active_students = set()
    return results


def one_step_update_single(
    rt: KTRuntime,
    student_id: str,
    observations: List[Dict[str, Any]],
    *,
    mode: str = "actual",
    tag: Optional[str] = None,
    temp_root: Optional[str] = None,
    reset_temp: bool = False,
    persist_temp_records: bool = True,
) -> Optional[Dict[str, Optional[str]]]:
    """Convenience wrapper to update a single student."""
    if not observations:
        return None
    sid = rt._canonical_student_id(student_id)
    tag_map = {sid: tag} if tag is not None else None
    result = one_step_update_all(
        rt,
        {sid: observations},
        mode=mode,
        tags=tag_map,
        temp_root=temp_root,
        reset_temp=reset_temp,
        persist_temp_records=persist_temp_records,
    )
    return result.get(sid)


# ---------- Helpers ----------

def _ensure_history(rt: KTRuntime) -> Dict[str, Dict[str, List[Dict[str, Any]]]]:
    history = getattr(rt, _HISTORY_ATTR, None)
    base_records = _load_base_records()
    _ensure_train_sizes(rt)
    if history is None:
        history = {}

    for sid in rt.iter_student_ids():
        dataset_id = _extract_dataset_id(rt, sid)
        initial_limit = _initial_train_size(rt, dataset_id)
        per_kc = history.setdefault(sid, defaultdict(list))
        # Expand history up to the desired limit using base records in order.
        existing_total = sum(len(items) for items in per_kc.values())
        if existing_total >= initial_limit:
            continue
        for record in base_records.get(dataset_id, [])[existing_total:initial_limit]:
            kc = record.get("kc")
            if not kc:
                continue
            per_kc[kc].append(
                _normalize_observation(
                    record,
                    kc,
                    len(per_kc[kc]),
                    allow_existing_question=True,
                )
            )

    setattr(rt, _HISTORY_ATTR, history)
    _ensure_train_sizes(rt)
    return history


def _ensure_kc_mapping(rt: KTRuntime) -> Dict[str, str]:
    mapping = getattr(rt, _MAPPING_ATTR, None)
    if mapping is not None:
        return mapping

    mapping_path = os.path.join(graph_dir, dataset, "merged_mapping.json")
    if os.path.isfile(mapping_path):
        with open(mapping_path, "r", encoding="utf-8") as f:
            mapping = json.load(f)
    else:
        mapping = {}

    setattr(rt, _MAPPING_ATTR, mapping)
    return mapping


def _ensure_train_sizes(rt: KTRuntime) -> Dict[str, int]:
    train_sizes = getattr(rt, _TRAIN_SIZE_ATTR, None)
    if train_sizes is None:
        initialize_train_uids()
        test_uids, train_size_map, train_uids = get_train_uids()

        initial_map = dict(train_size_map)
        train_sizes = dict(initial_map)

        for sid in rt.iter_student_ids():
            train_sizes.setdefault(sid, BURN_IN_SIZE)
            initial_map.setdefault(sid, BURN_IN_SIZE)

        setattr(rt, _TRAIN_SIZE_ATTR, train_sizes)
        setattr(rt, _INITIAL_TRAIN_SIZE_ATTR, initial_map)
        setattr(rt, _TRAIN_UIDS_ATTR, list(train_uids))
        setattr(rt, _TEST_UIDS_ATTR, list(test_uids))

    active = getattr(rt, "_active_students", set())
    practice_map = getattr(rt, "_practice_size_map", {})
    initial_map = getattr(rt, _INITIAL_TRAIN_SIZE_ATTR, {})

    for sid in active:
        size = int(practice_map.get(sid, BURN_IN_SIZE))
        train_sizes[sid] = size
        initial_map[sid] = size

    setattr(rt, _INITIAL_TRAIN_SIZE_ATTR, initial_map)

    return train_sizes


def _ensure_extra_records(rt: KTRuntime) -> Dict[str, List[Dict[str, Any]]]:
    extra = getattr(rt, _EXTRA_DATA_ATTR, None)
    if extra is not None:
        return extra
    extra = {}
    setattr(rt, _EXTRA_DATA_ATTR, extra)
    return extra


def _initial_train_size(rt: KTRuntime, dataset_id: str) -> int:
    practice_map = getattr(rt, "_practice_size_map", {})
    active = getattr(rt, "_active_students", set())
    if dataset_id in active:
        return int(practice_map.get(dataset_id, BURN_IN_SIZE))
    return BURN_IN_SIZE


def _load_base_records() -> Dict[str, List[Dict[str, Any]]]:
    cache = getattr(_load_base_records, "_cache", None)
    if cache is not None:
        return cache

    initialize_train_uids()
    datas = load_full_recordings()
    cache = {sid: [dict(item) for item in items] for sid, items in datas.items()}

    setattr(_load_base_records, "_cache", cache)
    return cache


def _normalize_kc_name(kc: str, mapping: Dict[str, str]) -> str:
    return mapping.get(kc, kc)


def _normalize_observation(
    obs: Dict[str, Any],
    kc: str,
    offset: int,
    allow_existing_question: bool = False,
) -> Dict[str, Any]:
    record = dict(obs)
    record["kc"] = kc
    try:
        record["response"] = int(record.get("response", 0))
    except (TypeError, ValueError):
        record["response"] = 0
    difficulty = record.get("difficulty", "medium")
    record["difficulty"] = str(difficulty).lower()

    if allow_existing_question and record.get("question"):
        return record

    question = record.get("question") or record.get("question_id")
    if not question:
        question = f"{kc}#auto_{offset:04d}"
    record["question"] = question
    return record


def _sync_question_count(graph: Dict[str, Any], kc: str, count: int) -> None:
    node = graph.get(kc)
    if node is None:
        return
    if hasattr(node, "question_count"):
        node.question_count = count
    elif isinstance(node, dict):
        node["question_count"] = count


def _extract_dataset_id(rt: KTRuntime, sid: str) -> str:
    return rt._canonical_student_id(sid)


def _resolve_rdiff(param_graph: Dict[str, Any], root_name: str | None) -> Iterable[float]:
    if root_name and root_name in param_graph:
        node = param_graph[root_name]
        if isinstance(node, dict):
            rdiff = node.get("r_diff")
            if isinstance(rdiff, (list, tuple)):
                return rdiff

    for params in param_graph.values():
        if isinstance(params, dict):
            rdiff = params.get("r_diff")
            if isinstance(rdiff, (list, tuple)):
                return rdiff

    return [0.7, 0.8, 0.9]


def _resolve_student_ids(
    rt: KTRuntime, raw: Dict[str, List[Dict[str, Any]]]
) -> Dict[str, List[Dict[str, Any]]]:
    resolved: Dict[str, List[Dict[str, Any]]] = {}
    for key, obs in raw.items():
        if not obs:
            continue
        sid = rt._canonical_student_id(str(key))
        path = rt.student_path(sid)
        if not os.path.isfile(path):
            continue
        resolved[sid] = obs
    return resolved


def _clone_history(history: Dict[str, List[Dict[str, Any]]]) -> Dict[str, List[Dict[str, Any]]]:
    return defaultdict(list, {kc: list(records) for kc, records in history.items()})


def _resolve_assumption_tag(
    sid: str,
    updated_kcs: Dict[str, List[Dict[str, Any]]],
    tags: Mapping[str, str],
) -> str:
    explicit = tags.get(sid)
    if explicit:
        return _slugify_tag(explicit)

    for kc, records in updated_kcs.items():
        if not records:
            continue
        response = records[-1].get("response", 0)
        base = f"{kc}_resp{response}"
        return _slugify_tag(base)
    return _slugify_tag("assumption")


def _slugify_tag(tag: str) -> str:
    tag = tag.strip()
    if not tag:
        return "assumption"
    tag = re.sub(r"\s+", "_", tag)
    tag = re.sub(r"[^A-Za-z0-9_\-]+", "_", tag)
    tag = tag.strip("_")
    return tag or "assumption"


def _calibrate_r_diff_local(
    uids: List[str],
    data_by_uid: Dict[str, List[Dict[str, Any]]],
    graphs_by_uid: Dict[str, Dict[str, Any]],
    train_size_map: Mapping[str, int],
    mapping: Mapping[str, str],
) -> List[float]:
    correct_difficulty: List[str] = []
    incorrect_difficulty: List[str] = []
    correct_posterior: List[float] = []
    incorrect_posterior: List[float] = []

    for uid in uids:
        data = data_by_uid.get(uid, [])
        graph = graphs_by_uid.get(uid, {})
        # `data` is already windowed upstream, so just iterate over it directly
        for item in data:
            kc_raw = item.get("kc")
            kc = mapping.get(kc_raw, kc_raw)
            node = graph.get(kc)
            if node is None:
                continue
            posterior1 = getattr(node, "posterior1", None)
            if posterior1 is None:
                continue
            response = int(item.get("response", 0))
            difficulty = str(item.get("difficulty", "medium")).lower()
            if response == 1:
                correct_difficulty.append(difficulty)
                correct_posterior.append(float(posterior1))
            else:
                incorrect_difficulty.append(difficulty)
                incorrect_posterior.append(float(posterior1))

    if not correct_difficulty and not incorrect_difficulty:
        return [0.7, 0.8, 0.9]

    return calibrate_r_diff_closed_form(
        correct_difficulty,
        incorrect_difficulty,
        correct_posterior,
        incorrect_posterior,
    )


def _run_parameter_update(
    rt: KTRuntime,
    sid: str,
    dataset_id: str,
    target_graph: Dict[str, Any],
    new_records: List[Dict[str, Any]],
    param_graph: Dict[str, Any],
    *,
    is_actual: bool,
    kc_mapping: Dict[str, str],
    train_sizes: Dict[str, int],
    initial_size: int,
    existing_extras: List[Dict[str, Any]],
    window_size: int = -1,
    num_sample_students: int = -1,
) -> Tuple[Optional[Dict[str, Any]], Optional[List[float]], List[Dict[str, Any]]]:
    if not new_records:
        return param_graph, None, existing_extras

    base_records = _load_base_records()

    # Build the historical burn-in pool from other students.
    # num_sample_students > 0 subsamples the pool; -1 uses every other student.
    if num_sample_students > 0:
        all_other_students = [s for s in base_records.keys() if s != dataset_id]

        if len(all_other_students) > num_sample_students:
            import random
            sampled_student_ids = random.sample(all_other_students, num_sample_students)
        else:
            sampled_student_ids = all_other_students

        historical = []
        for sampled_sid in sampled_student_ids:
            historical.extend([dict(item) for item in base_records.get(sampled_sid, [])[:initial_size]])
    else:
        historical = []
        for other_sid in base_records.keys():
            if other_sid != dataset_id:
                historical.extend([dict(item) for item in base_records.get(other_sid, [])[:initial_size]])

    extras_history = [dict(item) for item in existing_extras]
    combined_extras = extras_history + [dict(rec) for rec in new_records]

    # Full data sequence (keep the complete combined_extras for return value and history maintenance)
    full_sequence = historical + combined_extras
    if not full_sequence:
        return param_graph, None, combined_extras

    # Apply sliding window only to the current student's extra data; burn-in data is unaffected
    if window_size > 0:
        windowed_extras = combined_extras[-window_size:]  # window only the current student's data
        data_sequence = historical + windowed_extras  # burn-in data + windowed current-student data
    else:
        data_sequence = full_sequence  # use all data

    new_total = initial_size + len(combined_extras)

    if is_actual:
        # train_sizes records the true accumulated total, independent of the window
        train_sizes[dataset_id] = min(new_total, len(full_sequence))
        train_size_map: Dict[str, int] = train_sizes
    else:
        train_size_map = dict(train_sizes)
        train_size_map[dataset_id] = min(new_total, len(full_sequence))

    uids = [dataset_id]
    graphs_payload = {dataset_id: target_graph}
    datas_payload = {dataset_id: data_sequence}

    r_diff_new = _calibrate_r_diff_local(
        uids,
        datas_payload,
        graphs_payload,
        train_size_map,
        kc_mapping,
    )

    updated_param_graph = m_step(
        graphs_payload,
        datas_payload,
        uids,
        r_diff_new,
        param_graph,
        kc_mapping,
    )

    # Return the complete combined_extras so the history stays intact
    return updated_param_graph, r_diff_new, combined_extras
