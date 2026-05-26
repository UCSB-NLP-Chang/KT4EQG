"""Shared data utilities for KT baseline methods (BKT, DKT).

Converts raw dataset recordings into KC-level sequences suitable for
KC-level knowledge tracing models.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple


def load_question_info(path: str | Path) -> Dict[str, Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_qid_to_kc(question_info: Dict[str, Dict[str, Any]]) -> Dict[str, str]:
    """Map question_id -> KC name."""
    return {str(qid): q["kc"] for qid, q in question_info.items() if "kc" in q}


def load_recordings(path: str | Path) -> List[Dict[str, Any]]:
    """Load recordings.jsonl → list of student records.

    Each record: {"student_id": str, "exercises_logs": [qid, ...], "is_corrects": ["0"/"1", ...]}
    """
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def recordings_to_kc_sequences(
    recordings: List[Dict[str, Any]],
    qid_to_kc: Dict[str, str],
) -> Dict[str, List[Tuple[str, int]]]:
    """Convert raw recordings to KC-level sequences.

    Returns:
        {student_id: [(kc_name, correct), ...]}
        Entries whose question_id has no KC mapping are skipped.
    """
    result: Dict[str, List[Tuple[str, int]]] = {}
    for rec in recordings:
        sid = str(rec["student_id"])
        qlogs = rec["exercises_logs"]
        corrects = rec["is_corrects"]
        seq: List[Tuple[str, int]] = []
        for qid, c in zip(qlogs, corrects):
            kc = qid_to_kc.get(str(qid))
            if kc is None:
                continue
            seq.append((kc, int(c)))
        if seq:
            result[sid] = seq
    return result


def build_kc_index(kc_sequences: Dict[str, List[Tuple[str, int]]]) -> Tuple[Dict[str, int], List[str]]:
    """Build KC name ↔ integer index mapping.

    Returns:
        (kc_to_idx, idx_to_kc)
    """
    all_kcs: set[str] = set()
    for seq in kc_sequences.values():
        for kc, _ in seq:
            all_kcs.add(kc)
    idx_to_kc = sorted(all_kcs)
    kc_to_idx = {kc: i for i, kc in enumerate(idx_to_kc)}
    return kc_to_idx, idx_to_kc


def extract_burn_in(
    kc_sequences: Dict[str, List[Tuple[str, int]]],
    burn_in_size: int,
) -> Dict[str, List[Tuple[str, int]]]:
    """Extract first `burn_in_size` observations per student."""
    return {
        sid: seq[:burn_in_size]
        for sid, seq in kc_sequences.items()
    }


def load_practice_traces(traj_path: str | Path) -> Dict[str, List[Tuple[str, int]]]:
    """Load practice trajectories from eval run output.

    Returns:
        {student_id: [(kc, correct), ...]} from practice_trace
    """
    result: Dict[str, List[Tuple[str, int]]] = {}
    with open(traj_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            sid = str(row["student_id"])
            trace = row.get("practice_trace", [])
            seq = [(t.get("verified_kc") or t["kc"], int(t["sampled_correct"])) for t in trace]
            result[sid] = seq
    return result


def load_exam_set(path: str | Path) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_sids_from_graph_dirs(
    dataset_dir: str | Path,
    split: str,
    module_state_dirs: Sequence[str] = (
        "application_states", "computation_states", "counting_states",
    ),
) -> set[str]:
    """Extract unique student IDs from {module_states}/{split}/students_graphs/ filenames.

    Filename pattern: E_step_student_{sid}_step_{N}.json
    Returns the union of student IDs across all modules for the given split.
    """
    import re
    dataset_dir = Path(dataset_dir)
    all_sids: set[str] = set()
    for mod_dir in module_state_dirs:
        graph_dir = dataset_dir / mod_dir / split / "students_graphs"
        if not graph_dir.is_dir():
            continue
        for fname in graph_dir.iterdir():
            m = re.search(r"student_(\d+)_step", fname.name)
            if m:
                all_sids.add(m.group(1))
    return all_sids


def collect_kc_observations(
    kc_sequences: Dict[str, List[Tuple[str, int]]],
    max_seq_len: int = 0,
) -> Dict[str, List[List[int]]]:
    """Group observations by KC for BKT fitting.

    Args:
        kc_sequences: {student_id: [(kc, correct), ...]}
        max_seq_len: If > 0, truncate each student's per-KC sequence to this
            length (keeps the first ``max_seq_len`` observations). 0 = no limit.

    Returns:
        {kc_name: [[correct_0, correct_1, ...], ...]}
        Each inner list is one student's sequence of correct/incorrect
        for that KC (preserving temporal order within the full sequence).
    """
    kc_obs: Dict[str, Dict[str, List[int]]] = {}
    for sid, seq in kc_sequences.items():
        for kc, correct in seq:
            if kc not in kc_obs:
                kc_obs[kc] = {}
            if sid not in kc_obs[kc]:
                kc_obs[kc][sid] = []
            kc_obs[kc][sid].append(correct)
    # Convert to list of lists (one per student)
    result: Dict[str, List[List[int]]] = {}
    for kc, sid_map in kc_obs.items():
        seqs = [obs for obs in sid_map.values() if len(obs) >= 1]
        if max_seq_len > 0:
            seqs = [s[:max_seq_len] for s in seqs]
        result[kc] = seqs
    return result
