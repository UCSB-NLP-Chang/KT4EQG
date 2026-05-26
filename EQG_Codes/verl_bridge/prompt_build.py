from __future__ import annotations

import random
import os
import hashlib
from typing import Dict, Iterable, List, Optional, Tuple

from prompt.prompts import QUESTION_SYSTEM_PROMPT_TRAINABLE, question_prompt_trainable


def format_student_state_text(
    student_graph: Dict[str, object],
    leaves: Iterable[str],
    *,
    precision: int = 4,
    shuffle: Optional[bool] = None,
    seed: Optional[int] = None,
    student_id: Optional[str] = None,
    practice_size: Optional[int] = None,
) -> str:
    """Convert a student graph into the text format expected by simple_loop."""
    if shuffle is None:
        shuffle = bool(int(os.getenv("EQG_PROMPT_SHUFFLE", "1")))

    if seed is None and bool(int(os.getenv("EQG_PROMPT_HASH_SEED", "0"))):
        if student_id is not None:
            base = f"{student_id}:{practice_size if practice_size is not None else ''}"
            digest = hashlib.sha1(base.encode("utf-8")).hexdigest()[:8]
            seed = int(digest, 16)

    items = list(student_graph.items())
    if shuffle and len(items) > 1:
        rng = random.Random(seed) if seed is not None else random
        rng.shuffle(items)

    kcs = [node for name, node in items if name in set(leaves)]
    lines: List[str] = []
    for kc in kcs:
        value = getattr(kc, "posterior1", None)
        try:
            value_str = f"{float(value):.{precision}f}"
        except (TypeError, ValueError):
            value_str = str(value)
        lines.append(f"{kc.name}: {value_str}")
    return "\n".join(lines)


def _student_state_text(student_graph: Dict[str, object], leaves: Iterable[str]) -> str:
    return format_student_state_text(student_graph, leaves, precision=4)


def build_question_prompt(
    student_graph: Dict[str, object],
    leaves: List[str],
    *,
    student_id: Optional[str] = None,
    practice_size: Optional[int] = None,
) -> Tuple[str, str]:
    """Build system/user prompts exactly as QuestionSampler.sample_one (one-stage)."""
    student_state_text = format_student_state_text(
        student_graph,
        leaves,
        precision=4,
        student_id=student_id,
        practice_size=practice_size,
    )
    system_prompt = QUESTION_SYSTEM_PROMPT_TRAINABLE
    user_prompt = question_prompt_trainable(student_state_text)
    return system_prompt, user_prompt
