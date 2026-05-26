from __future__ import annotations
import os
import sys
import uuid
from typing import Any, Dict
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from kt.predict import predict_on_new_question
from kt.update import one_step_update_single
from kt.runtime import KTRuntime

from verifier.inference import VerifierScorer

def align_score(question: Dict[str, Any], device: str | None = None) -> float:
    verifier = VerifierScorer(device=device) if device is not None else VerifierScorer()
    
    q_text = question.get("question_text") or question.get("text") or ""
    kc = question.get("kc") or question.get("knowledge_concept") or ""
    diff = question.get("difficulty") or question.get("difficulty_level")
    v_align = verifier.score_alignment(q_text, kc, difficulty=diff)
    return v_align
    
def value_fn(question, rt, student_id):
    """Estimate the value of asking *question* for *student_id*.

    Returns the expected mastery gain proxy under KT2 by averaging the
    post-update mastery sums for both possible responses.
    """

    if "kc" not in question and "knowledge_concept" not in question:
        raise ValueError("question must contain 'kc' or 'knowledge_concept'")

    base_question = {
        "kc": question.get("kc") or question.get("knowledge_concept"),
        "difficulty": "medium",
    }

    p_correct = predict_on_new_question(rt, student_id, base_question)

    # V_edu is the expected posterior mass after a correct response.
    u1 = _posterior_sum_after_response(rt, student_id, base_question, response=1)
    v_edu = u1

    return v_edu


def _posterior_sum_after_response(rt, student_id, base_question, response: int) -> float:
    observation = {
        "kc": base_question["kc"],
        "difficulty": base_question["difficulty"],
        "response": int(response),
        "source": "value_fn",
    }
    tag = f"value_resp{response}_{uuid.uuid4().hex}"
    update_info = one_step_update_single(
        rt,
        student_id,
        [observation],
        mode="assumption",
        tag=tag,
        reset_temp=False,
    )
    if not update_info:
        raise RuntimeError("Assumption update returned no metadata.")

    temp_graph = update_info.get("student_graph")
    if not temp_graph:
        raise RuntimeError("Temporary student graph path missing.")

    graph = rt.load_student_graph(student_id, path=temp_graph)

    return _calculate_posterior_sum(graph)


def _calculate_posterior_sum(graph):
    total = 0.0
    for node in graph.values():
        if hasattr(node, "posterior1"):
            total += float(node.posterior1)
        elif isinstance(node, dict):
            p = node.get("posterior1")
            if p is not None:
                total += float(p)
    return total
