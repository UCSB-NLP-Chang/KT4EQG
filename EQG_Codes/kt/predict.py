"""New-question prediction (read-only) using KT2 modules."""
from typing import Dict, Any, Mapping, Optional
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from kt.runtime import KTRuntime
from kt.KT2.KT.KT import calculate_prior

DIFF_TO_ID = {"easy": 0, "medium": 1, "hard": 2}


def _parents_of(node_obj) -> list:
    # KC_tree.io builds KCNode objects; each node has .parents (list of KCNode)
    if hasattr(node_obj, "parents"):
        return [p.name for p in node_obj.parents]
    # Fallback to dict shape
    return list(node_obj.get("parents", [])) if isinstance(node_obj, dict) else []


def _calculate_prior(student_graph: Mapping[str, Any], kc: str, param_graph: Mapping[str, Any]) -> float:
    # Climb to root following the first parent chain, then accumulate gamma
    track = [kc]
    cur = kc
    while True:
        node = student_graph.get(cur)
        if node is None:
            break
        parents = _parents_of(node)
        if not parents:
            # At root
            root_gamma = _safe_read(param_graph, cur, "gamma_root", default=0.5)
            prior = float(root_gamma)
            break
        cur = parents[0]
        track.append(cur)
    # Accumulate gamma from near-root to leaf
    while len(track) > 1:
        node = track.pop()
        g = float(_safe_read(param_graph, node, "gamma", default=0.0))
        prior = prior + (1.0 - prior) * g
    return float(prior)


def _safe_read(d: Mapping[str, Any], node: str, key: str, default=None):
    if node in d and isinstance(d[node], dict) and key in d[node]:
        return d[node][key]
    return default


def predict_on_new_question(
    rt: KTRuntime,
    student_id: str,
    question: Dict[str, Any],
    *,
    graph_path: Optional[str] = None,
    graph_variant: Optional[str] = None,
    param_path: Optional[str] = None,
    param_graph: Optional[Mapping[str, Any]] = None,
) -> float:
    """P(correct | student's current graph, question), without mutating any state.
    Uses your KT2-style formula: pred = phi * post1 + epsilon * (1 - post1).
    - phi from root's r_diff[diff_id] (fallback: search any node's r_diff)
    - epsilon from param_graph[kc]['epsilon'] (fallback: 0.1)
    - post1 from student_graph[kc].posterior1 (fallback: prior from gamma_root/gamma)
    """
    kc = question.get("kc")
    if not kc:
        raise ValueError("question must contain 'kc'")
    diff = question.get("difficulty", "medium").lower()
    diff_id = DIFF_TO_ID.get(diff, 1)

    param_graph = rt.load_param_graph(param_path) if param_path else param_graph or rt.param_graph
    s_graph = rt.load_student_graph(student_id, path=graph_path, variant=graph_variant)

    # phi via r_diff
    rdiff = _safe_read(param_graph, rt.root_kc, "r_diff", default=None)
    if not (isinstance(rdiff, (list, tuple)) and len(rdiff) >= 3):
        for node_name, params in param_graph.items():
            if isinstance(params, dict):
                cand = params.get("r_diff", None)
                if isinstance(cand, (list, tuple)) and len(cand) >= 3:
                    rdiff = cand
                    break
    if not (isinstance(rdiff, (list, tuple)) and len(rdiff) >= 3):
        rdiff = [0.7, 0.8, 0.9]
    phi = float(rdiff[diff_id])

    # epsilon at kc
    epsilon = float(_safe_read(param_graph, kc, "epsilon", default=0.1))

    # posterior1 or prior
    # NOTE: Check logic. Why is it possible for node to be None??
    node = s_graph.get(kc)
    post1 = getattr(node, "posterior1", None) if node is not None else None
    if post1 is None:
        post1 = _calculate_prior(s_graph, kc, param_graph)
    else:
        post1 = float(post1)

    pred = phi * post1 + epsilon * (1.0 - post1)
    return float(pred)
