"""Utility helpers shared across verl_bridge scripts."""
from __future__ import annotations

import os
import sys
from typing import Dict, List, Optional

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.config import load_config
cfg = load_config()


def _load_state_root(split: Optional[str] = None) -> str:
    """Read the EM results root that stores KT2 graphs."""
    base = cfg.KT.state_graph_path
    if not base:
        raise ValueError("`state_graph_path` missing in config/config.yaml")
    target_split = split or os.getenv("EQG_STATE_SPLIT") or "train"
    candidate = os.path.join(base, target_split)
    return candidate if os.path.isdir(candidate) else base


def _leaf_concepts(student_graph: Dict[str, object]) -> List[str]:
    """Return KC names that have no children in the given student graph."""
    leaves: List[str] = []
    for name, node in student_graph.items():
        children = getattr(node, "children", None)
        if children is None:
            continue
        if len(children) == 0:
            leaves.append(name)
    if not leaves:
        raise ValueError("Failed to locate any leaf KCs from the student graph.")
    leaves.sort()
    return leaves
