"""Runtime holder for KT2 graphs (uses KC_tree.io)."""
from __future__ import annotations
import os, sys, shutil
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import re
import json
import glob
from typing import Dict, Any, Optional, Iterable, Tuple

from kt.KC_tree.io import load_graph, save_graph  # loads/saves KCNode graphs

ROOT_KC = "应用题模块"
_STUDENT_RE = re.compile(r"E_step_student_(.+?)_step_(\d+)\.json(?:\.json)?$")

class KTRuntime:
    """Runtime that manages parameter graph + all student graphs from an EM results dir.

    Directory layout (as you described):
      {em_results_root}/parameter_graphs/*.json
      {em_results_root}/students_graphs/*.json

    - Parameter graphs are raw JSON (dict of params).
    - Student graphs are KCNode dicts serialized by KC_tree.io.save_graph.
    """

    def __init__(
        self,
        em_results_root: str,
        *,
        burn_in_size: int = 10,
        temp_dir: Optional[str] = None,
    ):
        self.root = os.path.abspath(em_results_root)
        self.param_dir = os.path.join(self.root, "parameter_graphs")
        self.student_dir = os.path.join(self.root, "students_graphs")
        if not os.path.isdir(self.param_dir):
            raise FileNotFoundError(f"parameter_graphs not found: {self.param_dir}")
        if not os.path.isdir(self.student_dir):
            raise FileNotFoundError(f"students_graphs not found: {self.student_dir}")

        self.burn_in_size = burn_in_size

        # Discover available student graphs and their practice sizes.
        self._student_versions: Dict[str, Dict[int, str]] = self._scan_student_versions()
        self._practice_size_map: Dict[str, int] = {
            sid: max(sizes) for sid, sizes in self._student_versions.items()
        }

        # Temp directory for assumption/actual transient states.
        # Use a run-scoped override when provided to avoid cross-process conflicts.
        self._temp_dir = (
            os.path.abspath(temp_dir) if temp_dir else os.path.join(self.root, "_temp_states")
        )
        self._actual_student_dir = os.path.join(self._temp_dir, "students_graphs_actual")
        self._actual_param_dir = os.path.join(self._temp_dir, "parameter_graphs_actual")

        # Load the shared/burn-in parameter graph as the default.
        self.param_path = self._resolve_param_path_for_student(None, burn_in_size)
        self.param_graph = self._load_json(self.param_path)
        # Keep canonical parameter path for resets
        self._param_canonical_path = self.param_path

        # Cache for student graphs (KCNode dicts)
        self._student_cache: Dict[str, Dict[str, Any]] = {}

        # Infer root KC name (robust to gamma_root types)
        # Use a sample student graph for fallback if needed
        sample_sid = self.first_student_id()
        sample_graph = self.load_student_graph(sample_sid) if sample_sid else {}
        self._root_kc = self._infer_root_name(self.param_graph, sample_graph)

    # ---------- IO helpers ----------
    @staticmethod
    def _load_json(path: str) -> Dict[str, Any]:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    @staticmethod
    def _pick_latest_json(dir_path: str) -> str:
        files = sorted(glob.glob(os.path.join(dir_path, "*.json")))
        if not files:
            raise FileNotFoundError(f"No json found under {dir_path}")
        finals = [p for p in files if re.search(r"final", os.path.basename(p))]
        return finals[0] if finals else files[-1]

    def _scan_student_versions(self) -> Dict[str, Dict[int, str]]:
        versions: Dict[str, Dict[int, str]] = {}
        for p in glob.glob(os.path.join(self.student_dir, "*.json*")):
            m = _STUDENT_RE.match(os.path.basename(p))
            if not m:
                continue
            sid, size_str = m.group(1), m.group(2)
            size = int(size_str)
            versions.setdefault(sid, {})[size] = os.path.basename(p)
        if not versions:
            raise FileNotFoundError(f"No student graphs found under {self.student_dir}")
        return versions

    def practice_size_for_student(self, student_id: str) -> int:
        canon = self._canonical_student_id(student_id)
        return int(self._practice_size_map.get(canon, self.burn_in_size))

    def available_practice_sizes(self, student_id: str) -> Iterable[int]:
        """Return available practice sizes for a student (sorted ascending)."""
        canon = self._canonical_student_id(student_id)
        return sorted(self._student_versions.get(canon, {}).keys())

    def set_practice_size(self, student_id: str, practice_size: int) -> None:
        canon = self._canonical_student_id(student_id)
        self._practice_size_map[canon] = max(int(practice_size), self.burn_in_size)

    def iter_student_ids(self) -> Iterable[str]:
        seen = set()
        for canon in sorted(self._student_versions.keys()):
            if canon not in seen:
                seen.add(canon)
                yield canon

    def first_student_id(self) -> Optional[str]:
        for sid in self.iter_student_ids():
            return sid
        return None

    @staticmethod
    def _canonical_student_id(student_id: str) -> str:
        base = student_id
        if base.endswith(".json"):
            base = base[:-5]
        match = re.match(r"E_step_student_(.+?)_step", base)
        if match:
            return match.group(1)
        return base

    def _student_filename(self, student_id: str, practice_size: int) -> str:
        return f"E_step_student_{student_id}_step_{practice_size}.json"

    def student_path(self, student_id: str, practice_size: Optional[int] = None) -> str:
        canon = self._canonical_student_id(student_id)
        size = practice_size or self.practice_size_for_student(canon)
        actual = self._actual_student_path(canon, size)
        if os.path.isfile(actual):
            return actual
        fname = self._student_versions.get(canon, {}).get(size)
        if fname:
            return os.path.join(self.student_dir, fname)
        fallback = os.path.join(self.student_dir, self._student_filename(canon, size))
        if os.path.isfile(fallback):
            return fallback
        double = fallback + ".json"
        if os.path.isfile(double):
            return double
        raise FileNotFoundError(f"student graph not found for {canon} size={size}")

    def _actual_student_path(self, student_id: str, practice_size: int) -> str:
        filename = self._student_filename(self._canonical_student_id(student_id), practice_size)
        return os.path.join(self._actual_student_dir, filename)

    def _actual_param_path(self, student_id: Optional[str], practice_size: int) -> str:
        filename = self._param_filename(student_id, practice_size)
        return os.path.join(self._actual_param_dir, filename)

    def actual_student_dir(self) -> str:
        return self._actual_student_dir

    def actual_param_dir(self) -> str:
        return self._actual_param_dir

    def student_filename(self, student_id: str, practice_size: Optional[int] = None) -> str:
        canon = self._canonical_student_id(student_id)
        size = practice_size or self.practice_size_for_student(canon)
        return self._student_filename(canon, size)

    def canonical_param_filename(self) -> str:
        return os.path.basename(self._param_canonical_path)

    def temp_dir(self) -> str:
        os.makedirs(self._temp_dir, exist_ok=True)
        return self._temp_dir

    def load_student_graph(
        self,
        student_id: str,
        *,
        variant: Optional[str] = None,
        path: Optional[str] = None,
        practice_size: Optional[int] = None,
        refresh: bool = False,
    ) -> Dict[str, Any]:
        canon = self._canonical_student_id(student_id)
        size = practice_size or self.practice_size_for_student(canon)
        if path:
            dir_path, fname = os.path.dirname(path), os.path.basename(path)
        elif variant:
            dir_path = os.path.join(self.temp_dir(), "students_graphs")
            fname = f"{canon}__{variant}.json"
        else:
            target_path = self.student_path(canon, size)
            dir_path, fname = os.path.dirname(target_path), os.path.basename(target_path)

        cacheable = variant is None and path is None
        if cacheable and not refresh and canon in self._student_cache:
            return self._student_cache[canon]

        full_path = os.path.join(dir_path, fname)
        if not os.path.isfile(full_path):
            raise FileNotFoundError(f"student graph not found: {full_path}")

        graph = load_graph(dir_path, fname)

        if cacheable:
            self._student_cache[canon] = graph
            self._practice_size_map[canon] = size

        # Keep the parameter graph aligned with the student's practice size.
        self.param_path = self._resolve_param_path_for_student(canon, size)
        self.param_graph = self._load_json(self.param_path)

        return graph

    def save_student_graph(
        self,
        student_id: str,
        graph: Dict[str, Any],
        *,
        dest_dir: Optional[str] = None,
        filename: Optional[str] = None,
        practice_size: Optional[int] = None,
        cache: bool = True,
        update_practice_map: bool = True,
    ) -> str:
        canon = self._canonical_student_id(student_id)
        dir_path = dest_dir or self.student_dir
        os.makedirs(dir_path, exist_ok=True)
        if filename:
            fname = filename
        else:
            size = practice_size or self.practice_size_for_student(canon)
            fname = self._student_filename(canon, size)
        save_graph(graph, dir_path, fname)
        if cache:
            self._student_cache[canon] = graph
        elif dest_dir is None:
            self._student_cache.pop(canon, None)
        if practice_size is not None and update_practice_map:
            self._practice_size_map[canon] = practice_size
        return os.path.join(dir_path, fname)

    def save_param_graph(
        self,
        graph: Optional[Dict[str, Any]] = None,
        *,
        dest_dir: Optional[str] = None,
        filename: Optional[str] = None,
        update_runtime: bool = False,
        student_id: Optional[str] = None,
        practice_size: Optional[int] = None,
    ) -> str:
        graph = graph or self.param_graph
        dir_path = dest_dir or self.param_dir
        os.makedirs(dir_path, exist_ok=True)
        if filename:
            fname = filename if filename.endswith(".json") else f"{filename}.json"
        else:
            size = practice_size or self.burn_in_size
            fname = self._param_filename(student_id, size)
        path = os.path.join(dir_path, fname)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(graph, f, ensure_ascii=False, indent=4)
        if update_runtime or dest_dir is None:
            self.param_path = path
            self.param_graph = graph
            self._param_canonical_path = path
        return path

    def load_param_graph(self, path: Optional[str] = None) -> Dict[str, Any]:
        if path:
            target = path
        else:
            actual = self._actual_param_path(None, self.burn_in_size)
            target = actual if os.path.isfile(actual) else self.param_path
        return self._load_json(target)

    def ensure_actual_dirs(self) -> None:
        os.makedirs(self._actual_student_dir, exist_ok=True)
        os.makedirs(self._actual_param_dir, exist_ok=True)

    def clear_actual_temp_graphs(self) -> None:
        if os.path.isdir(self._actual_student_dir):
            shutil.rmtree(self._actual_student_dir)
        if os.path.isdir(self._actual_param_dir):
            shutil.rmtree(self._actual_param_dir)
        self._student_cache.clear()
        self._practice_size_map = {
            sid: max(sizes) for sid, sizes in self._student_versions.items()
        }
        if os.path.isfile(self._param_canonical_path):
            self.param_graph = self._load_json(self._param_canonical_path)
        self.param_path = self._param_canonical_path

    def _param_filename(self, student_id: Optional[str], practice_size: int) -> str:
        if practice_size <= self.burn_in_size or student_id is None:
            return f"parameter_graph_step_{self.burn_in_size}.json"
        canon = self._canonical_student_id(student_id)
        name = f"parameter_graph_student_{canon}_step_{practice_size}.json"
        if os.path.isfile(os.path.join(self.param_dir, name + ".json")):
            name = name + ".json"
        return name

    def _resolve_param_path_for_student(
        self, student_id: Optional[str], practice_size: int
    ) -> str:
        actual = self._actual_param_path(student_id, practice_size)
        if os.path.isfile(actual):
            return actual

        fname = self._param_filename(student_id, practice_size)
        candidate = os.path.join(self.param_dir, fname)
        if os.path.isfile(candidate):
            return candidate

        shared = os.path.join(self.param_dir, f"parameter_graph_step_{self.burn_in_size}.json")
        if os.path.isfile(shared):
            return shared

        return self._pick_latest_json(self.param_dir)

    # ---------- Root inference ----------
    @staticmethod
    def _infer_root_name(param_graph: Dict[str, Any], student_graph: Dict[str, Any]) -> Optional[str]:
        # 1) Prefer node in parameter graph that has non-None gamma_root
        for k, v in param_graph.items():
            if isinstance(v, dict) and (v.get("gamma_root", None) is not None):
                return v.get("name", k)
        # 2) Else find node in student graph with no parents
        for k, v in student_graph.items():
            if isinstance(v, dict):
                parents = v.get("parents", [])
                if isinstance(parents, list) and len(parents) == 0:
                    return v.get("name", k)
        # 3) Fallback: common root name or first key
        if ROOT_KC in student_graph:
            return ROOT_KC
        return next(iter(student_graph.keys()), None)

    @property
    def root_kc(self) -> Optional[str]:
        return self._root_kc
