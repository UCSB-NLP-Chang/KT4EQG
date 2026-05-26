# cache.py
import json
import os
import sys
from typing import Any, Dict, List

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# from config.config import (
#     graph_dir,
#     dataset_dir,
#     EM_output_dir,
#     burn_in_size,
#     dataset,
#     root_node,
# )
from config.config import load_config
cfg = load_config()
graph_dir = cfg.KT.graph_dir
dataset_dir = cfg.KT.dataset_dir
EM_output_dir = cfg.KT.EM_output_dir
burn_in_size = cfg.KT.burn_in_size
dataset = cfg.KT.dataset
root_node = cfg.KT.root_node

from kt.KC_tree.io import load_graph

global_datas: Dict[str, List[Dict[str, Any]]] = {}
global_graphs = {}
_temp_dataset_cache: Dict[str, List[Dict[str, Any]]] | None = None

recording_path = os.path.join(dataset_dir, dataset, "recordings.jsonl")
question_path = os.path.join(dataset_dir, dataset, "question_info.json")

graph_path = os.path.join(graph_dir, dataset)
test_path = os.path.join(
    f"{dataset_dir}/{dataset}/subtree_split/selected_classroom_students_{root_node}0.65.txt"
)

merged_mapping_path = os.path.join(graph_dir, dataset, "merged_mapping.json")
temp_recordings_path = os.path.join(dataset_dir, dataset, "recordings_temp.json")
_default_temp_recordings_path = temp_recordings_path

translation_mapping = {
    'Application_Module': 'Application Module',
    'Computation_Module': 'Computation Module',
    'Counting_Module': 'Counting Module',
    'Wine_Knowledge': 'Wine Knowledge',
    'Circuit_Design': 'Circuit Design',
    'Education_Theory': 'Education Theory'
}



def get_target_subtree_nodes(root_node):
    if root_node in translation_mapping.keys():
        root_node = translation_mapping[root_node]

    all_kcs = []
    single_graph = load_graph(graph_path, f"pruned_knowledge_graph.json")
    current_nodes = [single_graph[root_node]]

    while True:
        next_nodes = []
        for node in current_nodes:
            all_kcs.extend([child.name for child in node.children])
            next_nodes.extend([child for child in node.children])
        current_nodes = next_nodes
        if len(current_nodes) == 0:
            break
    all_kcs = list(set(all_kcs))

    with open(os.path.join(graph_path, f"subtree/{root_node}_subtree_nodes.json"), "w") as f:
        json.dump(all_kcs, f, indent=4, ensure_ascii=False)
    return all_kcs


def initialize_data(has_graph=True):
    global global_datas
    global global_graphs

    if global_datas and global_graphs:
        return global_datas, global_graphs

    uids = train_uids + test_uids
    uids = list(set(uids))

    # print("Loading data...")

    with open(question_path, "r") as f:
        questions = json.load(f)

    target_subtree_nodes = get_target_subtree_nodes(root_node)

    with open(merged_mapping_path, "r") as f:
        mapping = json.load(f)

    baseline_snapshot: Dict[str, List[Dict[str, Any]]] = {}

    for uid in uids:
        if has_graph:
            EM_output_path = os.path.join(
                EM_output_dir, dataset, f"EM_results-Set{root_node}-burn-in{burn_in_size}"
            )
            if not os.path.exists(
                EM_output_path + "/students_graphs" + f"/E_step_student_{uid}_step_final.json"
            ):
                single_graph = load_graph(graph_path, f"{root_node}_subtree.json")
            else:
                single_graph = load_graph(
                    EM_output_path + "/students_graphs", f"E_step_student_{uid}_step_final.json"
                )
            global_graphs[uid] = single_graph

    with open(recording_path, "r") as f:
        for line in f:
            data = json.loads(line)
            uid = str(data["student_id"])

            if uid not in uids:
                continue

            all_questions: List[Dict[str, Any]] = []
            for index in range(len(data["exercises_logs"])):
                tmp: Dict[str, Any] = {}
                que = data["exercises_logs"][index]
                response = int(data["is_corrects"][index])
                if response == -1:
                    breakpoint()
                if que not in questions.keys():
                    # These are questions not in the three classroom modules we selected (the translated question_info does not contain these questions, but the zh_question_info does)
                    continue
                # assert que in questions.keys()
                question = questions[que]
                tmp["question"] = question["content"]
                tmp["response"] = response
                tmp["kc"] = question["kc"]
                tmp["difficulty"] = question["difficulty"]

                kc = question["kc"]
                if kc in mapping.keys():
                    kc = mapping[kc]
                if kc not in target_subtree_nodes:
                    continue
                all_questions.append(tmp)
            if uid in train_uids:
                # print(
                #     "Filtering out questions not in the train subtree:",
                #     train_size[uid],
                #     len(all_questions),
                # )
                train_size[uid] = len(all_questions)
            else:
                # print(
                #     "Filtering out questions not in the test subtree:",
                #     train_size[uid],
                #     len(all_questions),
                # )
                pass
            limit = min(len(all_questions), train_size.get(uid, len(all_questions)))
            truncated = all_questions[:limit]
            global_datas[uid] = truncated
            baseline_snapshot[uid] = [dict(item) for item in truncated]

    temp_dataset = _load_temp_dataset()
    if temp_dataset:
        merged: Dict[str, List[Dict[str, Any]]] = {
            uid: [dict(item) for item in items]
            for uid, items in baseline_snapshot.items()
        }
        for uid, items in temp_dataset.items():
            snapshot = [dict(item) for item in items]
            merged[uid] = snapshot
            train_size[uid] = len(snapshot)
        global_datas.clear()
        global_datas.update(merged)


def load_data():
    return global_datas, global_graphs


train_uids: List[str] = []
test_uids: List[str] = []
train_size: Dict[str, int] = {}


def initialize_train_uids():
    global train_uids
    global train_size
    global test_uids

    if len(train_size) > 0:
        return global_datas, global_graphs

    train_uids = []
    with open(test_path, "r") as f:
        test_uids = f.readlines()
        test_uids = [uid.strip() for uid in test_uids]

    print(f"Number of train students: {len(train_uids)}")
    print(f"Number of test students: {len(test_uids)}")

    with open(recording_path, "r") as f:
        for line in f:
            data = json.loads(line)
            uid = str(data["student_id"])
            total_length = len(data["exercises_logs"])
            if uid in test_uids:
                train_size[uid] = burn_in_size
            elif uid in train_uids:
                train_size[uid] = total_length


def get_train_uids():
    return test_uids, train_size, train_uids


emission_dict = {}


def get_emission_dict(reset=False):
    global emission_dict
    if reset:
        emission_dict = {}
    return emission_dict


def set_temp_recordings_path(path: str | None) -> str:
    """Override temp recordings path for the current process."""
    global temp_recordings_path, _temp_dataset_cache
    temp_recordings_path = os.path.abspath(path) if path else _default_temp_recordings_path
    _temp_dataset_cache = None
    return temp_recordings_path


def get_temp_recordings_path() -> str:
    return temp_recordings_path


def _load_temp_dataset() -> Dict[str, List[Dict[str, Any]]] | None:
    global _temp_dataset_cache
    if _temp_dataset_cache is not None:
        return _temp_dataset_cache
    if not os.path.isfile(temp_recordings_path):
        return None
    if os.path.getsize(temp_recordings_path) == 0:
        return None
    try:
        with open(temp_recordings_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except json.JSONDecodeError:
        # Corrupted/partial temp file should not break KT update flows.
        return None
    if not isinstance(data, dict):
        return None
    _temp_dataset_cache = data
    return data


def load_full_recordings() -> Dict[str, List[Dict[str, Any]]]:
    """Load all available question-response records per student without truncation."""
    cache = getattr(load_full_recordings, "_cache", None)
    if cache is not None:
        return cache

    initialize_train_uids()

    with open(question_path, "r") as f:
        questions = json.load(f)

    target_subtree_nodes = get_target_subtree_nodes(root_node)

    with open(merged_mapping_path, "r") as f:
        mapping = json.load(f)

    full_records: Dict[str, List[Dict[str, Any]]] = {}

    with open(recording_path, "r") as f:
        for line in f:
            data = json.loads(line)
            uid = str(data["student_id"])

            all_questions: List[Dict[str, Any]] = []
            for index in range(len(data["exercises_logs"])):
                que = data["exercises_logs"][index]
                response = int(data["is_corrects"][index])
                if que not in questions.keys():
                    continue
                question = questions[que]
                kc = question["kc"]
                if kc in mapping.keys():
                    kc = mapping[kc]
                if kc not in target_subtree_nodes:
                    continue
                all_questions.append(
                    {
                        "question": question["content"],
                        "response": response,
                        "kc": kc,
                        "difficulty": question["difficulty"],
                    }
                )
            full_records[uid] = all_questions

    setattr(load_full_recordings, "_cache", full_records)
    return full_records


def _write_temp_dataset(data: Dict[str, List[Dict[str, Any]]]) -> None:
    global _temp_dataset_cache
    os.makedirs(os.path.dirname(temp_recordings_path), exist_ok=True)
    tmp_path = f"{temp_recordings_path}.tmp.{os.getpid()}"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    os.replace(tmp_path, temp_recordings_path)
    _temp_dataset_cache = data


def reset_temp_recordings() -> None:
    global _temp_dataset_cache
    if os.path.isfile(temp_recordings_path):
        os.remove(temp_recordings_path)
    _temp_dataset_cache = None


def append_temp_records(student_id: str, records: List[Dict[str, Any]]) -> None:
    if not records:
        return
    dataset = _load_temp_dataset()
    if dataset is None:
        if not global_datas:
            initialize_data(has_graph=False)
        baseline = {}
        for uid, items in global_datas.items():
            limit = train_size.get(uid, len(items))
            baseline[uid] = [dict(item) for item in items[:limit]]
        dataset = baseline
    else:
        dataset = {uid: [dict(item) for item in items] for uid, items in dataset.items()}

    dataset.setdefault(student_id, [])
    dataset[student_id].extend([dict(r) for r in records])
    _write_temp_dataset(dataset)

    if student_id in global_datas:
        global_datas[student_id].extend([dict(r) for r in records])
    else:
        global_datas[student_id] = [dict(r) for r in records]


def temp_recordings_exists() -> bool:
    return os.path.isfile(temp_recordings_path)


def clear_cached_data() -> None:
    global global_datas, global_graphs
    global_datas.clear()
    global_graphs.clear()
