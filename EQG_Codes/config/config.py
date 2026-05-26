from dataclasses import dataclass
from typing import Dict, Any, List, Optional
import yaml
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

with open('./config/config.yaml', 'r', encoding='utf-8') as file:
    config = yaml.safe_load(file)


@dataclass
class VerifierProtoConfig:
    train_ratio: float
    seed: int
    max_samples: int | None = None

@dataclass
class VerifierModelConfig:
    encoder_name: str
    proj_dim: int
    freeze_encoder: bool
    max_length: int = 256

@dataclass
class VerifierTrainConfig:
    batch_size: int
    max_epochs: int
    lr: float
    weight_decay: float
    tau: float
    device: str = "cuda"
    log_every: int = 50
    use_hard_negatives: bool = True
    encoder_lr: Optional[float] = None
    wandb: bool = False
    wandb_project: str = "eqg-verifier"
    early_stopping: bool = False
    early_metric: str = "cd_hit3"
    early_mode: str = "max"
    patience: int = 5
    min_delta: float = 0.001

@dataclass
class VerifierInferenceConfig:
    ckpt_path: str

@dataclass
class GeneratorConfig:
    output_dir: str
    csv_output_dir: str

@dataclass
class VerifierConfig:
    dataset: str
    raw_question_info: Dict[str, str]
    tree_dir: str
    output_dir: str
    proto: VerifierProtoConfig
    model: VerifierModelConfig
    train: VerifierTrainConfig
    inference: VerifierInferenceConfig

@dataclass
class KTConfig:
    # KC Graph and Parameter Graph
    dataset_dir: str
    graph_dir: str
    EM_output_dir: str
    KT_output_dir: str
    state_graph_path: str

    # EM
    dataset: str
    root_node: str
    burn_in_size: int
    window_size: int
    early_stopping: bool
    max_step: int
    early_stopping_threshold: float

    # KT
    save_intermediate_graph: bool

    # Parameters Initialization
    initial_r_diff: List[float]
    initial_gamma_root: float
    initial_transition: float
    initial_phi: float
    initial_epsilon: float
    num_sample_students: int = -1  # Number of students to sample for burn-in data
    state_graph_path_root: Optional[str] = None

@dataclass
class GlobalConfig:
    verifier: VerifierConfig | None = None
    KT: KTConfig | None = None
    generator: GeneratorConfig | None = None



def load_config(path: str = "config/config.yaml") -> GlobalConfig:
    with open(path, "r") as f:
        cfg_dict = yaml.safe_load(f)

    # Parse verifier section
    v_cfg_dict = cfg_dict.get("verifier", None)
    if v_cfg_dict:
        proto_dict = v_cfg_dict.get("proto", {})
        proto = VerifierProtoConfig(**proto_dict)
        model_dict = v_cfg_dict.get("model", {})
        model = VerifierModelConfig(**model_dict)
        train_dict = v_cfg_dict.get("train", {})
        train = VerifierTrainConfig(**train_dict)
        inference_dict = v_cfg_dict.get("inference", {})
        inference = VerifierInferenceConfig(**inference_dict)
        verifier_cfg = VerifierConfig(
            dataset=v_cfg_dict["dataset"],
            raw_question_info=v_cfg_dict["raw_question_info"],
            tree_dir=v_cfg_dict["tree_dir"],
            output_dir=v_cfg_dict["output_dir"],
            proto=proto,
            model=model,
            train=train,
            inference=inference,
        )
    else:
        verifier_cfg = None

    # Parse KT section
    kt_cfg_dict = cfg_dict.get("KT", None)
    if kt_cfg_dict:
        if not kt_cfg_dict.get("state_graph_path"):
            root = kt_cfg_dict.get("state_graph_path_root")
            root_node = kt_cfg_dict.get("root_node", "")
            dataset = kt_cfg_dict.get("dataset", "")
            if root:
                module_prefix = root_node.split("_", 1)[0].lower() if root_node else ""
                module_dir = f"{module_prefix}_states" if module_prefix else ""
                if module_dir:
                    root_candidate = root
                    if dataset and not os.path.basename(root_candidate.rstrip("/")) == dataset:
                        root_candidate = os.path.join(root_candidate, dataset)
                    kt_cfg_dict["state_graph_path"] = os.path.join(root_candidate, module_dir)
            if not kt_cfg_dict.get("state_graph_path"):
                raise ValueError("KT.state_graph_path or KT.state_graph_path_root+root_node must be set")
        kt_cfg = KTConfig(**kt_cfg_dict)
    else:
        kt_cfg = None

    # Parse generator section
    gen_cfg_dict = cfg_dict.get("generator", None)
    if gen_cfg_dict:
        gen_cfg = GeneratorConfig(**gen_cfg_dict)
    else:
        gen_cfg = None

    return GlobalConfig(
        verifier=verifier_cfg,
        KT=kt_cfg,
        generator=gen_cfg,
    )
