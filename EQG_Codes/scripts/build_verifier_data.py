# scripts/build_verifier_data.py
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.config import load_config
from verifier.data_builder import build_verifier_proto_data

if __name__ == "__main__":
    cfg = load_config()
    if cfg.verifier is None:
        raise RuntimeError("No verifier config found in configs/config.yaml")
    build_verifier_proto_data(cfg.verifier)
