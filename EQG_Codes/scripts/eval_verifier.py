#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Entry script to run global ranking evaluation for the verifier.

Usage:
    # Use default config and default ckpt name (simple_verifier.pt)
    python scripts/eval_verifier.py

    # Use specified ckpt
    python scripts/eval_verifier.py \
        --ckpt ../Verifier/XES3G5M/simple_verifier.pt

    # If you need to specify the config path
    python scripts/eval_verifier.py \
        --config config/config.yaml \
        --ckpt  ../Verifier/XES3G5M/simple_verifier.pt
"""

import os
import sys
import argparse

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

from verifier.global_eval import run_global_eval


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ckpt",
        type=str,
        default=None,
        help="Path to verifier checkpoint (.pt). "
             "If not set, will use the default path in global_eval.py "
             "(.../simple_verifier.pt).",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Optional path to config.yaml. If omitted, load_config() default is used.",
    )

    args = parser.parse_args()

    run_global_eval(ckpt_path=args.ckpt, config_path=args.config)


if __name__ == "__main__":
    main()
