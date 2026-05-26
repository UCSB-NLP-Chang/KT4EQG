#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Entry script to train the verifier.

Usage:
    cd EQG_Codes
    python scripts/train_verfier.py
"""

import os
import sys
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

from verifier.train import train_verifier


def main():
    """
    Thin wrapper around verifier.train.train_verifier().
    All hyperparameters and dataset are read from configs/config.yaml.
    """
    train_verifier()


if __name__ == "__main__":
    main()
