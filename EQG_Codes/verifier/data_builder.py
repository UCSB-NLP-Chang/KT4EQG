import json
import os
import sys
from dataclasses import dataclass
from typing import Dict, List, Tuple
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.config import VerifierConfig, load_config
import random

@dataclass
class QuestionRecord:
    qid: str
    text: str
    concept_name: str


def load_question_info(path: str) -> List[QuestionRecord]:
    """Load raw question_info.json and convert to a flat list."""
    with open(path, "r") as f:
        raw = json.load(f)

    records: List[QuestionRecord] = []
    for qid, info in raw.items():
        content = info.get("content", "").strip()
        kc = info.get("kc", "").strip()

        # Simple filter: skip if missing fields or empty text
        if not content or not kc:
            continue

        records.append(
            QuestionRecord(
                qid=str(qid),
                text=content,
                concept_name=kc,
            )
        )
    return records


def build_concept_vocab(records: List[QuestionRecord]) -> Dict[str, int]:
    concepts = sorted({r.concept_name for r in records})
    return {name: idx for idx, name in enumerate(concepts)}


def split_train_val(
    records: List[QuestionRecord],
    train_ratio: float,
    seed: int,
) -> Tuple[List[QuestionRecord], List[QuestionRecord]]:
    random.Random(seed).shuffle(records)
    n_train = int(len(records) * train_ratio)
    train = records[:n_train]
    val = records[n_train:]
    return train, val


def dump_jsonl(
    records: List[QuestionRecord],
    concept2id: Dict[str, int],
    out_path: str,
):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for r in records:
            item = {
                "qid": r.qid,
                "text": r.text,
                "concept_name": r.concept_name,
                "concept_id": concept2id[r.concept_name],
            }
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def dump_vocab_files(
    concept2id: Dict[str, int],
    output_dir: str,
):
    os.makedirs(output_dir, exist_ok=True)
    concept_vocab_path = os.path.join(output_dir, "concept_vocab.json")

    with open(concept_vocab_path, "w", encoding="utf-8") as f:
        json.dump(concept2id, f, ensure_ascii=False, indent=2)


def build_verifier_proto_data(vcfg: VerifierConfig):
    dataset = vcfg.dataset
    raw_path = vcfg.raw_question_info[dataset]
    out_dir = os.path.join(vcfg.output_dir, dataset)

    print(f"[Verifier] Building proto data for dataset={dataset}")
    print(f"  Raw question_info: {raw_path}")
    print(f"  Output dir       : {out_dir}")

    records = load_question_info(raw_path)
    print(f"[Verifier] Loaded {len(records)} raw questions")

    # Truncate total samples for small experiments (optional)
    if vcfg.proto.max_samples is not None and len(records) > vcfg.proto.max_samples:
        records = records[:vcfg.proto.max_samples]
        print(f"[Verifier] Truncated to max_samples={vcfg.proto.max_samples}")

    if not records:
        raise ValueError("[Verifier] No records left after filtering!")

    concept2id = build_concept_vocab(records)
    print(f"[Verifier] Found {len(concept2id)} concepts")

    train_recs, val_recs = split_train_val(
        records, vcfg.proto.train_ratio, vcfg.proto.seed
    )
    print(f"[Verifier] Train size={len(train_recs)}, Val size={len(val_recs)}")

    # Output jsonl
    train_path = os.path.join(out_dir, "verifier_proto_train.jsonl")
    val_path = os.path.join(out_dir, "verifier_proto_val.jsonl")
    dump_jsonl(train_recs, concept2id, train_path)
    dump_jsonl(val_recs, concept2id, val_path)
    print(f"[Verifier] Dumped train to {train_path}")
    print(f"[Verifier] Dumped val   to {val_path}")

    # Output vocab
    dump_vocab_files(concept2id, out_dir)
    print(f"[Verifier] Dumped vocab files to {out_dir}")


if __name__ == "__main__":
    cfg = load_config()
    if cfg.verifier is None:
        raise RuntimeError("No verifier config found in configs/config.yaml")
    build_verifier_proto_data(cfg.verifier)
