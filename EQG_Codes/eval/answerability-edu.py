#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Batch classify whether questions are answerable (well-posed) using Qwen3 via vLLM.

Example:
  python qwen3_answerable_batch_csv.py \
    --model Qwen/Qwen3-4B \
    --input_csv /mnt/data/stage2_test_outputs.csv \
    --question_col question_text \
    --outdir output \
    --tp 1
"""

import argparse
import csv
import json
import os
import re
from typing import Any, Dict, List, Optional

from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

# Because the generator is a text model (not multimodal), issues regarding images are ignored
SYSTEM_PROMPT = """You are a strict validator for math/logic/word problems.

Task:
Given ONE question, determine whether it is answerable (well-posed and solvable).

Evaluation Procedure:

Step 1: Assume all referenced charts/images/tables are available and complete.

Step 2: Evaluate whether the question would be answerable under that assumption.

Step 3: Ignore any reasoning like:
- "chart not provided"
- "image missing"
- "cannot see the figure"

Definition:
Answerable = 
- For quantitative/logical problems: has sufficient conditions, unambiguous quantities/entities, and a clear goal leading to a unique (or well-defined) solution.
- For conceptual/educational questions: asks about a recognizable concept, theory, or explanation such that a typical educated respondent can provide a standard, coherent answer.
Not answerable = 
missing key information EVEN IF the chart/image were available, ambiguous references, undefined variables, contradictory constraints, or unclear task.

Output format (MUST be valid JSON, nothing else):
{
  "answerable": true/false,
  "reason": "one short sentence",
  "missing_info": ["..."],         // empty list if none
  "ambiguities": ["..."],          // empty list if none
  "notes": "optional short note"   // can be empty string
}

Rules:
- Do NOT solve the problem.
- Keep fields short.
"""


def ensure_outdir(outdir: str) -> None:
    os.makedirs(outdir, exist_ok=True)


def read_questions_from_csv(path: str, question_col: Optional[str] = None) -> List[str]:
    """
    Reads questions from a CSV file.
    - If question_col is provided, use it.
    - Otherwise, auto-detect from common names or fall back to the first column.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"CSV not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f"CSV has no header row: {path}")

        headers = [h.strip() for h in reader.fieldnames if h is not None]

        # Auto-detect if not provided
        if question_col is None:
            common = [
                "question_text",  # <-- your attached CSV uses this
                "question",
                "prompt",
                "text",
                "problem",
                "query",
                "input",
            ]
            header_lower = {h.lower(): h for h in headers}
            for c in common:
                if c.lower() in header_lower:
                    question_col = header_lower[c.lower()]
                    break
            if question_col is None:
                question_col = headers[0]  # fallback

        if question_col not in headers:
            raise ValueError(
                f"--question_col '{question_col}' not found in CSV headers.\n"
                f"Available columns: {headers}"
            )

        questions: List[str] = []
        for row in reader:
            q = (row.get(question_col) or "").strip()
            if q:
                questions.append(q)

    if not questions:
        raise ValueError(f"No non-empty questions found in column '{question_col}' of {path}")

    return questions


def extract_json_object(text: str) -> Optional[Dict[str, Any]]:
    text = text.strip()

    try:
        obj = json.loads(text)
        if isinstance(obj, dict) and "answerable" in obj:
            return obj
    except Exception:
        pass

    m = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not m:
        return None

    candidate = m.group(0).strip()
    try:
        obj = json.loads(candidate)
        if isinstance(obj, dict) and "answerable" in obj:
            return obj
    except Exception:
        return None

    return None


def build_chat_prompt(tokenizer, question: str) -> str:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]

    # Best-effort disable thinking for cleaner JSON
    try:
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
    except TypeError:
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

    return prompt


def normalize_decision(obj: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}

    out["answerable"] = bool(obj.get("answerable", False))

    reason = obj.get("reason", "")
    out["reason"] = str(reason)[:500].strip()

    missing = obj.get("missing_info", [])
    if not isinstance(missing, list):
        missing = [str(missing)]
    out["missing_info"] = [str(x)[:300].strip() for x in missing][:20]

    amb = obj.get("ambiguities", [])
    if not isinstance(amb, list):
        amb = [str(amb)]
    out["ambiguities"] = [str(x)[:300].strip() for x in amb][:20]

    notes = obj.get("notes", "")
    out["notes"] = str(notes)[:500].strip()

    return out


def write_result_file(
    outpath: str,
    idx: int,
    question: str,
    raw_text: str,
    parsed: Optional[Dict[str, Any]],
) -> None:
    with open(outpath, "w", encoding="utf-8") as f:
        f.write(f"## index\n{idx}\n\n")
        f.write("## question\n")
        f.write(question + "\n\n")

        f.write("## raw_model_output\n")
        f.write(raw_text.strip() + "\n\n")

        f.write("## parsed_json\n")
        if parsed is None:
            f.write("null\n")
        else:
            f.write(json.dumps(parsed, ensure_ascii=False, indent=2) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-4B",
                        help="HF model name or local path (use an Instruct checkpoint).")
    parser.add_argument("--input_csv", type=str, required=True,
                        help="Path to CSV file containing questions.")
    parser.add_argument("--question_col", type=str, default=None,
                        help="Column name for questions. If omitted, auto-detect.")
    parser.add_argument("--outdir", type=str, default="output",
                        help="Output directory for per-question files.")
    parser.add_argument("--tp", type=int, default=1,
                        help="Tensor parallel size (number of GPUs to use).")
    parser.add_argument("--max_model_len", type=int, default=4096,
                        help="Max context length used by vLLM engine.")
    parser.add_argument("--max_tokens", type=int, default=256,
                        help="Max tokens to generate per question.")
    args = parser.parse_args()

    ensure_outdir(args.outdir)

    questions = read_questions_from_csv(args.input_csv, args.question_col)

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

    llm = LLM(
        model=args.model,
        tensor_parallel_size=args.tp,
        max_model_len=args.max_model_len,
        trust_remote_code=True,
    )

    sampling = SamplingParams(
        temperature=0.0,
        top_p=1.0,
        max_tokens=args.max_tokens,
        stop_token_ids=[tokenizer.eos_token_id] if tokenizer.eos_token_id is not None else None,
    )

    valid = 0
    invalid = 0

    prompts: List[str] = [build_chat_prompt(tokenizer, q) for q in questions]
    outputs = llm.generate(prompts, sampling)

    for i, (q, out) in enumerate(zip(questions, outputs), start=1):
        raw = out.outputs[0].text if out.outputs else ""
        obj = extract_json_object(raw)
        parsed = normalize_decision(obj) if obj is not None else None

        if parsed is not None and parsed.get("answerable") is True:
            valid += 1
        else:
            invalid += 1

        outpath = os.path.join(args.outdir, f"{i}.txt")
        write_result_file(outpath, i, q, raw, parsed)

    print(f"Total: {len(questions)}")
    print(f"Valid/answerable: {valid}")
    print(f"Invalid/not answerable: {invalid}")


if __name__ == "__main__":
    main()