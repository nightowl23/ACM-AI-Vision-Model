#!/usr/bin/env python3
import json
import argparse
from pathlib import Path
from bert_score import score as bert_score
import pandas as pd


def load_jsonl(path: Path):
    records = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    return records


def extract_top_category(path_string: str):
    path_string = path_string.strip("/")
    parts = path_string.split("/")
    return parts[0] if parts else "unknown"


def compute_bertscore(records, answer_field, model_type):
    for r in records:
        if answer_field not in r:
            raise KeyError(
                f"Missing field '{answer_field}'. Found keys: {list(r.keys())}"
            )

    candidates = [r[answer_field] for r in records]
    references = [r["ground truth"] for r in records]

    P, R, F1 = bert_score(
        cands=candidates,
        refs=references,
        lang="en",
        model_type=model_type,
        verbose=True
    )

    results = []
    for rec, p, r, f in zip(records, P, R, F1):
        out = rec.copy()
        out["bertscore_precision"] = float(p)
        out["bertscore_recall"] = float(r)
        out["bertscore_f1"] = float(f)
        out["top_category"] = extract_top_category(rec["path"])
        results.append(out)

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Compute BERTScore for Gemini / Woodpecker outputs"
    )
    parser.add_argument(
        "--gemini-output",
        type=Path,
        required=True,
        help="Path to JSONL outputs file"
    )
    parser.add_argument(
        "--answer-field",
        type=str,
        default="answer",
        choices=["answer", "answer_refined"],
        help="Which answer field to evaluate"
    )
    parser.add_argument(
        "--model-type",
        type=str,
        default="microsoft/deberta-large-mnli",
        help="Encoder model for BERTScore"
    )

    args = parser.parse_args()

    records = load_jsonl(args.gemini_output)
    print(f"Loaded {len(records)} records.")

    results = compute_bertscore(
        records,
        answer_field=args.answer_field,
        model_type=args.model_type
    )

    # Always save evaluation artifacts in evaluation/outputs
    eval_output_dir = Path("evaluation/outputs")
    eval_output_dir.mkdir(parents=True, exist_ok=True)

    suffix = "refined" if args.answer_field == "answer_refined" else "raw"

    output_jsonl = eval_output_dir / f"bertscore_outputs_{suffix}.jsonl"
    summary_csv = eval_output_dir / f"bertscore_summary_{suffix}.csv"
    category_csv = eval_output_dir / f"bertscore_by_top_category_{suffix}.csv"

    with output_jsonl.open("w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    df = pd.DataFrame(results)

    df[
        ["bertscore_precision", "bertscore_recall", "bertscore_f1"]
    ].mean().to_csv(summary_csv, header=["mean_score"])

    df.groupby("top_category")[
        ["bertscore_precision", "bertscore_recall", "bertscore_f1"]
    ].mean().to_csv(category_csv)

    print("\nSaved:")
    print(output_jsonl)
    print(summary_csv)
    print(category_csv)


if __name__ == "__main__":
    main()
