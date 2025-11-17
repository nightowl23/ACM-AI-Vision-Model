#!/usr/bin/env python3
from __future__ import annotations
import argparse
import json
import logging
import os
import time
from itertools import islice
from pathlib import Path
from typing import Dict, Iterable, Iterator

from PIL import Image
from tqdm import tqdm
import google.generativeai as genai


LOGGER = logging.getLogger(__name__)
DEFAULT_MODEL = "gemini-2.0-flash"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Query Gemini with Bingo annotation/image pairs.")
    
    # Use repo root-relative defaults
    repo_root = Path(__file__).parents[1]

    parser.add_argument(
        "--annotations",
        type=Path,
        default=repo_root / "annotation.jsonl",
        help="Path to annotation.jsonl"
    )
    parser.add_argument(
        "--images-root",
        type=Path,
        default=repo_root / "Bingo_benchmark_case",
        help="Path to Bingo image folder"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).parent / "outputs.jsonl",
        help="Output path for Gemini responses"
    )
    parser.add_argument(
        "--api-key-file",
        type=Path,
        default=Path(__file__).parent / "gemini_api_key.txt",
        help="Path to Gemini API key"
    )
    parser.add_argument("--model-name", default=DEFAULT_MODEL)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-output-tokens", type=int, default=256)
    return parser.parse_args()


def load_api_key(path: Path) -> str:
    # ENV var takes priority
    env_key = os.environ.get("GEMINI_API_KEY")
    if env_key:
        return env_key.strip()

    # Fallback to text file
    if path.exists():
        return path.read_text(encoding="utf-8").strip()

    raise RuntimeError("Missing Gemini API key. Set GEMINI_API_KEY or provide gemini_api_key.txt.")


def iter_annotations(path: Path) -> Iterator[Dict]:
    """Yield valid annotation records, skipping malformed entries."""
    with path.open("r", encoding="utf-8") as f:
        for idx, line in enumerate(f, start=1):
            if not line.strip():
                continue

            try:
                record = json.loads(line)
            except Exception:
                LOGGER.warning(f"Skipping line {idx}: invalid JSON")
                continue

            # Skip missing fields — FIXED
            if "path" not in record or "question" not in record or "ground truth" not in record:
                LOGGER.warning(f"Skipping line {idx}: missing fields")
                continue

            yield record


def resolve_image(images_root: Path, rel_path: str) -> Path:
    candidate = images_root / rel_path.lstrip("/")
    if candidate.exists():
        return candidate
    raise FileNotFoundError(f"Image not found: {candidate}")


def run_with_retries(func, max_attempts=5, delay=2):
    for attempt in range(1, max_attempts + 1):
        try:
            return func()
        except Exception as e:
            LOGGER.warning(f"Attempt {attempt} failed: {e}")
            if attempt == max_attempts:
                return f"[ERROR] {e}"
            time.sleep(delay)


def generate_answer(model, question: str, image_path: Path, temp: float, max_tokens: int) -> str:
    with Image.open(image_path) as img:
        rgb = img.convert("RGB")

    def call_model():
        resp = model.generate_content(
            [question, rgb],
            generation_config={
                "temperature": temp,
                "max_output_tokens": max_tokens,
                "top_p": 1.0,
                "top_k": 1,
            },
        )
        resp.resolve()
        return resp.text.strip() if resp.text else "[EMPTY]"

    return run_with_retries(call_model)


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = parse_args()

    api_key = load_api_key(args.api_key_file)
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(args.model_name)

    records = iter_annotations(args.annotations)
    if args.max_samples:
        records = islice(records, args.max_samples)

    # FIXED: always write output in same folder as script
    args.output.parent.mkdir(parents=True, exist_ok=True)

    with args.output.open("w", encoding="utf-8") as sink:
        for entry in tqdm(list(records), desc="Querying Gemini"):
            image_path = resolve_image(args.images_root, entry["path"])

            answer = generate_answer(
                model=model,
                question=entry["question"],
                image_path=image_path,
                temp=args.temperature,
                max_tokens=args.max_output_tokens,
            )

            sink.write(json.dumps({
                "path": entry["path"],
                "question": entry["question"],
                "answer": answer,
                "ground truth": entry["ground truth"],
            }, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
