#!/usr/bin/env python3
import json
from pathlib import Path
from tqdm import tqdm

from concept_extraction import extract_entities
from question_generation import generate_questions
from visual_validation import validate_questions
from claim_generation import generate_claims
from refinement import refine_answer

BASE_DIR = Path(__file__).resolve().parent

BINGO_IMAGES = BASE_DIR.parent / "Bingo_benchmark_case"
INPUTS = BASE_DIR.parent / "gemini_pipeline" / "outputs.jsonl"
OUTPUTS = BASE_DIR / "outputs_woodpecker.jsonl"

def main():
    with INPUTS.open() as f, OUTPUTS.open("w") as out:
        for line in tqdm(f, desc="Running Woodpecker"):
            rec = json.loads(line)

            image_path = BINGO_IMAGES / rec["path"].lstrip("/")
            raw_answer = rec["answer"]

            entities = extract_entities(raw_answer)
            questions = generate_questions(entities)
            validations = validate_questions(questions, image_path)
            claims = generate_claims(validations)

            refined = refine_answer(raw_answer, claims)

            out.write(json.dumps({
                **rec,
                "answer_raw": raw_answer,
                "answer_refined": refined
            }, ensure_ascii=False) + "\n")

if __name__ == "__main__":
    main()
