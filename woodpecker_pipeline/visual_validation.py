from gemini_utils import call_gemini

VERIFY_PROMPT = """
Answer the question using only the image.
Respond briefly.

Question:
{question}
"""

def validate_questions(questions, image_path):
    results = []
    for entity, question in questions:
        ans = call_gemini(
            VERIFY_PROMPT.format(question=question),
            image_path=image_path,
            max_tokens=32
        )
        results.append((entity, question, ans))
    return results
