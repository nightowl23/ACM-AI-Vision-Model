from gemini_utils import call_gemini

PROMPT = """
Rewrite the answer using only the verified visual claims below.
Remove unsupported or hallucinated details.

Original answer:
{answer}

Verified claims:
{claims}
"""

def refine_answer(answer, claims):
    return call_gemini(
        PROMPT.format(
            answer=answer,
            claims="\n".join(claims)
        ),
        max_tokens=512
    )
