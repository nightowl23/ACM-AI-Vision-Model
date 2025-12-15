from gemini_utils import call_gemini

PROMPT = """
Generate at most {max_q} simple visual verification questions for the entity "{entity}".
Questions should help verify existence, attributes, or actions.
Each question on a new line.
"""

# ===== Tunable knobs =====
MAX_QUESTIONS_PER_ENTITY = 2
MAX_TOTAL_QUESTIONS = 6

# Cache to avoid repeated Gemini calls for common entities
QUESTION_CACHE = {}


def generate_questions(entities):
    questions = []

    for ent in entities:
        # --- Cache hit ---
        if ent in QUESTION_CACHE:
            qs = QUESTION_CACHE[ent]

        # --- Cache miss: call Gemini once ---
        else:
            resp = call_gemini(
                PROMPT.format(
                    entity=ent,
                    max_q=MAX_QUESTIONS_PER_ENTITY
                )
            )

            # Clean + truncate
            qs = [
                q.strip()
                for q in resp.splitlines()
                if q.strip()
            ][:MAX_QUESTIONS_PER_ENTITY]

            QUESTION_CACHE[ent] = qs

        # Add entity-question pairs
        for q in qs:
            questions.append((ent, q))

        # Hard global cap (very important)
        if len(questions) >= MAX_TOTAL_QUESTIONS:
            break

    return questions[:MAX_TOTAL_QUESTIONS]
