from gemini_utils import call_gemini

PROMPT = """
Extract concrete visual entities from the following description.
Only include objects that could be visually verified.
Return a comma-separated list.

Description:
{text}
"""

def extract_entities(answer):
    resp = call_gemini(PROMPT.format(text=answer))
    return [e.strip().lower() for e in resp.split(",") if e.strip()]
