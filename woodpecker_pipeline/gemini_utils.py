import os
from pathlib import Path
import google.generativeai as genai
from PIL import Image

MODEL = "gemini-2.0-flash"

def load_api_key():
    # Priority 1: environment variable
    if "GEMINI_API_KEY" in os.environ:
        return os.environ["GEMINI_API_KEY"].strip()

    # Priority 2: shared key file (reuse same one as gemini_pipeline)
    key_path = Path(__file__).resolve().parent.parent / "gemini_pipeline" / "gemini_api_key.txt"
    if key_path.exists():
        return key_path.read_text().strip()

    raise RuntimeError("Gemini API key not found")

# 🔑 CONFIGURE GEMINI ON IMPORT
genai.configure(api_key=load_api_key())

_model = genai.GenerativeModel(MODEL)

def call_gemini(prompt, image_path=None, max_tokens=512):
    if image_path:
        img = Image.open(image_path).convert("RGB")
        resp = _model.generate_content(
            [prompt, img],
            generation_config={
                "temperature": 0.0,
                "max_output_tokens": max_tokens,
                "top_p": 1.0,
                "top_k": 1,
            },
        )
    else:
        resp = _model.generate_content(
            prompt,
            generation_config={
                "temperature": 0.0,
                "max_output_tokens": max_tokens,
                "top_p": 1.0,
                "top_k": 1,
            },
        )

    resp.resolve()
    return resp.text.strip() if resp.text else ""
