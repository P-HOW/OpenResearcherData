import os
from pathlib import Path
from openai import OpenAI

# Robust .env loading (works even if script is run from a different CWD)
# Requires: pip install python-dotenv
try:
    from dotenv import load_dotenv
except ImportError as e:
    raise RuntimeError("Missing dependency: python-dotenv. Run: pip install python-dotenv") from e

# Try common .env locations
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR  # adjust if your .env is higher up
ENV_PATHS = [
    SCRIPT_DIR / ".env",
    PROJECT_ROOT / ".env",
    Path.cwd() / ".env",
]

loaded = False
for p in ENV_PATHS:
    if p.exists():
        load_dotenv(dotenv_path=p, override=False)
        loaded = True
        break

API_KEY = os.environ.get("OPENAI_API_KEY", "").strip()

if not API_KEY:
    # Helpful debug (prints where we looked; does NOT print the key)
    raise RuntimeError(
        "OPENAI_API_KEY is missing. Tried loading .env from:\n"
        + "\n".join(str(x) for x in ENV_PATHS)
        + ("\n\nNo .env file was found." if not loaded else "\n\n.env was found but OPENAI_API_KEY was not set.")
    )

client = OpenAI(
    base_url="https://yinli.one/v1",
    api_key=API_KEY,
)

MODEL_NAME = "claude-3-5-sonnet-20240620"


def _extract_text(msg) -> str:
    content = getattr(msg, "content", None)
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for p in content:
            if isinstance(p, str):
                parts.append(p)
            elif isinstance(p, dict):
                t = p.get("text")
                if isinstance(t, str):
                    parts.append(t)
        return "".join(parts)
    return ""


def _call_compare(univ_a: str, univ_b: str, max_tokens: int) -> str:
    prompt = f"""You are a strict comparator.

Task: Determine whether Institution A and Institution B refer to the same real world institution.

Consider common abbreviations, alternative spellings, translations, and campus or institute naming variants that still refer to the same parent institution.

Output rules:
- If they are the same institution, output exactly: 1
- If they are not the same institution, output exactly: 0
- Output must be a single character: either 1 or 0
- Do not output any other text.

Institution A: "{univ_a}"
Institution B: "{univ_b}"
"""

    completion = client.chat.completions.create(
        model=MODEL_NAME,
        temperature=0,
        max_tokens=max_tokens,
        messages=[
            {"role": "system", "content": "Output only one character: 0 or 1. No other text."},
            {"role": "user", "content": prompt},
        ],
    )

    return _extract_text(completion.choices[0].message).strip()


def same_university(univ_a: str, univ_b: str) -> int:
    if not isinstance(univ_a, str) or not isinstance(univ_b, str):
        raise TypeError("Inputs must be strings.")
    if not univ_a.strip() or not univ_b.strip():
        return 0

    out = _call_compare(univ_a, univ_b, max_tokens=5)
    if not out or out[0] not in ("0", "1"):
        out = _call_compare(univ_a, univ_b, max_tokens=20)

    first = next((ch for ch in out if ch in ("0", "1")), "")
    if first not in ("0", "1"):
        raise ValueError(f"Invalid model output (expected 0/1): {out!r}")

    return int(first)


if __name__ == "__main__":
    a = "King Abdullah University of Science and Technology"
    b = "KAUST"
    print(same_university(a, b))  # expected 1
