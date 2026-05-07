"""Probe Gemma 4 thinking-mode toggles on the local llama.cpp server.

Sends the same prompt under several candidate "disable thinking" paths
and reports latency, output length, whether a <think> block appeared,
and whether the API surfaced a separate reasoning_content field. Run
this once to figure out which path actually works on the user's
specific llama.cpp build / model template; the winning path then
becomes the implementation in src/llm/local_client.py.

Usage:
    python scripts/probe_thinking_toggle.py

Reads LOCAL_LLM_BASE_URL / LOCAL_LLM_MODEL from .env (same as the rest
of the project).
"""

from __future__ import annotations

import os
import re
import sys
import time
from pathlib import Path

# Make the project root importable when run from anywhere
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

BASE_URL = os.getenv("LOCAL_LLM_BASE_URL", "http://localhost:8080/v1")
API_KEY = os.getenv("LOCAL_LLM_API_KEY", "not-needed")
MODEL = os.getenv("LOCAL_LLM_MODEL", "google/gemma-4-26b-a4b")

# A prompt that *should* trigger visible reasoning when thinking is on,
# so the difference between on/off is unambiguous. Math word problem is
# the cleanest signal — thinking models will emit a chain when they have it.
PROMPT = (
    "A train leaves City A at 9:00 AM traveling east at 60 km/h. Another "
    "train leaves City B at 10:00 AM traveling west at 80 km/h. The cities "
    "are 350 km apart. At what time will the trains meet? Give the answer "
    "and a one-line explanation."
)

THINK_BLOCK = re.compile(r"<think.*?</think>", re.DOTALL | re.IGNORECASE)


def _run_one(label: str, *, extra_body: dict | None = None,
             user_prefix: str = "") -> dict:
    """Send the prompt with a given thinking-toggle path; return diagnostics."""
    client = OpenAI(base_url=BASE_URL, api_key=API_KEY)
    user_content = (user_prefix + PROMPT).strip()
    kwargs = {
        "model": MODEL,
        "messages": [{"role": "user", "content": user_content}],
        "temperature": 0.2,
        "timeout": 120,
    }
    if extra_body:
        kwargs["extra_body"] = extra_body

    t0 = time.perf_counter()
    try:
        resp = client.chat.completions.create(**kwargs)
    except Exception as exc:
        return {
            "label": label,
            "ok": False,
            "error": f"{type(exc).__name__}: {exc}",
        }
    elapsed = time.perf_counter() - t0

    msg = resp.choices[0].message
    content = msg.content or ""
    # Some servers split thinking out into a separate field. Common names:
    # `reasoning_content` (DeepSeek convention, increasingly mirrored).
    reasoning_field = getattr(msg, "reasoning_content", None)

    has_think_tag = bool(THINK_BLOCK.search(content))
    visible_after_strip = THINK_BLOCK.sub("", content).strip()

    return {
        "label": label,
        "ok": True,
        "elapsed_s": round(elapsed, 2),
        "content_chars": len(content),
        "visible_chars": len(visible_after_strip),
        "has_think_tag_in_content": has_think_tag,
        "reasoning_field_present": bool(reasoning_field),
        "reasoning_field_chars": len(reasoning_field) if reasoning_field else 0,
        "first_120_chars": content[:120].replace("\n", " ⏎ "),
    }


def main() -> int:
    print(f"Endpoint : {BASE_URL}")
    print(f"Model    : {MODEL}")
    print(f"Prompt   : {PROMPT[:80]}...")
    print()

    cases = [
        # 1. Baseline — whatever the server does by default. User reports
        # this currently runs WITH thinking on, so it's our control.
        ("baseline (no toggle)", {}, ""),

        # 2. Qwen3-style chat_template_kwargs — many llama.cpp builds expose
        # this passthrough; the template decides whether to honor it.
        ("chat_template_kwargs.enable_thinking=false",
         {"chat_template_kwargs": {"enable_thinking": False}}, ""),

        # 3. Some templates use 'thinking' instead of 'enable_thinking'.
        ("chat_template_kwargs.thinking=false",
         {"chat_template_kwargs": {"thinking": False}}, ""),

        # 4. OpenAI o-series style (recent llama.cpp added partial support
        # for `reasoning_effort` on some thinking models).
        ("reasoning_effort=none",
         {"reasoning_effort": "none"}, ""),

        # 5. Inline control token. Qwen3 supports `/no_think` at end of
        # user message; some Gemma templates may share this convention.
        ("user-prefix '/no_think'",
         {}, "/no_think\n"),

        # 6. Belt-and-suspenders: combine kwargs + inline.
        ("kwargs.enable_thinking=false + '/no_think' prefix",
         {"chat_template_kwargs": {"enable_thinking": False}}, "/no_think\n"),
    ]

    results = []
    for label, extra, prefix in cases:
        print(f"-- {label} --")
        r = _run_one(label, extra_body=extra or None, user_prefix=prefix)
        results.append(r)
        if not r["ok"]:
            print(f"  ERROR: {r['error']}")
        else:
            print(f"  elapsed: {r['elapsed_s']}s  "
                  f"content: {r['content_chars']}ch  "
                  f"visible: {r['visible_chars']}ch  "
                  f"<think> tag in content: {r['has_think_tag_in_content']}  "
                  f"reasoning_content field: "
                  f"{r['reasoning_field_present']} "
                  f"({r['reasoning_field_chars']}ch)")
            print(f"  start: {r['first_120_chars']}")
        print()

    # Summary table — quick visual to pick the winner.
    print("=" * 78)
    print(f"{'case':<55}  {'sec':>6}  {'think?':>7}  {'reas?':>6}")
    print("-" * 78)
    for r in results:
        if not r["ok"]:
            print(f"{r['label']:<55}  {'ERR':>6}")
            continue
        thought = "YES" if (r["has_think_tag_in_content"]
                            or r["reasoning_field_present"]) else "no"
        print(f"{r['label']:<55}  {r['elapsed_s']:>6}  "
              f"{thought:>7}  "
              f"{r['reasoning_field_chars']:>6}")
    print()
    print("Reading the table:")
    print("  - 'think?' YES  = output contained <think>...</think> OR a")
    print("    separate reasoning_content field came back. Either is")
    print("    the model still reasoning; toggle did NOT disable it.")
    print("  - 'think?' no   = no thinking trace. Toggle worked.")
    print("  - Compare 'sec' between baseline and the winning row to")
    print("    quantify the speed-up worth wiring into fast_query().")
    return 0


if __name__ == "__main__":
    sys.exit(main())
