# -*- coding: utf-8 -*-
"""Short runnable example: one reference caption + one Test Caption -> score record.

Offline mode (default) reads examples/cached_score_response.txt, so it runs
without an API key. Online mode calls any OpenAI-compatible endpoint configured
through OPENAI_BASE_URL / OPENAI_API_KEY / OPENAI_MODEL.
"""
import argparse
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "utils"))

from parse_score import parse_score, validate  # noqa: E402

QUOTE = re.compile("[\"\u201c\u201d\u300c\u300d]" )


def load_json(name):
    with open(os.path.join(HERE, name), encoding="utf-8") as f:
        return json.load(f)


def build_prompt(ref_text, test_text):
    path = os.path.join(ROOT, "prompts", "score_judge.txt")
    with open(path, encoding="utf-8") as f:
        tpl = f.read()
    return tpl.format(gold_caption=ref_text, test_caption=test_text)


def call_model(prompt):
    import requests

    base = os.environ.get("OPENAI_BASE_URL", "").rstrip("/")
    key = os.environ.get("OPENAI_API_KEY", "")
    model = os.environ.get("OPENAI_MODEL", "gpt-5")
    if not base or not key:
        raise SystemExit("set OPENAI_BASE_URL and OPENAI_API_KEY, or use --offline")
    r = requests.post(
        base + "/chat/completions",
        headers={"Authorization": "Bearer " + key, "Content-Type": "application/json"},
        json={
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "max_completion_tokens": 2048,
            "reasoning_effort": "low",
        },
        timeout=300,
    )
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--offline", action="store_true",
                    help="use the cached model response (default behaviour)")
    ap.add_argument("--call", action="store_true", help="call the model")
    args = ap.parse_args()

    ref = load_json("sample_ref.json")
    test = load_json("sample_test.json")
    prompt = build_prompt(ref["gold_caption"], test["test_caption"])

    if args.call:
        response = call_model(prompt)
    else:
        with open(os.path.join(HERE, "cached_score_response.txt"), encoding="utf-8") as f:
            response = f.read()

    text_applicable = bool(QUOTE.search(ref["gold_caption"]) or QUOTE.search(test["test_caption"]))
    rec = parse_score(response, ref["sample_id"], text_applicable=text_applicable)
    validate(rec)

    print("prompt length: %d chars" % len(prompt))
    print(json.dumps(rec, ensure_ascii=False, indent=2))
    print("\nExplanation: text_applicable=%s -> text_score is %s in the holistic total."
          % (rec["text_applicable"], "included" if rec["text_applicable"] else "excluded"))


if __name__ == "__main__":
    main()
