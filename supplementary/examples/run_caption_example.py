# -*- coding: utf-8 -*-
"""Short runnable example: video -> sampled frames -> per-frame captions -> Test Caption.

Offline mode (default) only shows the sampling plan and writes the frames, so it
runs without an API key. Online mode (--call) sends each frame to any
OpenAI-compatible endpoint configured through OPENAI_BASE_URL / OPENAI_API_KEY /
OPENAI_MODEL, then joins the per-frame descriptions with a single space.
"""
import argparse
import base64
import os
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "utils"))

from video_frames import extract_frames  # noqa: E402

PROMPTS = {
    "zh": "caption_vg_zh.txt",
    "en": "caption_vg_en.txt",
    "jp": "caption_vg_jp.txt",
}


def load_prompt(culture):
    with open(os.path.join(ROOT, "prompts", PROMPTS[culture]), encoding="utf-8") as f:
        return f.read()


def call_model(jpg_path, prompt):
    import requests

    base = os.environ.get("OPENAI_BASE_URL", "").rstrip("/")
    key = os.environ.get("OPENAI_API_KEY", "")
    model = os.environ.get("OPENAI_MODEL", "gpt-5")
    if not base or not key:
        raise SystemExit("set OPENAI_BASE_URL and OPENAI_API_KEY, or drop --call")

    with open(jpg_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("ascii")
    r = requests.post(
        base + "/chat/completions",
        headers={"Authorization": "Bearer " + key, "Content-Type": "application/json"},
        json={
            "model": model,
            "messages": [{"role": "user", "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url",
                 "image_url": {"url": "data:image/jpeg;base64," + b64}},
            ]}],
            "max_completion_tokens": 4096,
            "reasoning_effort": "low",
        },
        timeout=300,
    )
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"].strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--culture", choices=sorted(PROMPTS), default="zh")
    ap.add_argument("--offline", action="store_true",
                    help="only extract frames (default behaviour)")
    ap.add_argument("--call", action="store_true", help="call the model for each frame")
    ap.add_argument("--out-dir", default=None)
    args = ap.parse_args()

    out_dir = args.out_dir or tempfile.mkdtemp(prefix="mugc_frames_")
    duration, times, frames = extract_frames(args.video, out_dir)
    prompt = load_prompt(args.culture)

    print("video        : %s" % args.video)
    print("duration     : %.3f s" % duration)
    print("floor(dur)   : %d" % int(duration))
    print("frame times  : %s" % times)
    print("frames       : %d (JPEG, original resolution)" % len(frames))
    print("prompt file  : prompts/%s" % PROMPTS[args.culture])

    if not args.call:
        print("\nOffline mode: frames extracted; no model call was made.")
        return

    descs = [call_model(p, prompt) for p in frames]
    caption = " ".join(descs)
    print("\nper-frame descriptions: %d" % len(descs))
    print("\nTest Caption:\n%s" % caption)


if __name__ == "__main__":
    main()
