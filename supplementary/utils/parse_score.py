# -*- coding: utf-8 -*-
"""Parse the fixed key-value scoring block into a JSON record.

The scoring model returns text such as:

    score output format:
    background_score: 100
    object_score: 92
    text_score: 0
    culture_score: 95
    total_score: 95

parse_score() converts it into the record described by
schema/score_record.schema.json.
"""
import re

FIELDS = ["background_score", "object_score", "text_score",
          "culture_score", "total_score"]


def _grab(text, field):
    m = re.search(field + r"\s*[:：]\s*(-?\d+(?:\.\d+)?)", text or "")
    if not m:
        raise ValueError("missing field: %s" % field)
    return float(m.group(1))


def parse_score(response_text, sample_id="unknown", text_applicable=None):
    """Return a dict matching schema/score_record.schema.json."""
    rec = {"sample_id": sample_id}
    for f in FIELDS:
        v = _grab(response_text, f)
        if not 0 <= v <= 100:
            raise ValueError("%s out of range: %s" % (f, v))
        rec[f] = int(v) if v.is_integer() else v
    if text_applicable is not None:
        rec["text_applicable"] = bool(text_applicable)
    return rec


def validate(rec):
    """Minimal validation without external dependencies."""
    missing = [f for f in ["sample_id"] + FIELDS if f not in rec]
    if missing:
        raise ValueError("missing keys: %s" % missing)
    for f in FIELDS:
        if not 0 <= float(rec[f]) <= 100:
            raise ValueError("%s out of range" % f)
    return True


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--response", required=True, help="path to a text file with the model response")
    ap.add_argument("--sample-id", default="unknown")
    args = ap.parse_args()
    with open(args.response, encoding="utf-8") as f:
        txt = f.read()
    rec = parse_score(txt, args.sample_id)
    validate(rec)
    import json
    print(json.dumps(rec, ensure_ascii=False, indent=2))
