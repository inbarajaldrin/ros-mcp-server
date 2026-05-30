#!/usr/bin/env python3
"""gen_simverify.py — derive `.simverify.json` ground-truth from raw `_results.json`.

WHAT IT DOES: copies each `Assembly_fmb_assembly_<n>_results.json` to
`..._results.simverify.json` with one transform — strips the `object_name=...` argument
from every `move_to_safe_height(...)` call in the per-object tool_sequences. Everything else
is byte-identical.

WHY (2026-05-30, Track B): `move_to_safe_height(object_name='X')` computes a retreat height
relative to that object's CURRENT pose; after the part is grasped/moved the object-relative
query goes bad in sim and produces an unreachable target. The bare `move_to_safe_height()`
uses a fixed global safe height, which replays cleanly. fmb1 shipped a hand-made
`.simverify.json` with this edit; this script reproduces it for any assembly. The
`verify_replay_fmbN_sim` definitions point their replay-data copy at the `.simverify.json`.

USAGE (a4500):
    ~/env_isaaclab/bin/python gen_simverify.py            # assemblies 1 2 3
    ~/env_isaaclab/bin/python gen_simverify.py 2 3        # only fmb2, fmb3
    GT_DIR=/path/to/ground_truth_resources python gen_simverify.py 2

Idempotent: re-running just regenerates the .simverify.json from the raw file.
"""
import json, re, sys, os

GT = os.environ.get(
    "GT_DIR",
    os.path.expanduser("~/Documents/ros-mcp-server/ablations/ground_truth_resources"),
)
NUMS = [int(a) for a in sys.argv[1:]] or [1, 2, 3]

# strip `object_name=<...>, ` (the first arg) from move_to_safe_height(...) only
PAT = re.compile(r"(move_to_safe_height\()object_name=([^,]+),\s*")


def fix_step(s):
    return PAT.sub(r"\1", s)


for n in NUMS:
    raw_p = os.path.join(GT, "Assembly_fmb_assembly_%d_results.json" % n)
    sv_p = os.path.join(GT, "Assembly_fmb_assembly_%d_results.simverify.json" % n)
    if not os.path.exists(raw_p):
        print("fmb%d: raw not found at %s — skipping" % (n, raw_p))
        continue
    d = json.load(open(raw_p))
    nchg = 0
    for obj in d.get("assembly_order", []):
        seq = obj.get("tool_sequence", [])
        for i, step in enumerate(seq):
            ns = fix_step(step)
            if ns != step:
                nchg += 1
                seq[i] = ns
    json.dump(d, open(sv_p, "w"), indent=2)
    print("fmb%d: wrote %s (stripped object_name from %d move_to_safe_height calls)"
          % (n, os.path.basename(sv_p), nchg))
