#!/usr/bin/env python3
"""Definitive order-gate trace analysis.

In Mode 2 the agent never calls motion primitives directly. In the
assembly-sequence-discovery phase it logs, per object, a write_assembly_results
call carrying {object_name, assembly_order, tool_sequence:[...]}. The tool_sequence
strings are the primitive calls (e.g. "ros-mcp-server__move_to_grasp(object_name=
'u_brown', grasp_id=1)", "ros-mcp-server__translate_object(action='insert', ...)").
In assembly-execution the orchestrator REPLAYS those logged sequences server-side.

So the per-object primitive ordering the gate would see at runtime == the order of
tool_sequence entries within each object's log, replayed across objects in
assembly_order. We reconstruct that and check, for each gated primitive call
(move_to_grasp / rotate_object / translate_object insert) on object X, whether all
of X's assembly-order predecessors were seated (a successful insert logged) before
that call in the replay order.

Q2 STAGING : a gated call on X while a predecessor (never seated yet) is missing.
Q1 DISTURBED: a predecessor seated then later unseated before a later-object call
              (cannot happen in pure logged-replay since each object's sequence is
              self-contained and inserts are terminal -> reported if seen).
We also flag INTRA-object ordering: within one object's own sequence the gate fires
on its FIRST gated call (move_to_grasp) — predecessors must already be seated from
prior objects. That's the real gate semantics.
"""
import json, glob, os, re
from collections import defaultdict

RUNS = os.path.expanduser("~/Documents/mcp-client-example/.mcp-client-data/ablations/runs")
EVAL = os.path.expanduser("~/Documents/ros-mcp-server/ablations/eval_resources")

order_of = {}; base_order = {}
for p in sorted(glob.glob(os.path.join(EVAL, "fmb*_assembly.json"))):
    j = json.load(open(p)); b = j["base_name"]
    names = [s["object_name"] for s in sorted(j["assembly_order"], key=lambda s: s["position"])]
    base_order[b] = names
    for i, n in enumerate(names):
        order_of[n] = (b, i)

RE_CALL = re.compile(r'(move_to_grasp|rotate_object|translate_object)\s*\(([^)]*)\)')
RE_OBJ = re.compile(r'object_name\s*=\s*[\'"]([^\'"]+)[\'"]')
RE_ACT = re.compile(r'action\s*=\s*[\'"]([^\'"]+)[\'"]')

def all_tool_events(d):
    acc = []
    def walk(n):
        if isinstance(n, dict):
            if "toolName" in n: acc.append(n)
            for v in n.values(): walk(v)
        elif isinstance(n, list):
            for v in n: walk(v)
    walk(d)
    return acc

staging = []      # later-object gated call before a never-seated predecessor
intra = []        # an object's own sequence references a predecessor not yet seated
logged_objs = 0; war_calls = 0; files_with_log = 0; files = 0
per = defaultdict(int)

for study in sorted(glob.glob(os.path.join(RUNS, "mode2_*"))):
    sn = os.path.basename(study)
    for chat in glob.glob(os.path.join(study, "**", "assembly-sequence-discovery", "**", "chat.json"), recursive=True):
        files += 1
        try: d = json.load(open(chat))
        except Exception: continue
        rel = os.path.relpath(chat, RUNS)
        evs = all_tool_events(d)
        # collect the per-object logged sequences (last write per object wins)
        logs = {}   # object_name -> (assembly_order, [tool_sequence strings])
        for ev in evs:
            if (ev.get("toolName") or "").split("__")[-1] != "write_assembly_results":
                continue
            war_calls += 1
            ti = ev.get("toolInput") or {}
            if isinstance(ti, str):
                try: ti = json.loads(ti)
                except Exception: continue
            obj = ti.get("object_name"); ts = ti.get("tool_sequence")
            ao = ti.get("assembly_order")
            if obj and isinstance(ts, list):
                logs[obj] = (ao, ts)
        if not logs: continue
        files_with_log += 1; per[sn] += 1
        # replay objects in assembly_order (by our ground-truth, not the agent's claim)
        objs_sorted = sorted(logs.keys(), key=lambda o: order_of.get(o, ("", 999))[1])
        seated = set()
        for obj in objs_sorted:
            logged_objs += 1
            if obj not in order_of: continue
            b, k = order_of[obj]; order = base_order[b]
            preds = order[:k]
            missing_at_start = [pp for pp in preds if pp not in seated]
            # the gate would fire on this object's first gated call (move_to_grasp etc.)
            ao, ts = logs[obj]
            has_gated = False
            for s in ts:
                for m in RE_CALL.finditer(s):
                    prim = m.group(1); args = m.group(2)
                    mo = RE_OBJ.search(args); ma = RE_ACT.search(args)
                    tobj = mo.group(1) if mo else obj
                    act = ma.group(1) if ma else None
                    gated = prim in ("move_to_grasp", "rotate_object") or (prim == "translate_object" and act == "insert")
                    if gated and tobj == obj:
                        has_gated = True
            if has_gated and missing_at_start:
                staging.append((sn, obj, k + 1, missing_at_start, sorted(seated), rel))
            # mark seated: this object's log existing == it was assembled in this discovery
            seated.add(obj)

out = []
out.append("==== order-gate analysis from logged write_assembly_results tool_sequences ====")
out.append(f"discovery chat.json scanned: {files}  with >=1 logged object: {files_with_log}")
out.append(f"write_assembly_results calls: {war_calls}  logged-object replays: {logged_objs}")
out.append("")
out.append(f"Q2 STAGING hits (object logged a gated primitive while an order-predecessor")
out.append(f"   had NOT been logged/seated yet -> gate WOULD have blocked it): {len(staging)}")
out.append("")
out.append("Interpretation: each hit = a real historical run where the agent assembled an")
out.append("object out of order (predecessor not yet done). If these runs were SUCCESSFUL,")
out.append("the hard-block would have blocked a legitimate maneuver. If they FAILED, the gate")
out.append("would have correctly prevented the wasted attempt.")
out.append("")
out.append("==== per-study discovery files with logs ====")
for s in sorted(per): out.append(f"  {s}: {per[s]}")
out.append("")
out.append("==== STAGING examples (up to 40) ====")
for h in staging[:40]:
    out.append(f"  [{h[0]}] {h[1]} order{h[2]} predecessors-not-yet-seated={h[3]} seated-so-far={h[4]} | {h[5]}")

txt = "\n".join(out)
open("/tmp/order_trace_v3_out.txt", "w").write(txt)
print(txt)
