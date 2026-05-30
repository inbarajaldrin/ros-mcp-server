#!/usr/bin/env python3
"""Test harness for real-mode occlusion-tolerant verify_assembly --check-all + the
ported assembly-order gate (server.py + queries/verify_assembly.py).

No hardware / no ROS runtime needed: exercises the real functions with a temp state
store and simulated /objects_poses_real drop-outs. Covers the GPT must-fix list +
the safety property (present-but-pose-wrong is NEVER rescued).

Run on a4500:  source /opt/ros/humble/setup.bash && .venv/bin/python ablations/test_real_checkall_gate.py
"""
import os, sys, json, tempfile, importlib.util

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "queries"))

# State store must point at a temp dir — set BEFORE importing server_p3 (it reads
# MCP_CLIENT_OUTPUT_DIR at import).
_TMP = tempfile.mkdtemp(prefix="realcheckall_test_")
os.environ["MCP_CLIENT_OUTPUT_DIR"] = _TMP

def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m

sp3 = _load("sp3_under_test", os.path.join(REPO, "server.py"))
va = _load("va_under_test", os.path.join(REPO, "queries", "verify_assembly.py"))

STATE = sp3._state_path()
ORDER = ["u_brown", "u_orange", "line_green", "inverted_u_yellow"]  # base1
BASE = "base1"

_fails = []
def check(name, cond):
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}")
    if not cond:
        _fails.append(name)

def _read_state():
    with open(STATE) as f:
        return json.load(f)

# ── A. State store + clear + run_id ───────────────────────────────────────────
print("A. state store / clear / run_id")
r = sp3.clear_primitive_state(run_id="run-A")
check("A1 clear returns run_id run-A", r.get("run_id") == "run-A")
st = _read_state()
check("A1 store = {run_id:run-A, objects:{}}", st == {"run_id": "run-A", "objects": {}})
check("A1 _CURRENT_RUN_ID set", sp3._CURRENT_RUN_ID == "run-A")
sp3._set_object_state("u_brown", assembled=True)
check("A2 certified set == {u_brown}", sp3._run_certified_seated_set() == {"u_brown"})
check("A3 clear wiped prior objects (atomic)", "objects" in _read_state() and "u_brown" in _read_state()["objects"])
# tamper run_id -> mismatch -> untrusted
_st = _read_state(); _st["run_id"] = "run-B"
with open(STATE, "w") as f: json.dump(_st, f)
check("A4 cross-run run_id mismatch -> seated set None", sp3._run_certified_seated_set() is None)

# ── B. Order gate contract (real) ─────────────────────────────────────────────
print("B. order gate (real)")
sp3.clear_primitive_state(run_id="run-A")
check("B1 order-1 object allowed (None)", sp3._assert_assembly_order("u_brown", "real", BASE) is None)
g = sp3._assert_assembly_order("u_orange", "real", BASE)
check("B2 uncertified predecessor -> BLOCK", isinstance(g, dict) and g.get("result") == "failure")
check("B2 message says CERTIFIED + names verify_assembly", g and "CERTIFIED" in g["error"] and "verify_assembly('u_brown')" in g["error"])
sp3._set_object_state("u_brown", assembled=True)
check("B3 predecessor certified -> allow (None)", sp3._assert_assembly_order("u_orange", "real", BASE) is None)
sp3._set_object_state("u_orange", assembled=True)
check("B4 already-seated target -> regrasp allowed (None)", sp3._assert_assembly_order("u_orange", "real", BASE) is None)
# no run certification -> fail closed
with open(STATE, "w") as f: json.dump({"objects": {}}, f)   # no run_id
g = sp3._assert_assembly_order("u_orange", "real", BASE)
check("B5 no run certification -> BLOCK (fail-closed)", isinstance(g, dict) and "no run certification" in g.get("error", ""))
# gate off
sp3.clear_primitive_state(run_id="run-A")
os.environ["MCP_ORDER_GATE"] = "0"
check("B6 MCP_ORDER_GATE=0 -> no-op (None)", sp3._assert_assembly_order("u_orange", "real", BASE) is None)
os.environ["MCP_ORDER_GATE"] = "1"
# sim mode -> gate is real-only
check("B7 sim mode -> gate no-op (None)", sp3._assert_assembly_order("u_orange", "sim", BASE) is None)

# ── C. _load_certified_set (CLI) ──────────────────────────────────────────────
print("C. _load_certified_set (CLI, run_id-scoped, fail-closed)")
with open(STATE, "w") as f:
    json.dump({"run_id": "R", "objects": {"a": {"assembled": True}, "b": {"assembled": False}, "c": {}}}, f)
check("C1 matching run_id -> {a}", va._load_certified_set(STATE, "R") == {"a"})
check("C2 run_id mismatch -> empty (fail-closed)", va._load_certified_set(STATE, "X") == set())
check("C3 missing path -> empty", va._load_certified_set(STATE + ".nope", "R") == set())
check("C3 no run_id arg -> empty", va._load_certified_set(STATE, None) == set())
with open(STATE, "w") as f: f.write("{ this is not json")
check("C4 torn JSON -> empty (fail-closed)", va._load_certified_set(STATE, "R") == set())

# ── D. Occlusion classification (the simulation) ──────────────────────────────
print("D. occlusion classification (_classify_component)")
# Simulated perception: u_brown OCCLUDED (absent), others present.
present = {"u_orange": 1, "line_green": 1, "inverted_u_yellow": 1}  # dict like current_poses; u_brown absent
certified = {"u_brown", "u_orange", "line_green"}                   # what was verified earlier this run
correct = {"u_orange", "line_green"}                               # present AND pose-correct now
def verify_fn(n):  # stand-in for verify_assembly_pose: present+correct -> True, else False
    return (n in present and n in correct), {}

k, resc = va._classify_component("u_brown", present, certified, "real", verify_fn)
check("D1 absent + certified -> assembled via RESCUE", k == "assembled" and resc is True)
k, resc = va._classify_component("u_orange", present, certified, "real", verify_fn)
check("D2 present + correct -> assembled (not rescued)", k == "assembled" and resc is False)
k, resc = va._classify_component("inverted_u_yellow", present, certified, "real", verify_fn)
check("D4 absent + NOT certified -> unassembled (fail-closed)", k == "unassembled" and resc is False)
# CRITICAL safety: present-but-pose-wrong is NEVER rescued even if certified.
present_wrong = {"u_orange": 1}            # u_orange present...
correct_wrong = set()                      # ...but pose WRONG now (knocked loose)
def verify_fn_wrong(n):
    return (n in present_wrong and n in correct_wrong), {}
k, resc = va._classify_component("u_orange", present_wrong, {"u_orange"}, "real", verify_fn_wrong)
check("D5 present-but-wrong + certified -> UNASSEMBLED (disturbance detected, NOT rescued)", k == "unassembled" and resc is False)
# sim mode: rescue never applies
k, resc = va._classify_component("u_brown", present, certified, "sim", verify_fn)
check("D6 sim mode: absent+certified NOT rescued (sim unchanged)", k == "unassembled" and resc is False)

print()
if _fails:
    print(f"RESULT: {len(_fails)} FAILED -> {_fails}")
    sys.exit(1)
print("RESULT: ALL TESTS PASSED")
