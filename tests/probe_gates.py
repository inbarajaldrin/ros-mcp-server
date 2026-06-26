#!/usr/bin/env python3
"""NO-LLM, live-ROS probe for the MCP server's state-truth gates.

Exercises the SAME gate handlers the MCP server runs (triggers.commit_object.handle_object_commit
and triggers.signal_phase_complete.handle_phase_signal), but driven directly from the CLI with
subprocess-backed query callbacks — no MCP client, no YAML study harness, no LLM. It lets you
check the commit and phase-complete gates against a live sim/real scene by hand.

It deliberately MIRRORS server_core.py's wiring (same query scripts, same flags):
  - commit  G1 verify_assembly(object)   G2 verify_grasp(release)   G3 is_at_safe_height
            (G4 checkpoint is OFF here, matching MCP_CHECKPOINT_GATE unset)
  - signal  _phase_complete = verify_*(check_all) + is_home
The grasp gate is grasp-aware in sim: it passes --base-name + --grasp-id so verify_grasp checks
the gripper width against THAT grasp's approach width (each grasp_id has its own width). Real
keeps --width-only (the full real grasp check additionally needs object orientation).

If server_core's wiring changes, update the callbacks below to match — that drift is the point
of keeping this probe close to the real flow.

Usage:
  # commit a single object (assembly phase 2):
  python3 tests/probe_gates.py --tool commit --object u_green --base base2 --grasp-id 1 --phase 2 --mode sim

  # signal a phase complete:
  python3 tests/probe_gates.py --tool signal --base base2 --phase 2 --mode sim \
      --status success --comment "all assembled, arm home"

  # disassembly is phase 1 (verify_disassembly is used automatically):
  python3 tests/probe_gates.py --tool commit --object u_green --base base2 --grasp-id 1 --phase 1 --mode sim

Env: ROS_DOMAIN_ID defaults to 0 (override with --domain); ROS_MCP_MODE is set from --mode so the
primitives/queries bind the right sim/real config.
"""
import argparse
import asyncio
import json
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from triggers.commit_object import handle_object_commit
from triggers.signal_phase_complete import handle_phase_signal
from primitives.shared.predicates import is_object_grasped, is_at_safe_height, is_home


def _run_query(script: str, args: str, *, domain: str, mode: str, timeout: int = 60) -> dict:
    """Run a queries/*.py script and parse the JSON between the __RESULT_JSON__ markers.

    Mirrors server_core._run_query: shell invocation, marker extraction, fail-closed dict.
    PYTHONPATH is PREPENDED (not replaced) so the ROS2 rclpy path in the ambient env survives.
    """
    cmd = f"{sys.executable} {ROOT}/queries/{script} {args}"
    env = dict(os.environ, ROS_DOMAIN_ID=str(domain), ROS_MCP_MODE=mode,
               PYTHONPATH=ROOT + ":" + os.environ.get("PYTHONPATH", ""))
    try:
        p = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=timeout, env=env)
    except subprocess.TimeoutExpired:
        return {"result": "failure", "error": f"{script} timeout (>{timeout}s)"}
    out = p.stdout
    if out and "__RESULT_JSON__" in out and "__END_RESULT_JSON__" in out:
        s = out.index("__RESULT_JSON__") + len("__RESULT_JSON__")
        e = out.index("__END_RESULT_JSON__")
        try:
            return json.loads(out[s:e].strip())
        except json.JSONDecodeError as ex:
            return {"result": "failure", "error": f"parse {script}: {ex}"}
    return {"result": "failure", "error": f"no result marker from {script}", "stderr": p.stderr[-300:]}


def make_callbacks(domain: str, mode: str, base: str, grasp_id):
    """Build the verify/grasp/state callbacks, wired exactly like server_core's injected ones."""

    def verify_assembly_obj(base_name, m, object_name=None):
        if object_name:
            return _run_query("verify_assembly.py",
                              f"--object-name {object_name} --base-name {base_name} --mode {m}",
                              domain=domain, mode=mode)
        return _run_query("verify_assembly.py", f"--base-name {base_name} --mode {m} --check-all",
                          domain=domain, mode=mode)

    def verify_disassembly_obj(base_name, m, object_name=None):
        if object_name:
            return _run_query("verify_disassembly.py",
                              f"--object-name {object_name} --base-name {base_name} --mode {m}",
                              domain=domain, mode=mode)
        return _run_query("verify_disassembly.py", f"--base-name {base_name} --mode {m} --check-all",
                          domain=domain, mode=mode)

    def verify_grasp_held(object_name, m):
        # sim: grasp-aware (base scopes the assembly dataset, grasp_id picks the approach width).
        # real: width-only (full real grasp check needs object orientation, unavailable here).
        if m == "sim":
            return _run_query("verify_grasp.py",
                              f"--object-name {object_name} --base-name {base} --mode sim --grasp-id {grasp_id}",
                              domain=domain, mode=mode)
        return _run_query("verify_grasp.py", f"--object-name {object_name} --mode real --width-only",
                          domain=domain, mode=mode)

    def robot_state(m):
        return _run_query("get_robot_state.py", f"--mode {m}", domain=domain, mode=mode, timeout=15)

    return verify_assembly_obj, verify_disassembly_obj, verify_grasp_held, robot_state


def run_commit(args):
    va, vd, vg, rs = make_callbacks(args.domain, args.mode, args.base, args.grasp_id)
    if args.grasp_id is None and args.mode == "sim":
        print("ERROR: --grasp-id is required for a sim commit (the release gate checks the "
              "gripper width against THAT grasp's approach width).", file=sys.stderr)
        sys.exit(2)
    res = handle_object_commit(
        object_name=args.object, phase=args.phase, mode=args.mode, base_name=args.base,
        verify_assembly_fn=va, verify_disassembly_fn=vd, verify_grasp_fn=vg, robot_state_fn=rs,
    )
    return res


def run_home(args):
    """MIRRORS server_core._assert_home_preconditions (the move_home pre-gate):
    empty (verify_grasp --width-only, explicit-verdict fail-closed) AND (at safe height OR
    already home). Same query flags, same predicate calls, same refusal logic — drift here
    means the probe needs updating, which is the point."""
    _va, _vd, _vg, rs = make_callbacks(args.domain, args.mode, args.base, args.grasp_id)
    gres = _run_query("verify_grasp.py", f"--object-name any --mode {args.mode} --width-only",
                      domain=args.domain, mode=args.mode, timeout=15)
    if not (isinstance(gres, dict) and gres.get("result") in ("success", "failure")):
        return {"result": "failure", "error": "Move-home gate: grasp check unavailable. Retry.",
                "leg": "empty(unavailable)"}
    holding, why = is_object_grasped(gres)
    if holding:
        return {"result": "failure", "leg": "empty", "verdict": why,
                "error": "Move-home gate: the gripper is not empty (holding a part or at a "
                         "partial width)."}
    state = rs(args.mode)
    if not isinstance(state, dict):
        return {"result": "failure", "leg": "state", "error": "Move-home gate: robot state unavailable."}
    at_safe, safe_why = is_at_safe_height(state)
    if not at_safe:
        home_ok, home_why = is_home(state)
        if not home_ok:
            return {"result": "failure", "leg": "safe_height", "verdict": f"{safe_why} / {home_why}",
                    "error": "Move-home gate: arm is not at safe height."}
        return {"result": "success", "leg": "home", "verdict": home_why, "empty_verdict": why}
    return {"result": "success", "leg": "safe_height", "verdict": safe_why, "empty_verdict": why}


def run_signal(args):
    va, vd, _vg, rs = make_callbacks(args.domain, args.mode, args.base, args.grasp_id)
    res = asyncio.run(handle_phase_signal(
        phase=args.phase, status=args.status, comment=args.comment, ctx=None, mode=args.mode,
        base_name=args.base, verify_assembly_fn=va, verify_disassembly_fn=vd, robot_state_fn=rs,
    ))
    return res


def main():
    p = argparse.ArgumentParser(description="No-LLM live-ROS probe for commit / signal_phase gates.")
    p.add_argument("--tool", required=True, choices=["commit", "signal", "home"])
    p.add_argument("--mode", required=True, choices=["sim", "real"])
    p.add_argument("--phase", type=int, default=2, choices=[1, 2, 3], help="1=disassembly, 2/3=assembly")
    p.add_argument("--base", default="", help="assembly base name (e.g. base2)")
    p.add_argument("--object", default="", help="object name (commit only)")
    p.add_argument("--grasp-id", type=int, default=None, help="grasp id used to grasp the object (commit/sim)")
    p.add_argument("--status", default="success", choices=["success", "failure"], help="agent-reported status (signal)")
    p.add_argument("--comment", default="", help="agent comment (signal)")
    p.add_argument("--domain", default="0", help="ROS_DOMAIN_ID (default 0)")
    args = p.parse_args()

    if args.tool == "commit" and not args.object:
        p.error("--object is required for --tool commit")

    res = {"commit": run_commit, "signal": run_signal, "home": run_home}[args.tool](args)
    print(json.dumps(res, indent=2))
    # Exit nonzero when a gate failed, so the probe is usable in scripts/CI.
    ok = res.get("result") == "success" or res.get("status") == "success"
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
