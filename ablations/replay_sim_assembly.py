#!/usr/bin/env python3
"""Replay an assembly through the cell's simulation primitives.

By default each part's passes are derived, as in replay_real_assembly's
--derive-sequence: every part is observed after the startup and the cell's
sequence planner chooses one or two passes.  Ruling A (lead, 2026-11-19): each
part is scored on its final pose after release and retract
(ablations/sim_seat_scorer.py) before the next part starts, and every seated
part is re-scored at the end of the run.

This is the standalone entrance for the same sim tool sequence that
``replay_verify.py`` invokes through unified_api.  It deliberately refuses
real mode: a simulator process must serve the ROS contract and this process
only starts the existing primitive subprocesses against that contract.  Before the
first command it proves it is inside a DroidStack simulator (ablations/sim_guard.py)
and refuses otherwise; every step it runs is a mode-sim primitive.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
from pathlib import Path
import subprocess
import sys

from ablations.replay_verify import _parse_tool_call
from ablations.sim_guard import require_sim_container


ROOT = Path(__file__).resolve().parents[1]


def _command_for_step(step: str, object_name: str | None = None,
                      grasp_id: int | None = None) -> tuple[list[str], int]:
    name, kwargs = _parse_tool_call(step)
    if kwargs.get("mode") != "sim":
        raise ValueError(f"standalone sim replay refuses non-sim step: {step}")
    short = name.removeprefix("ros_mcp_server__")
    py = [sys.executable, "-u", "-m"]
    if short == "move_home":
        return py + ["primitives.move_home", "--mode", "sim", "--joint-space"], 120
    if short == "control_gripper":
        cmd = py + ["primitives.control_gripper", str(kwargs["command"]), "--mode", "sim"]
        if str(kwargs["command"]).lower() == "close" and object_name and grasp_id is not None:
            from primitives.shared.config import grip_width_mm
            span_mm = grip_width_mm(object_name, grasp_id=grasp_id)
            cmd += ["--expect", object_name, "--expect-span-mm", str(span_mm), "--force", "40"]
        return cmd, 90
    if short == "move_to_grasp":
        # --base-name lets move_to_grasp pick the wrist whose reorient keeps the camera away from the
        # table, exactly as the real replay does (CLAUDE.md "Camera away from the table").
        return py + ["primitives.move_to_grasp", "--object-name", kwargs["object_name"],
                     "--grasp-id", str(kwargs["grasp_id"]), "--mode", "sim", "--pad-center",
                     "--base-name", "base1"], 120
    if short == "move_to_safe_height":
        cmd = py + ["primitives.move_to_safe_height", "--mode", "sim"]
        if kwargs.get("object_name"):
            cmd += ["--object-name", kwargs["object_name"]]
        return cmd, 90
    if short == "rotate_object":
        return py + ["primitives.rotate_object", "--object-name", kwargs["object_name"],
                     "--base-name", kwargs["base_name"], "--mode", "sim", "--budget-s", "150"], 150
    if short == "translate_object":
        action = str(kwargs["action"]).replace("_", "-")
        return py + ["primitives.translate_object", "--mode", "sim",
                     "--object-name", kwargs["object_name"], "--base-name", kwargs["base_name"],
                     f"--{action}"], 360
    raise ValueError(f"unsupported replay tool: {name}")


def _result_from_output(text: str) -> dict:
    begin, end = "__RESULT_JSON__", "__END_RESULT_JSON__"
    if begin not in text or end not in text:
        return {"result": "failure", "error": "primitive emitted no result envelope"}
    raw = text.rsplit(begin, 1)[1].split(end, 1)[0].strip()
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as exc:
        return {"result": "failure", "error": f"invalid primitive result JSON: {exc}"}
    return value if isinstance(value, dict) else {
        "result": "failure", "error": "primitive result was not an object"
    }


def _run(cmd: list[str], timeout: int, env: dict[str, str]) -> tuple[int, str, dict]:
    try:
        proc = subprocess.run(cmd, cwd=ROOT, env=env, text=True,
                              stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                              timeout=timeout)
        output = proc.stdout or ""
        result = _result_from_output(output)
        if proc.returncode and result.get("result") != "failure":
            result = {"result": "failure", "error": f"primitive exited {proc.returncode}",
                      "reported": result}
        return proc.returncode, output, result
    except subprocess.TimeoutExpired as exc:
        raw_output = exc.stdout or ""
        output = (raw_output.decode(errors="replace")
                  if isinstance(raw_output, bytes) else raw_output)
        return 124, output, {"result": "failure", "error": f"primitive timed out after {timeout}s"}


def derive_sim_sequence(obj: str, recorded: list[str], base_name: str,
                        quat_xyzw: list[float]) -> tuple[list[str], str]:
    """The production replay's derive step (replay_real_assembly.derive_sequence_for)
    fed by the simulator's observed part orientation: the same planner decides
    one or two passes from the face the part lies on."""
    from scipy.spatial.transform import Rotation as _R
    from primitives.shared.config import DEFAULT_BASE_ORIENTATION
    from primitives.shared.fold_symmetry import load_symmetry_data
    from utils.data_path_finder import (find_assembly_json_by_base_name,
                                        get_assembly_data_dir, get_symmetry_dir)
    from compliant_insertion_studio.planning.sequence_planner import (
        build_tool_sequence, plan_passes, quat_to_matrix)

    R_start = quat_to_matrix(quat_xyzw)
    jf = find_assembly_json_by_base_name(base_name or "base1", str(get_assembly_data_dir()))
    components = json.load(open(jf))["components"]
    target = next(c for c in components if c["name"] == obj)["rotation"]["quaternion"]
    R_target = (quat_to_matrix(DEFAULT_BASE_ORIENTATION)
                @ quat_to_matrix([target[k] for k in "xyzw"]))
    passes = plan_passes(R_start, R_target, load_symmetry_data(obj, str(get_symmetry_dir())))
    sequence = build_tool_sequence(recorded, passes)
    rpy = _R.from_matrix(R_start).as_euler("xyz", degrees=True)
    note = (f"observed rpy=({rpy[0]:.1f}, {rpy[1]:.1f}, {rpy[2]:.1f}) -> "
            f"{passes} pass{'es' if passes > 1 else ''}, {len(sequence)} steps "
            f"({'same as recorded' if sequence == recorded else 'RESHAPED from recorded'})")
    return sequence, note


def _landing_check(obj, base_name, executed, regrasp_step, row, run_step, transcript):
    """Re-plan from the landed face exactly as replay_real_assembly does after a place-down."""
    from ablations.replay_real_assembly import STAGING_MAX_RETRIES, staging_single_pass_ok
    first = next((i for i, st in enumerate(executed) if "__move_to_grasp(" in st), None)
    cycle = executed[first + 1:] if first is not None else []
    for attempt in range(STAGING_MAX_RETRIES + 1):
        q = (row["result"].get("current_object_orientation") or {}).get("quat")
        q = [q["x"], q["y"], q["z"], q["w"]] if isinstance(q, dict) else None
        check = staging_single_pass_ok(obj, base_name, q)
        if check is None:
            transcript.write(f"[landing-check] {obj}: could not evaluate the landing - continuing\n")
            return row, True
        ok, err = check
        note = f"[landing-check] {obj}: up-face error {err:.1f} deg -> {'single pass OK' if ok else 'NEEDS ANOTHER PASS'}"
        transcript.write(note + "\n")
        print("  " + note, flush=True)
        if ok:
            return row, True
        if attempt == STAGING_MAX_RETRIES or not cycle:
            row = dict(row, result={"result": "failure", "error":
                       f"{obj} landing still not insertable after {attempt} extra pass(es)"})
            return row, False
        for st in cycle + [regrasp_step]:
            row = run_step(obj, st)
            if row["result"].get("result") != "success":
                return row, False
    return row, False


def _scorer(kind: str, names: list[str], base: str) -> tuple[list[str], int]:
    return ([sys.executable, "-u", "-m", "ablations.sim_seat_scorer", f"--{kind}", ",".join(names),
             "--base", base], 60)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--assembly-json", required=True)
    ap.add_argument("--only", default=None)
    ap.add_argument("--no-startup", action="store_true")
    ap.add_argument("--derive-sequence", action=argparse.BooleanOptionalAction, default=True,
                    help="observe every part after the startup and let the cell's sequence planner "
                         "choose one or two passes (default, as replay_real_assembly)")
    ap.add_argument("--base-name", default="base1")
    ap.add_argument("--log-dir", default="ablations/logs/sim-replay")
    args = ap.parse_args(argv)
    refusal = require_sim_container()
    if refusal:
        print(json.dumps({"replay": "refused", "error": refusal}, indent=2))
        return 3

    path = Path(args.assembly_json).resolve()
    data = json.loads(path.read_text())
    entries = sorted(data.get("assembly_order", []), key=lambda row: row.get("assembly_order", 0))
    if args.only:
        entries = [row for row in entries if row.get("object_name") == args.only]
    if not entries:
        print(f"ERROR: no matching assembly entries in {path}", file=sys.stderr)
        return 2

    stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    log_dir = Path(args.log_dir).resolve()
    log_dir.mkdir(parents=True, exist_ok=True)
    transcript_path = log_dir / f"replay-sim-{stamp}.log"
    receipt_path = log_dir / f"replay-sim-{stamp}.jsonl"
    env = os.environ.copy()
    env["ROS_MCP_MODE"] = "sim"
    env["PYTHONPATH"] = f"{ROOT}:{env.get('PYTHONPATH', '')}"

    startup: list[tuple[str, str]] = [] if args.no_startup else [
        ("startup", "ros-mcp-server__move_home(mode='sim')"),
        ("startup", "ros-mcp-server__control_gripper(command='open', mode='sim')"),
    ]
    completed: list[str] = []
    scores: dict[str, dict] = {}
    total = len(startup) + sum(len(row.get("tool_sequence", [])) for row in entries)
    with transcript_path.open("w", buffering=1) as transcript, receipt_path.open("w", buffering=1) as receipt:
        counter = {"index": 0}
        active_grasp: dict[str, int] = {}

        def fail(row):
            print(json.dumps({"replay": "failure", "failed": row, "scores": scores,
                              "transcript": str(transcript_path),
                              "receipt": str(receipt_path)}, indent=2))
            return 1

        def run_step(obj: str, step: str, cmd=None, timeout=None):
            counter["index"] += 1
            index = counter["index"]
            if cmd is None:
                if "__move_to_grasp(" in step:
                    _name, parsed = _parse_tool_call(step)
                    active_grasp[obj] = int(parsed["grasp_id"])
                cmd, timeout = _command_for_step(step, None if obj == "startup" else obj,
                                                 active_grasp.get(obj))
            started = dt.datetime.now(dt.timezone.utc).isoformat()
            transcript.write(f"\n{'=' * 72}\nSTEP {index} [{obj}] {step}\n{'=' * 72}\n")
            rc, output, result = _run(cmd, timeout, env)
            transcript.write(output)
            row = {"schema": "ros-mcp.sim-replay-step/1", "index": index, "object": obj,
                   "tool_call": step, "command": cmd, "started_utc": started,
                   "finished_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
                   "returncode": rc, "result": result}
            receipt.write(json.dumps(row, sort_keys=True) + "\n")
            print(f"[{index:02d}/{total:02d}] {obj}: {result.get('result')}"
                  + (f" {json.dumps(result.get('pre_release'), sort_keys=True)}"
                     if result.get("pre_release") else ""), flush=True)
            return row

        for obj, step in startup:
            row = run_step(obj, step)
            if row["result"].get("result") != "success":
                return fail(row)

        if args.derive_sequence:
            names = [row["object_name"] for row in entries]
            cmd, timeout = _scorer("observe", names, args.base_name)
            row = run_step("derive", "__observe_parts__", cmd, timeout)
            observed = row["result"].get("parts", {})
            for entry in entries:
                obj = entry["object_name"]
                recorded = list(entry.get("tool_sequence", []))
                if obj in observed:
                    try:
                        entry["tool_sequence"], note = derive_sim_sequence(
                            obj, recorded, args.base_name, observed[obj]["quat_xyzw"])
                    except Exception as exc:  # the production rule: never a new way to fail
                        note = f"derive failed ({exc}) -> recorded sequence"
                else:
                    note = "observe failed -> recorded sequence"
                derive_row = {"schema": "ros-mcp.sim-derive/1", "object": obj, "note": note,
                              "observed": observed.get(obj), "tool_sequence": entry["tool_sequence"]}
                receipt.write(json.dumps(derive_row, sort_keys=True) + "\n")
                transcript.write(f"[derive-sequence] {obj}: {note}\n")
                print(f"  [derive-sequence] {obj}: {note}", flush=True)
            total = counter["index"] + sum(len(e["tool_sequence"]) + 1 for e in entries) + 1

        for entry in entries:
            obj = entry["object_name"]
            executed: list[str] = []
            placed_at_staging = False
            for step in entry.get("tool_sequence", []):
                row = run_step(obj, step)
                if row["result"].get("result") != "success":
                    return fail(row)
                # The real replay's landing check (replay_real_assembly, staging regrasp): a
                # landing is a fresh start state; if the regrasp observes a face that one pass
                # cannot insert, run another place-down pass, at most STAGING_MAX_RETRIES.
                if "__move_to_grasp(" in step and placed_at_staging:
                    row, ok = _landing_check(obj, args.base_name, executed, step, row, run_step, transcript)
                    if not ok:
                        return fail(row)
                    placed_at_staging = False
                executed.append(step)
                if "action='place_down'" in step:
                    placed_at_staging = True
            # Ruling A (lead, 2026-11-19): the seat is the part's final pose after
            # release and retract, scored before the next part starts.
            cmd, timeout = _scorer("score", [obj], args.base_name)
            row = run_step(obj, "__score_final_seat__", cmd, timeout)
            scores[obj] = row["result"].get("parts", {}).get(obj, {"error": row["result"].get("error")})
            if row["result"].get("result") != "success":
                return fail(row)
            completed.append(obj)

        # Re-score every seated part at the end: a later insert must not have disturbed it.
        cmd, timeout = _scorer("score", completed, args.base_name)
        row = run_step("final", "__rescore_all_seats__", cmd, timeout)
        scores["final_rescore"] = row["result"].get("parts", {})
        if row["result"].get("result") != "success":
            return fail(row)

    print(json.dumps({"replay": "success", "completed_objects": completed, "scores": scores,
                      "transcript": str(transcript_path), "receipt": str(receipt_path)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
