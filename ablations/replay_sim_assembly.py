#!/usr/bin/env python3
"""Replay a recorded assembly through the cell's simulation primitives.

This is the standalone entrance for the same sim tool sequence that
``replay_verify.py`` invokes through unified_api.  It deliberately refuses
real mode: a simulator process must serve the ROS contract and this process
only starts the existing primitive subprocesses against that contract.
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


ROOT = Path(__file__).resolve().parents[1]


def _command_for_step(step: str) -> tuple[list[str], int]:
    name, kwargs = _parse_tool_call(step)
    if kwargs.get("mode") != "sim":
        raise ValueError(f"standalone sim replay refuses non-sim step: {step}")
    short = name.removeprefix("ros_mcp_server__")
    py = [sys.executable, "-u", "-m"]
    if short == "move_home":
        return py + ["primitives.move_home", "--mode", "sim"], 120
    if short == "control_gripper":
        return py + ["primitives.control_gripper", str(kwargs["command"]), "--mode", "sim"], 90
    if short == "move_to_grasp":
        return py + ["primitives.move_to_grasp", "--object-name", kwargs["object_name"],
                     "--grasp-id", str(kwargs["grasp_id"]), "--mode", "sim"], 120
    if short == "move_to_safe_height":
        cmd = py + ["primitives.move_to_safe_height", "--mode", "sim"]
        if kwargs.get("object_name"):
            cmd += ["--object-name", kwargs["object_name"]]
        return cmd, 90
    if short == "rotate_object":
        return py + ["primitives.rotate_object", "--object-name", kwargs["object_name"],
                     "--base-name", kwargs["base_name"], "--mode", "sim"], 150
    if short == "translate_object":
        action = str(kwargs["action"]).replace("_", "-")
        if action == "insert":
            return py + ["ablations.sim_regular_insert", "--object-name", kwargs["object_name"],
                         "--base-name", kwargs["base_name"]], 30
        return py + ["primitives.translate_object", "--mode", "sim",
                     "--object-name", kwargs["object_name"], "--base-name", kwargs["base_name"],
                     f"--{action}"], 120
    raise ValueError(f"unsupported replay tool: {name}")


def _staging_view_command() -> tuple[list[str], int]:
    """The production replay's camera-aimed view of the whole staging region."""
    from ablations.staging_layout import STAGING_VIEW, TOP_DOWN_Q
    from primitives.shared.config import TABLE_HEIGHT
    return ([sys.executable, "-u", "-m", "primitives.move_cartesian", "--mode", "real",
             "--xyz", *(f"{v:.4f}" for v in STAGING_VIEW),
             "--quat", *(str(v) for v in TOP_DOWN_Q),
             "--look-at-z", f"{TABLE_HEIGHT:.4f}"], 120)


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
        output = (exc.stdout or "") if isinstance(exc.stdout, str) else ""
        return 124, output, {"result": "failure", "error": f"primitive timed out after {timeout}s"}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--assembly-json", required=True)
    ap.add_argument("--only", default=None)
    ap.add_argument("--no-startup", action="store_true")
    ap.add_argument("--fov-gated", action="store_true",
                    help="fly the production camera-aimed staging view before each part's first pick")
    ap.add_argument("--log-dir", default="ablations/logs/sim-replay")
    args = ap.parse_args(argv)

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

    steps: list[tuple[str, str]] = []
    if not args.no_startup:
        steps.extend([
            ("startup", "ros-mcp-server__move_home(mode='sim')"),
            ("startup", "ros-mcp-server__control_gripper(command='open', mode='sim')"),
        ])
    for row in entries:
        obj = row["object_name"]
        first_grasp = True
        for step in row.get("tool_sequence", []):
            if args.fov_gated and first_grasp and "__move_to_grasp(" in step:
                steps.append((obj, "__camera_staging_view__"))
                first_grasp = False
            steps.append((obj, step))

    completed: list[str] = []
    with transcript_path.open("w", buffering=1) as transcript, receipt_path.open("w", buffering=1) as receipt:
        for index, (obj, step) in enumerate(steps, 1):
            cmd, timeout = _staging_view_command() if step == "__camera_staging_view__" else _command_for_step(step)
            started = dt.datetime.now(dt.timezone.utc).isoformat()
            transcript.write(f"\n{'=' * 72}\nSTEP {index}/{len(steps)} [{obj}] {step}\n{'=' * 72}\n")
            rc, output, result = _run(cmd, timeout, env)
            transcript.write(output)
            row = {"schema": "ros-mcp.sim-replay-step/1", "index": index, "object": obj,
                   "tool_call": step, "command": cmd, "started_utc": started,
                   "finished_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
                   "returncode": rc, "result": result}
            receipt.write(json.dumps(row, sort_keys=True) + "\n")
            print(f"[{index:02d}/{len(steps):02d}] {obj}: {result.get('result')}", flush=True)
            if result.get("result") != "success":
                print(json.dumps({"replay": "failure", "failed": row,
                                  "transcript": str(transcript_path),
                                  "receipt": str(receipt_path)}, indent=2))
                return 1
            if obj != "startup" and obj not in completed:
                completed.append(obj)

    print(json.dumps({"replay": "success", "completed_objects": completed,
                      "transcript": str(transcript_path), "receipt": str(receipt_path)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
