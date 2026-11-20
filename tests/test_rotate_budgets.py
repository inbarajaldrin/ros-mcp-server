"""Every caller of rotate_object passes its timeout as the budget (lead 2026-11-20, review r2 P3-8): the replay,
run_assembly_step and the four MCP servers, one value (primitives.shared.budgets.ROTATE_TIMEOUT_S).

The servers need the mcp package, which the cell PC's test Python lacks: each server's rotate_object tool is compiled
from its source with the @mcp.tool() decorator dropped and its collaborators stubbed, then CALLED; the test reads the
command and timeout it hands to _run_with_retry."""
import ast
import json
import os
import subprocess
import sys
import types
import typing
from pathlib import Path

import pytest

import tests.ros_stubs as ros_stubs

ros_stubs.install()
from primitives.shared.budgets import ROTATE_TIMEOUT_S  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
SERVERS = ["server.py", "server_mode2.py", "server_p3.py", "server_ablation_quat.py"]


def _call_server_rotate(fname):
    tree = ast.parse((ROOT / fname).read_text())
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "rotate_object")
    fn.decorator_list = []
    calls = []
    ns = {
        "Annotated": typing.Annotated, "Optional": typing.Optional, "List": typing.List,
        "Field": lambda *a, **k: None, "Mode": str, "RotateObjectResult": dict,
        "os": os, "sys": sys, "json": json, "__file__": str(ROOT / fname),
        "subprocess": types.SimpleNamespace(run=lambda *a, **k: types.SimpleNamespace(returncode=0, stdout="", stderr=""),
                                            TimeoutExpired=subprocess.TimeoutExpired),
        "_parse_result": lambda cls, raw: raw,
        "_remap_mode": lambda mode: mode,                                  # server_mode2 / server_ablation_quat
        "_get_object_state": lambda name: {"orientation": [0.0, 0.0, 0.0, 1.0]},   # server_p3's state store
        "_run_primitive": object(),
        "_run_with_retry": lambda func, script, cmd, **kw: calls.append((script, cmd, kw)) or {"result": "success"},
    }
    exec(compile(ast.Module(body=[fn], type_ignores=[]), fname, "exec"), ns)
    params = [a.arg for a in fn.args.args]
    kwargs = {"object_name": "u_orange", "base_name": "base1"}
    if "mode" in params:
        kwargs["mode"] = "real"
    if "current_object_orientation" in params:
        kwargs["current_object_orientation"] = [0.0, 0.0, 0.0, 1.0]
    ns["rotate_object"](**kwargs)
    assert len(calls) == 1, fname
    return calls[0]


@pytest.mark.parametrize("fname", SERVERS)
def test_each_mcp_server_passes_its_timeout_as_the_budget(fname):
    script, cmd, kw = _call_server_rotate(fname)
    assert script == "rotate_object.py"
    assert kw["timeout"] == ROTATE_TIMEOUT_S
    argv = cmd.split()
    assert argv[argv.index("--budget-s") + 1] == str(ROTATE_TIMEOUT_S)


def test_run_assembly_step_passes_its_timeout_as_the_budget(monkeypatch):
    from compliant_insertion_studio.scripts import run_assembly_step as ras
    seen = []
    monkeypatch.setattr(ras, "_run", lambda label, argv, timeout_s: seen.append((argv, timeout_s)) or {})
    ras._rotate_object("u_orange", "base1", (0.0, 0.0, 0.0, 1.0))
    argv, timeout_s = seen[0]
    assert timeout_s == ROTATE_TIMEOUT_S and argv[argv.index("--budget-s") + 1] == str(ROTATE_TIMEOUT_S)


def test_the_replay_uses_the_same_value():
    from ablations import replay_real_assembly as rra
    argv, timeout = rra.build_argv("rotate_object", {"object_name": "u_orange", "base_name": "base1"},
                                   rra.AssemblyState())
    assert timeout == ROTATE_TIMEOUT_S and argv[argv.index("--budget-s") + 1] == str(ROTATE_TIMEOUT_S)
