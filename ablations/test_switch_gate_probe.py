#!/usr/bin/env python3
"""NO-LLM probe for the first-failure @switch fix (option a). Run on a4500 with
PYTHONPATH=~/Documents/ros-mcp-server. Exercises _handle_phase_2 Gate 0 directly."""
import os, tempfile, sys

def fresh_env():
    d = tempfile.mkdtemp()
    os.environ["MCP_CLIENT_OUTPUT_DIR"] = d
    return d

def stub_assembled(base, mode):   # sim_passed = True
    return {"result": "success"}

PASS = True
def check(name, cond):
    global PASS
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}")
    PASS = PASS and cond

# import fresh each case so module-level caches don't leak across temp dirs
import importlib
import triggers.signal_verify_results as svr
from triggers.signal_phase_complete import _handle_phase_2

print("CASE 1: first unresolved rejection + sim_passed -> RE-PROMPT (not switch), marks reprompted")
fresh_env()
svr.record_rejection("u_green", "translate_object(insert)")
r1 = _handle_phase_2("success", "", "sim", base_name="base2", verify_assembly_fn=stub_assembled)
check("requires_response True", r1.get("requires_response") is True)
check("no action:switch yet", r1.get("action") != "switch")
check("reprompted flag set", svr.get_unresolved_rejections().get("u_green", {}).get("reprompted") is True)

print("CASE 2: still unresolved AFTER re-prompt + sim_passed -> @switch (action:switch)")
r2 = _handle_phase_2("success", "", "sim", base_name="base2", verify_assembly_fn=stub_assembled)
check("action == switch", r2.get("action") == "switch")
check("status == failure", r2.get("status") == "failure")
check("requires_response False", r2.get("requires_response") is False)

print("CASE 3: toggle MCP_SWITCH_ON_UNFIXABLE_LOG=0 -> NO switch, re-prompt only")
fresh_env()
os.environ["MCP_SWITCH_ON_UNFIXABLE_LOG"] = "0"
svr.record_rejection("u_green", "x"); svr.mark_reprompted("u_green")
r3 = _handle_phase_2("success", "", "sim", base_name="base2", verify_assembly_fn=stub_assembled)
check("no action:switch when toggled off", r3.get("action") != "switch")
check("re-prompts instead", r3.get("requires_response") is True)
del os.environ["MCP_SWITCH_ON_UNFIXABLE_LOG"]

print("CASE 4: reprompted but sim NOT passed -> NOT a switch (escalate territory, not 'completed')")
fresh_env()
svr.record_rejection("u_green", "x"); svr.mark_reprompted("u_green")
r4 = _handle_phase_2("success", "", "sim", base_name="base2", verify_assembly_fn=lambda b, m: {"result": "failure"})
check("no switch when sim not passed", r4.get("action") != "switch")

print("CASE 5: clean (no rejections) -> no false switch")
fresh_env()
r5 = _handle_phase_2("success", "", "sim", base_name="base2", verify_assembly_fn=stub_assembled)
check("no action:switch on clean run", r5.get("action") != "switch")

print("\nRESULT:", "ALL GREEN" if PASS else "FAILURES PRESENT")
sys.exit(0 if PASS else 1)
