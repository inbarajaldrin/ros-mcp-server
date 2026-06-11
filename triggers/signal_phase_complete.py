"""Signal phase completion to the MCP client.

Related files (phase signal pipeline):
  - triggers/signal_phase_complete.py (this file) -- handles phase signals, gates on logged results
  - triggers/signal_verify_results.py -- post-replay verification, manages verify_rejection.json
  - resource.py -- write_assembly_results clears rejections via clear_rejection()
  - ablations/replay_verify.py -- orchestrator replay script, calls signal_verify_results
  - elicitations/verify_assembly.py -- Pydantic schemas for human verification (phase 3 real)

The agent calls this at the end of each assembly phase. The MCP client
watches for this tool's result to trigger the next phase.

Sim verification runs ALWAYS (on both success and failure signals) to determine
true assembly/disassembly state. If the agent signals failure but sim says the
task actually succeeded and logs exist, the status is overridden to success so
the gate replay can run. This ensures the system (not the agent) decides
whether to escalate or switch.

Phase 2 gate sequence:
  Pre-check: verify_assembly(check_all) runs regardless of agent-reported status.
  Gate 0: Check verify_rejection.json -- reject if unresolved replay failures exist.
          Trigger @switch (requires_response=False) if any object has >= 3 failed attempts.
  Gate 1: verify_assembly(check_all) -- confirm all objects are actually assembled.
  Gate 2: Check Assembly_*_results.json exists on disk.

Phase behaviors:
  - Phase 1 (Disassembly): Sim verify always + gates on results file + verify_disassembly.
  - Phase 2 (Assembly): Sim verify always + Gate 0 (rejection) + Gate 1 (results file) + Gate 2.
  - Phase 3 sim: Gates on verify_assembly(check_all).
  - Phase 3 real: Invokes verify_assembly elicitation for human verification.
"""

import os
from pathlib import Path
from typing import Literal, Dict, Any, Optional, Callable
from mcp.server.fastmcp import Context
from mcp.server.session import ServerSession

from primitives.shared import predicates as P


def _get_logs_dir() -> Path:
    """Return the logs directory path, respecting MCP_CLIENT_OUTPUT_DIR."""
    base_output_dir = os.getenv("MCP_CLIENT_OUTPUT_DIR", "").strip()
    if base_output_dir:
        return Path(os.path.join(base_output_dir, "logs"))
    return Path(__file__).parent.parent / "logs"


def _has_assembly_results() -> bool:
    """Check if Assembly_*_results.json files exist in logs dir.

    DORMANT (CLEANUP item 2): the results-file gate helper. Only the dormant _handle_phase_*
    handlers call it; the active slim path does not. The stitcher produces the data now.
    Delete with the dormant handlers once the slim path + stitcher are proven.
    """
    logs_dir = _get_logs_dir()
    return logs_dir.exists() and len(list(logs_dir.glob("Assembly_*_results.json"))) > 0


def _remaining_objects(res) -> list:
    """Objects still not in their target state, for the failure message."""
    if not isinstance(res, dict):
        return []
    return res.get("unassembled_objects") or res.get("still_assembled_objects") or []


def _phase_complete(verify_fn, base_name, mode, robot_state_fn):
    """Authoritative phase completion: check_all verify + arm home. Returns (complete, reason).

    is_home implies the gripper is released (move_home refuses while holding), so the two
    checks cover the whole "phase done + neutral" condition. Fail-closed throughout.
    """
    if not (base_name and verify_fn):
        return False, "verification unavailable"
    try:
        res = verify_fn(base_name, mode)  # check_all
    except Exception as e:
        return False, f"verification unavailable ({type(e).__name__})"
    if not (isinstance(res, dict) and res.get("result") == "success"):
        rem = _remaining_objects(res)
        return False, ("not all objects verified" + (f" (remaining: {rem})" if rem else ""))
    if robot_state_fn is None:
        return False, "robot state unavailable"
    try:
        state = robot_state_fn(mode)
    except Exception as e:
        return False, f"robot state unavailable ({type(e).__name__})"
    home_ok, _ = P.is_home(state)
    if not home_ok:
        return False, "arm not home"
    return True, "complete"


async def handle_phase_signal(
    phase: Literal[1, 2, 3],
    status: Literal["success", "failure"],
    comment: str,
    ctx: Context[ServerSession, None],
    mode: Literal["sim", "real"],
    elicit_user_fn=None,                          # DORMANT: phase-3-real elicitation dropped
    base_name: Optional[str] = None,
    verify_disassembly_fn: Optional[Callable] = None,
    verify_assembly_fn: Optional[Callable] = None,
    robot_state_fn: Optional[Callable] = None,    # () -> get_robot_state dict, for is_home
    has_committed_progress_fn: Optional[Callable] = None,  # () -> bool: >=1 commit this phase
) -> Dict[str, Any]:
    """Authoritatively decide phase completion (uniform sim+real).

    Gates on check_all verify (verify_disassembly for phase 1, verify_assembly for 2/3) +
    is_home. The agent's `status` drives the client's hook-based escalate/switch (the YAML
    @escalate/@switch hooks match on this result JSON); the system's verdict can OVERRIDE it.

    Result schema (uniform, `message` ALWAYS present): {phase, status, requires_response, message}.
      - status=failure + genuinely incomplete -> stays failure -> YAML hook escalates. comment=why.
      - status=failure + actually complete     -> OVERRIDE to success (hallucination safeguard).
      - status=success + incomplete            -> failure, requires_response (keep working).
      - status=success + complete              -> success.

    DORMANT (code preserved, off the active path): _handle_phase_{1,2,3} below (the old
    @switch/replay Gate 0 + results-file gate + phase-3-real elicitation), re-enableable.
    """
    verify_fn = verify_disassembly_fn if phase == 1 else verify_assembly_fn
    complete, reason = _phase_complete(verify_fn, base_name, mode, robot_state_fn)
    label = "Disassembly" if phase == 1 else "Assembly"

    if status != "success":
        if complete:
            # OVERRIDE: agent reported failure but the phase is authoritatively complete.
            # Don't escalate; carry the agent comment; (stitcher trigger = Step 5).
            return {"phase": phase, "status": "success", "requires_response": False,
                    "message": (f"{label} verified complete despite agent signaling failure "
                                f"(agent comment: {comment or 'none'}). Overriding to success."),
                    "override": True}
        # Genuine failure -> status stays failure. Route the cascade by whether there is
        # committed work this phase: if >=1 object is committed (the floor file exists), emit
        # action="switch" so the client preserves the ledger+scene+context and a stronger tier
        # CONTINUES from the committed state (design B). With nothing committed, omit action ->
        # the YAML @escalate hook does a clean restart (nothing to preserve). The floor file is
        # the cascade-vs-restart discriminator.
        result = {"phase": phase, "status": "failure", "requires_response": False,
                  "message": f"{label} incomplete ({reason}). Agent comment: {comment or 'none'}."}
        if has_committed_progress_fn is not None and has_committed_progress_fn():
            result["action"] = "switch"
        return result

    # Agent reported success -> must be authoritatively complete.
    if not complete:
        return {"phase": phase, "status": "failure", "requires_response": True,
                "message": f"{label} not complete: {reason}. Keep working, then signal again."}
    return {"phase": phase, "status": "success", "requires_response": False,
            "message": f"Phase {phase} ({label}) complete."}


# ====================================================================================
# DORMANT — the old per-phase handlers, NO LONGER CALLED by handle_phase_signal (the slim
# _phase_complete path replaced them). Preserved for reference / re-enable / the
# ablations/test_switch_gate_probe.py probe.
#
# ░░ CLEANUP CHECKLIST ░░  Once the slim path + the stitcher are PROVEN in a live mode2
# run, delete in this order (each line is independently removable):
#   [ ] 1. _handle_phase_1 / _handle_phase_2 / _handle_phase_3 (the three functions below)
#   [ ] 2. _has_assembly_results() above            — dormant results-file gate helper
#   [ ] 3. triggers/signal_verify_results.py        — the @switch/replay log-repair machinery
#          (+ any orchestrator replay wiring); confirm nothing live imports it first
#   [ ] 4. elicitations/verify_assembly.py          — phase-3-real human elicitation, ONLY if
#          signal_operator fully covers the human-in-the-loop need
#   [ ] 5. handle_phase_signal params elicit_user_fn + ctx (if unused) + the server tool's
#          _handle_elicitation wiring for signal_phase_complete
# Do NOT delete before that live run passes — this is the rollback path.
# ====================================================================================

def _handle_phase_1(
    status: Literal["success", "failure"],
    comment: str,
    mode: Literal["sim", "real"],
    base_name: Optional[str] = None,
    verify_disassembly_fn: Optional[Callable] = None,
) -> Dict[str, Any]:
    """Phase 1: Disassembly sequence discovery.

    On success, gates on:
      1. Disassembly results JSON logged to disk.
      2. verify_disassembly(check_all) confirms all objects are actually disassembled.
    """
    # Always run sim verification to determine true assembly state,
    # regardless of what the agent reported as status.
    sim_passed = False
    if base_name and verify_disassembly_fn:
        verification = verify_disassembly_fn(base_name, mode)
        sim_passed = isinstance(verification, dict) and verification.get("result") == "success"

    if status != "success":
        if sim_passed:
            # Agent said failure but sim says disassembly is actually complete.
            logs_dir = _get_logs_dir()
            has_results = logs_dir.exists() and len(list(logs_dir.glob("Disassembly_*_results.json"))) > 0
            if has_results:
                # Logs exist — override to success so @complete-phase fires.
                return {
                    "status": "success",
                    "requires_response": False,
                    "message": "Disassembly verification passed despite agent signaling failure. Proceeding with logged results.",
                    "override": True,
                }
            else:
                # Sim passed but no logs written — prompt agent to write them.
                return {
                    "status": "failure",
                    "requires_response": True,
                    "message": "Disassembly verification passed but you have not logged any results. Use write_disassembly_results to log the grasp_id for each object you disassembled, then signal phase 1 complete again.",
                    "override": True,
                }
        return {"requires_response": False}

    # Gate 1: results file must exist
    logs_dir = _get_logs_dir()
    has_results = (
        logs_dir.exists()
        and len(list(logs_dir.glob("Disassembly_*_results.json"))) > 0
    )

    if not has_results:
        return {
            "status": "failure",
            "requires_response": True,
            "message": "You have not logged any disassembly results. Use write_disassembly_results to log the grasp_id for each object you disassembled, then signal phase 1 complete again.",
        }

    # Gate 2: verify all objects are actually disassembled
    if not sim_passed:
        if base_name and verify_disassembly_fn:
            verification = verify_disassembly_fn(base_name, mode)
            if isinstance(verification, dict) and verification.get("result") == "failure":
                still_assembled = verification.get("still_assembled_objects", [])
                return {
                    "status": "failure",
                    "requires_response": True,
                    "message": f"Disassembly verification failed. Still assembled: {still_assembled}. Disassemble remaining objects and signal phase 1 complete again.",
                    "verification": verification,
                }

    return {"requires_response": False}


def _handle_phase_2(
    status: Literal["success", "failure"],
    comment: str,
    mode: Literal["sim", "real"],
    base_name: Optional[str] = None,
    verify_assembly_fn: Optional[Callable] = None,
) -> Dict[str, Any]:
    """Phase 2: Assembly sequence discovery.

    On success, gates on:
      0. No unresolved verify rejections (orchestrator replay failures).
      1. Assembly results JSON logged to disk.
      2. verify_assembly(check_all) confirms all objects are actually assembled.
    """
    # Always run sim verification to determine true assembly state,
    # regardless of what the agent reported as status.
    sim_passed = False
    if base_name and verify_assembly_fn:
        verification = verify_assembly_fn(base_name, mode)
        sim_passed = isinstance(verification, dict) and verification.get("result") == "success"

    if status != "success":
        if sim_passed:
            if _has_assembly_results():
                # Sim passed + logs exist — override to success so gate replay can run.
                return {
                    "status": "success",
                    "requires_response": False,
                    "message": "Assembly verification passed despite agent signaling failure. Proceeding with gate replay.",
                    "override": True,
                }
            else:
                # Sim passed but no logs written — prompt agent to write them.
                return {
                    "status": "failure",
                    "requires_response": True,
                    "message": "Assembly verification passed but you have not logged any results. Use write_assembly_results to log the tool_sequence for each object you assembled, then signal phase 2 complete again.",
                    "override": True,
                }
        return {"requires_response": False}

    # ░░ CLEANUP item 3 ░░ Gate 0 — the @switch/replay log-repair. Obsolete with the stitcher
    # (no written log to repair). Dormant: this whole block + signal_verify_results.py go once
    # the slim path + stitcher are proven.
    # Gate 0: check for unresolved orchestrator replay rejections
    from triggers.signal_verify_results import (
        get_unresolved_rejections, has_exhausted_object, mark_reprompted
    )

    exhausted = has_exhausted_object()
    if exhausted:
        return {
            "status": "failure",
            "requires_response": False,
            "action": "switch",
            "message": (
                f"Object '{exhausted}' has exhausted maximum replay attempts. "
                f"Switching to next model to fix logs."
            ),
        }

    unresolved = get_unresolved_rejections()
    if unresolved:
        # First-failure @switch (author decision 2026-06-01, option (a)): the assembly is
        # physically complete (sim_passed) but a logged sequence is STILL wrong after the
        # model was already re-prompted to fix it. That is the defining "completed-but-logged-
        # wrong, can't self-correct" case → switch to the next model KEEPING CONTEXT (it sees
        # the assembled scene + the rejected sequence and only needs to re-log), rather than
        # re-prompting the same model forever or escalating (which redoes the phase fresh).
        # Decoupled from the 3-strike `attempts` counter (see track-b-switch-gate-scope.md).
        # Toggle: MCP_SWITCH_ON_UNFIXABLE_LOG=0 reverts to re-prompt-only (3-strike path).
        # Default OFF (2026-06-01): superseded by the def-level gate onFail -> @switch (direct switch,
        # no same-model re-prompt). Toggle on only to restore the old reprompt-then-switch path.
        switch_on_unfixable = os.getenv("MCP_SWITCH_ON_UNFIXABLE_LOG", "0") != "0"
        already_reprompted = [n for n, s in unresolved.items() if s.get("reprompted")]
        if switch_on_unfixable and sim_passed and already_reprompted:
            return {
                "status": "failure",
                "requires_response": False,
                "action": "switch",
                "message": (
                    f"Assembly is physically complete but the logged tool_sequence for "
                    f"{already_reprompted} is still incorrect after a re-prompt. This model "
                    f"cannot self-correct its log; switching to the next model to fix the "
                    f"logs (keeping context, assembly retained)."
                ),
            }
        # First time we see these unresolved: give the SAME model one chance to fix, and mark
        # it so a repeat lands in the @switch path above.
        objects = list(unresolved.keys())
        for name in objects:
            mark_reprompted(name)
        details = "; ".join(
            f"'{name}' (attempt {s['attempts']}, failed at: {s.get('last_failed_step', '?')})"
            for name, s in unresolved.items()
        )
        return {
            "status": "failure",
            "requires_response": True,
            "message": (
                f"You must update the logged tool_sequence for {objects} "
                f"using write_assembly_results before signaling completion. "
                f"Your assembly was correct but your logs are incomplete. "
                f"Review your conversation history for the actual steps you "
                f"performed. Details: {details}"
            ),
        }

    # Gate 1: verify all objects are actually assembled
    if not sim_passed:
        if base_name and verify_assembly_fn:
            verification = verify_assembly_fn(base_name, mode)
            if isinstance(verification, dict) and verification.get("result") == "failure":
                unassembled = verification.get("unassembled_objects", [])
                return {
                    "status": "failure",
                    "requires_response": True,
                    "message": f"Assembly verification failed. Unassembled: {unassembled}. Assemble remaining objects and signal phase 2 complete again.",
                    "verification": verification,
                }

    # Gate 2: results file must exist
    if not _has_assembly_results():
        return {
            "status": "failure",
            "requires_response": True,
            "message": "You have not logged any assembly results. Use write_assembly_results to log the tool_sequence for each object you assembled, then signal phase 2 complete again.",
        }

    return {"requires_response": False}


async def _handle_phase_3(
    status: Literal["success", "failure"],
    comment: str,
    ctx: Context[ServerSession, None],
    elicit_user_fn,
    mode: Literal["sim", "real"],
    base_name: Optional[str] = None,
    verify_assembly_fn: Optional[Callable] = None,
) -> Dict[str, Any]:
    """Phase 3: Assembly execution.

    Sim: gates on verify_assembly(check_all).
    Real: invokes human verification elicitation.
    """
    if mode == "sim":
        # Gate: verify all objects are actually assembled
        if status == "success" and base_name and verify_assembly_fn:
            verification = verify_assembly_fn(base_name, mode)
            if isinstance(verification, dict) and verification.get("result") == "failure":
                unassembled = verification.get("unassembled_objects", [])
                return {
                    "status": "failure",
                    "requires_response": True,
                    "message": f"Assembly verification failed. Unassembled: {unassembled}. Assemble remaining objects and signal phase 3 complete again.",
                    "verification": verification,
                }
        return {"requires_response": False}

    # ░░ CLEANUP item 4 ░░ phase-3-real human elicitation. Dropped from the active path —
    # real now gates on automated check_all (+ occlusion certification); signal_operator is
    # the explicit human-in-the-loop. This block + elicitations/verify_assembly.py go once
    # signal_operator is confirmed to cover the need.
    # Real mode: human verification elicitation
    if elicit_user_fn is None:
        return {}

    try:
        import importlib.util

        # Dynamically load verify_assembly elicitation module
        script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        verify_assembly_path = os.path.join(script_dir, "elicitations", "verify_assembly.py")

        spec = importlib.util.spec_from_file_location(
            "elicitations.verify_assembly", verify_assembly_path
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        # Build message from phase result
        message = module.build_elicitation_message({"phase": 3, "status": status, "comment": comment})

        # Call elicit_user (same pattern as prepare_workspace in server.py)
        elicit_response = await elicit_user_fn(ctx, "verify_assembly", message, {"phase": 3, "status": status})

        return {"human_verification": elicit_response}

    except Exception as e:
        return {"human_verification": {"status": "error", "message": f"Human verification failed: {str(e)}"}}
