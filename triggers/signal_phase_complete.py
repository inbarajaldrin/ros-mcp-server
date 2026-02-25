"""Signal phase completion to the MCP client.

Related files (phase signal pipeline):
  - triggers/signal_phase_complete.py (this file) — handles phase signals, gates on logged results
  - elicitations/verify_assembly.py   — Pydantic schemas for human verification of real-world assembly (phase 3)

The agent calls this at the end of each assembly phase. The MCP client
watches for this tool's result to trigger the next phase.

Phase behaviors:
  - Phase 1 (Disassembly sequence discovery): Gates on Disassembly_*_results.json
    + verify_disassembly(check_all) in sim.
  - Phase 2 (Assembly sequence discovery): Gates on Assembly_*_results.json
    + verify_assembly(check_all) in sim.
  - Phase 3 sim: Gates on verify_assembly(check_all).
  - Phase 3 real: Invokes verify_assembly elicitation for human verification.
"""

import os
from pathlib import Path
from typing import Literal, Dict, Any, Optional, Callable
from mcp.server.fastmcp import Context
from mcp.server.session import ServerSession


def _get_logs_dir() -> Path:
    """Return the logs directory path, respecting MCP_CLIENT_OUTPUT_DIR."""
    base_output_dir = os.getenv("MCP_CLIENT_OUTPUT_DIR", "").strip()
    if base_output_dir:
        return Path(os.path.join(base_output_dir, "logs"))
    return Path(__file__).parent.parent / "logs"


def _has_assembly_results() -> bool:
    """Check if Assembly_*_results.json files exist in logs dir."""
    logs_dir = _get_logs_dir()
    return logs_dir.exists() and len(list(logs_dir.glob("Assembly_*_results.json"))) > 0


async def handle_phase_signal(
    phase: Literal[1, 2, 3],
    status: Literal["success", "failure"],
    comment: str,
    ctx: Context[ServerSession, None],
    elicit_user_fn=None,
    mode: Literal["sim", "real"] = "sim",
    base_name: Optional[str] = None,
    verify_disassembly_fn: Optional[Callable] = None,
    verify_assembly_fn: Optional[Callable] = None,
) -> Dict[str, Any]:
    """Handle phase completion signal.

    Args:
        phase: Which phase completed (1, 2, or 3).
        status: Whether the phase succeeded or failed.
        comment: Optional comment (should explain failure).
        ctx: MCP Context (needed for phase 3 elicitation).
        elicit_user_fn: Reference to the elicit_user tool function from server.py.
        mode: sim or real.
        base_name: Assembly base name for verification gating.
        verify_disassembly_fn: Callback(base_name, mode) -> verify result dict.
        verify_assembly_fn: Callback(base_name, mode) -> verify result dict.

    Returns:
        Structured result dict consumed by the MCP client.
    """
    if phase == 1:
        result = _handle_phase_1(status, comment, mode, base_name, verify_disassembly_fn)
    elif phase == 2:
        result = _handle_phase_2(status, comment, mode, base_name, verify_assembly_fn)
    elif phase == 3:
        result = await _handle_phase_3(status, comment, ctx, elicit_user_fn, mode, base_name, verify_assembly_fn)

    # Stamp every return with phase/status so the MCP client can route it
    # Allow handlers to override status (e.g. gate check failing)
    effective_status = result.pop("status", status)
    envelope = {"phase": phase, "status": effective_status}
    return {**envelope, **result}


def _handle_phase_1(
    status: Literal["success", "failure"],
    comment: str,
    mode: Literal["sim", "real"] = "sim",
    base_name: Optional[str] = None,
    verify_disassembly_fn: Optional[Callable] = None,
) -> Dict[str, Any]:
    """Phase 1: Disassembly sequence discovery.

    On success, gates on:
      1. Disassembly results JSON logged to disk.
      2. verify_disassembly(check_all) confirms all objects are actually disassembled.
    """
    if status != "success":
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
            "message": "You have not logged any disassembly results. Use write_disassembly_results to log the grasp_id and gripper_state for each object you disassembled, then signal phase 1 complete again.",
        }

    # Gate 2: verify all objects are actually disassembled
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
    mode: Literal["sim", "real"] = "sim",
    base_name: Optional[str] = None,
    verify_assembly_fn: Optional[Callable] = None,
) -> Dict[str, Any]:
    """Phase 2: Assembly sequence discovery.

    On success, gates on:
      1. Assembly results JSON logged to disk.
      2. verify_assembly(check_all) confirms all objects are actually assembled.
    """
    if status != "success":
        return {"requires_response": False}

    # Gate 1: results file must exist
    if not _has_assembly_results():
        return {
            "status": "failure",
            "requires_response": True,
            "message": "You have not logged any assembly results. Use write_assembly_results to log the tool_sequence for each object you assembled, then signal phase 2 complete again.",
        }

    # Gate 2: verify all objects are actually assembled
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

    return {"requires_response": False}


async def _handle_phase_3(
    status: Literal["success", "failure"],
    comment: str,
    ctx: Context[ServerSession, None],
    elicit_user_fn,
    mode: Literal["sim", "real"] = "sim",
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
