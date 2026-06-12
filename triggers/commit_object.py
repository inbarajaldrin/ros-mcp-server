"""Commit a SINGLE object's assembly/disassembly (the `commit_object` tool).

Companion to signal_phase_complete.py, but per-object and STATE-TRUTH-DRIVEN: the agent
commits one object; the SYSTEM verifies live state rather than trusting an agent
self-report or a written log. A successful return is the *stitcher commit boundary* the
MCP client keys off to window the execution log for that object — the tool is tagged
_meta category "commit", so the client's RecipeManager closes the window on it by category
alone (no tool-name hardcoding). Success/failure is reported uniformly as result:.

Gates (uniform sim+real — every gate works on both; all evaluated, all failures reported
together so the agent fixes everything in one round-trip):

  G1 verify    is_assembled(X)   [phase 2/3]  or  is_disassembled(X)  [phase 1]   == True
  G2 released  is_object_grasped(X) == False        (the gripper let go of the part)
  G3 safe ht   is_at_safe_height == True            (arm retracted to safe height)

Uniform across every object — no object is special. HOME is NOT checked here; it is
enforced only at phase transitions (signal_phase_complete / onStart), so there is no
per-object "final object" / rule-(c) detection.

No agent status/comment and no elicitation — see the @mcp.tool in server_core.py. The
four *_fn callbacks are injected by that wrapper; this module stays free of ROS/subprocess
and is unit-testable with stub callbacks. Verdicts come from primitives.shared.predicates.
"""
from typing import Literal, Dict, Any, Optional, Callable

from primitives.shared import predicates as P


def handle_object_commit(
    object_name: str,
    phase: Literal[1, 2, 3],
    mode: Literal["sim", "real"],
    base_name: str = "",
    verify_assembly_fn: Optional[Callable] = None,     # (base_name, mode, object_name=None) -> result dict
    verify_disassembly_fn: Optional[Callable] = None,  # (base_name, mode, object_name=None) -> result dict
    verify_grasp_fn: Optional[Callable] = None,        # (object_name, mode) -> verify_grasp result dict
    robot_state_fn: Optional[Callable] = None,         # (mode) -> get_robot_state dict
    ordered_objects: Optional[list] = None,            # phase-appropriate order (reversed for disassembly); None => G4 OFF
    checkpoints_fn: Optional[Callable] = None,         # () -> set of saved checkpoint basenames; None => G4 OFF
) -> Dict[str, Any]:
    """Evaluate the per-object commit gates and return the client-facing envelope.

    Returns on success: {result:"success", requires_response:False,
                         message:"'X' committed.", commit:True}
    Returns on failure: {result:"failure", requires_response:True,
                         message:<gate remedies joined by ' · '>, failed_gates:[...]}
    """
    # Guard: object_name must be a real name. An empty/blank name would misroute the
    # dual verify callback to check_all (commit "no object") — reject it up front.
    if not object_name or not object_name.strip():
        return {
            "result": "failure",
            "requires_response": True,
            "message": "object_name is required to commit an object.",
            "failed_gates": ["object_name"],
        }

    is_disassembly = phase == 1
    verify_fn = verify_disassembly_fn if is_disassembly else verify_assembly_fn

    # NOTE: gates read live state across several subprocess calls. This is safe because the
    # arm is STOPPED at a commit (agent has retracted) and sim poses are deterministic — the
    # locked principle "sim determinism is a prerequisite" (out of client scope). The gates
    # do not defend against a part physically shifting between reads of a static scene.
    failures = []  # list of (gate_key, remedy_message)

    # NOTE: there is intentionally NO order gate here. The order PRE-GATE (currently
    # _assert_assembly_order, to be replaced by predicates.is_order_correct) blocks
    # out-of-order picks before they execute, so a commit can never be reached out of
    # order — a commit-time order check would be dead code. See the pre-gate registry note.

    # --- G1: per-object verify (assembled / disassembled) ------------------------------
    obj_verified = False
    if verify_fn is None:
        failures.append(("verify", "verify callback unavailable (fail-closed)."))
    else:
        try:
            vres = verify_fn(base_name, mode, object_name)
        except Exception as e:
            vres = {"result": "failure", "error": f"verify unavailable ({type(e).__name__})"}
        if is_disassembly:
            obj_verified, _ = P.is_disassembled(vres)
            g1_msg = f"'{object_name}' still in assembly. Disassembly incomplete."
        else:
            obj_verified, _ = P.is_assembled(vres)
            g1_msg = f"'{object_name}' not seated. Assembly incomplete."
        if not obj_verified:
            failures.append(("verify", g1_msg))

    # --- G2: released (gripper not holding the part) -----------------------------------
    # G2 is the only gate whose PASS condition is the predicate being False, so a plain
    # fail-closed (no-read -> "not holding") would fail OPEN. Require an explicit verdict
    # first: anything that is not a real {"result": success|failure} dict fails the gate.
    if verify_grasp_fn is None:
        failures.append(("released", "grasp callback unavailable (fail-closed)."))
    else:
        try:
            gres = verify_grasp_fn(object_name, mode)
        except Exception:
            gres = None
        if not (isinstance(gres, dict) and gres.get("result") in ("success", "failure")):
            failures.append(("released", f"grasp check unavailable for '{object_name}'. Retry."))
        else:
            holding, _ = P.is_object_grasped(gres)
            if holding:
                failures.append(("released", f"'{object_name}' still in the gripper. Release it."))

    # --- G3: safe height (arm retracted) -----------------------------------------------
    try:
        state = robot_state_fn(mode) if robot_state_fn is not None else None
    except Exception:
        state = None
    if state is None:
        failures.append(("safe_height", "robot state unavailable (fail-closed)."))
    else:
        at_safe, _ = P.is_at_safe_height(state)
        if not at_safe:
            failures.append(("safe_height", "Arm not at safe height. Move to safe height."))

    # NOTE: NO home gate here. The per-object protocol is uniform — every object ends at
    # SAFE HEIGHT (verify + released + safe height), no object is special. Home is enforced
    # only at phase transitions (signal_phase_complete at the end, onStart at the start),
    # so there is no per-object "final object" / rule-(c) detection.

    # --- G4: per-object checkpoint saved -----------------------------------------------
    # Opt-in: the server_core wrapper passes ordered_objects + checkpoints_fn ONLY when
    # MCP_CHECKPOINT_GATE is on (sim). When either is None the gate is OFF -> skipped, so
    # commit behaves exactly as before. DECISION delegated to predicates.is_checkpoint_saved
    # (k==0 in the phase-appropriate order => the shared 'init' baseline; else "{object}").
    if ordered_objects is not None and checkpoints_fn is not None:
        try:
            saved_checkpoints = checkpoints_fn()
        except Exception:
            saved_checkpoints = set()  # fail-closed: no readable checkpoints -> gate fails below
        ck_ok, ck_reason = P.is_checkpoint_saved(object_name, ordered_objects, saved_checkpoints)
        if not ck_ok:
            failures.append(("checkpoint", ck_reason))

    # --- compose envelope --------------------------------------------------------------
    if failures:
        return {
            "result": "failure",
            "requires_response": True,
            "message": " · ".join(msg for _, msg in failures),
            "failed_gates": [key for key, _ in failures],
        }
    return {
        "result": "success",
        "requires_response": False,
        "message": f"'{object_name}' committed.",
        "commit": True,
    }
