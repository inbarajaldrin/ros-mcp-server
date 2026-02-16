"""Pre-assembly check - invoke human to fix scene when clearance verification fails.

Related files (clearance check pipeline):
  - queries/verify_clearance.py    — ROS query that checks poses and computes clearance
  - triggers/pre_assembly_check.py (this file) — handles failure by invoking human elicitation
  - elicitations/fix_scene.py      — Pydantic schemas for human interaction

When verify_clearance query returns failure (missing objects or clearance issues),
this trigger loads the fix_scene elicitation schema and asks the human
to fix the issue. Supports retry (re-run query) and skip (proceed anyway).
"""

import os
import importlib.util
import logging
from typing import Dict, Any
from mcp.server.fastmcp import Context
from mcp.server.session import ServerSession

logger = logging.getLogger("RosMCPServer")


async def handle_clearance_failure(
    result: Dict[str, Any],
    base_name: str,
    mode: str,
    ctx: Context[ServerSession, None],
    elicit_user_fn,
    run_query_fn,
) -> Dict[str, Any]:
    """Handle clearance failure by invoking human elicitation.

    Args:
        result: The failed verification result from the query.
        base_name: Name of the base object.
        mode: Robot mode ("sim" or "real").
        ctx: MCP Context for elicitation.
        elicit_user_fn: Reference to _handle_elicitation from server.py.
        run_query_fn: Callable to re-run the query on retry.

    Returns:
        Updated result dict with elicitation outcome.
    """
    has_missing = "missing_objects" in result
    has_clearance_issues = "objects_with_clearance_issues" in result

    if not has_missing and not has_clearance_issues:
        return result

    try:
        # Dynamically load fix_scene elicitation module
        script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        fix_scene_path = os.path.join(script_dir, "elicitations", "fix_scene.py")

        spec = importlib.util.spec_from_file_location(
            "elicitations.fix_scene", fix_scene_path
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        # Build message using the elicitation's message builder
        message = module.build_elicitation_message(result)

        # Call elicitation handler with result context for dynamic schema
        elicit_response = await elicit_user_fn(ctx, "fix_scene", message, result)

        if elicit_response.get("action") == "accept":
            user_action = elicit_response.get("user_action")

            if user_action == "retry":
                result = run_query_fn(base_name, mode)
                result["retried"] = True
            elif user_action == "skip":
                result["skipped"] = True
                result["warning"] = "Proceeding without proper environment setup may cause assembly failures"

        elif elicit_response.get("action") == "decline":
            result["declined"] = True
            result["result"] = "declined"

    except Exception as e:
        if "Method not found" not in str(e):
            logger.warning(f"Elicitation failed: {e}")

    return result
