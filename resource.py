from mcp.server.fastmcp import FastMCP, Context
from mcp.server.session import ServerSession
from pydantic import BaseModel, Field
import json
import os
from pathlib import Path
from typing import Dict, Any, Optional

mcp = FastMCP("FMB Assembly and Disassembly Logbook")

# ========== DIRECTORY CONFIGURATION ==========
BASE_OUTPUT_DIR = os.getenv("MCP_CLIENT_OUTPUT_DIR", "").strip()
if BASE_OUTPUT_DIR:
    LOGS_DIR = Path(BASE_OUTPUT_DIR) / "logs"
else:
    LOGS_DIR = Path(__file__).parent / "logs"

LOGS_DIR.mkdir(parents=True, exist_ok=True)

# ========== MODE CONFIGURATION ==========
# Each mode defines its file prefix, order key, and type-specific required trial fields.
# Common required fields (trial_id, grasp_id, result) are always validated.
MODES = {
    "assembly": {
        "prefix": "Assembly",
        "order_key": "assembly_order",
        "trial_fields": {"tool_sequence"},
    },
    "disassembly": {
        "prefix": "Disassembly",
        "order_key": "disassembly_order",
        "trial_fields": {"gripper_state"},
    },
}


# ========== GENERIC HELPERS ==========

def _results_file(mode: str, assembly_id: str) -> Path:
    prefix = MODES[mode]["prefix"]
    return LOGS_DIR / f"{prefix}_{assembly_id}_results.json"


def _load_json(mode: str, assembly_id: str) -> dict:
    order_key = MODES[mode]["order_key"]
    default = {"assembly_id": assembly_id, "base_name": "", order_key: []}
    path = _results_file(mode, assembly_id)
    if not path.exists():
        return default
    try:
        content = path.read_text().strip()
        if not content:
            return default
        data = json.loads(content)
        data.setdefault("assembly_id", assembly_id)
        data.setdefault("base_name", "")
        data.setdefault(order_key, [])
        return data
    except (json.JSONDecodeError, Exception):
        return default


def _save_json(mode: str, assembly_id: str, data: dict) -> None:
    path = _results_file(mode, assembly_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2))


def _err(msg: str) -> dict:
    return {"success": False, "error": msg}


def _ok() -> dict:
    return {"success": True}


def _read_results(mode: str, assembly_id: str, result: Optional[str]) -> dict:
    """Shared read logic for both assembly and disassembly results."""
    order_key = MODES[mode]["order_key"]
    data = _load_json(mode, assembly_id)

    if result is None:
        return data

    result_lower = result.lower()
    if result_lower not in ("success", "failure"):
        return {"error": f"Invalid result filter: {result}. Must be 'success' or 'failure'"}

    filtered_order = []
    for item in data.get(order_key, []):
        matching = [t for t in item.get("trials", []) if t.get("result", "").lower() == result_lower]
        if matching:
            first = matching[0]
            entry = {
                order_key: item.get(order_key),
                "object_name": item.get("object_name"),
                "grasp_id": first.get("grasp_id"),
            }
            for field in MODES[mode]["trial_fields"]:
                entry[field] = first.get(field)
            filtered_order.append(entry)

    return {
        "assembly_id": data.get("assembly_id"),
        "base_name": data.get("base_name"),
        order_key: filtered_order,
    }


def _write_results(mode: str, assembly_id: str, base_name: str,
                   object_name: str, order_position: int, trial: Dict[str, Any]) -> dict:
    """Shared write logic for both assembly and disassembly results."""
    order_key = MODES[mode]["order_key"]

    try:
        data = _load_json(mode, assembly_id)
    except Exception as e:
        return _err(f"Error loading log: {str(e)}")

    # Validate order_position
    try:
        order_position = int(order_position)
    except (ValueError, TypeError):
        return _err(f"{order_key} must be an integer, got: {order_position}")

    # Validate base_name
    if not isinstance(base_name, str) or not base_name.strip():
        return _err("base_name must be a non-empty string")
    base_name = base_name.strip()

    existing_base = data.get("base_name", "").strip()
    if existing_base and existing_base != base_name:
        return _err(f"base_name mismatch: assembly already has base_name '{existing_base}', cannot change to '{base_name}'")

    data["base_name"] = base_name
    data["assembly_id"] = assembly_id

    # Parse trial from JSON string if needed
    if isinstance(trial, str):
        try:
            trial = json.loads(trial)
        except json.JSONDecodeError as e:
            return _err(f"Invalid JSON in trial: {str(e)}")

    if not isinstance(trial, dict):
        return _err(f"trial must be a dictionary, got: {type(trial).__name__}")

    # Validate required fields
    required_fields = {"trial_id", "grasp_id", "result"} | MODES[mode]["trial_fields"]
    missing = required_fields - set(trial.keys())
    if missing:
        return _err(f"trial missing required fields: {sorted(missing)}")

    # Validate trial_id
    try:
        trial_id = int(trial["trial_id"])
    except (ValueError, TypeError):
        return _err(f"trial_id must be an integer, got: {trial.get('trial_id')}")

    # Validate grasp_id
    try:
        grasp_id = int(trial["grasp_id"])
    except (ValueError, TypeError):
        return _err(f"grasp_id must be an integer, got: {trial.get('grasp_id')}")

    # Validate result
    result_val = trial.get("result", "").lower()
    if result_val not in ("success", "failure"):
        return _err(f"Invalid result: {trial.get('result')}. Must be 'success' or 'failure'")

    # Validate comment (optional)
    comment = trial.get("comment")
    if comment is not None and not isinstance(comment, str):
        return _err(f"comment must be a string, got: {type(comment).__name__}")

    # Build validated trial with common fields
    validated_trial = {
        "trial_id": trial_id,
        "grasp_id": grasp_id,
        "result": result_val,
    }
    if comment is not None:
        validated_trial["comment"] = comment

    # Mode-specific field validation and inclusion
    if mode == "assembly":
        tool_sequence = trial.get("tool_sequence")
        if not isinstance(tool_sequence, list):
            return _err(f"tool_sequence must be a list, got: {type(tool_sequence).__name__}")
        for i, tool in enumerate(tool_sequence):
            if not isinstance(tool, str):
                return _err(f"tool_sequence[{i}] must be a string, got: {type(tool).__name__}")
        validated_trial["tool_sequence"] = tool_sequence
    elif mode == "disassembly":
        gripper_state = trial.get("gripper_state")
        if gripper_state not in ("open", "half-open"):
            return _err(f"Invalid gripper_state: {gripper_state}. Must be 'open' or 'half-open'")
        validated_trial["gripper_state"] = gripper_state

    # Find or create object entry
    order_list = data.get(order_key, [])
    object_index = None
    for i, item in enumerate(order_list):
        if item.get("object_name") == object_name:
            if item.get(order_key) != order_position:
                return _err(
                    f"{order_key} mismatch: object '{object_name}' already has "
                    f"{order_key} {item.get(order_key)}, cannot change to {order_position}"
                )
            object_index = i
            break

    if object_index is not None:
        order_list[object_index]["trials"].append(validated_trial)
    else:
        order_list.append({
            order_key: order_position,
            "object_name": object_name,
            "trials": [validated_trial],
        })

    data[order_key] = order_list
    _save_json(mode, assembly_id, data)
    return _ok()


def _clear_results(mode: str, assembly_id: str) -> dict:
    """Shared clear logic."""
    path = _results_file(mode, assembly_id)
    if not path.exists():
        return {"success": False, "message": "Log not found"}
    try:
        path.unlink()
        return _ok()
    except Exception as e:
        return _err(str(e))


def _list_results(mode: str) -> dict:
    """Shared list logic."""
    prefix = MODES[mode]["prefix"]
    pattern = f"{prefix}_*_results.json"
    assembly_ids = []
    for f in LOGS_DIR.glob(pattern):
        name = f.stem
        if name.startswith(f"{prefix}_") and name.endswith("_results"):
            assembly_ids.append(name[len(prefix) + 1 : -len("_results")])
    return {"assembly_ids": assembly_ids, "count": len(assembly_ids)}


# ========== UNIFIED MCP TOOLS ==========

def _validate_task_type(task_type: str) -> Optional[dict]:
    """Return an error dict if task_type is invalid, else None."""
    if task_type not in MODES:
        return _err(f"Invalid task_type: '{task_type}'. Must be one of: {sorted(MODES.keys())}")
    return None


@mcp.tool()
def read_results(task_type: str, assembly_id: str, result: Optional[str] = None) -> dict:
    """
    Read assembly or disassembly results for a specific assembly.

    Args:
        task_type: "assembly" or "disassembly"
        assembly_id: The ID of the assembly
        result: Optional filter - "success" or "failure". If omitted, returns all trials.

    Returns:
        Dictionary containing assembly_id, base_name, and ordered trials.
        When result filter is applied, returns first matching trial per object.
    """
    if err := _validate_task_type(task_type):
        return err
    return _read_results(task_type, assembly_id, result)


@mcp.tool()
def write_assembly_results(assembly_id: str, base_name: str, object_name: str, assembly_order: int, trial: Dict[str, Any]) -> dict:
    """
    Write an assembly trial result for a specific object.

    Args:
        assembly_id: The ID of the assembly
        base_name: The name of the base object being assembled into
        object_name: The name of the object being assembled
        assembly_order: The fixed sequence position (assembly order)
        trial: Trial data with:
            - trial_id: integer (required)
            - grasp_id: integer (required)
            - tool_sequence: list of strings (required) - the exact MCP tool calls
              made during the trial, in order. Format each entry as
              "server__tool_name(key = 'value', key2 = 'value2')"
            - result: "success" or "failure" (required)
            - comment: string (optional)

    Returns:
        Dictionary with confirmation or error message
    """
    return _write_results("assembly", assembly_id, base_name, object_name, assembly_order, trial)


@mcp.tool()
def write_disassembly_results(assembly_id: str, base_name: str, object_name: str, disassembly_order: int, trial: Dict[str, Any]) -> dict:
    """
    Write a disassembly trial result for a specific object.

    Args:
        assembly_id: The ID of the assembly
        base_name: The name of the base object being disassembled from
        object_name: The name of the object being disassembled
        disassembly_order: The disassembly order position (which object to disassemble in sequence)
        trial: Trial data with:
            - trial_id: integer (required)
            - grasp_id: integer (required)
            - gripper_state: "open" or "half-open" (required) (which gripper state was used to access the object)
            - result: "success" or "failure" (required)
            - comment: string (optional)

    Returns:
        Dictionary with confirmation or error message
    """
    return _write_results("disassembly", assembly_id, base_name, object_name, disassembly_order, trial)


@mcp.tool()
def clear_results(task_type: str, assembly_id: str) -> dict:
    """
    Clear/delete all results for an assembly.

    Args:
        task_type: "assembly" or "disassembly"
        assembly_id: The ID of the assembly to clear

    Returns:
        Dictionary with confirmation or error message
    """
    if err := _validate_task_type(task_type):
        return err
    return _clear_results(task_type, assembly_id)


@mcp.tool()
def list_results(task_type: str) -> dict:
    """
    List all assemblies that have results.

    Args:
        task_type: "assembly" or "disassembly"

    Returns:
        Dictionary containing all assembly IDs with results
    """
    if err := _validate_task_type(task_type):
        return err
    return _list_results(task_type)


# ========== ELICITATION TOOL FOR LOG MANAGEMENT ==========

class ClearLogsConfirmation(BaseModel):
    """Schema for confirming log deletion."""
    confirm_delete: bool = Field(
        default=False,
        description="Delete ALL listed log files? This cannot be undone."
    )

@mcp.tool()
async def clear_all_logs(ctx: Context[ServerSession, None]) -> dict:
    """
    Clear/delete all log files with user confirmation via elicitation.

    Shows the user a list of existing log files and asks for confirmation before deleting.
    This includes disassembly results and assembly results.

    Returns:
        Dictionary with deletion result
    """
    # Get all log files
    log_patterns = [
        "Disassembly_*_results.json",
        "Assembly_*_results.json",
    ]

    all_files = []
    for pattern in log_patterns:
        files = list(LOGS_DIR.glob(pattern))
        all_files.extend([f.name for f in files])

    all_files.sort()

    if not all_files:
        return {"status": "info", "message": "No log files found in logs directory", "logs_dir": str(LOGS_DIR)}

    file_list = "\n".join(f"  - {f}" for f in all_files)

    try:
        result = await ctx.elicit(
            message=f"Found {len(all_files)} log file(s):\n{file_list}\n\nWould you like to delete them all?",
            schema=ClearLogsConfirmation
        )

        if result.action == "accept" and result.data:
            if not result.data.confirm_delete:
                return {"status": "cancelled", "message": "Deletion cancelled by user"}

            deleted_files = []
            errors = []
            for filename in all_files:
                try:
                    (LOGS_DIR / filename).unlink()
                    deleted_files.append(filename)
                except Exception as e:
                    errors.append(f"{filename}: {str(e)}")

            return {
                "status": "success" if not errors else "partial",
                "deleted_files": deleted_files,
                "errors": errors if errors else None,
                "message": f"Deleted {len(deleted_files)} file(s)",
            }

        else:
            return {"status": "cancelled", "message": "Elicitation declined or cancelled", "files_found": all_files}

    except Exception as e:
        error_msg = str(e)
        if "Method not found" in error_msg:
            return {
                "status": "error",
                "message": "Elicitation not supported by this client",
                "files_found": all_files,
                "hint": "Use clear_assembly_results or clear_disassembly_results tools directly",
            }
        return {"status": "error", "message": f"Elicitation failed: {error_msg}", "files_found": all_files}


if __name__ == "__main__":
    mcp.run()
