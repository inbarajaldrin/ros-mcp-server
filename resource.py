from mcp.server.fastmcp import FastMCP, Context
from mcp.server.session import ServerSession
from pydantic import BaseModel, Field
import json
import os
from pathlib import Path
from typing import Annotated, Dict, Any, Literal, Optional, List

# Assembly IDs available in the FMB benchmark
AssemblyId = Literal["fmb_assembly_1", "fmb_assembly_2", "fmb_assembly_3"]

mcp = FastMCP("FMB Assembly and Disassembly Logbook")




# ========== DIRECTORY CONFIGURATION ==========
BASE_OUTPUT_DIR = os.getenv("MCP_CLIENT_OUTPUT_DIR", "").strip()
if BASE_OUTPUT_DIR:
    LOGS_DIR = Path(BASE_OUTPUT_DIR) / "logs"
else:
    LOGS_DIR = Path(__file__).parent / "logs"

LOGS_DIR.mkdir(parents=True, exist_ok=True)

# ========== MODE CONFIGURATION ==========
# Each mode defines its file prefix, order key, and type-specific required fields.
# Common required field: grasp_id. Only successful results are logged.
MODES = {
    "assembly": {
        "prefix": "Assembly",
        "order_key": "assembly_order",
        "extra_fields": {"tool_sequence"},
        "has_grasp_id": False,
    },
    "disassembly": {
        "prefix": "Disassembly",
        "order_key": "disassembly_order",
        "extra_fields": set(),
        "has_grasp_id": True,
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


def _validate_base_name(data: dict, base_name: str) -> Optional[dict]:
    """Validate and check base_name consistency. Returns error dict or None."""
    if not isinstance(base_name, str) or not base_name.strip():
        return _err("base_name must be a non-empty string")
    existing_base = data.get("base_name", "").strip()
    if existing_base and existing_base != base_name.strip():
        return _err(f"base_name mismatch: assembly already has base_name '{existing_base}', cannot change to '{base_name.strip()}'")
    return None


def _validate_grasp_id(grasp_id) -> tuple[Optional[int], Optional[dict]]:
    """Validate grasp_id. Returns (validated_id, None) or (None, error_dict)."""
    try:
        return int(grasp_id), None
    except (ValueError, TypeError):
        return None, _err(f"grasp_id must be an integer, got: {grasp_id}")


def _validate_assembly_extras(tool_sequence) -> Optional[dict]:
    """Validate assembly-specific fields. Returns error dict or None."""
    if not isinstance(tool_sequence, list):
        return _err(f"tool_sequence must be a list, got: {type(tool_sequence).__name__}")
    for i, tool in enumerate(tool_sequence):
        if not isinstance(tool, str):
            return _err(f"tool_sequence[{i}] must be a string, got: {type(tool).__name__}")
    return None


def _validate_disassembly_extras() -> Optional[dict]:
    """Validate disassembly-specific fields. Returns error dict or None."""
    return None


def _find_object(order_list: list, object_name: str) -> Optional[int]:
    """Find index of object in order list by name. Returns index or None."""
    for i, item in enumerate(order_list):
        if item.get("object_name") == object_name:
            return i
    return None


# ========== CORE LOGIC ==========

def _read_results(mode: str, assembly_id: str) -> dict:
    """Read results for an assembly. Returns active entries (no previous history)."""
    return _load_json(mode, assembly_id)


def _write_results(mode: str, assembly_id: str, base_name: str,
                   object_name: str, order_position: int,
                   comment: Optional[str] = None, **extra_fields) -> dict:
    """Write a successful result for an object. Only call for successful outcomes."""
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
    if err := _validate_base_name(data, base_name):
        return err
    base_name = base_name.strip()

    # Validate grasp_id if this mode requires it
    if MODES[mode]["has_grasp_id"]:
        grasp_id = extra_fields.get("grasp_id")
        if grasp_id is None:
            return _err("grasp_id is required for disassembly results")
        grasp_id, err = _validate_grasp_id(grasp_id)
        if err:
            return err
        extra_fields["grasp_id"] = grasp_id

    # Validate mode-specific fields
    if mode == "assembly":
        if err := _validate_assembly_extras(extra_fields.get("tool_sequence")):
            return err
    data["base_name"] = base_name
    data["assembly_id"] = assembly_id

    # Build entry
    entry = {
        order_key: order_position,
        "object_name": object_name,
    }
    if MODES[mode]["has_grasp_id"]:
        entry["grasp_id"] = extra_fields["grasp_id"]
    for field in MODES[mode]["extra_fields"]:
        entry[field] = extra_fields[field]
    if comment:
        entry["comment"] = comment
    entry["previous"] = []

    # Find or create object entry
    order_list = data.get(order_key, [])
    obj_idx = _find_object(order_list, object_name)

    if obj_idx is not None:
        existing = order_list[obj_idx]
        if existing.get(order_key) != order_position:
            return _err(
                f"{order_key} mismatch: object '{object_name}' already has "
                f"{order_key} {existing.get(order_key)}, cannot change to {order_position}"
            )
        # Object already exists — overwrite (move current to previous)
        prev = existing.get("previous", [])
        # Archive current active fields
        archived = {k: v for k, v in existing.items() if k not in ("previous",)}
        prev.append(archived)
        entry["previous"] = prev
        order_list[obj_idx] = entry
    else:
        order_list.append(entry)

    data[order_key] = order_list
    _save_json(mode, assembly_id, data)
    return _ok()


def _update_results(mode: str, assembly_id: str, object_name: str,
                    comment: Optional[str] = None, **extra_fields) -> dict:
    """Update an existing object's result. Moves current to previous, sets new values."""
    order_key = MODES[mode]["order_key"]

    try:
        data = _load_json(mode, assembly_id)
    except Exception as e:
        return _err(f"Error loading log: {str(e)}")

    order_list = data.get(order_key, [])
    obj_idx = _find_object(order_list, object_name)

    if obj_idx is None:
        return _err(f"Object '{object_name}' not found in {mode} results for '{assembly_id}'")

    existing = order_list[obj_idx]

    # Validate mode-specific fields if provided
    if mode == "assembly" and "tool_sequence" in extra_fields:
        if err := _validate_assembly_extras(extra_fields["tool_sequence"]):
            return err
    # Archive current to previous
    prev = existing.get("previous", [])
    archived = {k: v for k, v in existing.items() if k not in ("previous",)}
    prev.append(archived)

    # Build updated entry — keep existing values for fields not provided
    updated = {
        order_key: existing[order_key],
        "object_name": object_name,
    }
    if MODES[mode]["has_grasp_id"]:
        updated["grasp_id"] = existing.get("grasp_id")
    for field in MODES[mode]["extra_fields"]:
        if field in extra_fields:
            updated[field] = extra_fields[field]
        else:
            updated[field] = existing.get(field)
    if comment is not None:
        updated["comment"] = comment
    elif "comment" in existing:
        updated["comment"] = existing["comment"]
    updated["previous"] = prev

    order_list[obj_idx] = updated
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


class AssemblyEntry(BaseModel):
    """A single object entry in the assembly order list."""
    assembly_order: int = Field(description="The sequence position in the assembly")
    object_name: str
    tool_sequence: List[str] = Field(description="MCP tool calls made in order")
    comment: Optional[str] = None
    previous: List[dict] = Field(default_factory=list, description="Archived older versions of this entry")

class DisassemblyEntry(BaseModel):
    """A single object entry in the disassembly order list."""
    disassembly_order: int = Field(description="The sequence position in the disassembly")
    object_name: str
    grasp_id: int
    comment: Optional[str] = None
    previous: List[dict] = Field(default_factory=list, description="Archived older versions of this entry")

class AssemblyResults(BaseModel):
    """Return schema for reading assembly results."""
    assembly_id: str
    base_name: str
    assembly_order: List[AssemblyEntry] = Field(default_factory=list, description="Ordered list of assembly results")

class DisassemblyResults(BaseModel):
    """Return schema for reading disassembly results."""
    assembly_id: str
    base_name: str
    disassembly_order: List[DisassemblyEntry] = Field(default_factory=list, description="Ordered list of disassembly results")

@mcp.tool()
def read_results(
    task_type: Literal["assembly", "disassembly"],
    assembly_id: AssemblyId,
) -> AssemblyResults | DisassemblyResults:
    """Read assembly or disassembly results for a specific assembly.

    Returns:
        assembly_id: the assembly identifier
        base_name: the assembly base
        assembly_order or disassembly_order: list of entries, each with object_name, order position, grasp_id (disassembly only), tool_sequence (assembly only), comment, and previous (archived versions)"""
    if err := _validate_task_type(task_type):
        return err
    return _read_results(task_type, assembly_id)


@mcp.tool()
def write_assembly_results(
    assembly_id: AssemblyId,
    base_name: str,
    object_name: str,
    assembly_order: int,
    tool_sequence: Annotated[List[str], Field(description='The exact MCP tool calls made in order. Format each entry as "server__tool_name(key = \'value\', key2 = \'value2\')". The grasp_id is captured within the tool calls themselves.')],
    comment: str = "",
) -> dict:
    """Log a successful assembly result for an object. Only call this when assembly verification succeeds.

    Returns:
        success: True if logged successfully, False on validation error
        error: validation error message (only on failure)"""
    return _write_results("assembly", assembly_id, base_name, object_name,
                          assembly_order, comment=comment or None,
                          tool_sequence=tool_sequence)


@mcp.tool()
def update_assembly_results(
    assembly_id: AssemblyId,
    object_name: str,
    tool_sequence: List[str],
    comment: str = "",
) -> dict:
    """Update an existing object's assembly result with a corrected sequence. Use this when reverification finds a better or corrected tool sequence.

    Returns:
        success: True if updated successfully, False on validation error
        error: validation error message (only on failure)"""
    return _update_results("assembly", assembly_id, object_name,
                           comment=comment or None,
                           tool_sequence=tool_sequence)


@mcp.tool()
def write_disassembly_results(
    assembly_id: AssemblyId,
    base_name: str,
    object_name: str,
    disassembly_order: int,
    grasp_id: int,
    comment: str = "",
) -> dict:
    """Log a successful disassembly result for an object. Only call this when disassembly verification succeeds.

    Returns:
        success: True if logged successfully, False on validation error
        error: validation error message (only on failure)"""
    return _write_results("disassembly", assembly_id, base_name, object_name,
                          disassembly_order, comment=comment or None,
                          grasp_id=grasp_id)


@mcp.tool()
def clear_results(
    task_type: Literal["assembly", "disassembly"],
    assembly_id: AssemblyId,
) -> dict:
    """Clear all results for an assembly.

    Returns:
        success: True if cleared, False if log not found or deletion failed
        error: failure reason (only on failure)"""
    if err := _validate_task_type(task_type):
        return err
    return _clear_results(task_type, assembly_id)


class ResultListResponse(BaseModel):
    """Return schema for listing assemblies."""
    assembly_ids: List[str] = Field(description="IDs of assemblies that have results")
    count: int

@mcp.tool()
def list_results(
    task_type: Literal["assembly", "disassembly"],
) -> ResultListResponse:
    """List all assemblies that have results.

    Returns:
        assembly_ids: list of assembly IDs that have results
        count: number of assemblies found"""
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
    """Clear all log files with user confirmation via elicitation. Shows existing log files and asks for confirmation before deleting.

    Returns:
        status: "success", "partial", "cancelled", "info", or "error"
        message: description of what happened
        deleted_files: list of filenames deleted (on success)
        errors: list of deletion errors (on partial failure)
        files_found: list of log files found (on cancel/error)"""
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
                "hint": "Use clear_results tool directly",
            }
        return {"status": "error", "message": f"Elicitation failed: {error_msg}", "files_found": all_files}


if __name__ == "__main__":
    mcp.run()
