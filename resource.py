from mcp.server.fastmcp import FastMCP
from pydantic import BaseModel, Field
import json
import os
from pathlib import Path
from typing import Annotated, Literal, Optional, List

from utils.data_path_finder import get_assembly_data_dir, get_aruco_data_dir

# Assembly IDs available in the FMB benchmark
AssemblyId = Literal["fmb_assembly_1", "fmb_assembly_2", "fmb_assembly_3"]

mcp = FastMCP("FMB Assembly Resources")




# ========== DIRECTORY CONFIGURATION ==========
BASE_OUTPUT_DIR = os.getenv("MCP_CLIENT_OUTPUT_DIR", "").strip()
if BASE_OUTPUT_DIR:
    LOGS_DIR = Path(BASE_OUTPUT_DIR) / "logs"
else:
    LOGS_DIR = Path(__file__).parent / "logs"

LOGS_DIR.mkdir(parents=True, exist_ok=True)

GROUND_TRUTH_DIR = Path(__file__).parent / "ablations" / "ground_truth_resources"

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


# ========== ASSEMBLY CONFIG HELPERS ==========

def _assembly_id_to_filename(assembly_id: str) -> str:
    """Convert assembly_id (e.g. 'fmb_assembly_1') to JSON filename (e.g. 'fmb_assembly1.json').

    Convention: strip the underscore before the trailing digit(s).
    """
    # "fmb_assembly_1" → "fmb_assembly" + "1" → "fmb_assembly1"
    parts = assembly_id.rsplit("_", 1)
    return f"{''.join(parts)}.json"


def _load_assembly_config(assembly_id: str) -> dict:
    """Load assembly config JSON for an assembly_id. Returns {} on failure."""
    try:
        data_dir = get_assembly_data_dir()
    except FileNotFoundError:
        return {}
    path = data_dir / _assembly_id_to_filename(assembly_id)
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return {}


def _assembly_id_to_base_name(assembly_id: str) -> Optional[str]:
    """Find the base (board) component name for an assembly_id."""
    config = _load_assembly_config(assembly_id)
    for comp in config.get("components", []):
        if comp.get("type") == "board":
            return comp["name"]
    return None


def _load_grasp_points(object_name: str) -> list:
    """Load grasp IDs and gripper widths for an object from grasp points JSON.

    Returns list of {id, gripper_width_mm} dicts.
    """
    try:
        data_dir = get_aruco_data_dir() / "grasp_points"
    except FileNotFoundError:
        return []

    grasp_file = data_dir / f"{object_name}_grasp_points.json"
    if not grasp_file.exists():
        matches = list(data_dir.glob(f"{object_name}*_grasp_points.json"))
        if matches:
            grasp_file = matches[0]
        else:
            return []

    try:
        with open(grasp_file, 'r') as f:
            data = json.load(f)

        points = []
        for gp in data.get('grasp_points', []):
            gp_id = gp.get('id', 0)
            grasp_validity = gp.get('grasp_validity', {})
            w_x = grasp_validity.get('x_axis_gripper_width_mm')
            # w_y = grasp_validity.get('y_axis_gripper_width_mm')
            if w_x is not None:
                points.append({"id": gp_id, "gripper_width_mm": w_x})
        return points
    except Exception:
        return []


# ========== MCP RESOURCES ==========

@mcp.resource("assembly://{assembly_id}/objects")
def assembly_objects(assembly_id: str) -> str:
    """Objects in this assembly with their assembly order, subtypes, and grasp points.

    Returns:
        assembly_id: assembly identifier
        base_name: name of the fixed base (board)
        objects: list sorted by assembly_order, each containing:
            name: object name
            subtype: object geometry — "block", "peg", or "socket"
            assembly_order: position in assembly sequence
            grasp_points: list of {id, gripper_width_mm}
                gripper_width_mm: width in mm to open gripper before grasping and after releasing at this point.
                    Includes tolerance — do not open beyond this value."""
    config = _load_assembly_config(assembly_id)
    if not config:
        return json.dumps({"error": f"Assembly config not found for '{assembly_id}'"})

    base_name = None
    objects = []
    for comp in config.get("components", []):
        if comp.get("type") == "board":
            base_name = comp["name"]
            continue
        obj_name = comp["name"]
        objects.append({
            "name": obj_name,
            "subtype": comp.get("subtype", ""),
            "assembly_order": comp.get("assembly_order"),
            "grasp_points": _load_grasp_points(obj_name),
        })

    objects.sort(key=lambda o: o.get("assembly_order", 0))
    return json.dumps({
        "assembly_id": assembly_id,
        "base_name": base_name,
        "objects": objects,
    }, indent=2)


@mcp.resource("assembly://{assembly_id}/ground_truth/{task_type}")
def ground_truth_resource(assembly_id: str, task_type: str) -> str:
    """Ground truth results for this assembly (static, verified sequences).

    Same schema as results_resource but reads from ablations/ground_truth_resources/.
    Used in Mode 1 isolated experiments to inject fixed inputs independent of
    prior phase performance.

    task_type: "disassembly" or "assembly"

    Returns: same structure as results_resource."""
    if task_type not in MODES:
        return json.dumps({"error": f"Invalid task_type: '{task_type}'. Must be one of: {sorted(MODES.keys())}"})
    prefix = MODES[task_type]["prefix"]
    path = GROUND_TRUTH_DIR / f"{prefix}_{assembly_id}_results.json"
    if not path.exists():
        return json.dumps({"error": f"Ground truth not found: {path.name}"})
    try:
        return path.read_text()
    except OSError as e:
        return json.dumps({"error": str(e)})


@mcp.resource("assembly://{assembly_id}/ground_truth_ablation_quat/{task_type}")
def ground_truth_ablation_quat_resource(assembly_id: str, task_type: str) -> str:
    """Ground truth with orientation placeholders for quaternion accuracy ablation.

    Same as ground_truth_resource but tool sequences include
    current_object_orientation and grasp_id as placeholder params that the
    agent must fill from live tool outputs.

    task_type: "disassembly" or "assembly"

    Returns: same structure as results_resource."""
    if task_type not in MODES:
        return json.dumps({"error": f"Invalid task_type: '{task_type}'. Must be one of: {sorted(MODES.keys())}"})
    prefix = MODES[task_type]["prefix"]
    path = GROUND_TRUTH_DIR / "ablation_quat" / f"{prefix}_{assembly_id}_results.json"
    if not path.exists():
        return json.dumps({"error": f"Ground truth not found: {path.name}"})
    try:
        return path.read_text()
    except OSError as e:
        return json.dumps({"error": str(e)})


@mcp.resource("assembly://{assembly_id}/results/{task_type}")
def results_resource(assembly_id: str, task_type: str) -> str:
    """Results logged during assembly or disassembly for this assembly.

    task_type: "disassembly" or "assembly"

    Returns (disassembly):
        assembly_id, base_name
        disassembly_order: list of completed objects, each containing:
            disassembly_order: sequence position
            object_name, grasp_id
            comment: optional note

    Returns (assembly):
        assembly_id, base_name
        assembly_order: list of completed objects, each containing:
            assembly_order: sequence position
            object_name
            tool_sequence: MCP tool calls executed in order
            comment: optional note"""
    if task_type not in MODES:
        return json.dumps({"error": f"Invalid task_type: '{task_type}'. Must be one of: {sorted(MODES.keys())}"})
    return json.dumps(_read_results(task_type, assembly_id), indent=2)


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
    return {"result": "failure", "error": msg}


def _ok() -> dict:
    return {"result": "success"}


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
        # Object already exists — overwrite
        order_list[obj_idx] = entry
    else:
        order_list.append(entry)

    data[order_key] = order_list
    _save_json(mode, assembly_id, data)

    # Clear any orchestrator replay rejection for this object
    if mode == "assembly":
        try:
            from triggers.signal_verify_results import clear_rejection
            clear_rejection(object_name)
        except ImportError:
            pass  # signal_verify_results not available (e.g. standalone resource server)

    return _ok()



def _clear_results(mode: str, assembly_id: str) -> dict:
    """Shared clear logic."""
    path = _results_file(mode, assembly_id)
    if not path.exists():
        return _err("Log not found")
    try:
        path.unlink()
        return _ok()
    except Exception as e:
        return _err(str(e))



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

class DisassemblyEntry(BaseModel):
    """A single object entry in the disassembly order list."""
    disassembly_order: int = Field(description="The sequence position in the disassembly")
    object_name: str
    grasp_id: int
    comment: Optional[str] = None

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
def write_assembly_results(
    assembly_id: AssemblyId,
    base_name: str,
    object_name: str,
    assembly_order: int,
    tool_sequence: Annotated[List[str], Field(description='MCP tool calls in order. Format each entry as "server__tool_name(key=\'value\', key2=\'value2\')".')],
    comment: str = "",
) -> dict:
    """Log a successful assembly result for an object. Only call this when assembly verification succeeds.

    Returns:
        result: "success" or "failure"
        error: validation error message (only on failure)"""
    return _write_results("assembly", assembly_id, base_name, object_name,
                          assembly_order, comment=comment or None,
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
        result: "success" or "failure"
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
        result: "success" or "failure"
        error: failure reason (only on failure)"""
    if err := _validate_task_type(task_type):
        return err
    return _clear_results(task_type, assembly_id)



if __name__ == "__main__":
    mcp.run()
