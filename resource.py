from mcp.server.fastmcp import FastMCP, Context
from mcp.server.session import ServerSession
from pydantic import BaseModel, Field
import json
import os
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

mcp = FastMCP("FMB Assembly and Disassembly Resources")

# ========== DIRECTORY CONFIGURATION ==========
# Output directories - use MCP_CLIENT_OUTPUT_DIR if set, otherwise use relative paths
# Directories are created lazily when needed by tools
BASE_OUTPUT_DIR = os.getenv("MCP_CLIENT_OUTPUT_DIR", "").strip()
if BASE_OUTPUT_DIR:
    RESOURCES_DIR = Path(BASE_OUTPUT_DIR) / "resources"
else:
    RESOURCES_DIR = Path(__file__).parent / "resources"

# Ensure resources directory exists
RESOURCES_DIR.mkdir(parents=True, exist_ok=True)

# ========== HELPER FUNCTIONS ==========

def normalize_assembly_id(assembly_id: str) -> str:
    """Normalize assembly_id by removing any 'Assembly' or 'assembly' prefix"""
    # Remove 'Assembly' or 'assembly' prefix if present
    if assembly_id.startswith("Assembly"):
        normalized = assembly_id[8:]  # Remove "Assembly" (8 chars)
    elif assembly_id.startswith("assembly"):
        normalized = assembly_id[8:]  # Remove "assembly" (8 chars)
    else:
        normalized = assembly_id
    
    # Remove leading underscores if any
    normalized = normalized.lstrip("_")
    
    return normalized

def validate_assembly_id(assembly_id: str) -> Tuple[bool, str]:
    """
    Validate that assembly_id is a pure numeric string (e.g., "3").
    Rejects inputs with prefixes, spaces, or non-numeric characters.
    
    Returns:
        (is_valid, error_message): Tuple where is_valid is True if valid, False otherwise.
                                   error_message is empty if valid, otherwise contains error description.
    """
    # Strip whitespace
    assembly_id = assembly_id.strip()
    
    # Check if empty
    if not assembly_id:
        return False, "assembly_id cannot be empty"
    
    # Check if it contains "Assembly" or "assembly" prefix (not allowed)
    if assembly_id.lower().startswith("assembly"):
        return False, f"assembly_id must be a numeric ID only, not a prefix (got: '{assembly_id}'). Use format like '3'"
    
    # Check if it contains spaces
    if " " in assembly_id:
        return False, f"assembly_id must be numeric only and cannot contain spaces (got: '{assembly_id}'). Use format like '3'"
    
    # Check if it's purely numeric (only digits)
    if not assembly_id.isdigit():
        return False, f"assembly_id must be a numeric ID only (got: '{assembly_id}'). Use format like '3'"
    
    return True, ""


def get_assembly_results_file(assembly_id: str) -> Path:
    """Get the assembly results file path for a specific assembly"""
    normalized_id = normalize_assembly_id(assembly_id)
    return RESOURCES_DIR / f"Assembly_{normalized_id}_results.json"


def get_disassembly_tool_sequence_file() -> Path:
    """Get the global disassembly tool sequence file path (no assembly_id needed)"""
    return RESOURCES_DIR / "disassembly_tool_sequence.json"

def load_disassembly_tool_sequence():
    """Load disassembly tool sequence from global JSON file.

    Returns list of tool call strings:
    ["move_to_object", "move_down", "control_gripper --command close", ...]
    """
    tool_seq_file = get_disassembly_tool_sequence_file()
    if not tool_seq_file.exists():
        return []

    try:
        with open(tool_seq_file, 'r') as f:
            content = f.read().strip()
            if not content:
                return []
            data = json.loads(content)
            if isinstance(data, list):
                return data
            return []
    except json.JSONDecodeError:
        return []
    except Exception:
        return []

def save_disassembly_tool_sequence(data):
    """Save disassembly tool sequence to global JSON file"""
    tool_seq_file = get_disassembly_tool_sequence_file()
    with open(tool_seq_file, 'w') as f:
        json.dump(data, f, indent=2)

def get_disassembly_results_file(assembly_id: str) -> Path:
    """Get the disassembly results file path for a specific assembly"""
    normalized_id = normalize_assembly_id(assembly_id)
    return RESOURCES_DIR / f"Disassembly_{normalized_id}_results.json"

def load_disassembly_results(assembly_id: str):
    """Load disassembly results from JSON file for a specific assembly.

    Returns dict with structure:
    {
        "assembly_id": "3",
        "base_name": "base3",
        "disassembly_order": [
            {
                "assembly_order": 1,
                "object_name": "line_brown",
                "trials": [
                    {"trial_id": 1, "grasp_id": 1, "gripper_state": "open", "result": "success"}
                ]
            }
        ]
    }
    """
    results_file = get_disassembly_results_file(assembly_id)
    if not results_file.exists():
        return {"assembly_id": assembly_id, "base_name": "", "disassembly_order": []}

    try:
        with open(results_file, 'r') as f:
            content = f.read().strip()
            if not content:
                return {"assembly_id": assembly_id, "base_name": "", "disassembly_order": []}
            data = json.loads(content)
            # Ensure required keys exist
            if "assembly_id" not in data:
                data["assembly_id"] = assembly_id
            if "base_name" not in data:
                data["base_name"] = ""
            if "disassembly_order" not in data:
                data["disassembly_order"] = []
            return data
    except json.JSONDecodeError:
        return {"assembly_id": assembly_id, "base_name": "", "disassembly_order": []}
    except Exception:
        return {"assembly_id": assembly_id, "base_name": "", "disassembly_order": []}

def save_disassembly_results(assembly_id: str, data):
    """Save disassembly results to JSON file for a specific assembly"""
    results_file = get_disassembly_results_file(assembly_id)
    with open(results_file, 'w') as f:
        json.dump(data, f, indent=2)

def load_assembly_results(assembly_id: str):
    """Load assembly results from JSON file for a specific assembly.

    Returns dict with structure:
    {
        "assembly_id": "3",
        "base_name": "base3",
        "assembly_order": [
            {
                "assembly_order": 1,
                "object_name": "line_brown",
                "trials": [
                    {"trial_id": 1, "grasp_id": 1, "tool_sequence": [...], "result": "success", "comment": "..."}
                ]
            }
        ]
    }
    """
    results_file = get_assembly_results_file(assembly_id)
    if not results_file.exists():
        return {"assembly_id": assembly_id, "base_name": "", "assembly_order": []}

    try:
        with open(results_file, 'r') as f:
            content = f.read().strip()
            if not content:
                return {"assembly_id": assembly_id, "base_name": "", "assembly_order": []}
            data = json.loads(content)
            # Ensure required keys exist
            if "assembly_id" not in data:
                data["assembly_id"] = assembly_id
            if "base_name" not in data:
                data["base_name"] = ""
            if "assembly_order" not in data:
                data["assembly_order"] = []
            return data
    except json.JSONDecodeError:
        return {"assembly_id": assembly_id, "base_name": "", "assembly_order": []}
    except Exception:
        return {"assembly_id": assembly_id, "base_name": "", "assembly_order": []}

def save_assembly_results(assembly_id: str, data):
    """Save assembly results to JSON file for a specific assembly"""
    results_file = get_assembly_results_file(assembly_id)
    with open(results_file, 'w') as f:
        json.dump(data, f, indent=2)

def get_knowledge_file() -> Path:
    """Get the knowledge observations file path (global, not per-assembly)"""
    return RESOURCES_DIR / "knowledge_observations.json"

def load_knowledge():
    """Load knowledge observations from JSON file.

    Returns dict with structure:
    {
        "observations": [
            {
                "obs_id": "obs_001",
                "observation": "Objects with 90° orientation offset require regrasp",
                "status": "hypothesis",
                "assembly_id": "3"
            }
        ]
    }
    """
    knowledge_file = get_knowledge_file()
    if not knowledge_file.exists():
        return {"observations": []}

    try:
        with open(knowledge_file, 'r') as f:
            content = f.read().strip()
            if not content:
                return {"observations": []}
            data = json.loads(content)
            # Ensure required keys exist
            if "observations" not in data:
                data["observations"] = []
            return data
    except json.JSONDecodeError:
        return {"observations": []}
    except Exception:
        return {"observations": []}

def save_knowledge(data):
    """Save knowledge observations to JSON file"""
    knowledge_file = get_knowledge_file()
    with open(knowledge_file, 'w') as f:
        json.dump(data, f, indent=2)


# ========== TOOLS ==========

# ========== ASSEMBLY RESULTS TOOLS ==========

@mcp.tool()
def read_assembly_results(assembly_id: str, result: Optional[str] = None) -> str:
    """
    Read assembly results for a specific assembly.

    Args:
        assembly_id: The ID of the assembly
        result: Optional filter - "success" or "failure". If omitted, returns all trials.

    Returns:
        JSON string containing:
        - assembly_id
        - base_name
        - sequence: list of {sequence_id, object_name, trials or filtered trial}

        When result filter is applied, returns first matching trial per object with:
        {sequence_id, object_name, grasp_id, tool_sequence}
    """
    data = load_assembly_results(assembly_id)

    if result is None:
        # Return all trials
        return json.dumps(data, indent=2)

    result_lower = result.lower()
    if result_lower not in ["success", "failure"]:
        return json.dumps({"error": f"Invalid result filter: {result}. Must be 'success' or 'failure'"}, indent=2)

    # Filter to get only matching trials per object
    filtered_order = []
    for item in data.get("assembly_order", []):
        trials = item.get("trials", [])
        matching_trials = [t for t in trials if t.get("result", "").lower() == result_lower]
        if matching_trials:
            # Return first matching trial for this object
            filtered_order.append({
                "assembly_order": item.get("assembly_order"),
                "object_name": item.get("object_name"),
                "grasp_id": matching_trials[0].get("grasp_id"),
                "tool_sequence": matching_trials[0].get("tool_sequence", [])
            })

    return json.dumps({
        "assembly_id": data.get("assembly_id"),
        "base_name": data.get("base_name"),
        "assembly_order": filtered_order
    }, indent=2)

@mcp.tool()
def write_assembly_results(assembly_id: str, base_name: str, object_name: str, assembly_order: int, trial: Dict[str, Any]) -> str:
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
            - tool_sequence: list of strings (required) - ordered tool calls
            - result: "success" or "failure" (required)
            - comment: string (optional)

    Returns:
        JSON string with confirmation or error message
    """
    try:
        data = load_assembly_results(assembly_id)
    except Exception as e:
        return json.dumps({"success": False, "error": f"Error loading resource: {str(e)}"}, indent=2)

    # Validate assembly_order
    try:
        assembly_order = int(assembly_order)
    except (ValueError, TypeError):
        return json.dumps({"success": False, "error": f"assembly_order must be an integer, got: {assembly_order}"}, indent=2)

    # Validate base_name
    if not isinstance(base_name, str) or not base_name.strip():
        return json.dumps({"success": False, "error": "base_name must be a non-empty string"}, indent=2)

    base_name = base_name.strip()

    # Check base_name consistency
    existing_base = data.get("base_name", "").strip()
    if existing_base and existing_base != base_name:
        return json.dumps({"success": False, "error": f"base_name mismatch: assembly already has base_name '{existing_base}', cannot change to '{base_name}'"}, indent=2)

    data["base_name"] = base_name
    data["assembly_id"] = assembly_id

    # Handle trial as JSON string
    if isinstance(trial, str):
        try:
            trial = json.loads(trial)
        except json.JSONDecodeError as e:
            return json.dumps({"success": False, "error": f"Invalid JSON in trial: {str(e)}"}, indent=2)

    if not isinstance(trial, dict):
        return json.dumps({"success": False, "error": f"trial must be a dictionary, got: {type(trial).__name__}"}, indent=2)

    # Validate trial fields
    required_fields = {"trial_id", "grasp_id", "tool_sequence", "result"}
    missing_fields = required_fields - set(trial.keys())
    if missing_fields:
        return json.dumps({"success": False, "error": f"trial missing required fields: {list(missing_fields)}"}, indent=2)

    # Validate trial_id
    try:
        trial_id = int(trial.get("trial_id"))
    except (ValueError, TypeError):
        return json.dumps({"success": False, "error": f"trial_id must be an integer, got: {trial.get('trial_id')}"}, indent=2)

    # Validate grasp_id
    try:
        grasp_id = int(trial.get("grasp_id"))
    except (ValueError, TypeError):
        return json.dumps({"success": False, "error": f"grasp_id must be an integer, got: {trial.get('grasp_id')}"}, indent=2)

    # Validate tool_sequence
    tool_sequence = trial.get("tool_sequence")
    if not isinstance(tool_sequence, list):
        return json.dumps({"success": False, "error": f"tool_sequence must be a list, got: {type(tool_sequence).__name__}"}, indent=2)

    for i, tool in enumerate(tool_sequence):
        if not isinstance(tool, str):
            return json.dumps({"success": False, "error": f"tool_sequence[{i}] must be a string, got: {type(tool).__name__}"}, indent=2)

    # Validate result
    result_val = trial.get("result", "").lower()
    if result_val not in ["success", "failure"]:
        return json.dumps({"success": False, "error": f"Invalid result: {trial.get('result')}. Must be 'success' or 'failure'"}, indent=2)

    # Validate comment (optional)
    comment = trial.get("comment")
    if comment is not None and not isinstance(comment, str):
        return json.dumps({"success": False, "error": f"comment must be a string, got: {type(comment).__name__}"}, indent=2)

    # Build validated trial
    validated_trial = {
        "trial_id": trial_id,
        "grasp_id": grasp_id,
        "tool_sequence": tool_sequence,
        "result": result_val
    }
    if comment is not None:
        validated_trial["comment"] = comment

    # Find or create object entry in assembly_order
    assembly_order_list = data.get("assembly_order", [])
    object_found = False
    object_index = -1

    for i, item in enumerate(assembly_order_list):
        if item.get("object_name") == object_name:
            object_found = True
            object_index = i
            # Check assembly_order consistency
            if item.get("assembly_order") != assembly_order:
                return json.dumps({"success": False, "error": f"assembly_order mismatch: object '{object_name}' already has assembly_order {item.get('assembly_order')}, cannot change to {assembly_order}"}, indent=2)
            break

    if object_found:
        # Append trial to existing object
        assembly_order_list[object_index]["trials"].append(validated_trial)
    else:
        # Create new object entry
        assembly_order_list.append({
            "assembly_order": assembly_order,
            "object_name": object_name,
            "trials": [validated_trial]
        })

    data["assembly_order"] = assembly_order_list
    save_assembly_results(assembly_id, data)

    return json.dumps({"success": True}, indent=2)

@mcp.tool()
def clear_assembly_results(assembly_id: str) -> str:
    """
    Clear/delete all assembly results for an assembly.

    Args:
        assembly_id: The ID of the assembly to clear

    Returns:
        JSON string with confirmation or error message
    """
    results_file = get_assembly_results_file(assembly_id)

    if not results_file.exists():
        return json.dumps({"success": False, "message": "Resource not found"}, indent=2)

    try:
        results_file.unlink()
        return json.dumps({"success": True}, indent=2)
    except Exception as e:
        return json.dumps({"success": False, "error": str(e)}, indent=2)

@mcp.tool()
def list_assembly_results() -> str:
    """
    List all assemblies that have assembly results.

    Returns:
        JSON string containing all assembly IDs with results
    """
    results_files = list(RESOURCES_DIR.glob("Assembly_*_results.json"))
    assembly_ids = []

    for file in results_files:
        name = file.stem  # "Assembly_3_results"
        if name.startswith("Assembly_") and name.endswith("_results"):
            assembly_id = name.replace("Assembly_", "").replace("_results", "")
            assembly_ids.append(assembly_id)

    return json.dumps({
        "assembly_ids": assembly_ids,
        "count": len(assembly_ids)
    }, indent=2)

# ========== DISASSEMBLY TOOL SEQUENCE INFO TOOLS ==========

@mcp.tool()
def read_disassembly_tool_sequence() -> str:
    """
    Read the global disassembly tool sequence (list of tool calls).

    Returns:
        JSON string containing the tool_sequence list:
        ["move_to_object", "move_down", "control_gripper --command close", ...]
    """
    tool_sequence = load_disassembly_tool_sequence()
    return json.dumps({
        "tool_sequence": tool_sequence
    }, indent=2)

@mcp.tool()
def write_disassembly_tool_sequence(tool_sequence: List[str]) -> str:
    """
    Write the global disassembly tool sequence (ordered list of tool calls).

    Args:
        tool_sequence: List of tool call strings in order.
                       Example: ["move_to_object", "move_down", "control_gripper --command close", ...]

    Returns:
        JSON string with confirmation or error message
    """
    # Handle case where tool_sequence might be a JSON string
    if isinstance(tool_sequence, str):
        try:
            tool_sequence = json.loads(tool_sequence)
        except json.JSONDecodeError as e:
            return json.dumps({"success": False, "error": f"Invalid JSON in tool_sequence: {str(e)}"}, indent=2)

    # Validate it's a list
    if not isinstance(tool_sequence, list):
        return json.dumps({"success": False, "error": f"tool_sequence must be a list, got: {type(tool_sequence).__name__}"}, indent=2)

    # Validate all elements are strings
    for i, tool in enumerate(tool_sequence):
        if not isinstance(tool, str):
            return json.dumps({"success": False, "error": f"tool_sequence[{i}] must be a string, got: {type(tool).__name__}"}, indent=2)

    save_disassembly_tool_sequence(tool_sequence)
    return json.dumps({"success": True}, indent=2)

@mcp.tool()
def clear_disassembly_tool_sequence() -> str:
    """
    Clear/delete the global disassembly tool sequence.

    Returns:
        JSON string with confirmation or error message
    """
    tool_seq_file = get_disassembly_tool_sequence_file()

    if not tool_seq_file.exists():
        return json.dumps({"success": False, "message": "Resource not found"}, indent=2)

    try:
        tool_seq_file.unlink()
        return json.dumps({"success": True}, indent=2)
    except Exception as e:
        return json.dumps({"success": False, "error": str(e)}, indent=2)

# ========== DISASSEMBLY RESULTS TOOLS ==========

@mcp.tool()
def read_disassembly_results(assembly_id: str, result: Optional[str] = None) -> str:
    """
    Read disassembly results for a specific assembly.

    Args:
        assembly_id: The ID of the assembly
        result: Optional filter - "success" or "failure". If omitted, returns all trials.

    Returns:
        JSON string containing:
        - assembly_id
        - base_name
        - disassembly_order: list of {assembly_order, object_name, trials or filtered trial}

        When result filter is applied, returns first matching trial per object with:
        {assembly_order, object_name, grasp_id, gripper_state}
    """
    data = load_disassembly_results(assembly_id)

    if result is None:
        # Return all trials
        return json.dumps(data, indent=2)

    result_lower = result.lower()
    if result_lower not in ["success", "failure"]:
        return json.dumps({"error": f"Invalid result filter: {result}. Must be 'success' or 'failure'"}, indent=2)

    # Filter to get only matching trials per object
    filtered_order = []
    for item in data.get("disassembly_order", []):
        trials = item.get("trials", [])
        matching_trials = [t for t in trials if t.get("result", "").lower() == result_lower]
        if matching_trials:
            # Return first matching trial for this object
            filtered_order.append({
                "assembly_order": item.get("assembly_order"),
                "object_name": item.get("object_name"),
                "grasp_id": matching_trials[0].get("grasp_id"),
                "gripper_state": matching_trials[0].get("gripper_state")
            })

    return json.dumps({
        "assembly_id": data.get("assembly_id"),
        "base_name": data.get("base_name"),
        "disassembly_order": filtered_order
    }, indent=2)

@mcp.tool()
def write_disassembly_results(assembly_id: str, base_name: str, object_name: str, disassembly_order: int, trial: Dict[str, Any]) -> str:
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
            - gripper_state: "open" or "half-open" (required)
            - result: "success" or "failure" (required)

    Returns:
        JSON string with confirmation or error message
    """
    try:
        data = load_disassembly_results(assembly_id)
    except Exception as e:
        return json.dumps({"success": False, "error": f"Error loading resource: {str(e)}"}, indent=2)

    # Validate disassembly_order
    try:
        disassembly_order = int(disassembly_order)
    except (ValueError, TypeError):
        return json.dumps({"success": False, "error": f"disassembly_order must be an integer, got: {disassembly_order}"}, indent=2)

    # Validate base_name
    if not isinstance(base_name, str) or not base_name.strip():
        return json.dumps({"success": False, "error": "base_name must be a non-empty string"}, indent=2)

    base_name = base_name.strip()

    # Check base_name consistency
    existing_base = data.get("base_name", "").strip()
    if existing_base and existing_base != base_name:
        return json.dumps({"success": False, "error": f"base_name mismatch: assembly already has base_name '{existing_base}', cannot change to '{base_name}'"}, indent=2)

    data["base_name"] = base_name
    data["assembly_id"] = assembly_id

    # Handle trial as JSON string
    if isinstance(trial, str):
        try:
            trial = json.loads(trial)
        except json.JSONDecodeError as e:
            return json.dumps({"success": False, "error": f"Invalid JSON in trial: {str(e)}"}, indent=2)

    if not isinstance(trial, dict):
        return json.dumps({"success": False, "error": f"trial must be a dictionary, got: {type(trial).__name__}"}, indent=2)

    # Validate trial fields - NO tool_sequence, NO comment for disassembly (but has gripper_state)
    required_fields = {"trial_id", "grasp_id", "gripper_state", "result"}
    missing_fields = required_fields - set(trial.keys())
    if missing_fields:
        return json.dumps({"success": False, "error": f"trial missing required fields: {list(missing_fields)}"}, indent=2)

    # Reject extra fields
    allowed_fields = {"trial_id", "grasp_id", "gripper_state", "result"}
    extra_fields = set(trial.keys()) - allowed_fields
    if extra_fields:
        return json.dumps({"success": False, "error": f"Invalid fields in trial: {list(extra_fields)}. Only 'trial_id', 'grasp_id', 'gripper_state', and 'result' are allowed."}, indent=2)

    # Validate trial_id
    try:
        trial_id = int(trial.get("trial_id"))
    except (ValueError, TypeError):
        return json.dumps({"success": False, "error": f"trial_id must be an integer, got: {trial.get('trial_id')}"}, indent=2)

    # Validate grasp_id
    try:
        grasp_id = int(trial.get("grasp_id"))
    except (ValueError, TypeError):
        return json.dumps({"success": False, "error": f"grasp_id must be an integer, got: {trial.get('grasp_id')}"}, indent=2)

    # Validate gripper_state
    gripper_state = trial.get("gripper_state")
    if gripper_state not in ["open", "half-open"]:
        return json.dumps({"success": False, "error": f"Invalid gripper_state: {gripper_state}. Must be 'open' or 'half-open'"}, indent=2)

    # Validate result
    result_val = trial.get("result", "").lower()
    if result_val not in ["success", "failure"]:
        return json.dumps({"success": False, "error": f"Invalid result: {trial.get('result')}. Must be 'success' or 'failure'"}, indent=2)

    # Build validated trial (simpler than assembly - no tool_sequence, no comment, but has gripper_state)
    validated_trial = {
        "trial_id": trial_id,
        "grasp_id": grasp_id,
        "gripper_state": gripper_state,
        "result": result_val
    }

    # Find or create object entry in disassembly_order
    disassembly_order_list = data.get("disassembly_order", [])
    object_found = False
    object_index = -1

    for i, item in enumerate(disassembly_order_list):
        if item.get("object_name") == object_name:
            object_found = True
            object_index = i
            # Check disassembly_order consistency
            if item.get("disassembly_order") != disassembly_order:
                return json.dumps({"success": False, "error": f"disassembly_order mismatch: object '{object_name}' already has disassembly_order {item.get('disassembly_order')}, cannot change to {disassembly_order}"}, indent=2)
            break

    if object_found:
        # Append trial to existing object
        disassembly_order_list[object_index]["trials"].append(validated_trial)
    else:
        # Create new object entry
        disassembly_order_list.append({
            "disassembly_order": disassembly_order,
            "object_name": object_name,
            "trials": [validated_trial]
        })

    data["disassembly_order"] = disassembly_order_list
    save_disassembly_results(assembly_id, data)

    return json.dumps({"success": True}, indent=2)

@mcp.tool()
def clear_disassembly_results(assembly_id: str) -> str:
    """
    Clear/delete all disassembly results for an assembly.

    Args:
        assembly_id: The ID of the assembly to clear

    Returns:
        JSON string with confirmation or error message
    """
    results_file = get_disassembly_results_file(assembly_id)

    if not results_file.exists():
        return json.dumps({"success": False, "message": "Resource not found"}, indent=2)

    try:
        results_file.unlink()
        return json.dumps({"success": True}, indent=2)
    except Exception as e:
        return json.dumps({"success": False, "error": str(e)}, indent=2)

@mcp.tool()
def list_disassembly_results() -> str:
    """
    List all assemblies that have disassembly results.

    Returns:
        JSON string containing all assembly IDs with disassembly results
    """
    results_files = list(RESOURCES_DIR.glob("Disassembly_*_results.json"))
    assembly_ids = []

    for file in results_files:
        name = file.stem  # "Disassembly_3_results"
        if name.startswith("Disassembly_") and name.endswith("_results"):
            assembly_id = name.replace("Disassembly_", "").replace("_results", "")
            assembly_ids.append(assembly_id)

    return json.dumps({
        "assembly_ids": assembly_ids,
        "count": len(assembly_ids)
    }, indent=2)

# ========== KNOWLEDGE TOOLS ==========

@mcp.tool()
def write_knowledge(obs_id: str, observation: str, status: str, assembly_id: str) -> str:
    """
    Write a knowledge observation (hypothesis or rule).

    Args:
        obs_id: Unique identifier for the observation
        observation: The observation text
        status: "hypothesis" (unverified pattern) or "rule" (validated across assemblies)
        assembly_id: The assembly this observation came from

    Returns:
        JSON string with confirmation or error message
    """
    # Validate status
    status_lower = status.lower()
    if status_lower not in ["hypothesis", "rule"]:
        return json.dumps({"success": False, "error": f"Invalid status: {status}. Must be 'hypothesis' or 'rule'"}, indent=2)

    # Validate obs_id
    if not isinstance(obs_id, str) or not obs_id.strip():
        return json.dumps({"success": False, "error": "obs_id must be a non-empty string"}, indent=2)

    # Validate observation
    if not isinstance(observation, str) or not observation.strip():
        return json.dumps({"success": False, "error": "observation must be a non-empty string"}, indent=2)

    # Validate assembly_id
    if not isinstance(assembly_id, str) or not assembly_id.strip():
        return json.dumps({"success": False, "error": "assembly_id must be a non-empty string"}, indent=2)

    try:
        data = load_knowledge()
    except Exception as e:
        return json.dumps({"success": False, "error": f"Error loading knowledge: {str(e)}"}, indent=2)

    observations = data.get("observations", [])

    # Check if obs_id already exists
    for obs in observations:
        if obs.get("obs_id") == obs_id:
            return json.dumps({"success": False, "error": f"obs_id '{obs_id}' already exists"}, indent=2)

    # Add new observation
    observations.append({
        "obs_id": obs_id.strip(),
        "observation": observation.strip(),
        "status": status_lower,
        "assembly_id": assembly_id.strip()
    })

    data["observations"] = observations
    save_knowledge(data)

    return json.dumps({"success": True}, indent=2)

@mcp.tool()
def clear_knowledge(obs_id: str) -> str:
    """
    Delete a knowledge observation by obs_id.

    Args:
        obs_id: The ID of the observation to delete

    Returns:
        JSON string with confirmation or error message
    """
    try:
        data = load_knowledge()
    except Exception as e:
        return json.dumps({"success": False, "error": f"Error loading knowledge: {str(e)}"}, indent=2)

    observations = data.get("observations", [])
    initial_count = len(observations)

    # Filter out the observation with matching obs_id
    filtered_observations = [obs for obs in observations if obs.get("obs_id") != obs_id]

    if len(filtered_observations) == initial_count:
        return json.dumps({"success": False, "message": f"obs_id '{obs_id}' not found"}, indent=2)

    data["observations"] = filtered_observations
    save_knowledge(data)

    return json.dumps({"success": True}, indent=2)

@mcp.tool()
def list_knowledge() -> str:
    """
    List all knowledge observations.

    Returns:
        JSON string containing all observations with their IDs, texts, statuses, and assembly IDs
    """
    data = load_knowledge()
    return json.dumps(data, indent=2)

# ========== ELICITATION TOOL FOR LOG MANAGEMENT ==========

class ClearLogsConfirmation(BaseModel):
    """Schema for confirming log deletion."""
    confirm_delete: bool = Field(
        default=False,
        description="Delete ALL listed resource files? This cannot be undone."
    )

@mcp.tool()
async def clear_all_resources(ctx: Context[ServerSession, None]) -> str:
    """
    Clear/delete all resource files with user confirmation via elicitation.

    Shows the user a list of existing resource files and asks for confirmation before deleting.
    This includes disassembly results, assembly results, disassembly tool sequences, and knowledge observations.

    Returns:
        JSON string with deletion result
    """
    # Get all resource files
    log_patterns = [
        "Disassembly_*_results.json",
        "Assembly_*_results.json",
        "disassembly_tool_sequence.json",
        "knowledge_observations.json"
    ]

    all_files = []
    for pattern in log_patterns:
        files = list(RESOURCES_DIR.glob(pattern))
        all_files.extend([f.name for f in files])

    all_files.sort()

    if not all_files:
        return json.dumps({
            "status": "info",
            "message": "No log files found in resources directory",
            "resources_dir": str(RESOURCES_DIR)
        }, indent=2)

    # Build file list message
    file_list = "\n".join(f"  - {f}" for f in all_files)

    try:
        # Use elicitation to ask for confirmation
        result = await ctx.elicit(
            message=f"Found {len(all_files)} log file(s):\n{file_list}\n\nWould you like to delete them all?",
            schema=ClearLogsConfirmation
        )

        if result.action == "accept" and result.data:
            if not result.data.confirm_delete:
                return json.dumps({
                    "status": "cancelled",
                    "message": "Deletion cancelled by user"
                }, indent=2)

            # Delete all files
            deleted_files = []
            errors = []
            for filename in all_files:
                file_path = RESOURCES_DIR / filename
                try:
                    file_path.unlink()
                    deleted_files.append(filename)
                except Exception as e:
                    errors.append(f"{filename}: {str(e)}")

            return json.dumps({
                "status": "success" if not errors else "partial",
                "deleted_files": deleted_files,
                "errors": errors if errors else None,
                "message": f"Deleted {len(deleted_files)} file(s)"
            }, indent=2)

        else:
            return json.dumps({
                "status": "cancelled",
                "message": "Elicitation declined or cancelled",
                "files_found": all_files
            }, indent=2)

    except Exception as e:
        error_msg = str(e)
        if "Method not found" in error_msg:
            return json.dumps({
                "status": "error",
                "message": "Elicitation not supported by this client",
                "files_found": all_files,
                "hint": "Use clear_assembly_results, clear_disassembly_results, clear_disassembly_tool_sequence, or clear_knowledge tools directly"
            }, indent=2)
        return json.dumps({
            "status": "error",
            "message": f"Elicitation failed: {error_msg}",
            "files_found": all_files
        }, indent=2)


if __name__ == "__main__":
    mcp.run()
