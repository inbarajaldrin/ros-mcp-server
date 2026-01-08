from mcp.server.fastmcp import FastMCP
import json
import os
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

mcp = FastMCP("FMB Assembly Steps and Grasp IDs Management")

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

def get_grasp_log_file(assembly_id: str) -> Path:
    """Get the grasp log file path for a specific assembly"""
    normalized_id = normalize_assembly_id(assembly_id)
    return RESOURCES_DIR / f"Assembly_{normalized_id}_object_grasp_log.json"

def get_assembly_file(assembly_id: str) -> Path:
    """Get the assembly sequence log file path for a specific assembly"""
    normalized_id = normalize_assembly_id(assembly_id)
    return RESOURCES_DIR / f"Assembly_{normalized_id}_sequence_log.json"

def get_final_sequence_file(assembly_id: str) -> Path:
    """Get the final sequence file path for a specific assembly"""
    normalized_id = normalize_assembly_id(assembly_id)
    return RESOURCES_DIR / f"Assembly_{normalized_id}_final_sequence.json"

def get_disassembly_grasp_log_file(assembly_id: str) -> Path:
    """Get the disassembly grasp log file path for a specific assembly"""
    normalized_id = normalize_assembly_id(assembly_id)
    return RESOURCES_DIR / f"Disassembly_{normalized_id}_grasp_log.json"

def load_grasp_resource(assembly_id: str):
    """Load grasp resource from JSON file for a specific assembly"""
    grasp_file = get_grasp_log_file(assembly_id)
    if not grasp_file.exists():
        return {}
    
    try:
        with open(grasp_file, 'r') as f:
            content = f.read().strip()
            if not content:
                return {}
            return json.loads(content)
    except json.JSONDecodeError:
        # File exists but has invalid JSON - return empty dict
        return {}
    except Exception:
        # Any other error - return empty dict
        return {}

def save_grasp_resource(assembly_id: str, data):
    """Save grasp resource to JSON file for a specific assembly"""
    grasp_file = get_grasp_log_file(assembly_id)
    with open(grasp_file, 'w') as f:
        json.dump(data, f, indent=2)

def load_assembly_sequence_resource(assembly_id: str):
    """Load assembly sequence resource from JSON file for a specific assembly. Returns dict with 'assembled_into' and 'sequence' keys."""
    assembly_file = get_assembly_file(assembly_id)
    if not assembly_file.exists():
        return {"assembled_into": "", "sequence": []}
    
    try:
        with open(assembly_file, 'r') as f:
            content = f.read().strip()
            if not content:
                return {"assembled_into": "", "sequence": []}
            data = json.loads(content)
            # Handle backward compatibility: if it's a list (old format), convert to new format
            if isinstance(data, list):
                return {"assembled_into": "", "sequence": data}
            # New format: ensure it has the required keys
            if isinstance(data, dict):
                if "sequence" not in data:
                    data["sequence"] = []
                if "assembled_into" not in data:
                    data["assembled_into"] = ""
                return data
            return {"assembled_into": "", "sequence": []}
    except json.JSONDecodeError:
        # File exists but has invalid JSON - return empty structure
        return {"assembled_into": "", "sequence": []}
    except Exception:
        # Any other error - return empty structure
        return {"assembled_into": "", "sequence": []}

def save_assembly_sequence_resource(assembly_id: str, data):
    """Save assembly sequence resource to JSON file for a specific assembly"""
    assembly_file = get_assembly_file(assembly_id)
    with open(assembly_file, 'w') as f:
        json.dump(data, f, indent=2)

def load_final_sequence(assembly_id: str):
    """Load final sequence from JSON file for a specific assembly"""
    final_file = get_final_sequence_file(assembly_id)
    if not final_file.exists():
        return []
    
    try:
        with open(final_file, 'r') as f:
            content = f.read().strip()
            if not content:
                return []
            return json.loads(content)
    except json.JSONDecodeError:
        # File exists but has invalid JSON - return empty list
        return []
    except Exception:
        # Any other error - return empty list
        return []

def save_final_sequence(assembly_id: str, data):
    """Save final sequence to JSON file for a specific assembly"""
    final_file = get_final_sequence_file(assembly_id)
    with open(final_file, 'w') as f:
        json.dump(data, f, indent=2)

def load_disassembly_grasp_resource(assembly_id: str):
    """Load disassembly grasp resource from JSON file for a specific assembly"""
    grasp_file = get_disassembly_grasp_log_file(assembly_id)
    if not grasp_file.exists():
        return {}
    
    try:
        with open(grasp_file, 'r') as f:
            content = f.read().strip()
            if not content:
                return {}
            return json.loads(content)
    except json.JSONDecodeError:
        # File exists but has invalid JSON - return empty dict
        return {}
    except Exception:
        # Any other error - return empty dict
        return {}

def save_disassembly_grasp_resource(assembly_id: str, data):
    """Save disassembly grasp resource to JSON file for a specific assembly"""
    grasp_file = get_disassembly_grasp_log_file(assembly_id)
    with open(grasp_file, 'w') as f:
        json.dump(data, f, indent=2)


# ========== RESOURCES ==========

@mcp.resource("Assembly{assembly_id}/object_name/{object_name}/grasp_configs")
def get_object_grasp_configs(assembly_id: str, object_name: str) -> str:
    """Get grasp configurations for an object in a specific assembly (list of grasp_id, gripper_state, and result). Each config includes grasp_id (int), gripper_state ("open" or "half-open"), and result ("SUCCESS" or "FAILURE"). IMPORTANT: The gripper must be set to the specified gripper_state BEFORE moving to grasp to access the grasp_id."""
    data = load_grasp_resource(assembly_id)
    obj_data = data.get(object_name, [])
    
    if isinstance(obj_data, list):
        return json.dumps(obj_data, indent=2)
    else:
        return json.dumps([], indent=2)

@mcp.resource("Assembly{assembly_id}/object_name/{object_name}/grasp_configs/{grasp_id}/result")
def get_object_status_for_id(assembly_id: str, object_name: str, grasp_id: str) -> str:
    """Get all configurations (grasp_id, gripper_state, and result) for a specific grasp_id of an object_name in an assembly. Returns all attempts (both SUCCESS and FAILURE) with their modes. IMPORTANT: The gripper must be set to the specified gripper_state BEFORE moving to grasp to access the grasp_id."""
    data = load_grasp_resource(assembly_id)
    obj_data = data.get(object_name, [])
    grasp_id_int = int(grasp_id)
    
    if isinstance(obj_data, list):
        matching_configs = [config for config in obj_data if config.get("grasp_id") == grasp_id_int]
        if matching_configs:
            return json.dumps(matching_configs, indent=2)
    
    return json.dumps([], indent=2)

@mcp.resource("Assembly{assembly_id}/object_name/{object_name}/grasp_configs/result/{result}")
def get_object_grasp_configs_by_result(assembly_id: str, object_name: str, result: str) -> str:
    """Get all grasp configurations filtered by result (SUCCESS or FAILURE) for an object in a specific assembly. Returns all configs matching the specified result. IMPORTANT: The gripper must be set to the specified gripper_state BEFORE moving to grasp to access the grasp_id."""
    data = load_grasp_resource(assembly_id)
    obj_data = data.get(object_name, [])
    result_upper = result.upper()
    
    if result_upper not in ["SUCCESS", "FAILURE"]:
        return json.dumps({"error": f"Invalid result: {result}. Must be 'SUCCESS' or 'FAILURE'"}, indent=2)
    
    if isinstance(obj_data, list):
        matching_configs = [config for config in obj_data if config.get("result", "").upper() == result_upper]
        return json.dumps(matching_configs, indent=2)
    
    return json.dumps([], indent=2)

@mcp.resource("Assembly{assembly_id}/sequence")
def get_assembly_sequence(assembly_id: str) -> str:
    """Get sequence for a specific Assembly ID. Returns dict with 'assembled_into' (string) and 'sequence' (list). Each item in sequence has sequence_id (fixed), object_name (fixed), and tools_trials (list of trials with trial_id, grasp_id, tools ordered sequence, and result)."""
    data = load_assembly_sequence_resource(assembly_id)
    return json.dumps(data, indent=2)

@mcp.resource("Assembly{assembly_id}/sequence/{object_name}/tools_trials")
def get_assembly_object_tools_trials(assembly_id: str, object_name: str) -> str:
    """Get tools_trials for a specific object in an assembly sequence. Each trial has trial_id, grasp_id, tools (ordered sequence), and result."""
    data = load_assembly_sequence_resource(assembly_id)
    sequence = data.get("sequence", [])
    
    for item in sequence:
        if item.get("object_name") == object_name:
            return json.dumps({"tools_trials": item.get("tools_trials", [])}, indent=2)
    
    return json.dumps({"error": f"Object '{object_name}' not found in Assembly{assembly_id}"}, indent=2)

@mcp.resource("Assembly{assembly_id}/sequence/{object_name}/tools_trials/{trial_id}")
def get_assembly_object_trial(assembly_id: str, object_name: str, trial_id: str) -> str:
    """Get a specific trial for an object in an assembly sequence. Returns trial_id, grasp_id, tools (ordered sequence), and result."""
    data = load_assembly_sequence_resource(assembly_id)
    sequence = data.get("sequence", [])
    trial_id_int = int(trial_id)
    
    for item in sequence:
        if item.get("object_name") == object_name:
            trials = item.get("tools_trials", [])
            for trial in trials:
                if trial.get("trial_id") == trial_id_int:
                    return json.dumps(trial, indent=2)
            return json.dumps({"error": f"Trial {trial_id} not found for object '{object_name}' in Assembly{assembly_id}"}, indent=2)
    
    return json.dumps({"error": f"Object '{object_name}' not found in Assembly{assembly_id}"}, indent=2)

@mcp.resource("Assembly{assembly_id}/sequence/{object_name}/tools_trials/{trial_id}/result")
def get_assembly_object_trial_result(assembly_id: str, object_name: str, trial_id: str) -> str:
    """Get all configurations (trial_id, grasp_id, tools, and result) for a specific trial_id of an object_name in an assembly. Returns all attempts (both SUCCESS and FAILURE) with their configurations."""
    data = load_assembly_sequence_resource(assembly_id)
    sequence = data.get("sequence", [])
    trial_id_int = int(trial_id)
    
    for item in sequence:
        if item.get("object_name") == object_name:
            trials = item.get("tools_trials", [])
            matching_trials = [trial for trial in trials if trial.get("trial_id") == trial_id_int]
            if matching_trials:
                return json.dumps(matching_trials, indent=2)
            return json.dumps([], indent=2)
    
    return json.dumps([], indent=2)

@mcp.resource("Assembly{assembly_id}/sequence/{object_name}/tools_trials/result/{result}")
def get_assembly_object_trials_by_result(assembly_id: str, object_name: str, result: str) -> str:
    """Get all tools_trials filtered by result (SUCCESS or FAILURE) for an object in an assembly sequence. Returns all trials matching the specified result."""
    data = load_assembly_sequence_resource(assembly_id)
    sequence = data.get("sequence", [])
    result_upper = result.upper()
    
    if result_upper not in ["SUCCESS", "FAILURE"]:
        return json.dumps({"error": f"Invalid result: {result}. Must be 'SUCCESS' or 'FAILURE'"}, indent=2)
    
    for item in sequence:
        if item.get("object_name") == object_name:
            trials = item.get("tools_trials", [])
            matching_trials = [trial for trial in trials if trial.get("result", "").upper() == result_upper]
            return json.dumps({"tools_trials": matching_trials}, indent=2)
    
    return json.dumps({"error": f"Object '{object_name}' not found in Assembly{assembly_id}"}, indent=2)

# ========== DISASSEMBLY RESOURCES ==========

@mcp.resource("Disassembly{assembly_id}/object_name/{object_name}/grasp_configs")
def get_disassembly_object_grasp_configs(assembly_id: str, object_name: str) -> str:
    """Get disassembly grasp configurations for an object in a specific assembly (list of grasp_id and result). Each config includes grasp_id (int) and result ("SUCCESS" or "FAILURE")."""
    data = load_disassembly_grasp_resource(assembly_id)
    obj_data = data.get(object_name, [])
    
    if isinstance(obj_data, list):
        return json.dumps(obj_data, indent=2)
    else:
        return json.dumps([], indent=2)

@mcp.resource("Disassembly{assembly_id}/object_name/{object_name}/grasp_configs/{grasp_id}/result")
def get_disassembly_object_status_for_id(assembly_id: str, object_name: str, grasp_id: str) -> str:
    """Get all disassembly configurations (grasp_id and result) for a specific grasp_id of an object_name in an assembly. Returns all attempts (both SUCCESS and FAILURE)."""
    data = load_disassembly_grasp_resource(assembly_id)
    obj_data = data.get(object_name, [])
    grasp_id_int = int(grasp_id)
    
    if isinstance(obj_data, list):
        matching_configs = [config for config in obj_data if config.get("grasp_id") == grasp_id_int]
        if matching_configs:
            return json.dumps(matching_configs, indent=2)
    
    return json.dumps([], indent=2)

@mcp.resource("Disassembly{assembly_id}/object_name/{object_name}/grasp_configs/result/{result}")
def get_disassembly_object_grasp_configs_by_result(assembly_id: str, object_name: str, result: str) -> str:
    """Get all disassembly grasp configurations filtered by result (SUCCESS or FAILURE) for an object in a specific assembly. Returns all configs matching the specified result."""
    data = load_disassembly_grasp_resource(assembly_id)
    obj_data = data.get(object_name, [])
    result_upper = result.upper()
    
    if result_upper not in ["SUCCESS", "FAILURE"]:
        return json.dumps({"error": f"Invalid result: {result}. Must be 'SUCCESS' or 'FAILURE'"}, indent=2)
    
    if isinstance(obj_data, list):
        matching_configs = [config for config in obj_data if config.get("result", "").upper() == result_upper]
        return json.dumps(matching_configs, indent=2)
    
    return json.dumps([], indent=2)

# ========== TOOLS FOR OBJECT GRASP RESOURCE (Assembly{id}/object_name/grasp_configs) ==========

@mcp.tool()
def read_object_grasp_resource(assembly_id: str, object_name: str) -> str:
    """
    Read the grasp configurations for an object in a specific assembly (grasp_configs with grasp_id, gripper_state, and result)
    
    Args:
        assembly_id: The ID of the assembly
        object_name: The name of the object
    
    Returns:
        JSON string containing grasp_configs list with grasp_id, gripper_state, and result (includes both SUCCESS and FAILURE attempts)
        Each config has: {"grasp_id": <int>, "gripper_state": "open"|"half-open", "result": "SUCCESS"|"FAILURE"}
    
    IMPORTANT: The gripper must be set to the specified gripper_state BEFORE moving to grasp to access the grasp_id.
    The sequence should be: 1) Set gripper to the gripper_state (open or half-open), 2) Move to grasp position, 3) Execute grasp using the grasp_id.
    """
    data = load_grasp_resource(assembly_id)
    obj_data = data.get(object_name, [])
    
    if not obj_data:
        return json.dumps({
            "assembly_id": assembly_id,
            "object_name": object_name,
            "grasp_configs": [],
            "message": "Resource not found"
        }, indent=2)
    
    if isinstance(obj_data, list):
        return json.dumps({
            "assembly_id": assembly_id,
            "object_name": object_name,
            "grasp_configs": obj_data
        }, indent=2)
    else:
        return json.dumps({
            "assembly_id": assembly_id,
            "object_name": object_name,
            "grasp_configs": []
        }, indent=2)

@mcp.tool()
def write_object_grasp_resource(assembly_id: str, object_name: str, grasp_configs: Optional[List[Dict[str, Any]]] = None) -> str:
    """
    Write or update grasp configurations for an object in a specific assembly (grasp_configs list with grasp_id, gripper_state, and result)
    
    Args:
        assembly_id: The ID of the assembly
        object_name: The name of the object
        grasp_configs: List of grasp configurations, each with grasp_id, gripper_state, and result.
                       Format: [{"grasp_id": 1, "gripper_state": "open", "result": "SUCCESS"}, ...]
                       - grasp_id: integer (required)
                       - gripper_state: "open" or "half-open" (required)
                       - result: "SUCCESS" or "FAILURE" (required)
                       Both success and failure attempts are stored.
    
    Returns:
        JSON string with confirmation or error message
    """
    try:
        data = load_grasp_resource(assembly_id)
    except json.JSONDecodeError as e:
        return json.dumps({"success": False, "error": f"Error loading resource file: {str(e)}"}, indent=2)
    except Exception as e:
        return json.dumps({"success": False, "error": f"Unexpected error loading resource: {str(e)}"}, indent=2)
    
    if grasp_configs is not None:
        # Handle case where grasp_configs might be a JSON string
        if isinstance(grasp_configs, str):
            try:
                grasp_configs = json.loads(grasp_configs)
            except json.JSONDecodeError as e:
                return json.dumps({"success": False, "error": f"Invalid JSON in grasp_configs: {str(e)}"}, indent=2)
        
        # Ensure it's a list
        if not isinstance(grasp_configs, list):
            return json.dumps({"success": False, "error": f"grasp_configs must be a list, got: {type(grasp_configs).__name__}"}, indent=2)
        processed_configs = []
        for i, config in enumerate(grasp_configs):
            # Ensure config is a dict
            if not isinstance(config, dict):
                return json.dumps({"success": False, "error": f"Config at index {i} must be a dictionary, got: {type(config).__name__}"}, indent=2)
            
            # Check for required fields
            if "grasp_id" not in config:
                return json.dumps({"success": False, "error": f"Config at index {i} must have a 'grasp_id' field"}, indent=2)
            
            if "result" not in config:
                return json.dumps({"success": False, "error": f"Config at index {i} must have a 'result' field"}, indent=2)
            
            if "gripper_state" not in config:
                return json.dumps({"success": False, "error": f"Config at index {i} must have a 'gripper_state' field"}, indent=2)
            
            # Reject any extra fields - only allow these three fields
            allowed_fields = {"grasp_id", "result", "gripper_state"}
            extra_fields = set(config.keys()) - allowed_fields
            if extra_fields:
                return json.dumps({"success": False, "error": f"Invalid fields found: {list(extra_fields)}. Only 'grasp_id', 'gripper_state', and 'result' are allowed."}, indent=2)
            
            # Validate grasp_id is an integer
            try:
                grasp_id = int(config.get("grasp_id"))
            except (ValueError, TypeError):
                return json.dumps({"success": False, "error": f"grasp_id must be an integer, got: {config.get('grasp_id')}"}, indent=2)
            
            # Validate result
            result = config.get("result", "").upper()
            if result not in ["SUCCESS", "FAILURE"]:
                return json.dumps({"success": False, "error": f"Invalid result: {config.get('result')}. Must be 'SUCCESS' or 'FAILURE'"}, indent=2)
            
            # Validate gripper_state
            gripper_state = config.get("gripper_state")
            if gripper_state not in ["open", "half-open"]:
                return json.dumps({"success": False, "error": f"Invalid gripper_state: {gripper_state}. Must be 'open' or 'half-open'"}, indent=2)
            
            # Store all three fields (only these three, no extra fields)
            processed_configs.append({
                "grasp_id": grasp_id,
                "gripper_state": gripper_state,
                "result": result  # Store as uppercase
            })
        
        # Store as list - includes both success and failure attempts with all three fields
        data[object_name] = processed_configs
    
    save_grasp_resource(assembly_id, data)
    
    return json.dumps({"success": True}, indent=2)

@mcp.tool()
def clear_object_grasp_resource(assembly_id: str, object_name: str) -> str:
    """
    Clear/delete a grasp resource for an object in a specific assembly
    
    Args:
        assembly_id: The ID of the assembly
        object_name: The name of the object to clear
    
    Returns:
        JSON string with confirmation or error message
    """
    data = load_grasp_resource(assembly_id)
    
    if object_name not in data:
        return json.dumps({"success": False}, indent=2)
    
    del data[object_name]
    
    save_grasp_resource(assembly_id, data)
    
    return json.dumps({"success": True}, indent=2)

@mcp.tool()
def list_object_grasp_resource() -> str:
    """
    List all assemblies in the grasp resource
    
    Returns:
        JSON string containing all assembly IDs that have grasp logs
    
    Note: When using grasp configurations from this resource, remember that the gripper must be set to the specified gripper_state BEFORE moving to grasp to access the grasp_id.
    """
    # Find all Assembly_{id}_object_grasp_log.json files
    grasp_files = list(RESOURCES_DIR.glob("Assembly_*_object_grasp_log.json"))
    assembly_ids = []
    
    for file in grasp_files:
        # Extract assembly ID from filename like "Assembly_1_object_grasp_log.json"
        name = file.stem  # "Assembly_1_object_grasp_log"
        if name.startswith("Assembly_") and name.endswith("_object_grasp_log"):
            # Remove "Assembly_" prefix and "_object_grasp_log" suffix
            assembly_id = name.replace("Assembly_", "").replace("_object_grasp_log", "")
            assembly_ids.append(assembly_id)
    
    return json.dumps({
        "assembly_ids": assembly_ids,
        "count": len(assembly_ids)
    }, indent=2)

# ========== TOOLS FOR ASSEMBLY SEQUENCE RESOURCE (Assembly{id}/sequence/object_name/grasp_id) ==========

@mcp.tool()
def read_assembly_sequence_resource(assembly_id: str) -> str:
    """
    Read the complete assembly sequence resource for an assembly (dict with assembled_into and sequence)
    
    Args:
        assembly_id: The ID of the assembly
    
    Returns:
        JSON string containing dict with 'assembled_into' (string) and 'sequence' (list). Each item in sequence has sequence_id (fixed), object_name (fixed), and tools_trials (list of trials with trial_id, grasp_id, tools ordered sequence, and result)
    """
    data = load_assembly_sequence_resource(assembly_id)
    
    if not data.get("sequence"):
        return json.dumps({
            "assembly_id": assembly_id,
            "assembled_into": data.get("assembled_into", ""),
            "sequence": [],
            "message": "Resource not found"
        }, indent=2)
    
    return json.dumps({
        "assembly_id": assembly_id,
        "assembled_into": data.get("assembled_into", ""),
        "sequence": data.get("sequence", [])
    }, indent=2)

@mcp.tool()
def write_assembly_sequence_resource(assembly_id: str, object_name: str, sequence_id: int, assembled_into: str, tools_trials: Optional[List[Dict[str, Any]]] = None) -> str:
    """
    Write or update an assembly sequence resource for a specific object. assembled_into is stored at the top level (once for the entire assembly). sequence_id and object_name are fixed and cannot be changed after creation.
    
    Args:
        assembly_id: The ID of the assembly
        object_name: The name of the object
        sequence_id: The sequence ID (integer, fixed)
        assembled_into: The name of the object/base that objects are being assembled into (string, stored at top level)
        tools_trials: Optional list of trial objects, each with:
                      - trial_id: integer (required)
                      - grasp_id: integer (required)
                      - tools: ordered list of strings (required) - sequence of tool names executed in order
                                NOTE: Add flags also if called under one tool (e.g., ["tool_name", "tool_name --flag1", "tool_name --flag2"])
                                NOTE: Gripper state is included in the tools sequence (e.g., "control_gripper --command open")
                      - result: "SUCCESS" or "FAILURE" (required)
                      - comment: string (optional) - optional comment/note about the trial
                      Format: [{"trial_id": 1, "grasp_id": 1, "tools": ["tool_name_1", "tool_name_2"], "result": "SUCCESS", "comment": "optional note"}, ...]
    
    Returns:
        JSON string with confirmation or error message
    """
    try:
        data = load_assembly_sequence_resource(assembly_id)
    except json.JSONDecodeError as e:
        return json.dumps({"success": False, "error": f"Error loading resource file: {str(e)}"}, indent=2)
    except Exception as e:
        return json.dumps({"success": False, "error": f"Unexpected error loading resource: {str(e)}"}, indent=2)
    
    sequence = data.get("sequence", [])
    
    # Validate sequence_id is an integer
    try:
        sequence_id = int(sequence_id)
    except (ValueError, TypeError):
        return json.dumps({"success": False, "error": f"sequence_id must be an integer, got: {sequence_id}"}, indent=2)
    
    # Validate assembled_into is a string
    if not isinstance(assembled_into, str):
        return json.dumps({"success": False, "error": f"assembled_into must be a string, got: {type(assembled_into).__name__}"}, indent=2)
    
    if not assembled_into.strip():
        return json.dumps({"success": False, "error": "assembled_into cannot be empty"}, indent=2)
    
    assembled_into = assembled_into.strip()
    
    # Check if assembled_into is already set and matches (or set it if empty)
    existing_assembled_into = data.get("assembled_into", "").strip()
    if existing_assembled_into:
        if existing_assembled_into != assembled_into:
            return json.dumps({"success": False, "error": f"assembled_into mismatch: assembly already has assembled_into '{existing_assembled_into}', cannot change to '{assembled_into}'"}, indent=2)
    else:
        # Set assembled_into at top level
        data["assembled_into"] = assembled_into
    
    # Find existing object or create new
    object_found = False
    object_index = -1
    for i, item in enumerate(sequence):
        if item.get("object_name") == object_name:
            object_found = True
            object_index = i
            # Check if sequence_id matches (it should be fixed)
            if item.get("sequence_id") != sequence_id:
                return json.dumps({"success": False, "error": f"sequence_id mismatch: object '{object_name}' already exists with sequence_id {item.get('sequence_id')}, cannot change to {sequence_id}"}, indent=2)
            break
    
    # Process tools_trials (use empty list if None)
    if tools_trials is None:
        tools_trials = []
    
    # Handle case where tools_trials might be a JSON string
    if isinstance(tools_trials, str):
        try:
            tools_trials = json.loads(tools_trials)
        except json.JSONDecodeError as e:
            return json.dumps({"success": False, "error": f"Invalid JSON in tools_trials: {str(e)}"}, indent=2)
    
    # Ensure it's a list
    if not isinstance(tools_trials, list):
        return json.dumps({"success": False, "error": f"tools_trials must be a list, got: {type(tools_trials).__name__}"}, indent=2)
    
    # Validate each trial
    validated_trials = []
    for j, trial in enumerate(tools_trials):
        if not isinstance(trial, dict):
            return json.dumps({"success": False, "error": f"Trial at index {j} must be a dictionary, got: {type(trial).__name__}"}, indent=2)
        
        # Check for required fields in trial
        trial_required_fields = {"trial_id", "grasp_id", "tools", "result"}
        trial_missing_fields = trial_required_fields - set(trial.keys())
        if trial_missing_fields:
            return json.dumps({"success": False, "error": f"Trial at index {j} missing required fields: {list(trial_missing_fields)}"}, indent=2)
        
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
        
        # Validate tools is an ordered list (sequence)
        tools = trial.get("tools")
        if not isinstance(tools, list):
            return json.dumps({"success": False, "error": f"tools must be an ordered list (sequence), got: {type(tools).__name__}"}, indent=2)
        
        # Validate each tool in the sequence is a string
        for k, tool in enumerate(tools):
            if not isinstance(tool, str):
                return json.dumps({"success": False, "error": f"Tool at index {k} in trial {j} must be a string, got: {type(tool).__name__}"}, indent=2)
        
        # Validate result
        result = trial.get("result", "").upper()
        if result not in ["SUCCESS", "FAILURE"]:
            return json.dumps({"success": False, "error": f"Invalid result: {trial.get('result')}. Must be 'SUCCESS' or 'FAILURE'"}, indent=2)
        
        # Validate comment (optional)
        comment = trial.get("comment")
        if comment is not None and not isinstance(comment, str):
            return json.dumps({"success": False, "error": f"comment must be a string, got: {type(comment).__name__}"}, indent=2)
        
        # Reject any extra fields in trial
        trial_allowed_fields = {"trial_id", "grasp_id", "tools", "result", "comment"}
        trial_extra_fields = set(trial.keys()) - trial_allowed_fields
        if trial_extra_fields:
            return json.dumps({"success": False, "error": f"Invalid fields found in trial {j}: {list(trial_extra_fields)}. Only 'trial_id', 'grasp_id', 'tools', 'result', and 'comment' are allowed."}, indent=2)
        
        # Build trial dict (include comment only if provided)
        trial_dict = {
            "trial_id": trial_id,
            "grasp_id": grasp_id,
            "tools": tools,
            "result": result
        }
        if comment is not None:
            trial_dict["comment"] = comment
        
        validated_trials.append(trial_dict)
    
    # Update or create the object entry
    if object_found:
        sequence[object_index]["tools_trials"] = validated_trials
    else:
        sequence.append({
            "sequence_id": sequence_id,
            "object_name": object_name,
            "tools_trials": validated_trials
        })
    
    # Update the sequence in data
    data["sequence"] = sequence
    save_assembly_sequence_resource(assembly_id, data)
    
    return json.dumps({"success": True}, indent=2)

@mcp.tool()
def clear_assembly_sequence_resource(assembly_id: str) -> str:
    """
    Clear/delete an assembly sequence resource
    
    Args:
        assembly_id: The ID of the assembly to clear
    
    Returns:
        JSON string with confirmation or error message
    """
    assembly_file = get_assembly_file(assembly_id)
    
    if not assembly_file.exists():
        return json.dumps({"success": False}, indent=2)
    
    # Delete the file
    try:
        assembly_file.unlink()
        return json.dumps({"success": True}, indent=2)
    except Exception as e:
        return json.dumps({"success": False, "error": str(e)}, indent=2)

@mcp.tool()
def list_assembly_sequence_resource() -> str:
    """
    List all assemblies in the assembly sequence resource
    
    Returns:
        JSON string containing all assembly IDs
    """
    # Find all Assembly_{id}_sequence_log.json files
    assembly_files = list(RESOURCES_DIR.glob("Assembly_*_sequence_log.json"))
    assembly_ids = []
    
    for file in assembly_files:
        # Extract assembly ID from filename like "Assembly_1_sequence_log.json"
        name = file.stem  # "Assembly_1_sequence_log"
        if name.startswith("Assembly_") and name.endswith("_sequence_log"):
            # Remove "Assembly_" prefix and "_sequence_log" suffix
            assembly_id = name.replace("Assembly_", "").replace("_sequence_log", "")
            assembly_ids.append(assembly_id)
    
    return json.dumps({
        "assembly_ids": assembly_ids,
        "count": len(assembly_ids)
    }, indent=2)

# ========== TOOLS FOR DISASSEMBLY GRASP RESOURCE (Disassembly{id}/object_name/grasp_configs) ==========

@mcp.tool()
def read_disassembly_grasp_resource(assembly_id: str, object_name: str) -> str:
    """
    Read the disassembly grasp configurations for an object in a specific assembly (grasp_configs with grasp_id and result)
    
    Args:
        assembly_id: The ID of the assembly
        object_name: The name of the object
    
    Returns:
        JSON string containing grasp_configs list with grasp_id and result (includes both SUCCESS and FAILURE attempts)
        Each config has: {"grasp_id": <int>, "result": "SUCCESS"|"FAILURE"}
    """
    data = load_disassembly_grasp_resource(assembly_id)
    obj_data = data.get(object_name, [])
    
    if not obj_data:
        return json.dumps({
            "assembly_id": assembly_id,
            "object_name": object_name,
            "grasp_configs": [],
            "message": "Resource not found"
        }, indent=2)
    
    if isinstance(obj_data, list):
        return json.dumps({
            "assembly_id": assembly_id,
            "object_name": object_name,
            "grasp_configs": obj_data
        }, indent=2)
    else:
        return json.dumps({
            "assembly_id": assembly_id,
            "object_name": object_name,
            "grasp_configs": []
        }, indent=2)

@mcp.tool()
def write_disassembly_grasp_resource(assembly_id: str, object_name: str, grasp_configs: Optional[List[Dict[str, Any]]] = None) -> str:
    """
    Write or update disassembly grasp configurations for an object in a specific assembly (grasp_configs list with grasp_id and result)
    
    Args:
        assembly_id: The ID of the assembly
        object_name: The name of the object
        grasp_configs: List of grasp configurations, each with grasp_id and result.
                       Format: [{"grasp_id": 1, "result": "SUCCESS"}, ...]
                       - grasp_id: integer (required)
                       - result: "SUCCESS" or "FAILURE" (required)
                       Both success and failure attempts are stored.
    
    Returns:
        JSON string with confirmation or error message
    """
    try:
        data = load_disassembly_grasp_resource(assembly_id)
    except json.JSONDecodeError as e:
        return json.dumps({"success": False, "error": f"Error loading resource file: {str(e)}"}, indent=2)
    except Exception as e:
        return json.dumps({"success": False, "error": f"Unexpected error loading resource: {str(e)}"}, indent=2)
    
    if grasp_configs is not None:
        # Handle case where grasp_configs might be a JSON string
        if isinstance(grasp_configs, str):
            try:
                grasp_configs = json.loads(grasp_configs)
            except json.JSONDecodeError as e:
                return json.dumps({"success": False, "error": f"Invalid JSON in grasp_configs: {str(e)}"}, indent=2)
        
        # Ensure it's a list
        if not isinstance(grasp_configs, list):
            return json.dumps({"success": False, "error": f"grasp_configs must be a list, got: {type(grasp_configs).__name__}"}, indent=2)
        processed_configs = []
        for i, config in enumerate(grasp_configs):
            # Ensure config is a dict
            if not isinstance(config, dict):
                return json.dumps({"success": False, "error": f"Config at index {i} must be a dictionary, got: {type(config).__name__}"}, indent=2)
            
            # Check for required fields
            if "grasp_id" not in config:
                return json.dumps({"success": False, "error": f"Config at index {i} must have a 'grasp_id' field"}, indent=2)
            
            if "result" not in config:
                return json.dumps({"success": False, "error": f"Config at index {i} must have a 'result' field"}, indent=2)
            
            # Reject any extra fields - only allow these two fields
            allowed_fields = {"grasp_id", "result"}
            extra_fields = set(config.keys()) - allowed_fields
            if extra_fields:
                return json.dumps({"success": False, "error": f"Invalid fields found: {list(extra_fields)}. Only 'grasp_id' and 'result' are allowed."}, indent=2)
            
            # Validate grasp_id is an integer
            try:
                grasp_id = int(config.get("grasp_id"))
            except (ValueError, TypeError):
                return json.dumps({"success": False, "error": f"grasp_id must be an integer, got: {config.get('grasp_id')}"}, indent=2)
            
            # Validate result
            result = config.get("result", "").upper()
            if result not in ["SUCCESS", "FAILURE"]:
                return json.dumps({"success": False, "error": f"Invalid result: {config.get('result')}. Must be 'SUCCESS' or 'FAILURE'"}, indent=2)
            
            # Store only these two fields (no gripper_state)
            processed_configs.append({
                "grasp_id": grasp_id,
                "result": result  # Store as uppercase
            })
        
        # Store as list - includes both success and failure attempts
        data[object_name] = processed_configs
    
    save_disassembly_grasp_resource(assembly_id, data)
    
    return json.dumps({"success": True}, indent=2)

@mcp.tool()
def clear_disassembly_grasp_resource(assembly_id: str, object_name: str) -> str:
    """
    Clear/delete a disassembly grasp resource for an object in a specific assembly
    
    Args:
        assembly_id: The ID of the assembly
        object_name: The name of the object to clear
    
    Returns:
        JSON string with confirmation or error message
    """
    data = load_disassembly_grasp_resource(assembly_id)
    
    if object_name not in data:
        return json.dumps({"success": False}, indent=2)
    
    del data[object_name]
    
    save_disassembly_grasp_resource(assembly_id, data)
    
    return json.dumps({"success": True}, indent=2)

@mcp.tool()
def list_disassembly_grasp_resource() -> str:
    """
    List all assemblies in the disassembly grasp resource
    
    Returns:
        JSON string containing all assembly IDs that have disassembly grasp logs
    
    Note: Disassembly grasp configurations only store grasp_id and result (no gripper_state).
    """
    # Find all Disassembly_{id}_grasp_log.json files
    disassembly_files = list(RESOURCES_DIR.glob("Disassembly_*_grasp_log.json"))
    assembly_ids = []
    
    for file in disassembly_files:
        # Extract assembly ID from filename like "Disassembly_1_grasp_log.json"
        name = file.stem  # "Disassembly_1_grasp_log"
        if name.startswith("Disassembly_") and name.endswith("_grasp_log"):
            # Remove "Disassembly_" prefix and "_grasp_log" suffix
            assembly_id = name.replace("Disassembly_", "").replace("_grasp_log", "")
            assembly_ids.append(assembly_id)
    
    return json.dumps({
        "assembly_ids": assembly_ids,
        "count": len(assembly_ids)
    }, indent=2)

# ========== TOOLS FOR FINAL SEQUENCE (Assembly{id}_final_sequence.json) ==========
# NOTE: Final sequence tools have been removed per user request

if __name__ == "__main__":
    mcp.run()
