"""
Path Finder Utility
Resolves the grasp/assembly data directory. PRIMARY source is the IN-REPO ros-mcp-server/data/
(self-contained; curated + pushed from the Mac aruco-grasp-annotator pipeline). The legacy recursive
search for a local aruco-grasp-annotator/data checkout is kept ONLY as a fallback — that checkout on
dual-a4500 is STALE and must not be the source of truth.
Provides shared assembly JSON loading functions.
"""
import os
import json
import glob as glob_module
from pathlib import Path
from typing import Optional, Dict, Any, List

# In-repo data dir: ros-mcp-server/data (this file is ros-mcp-server/utils/data_path_finder.py).
_REPO_DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def find_aruco_data_dir() -> Optional[Path]:
    """
    Recursively search Documents folder for aruco-grasp-annotator/data directory.
    
    Returns:
        Path to data directory if found, None otherwise
    """
    # Recursively search only in Documents folder
    documents_dir = Path.home() / "Documents"
    if not documents_dir.exists():
        return None
    
    # Search for aruco-grasp-annotator directory
    target_name = "aruco-grasp-annotator"
    for path in documents_dir.rglob(target_name):
        if path.is_dir():
            data_dir = path / "data"
            # Verify it has the expected structure. Accept either the candidate-native
            # layout (grasp_candidates/, the current path) or the legacy grasp_points/ layout
            # so the finder works during/after the candidate hard-switch.
            if data_dir.exists() and (
                (data_dir / "grasp_candidates").exists() or (data_dir / "grasp_points").exists()
            ):
                return data_dir

    return None


def get_aruco_data_dir() -> Path:
    """
    Resolve the grasp/assembly data directory.

    Order: $ROS_MCP_DATA_DIR override -> the in-repo ros-mcp-server/data/ (PRIMARY, curated from the Mac
    pipeline) -> recursive search for a local aruco-grasp-annotator/data checkout (STALE fallback only).

    Returns:
        Path to data directory (raises FileNotFoundError if none found)
    """
    env = os.environ.get("ROS_MCP_DATA_DIR")
    if env and Path(env).exists():
        return Path(env)

    if _REPO_DATA_DIR.exists() and (
        (_REPO_DATA_DIR / "grasp_candidates").exists() or (_REPO_DATA_DIR / "grasp_points").exists()
    ):
        return _REPO_DATA_DIR

    found_dir = find_aruco_data_dir()
    if found_dir:
        return found_dir

    raise FileNotFoundError(
        "No grasp data dir: set ROS_MCP_DATA_DIR, populate ros-mcp-server/data/, "
        "or place an aruco-grasp-annotator/data checkout in Documents."
    )


def get_symmetry_dir() -> Path:
    """Get symmetry directory path."""
    return get_aruco_data_dir() / "symmetry"


def get_grasp_candidates_dir() -> Path:
    """Get the grasp_candidates directory (schema_version 2 candidate JSONs)."""
    return get_aruco_data_dir() / "grasp_candidates"


def load_grasp_candidate(object_name: str, candidate_id: int):
    """Load one grasp candidate (schema_version 2) by composite id.

    candidate_id = grasp_point_id*100 + direction_id (e.g. 101 = gp 1 / dir 1, 202 = gp 2 / dir 2).

    Returns a (candidate_dict, gripper_meta_dict) tuple, or (None, None) if the file or the
    candidate id is not found. gripper_meta carries the file-level `gripper` block (max_width_mm,
    clearance_mm, tip_thickness_mm) — the candidate dict alone does NOT include clearance_mm, so
    callers that derive W_grip = width_mm - clearance_mm must read it from gripper_meta (never a
    duplicated constant).
    """
    try:
        cand_dir = get_grasp_candidates_dir()
    except FileNotFoundError:
        return None, None

    cand_file = cand_dir / f"{object_name}_grasp_candidates.json"
    if not cand_file.exists():
        return None, None

    try:
        with open(cand_file, 'r') as f:
            data = json.load(f)
    except (json.JSONDecodeError, IOError):
        return None, None

    gripper_meta = data.get('gripper', {})
    for cand in data.get('grasp_candidates', []):
        cid = cand.get('grasp_point_id', 0) * 100 + cand.get('direction_id', 0)
        if cid == candidate_id:
            return cand, gripper_meta

    return None, gripper_meta


def get_assembly_data_dir() -> Path:
    """Get assembly data directory path (same as data dir)."""
    return get_aruco_data_dir()


def find_assembly_json_by_base_name(base_name: str, data_dir: str = None, logger=None) -> Optional[str]:
    """
    Find the assembly JSON file that contains the given base name.

    Args:
        base_name: Name of the base object to search for
        data_dir: Directory to search for JSON files (defaults to assembly data dir)
        logger: Optional logger for debug output

    Returns:
        Path to the matching JSON file, or None if not found
    """
    if data_dir is None:
        try:
            data_dir = str(get_assembly_data_dir())
        except FileNotFoundError:
            return None

    import os
    if not os.path.exists(data_dir):
        if logger:
            logger.error(f"Data directory not found: {data_dir}")
        return None

    json_files = glob_module.glob(os.path.join(data_dir, "*.json"))

    for json_file in json_files:
        try:
            with open(json_file, 'r') as f:
                config = json.load(f)
            for component in config.get('components', []):
                if component.get('name', '') == base_name:
                    return json_file
        except (json.JSONDecodeError, IOError) as e:
            if logger:
                logger.debug(f"Skipping invalid JSON file {json_file}: {e}")
            continue

    if logger:
        logger.warn(f"No assembly JSON found for base '{base_name}' in {data_dir}")
    return None


def load_assembly_config(base_name: str, data_dir: str = None, logger=None) -> Dict[str, Any]:
    """
    Load assembly configuration from JSON file for a given base name.

    Args:
        base_name: Name of the base object
        data_dir: Directory to search (defaults to assembly data dir)
        logger: Optional logger

    Returns:
        Assembly config dict, or empty dict if not found
    """
    json_file = find_assembly_json_by_base_name(base_name, data_dir, logger)
    if json_file is None:
        return {}
    try:
        with open(json_file, 'r') as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        if logger:
            logger.error(f"Error loading assembly config from {json_file}: {e}")
        return {}


def load_assembly_order_map() -> Dict[str, int]:
    """Load assembly_order from all assembly JSON files.

    Returns:
        Dict mapping object_name -> assembly_order (int)
    """
    order_map = {}
    try:
        data_dir = get_assembly_data_dir()
    except FileNotFoundError:
        return order_map

    for json_file in glob_module.glob(str(data_dir / "*.json")):
        try:
            with open(json_file, 'r') as f:
                config = json.load(f)
            for component in config.get('components', []):
                name = component.get('name', '')
                order = component.get('assembly_order')
                if name and order is not None:
                    order_map[name] = order
        except (json.JSONDecodeError, IOError):
            continue

    return order_map


def load_disassembly_order_map() -> Dict[str, int]:
    """Load disassembly_order (reverse of assembly_order) from all assembly JSON files.

    Disassembly order: last assembled = first to disassemble.
    Formula: max_assembly_order - assembly_order + 1 (board order 0 excluded).

    Returns:
        Dict mapping object_name -> disassembly_order (int). Boards are excluded.
    """
    order_map = {}
    try:
        data_dir = get_assembly_data_dir()
    except FileNotFoundError:
        return order_map

    for json_file in glob_module.glob(str(data_dir / "*.json")):
        try:
            with open(json_file, 'r') as f:
                config = json.load(f)
            components = config.get('components', [])
            max_order = max(
                (c.get('assembly_order', 0) for c in components if c.get('type') != 'board'),
                default=0
            )
            for component in components:
                name = component.get('name', '')
                order = component.get('assembly_order')
                if name and order is not None and order > 0:
                    order_map[name] = max_order - order + 1
        except (json.JSONDecodeError, IOError):
            continue

    return order_map


def find_object_cad_file(object_name: str) -> Optional[Path]:
    """
    Find CAD .obj file for an object by name.

    Searches in common locations:
    - aruco-grasp-annotator/data/models/
    - aruco-grasp-annotator/data/objects/
    - etc.

    Args:
        object_name: Name of the object (e.g., 'u_orange', 'fork')

    Returns:
        Path to .obj file if found, None otherwise
    """
    try:
        data_dir = get_aruco_data_dir()
    except FileNotFoundError:
        return None

    # Search for .obj files with matching name in data directory and subdirectories
    for obj_file in data_dir.rglob(f"{object_name}.obj"):
        return obj_file

    # Also try with common prefixes
    for obj_file in data_dir.rglob(f"*{object_name}*.obj"):
        return obj_file

    return None

