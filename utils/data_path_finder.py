"""
Path Finder Utility
Locates the CAD data (part meshes, ArUco layouts, grasp points, symmetry, assembly JSONs) and provides shared
assembly JSON loading functions.

The data lives IN this repo at assets/cad_data/ (copied 2026-11-16 from the former external aruco-grasp-annotator
data dir, so the repo runs without it). FMB_CAD_DATA overrides the location.
"""
import json
import os
import glob as glob_module
from pathlib import Path
from typing import Optional, Dict, Any, List

REPO_CAD_DATA = Path(__file__).resolve().parents[1] / "assets" / "cad_data"


def find_aruco_data_dir() -> Optional[Path]:
    """The CAD data directory: $FMB_CAD_DATA if set, else <repo>/assets/cad_data. None if it has no grasp_points/."""
    data_dir = Path(os.environ["FMB_CAD_DATA"]) if os.environ.get("FMB_CAD_DATA") else REPO_CAD_DATA
    return data_dir if (data_dir / "grasp_points").is_dir() else None


def get_aruco_data_dir() -> Path:
    """The CAD data directory (raises FileNotFoundError if it is missing)."""
    found_dir = find_aruco_data_dir()
    if found_dir:
        return found_dir

    raise FileNotFoundError(
        f"No CAD data directory with grasp_points/ at {os.environ.get('FMB_CAD_DATA') or REPO_CAD_DATA} "
        "(set FMB_CAD_DATA or restore assets/cad_data)."
    )


def get_symmetry_dir() -> Path:
    """Get symmetry directory path."""
    return get_aruco_data_dir() / "symmetry"


def get_assembly_data_dir() -> Path:
    """Get assembly data directory path (same as data dir)."""
    return get_aruco_data_dir()


ASSEMBLY_JSON_FOR_BASE = {"base1": "fmb_assembly1_seated.json", "base3": "fmb_assembly3.json"}


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

    json_files = sorted(glob_module.glob(os.path.join(data_dir, "*.json")))
    # Two JSONs name each of base1 and base3, and the first match used to depend on the filesystem's directory order
    # (an unsorted glob). Pin the file every run so far resolved to (checked 2026-11-16 on the cell PC), so a clone on
    # another machine picks the same one; any other base takes the first match in sorted order.
    pinned = os.path.join(data_dir, ASSEMBLY_JSON_FOR_BASE.get(base_name, ""))
    if base_name in ASSEMBLY_JSON_FOR_BASE and os.path.isfile(pinned):
        json_files = [pinned] + [f for f in json_files if f != pinned]

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
    - <cad data>/models/
    - <cad data>/objects/
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

