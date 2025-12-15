"""
Path Finder Utility
Recursively searches for aruco-grasp-annotator data directory in Documents folder.
"""
from pathlib import Path
from typing import Optional


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
            # Verify it has the expected structure (grasp subdirectory)
            if data_dir.exists() and (data_dir / "grasp").exists():
                return data_dir
    
    return None


def get_aruco_data_dir() -> Path:
    """
    Get aruco-grasp-annotator data directory.
    
    Returns:
        Path to data directory (raises FileNotFoundError if not found)
    """
    found_dir = find_aruco_data_dir()
    if found_dir:
        return found_dir
    
    raise FileNotFoundError(
        "Could not find aruco-grasp-annotator data directory in Documents folder."
    )


def get_symmetry_dir() -> Path:
    """Get symmetry directory path."""
    return get_aruco_data_dir() / "symmetry"


def get_assembly_data_dir() -> Path:
    """Get assembly data directory path (same as data dir)."""
    return get_aruco_data_dir()

