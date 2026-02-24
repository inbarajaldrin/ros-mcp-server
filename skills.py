from typing import Annotated, Literal

from pydantic import Field
from fastmcp import FastMCP

mcp = FastMCP("Skills")

@mcp.prompt()
def grasp_workflow() -> str:
    """Workflow for grasping an object."""
    return """Grasp Workflow:
1. Call move_to_object to move to the grasp point (make sure the gripper is open before this call)
2. Call control_gripper to grasp the object
3. Call move_to_safe_height to move to the safe height (z=0.3m)"""

@mcp.prompt()
def regrasp_workflow() -> str:
    """Workflow for regrasping an object with a top-down EE orientation."""
    return """Regrasp Workflow:
1. Call rotate_object to move the object to the target orientation relative to base orientation (which results in a non-optimal end effector orientation)
2. Call move_to_clear_space to move to the clear space (make sure the objects is already grasped and rotated)
3. Call move_down to place the object on the table
4. Call control_gripper to release the object
5. Call move_ee_top_down to move the EE to the top-down orientation at z=0.3m
6. Call move_to_grasp to grasp the object again using the grasp id from the disassembly results
7. IMPORTANT: Call rotate_object again to restore the object's orientation. Opening the gripper to drop the object typically causes minor orientation changes, so this step is REQUIRED to correct the orientation back to the target before continuing.
8. Continue with what you were doing"""

@mcp.prompt()
def assembly_workflow() -> str:
    """Workflow for assembling an object onto the base."""
    return """Assembly Workflow:
1. Call move_to_base to move to the base
2. Call perform_insert to move down to the final position (Make sure the object orientation is correct before this call)
3. Call control_gripper to release the object
4. Call move_to_safe_height to move to the safe height (z=0.3m)"""

@mcp.prompt()
def disassembly_workflow() -> str:
    """Workflow for disassembling an object from the base."""
    return """Disassembly Workflow:
1. Call move_away_from_base once you are holding the object and in safe height to move the object away from the base
2. Call control_gripper to release the object"""

if __name__ == "__main__":
    mcp.run()
