"""Setup Real Scene Elicitation - Ask a human operator to prepare the physical workspace.

Related files (workspace preparation pipeline):
  - server.py (prepare_workspace tool) — invokes this elicitation before clearance check
  - elicitations/setup_real_scene.py   (this file) — Pydantic schemas for real scene setup confirmation

Used by the prepare_workspace tool at the start of Phase 3 to get human confirmation that:
  1. Physical objects are placed in the scene
  2. The real robot is spawned and ready
before running clearance verification and starting real-world execution.
"""

from pydantic import BaseModel, Field


class SetupRealSceneSchema(BaseModel):
    """Schema for human confirmation that the real scene is ready."""

    objects_placed: bool = Field(
        default=False,
        description="Are all assembly objects placed in the physical workspace?"
    )

    robot_spawned: bool = Field(
        default=False,
        description="Is the real robot spawned and ready?"
    )

    action: str = Field(
        default="proceed",
        description="What would you like to do?",
        json_schema_extra={"enum": ["proceed", "abort"]}
    )

    feedback: str = Field(
        default="",
        description="Optional notes about the scene setup"
    )


def get_elicitation_schema(result: dict = None):
    """Return the elicitation schema for real scene setup.

    Args:
        result: Optional context data (unused, single schema).
    """
    return SetupRealSceneSchema


def build_elicitation_message(result: dict) -> str:
    """Build a human-readable elicitation message for scene setup.

    Args:
        result: Context dict with phase info.

    Returns:
        Formatted message string for the human operator.
    """
    message = "Phase 3 — Prepare Real Scene\n\n"
    message += "The assembly sequence has been verified in simulation.\n"
    message += "Before starting real-world assembly:\n\n"
    message += "1. Place all assembly objects in the physical workspace\n"
    message += "2. Spawn the real robot\n\n"
    message += "Confirm when ready to proceed."
    return message
