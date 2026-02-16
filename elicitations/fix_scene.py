"""Fix Scene Elicitation - Schemas for human interaction when clearance check fails.

Related files (clearance check pipeline):
  - queries/verify_clearance.py    — ROS query that checks poses and computes clearance
  - triggers/pre_assembly_check.py — handles failure by invoking human elicitation
  - elicitations/fix_scene.py      (this file) — Pydantic schemas for human interaction

Used by the pre_assembly_check trigger when objects are missing or have insufficient
clearance. Presents the human with options to fix the scene and retry or skip.
"""

from pydantic import BaseModel, Field


class MissingObjectsSchema(BaseModel):
    """Schema when objects are missing from the scene."""

    objects_placed: bool = Field(
        default=False,
        description="Have you placed the missing objects in the workspace?"
    )

    action: str = Field(
        default="retry",
        description="What would you like to do?",
        json_schema_extra={"enum": ["retry", "skip"]}
    )


class ClearanceIssuesSchema(BaseModel):
    """Schema when objects have insufficient clearance."""

    objects_repositioned: bool = Field(
        default=False,
        description="Have you moved the objects further apart so the gripper can reach them?"
    )

    action: str = Field(
        default="retry",
        description="What would you like to do?",
        json_schema_extra={"enum": ["retry", "skip"]}
    )


class MissingAndClearanceSchema(BaseModel):
    """Schema when both missing objects and clearance issues exist."""

    objects_placed: bool = Field(
        default=False,
        description="Have you placed the missing objects in the workspace?"
    )

    objects_repositioned: bool = Field(
        default=False,
        description="Have you moved the objects further apart so the gripper can reach them?"
    )

    action: str = Field(
        default="retry",
        description="What would you like to do?",
        json_schema_extra={"enum": ["retry", "skip"]}
    )


def get_elicitation_schema(result: dict = None):
    """Return the appropriate elicitation schema based on the error type.

    Args:
        result: The failed verification result. If None, returns the combined schema.
    """
    if result is None:
        return MissingAndClearanceSchema

    has_missing = bool(result.get("missing_objects"))
    has_clearance = bool(result.get("objects_with_clearance_issues"))

    if has_missing and has_clearance:
        return MissingAndClearanceSchema
    elif has_missing:
        return MissingObjectsSchema
    elif has_clearance:
        return ClearanceIssuesSchema
    else:
        return MissingAndClearanceSchema


def build_elicitation_message(result: dict) -> str:
    """Build an elicitation message from clearance verification results.

    Args:
        result: The failed verification result containing missing_objects and/or clearance_issues

    Returns:
        Formatted message string for the user
    """
    message_parts = []

    missing = result.get("missing_objects", [])
    clearance_issues = result.get("objects_with_clearance_issues", [])

    if missing:
        message_parts.append(f"Missing objects: {', '.join(missing)}")

    if clearance_issues:
        message_parts.append(f"Clearance issues: {', '.join(clearance_issues)} (too close to other objects for gripper access)")

    return "\n".join(message_parts)
