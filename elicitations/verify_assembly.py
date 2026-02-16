"""Verify Assembly Elicitation - Ask a human operator to verify real-world assembly results.

Related files (phase signal pipeline):
  - triggers/signal_phase_complete.py — handles phase signals, invokes human for phase 3
  - elicitations/verify_assembly.py   (this file) — Pydantic schemas for human verification of real-world assembly

Used by Phase 3 (real-world execution) to get human confirmation that the
physical assembly was completed correctly before reporting final results.
"""

from pydantic import BaseModel, Field


class AssemblyVerificationSchema(BaseModel):
    """Schema for human verification of real-world assembly."""

    assembly_correct: bool = Field(
        default=False,
        description="Is the physical assembly correctly completed?"
    )

    action: str = Field(
        default="confirm",
        description="What would you like to do?",
        json_schema_extra={"enum": ["confirm", "retry", "abort"]}
    )

    feedback: str = Field(
        default="",
        description="Optional feedback or notes about the assembly result"
    )


class AssemblyFailureSchema(BaseModel):
    """Schema when the agent reports assembly failure."""

    acknowledged: bool = Field(
        default=False,
        description="Do you acknowledge the assembly failure?"
    )

    action: str = Field(
        default="abort",
        description="What would you like to do?",
        json_schema_extra={"enum": ["retry", "abort"]}
    )

    feedback: str = Field(
        default="",
        description="Optional feedback or instructions for retry"
    )


def get_elicitation_schema(result: dict = None):
    """Return the appropriate elicitation schema based on the phase result.

    Args:
        result: The phase completion result. If None, returns the default schema.
    """
    if result is None:
        return AssemblyVerificationSchema

    status = result.get("status", "success")
    if status == "failure":
        return AssemblyFailureSchema
    return AssemblyVerificationSchema


def build_elicitation_message(result: dict) -> str:
    """Build a human-readable elicitation message from phase 3 results.

    Args:
        result: The phase completion result dict.

    Returns:
        Formatted message string for the human operator.
    """
    status = result.get("status", "unknown")
    comment = result.get("comment", "")

    if status == "success":
        message = "Phase 3 (Real-World Execution) - Assembly Complete\n\n"
        message += "The robot reports the physical assembly is complete.\n"
        message += "Please verify the assembly is correct before confirming."
    else:
        message = "Phase 3 (Real-World Execution) - Assembly Issue\n\n"
        message += "The robot reports a problem with the physical assembly."

    if comment:
        message += f"\n\nAgent comment: {comment}"

    return message
