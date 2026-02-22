"""Signal Operator Elicitation - General-purpose human-in-the-loop confirmation.

Related files:
  - server.py (signal_operator tool) — invokes this elicitation with agent-provided message
  - elicitations/signal_operator.py   (this file) — Pydantic schema for operator confirmation

Used by the signal_operator tool whenever the agent needs a human operator to perform
a physical action (e.g., swap a part, inspect the workspace) and confirm before the
robot continues. The agent provides the message content; this schema only handles
the proceed/abort decision.
"""

from pydantic import BaseModel, Field


class SignalOperatorSchema(BaseModel):
    """Schema for operator confirmation after a robot-requested action."""

    action: str = Field(
        default="proceed",
        description="What would you like to do?",
        json_schema_extra={"enum": ["proceed", "abort"]}
    )

    feedback: str = Field(
        default="",
        description="Optional notes for the robot"
    )


def get_elicitation_schema(context_data: dict = None):
    """Return the elicitation schema for operator signaling.

    Args:
        context_data: Optional context (unused, single schema).
    """
    return SignalOperatorSchema


def build_elicitation_message(context_data: dict) -> str:
    """Pass through the agent's message.

    Args:
        context_data: Dict with "message" key containing the agent's message.

    Returns:
        The agent's message string.
    """
    return context_data.get("message", "The robot is waiting for operator confirmation.")
