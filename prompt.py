from typing import Annotated, Literal

from pydantic import Field
from fastmcp import FastMCP

mcp = FastMCP("Prompts")

@mcp.prompt()
def system_context() -> str:
    """
    Common system context for all phases.
    Injected via systemPrompt in YAML.
    """
    return """You are an autonomous agent operating a 6 DOF robotic arm and a parallel gripper.

**Environment context:**

Common parameters used across all tools:
- mode: "sim" = simulation environment, "real" = physical robot.
- object_name: name of the object being manipulated.
- base_name: name of the fixed assembly base.
- grasp_id: grasp point ID from the assembly resource."""


@mcp.prompt()
def phase_1_disassembly_sequence_discovery(
    orchestrator: Annotated[Literal["enabled"], Field(description="Set to 'enabled' to require orchestration: once a working sequence is verified for one object, orchestrate remaining objects (fallback to individual calls on failure)")] = None,
    grasp_hint: Annotated[Literal["enabled"], Field(description="Set to 'enabled' to include grasp z-height accessibility hint")] = None,
) -> str:
    """
    Phase 1: Disassembly sequence discovery in a simulated environment.
    Discovers accessible grasp IDs and disassembly sequence.
    """
    if orchestrator == "enabled":
        orchestrator_instruction = (
            "\nOnce you have verified a working sequence for one object, "
            "orchestrate a composed policy for the remaining objects."
        )
    else:
        orchestrator_instruction = ""

    if grasp_hint == "enabled":
        grasp_hint_text = "\nGrasp points with higher z position are more accessible for top-down grasping."
    else:
        grasp_hint_text = ""

    return f"""Phase 1: Disassembly sequence discovery in a simulated environment.

**Initialization:**

You will be provided the fully assembled assembly in the environment.

Save scene state at the start as "init.json" before touching anything — this is the assembled baseline.

Then open the gripper and move the robot home before disassembling the first object.

**Task**

Your goal is to disassemble the objects into the clear region of the workspace,
thereby figuring out which grasp ids let you perform the disassembly.
Disassemble in reverse assembly order. The grasp ids you commit in this run are
reused for assembly later.

**Execution**

Perform disassembly one object at a time.{grasp_hint_text}{orchestrator_instruction}
If you exhaust all strategies for an object, signal failure — do not skip to the next object.

**Verification**

Run verify disassembly once you've run all tools required to move one object away from the fixed base.

**SUCCESS** — verify disassembly returns success:
1. Move to safe height.
2. Save scene state as "{{object_name}}.json".
3. Commit the object with commit_object — the grasp id you pass is the record of what worked. Then move on to the next object.

**FAILURE** — verify disassembly returns failure:
1. Figure out if there are any steps you can continue to take to complete the disassembly.
2. If not, restore scene state and try a different grasp id. Not all grasp points are accessible from every object orientation.

**Post Execution**
Signal you are done with the disassembly of all objects.
"""

@mcp.prompt()
def phase_2_assembly_sequence_discovery(
    orchestrator: Annotated[Literal["enabled"], Field(description="Set to 'enabled' to require orchestration: once a working sequence is verified for one object, orchestrate remaining objects (fallback to individual calls on failure)")] = None,
) -> str:
    """
    Phase 2: Assembly sequence discovery in a simulated environment.
    """
    if orchestrator == "enabled":
        orchestrator_instruction = (
            "\nOnce you have verified a working sequence for one object, "
            "orchestrate a composed policy for the remaining objects."
        )
    else:
        orchestrator_instruction = ""

    return f"""Phase 2: Assembly sequence discovery in a simulated environment.

**Initialization:**

Read the disassembly results to identify the grasp ids successfully used to disassemble each object from a fully assembled state. Use the same grasp ids to assemble them back when their current orientation matches their target orientation.

Save scene state at the start before beginning assembly as "init.json".

Then open the gripper and move the robot home before assembling the first object.

**Task**

Your goal is to perform assembly of the objects scattered in the scene onto the base object.

**Execution**

Grasp points with higher z position are more accessible for top-down grasping.
If you exhaust all strategies for an object, signal failure — do not skip to the next object.{orchestrator_instruction}

**Verification**

Run verify assembly once you've run all tools required to assemble one object into the fixed base.

**SUCCESS** — verify assembly returns success:
1. Move to safe height.
2. Save scene state with "{{object_name}}.json".
3. Commit the object with commit_object, then move on to the next object.

**FAILURE** — verify assembly returns failure:
1. Figure out if there are any steps you can continue to take to complete the assembly.
2. If not, restore scene state and try a different sequence you haven't tried before.

**Post Execution**
Signal you are done with the assembly of all objects.
"""

@mcp.prompt()
def phase_3_perform_assembly(mode: Annotated[Literal["sim", "real"], Field(description="Environment: sim or real")]) -> str:
    """
    Phase 3: Performing Assembly in simulation or real world.
    """
    if mode == "real":
        setup_instruction = ("Signal the operator (signal_operator) to set up and verify the workspace, "
                             "and wait for confirmation before starting.\n")
        init_save = ""
        success_block = """1. Move to safe height.
2. Commit the object with commit_object, then move on to the next object."""
    else:
        setup_instruction = ""
        init_save = '\nSave scene state at the start as "init.json".\n'
        success_block = """1. Move to safe height.
2. Save scene state as "{object_name}.json".
3. Commit the object with commit_object, then move on to the next object."""

    return f"""Phase 3: Performing Assembly ({mode} mode).

**Initialization:**

{setup_instruction}Read the assembly recipe to identify the tool call sequence for each object.
{init_save}
Then open the gripper and move the robot home before assembling the first object.

**Task**

Perform assembly of the objects onto the fixed base following the recipe.

**Execution**

Follow the recipe's object order and tool calls with their arguments, one object at a time. Do not skip any object.
If you exhaust all strategies for an object, signal failure — do not skip to the next object.

**Verification**

Run verify assembly once you've run all tools required to assemble one object into the fixed base.

**SUCCESS** — verify assembly returns success:
{success_block}

**FAILURE** — verify assembly returns failure or any of the steps failed:
1. Grasp the object, place it down in clear space, and start that object's cycle over (regrasp and retry).
2. If the object is still not assembled after two tries, signal the operator for assistance.

**Post Execution**
Signal you are done with the final assembly of all objects.
"""

@mcp.prompt()
def replace_defective_part(
    object_name: Annotated[str, Field(description="Name of the defective object")],
    base_name: Annotated[str, Field(description="Name of the base assembly")],
) -> str:
    """
    An operator reports a defective part during assembly.
    """
    return f"""{object_name} assembled in {base_name} is defective. Disassemble that object and place it in the clear space and notify me. I'll replace the defective component with a new one and then you can continue the assembly."""

if __name__ == "__main__":
    mcp.run()
