from mcp.server.fastmcp import FastMCP

mcp = FastMCP("Prompts")

@mcp.prompt()
def phase_1_identify_accessible_grasp_ids() -> str:
    """
    Returns a prompt for identifying accessible grasp ids in a simulation environment.
    """
    return f"""**Initialize:**

You are working in a sim environment
Identify available objects and their grasp ids.
Save scene state

**Task**

Your task is to find which grasp ids are accessible per object and if we need to perform half open before moving to grasp.

**Execution**

Use the tools to find the right control gripper mode for each grasp id of each object.
Gripper is in open mode by default. But for some objects in the scene, the gripper needs to be half open before accessing that specific grasp id.
And some grasp ids are not accesible at all.
Verify if the object is actually grabbed by moving to safe height and using verify_grasp.
Once you've found information about a grasp id for an object save it to grasp logs. Save both SUCCESS and FAILURE.
If it was a FAILURE, restore scene state.

**Verification**

> To verify if a grasp was successful:
> After moving to safe height, use `verify_grasp` to check if the object is within 6cm radius from the gripper center:
> - **SUCCESS**: verify_grasp returns success (object is within grasp radius)
> - **FAILURE**: verify_grasp returns failure (object is outside grasp radius) - the object was not successfully grasped

**Output**
Use `write_grasp_resource(assembly_id, object_name, grasp_configs)` to log each attempt.
Each config should have: {{"grasp_id": <int>, "gripper_state": "open"|"half-open", "result": "SUCCESS"|"FAILURE"}}
Resource endpoint: Assembly{{assembly_id}}/object_name/{{object_name}}/grasp_configs/result/{{result}}"""

@mcp.prompt()
def phase_2_identify_grasp_id_per_object() -> str:
    """
    Returns a prompt for performing disassembly and identifying grasp id per object in a simulation environment.
    """
    return """**Initialize:**

You are working in a sim environment
Find the objects available in the scene. Save scene.
Identify accessible grasp ids per objectfrom  the grasp log resource.
You will be provided the fully assembled assembly in the scene.

**Task**

Your goal is to dissassemble the assembly there by figuring out which of the grasp ids out of the accessible grasp ids let you perform the dissassembly. 
You can refer the instruction manual to figure out the sequence in which you need to dissassemble them. 
The information you collect in this run will be used later to perform assembly.

**Execution**

Use existing tools to perform disassembly one object at a time. 
Use half open gripper state to access all ids of an object in the assembled state.
If it was a SUCCESS, save scene state and log the grasp id using which you were able to dissassemble that object. Move on to the next object.
If it was a FAILURE, restore scene state and try a different grasp id of the same object.

**Verification**

> To verify if a disassembly was successful: run verify dissasembly once you think you've ran all tools required to move the object out of the assembly.
> 1.**SUCCESS**: verify dissassembly returns success.
> 2.**FAILURE**: If the grasp ids are out of reach then verify dissassembly will fail. 

**Output**
Use `write_disassembly_grasp_resource(assembly_id, object_name, grasp_configs)` to log each attempt.
Each config should have: {{"grasp_id": <int>, "result": "SUCCESS"|"FAILURE"}} (no gripper_state)
Resource endpoint: Disassembly{{assembly_id}}/object_name/{{object_name}}/grasp_configs/result/{{result}}"""

@mcp.prompt()
def phase_3_identify_assembly_sequence() -> str:
    """
    Returns a prompt for performing assembly using grasp ids from disassembly log in a simulation environment.
    """
    return """**Initialize:**

You are working in a sim environment
Find the objects available in the scene. Save scene.
Identify grasp ids collected during disassembly which will have to be used per object from the disassembly log to perform this assembly.
Identify the gripper state requried to access these ids per object from the grasp log.

**Task**

Your goal is to perform assembly of these objects using the provided tools. The same tool sequence might not work for all objects. 

**Execution**

Use available tools to perform assembly.
Use only the grasp id from the dissassembly log to perform assembly. Do not try a different id.
When performing assembly onto the base, half-open gripper state as to not disturb the previously placed parts.
If it was a SUCCESS, record the tools and arguments executed in sequence, save scene state and move on to the next object.
If it was a FAILURE, restore scene state and record the tools and arguments executed in sequence that didnt work and retry a different seqeuence you havent tried before.

**Verification**

> To verify if an assembly was successful: run verify assembly once you think you've ran all tools required to move the object into the fixed base.
1.SUCCESS: verify assembly returns success.
2.FAILURE: If the objects need to be regrasped from a different orientation then the assembly might fail.

**Output**
Use `write_assembly_resource(assembly_id, object_name, sequence_id, assembled_into, tools_trials)` to log each trial.
Each trial should have: {{"trial_id": <int>, "grasp_id": <int>, "gripper_state": "open"|"half-open", "tools": [<ordered list of tool names with flags>], "result": "SUCCESS"|"FAILURE"}}
Note: Include flags in tool names if used (e.g., "translate_object --move-to-base", "move_to_grasp --move-to-object")
The assembled_into parameter specifies what object/base all objects are being assembled into (e.g., "base", "part_a", etc.). This is stored once at the top level of the assembly log and should be the same for all objects in the assembly.
Resource endpoint: Assembly{{assembly_id}}/sequence/{{object_name}}/tools_trials/result/{{result}}"""

@mcp.prompt()
def phase_4_perform_assembly_sequence(mode: str = "real") -> str:
    """
    Returns a prompt for performing assembly based on assembly log data.
    
    Args:
        mode: The environment mode - either "sim" or "real" (default: "real")
    """
    environment = "sim environment" if mode == "sim" else "real environment"
    world_context = "simulation" if mode == "sim" else "real world"
    
    return f"""**Initialize:**

You are working in a {environment}.
Read the assembly log to identify the objects you are dealing with.

**Task**

You are to perform assembly of the objects onto a fixed base in the {world_context} based on the data assembly resource collected using a Digital twin from your previous runs.

**Execution**

Use the available tools to perform assembly using the information of the assembly log in the same order. Do not skip any object assembly sequence.
Follow the sequence of objects and the tool calls with arguements to perform assembly one by one.
When performing assembly onto the base, half open gripper as to not disturb the previously placed parts.
Verify assembly after each object assembly.
Move home after each object is assembled.

**Verification**

> To verify if an assembly was successful: run verify assembly once you you've ran all tools required to move one object into the fixed base.
> 1.**SUCCESS**: verify assembly returns success.
> 2.**FAILURE**: verify assembly returns failure. Pause and Request for assistance on further instructions.

**Output**

Fully assembled Assembly in the {world_context}."""


if __name__ == "__main__":
    mcp.run()