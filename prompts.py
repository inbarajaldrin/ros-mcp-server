"""
Prompts and task descriptions for the ROS MCP Server.
These prompts can be automatically loaded and used by the server.
"""

LEGO_SORTING_TASK = """
Sorting lego blocks using a robotic arm with mobile base into a drop container localized using aruco markers.

Step 1: move camera down

Step 2: read /camera/image_rgb to analyze the image and pick a color of lego block closest to the robot to pick first.

Step 3: add yolo prompt- if your picking and placing a red lego block, then update the prompt as "red object", if your picking and placing a yellow lego block then "yellow object" and so on.

Step 4: read /objects_poses topic to see which objects are available to pick up. select one object 

if there are two or more objects with the same color then pick the closest object (x position) and not based on confidence score.

Step 5: move to lego. (if you used a prompt like "lego object to move to lego, then capture an image again and update the prompt before you mvoe to grasp.")

Step 6: move to grasp object

Step 7: close gripper and read the final output, if the output is close to the closing range of gripper then the grasp failed, reset joints and open gripper and move to step 5 again and continue from there. Keep trying to grasp the object until you succeed.

Step 8: once the grasp is sucessful, reset joints and move camera reset 

Step 9: move linear -0.4 for 0.1 secs

Step 10: control wheels to move to yaw=0 (location of aruco marker drop container)

Step 11: read /aruco_poses to identify ids available for drop

Step 12: Read database to see if any ids have been previously assigned to the object color you have at hand. then add information to database saying you are processing the current lego block of "color" and "id" and update the "count" +1 and set it to processing "status". if the color is not assigned any aruco then based on available aruco ids, assign one id to the color of the lego block you are holding and update database using mqtt.

Step 13: move to aruco

Step 14: move to drop at the aruco id selected

Step 15: open gripper

Step 16: update database to mark the current task as completed

Step 17: reset joints

Step 18: move linear -0.4 for 0.2 secs

Step 19: control wheels to move to yaw=80 (location of lego blocks)

Step 20: Loop back to step 1 and continue the tasks till there are no lego blocks in the scene.
"""

# Dictionary to store all prompts for easy access
PROMPTS = {
    "lego_sorting_task": LEGO_SORTING_TASK,
}

def get_prompt(prompt_name: str) -> str:
    """
    Get a prompt by name.
    
    Args:
        prompt_name: Name of the prompt to retrieve
        
    Returns:
        str: The prompt text, or empty string if not found
    """
    return PROMPTS.get(prompt_name, "")

def list_prompts() -> list:
    """
    Get a list of all available prompt names.
    
    Returns:
        list: List of prompt names
    """
    return list(PROMPTS.keys())

