import numpy as np

TABLE_HEIGHT = -0.08       # Table surface Z in robot base frame (meters)
TABLE_COLLISION_MARGIN_SIDEWAYS = 0.08  # Safety margin for sideways/horizontal EE orientations
TABLE_COLLISION_MARGIN_FACEDOWN = 0.01  # Safety margin for face_down EE orientation 
GRIPPER_CENTER_TOOL_OFFSET = np.array([0.0, 0.0, 0.2286])  # 3D offset from EE flange to gripper center in tool frame (meters) - REAL
# GRIPPER_CENTER_TOOL_OFFSET = np.array([-0.0005, 0.0025, 0.2286])  # 3D offset from EE flange to gripper center in tool frame (meters)- SIM
SAFE_HEIGHT_ABOVE_TABLE = 0.4      # Safe/hover height above table surface (meters)
SAFE_HEIGHT = TABLE_HEIGHT + SAFE_HEIGHT_ABOVE_TABLE  # Absolute safe height Z in robot base frame
ROTATE_ABOUT_GRIPPER_CENTER = False # If True, rotate about gripper center; if False, rotate about flange
DEFAULT_BASE_ORIENTATION = [0.0, 0.0, 0.0, 1.0]  # Base object orientation [x, y, z, w] quaternion (identity = no rotation)
DEFAULT_BASE_POSITION = [0.0, -0.4, -0.0625]     # Default base object position [x, y, z] in robot base frame (meters)
