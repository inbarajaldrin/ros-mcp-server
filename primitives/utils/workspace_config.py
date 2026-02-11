import numpy as np

TABLE_HEIGHT = -0.0825       # Table surface Z in robot base frame (meters)
TABLE_COLLISION_MARGIN = 0.1 # Safety margin above table surface — flag collision when joint Z < TABLE_HEIGHT + this
GRIPPER_CENTER_TOOL_OFFSET = np.array([-0.0, 0.0, 0.23])  # 3D offset from EE flange to gripper center in tool frame (meters)
SAFE_HEIGHT_ABOVE_TABLE = 0.4      # Safe/hover height above table surface (meters)
SAFE_HEIGHT = TABLE_HEIGHT + SAFE_HEIGHT_ABOVE_TABLE  # Absolute safe height Z in robot base frame
ROTATE_ABOUT_GRIPPER_CENTER = False # If True, rotate about gripper center; if False, rotate about flange
