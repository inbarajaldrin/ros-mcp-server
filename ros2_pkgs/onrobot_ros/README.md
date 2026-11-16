# 🤖 OnRobot Gripper Control (ROS 2)

This **ROS 2 package** provides a lightweight interface to control an **OnRobot gripper** via a simple ROS 2 node. The node subscribes to a topic and interprets string commands to open or close the gripper.

---

## ✨ Features

- ROS 2 Humble compatible
- Listens to `/gripper_command` topic
- Simple string-based control interface
- **Real-time gripper state feedback** (width, grasp detection, motion status)
- Easy integration with larger robot stacks
- Tested with OnRobot hardware

## 🔧 Test

1. Launch the gripper control node in one terminal:

   ```bash
   ros2 run onrobot_ros gripper_control

2. In a new terminal, source your ROS 2 Humble setup and workspace overlay: `source /opt/ros/humble/setup.bash` `source ~/your_ws/install/setup.bash`

3. Send a command to the gripper:

    To close the gripper:
   ```bash
   ros2 topic pub --once /gripper_command std_msgs/String "{data: 'close'}"
   ```
    To open the gripper:
   ```bash
   ros2 topic pub --once /gripper_command std_msgs/String "{data: 'open'}"
   ```
    To set a particular width in **mm**:
   ```bash
   ros2 topic pub --once /gripper_command std_msgs/String "{data: '200'}"
   ```

## 📊 Gripper State Feedback

The node publishes real-time feedback on several topics:

### **Current Width**
```bash
# Monitor current gripper width in mm
ros2 topic echo /gripper_width std_msgs/Float32
```

### **Grasp Detection** 
```bash
# Monitor if an object is grasped (true/false)
ros2 topic echo /gripper_grasp_detected std_msgs/Bool
```

### **Motion Status**
```bash
# Monitor if gripper is currently moving (true/false)
ros2 topic echo /gripper_motion_ongoing std_msgs/Bool
```

### **Width with Offset**
```bash
# Monitor gripper width including fingertip offset
ros2 topic echo /gripper_width_offset std_msgs/Float32
```

### **Detailed Status**
```bash
# Monitor comprehensive status information
ros2 topic echo /gripper_status std_msgs/String
```

## 🎯 Example: Grasp Confirmation Workflow

1. **Close gripper around object:**
   ```bash
   ros2 topic pub --once /gripper_command std_msgs/String "{data: 'close'}"
   ```

2. **Monitor grasp detection:**
   ```bash
   ros2 topic echo /gripper_grasp_detected std_msgs/Bool
   ```

3. **Check current width to confirm grip:**
   ```bash
   ros2 topic echo /gripper_width std_msgs/Float32
   ```

4. **Wait for motion to complete:**
   ```bash
   ros2 topic echo /gripper_motion_ongoing std_msgs/Bool
   ```
## ⚙️ Configuration

You can configure the status publishing rate using ROS parameters:

```bash
ros2 run onrobot_ros gripper_control --ros-args -p status_publish_rate:=20.0
```

Default rate is 10 Hz.

