"""Import a ROS primitive's module without ROS: stand-in modules for rclpy and the message packages.

Only for calling pure planning methods on an instance built without Node.__init__; nothing here joins a ROS
graph (this repo's tests run on the cell PC, beside the real arm)."""
import sys
import types

ROS_MODULES = ["rclpy", "rclpy.node", "rclpy.action", "rclpy.qos", "rclpy.duration", "rclpy.time",
               "rclpy.executors", "rclpy.callback_groups", "tf2_msgs", "tf2_msgs.msg", "geometry_msgs",
               "geometry_msgs.msg", "sensor_msgs", "sensor_msgs.msg", "control_msgs", "control_msgs.action",
               "control_msgs.msg", "trajectory_msgs", "trajectory_msgs.msg", "builtin_interfaces",
               "builtin_interfaces.msg", "std_msgs", "std_msgs.msg", "action_msgs", "action_msgs.msg"]


class _Any:
    def __init__(self, *args, **kwargs):
        pass

    def __getattr__(self, name):
        return _Any()

    def __call__(self, *args, **kwargs):
        return _Any()


class _Module(types.ModuleType):
    def __getattr__(self, name):
        if name.startswith("__"):
            raise AttributeError(name)
        return type(name, (_Any,), {})


class GoalStatus:
    """action_msgs/msg/GoalStatus constants."""
    STATUS_UNKNOWN, STATUS_ACCEPTED, STATUS_EXECUTING, STATUS_CANCELING = 0, 1, 2, 3
    STATUS_SUCCEEDED, STATUS_CANCELED, STATUS_ABORTED = 4, 5, 6


class FollowJointTrajectory(_Any):
    """control_msgs/action/FollowJointTrajectory: a settable Goal and the result codes."""
    class Goal(_Any):
        pass

    class Result:
        SUCCESSFUL, INVALID_GOAL, INVALID_JOINTS = 0, -1, -2
        OLD_HEADER_TIMESTAMP, PATH_TOLERANCE_VIOLATED, GOAL_TOLERANCE_VIOLATED = -3, -4, -5


def install():
    """Stand in for every ROS module, installed or not: the suite never imports a real rclpy (a real one would let a
    test shut a context down or join a graph on the cell PC).  tests/conftest.py installs these before any test
    module is imported."""
    for name in ROS_MODULES:
        if not isinstance(sys.modules.get(name), _Module):
            sys.modules[name] = _Module(name)
    if isinstance(sys.modules["action_msgs.msg"], _Module):
        sys.modules["action_msgs.msg"].GoalStatus = GoalStatus
    if isinstance(sys.modules["control_msgs.action"], _Module):
        sys.modules["control_msgs.action"].FollowJointTrajectory = FollowJointTrajectory
