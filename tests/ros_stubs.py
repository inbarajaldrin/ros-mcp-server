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


def install():
    """Stand in for every ROS module that is not importable here; a real install is left alone."""
    for name in ROS_MODULES:
        if name in sys.modules:
            continue
        try:
            __import__(name)
        except ImportError:
            sys.modules[name] = _Module(name)
