"""The suite never imports a real rclpy or ROS message package: stand-ins go in before any test module loads."""
import tests.ros_stubs as _ros_stubs

_ros_stubs.install()
