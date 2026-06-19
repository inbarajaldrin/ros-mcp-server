#!/usr/bin/env python3
"""Curated CLI to open/close the sim RG2 (talks to rg2_gripper_driver).

  rg2_gripper.py open          # 100 mm contact (full open)
  rg2_gripper.py close         # 0 mm
  rg2_gripper.py half          # 35 mm
  rg2_gripper.py 55            # 55 mm contact gap

Publishes a std_msgs/Float64 CONTACT width (mm) on /rg2sim/gripper_cmd.
"""
import sys
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64

WORDS = {"open": 100.0, "close": 0.0, "half": 35.0}


def main():
    if len(sys.argv) != 2:
        print(__doc__)
        sys.exit(1)
    a = sys.argv[1].lower()
    contact = WORDS.get(a)
    if contact is None:
        try:
            contact = float(a)
        except ValueError:
            print(f"unknown command: {a}\n{__doc__}")
            sys.exit(1)
    rclpy.init()
    n = Node("rg2_gripper_cli")
    pub = n.create_publisher(Float64, "/rg2sim/gripper_cmd", 10)
    msg = Float64(data=float(contact))
    import time
    # wait for the driver's subscription to be discovered, else a short-lived publisher
    # drops the message before DDS discovery completes (the open-didn't-land bug).
    t = time.time()
    while pub.get_subscription_count() == 0 and time.time() - t < 5.0:
        rclpy.spin_once(n, timeout_sec=0.05)
    if pub.get_subscription_count() == 0:
        print("WARNING: no rg2_gripper_driver subscribed to /rg2sim/gripper_cmd — is it running?")
    for _ in range(5):                          # publish a few times so it lands
        pub.publish(msg)
        rclpy.spin_once(n, timeout_sec=0.0)
        time.sleep(0.05)
    print(f"commanded contact = {contact:.1f} mm")
    n.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
