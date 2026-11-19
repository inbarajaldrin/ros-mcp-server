#!/usr/bin/env python3
"""Request the twin's exact, non-compliant insert for an already-held part."""

import argparse
import json
import time
import uuid

import rclpy
from rclpy.node import Node
from std_msgs.msg import String


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--object-name", required=True)
    parser.add_argument("--base-name", required=True)
    args = parser.parse_args()
    rclpy.init()
    node = Node("sim_regular_insert")
    publisher = node.create_publisher(String, "/sim_regular_insert_command", 10)
    result = None
    request_id = uuid.uuid4().hex

    def receive(message):
        nonlocal result
        value = json.loads(message.data)
        if value.get("id") == request_id:
            result = value

    node.create_subscription(String, "/sim_regular_insert_result", receive, 10)
    deadline = time.time() + 20
    next_publish = 0.0
    while result is None and time.time() < deadline:
        if time.time() >= next_publish:
            message = String()
            message.data = json.dumps({"id": request_id, "object": args.object_name,
                                       "base": args.base_name})
            publisher.publish(message)
            next_publish = time.time() + 0.25
        rclpy.spin_once(node, timeout_sec=0.05)
    node.destroy_node()
    rclpy.shutdown()
    if result is None:
        result = {"result": "failure", "error": "sim regular insert timed out"}
    else:
        result = {"result": "success" if result.get("ok") else "failure",
                  "object_name": args.object_name, "base_name": args.base_name,
                  "movement_type": "regular_insert", **result}
    print("__RESULT_JSON__")
    print(json.dumps(result))
    print("__END_RESULT_JSON__")
    return 0 if result.get("result") == "success" else 1


if __name__ == "__main__":
    raise SystemExit(main())
