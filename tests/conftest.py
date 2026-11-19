"""Test-suite guards (this repo's tests run on the cell PC, beside the real UR5e).

- The suite never imports a real rclpy or ROS message package: stand-ins go in before any test module loads.
- Autouse, every test: cell_io's dashboard calls are replaced (no `stop`, no `safetystatus` can reach the robot,
  even if a test forgets its own patch), and every network connection that is not loopback or a unix socket is
  refused.  Tests that need a dashboard behaviour patch it themselves on top of this.
"""
import socket

import pytest

import tests.ros_stubs as _ros_stubs

_ros_stubs.install()

ROBOT_BLOCKED = "test suite: no connection to anything but loopback (the real robot is on this network)"


def _is_local(address):
    if isinstance(address, (str, bytes)):                 # AF_UNIX path
        return True
    host = address[0] if isinstance(address, tuple) and address else ""
    return host in ("localhost", "127.0.0.1", "::1", "") or str(host).startswith("127.")


@pytest.fixture(autouse=True)
def no_robot_network(monkeypatch):
    calls = {"dashboard_stop": 0, "robot_safety_status": 0}
    try:
        from ablations import cell_io
    except Exception:                                     # a test environment without the ablations package
        cell_io = None
    if cell_io is not None:
        def dashboard_stop(*args, **kwargs):
            calls["dashboard_stop"] += 1
            return False
        def robot_safety_status(*args, **kwargs):
            calls["robot_safety_status"] += 1
            return "NORMAL"
        monkeypatch.setattr(cell_io, "dashboard_stop", dashboard_stop)
        monkeypatch.setattr(cell_io, "robot_safety_status", robot_safety_status)

    real_connect = socket.socket.connect
    real_connect_ex = socket.socket.connect_ex
    real_create = socket.create_connection

    def connect(self, address):
        if not _is_local(address):
            raise ConnectionRefusedError(f"{ROBOT_BLOCKED}: {address!r}")
        return real_connect(self, address)

    def connect_ex(self, address):
        if not _is_local(address):
            raise ConnectionRefusedError(f"{ROBOT_BLOCKED}: {address!r}")
        return real_connect_ex(self, address)

    def create_connection(address, *args, **kwargs):
        if not _is_local(address):
            raise ConnectionRefusedError(f"{ROBOT_BLOCKED}: {address!r}")
        return real_create(address, *args, **kwargs)

    monkeypatch.setattr(socket.socket, "connect", connect)
    monkeypatch.setattr(socket.socket, "connect_ex", connect_ex)
    monkeypatch.setattr(socket, "create_connection", create_connection)
    yield calls
