"""seams.client on Ctrl-C / SIGTERM (review rc1 P2-2): rclpy is initialised WITHOUT its signal handlers, so a SIGINT is
Python's KeyboardInterrupt with the context still alive and _action's branch can send the goal cancel AND wait for
it; SIGTERM takes the same path.  Offline: fake rclpy modules (nothing joins a ROS graph)."""
import signal
import sys
import types

import pytest

from seams import client


def test_rclpy_is_initialised_without_its_signal_handlers_and_sigterm_is_a_keyboard_interrupt(monkeypatch):
    sig = types.ModuleType("rclpy.signals")              # the option enum (the suite may stub rclpy itself)
    sig.SignalHandlerOptions = types.SimpleNamespace(NO="NO", ALL="ALL")
    monkeypatch.setitem(sys.modules, "rclpy.signals", sig)
    seen = {}
    fake = types.SimpleNamespace(init=lambda **kw: seen.update(kw))
    saved = signal.getsignal(signal.SIGTERM)
    try:
        client._init_without_signal_handlers(fake)
        assert seen == {"signal_handler_options": sig.SignalHandlerOptions.NO}
        with pytest.raises(KeyboardInterrupt):
            signal.getsignal(signal.SIGTERM)(signal.SIGTERM, None)
    finally:
        signal.signal(signal.SIGTERM, saved)


def test_a_keyboard_interrupt_during_the_goal_cancels_it_and_waits_for_the_cancel(monkeypatch):
    calls = []

    class Fut:
        def __init__(self, name, value=None):
            self.name, self.value = name, value

        def result(self):
            return self.value

    class GH:
        accepted = True

        def get_result_async(self):
            return Fut("result")

        def cancel_goal_async(self):
            calls.append("cancel sent")
            return Fut("cancel")

    class AC:
        def __init__(self, node, T, name):
            pass

        def wait_for_server(self, timeout_sec):
            return True

        def send_goal_async(self, goal):
            return Fut("send", GH())

    def spin(node, fut, timeout_sec=None):
        calls.append(f"spin {fut.name}")
        if fut.name == "result":
            raise KeyboardInterrupt                      # Ctrl-C while the insert runs
    rclpy = types.ModuleType("rclpy")
    rclpy.spin_until_future_complete = spin
    action = types.ModuleType("rclpy.action")
    action.ActionClient = AC
    monkeypatch.setitem(sys.modules, "rclpy", rclpy)
    monkeypatch.setitem(sys.modules, "rclpy.action", action)
    with pytest.raises(KeyboardInterrupt):
        client._action(object(), object, "/ur5e/drive_mate", object(), 60.0)
    assert calls == ["spin send", "spin result", "cancel sent", "spin cancel"]   # the cancel is sent AND awaited
