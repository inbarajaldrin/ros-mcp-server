"""goal_guard (review r2): a signalled or exiting primitive cancels its running arm goal and confirms the stop."""
import subprocess
import sys
import textwrap
import types
from pathlib import Path

import pytest

from primitives.shared import goal_guard as gg

ROOT = Path(__file__).resolve().parents[1]


class FakeFuture:
    def __init__(self, value=None, completes=True):
        self.value, self.completes, self._done, self.callbacks = value, completes, False, []

    def add_done_callback(self, cb):
        self.callbacks.append(cb)
        if self._done:
            cb(self)

    def complete(self):
        if self.completes and not self._done:
            self._done = True
            for cb in list(self.callbacks):
                cb(self)

    def done(self):
        return self._done

    def result(self):
        return self.value


class FakeHandle:
    def __init__(self, status_after_cancel=gg.GOAL_STATUS_CANCELED, cancel_answers=True):
        self.accepted = True
        self.result_future = FakeFuture(types.SimpleNamespace(status=status_after_cancel))
        self.cancel_future = FakeFuture(object(), completes=cancel_answers)
        self.cancel_requested = False

    def get_result_async(self):
        return self.result_future

    def cancel_goal_async(self):
        self.cancel_requested = True
        return self.cancel_future


def spin(node, future, timeout_s):
    future.complete()
    return future.done()


def accepted_goal(handle):
    send = FakeFuture(handle)
    goal = gg.TrackedGoal("node", send)
    send.complete()
    return goal


def test_a_running_goal_is_cancelled_and_confirmed():
    handle = FakeHandle()
    goal = accepted_goal(handle)
    assert not goal.finished
    assert gg.cancel_goal(goal, spin=spin) is None
    assert handle.cancel_requested and goal.status == gg.GOAL_STATUS_CANCELED


@pytest.mark.parametrize("status,answers", [(2, True), (gg.GOAL_STATUS_CANCELED, False)])
def test_an_unconfirmed_cancel_says_so(status, answers):
    goal = accepted_goal(FakeHandle(status_after_cancel=status, cancel_answers=answers))
    why = gg.cancel_goal(goal, spin=spin)
    assert why.startswith("CANCEL_UNCONFIRMED") and "may be moving" in why


def test_a_goal_accepted_late_is_cancelled_and_a_finished_one_is_left_alone():
    handle = FakeHandle()
    goal = gg.TrackedGoal("node", FakeFuture(handle))            # the response has not come yet
    assert gg.cancel_goal(goal, spin=spin) is None and handle.cancel_requested
    done = FakeHandle()
    goal = accepted_goal(done)
    done.result_future.value = types.SimpleNamespace(status=gg.GOAL_STATUS_SUCCEEDED)
    done.result_future.complete()
    assert goal.finished and gg.cancel_goal(goal, spin=spin) is None and not done.cancel_requested


def test_a_rejected_goal_needs_no_cancel():
    handle = FakeHandle()
    handle.accepted = False
    goal = accepted_goal(handle)
    assert goal.finished and gg.cancel_goal(goal, spin=spin) is None and not handle.cancel_requested


FAKE_ROS = textwrap.dedent('''
    import os, signal, sys, time, types
    sys.path.insert(0, {root!r})
    log = []

    class Future:
        def __init__(self, value=None):
            self.value, self._done, self.cbs = value, False, []
        def add_done_callback(self, cb):
            self.cbs.append(cb)
            if self._done: cb(self)
        def complete(self):
            if not self._done:
                self._done = True
                for cb in list(self.cbs): cb(self)
        def done(self): return self._done
        def result(self): return self.value

    class InvalidHandle(Exception):
        pass

    class Handle:
        accepted = True
        def __init__(self, client): self.client, self.result_future = client, Future(types.SimpleNamespace(status=5))
        def get_result_async(self): return self.result_future
        def cancel_goal_async(self):
            if self.client.destroyed:                     # Humble: a destroyed ActionClient cannot send a cancel
                raise InvalidHandle("action client destroyed")
            log.append("cancel"); print("CANCEL REQUESTED", flush=True); return Future(object())

    class Node:
        def __init__(self, name): self.name = name
        def destroy_node(self): log.append("destroy")

    class ActionClient:
        def __init__(self, node, action_type, name):
            self._node, self._action_type, self.destroyed = node, action_type, False
        def send_goal_async(self, goal, **kw):
            f = Future(Handle(self)); f.complete(); return f
        def destroy(self):
            log.append("client destroy"); self.destroyed = True

    state = {{"ok": True}}
    rclpy = types.ModuleType("rclpy")
    rclpy.init = lambda **kw: log.append("init " + str(kw.get("signal_handler_options")))
    rclpy.ok = lambda: state["ok"]
    def shutdown(*a, **k): log.append("shutdown"); state["ok"] = False
    rclpy.shutdown = shutdown
    rclpy.try_shutdown = shutdown
    def spin_until_future_complete(node, future, timeout_sec=None): future.complete()
    rclpy.spin_until_future_complete = spin_until_future_complete
    action = types.ModuleType("rclpy.action"); action.ActionClient = ActionClient
    node_mod = types.ModuleType("rclpy.node"); node_mod.Node = Node
    signals = types.ModuleType("rclpy.signals"); signals.SignalHandlerOptions = types.SimpleNamespace(NO="NO")
    sys.modules.update({{"rclpy": rclpy, "rclpy.action": action, "rclpy.node": node_mod, "rclpy.signals": signals}})
    FollowJointTrajectory = type("FollowJointTrajectory", (), {{}})
    Grip = type("Grip", (), {{}})

    from primitives.shared.goal_guard import init_ros
    init_ros()
    node = Node("primitive")
''')


def run_script(body):
    code = FAKE_ROS.format(root=str(ROOT)) + textwrap.dedent(body)
    return subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=30)


def test_sigterm_mid_motion_cancels_before_the_node_goes():
    """The replay's outer kill (SIGTERM) on a primitive that catches KeyboardInterrupt and destroys its node."""
    out = run_script('''
        ActionClient(node, FollowJointTrajectory, "arm").send_goal_async(object())
        try:
            os.kill(os.getpid(), signal.SIGTERM)
            time.sleep(5)
            print("NOT INTERRUPTED")
        except KeyboardInterrupt:
            pass
        finally:
            node.destroy_node()
            rclpy.shutdown()
        print("LOG", log)
    ''')
    assert "NOT INTERRUPTED" not in out.stdout
    assert "LOG ['init NO', 'cancel', 'destroy', 'shutdown']" in out.stdout, out.stdout + out.stderr
    assert "cancelled and confirmed stopped" in out.stderr


def test_an_unhandled_sigterm_still_cancels_at_exit():
    out = run_script('''
        ActionClient(node, FollowJointTrajectory, "arm").send_goal_async(object())
        os.kill(os.getpid(), signal.SIGTERM)
        time.sleep(5)
        print("NOT INTERRUPTED")
    ''')
    assert out.returncode != 0 and "NOT INTERRUPTED" not in out.stdout
    assert "CANCEL REQUESTED" in out.stdout, out.stdout + out.stderr      # from goal_guard's exit hook
    assert "cancelled and confirmed stopped" in out.stderr


def test_a_gripper_goal_is_not_cancelled():
    out = run_script('''
        ActionClient(node, Grip, "grip").send_goal_async(object())
        node.destroy_node(); rclpy.shutdown()
        print("LOG", log)
    ''')
    assert "LOG ['init NO', 'destroy', 'shutdown']" in out.stdout, out.stdout + out.stderr


def test_no_guarded_primitive_waits_without_a_timeout():
    """With rclpy's own signal handlers off (goal_guard), nothing wakes an untimed executor wait, so a SIGINT /
    SIGTERM would only be serviced when the wait returns.  Every primitive that starts with init_ros waits in
    timed steps."""
    import re
    offenders = []
    for path in sorted((ROOT / "primitives").rglob("*.py")):
        text = path.read_text()
        if "init_ros(" not in text:
            continue
        for n, line in enumerate(text.splitlines(), 1):
            code = line.split("#")[0]
            if re.search(r"rclpy\.spin\(", code) or (
                    "spin_until_future_complete(" in code and "timeout_sec" not in code):
                offenders.append(f"{path.relative_to(ROOT)}:{n}")
    assert offenders == []


def test_the_cancel_runs_before_the_action_client_is_destroyed():
    """Review r3 P1-1: move_down, rotate_object, translate_object, move_to_grasp, both move_to_safe_heights and
    scan_workspace destroy their action client before their node.  Their exact exit order, on SIGINT mid-motion."""
    out = run_script('''
        client = ActionClient(node, FollowJointTrajectory, "arm")
        client.send_goal_async(object())
        try:
            os.kill(os.getpid(), signal.SIGINT)
            time.sleep(5)
        except KeyboardInterrupt:
            pass                                          # these primitives only log here
        finally:
            client.destroy()
            node.destroy_node()
            rclpy.shutdown()
        print("LOG", log)
    ''')
    assert "LOG ['init NO', 'cancel', 'client destroy', 'destroy', 'shutdown']" in out.stdout, out.stdout + out.stderr
    assert "CANCEL_UNCONFIRMED" not in out.stderr and "cancelled and confirmed stopped" in out.stderr


def test_a_shutdown_from_another_thread_defers_the_cancel_to_the_main_thread():
    """Review r3 P2-1: no executor spin off the main thread; the main thread's own exit path cancels."""
    out = run_script('''
        import threading
        client = ActionClient(node, FollowJointTrajectory, "arm")
        client.send_goal_async(object())
        t = threading.Thread(target=rclpy.shutdown); t.start(); t.join()
        print("AFTER THREAD", log, flush=True)
    ''')
    assert "AFTER THREAD ['init NO', 'shutdown']" in out.stdout, out.stdout + out.stderr
    assert "cancel deferred: not on the main thread" in out.stderr


def test_a_second_signal_during_the_cancel_still_destroys_the_node():
    """Review r3 P3-4: KeyboardInterrupt inside the cancel; the original destroy_node runs, then it is re-raised."""
    out = run_script('''
        client = ActionClient(node, FollowJointTrajectory, "arm")
        client.send_goal_async(object())
        def interrupting(node, future, timeout_sec=None):
            raise KeyboardInterrupt("second signal")
        rclpy.spin_until_future_complete = interrupting
        try:
            node.destroy_node()
        except KeyboardInterrupt:
            print("RE-RAISED", flush=True)
        print("LOG", log, flush=True)
    ''')
    assert "RE-RAISED" in out.stdout and "'destroy'" in out.stdout, out.stdout + out.stderr
