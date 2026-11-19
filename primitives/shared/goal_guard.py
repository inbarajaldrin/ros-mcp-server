"""The arm never keeps moving after the client that sent it exits (CLAUDE.md; review r2).

Every primitive that sends an arm trajectory starts ROS with init_ros() instead of rclpy.init().  Then:

- SIGINT and SIGTERM raise KeyboardInterrupt in the main thread, so the primitive's own except/finally paths run.
  rclpy's own signal handlers are not installed: they shut the context down on SIGINT, after which nothing can
  cancel.
- Every FollowJointTrajectory goal sent through rclpy's ActionClient is tracked from the send.
- Before a node is destroyed, before rclpy shuts down, and at interpreter exit, every tracked arm goal that is
  still pending or running is cancelled.  The cancel is confirmed from the goal's final status (canceled /
  succeeded / aborted); otherwise a CANCEL_UNCONFIRMED line is written to stderr ("the arm may be moving - stop
  it").  A goal still awaiting its response is cancelled once it is accepted.
- Gripper goals are not touched: an exit must not stop a gripper mid-close.

A goal that finished normally is never cancelled, so the normal exit paths are unchanged.
"""
from __future__ import annotations

import atexit
import signal
import sys

CANCEL_CONFIRM_TIMEOUT_S = 5.0
GOAL_STATUS_SUCCEEDED, GOAL_STATUS_CANCELED, GOAL_STATUS_ABORTED = 4, 5, 6   # action_msgs/GoalStatus
_FINAL = (GOAL_STATUS_SUCCEEDED, GOAL_STATUS_CANCELED, GOAL_STATUS_ABORTED)

_tracked: list["TrackedGoal"] = []
_installed = False


class TrackedGoal:
    """One sent arm goal: its node, send future, handle once accepted, and final status once known."""

    def __init__(self, node, send_future):
        self.node, self.send_future = node, send_future
        self.handle, self.result_future, self.status, self.rejected = None, None, None, False
        send_future.add_done_callback(self._on_response)

    def _on_response(self, future):
        try:
            handle = future.result()
        except Exception:
            self.rejected = True
            return
        if handle is None or not handle.accepted:
            self.rejected = True
            return
        self.handle = handle
        self.result_future = handle.get_result_async()
        self.result_future.add_done_callback(self._on_result)

    def _on_result(self, future):
        try:
            self.status = future.result().status
        except Exception:
            self.status = None

    @property
    def finished(self):
        return self.rejected or self.status in _FINAL


def _spin_until(node, future, timeout_s):
    import rclpy
    if not future.done():
        rclpy.spin_until_future_complete(node, future, timeout_sec=timeout_s)
    return future.done()


def cancel_goal(goal: TrackedGoal, wait_s: float = CANCEL_CONFIRM_TIMEOUT_S, spin=_spin_until) -> str | None:
    """Cancel one tracked goal and confirm it ended.  None when it ended (or never ran), else CANCEL_UNCONFIRMED."""
    if goal.finished:
        return None
    if goal.handle is None:
        if not spin(goal.node, goal.send_future, wait_s):
            return "CANCEL_UNCONFIRMED: the goal response never came; the arm may be moving - stop it"
        if goal.handle is None:
            return None                                        # rejected: nothing runs
    if goal.finished:
        return None
    if not spin(goal.node, goal.handle.cancel_goal_async(), wait_s):
        return "CANCEL_UNCONFIRMED: no cancel response; the arm may be moving - stop it"
    if goal.status is None and goal.result_future is not None:
        spin(goal.node, goal.result_future, wait_s)
    if goal.status in _FINAL:
        return None
    return f"CANCEL_UNCONFIRMED: goal status {goal.status} after the cancel; the arm may be moving - stop it"


def cancel_active_goals(node=None, spin=_spin_until) -> list[str]:
    """Cancel every tracked arm goal still pending or running (of `node`, or all).  Returns the unconfirmed ones."""
    problems = []
    for goal in list(_tracked):
        if node is not None and goal.node is not node:
            continue
        if not goal.finished:
            why = cancel_goal(goal, spin=spin)
            if why:
                problems.append(why)
                print(f"[goal_guard] {why}", file=sys.stderr, flush=True)
            else:
                print("[goal_guard] the arm goal was cancelled and confirmed stopped", file=sys.stderr, flush=True)
        if goal.finished:
            _tracked.remove(goal)
    return problems


def _raise_interrupt(signum, frame):
    raise KeyboardInterrupt(f"signal {signum}")


def _is_arm_goal(client) -> bool:
    action_type = getattr(client, "_action_type", None)
    return getattr(action_type, "__name__", "") == "FollowJointTrajectory"


def install():
    """Patch the ROS entry points once: track arm goals, cancel them before destroy/shutdown/exit."""
    global _installed
    if _installed:
        return
    import rclpy
    from rclpy.action import ActionClient
    from rclpy.node import Node

    original_send = ActionClient.send_goal_async

    def send_goal_async(self, goal, *args, **kwargs):
        future = original_send(self, goal, *args, **kwargs)
        if _is_arm_goal(self):
            _tracked.append(TrackedGoal(self._node, future))
        return future

    ActionClient.send_goal_async = send_goal_async

    original_destroy = Node.destroy_node

    def destroy_node(self, *args, **kwargs):
        try:
            if rclpy.ok():
                cancel_active_goals(self)
        except Exception as error:                                  # never block the destroy
            print(f"[goal_guard] cancel before destroy failed: {error}", file=sys.stderr, flush=True)
        return original_destroy(self, *args, **kwargs)

    Node.destroy_node = destroy_node

    for name in ("shutdown", "try_shutdown"):
        original = getattr(rclpy, name, None)
        if original is None:
            continue

        def wrapped(*args, _original=original, **kwargs):
            try:
                if rclpy.ok():
                    cancel_active_goals()
            except Exception as error:
                print(f"[goal_guard] cancel before shutdown failed: {error}", file=sys.stderr, flush=True)
            return _original(*args, **kwargs)

        setattr(rclpy, name, wrapped)

    def at_exit():
        try:
            if rclpy.ok():
                cancel_active_goals()
            elif any(not g.finished for g in _tracked):
                print("[goal_guard] CANCEL_UNCONFIRMED: exiting with an arm goal running and ROS already shut down; "
                      "the arm may be moving - stop it", file=sys.stderr, flush=True)
        except Exception as error:
            print(f"[goal_guard] cancel at exit failed: {error}", file=sys.stderr, flush=True)

    atexit.register(at_exit)
    _installed = True


def init_ros(args=None, **kwargs):
    """rclpy.init without rclpy's signal handlers; SIGINT/SIGTERM raise KeyboardInterrupt; arm goals guarded."""
    import rclpy
    from rclpy.signals import SignalHandlerOptions
    rclpy.init(args=args, signal_handler_options=SignalHandlerOptions.NO, **kwargs)
    signal.signal(signal.SIGINT, _raise_interrupt)
    signal.signal(signal.SIGTERM, _raise_interrupt)
    install()
