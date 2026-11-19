"""move_to_clear_area's execution watchdog (review P2-5): planning time is not execution time, and a timeout
cancels the goal and confirms it stopped instead of exiting while the arm keeps moving."""
import types

import pytest

import tests.ros_stubs as ros_stubs

ros_stubs.install()
from primitives.core import move_to_clear_area as mca  # noqa: E402


class FakeFuture:
    def __init__(self, value=None, completes=True):
        self.value, self.completes, self._done, self.callbacks = value, completes, False, []

    def add_done_callback(self, cb):
        self.callbacks.append(cb)

    def complete(self):
        if self.completes and not self._done:
            self._done = True
            for cb in self.callbacks:
                cb(self)

    def done(self):
        return self._done

    def result(self):
        return self.value


class FakeHandle:
    def __init__(self, final_status, cancel_answers=True):
        self.accepted = True
        self.result_future = FakeFuture(types.SimpleNamespace(status=final_status))
        self.cancel_future = FakeFuture(object(), completes=cancel_answers)
        self.cancel_requested = False

    def get_result_async(self):
        return self.result_future

    def cancel_goal_async(self):
        self.cancel_requested = True
        return self.cancel_future


class Logger:
    def info(self, *a): pass
    warn = error = info


def node():
    n = object.__new__(mca.MoveToClearArea)
    n.trajectory_duration_s, n.goal_sent_at, n.goal_accepted_at = None, None, None
    n._goal_handle = n._send_goal_future = n._get_result_future = None
    n._cancel_on_accept = n._cancelling = False
    n._cancel_result_status = None
    n.get_logger = lambda: Logger()

    def spin_until(future, timeout_s):
        future.complete()
        return future.done()
    n._spin_until = spin_until
    return n


def test_planning_time_does_not_count_against_execution():
    n = node()
    assert n.watchdog_refusal(100.0, 0.0) is None                       # long planning, nothing sent
    assert n.watchdog_refusal(mca.PLANNING_TIMEOUT_S + 1, 0.0).startswith("PLANNING_TIMEOUT")
    n.goal_sent_at, n.trajectory_duration_s = 100.0, 45.0
    assert n.watchdog_refusal(105.0, 0.0) is None
    assert n.watchdog_refusal(111.0, 0.0).startswith("GOAL_ACCEPT_TIMEOUT")
    n.goal_accepted_at = 101.0
    # a 45 s carry accepted after 100 s of planning: the old 30 s whole-run clock had already expired
    assert n.watchdog_refusal(101.0 + 54.0, 0.0) is None
    assert n.watchdog_refusal(101.0 + 56.0, 0.0).startswith("EXECUTION_TIMEOUT")
    n.trajectory_duration_s = 3.0
    assert n.watchdog_refusal(101.0 + 29.0, 0.0) is None                # never shorter than the old 30 s
    assert n.watchdog_refusal(101.0 + 31.0, 0.0).startswith("EXECUTION_TIMEOUT")


def _accepted(n, handle):
    n.goal_sent_at = 1.0
    n.goal_response(FakeFuture(handle))
    assert n._goal_handle is handle and n.goal_accepted_at is not None


def test_timeout_cancels_and_confirms_the_stop():
    n = node()
    handle = FakeHandle(final_status=mca.GOAL_STATUS_CANCELED)
    _accepted(n, handle)
    assert n.cancel_active_goal() is None
    assert handle.cancel_requested and n._cancel_result_status == mca.GOAL_STATUS_CANCELED


@pytest.mark.parametrize("status,cancel_answers", [(2, True), (mca.GOAL_STATUS_CANCELED, False)])
def test_an_unconfirmed_cancel_is_reported_as_such(status, cancel_answers):
    n = node()
    handle = FakeHandle(final_status=status, cancel_answers=cancel_answers)
    _accepted(n, handle)
    why = n.cancel_active_goal()
    assert why.startswith("CANCEL_UNCONFIRMED") and "may be moving" in why


def test_a_goal_accepted_after_the_timeout_is_cancelled():
    n = node()
    handle = FakeHandle(final_status=mca.GOAL_STATUS_CANCELED)
    n.goal_sent_at = 1.0
    n._send_goal_future = FakeFuture(handle)
    n._send_goal_future.add_done_callback(n.goal_response)
    assert n.cancel_active_goal() is None
    assert handle.cancel_requested


def test_nothing_sent_nothing_to_cancel():
    assert node().cancel_active_goal() is None
