"""drive-mate subscribes to the TCP pose (500 Hz) only while a goal runs (lead 2026-11-20: idle, the always-on Python
callback cost the server ~31 % CPU on the cell PC, whose load drops the robot program).  Offline, on the gate test's
rig: execute() subscribes when a goal starts, waits for a FRESH sample (a pose left from an earlier goal is never
used), refuses 'no TCP pose' when none comes, and unsubscribes when the goal ends, however it ends."""
from types import SimpleNamespace as NS

from tests.test_drive_mate_gate import _GH, rig  # noqa: F401  (the fixture)


def _subs(srv, deliver=True):
    calls = NS(created=[], destroyed=[])

    def create_subscription(msg, topic, cb, depth, callback_group=None):
        calls.created.append((topic, depth))
        if deliver:
            cb(NS(pose=NS(position=NS(x=0.03, y=-0.45, z=0.2), orientation=NS(x=1.0, y=0.0, z=0.0, w=0.0))))
        return object()
    srv.create_subscription = create_subscription
    srv.destroy_subscription = lambda sub: calls.destroyed.append(sub) or True
    srv._tcp_sub, srv._reads = None, None
    return calls


def test_the_tcp_pose_is_subscribed_only_while_a_goal_runs(rig):
    srv, world = rig
    calls = _subs(srv)
    gh = _GH()
    srv.execute(gh)
    assert calls.created == [("/tcp_pose_broadcaster/pose", 1)] and len(calls.destroyed) == 1
    assert gh.ended == "succeeded" and srv._tcp_sub is None and srv.tcp is None


def test_no_fresh_tcp_sample_refuses_and_a_stale_one_is_never_used(rig):
    srv, world = rig                                   # the rig leaves a pose in srv.tcp: stale for the next goal
    calls = _subs(srv, deliver=False)
    srv.TCP_FIRST_SAMPLE_S = 0.05
    gh = _GH()
    res = srv.execute(gh)
    assert res.refusal == "insert_failed_to_start" and gh.ended == "aborted" and world.starts == []
    assert len(calls.destroyed) == 1 and srv._tcp_sub is None


def test_the_subscription_ends_with_a_latched_goal_too(rig):
    srv, world = rig
    calls = _subs(srv)
    srv.latch.trip("earlier insert left the controllers switched")
    gh = _GH()
    res = srv.execute(gh)
    assert res.refusal.startswith("DRIVE_MATE_LATCHED") and len(calls.destroyed) == len(calls.created) == 1
