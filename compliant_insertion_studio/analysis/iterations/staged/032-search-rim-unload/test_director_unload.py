#!/usr/bin/env python3
"""Synthetic check of the SearchDirector rim-stall unload (staged/032).

Kinematics are a toy: in 'pinned' the TCP never moves; in 'slides_when_unloaded' it moves
toward the reference at 4 mm/s only while the commanded press is light; in 'free' it
tracks the reference. Force-direction sign convention is ignored — motion is applied
toward the reference directly, which is what the real command achieves.
"""
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[5]))
from compliant_insertion_studio.wrapper.contact_search_fsm import SearchDirector  # noqa: E402


def run(mode, seconds=40.0, dt=0.01):
    sd = SearchDirector()
    sd.set_center((0.0, 0.0))
    # start 2.04 mm behind the first reference point, like the 2026-10-07 freeze
    x, y = sd.r0 - 0.00204, 0.0
    t, unloads, events = 0.0, 0, []
    was_unloading = False
    while t < seconds:
        fx, fy, fz = sd.update(t, -5.0 if not sd.last_unloading else -1.5, x, y)
        if sd.last_unloading and not was_unloading:
            unloads += 1
            events.append(f"t={t:5.2f}s UNLOAD #{sd._unload_attempts} Fz={fz:+.1f}N "
                          f"v4_suppressed_until={sd.v4_suppressed_until:.2f}")
        if was_unloading and not sd.last_unloading:
            events.append(f"t={t:5.2f}s unload ended (lag={1000*sd.last_e_mag:.2f}mm) Fz={fz:+.1f}N")
        was_unloading = sd.last_unloading
        if sd.abort_reason:
            events.append(f"t={t:5.2f}s ABORT: {sd.abort_reason}")
            break
        px, py = sd.last_p_ref if hasattr(sd, 'last_p_ref') and sd.last_p_ref else (x, y)
        ex, ey = px - x, py - y
        d = math.hypot(ex, ey)
        speed = {'pinned': 0.0,
                 'slides_when_unloaded': 0.004 if fz > -2.0 else 0.0,
                 'free': 0.004}[mode]
        if d > 1e-9 and speed:
            step = min(d, speed * dt)
            x += ex / d * step
            y += ey / d * step
        t += dt
    return unloads, events, sd


for mode in ('pinned', 'slides_when_unloaded', 'free'):
    n, ev, sd = run(mode)
    print(f"== {mode}: unloads={n}, final r={1000*sd.last_radius_m:.2f}mm, aborted={bool(sd.abort_reason)}")
    for e in ev[:8]:
        print("   ", e)
