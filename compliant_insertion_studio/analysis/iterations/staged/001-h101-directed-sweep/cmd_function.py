"""Replay approximation for H101 — directed sweep replacing Stankowski spiral.

Behaviour at each tick post-contact:
  - Compute c_to_s = seat_xy - tcp_xy (the data-derived operator-direction prior, I003).
  - Apply F_lat = 5 N along c_to_s (operator median 2.0-2.9 N reactive; 5 N command
    produces the right ~1.5 mm displacement under empirical admittance — I004/I005).
  - Hold Fz = -9 N (matches GOLD operator demos, I001).
  - If no Fz_t collapse after `dwell_s` seconds, rotate direction by `rotate_deg`
    and try again, up to `n_directions` total. This converts the directed push
    into a sparse pseudo-spiral over 6 cardinal+diagonal directions (FINDINGS §8 #4).
  - When |F_lat| starts to be reacted (|F_lat_base_N| > 2 N) we hold the direction
    (peg has caught chamfer geometry — keep pushing).
"""
import math

DIRECTED_F_N      = 3.0
DIRECTED_FZ_N     = 9.0      # commanded as -DIRECTED_FZ_N on the Fz axis
DWELL_S           = 1.5      # per-direction dwell before rotating
ROTATE_DEG        = 60.0     # rotate by this much each dwell window
N_DIRECTIONS_MAX  = 6        # 6 × 60° covers full circle
F_LAT_LATCH_N     = 2.0      # if reaction force > this, latch the direction


def cmd_wrench(state):
    tcp_xy = state.get("tcp_xy_m", (0.0, 0.0))
    hole_xy = state.get("hole_xy_m", None)
    t_post = float(state.get("t_since_contact_s", 0.0))
    f_lat = float(state.get("F_lat_base_N", 0.0))
    fz_t = float(state.get("fz_t_N", 0.0))

    # Pre-contact: zero lateral, light Fz (replay only kicks in post-contact anyway)
    if t_post < 0.0:
        return (0.0, 0.0, -DIRECTED_FZ_N, 0.0, 0.0, 0.0)

    # Once Fz_t collapsed (<2N) the peg is in slot — stop lateral push (FINDINGS §5).
    if abs(fz_t) < 2.0 and t_post > 0.5:
        return (0.0, 0.0, -DIRECTED_FZ_N, 0.0, 0.0, 0.0)

    # Direction prior: c_to_s
    if hole_xy is None:
        return (0.0, 0.0, -DIRECTED_FZ_N, 0.0, 0.0, 0.0)
    dx = hole_xy[0] - tcp_xy[0]
    dy = hole_xy[1] - tcp_xy[1]
    dist = math.hypot(dx, dy)
    if dist < 1e-4:
        # Already at seat — fall back to gentle hold (let admittance settle).
        return (0.0, 0.0, -DIRECTED_FZ_N, 0.0, 0.0, 0.0)
    ux, uy = dx / dist, dy / dist

    # Latch: if reaction is building, don't rotate — peg has caught chamfer geometry.
    latched = abs(f_lat) > F_LAT_LATCH_N

    # Rotate direction every DWELL_S until we reach N_DIRECTIONS_MAX or we latch.
    if not latched:
        idx = min(int(t_post / DWELL_S), N_DIRECTIONS_MAX - 1)
        theta = math.radians(idx * ROTATE_DEG)
        c, s = math.cos(theta), math.sin(theta)
        ux, uy = (c * ux - s * uy, s * ux + c * uy)

    # base_link convention: wrapper applies the X/Y sign flip — provide
    # base-frame command here; the cmd_function operates in the same frame
    # the simulator reports F_lat_base_N in (= base_link).
    fx = -DIRECTED_F_N * ux
    fy = -DIRECTED_F_N * uy
    fz = -DIRECTED_FZ_N
    return (fx, fy, fz, 0.0, 0.0, 0.0)
