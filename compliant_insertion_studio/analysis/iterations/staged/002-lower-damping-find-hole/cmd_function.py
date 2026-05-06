"""Replay approximation for staged patch 002 — lower damping during find_hole.

Patch behaviour:
  - Commanded wrench is UNCHANGED. The patch only changes the UR force-mode
    `damping_factor` from 0.30 to 0.18 during the FIND_HOLE state (and
    correction.action.damping_search from 0.50 to 0.30 during Mode B bursts).
  - Under linear admittance, lateral velocity per N of commanded F_lat scales
    roughly as 1/(1 + damping). Going from damping=0.30 to damping=0.18 raises
    velocity per N by approximately (1.30/1.18) ≈ 1.10 if interpreted as a
    direct numerical denominator, or up to ~1.4× if the UR controller's
    internal scaling treats `damping_factor` as an exponential weight.
  - For the replay simulator (which compares predicted-vs-actual TCP xy
    displacement under the same commanded-wrench delta), this patch therefore
    produces an UNCHANGED `cmd_wrench` output. The simulator's admittance gain
    K_LIN_M_PER_NS is what would change at-robot, not the cmd. We document this
    so the replay's "fraction_in_gold_band" reflects only the part of the
    H101-style direction logic, NOT the damping change. This staged patch's
    evidence_score therefore does not count replay band landings as positive
    signal — the score must rely on backing-invariants count.

Practically: for an honest replay we return the EXISTING force-mode commanded
wrench profile during find_hole — i.e. the same (0, +5N Fy, -8N Fz) sustained
push the FSM already issues post-contact. This makes the replay a no-op
relative to the actual May-4 traces (delta_cmd = 0).
"""

# Mirror the FSM's nominal find_hole-state commanded wrench for u_orange/base1
# (per cmd_wrench_raw of insert_u_orange_20260505_193941):
NOMINAL_FX = 0.0
NOMINAL_FY = 5.0     # operator-tuned default; same in patch (cmd unchanged)
NOMINAL_FZ = -8.0    # same in patch


def cmd_wrench(state):
    t_post = float(state.get("t_since_contact_s", 0.0))
    fz_t = float(state.get("fz_t_N", 0.0))

    # Pre-contact: zero lateral, light Fz
    if t_post < 0.0:
        return (0.0, 0.0, -9.0, 0.0, 0.0, 0.0)

    # Once Fz_t collapsed (<2N) the peg is in slot — stop lateral push
    if abs(fz_t) < 2.0 and t_post > 0.5:
        return (0.0, 0.0, -9.0, 0.0, 0.0, 0.0)

    # Find-hole nominal command (unchanged by the patch — the patch lowers
    # damping_factor at the controller level, not the commanded wrench).
    return (NOMINAL_FX, NOMINAL_FY, NOMINAL_FZ, 0.0, 0.0, 0.0)
