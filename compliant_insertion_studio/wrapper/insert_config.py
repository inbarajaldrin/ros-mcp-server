"""
YAML config loader + termination predicate evaluator.

Phase 5 v0 (2026-05-04). Reads compliant_insertion_studio/configs/<shape>.yaml
which inherits from defaults.yaml via deep-merge. The wrapper's ACTIVE loop
calls TerminationEvaluator.eval() each tick and exits when the predicate fires
sustainedly.
"""
from __future__ import annotations

import math
from collections import deque
from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml


def deep_merge(base: Any, override: Any) -> Any:
    """Recursively merge `override` into `base`. For dict leaves, override wins
    per-key. For non-dict, override replaces base entirely.
    """
    if not isinstance(base, dict) or not isinstance(override, dict):
        return deepcopy(override)
    result = deepcopy(base)
    for k, v in override.items():
        if k in result:
            result[k] = deep_merge(result[k], v)
        else:
            result[k] = deepcopy(v)
    return result


def load_config(path: str | Path) -> dict:
    """Load a YAML config; if it has a top-level `base:` key, recursively load
    that file (resolved relative to this one) and deep-merge.
    """
    p = Path(path).resolve()
    if not p.exists():
        raise FileNotFoundError(f"insert config not found: {p}")
    with open(p) as f:
        cfg = yaml.safe_load(f) or {}
    base_ref = cfg.pop("base", None)
    if base_ref:
        base_path = (p.parent / base_ref).resolve()
        base_cfg = load_config(base_path)
        cfg = deep_merge(base_cfg, cfg)
    return cfg


class CorrectionEvaluator:
    """Phase 5 Mode B — adaptive correction (back-off-on-stuck + exploration).

    Detection: NET z-descent over a sliding window < threshold (independent of
    residual magnitude — lack of progress IS the signal). The previous design's
    instantaneous v_z EMA never fired because force-mode compliance jiggle
    spiked v_z momentarily even during 3-min stuck states.

    Correction action: per-attempt magnitude ADAPTS to the residual + a
    minimum perturbation floor. If residual gives a clear direction (>~1 N or
    >0.05 Nm), use opposite-of-residual; otherwise rotate through cardinal
    directions (+X, -X, +Y, -Y, +Y_torque, -Y_torque, +X_torque, -X_torque) so
    we always perturb the part even when residual is too small to interpret.

    No hard cap on correction count beyond a generous safety ceiling — operator
    feedback was that some attempts need a small nudge, others a big one,
    and a fixed count would be wrong. We log every attempt and ABORT only at
    the safety ceiling.

    Per-tick `update()` returns one of:
      ("none",   None)                       — keep going
      ("apply",  payload)                    — poke mode (v1): one-shot delta wrench
      ("apply_spiral", payload)              — spiral mode (v2): full base-link
                                               wrench setpoint + gain/damping override,
                                               re-issued during burst at command period
      ("revert", None)                       — wrapper reverts to default wrench
                                               (and nominal gain/damping for spiral)
      ("abort",  reason_str)                 — safety ceiling hit

    Spiral mode (v2, 2026-05-04): replaces poke mode for peg-on-rim wedge
    breaking. Burst is [retract_phase_s of unload Fz=+retract_fz_N]
    then [duration_s - retract_phase_s of spiral with rotating lateral
    force F_amp(t) = lerp(start, end, t_within_search/search_duration) at
    freq_hz, axial Fz=-search_fz_N]. Gain/damping drop to gain_search /
    damping_search during the entire burst (UR docs warn high gain unstable
    on hard contact; low damping per Chhatpar 2001 / FANUC for search).
    """

    # Combined F+T exploration directions — applied when residual is too small to
    # give a meaningful direction. Each entry combines a lateral push WITH a
    # counter-torque so each correction perturbs both translation AND rotation,
    # giving the peg multiple ways to escape a wedge in one pulse.
    # Order: first 4 are "natural pairings" (push + rotation that tilts the peg
    # in the same direction the push goes), next 4 are "cross combinations".
    _EXPLORE_PATTERN = [
        # (dfx, dfy, dtx, dty) unit vectors
        ( 1.0,  0.0,  0.0,  1.0),  # +X push + +Ty (roll forward into push)
        (-1.0,  0.0,  0.0, -1.0),  # -X + -Ty
        ( 0.0,  1.0,  1.0,  0.0),  # +Y + +Tx
        ( 0.0, -1.0, -1.0,  0.0),  # -Y + -Tx
        ( 1.0,  0.0,  0.0, -1.0),  # +X + -Ty (cross — push one way, tilt other)
        (-1.0,  0.0,  0.0,  1.0),  # -X + +Ty
        ( 0.0,  1.0, -1.0,  0.0),  # +Y + -Tx
        ( 0.0, -1.0,  1.0,  0.0),  # -Y + +Tx
    ]

    def __init__(self, correction_cfg: dict | None):
        self.cfg = correction_cfg or {}
        self.enabled = bool(self.cfg)
        if not self.enabled:
            return
        self.stuck_cfg   = self.cfg.get("stuck",   {}) or {}
        self.trigger_cfg = self.cfg.get("trigger", {}) or {}
        self.action_cfg  = self.cfg.get("action",  {}) or {}
        self.max_corrections = int(self.cfg.get("max_corrections", 12))

        # Buffer for windowed net-descent computation
        self.window_s = float(self.stuck_cfg.get("descent_window_s", 2.0))
        self.z_buffer: deque = deque()  # of (t_s, tcp_z)
        # Buffer for smoothed fz (anti-jitter) — force-mode dynamics oscillate
        # the instantaneous fz reading, so a single off-tick can be 3-4 N
        # below threshold even though the part is firmly in contact.
        self.fz_window_s = float(self.stuck_cfg.get("fz_smooth_window_s", 0.5))
        self.fz_buffer: deque = deque()

        self.state: str = "NORMAL"
        self.state_until_t: float = 0.0
        self.correction_count: int = 0
        self.contact_t: float | None = None
        self.stuck_first_t: float | None = None
        # Spiral-mode burst state — set on entry to CORRECTING when mode==spiral
        self.correcting_start_t: float | None = None
        self.last_spiral_command_t: float = 0.0
        # Per-burst mode: "spiral" (XY search, when peg is far from target) or
        # "yaw_unlock" (Tz oscillation only, when peg is xy-localized but
        # rotationally misaligned with an asymmetric slot like u_orange's
        # U-shape). Set when entering CORRECTING. Per GPT/Li2021 followup:
        # u_orange is SE(2) — x, y, AND yaw-about-peg-axis must align. XY
        # spiral can localize but can't resolve yaw lock; the operator's
        # manual demos succeeded by twisting the peg, which the algorithm
        # didn't replicate.
        self.burst_mode: str = "spiral"

        # === Hole detection (added 2026-05-04, tuned 2026-05-04 PM) ===
        # During the spiral search, if z descent is sustained (peg has found
        # the chamfer / slot opening, slowly grinding in), record the TCP xy
        # at that moment as `hole_observed_xy` and exit the spiral early.
        # Initial 5 mm/s threshold was way too high — successful breakthroughs
        # have peak descent ~0.1-0.5 mm/s. Lowered to 0.5 mm/s with longer
        # 0.5s averaging window to distinguish sustained descent from
        # spurious spikes from force-mode oscillation.
        self.spike_window_s    = float(self.cfg.get("spike_detect_window_s", 0.5))    # 500ms
        self.spike_threshold_m_s = float(self.cfg.get("spike_threshold_m_s", 0.0005)) # 0.5 mm/s

        # Post-hole-detection grace period (added 2026-05-04 evening):
        # After detecting hole, suppress new Mode B triggers for this long
        # so the peg has time to descend into the slot without the spiral
        # pushing it back out. The ONLY way to exit this grace early is if
        # the predicate fires (success) — otherwise grace expires and we
        # resume normal stuck-detection.
        self.post_hole_grace_s = float(self.cfg.get("post_hole_grace_s", 5.0))
        self.hole_detect_t: float | None = None    # last hole-detection timestamp
        self.hole_observed_xy: tuple[float, float] | None = None
        self.hole_observed_z:  float | None = None
        self.hole_observed_t:  float | None = None
        self.hole_observed_correction: int | None = None
        # Short rolling z-buffer for instantaneous descent rate
        self._spike_z_buffer: deque = deque()

        # Diagnostic-log throttle: emit a status line every ~1 s while in
        # NORMAL post-contact, so the operator can SEE what the evaluator is
        # observing (net descent, F/T residual) — not just silence when nothing
        # fires.
        self._last_diag_t: float = 0.0
        self._diag_period_s: float = 1.0

    def get_diag(self, t_now: float, fz_inst: float, tcp_z: float,
                 F_lat: tuple[float, float], T_lat: tuple[float, float]) -> str | None:
        """Returns a one-line status string if it's time to emit one, else None.
        Wrapper calls this each tick and logs the result. Uses the SAME smoothed
        fz that update() uses, so the diag matches the FSM's view of state.
        """
        if not self.enabled or self.contact_t is None:
            return None
        if (t_now - self._last_diag_t) < self._diag_period_s:
            return None
        self._last_diag_t = t_now
        if len(self.z_buffer) < 10:
            return f"Mode B [post-contact]: warming up (buffer={len(self.z_buffer)})"
        # Use smoothed fz so diag matches FSM view
        if self.fz_buffer:
            fz_smoothed = sum(v for _, v in self.fz_buffer) / len(self.fz_buffer)
        else:
            fz_smoothed = fz_inst
        t0, z0 = self.z_buffer[0]
        rate_mm_s = (z0 - tcp_z) / max(t_now - t0, 0.1) * 1000
        F_mag = (F_lat[0]**2 + F_lat[1]**2) ** 0.5
        T_mag = (T_lat[0]**2 + T_lat[1]**2) ** 0.5
        stuck_thr = float(self.stuck_cfg.get("min_descent_rate_m_s", 0.0001)) * 1000
        in_contact = abs(fz_smoothed) > float(self.stuck_cfg.get("fz_min_N", 6.0))
        no_progress = rate_mm_s < stuck_thr
        sus = (t_now - self.stuck_first_t) if self.stuck_first_t else 0.0
        return (f"Mode B [{self.state}/n={self.correction_count}]: "
                f"fz_smooth={fz_smoothed:+.2f}N (inst={fz_inst:+.2f}) "
                f"net_descent={rate_mm_s:+.3f}mm/s "
                f"(thr<{stuck_thr:.3f}, contact={in_contact}, no_progress={no_progress}, "
                f"stuck_sustain={sus:.1f}/{self.stuck_cfg.get('sustain_s',2.0)}s) "
                f"F_lat={F_mag:.2f}N T_lat={T_mag:.3f}Nm")

    def note_contact(self, t_now: float):
        if self.contact_t is None:
            self.contact_t = t_now

    def _spiral_setpoint(self, t_now: float,
                         tcp_xy: tuple[float, float] | None = None,
                         target_xy: tuple[float, float] | None = None) -> tuple[str, dict]:
        """Compute the time-parameterized force-mode setpoint for spiral mode.

        Phase A (first retract_phase_s): unload via Fz = +retract_fz_N (pull up).
                                         Lateral = 0. Lets static friction relax.
        Phase B (rest of duration_s):   spiral search with bias-plus-rotation.
            - Fz = -search_fz_N (gentle push)
            - Lateral force = bias_force_toward_target + rotating_force
              The bias steers the peg toward `target_xy` (the prior-attempt's
              hole_observed_xy if available, else CAD-derived predicted xy).
              The rotation searches the local neighborhood. Without bias,
              the spiral was purely omnidirectional and could drift AWAY from
              the known hole — observed 2026-05-04 by operator.

        Returns ("apply_spiral", payload). Payload includes:
          - wrench_baselink: (Fx, Fy, Fz, Tx, Ty, Tz)
          - gain, damping
          - phase: "retract" or "search"
          - mode: human-readable label
          - bias_force_N: magnitude of bias (in addition to rotation)
        """
        if self.correcting_start_t is None:
            self.correcting_start_t = t_now
        t_in_burst = max(0.0, t_now - self.correcting_start_t)
        retract_phase_s = float(self.action_cfg.get("retract_phase_s", 0.15))
        retract_fz_N    = float(self.action_cfg.get("retract_fz_N",    2.0))
        search_fz_N     = float(self.action_cfg.get("search_fz_N",     3.0))
        F_lat_start     = float(self.action_cfg.get("search_F_lat_start_N", 1.5))
        F_lat_end       = float(self.action_cfg.get("search_F_lat_end_N",   4.0))
        # v3: rotating-torque amplitudes (alongside force; same theta + freq).
        # Default 0 = legacy force-only spiral.
        T_lat_start     = float(self.action_cfg.get("search_T_lat_start_Nm", 0.0))
        T_lat_end       = float(self.action_cfg.get("search_T_lat_end_Nm",   0.0))
        freq_hz         = float(self.action_cfg.get("search_freq_hz",  1.5))
        gain            = float(self.action_cfg.get("gain_search",     0.5))
        damping         = float(self.action_cfg.get("damping_search",  0.2))
        duration_s      = float(self.action_cfg.get("duration_s",      1.5))

        if t_in_burst < retract_phase_s:
            # Retract / unload phase
            wrench = (0.0, 0.0, retract_fz_N, 0.0, 0.0, 0.0)
            payload = {
                "wrench_baselink": wrench,
                "gain":    gain,
                "damping": damping,
                "phase":   "retract",
                "t_in_burst": t_in_burst,
                "mode":    f"spiral.retract Fz=+{retract_fz_N:.1f}",
            }
            return ("apply_spiral", payload)

        # === Position-spiral search (2026-05-04 PM, from danielstankw/UR5e ref) ===
        # Reference implementation uses Archimedean spiral as POSITION REFERENCE,
        # not rotating force. PD controller drives EE along the expanding spiral.
        # This is a fundamentally different (and better) algorithm than oscillating
        # force — peg has a deterministic motion target each tick, can't drift
        # under noise, and tangential speed is exactly the published 1.5-3 mm/s.
        #
        #   theta_dot = 2π·v / (p·sqrt(1+θ²))    [v=tangential speed, p=ring pitch]
        #   radius = (p/2π)·θ
        #   spiral_dx = radius·cos(θ), spiral_dy = radius·sin(θ)
        #   desired_xy = origin_xy + (spiral_dx, spiral_dy)
        #   Fxy = kp · (desired_xy - current_xy)
        #
        # Critical constraint from Stankowski/Chhatpar: pitch p ≤ 2 × clearance,
        # else spiral can skip past the hole between rings.
        position_spiral = bool(self.action_cfg.get("position_spiral_enabled", True))
        if position_spiral and tcp_xy is not None:
            v_tangential = float(self.action_cfg.get("spiral_tangential_speed_m_s", 0.0015))
            p_pitch = float(self.action_cfg.get("spiral_pitch_m", 0.0006))
            kp_pos = float(self.action_cfg.get("spiral_position_kp_N_per_m", 1500.0))
            F_pos_max = float(self.action_cfg.get("spiral_F_pos_max_N", 8.0))

            # Initialize origin at start of search phase
            if not hasattr(self, "_spiral_origin_xy") or self._spiral_origin_xy is None:
                # If target_xy is available (hole_xy_prior or CAD), use it as origin —
                # spiral expands FROM target, ensuring rings cover hole region.
                # Else use current tcp_xy.
                if target_xy is not None:
                    self._spiral_origin_xy = (float(target_xy[0]), float(target_xy[1]))
                else:
                    self._spiral_origin_xy = (float(tcp_xy[0]), float(tcp_xy[1]))
                self._spiral_theta = 0.0
                self._spiral_last_t = t_now

            # Advance theta by integrated theta_dot
            dt = max(t_now - self._spiral_last_t, 1e-3)
            self._spiral_last_t = t_now
            theta_dot = (2.0 * math.pi * v_tangential) / max(
                p_pitch * math.sqrt(1.0 + self._spiral_theta ** 2), 1e-6)
            self._spiral_theta += theta_dot * dt
            radius = (p_pitch / (2.0 * math.pi)) * self._spiral_theta
            spiral_dx = radius * math.cos(self._spiral_theta)
            spiral_dy = radius * math.sin(self._spiral_theta)

            # Desired EE xy = origin + spiral offset
            desired_x = self._spiral_origin_xy[0] + spiral_dx
            desired_y = self._spiral_origin_xy[1] + spiral_dy

            # PD-derived lateral force (position error * kp).
            # CRITICAL frame note (2026-05-04 PM): tcp_xy comes from
            # /tcp_pose_broadcaster/pose which publishes in `base` frame, but
            # _start_force_mode's `override_wrench_baselink` expects the
            # BASE_LINK convention (wrapper applies its own X/Y sign flip
            # to send to URScript through `base` frame). base and base_link
            # are 180° about Z, so we must NEGATE err to express the desired
            # force in base_link convention. Without this negation, the
            # wrapper's sign flip double-inverts and the PD pushes peg
            # AWAY from target — observed empirically as sustained +2.3 mm/s
            # drift away from truth in v55.
            err_x_base = desired_x - tcp_xy[0]
            err_y_base = desired_y - tcp_xy[1]
            err_x_baselink = -err_x_base
            err_y_baselink = -err_y_base
            Fx = kp_pos * err_x_baselink
            Fy = kp_pos * err_y_baselink
            # Cap magnitude to avoid runaway commands when err is large
            F_mag = math.hypot(Fx, Fy)
            if F_mag > F_pos_max:
                Fx = Fx * F_pos_max / F_mag
                Fy = Fy * F_pos_max / F_mag

            Fz = -search_fz_N
            # Tilt rocking: Tx/Ty oscillation matching operator demo pattern.
            # Pure circular spiral works for round pegs; U-shaped pegs need
            # rocking to navigate asymmetric chamfer entry. Operator demos
            # show tilt torque (Tx/Ty) is the primary alignment action,
            # not yaw or pure xy push.
            t_in_search_pos = max(0.0, t_in_burst - retract_phase_s)
            T_lat_search = float(self.action_cfg.get("tilt_rock_T_amp_Nm", 0.30))
            tilt_freq_hz = float(self.action_cfg.get("tilt_rock_freq_hz", 1.5))
            tilt_theta = 2.0 * math.pi * tilt_freq_hz * t_in_search_pos
            Tx = T_lat_search * math.cos(tilt_theta)
            Ty = T_lat_search * math.sin(tilt_theta)
            wrench = (Fx, Fy, Fz, Tx, Ty, 0.0)
            payload = {
                "wrench_baselink": wrench,
                "gain":    gain,
                "damping": damping,
                "phase":   "search_position",
                "t_in_burst": t_in_burst,
                "mode":    (f"position_spiral r={radius*1000:.2f}mm θ={math.degrees(self._spiral_theta) % 360:.0f}° "
                            f"err=({err_x_base*1000:+.1f},{err_y_base*1000:+.1f})mm F=({Fx:+.1f},{Fy:+.1f})N "
                            f"T=({Tx:+.2f},{Ty:+.2f})Nm"),
                "spiral_target_xy_m": (desired_x, desired_y),
                "spiral_radius_m": radius,
                "spiral_theta_rad": self._spiral_theta,
            }
            return ("apply_spiral", payload)

        # Search phase: bias toward target_xy + rotating lateral force/torque.
        # Bias is a constant lateral force pointing from current TCP xy to
        # target_xy (typically hole_xy_prior from a successful detection in a
        # previous attempt). Rotating component still searches the local
        # neighborhood. Without the bias, the spiral was purely omnidirectional
        # and could drift away from the known hole.
        t_in_search = t_in_burst - retract_phase_s
        search_dur  = max(1e-3, duration_s - retract_phase_s)
        progress    = min(1.0, t_in_search / search_dur)
        F_amp       = F_lat_start + progress * (F_lat_end - F_lat_start)
        T_amp       = T_lat_start + progress * (T_lat_end - T_lat_start)
        # Phase-align the spiral start direction with the target offset so
        # that the FIRST force vector kicks the peg TOWARD target (chamfer)
        # before rotating. Without alignment, theta=0 → Fx=+F_amp,Fy=0 always
        # — peg gets initial push in +x regardless of where the target is. If
        # target is in -x direction, peg gets shoved AWAY first; by the time
        # spiral rotates around to -x phase, peg has drifted out of reach.
        # Aligning so cos(theta_0)=bias_dir.x, sin(theta_0)=bias_dir.y means
        # the spiral STARTS with force toward target, then rotates around it.
        # 2026-05-04 fix for the U-shape asymmetric chamfer issue: contact xy
        # randomness puts peg on either side of the U; first-kick direction
        # determines whether peg slides INTO chamfer or away from it.
        theta_0 = 0.0
        if tcp_xy is not None and target_xy is not None:
            dx0 = target_xy[0] - tcp_xy[0]
            dy0 = target_xy[1] - tcp_xy[1]
            offset0 = math.hypot(dx0, dy0)
            if offset0 > 1e-4:
                theta_0 = math.atan2(dy0, dx0)
        theta       = theta_0 + 2.0 * math.pi * freq_hz * t_in_search
        cos_th      = math.cos(theta)
        sin_th      = math.sin(theta)

        # Bias toward target_xy — gentle, no deadband (2026-05-04 PM v2).
        # Removing bias entirely (deadband-only) was wrong — peg landed at
        # right xy but had no force pushing it INTO the chamfer center, just
        # parked on rim. Restoring bias but at HALF the original kp (200 vs
        # 400 N/m) so it's a gentle nudge, not a strong yank that overshoots.
        # At 5mm offset → 1N bias (within ~1/4 of search F_lat_end).
        bias_force_N = 0.0
        bias_dir = (0.0, 0.0)
        bias_xy_label = ""
        if tcp_xy is not None and target_xy is not None:
            dx = target_xy[0] - tcp_xy[0]
            dy = target_xy[1] - tcp_xy[1]
            offset_m = math.hypot(dx, dy)
            if offset_m > 1e-4:
                bias_kp = float(self.action_cfg.get(
                    "search_bias_kp_N_per_m", 200.0))
                bias_max = float(self.action_cfg.get(
                    "search_bias_max_N", F_lat_end))
                bias_force_N = min(bias_kp * offset_m, bias_max)
                if bias_force_N > 0:
                    bias_dir = (dx / offset_m, dy / offset_m)
                    bias_xy_label = (f" bias=({bias_force_N:.1f}N→"
                                     f"({dx*1000:+.1f},{dy*1000:+.1f})mm)")

        Fx = bias_force_N * bias_dir[0] + F_amp * cos_th
        Fy = bias_force_N * bias_dir[1] + F_amp * sin_th
        Fz = -search_fz_N
        Tx = T_amp * cos_th
        Ty = T_amp * sin_th
        wrench = (Fx, Fy, Fz, Tx, Ty, 0.0)
        payload = {
            "wrench_baselink": wrench,
            "gain":    gain,
            "damping": damping,
            "phase":   "search",
            "t_in_burst": t_in_burst,
            "bias_force_N": float(bias_force_N),
            "mode":     (f"spiral.search F={F_amp:.1f}N T={T_amp:.2f}Nm "
                         f"theta={math.degrees(theta) % 360:.0f}deg{bias_xy_label}"),
        }
        return ("apply_spiral", payload)

    def _yaw_unlock_setpoint(self, t_now: float) -> tuple[str, dict]:
        """YAW_UNLOCK action: Tz oscillation about peg axis to break rotational
        wedges in asymmetric (U/key/square) peg-in-hole. Pauses XY spiral so
        the peg stays at its localized xy position while yaw varies.

        Per GPT/Li 2021 followup (2026-05-04 PM): when peg is xy-localized but
        not descending, this is SE(2) yaw lock — peg-tip is in chamfer but
        the U-arms hit slot walls. Need to "wiggle peg about its own axis"
        until yaw matches slot opening. Recommended: ±0.15-0.30 Nm initial,
        1-2 Hz reciprocating, with light Fz held to keep contact.

        Tz is commanded in base_link frame. For face-down EE the peg's axial
        direction (= -world z) and base_link z are anti-parallel, so Tz_baselink
        positive = peg yaws clockwise looking from above. Sign doesn't matter
        for an oscillating scan since both ± yaw are sampled.
        """
        if self.correcting_start_t is None:
            self.correcting_start_t = t_now
        t_in_burst = max(0.0, t_now - self.correcting_start_t)
        yu = self.action_cfg.get("yaw_unlock", {}) or {}
        T_amp     = float(yu.get("T_amp_Nm",   0.30))
        freq_hz   = float(yu.get("freq_hz",    1.5))
        fz_N      = float(yu.get("fz_N",       3.0))
        gain      = float(yu.get("gain",       0.5))
        damping   = float(yu.get("damping",    0.5))
        # Sinusoidal Tz at freq_hz; zero F_xy so peg doesn't drift away from
        # the localized xy position while yaw is being scanned.
        Tz = T_amp * math.sin(2.0 * math.pi * freq_hz * t_in_burst)
        wrench = (0.0, 0.0, -fz_N, 0.0, 0.0, Tz)
        payload = {
            "wrench_baselink": wrench,
            "gain":     gain,
            "damping":  damping,
            "phase":    "yaw_unlock",
            "t_in_burst": t_in_burst,
            "mode":     f"yaw_unlock Tz={Tz:+.2f}Nm Fz={-fz_N:.1f}N",
        }
        return ("apply_spiral", payload)

    def update(self, t_now: float, fz_inst: float, tcp_z: float,
               F_lat_baselink: tuple[float, float],
               T_lat_baselink: tuple[float, float],
               tcp_xy: tuple[float, float] | None = None,
               target_xy: tuple[float, float] | None = None,
               at_seat: bool = False) -> tuple[str, object]:
        """`at_seat` (added 2026-05-04): the wrapper passes True when the
        termination predicate's `tcp_z_reached_predicted` is satisfied
        (|tcp_z - predicted_seat_z| ≤ 5mm). When True, Mode B suppresses
        new corrections — the peg is at the seat depth and any further spiral
        would just disturb it. The predicate's other conditions (motion_stopped
        + descended_post_contact) will then naturally satisfy as the peg
        settles, firing the predicate → DONE.
        """
        if not self.enabled:
            return ("none", None)

        # --- Buffer z for net-descent computation (always, even outside NORMAL) ---
        self.z_buffer.append((t_now, tcp_z))
        while self.z_buffer and (t_now - self.z_buffer[0][0]) > self.window_s:
            self.z_buffer.popleft()

        # --- Buffer + smooth fz (anti-jitter) ---
        self.fz_buffer.append((t_now, fz_inst))
        while self.fz_buffer and (t_now - self.fz_buffer[0][0]) > self.fz_window_s:
            self.fz_buffer.popleft()
        # Mean fz over the smoothing window — rejects single-tick dips from
        # force-mode oscillation that would otherwise flip the contact flag and
        # reset the stuck-sustain timer.
        if self.fz_buffer:
            fz_smoothed = sum(v for _, v in self.fz_buffer) / len(self.fz_buffer)
        else:
            fz_smoothed = fz_inst

        # --- COOLDOWN ---
        if self.state == "COOLDOWN":
            if t_now >= self.state_until_t:
                self.state = "NORMAL"
                self.stuck_first_t = None
            return ("none", None)

        # --- CORRECTING ---
        action_mode = str(self.action_cfg.get("mode", "poke")).lower()

        if self.state == "CORRECTING":
            if t_now >= self.state_until_t:
                cooldown_s = float(self.action_cfg.get("cooldown_s", 1.0))
                self.state = "COOLDOWN"
                self.state_until_t = t_now + cooldown_s
                self.correcting_start_t = None
                return ("revert", None)

            # === Mid-burst convergence check (added 2026-05-04 PM) ===
            # Track tcp_xy distance to target through the burst. If peg is
            # CLOSER to target than at burst start AND has started moving
            # AWAY from minimum (overshoot detected), exit burst NOW so the
            # next NORMAL phase + Fz=-9N can drive descent without spiral
            # drag pushing peg past chamfer.
            #
            # Triggers operator's question: are we using drift-toward-vs-away
            # to validate corrections? Without this exit, peg can converge
            # to chamfer mid-burst then overshoot 5+ mm before burst ends.
            converge_min_thresh_m = 0.001  # require ≥1mm closer than start to trigger
            converge_overshoot_m = 0.0005  # 0.5mm past minimum = overshoot detected
            if (tcp_xy is not None and target_xy is not None
                    and self.correcting_start_t is not None):
                dist_now = math.hypot(target_xy[0]-tcp_xy[0],
                                      target_xy[1]-tcp_xy[1])
                if not hasattr(self, '_burst_dist_start') or self._burst_dist_start is None:
                    self._burst_dist_start = dist_now
                    self._burst_dist_min = dist_now
                else:
                    if dist_now < self._burst_dist_min:
                        self._burst_dist_min = dist_now
                    converged = (self._burst_dist_start - self._burst_dist_min) > converge_min_thresh_m
                    overshooting = (dist_now - self._burst_dist_min) > converge_overshoot_m
                    if converged and overshooting:
                        # Peg passed through closest-approach to target. Exit
                        # spiral so Fz=-9N can pull it down before drift continues.
                        cooldown_s = float(self.action_cfg.get("cooldown_s", 1.0))
                        self.state = "COOLDOWN"
                        self.state_until_t = t_now + cooldown_s
                        self.correcting_start_t = None
                        d_start = self._burst_dist_start * 1000
                        d_min   = self._burst_dist_min * 1000
                        d_now   = dist_now * 1000
                        self._burst_dist_start = None
                        self._burst_dist_min = None
                        return ("revert", {
                            "reason": "xy_converged_overshoot",
                            "dist_start_mm": d_start,
                            "dist_min_mm":   d_min,
                            "dist_now_mm":   d_now,
                            "correction":    self.correction_count,
                        })

            # YAW_UNLOCK burst: re-issue Tz oscillation setpoint each tick.
            # No spike detection (no hole search; we're already at hole).
            if self.burst_mode == "yaw_unlock" and self.correcting_start_t is not None:
                cmd_period = float(self.action_cfg.get("spiral_command_period_s", 0.05))
                if (t_now - self.last_spiral_command_t) >= cmd_period:
                    self.last_spiral_command_t = t_now
                    return self._yaw_unlock_setpoint(t_now)
                return ("none", None)

            # Spiral mode: re-issue full wrench setpoint at the spiral command
            # period (typ. 20 Hz). Other modes: stay quiescent during CORRECTING.
            if action_mode == "spiral" and self.correcting_start_t is not None:
                # === Hole-detection check (per-tick during spiral) ===
                # Track short-window z-descent rate. If it spikes above
                # spike_threshold (default 5 mm/s), the peg has found the
                # chamfer / hole opening — we're inside the slot edge and
                # about to drop in. Record the location and exit the spiral
                # immediately so the peg can settle without further lateral
                # disruption from the rotating force.
                self._spike_z_buffer.append((t_now, tcp_z))
                while (self._spike_z_buffer
                       and (t_now - self._spike_z_buffer[0][0]) > self.spike_window_s):
                    self._spike_z_buffer.popleft()
                if (len(self._spike_z_buffer) >= 3
                        and self.hole_observed_xy is None
                        and tcp_xy is not None):
                    t_old, z_old = self._spike_z_buffer[0]
                    dt = t_now - t_old
                    if dt > 0:
                        rate = (z_old - tcp_z) / dt   # +ve = descending
                        if rate >= self.spike_threshold_m_s:
                            self.hole_observed_xy = (float(tcp_xy[0]), float(tcp_xy[1]))
                            self.hole_observed_z  = float(tcp_z)
                            self.hole_observed_t  = float(t_now)
                            self.hole_observed_correction = int(self.correction_count)
                            self.hole_detect_t = float(t_now)   # start grace period
                            # Force-exit spiral: jump to COOLDOWN immediately.
                            cooldown_s = float(self.action_cfg.get("cooldown_s", 1.0))
                            self.state = "COOLDOWN"
                            self.state_until_t = t_now + cooldown_s
                            self.correcting_start_t = None
                            return ("revert", {
                                "reason": "hole_detected",
                                "hole_xy": self.hole_observed_xy,
                                "hole_z":  self.hole_observed_z,
                                "rate_mm_s": rate * 1000.0,
                                "correction": self.correction_count,
                            })
                cmd_period = float(self.action_cfg.get("spiral_command_period_s", 0.05))
                if (t_now - self.last_spiral_command_t) >= cmd_period:
                    self.last_spiral_command_t = t_now
                    return self._spiral_setpoint(t_now, tcp_xy=tcp_xy, target_xy=target_xy)
            return ("none", None)

        # --- NORMAL: detect stuck via NET z descent over window, not v_z EMA ---
        warmup_s = float(self.stuck_cfg.get("warmup_after_contact_s", 0.5))
        if self.contact_t is None or (t_now - self.contact_t) < warmup_s:
            return ("none", None)

        # === At-seat gate (added 2026-05-04) ===
        # If the wrapper says we're at the predicted seat z (within 5mm),
        # suppress new Mode B triggers. The peg is seated; any further spiral
        # would disturb it. Predicate's motion_stopped + descended_post_contact
        # will satisfy on their own → DONE.
        if at_seat:
            self.stuck_first_t = None  # reset stuck timer (we're not stuck, we're done)
            return ("none", None)

        # === Post-hole-detection grace period (added 2026-05-04 PM) ===
        # After hole detected, give peg time to settle into slot without
        # spiral disrupting. Spiral was pushing peg back OUT of chamfer
        # immediately after revert. Grace lets default wrench (-Fz=9N)
        # drive descent uninterrupted.
        if self.hole_detect_t is not None:
            t_since_hole = t_now - self.hole_detect_t
            if t_since_hole < self.post_hole_grace_s:
                self.stuck_first_t = None  # don't accumulate stuck time during grace
                return ("none", None)
            # else: grace expired; resume normal stuck detection

        # Need enough samples to compute net descent (≥ window_s of buffer)
        if len(self.z_buffer) < 10 or (t_now - self.z_buffer[0][0]) < self.window_s * 0.5:
            return ("none", None)

        t0, z0 = self.z_buffer[0]
        net_descent_rate_m_s = (z0 - tcp_z) / max(t_now - t0, 0.1)  # +ve = descending

        fz_min = float(self.stuck_cfg.get("fz_min_N", 6.0))
        descent_min_rate = float(self.stuck_cfg.get("min_descent_rate_m_s", 0.0001))  # 0.1 mm/s
        sustain_s = float(self.stuck_cfg.get("sustain_s", 2.0))

        in_contact = abs(fz_smoothed) > fz_min
        no_progress = net_descent_rate_m_s < descent_min_rate
        is_stuck = in_contact and no_progress

        if not is_stuck:
            self.stuck_first_t = None
            return ("none", None)

        if self.stuck_first_t is None:
            self.stuck_first_t = t_now
        if (t_now - self.stuck_first_t) < sustain_s:
            return ("none", None)

        if self.correction_count >= self.max_corrections:
            return ("abort", f"safety ceiling hit ({self.max_corrections} corrections)")

        # --- Branch: spiral mode (v2) vs poke mode (v1 legacy) ---
        if action_mode == "spiral":
            # Decide burst mode: YAW_UNLOCK if peg is xy-localized
            # (within yaw_unlock_xy_threshold of target), else XY spiral.
            # Per GPT/Li2021 SE(2) analysis: when xy is good but z stuck,
            # the failure is rotational misalignment, not xy.
            yu = self.action_cfg.get("yaw_unlock", {}) or {}
            yu_enabled = bool(yu.get("enabled", False))
            yu_xy_thresh_m = float(yu.get("xy_threshold_mm", 3.0)) / 1000.0
            yu_duration_s = float(yu.get("duration_s",
                                          self.action_cfg.get("duration_s", 1.5)))
            xy_offset_m = None
            if tcp_xy is not None and target_xy is not None:
                xy_offset_m = math.hypot(target_xy[0]-tcp_xy[0],
                                         target_xy[1]-tcp_xy[1])
            if (yu_enabled and xy_offset_m is not None
                    and xy_offset_m <= yu_xy_thresh_m):
                self.burst_mode = "yaw_unlock"
                duration_s = yu_duration_s
            else:
                self.burst_mode = "spiral"
                duration_s = float(self.action_cfg.get("duration_s", 1.5))

            self.state = "CORRECTING"
            self.state_until_t = t_now + duration_s
            self.correcting_start_t = t_now
            self.last_spiral_command_t = 0.0   # force first setpoint
            self.correction_count += 1
            self.stuck_first_t = None
            # Reset mid-burst convergence tracking (start fresh each burst)
            self._burst_dist_start = None
            self._burst_dist_min = None
            # Reset position-spiral state (fresh spiral from origin each burst)
            self._spiral_origin_xy = None
            self._spiral_theta = 0.0
            self._spiral_last_t = t_now
            # Emit first setpoint immediately (will be retract phase for spiral,
            # or first Tz tick for yaw_unlock).
            if self.burst_mode == "yaw_unlock":
                payload = self._yaw_unlock_setpoint(t_now)[1]
            else:
                payload = self._spiral_setpoint(t_now, tcp_xy=tcp_xy, target_xy=target_xy)[1]
            payload["n"] = self.correction_count
            payload["burst_mode"] = self.burst_mode
            payload["xy_offset_mm"] = (xy_offset_m * 1000) if xy_offset_m is not None else None
            payload["residual_F_N"]  = (F_lat_baselink[0]**2 + F_lat_baselink[1]**2) ** 0.5
            payload["residual_T_Nm"] = (T_lat_baselink[0]**2 + T_lat_baselink[1]**2) ** 0.5
            payload["net_descent_rate_mm_s"] = net_descent_rate_m_s * 1000
            return ("apply_spiral", payload)

        # --- Compute correction wrench delta (v1 legacy poke mode) ---
        F_gain     = float(self.action_cfg.get("F_gain", 0.5))
        T_gain     = float(self.action_cfg.get("T_gain", 1.0))
        max_dF     = float(self.action_cfg.get("max_delta_F_N",  6.0))
        max_dT     = float(self.action_cfg.get("max_delta_T_Nm", 0.5))
        min_perturb_N  = float(self.action_cfg.get("min_perturb_N",  3.0))
        min_perturb_Nm = float(self.action_cfg.get("min_perturb_Nm", 0.2))
        # "Direction is meaningful" thresholds — how big does residual need to be
        # to USE it as a direction-finder vs fall back to exploration cycling
        F_dir_floor = float(self.action_cfg.get("F_direction_floor_N", 1.0))
        T_dir_floor = float(self.action_cfg.get("T_direction_floor_Nm", 0.05))

        F_mag = (F_lat_baselink[0] ** 2 + F_lat_baselink[1] ** 2) ** 0.5
        T_mag = (T_lat_baselink[0] ** 2 + T_lat_baselink[1] ** 2) ** 0.5
        clamp = lambda v, lo, hi: max(lo, min(hi, v))

        # HYBRID action — every correction always perturbs F + T together.
        #
        # Iteration 4 (2026-05-04) — TARGET-DIRECTED instead of counter-residual.
        # Iter 3 analysis showed counter-residual pushed AWAY from target in
        # 58% of corrections — when peg is wedged at the (-X,-Y) corner of the
        # rim, wrist-sensor reads +X+Y (force from gripper holding part against
        # rim push), and "opposite-residual" = -X-Y = away from target. Wrong.
        # The CORRECT direction is TOWARD the target hole center: derive from
        # (target_xy - tcp_xy). Residual still tells us there IS a wall to push
        # past (magnitude), but the CAD-derived target gives the right direction.
        idx = self.correction_count % len(self._EXPLORE_PATTERN)
        explore_pat = self._EXPLORE_PATTERN[idx]

        # Direction toward target (if we have CAD info)
        target_dir = None
        if tcp_xy is not None and target_xy is not None:
            dx = target_xy[0] - tcp_xy[0]
            dy = target_xy[1] - tcp_xy[1]
            mag = (dx*dx + dy*dy) ** 0.5
            if mag > 1e-4:  # > 0.1 mm — meaningful offset
                target_dir = (dx/mag, dy/mag)

        if target_dir is not None:
            # Push toward the hole, magnitude scaled by residual or floor.
            scale_F = max(min_perturb_N, F_gain * F_mag) if F_mag > F_dir_floor else min_perturb_N
            ux, uy = target_dir
            f_mode = "toward_target"
        elif F_mag > F_dir_floor:
            # Fallback: counter-residual (no CAD target available)
            scale_F = max(min_perturb_N, F_gain * F_mag)
            ux = -F_lat_baselink[0] / F_mag
            uy = -F_lat_baselink[1] / F_mag
            f_mode = "counter_F"
        else:
            scale_F = min_perturb_N
            ux, uy = explore_pat[0], explore_pat[1]
            f_mode = "explore_F"
        dfx = clamp(scale_F * ux, -max_dF, max_dF)
        dfy = clamp(scale_F * uy, -max_dF, max_dF)

        if T_mag > T_dir_floor:
            scale_T = max(min_perturb_Nm, T_gain * T_mag)
            utx = -T_lat_baselink[0] / T_mag
            uty = -T_lat_baselink[1] / T_mag
            t_mode = "counter_T"
        else:
            scale_T = min_perturb_Nm
            utx, uty = explore_pat[2], explore_pat[3]
            t_mode = "explore_T"
        dtx = clamp(scale_T * utx, -max_dT, max_dT)
        dty = clamp(scale_T * uty, -max_dT, max_dT)

        mode = f"{f_mode}+{t_mode}"

        duration_s = float(self.action_cfg.get("duration_s", 0.4))
        self.state = "CORRECTING"
        self.state_until_t = t_now + duration_s
        self.correction_count += 1
        self.stuck_first_t = None
        return ("apply", {
            "delta":   (dfx, dfy, dtx, dty),
            "mode":    mode,
            "residual_F_N":  F_mag,
            "residual_T_Nm": T_mag,
            "net_descent_rate_mm_s": net_descent_rate_m_s * 1000,
            "n":       self.correction_count,
        })


def resolve_config_for_object(object_name: str, configs_dir: str | Path | None = None) -> Path | None:
    """Phase 5 v1: returns the universal `defaults.yaml`. Per-shape YAMLs were
    deleted — the termination predicate is now shape-agnostic via the CAD chain
    (predicted_tcp_at_seat). If a per-object override file `configs/<obj>.yaml`
    is present (e.g. for a future part needing custom force-mode params), it
    takes precedence.
    """
    if configs_dir is None:
        configs_dir = Path(__file__).resolve().parent.parent / "configs"
    configs_dir = Path(configs_dir)
    direct = configs_dir / f"{object_name}.yaml"
    if direct.exists():
        return direct
    universal = configs_dir / "defaults.yaml"
    return universal if universal.exists() else None


# ---------------------------------------------------------------------------
# Termination predicate evaluator
# ---------------------------------------------------------------------------

class TerminationEvaluator:
    """Stateful predicate evaluator — Phase 5 v1.

    Call eval() once per ACTIVE-loop tick. Returns (fired: bool, debug: dict).
    fired=True means the wrapper should exit the ACTIVE loop into DONE.

    Supported predicates (combined with AND/OR per `combinator`):
      - motion_stopped: v_lat ≤ v_lat_max_m_s AND v_z ≤ v_z_max_m_s (when set)
      - tcp_z_reached_predicted: |tcp_z − predicted_tcp_z_at_seat| ≤ tolerance_m
                                 (universal v1, requires CAD-derived prediction)
      - descended_post_contact: tcp descended ≥ min_m past first-contact z
                                (universal v0, no CAD needed)
      - z_reached (legacy v0.5): tcp descended ≥ descended_min_m past hover_z
                                (per-shape hardcoded — kept for backward compat)
    """

    def __init__(self, term_cfg: dict, hover_z_m: float,
                 predicted_tcp_z_at_seat: float | None = None):
        self.cfg = term_cfg or {}
        self.predicates_cfg = self.cfg.get("predicates", {}) or {}
        self.combinator = (self.cfg.get("combinator") or "AND").upper()
        self.sustain_s = float(self.cfg.get("sustain_s", 1.0))
        self.hover_z = float(hover_z_m)
        self.predicted_tcp_z_at_seat = predicted_tcp_z_at_seat

        # contact detection (in-evaluator so descended_post_contact works)
        self.contact_z: float | None = None
        self.contact_threshold_N = 6.0

        self._first_met_t: float | None = None
        self._last_xy: tuple[float, float] | None = None
        self._last_t: float | None = None
        self._last_z: float | None = None
        self._v_lat_ema: float = 0.0
        self._v_z_ema: float = 0.0
        self._ema_alpha: float = 0.2

    def _update_velocity(self, x: float, y: float, z: float,
                         t_now: float) -> tuple[float, float]:
        if self._last_xy is None or self._last_t is None or self._last_z is None:
            self._last_xy = (x, y)
            self._last_z = z
            self._last_t = t_now
            return 0.0, 0.0
        dt = max(t_now - self._last_t, 1e-3)
        vx = (x - self._last_xy[0]) / dt
        vy = (y - self._last_xy[1]) / dt
        vz = (z - self._last_z) / dt
        self._last_xy = (x, y)
        self._last_z = z
        self._last_t = t_now
        v_lat = (vx * vx + vy * vy) ** 0.5
        self._v_lat_ema = (self._ema_alpha * v_lat
                           + (1 - self._ema_alpha) * self._v_lat_ema)
        self._v_z_ema = (self._ema_alpha * vz
                         + (1 - self._ema_alpha) * self._v_z_ema)
        return self._v_lat_ema, self._v_z_ema

    def note_contact(self, tcp_z: float):
        """Wrapper calls this once when fz_smoothed first crosses contact threshold.
        Captures contact_z so descended_post_contact predicate can fire later.
        """
        if self.contact_z is None:
            self.contact_z = float(tcp_z)

    def eval(self, tcp_x: float, tcp_y: float, tcp_z: float,
             t_now: float, fz_smoothed: float | None = None) -> tuple[bool, dict]:
        v_lat, v_z = self._update_velocity(tcp_x, tcp_y, tcp_z, t_now)

        # Auto-detect contact based on fz_smoothed if wrapper passes it
        if fz_smoothed is not None and self.contact_z is None and fz_smoothed > self.contact_threshold_N:
            self.contact_z = tcp_z

        descended_from_hover = self.hover_z - tcp_z
        descended_post_contact = (self.contact_z - tcp_z) if self.contact_z is not None else None

        results: dict[str, bool] = {}

        ms = self.predicates_cfg.get("motion_stopped") or {}
        if ms.get("enabled", True):
            v_lat_ok = v_lat <= float(ms.get("v_lat_max_m_s", 0.005))
            v_z_max = ms.get("v_z_max_m_s")
            v_z_ok = (abs(v_z) <= float(v_z_max)) if v_z_max is not None else True
            results["motion_stopped"] = v_lat_ok and v_z_ok

        # Universal CAD-derived target (v1)
        ttp = self.predicates_cfg.get("tcp_z_reached_predicted") or {}
        if ttp.get("enabled", True) and self.predicted_tcp_z_at_seat is not None:
            tol = float(ttp.get("tolerance_m", 0.005))
            err = tcp_z - self.predicted_tcp_z_at_seat
            results["tcp_z_reached_predicted"] = abs(err) <= tol

        # Universal contact-relative (v0, no CAD needed)
        dpc = self.predicates_cfg.get("descended_post_contact") or {}
        if dpc.get("enabled", False):
            min_m = float(dpc.get("min_m", 0.025))
            results["descended_post_contact"] = (
                descended_post_contact is not None
                and descended_post_contact >= min_m
            )

        # Legacy v0.5 (per-shape hardcoded descent from hover) — backward compat
        zr = self.predicates_cfg.get("z_reached") or {}
        if zr.get("enabled", False):
            thr = zr.get("descended_min_m")
            if thr is not None:
                results["z_reached"] = descended_from_hover >= float(thr)

        if not results:
            return False, {
                "v_lat": v_lat, "v_z": v_z, "descended": descended_from_hover,
                "error": "no termination predicates enabled",
            }

        if self.combinator == "AND":
            all_met = all(results.values())
        elif self.combinator == "OR":
            all_met = any(results.values())
        else:
            return False, {"v_lat": v_lat, "v_z": v_z, "descended": descended_from_hover,
                           "error": f"unknown combinator {self.combinator!r}"}

        if all_met:
            if self._first_met_t is None:
                self._first_met_t = t_now
            sustained_s = t_now - self._first_met_t
            fired = sustained_s >= self.sustain_s
        else:
            self._first_met_t = None
            sustained_s = 0.0
            fired = False

        return fired, {
            "v_lat": v_lat,
            "v_z": v_z,
            "descended_from_hover": descended_from_hover,
            "descended_post_contact": descended_post_contact,
            "predicted_tcp_z_at_seat": self.predicted_tcp_z_at_seat,
            "results": results,
            "sustained_s": sustained_s,
        }
