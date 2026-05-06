"""Contact-guided peg-in-hole insertion state machine.

# Reference: https://github.com/swri-robotics/ConnTact (architecture)
# Reference: https://github.com/danielstankw/UR5e-robot-control (spiral math)

This is a clean rewrite of the Mode-B / CorrectionEvaluator approach. Instead
of continuously running force-oscillation corrections triggered by stuck
detection, we explicitly model the insertion as a sequence of discrete
states with measured (not assumed) exit conditions.

Operator constraints honored:
  - No hardcoded z thresholds — surface_z is MEASURED at APPROACH exit,
    hole drop is RELATIVE to that surface, INSERT exits on motion-stopped
    not absolute z target.
  - Per-part agnostic — same FSM works for any peg/slot pair.
  - Compensates pose-estimation error — search on surface IS the
    compensation; algorithm doesn't assume the predicted xy is correct.
  - Handles asymmetric pegs (U-shape) — ConnTact pattern with all-compliant
    INSERT phase + optional tilt rocking lets chamfer self-align.

States:
    APPROACH        peg descending in air, hunting for surface contact
    FIND_HOLE       peg on surface, spiral xy until z drops into chamfer
    INSERT          peg in chamfer, push down with full compliance to bottom
    DONE            seated successfully
    SAFETY_RETRACT  emergency back-off (force/timeout limit)
"""
from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class FSMAction:
    """One tick's output from the FSM."""
    # Lifecycle
    kind: str = "continue"           # "continue" | "exit_done" | "exit_abort"
    abort_reason: str = ""

    # Wrench update for force_mode_controller
    new_wrench: bool = False         # True -> wrapper should re-call SetForceMode
    wrench_baselink: tuple[float, float, float, float, float, float] = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    selection_vector: tuple[bool, bool, bool, bool, bool, bool] = (True, True, True, False, False, False)
    gain: float = 1.0
    damping: float = 0.7

    # Diagnostic info
    state: str = ""
    msg: str = ""

    # State-transition info (set when state changes)
    transitioned: bool = False
    new_state: str = ""
    transition_msg: str = ""


class SearchDirector:
    """Active search director — Archimedean spiral with PD position-tracking.

    v3 control law (analysis/SEARCH_CONTROL_LAW.md, iteration 3):

    Generates a spiral reference path centered on the first-contact xy and
    drives the peg along it via a saturated PD lateral force command, until
    the v4 detector fires the rim-cross transition.

    Why spiral and not F/T direction-following:
      Cross-demo analysis (10 GUIDED demos) shows F/T direction signals
      (-r_cop, -F_lat) are noise on flat rim (Q1 alignment +0.16 / -0.13)
      and only become directional at chamfer edge (Q4 +0.77 / +0.61). In
      autonomous force-mode, -F_lat suffers a friction positive-feedback
      confound (sensed F_lat opposes commanded F_lat in equilibrium, so
      -F_lat reinforces whatever direction the controller started in).
      Iter 1+2 demonstrated this empirically: peg drifted in a self-
      confirming direction, never crossed the rim.

      Spiral search with v4 detection is the published-evidence-backed
      answer when early F/T direction is ambiguous (Chhatpar & Branicky
      2001; Jasim et al. 2014). v4's empirical 10/10 success removes the
      historical pain point of spiral search (knowing when the hole is
      found).

    Reference path (spiral):
      Center: tcp_xy at first SEARCH tick.
      Radius: r(theta) = r0 + (pitch / 2π) * theta  [Archimedean]
      Tangential path speed: v_s   (constant arc-length rate).
      Termination: v4 fires (FSM-side), or radius >= R_max (abort), or
      stall detector trips (abort).

    Outer loop (force command, base_link):
      e = p_ref - p_tcp
      Fxy_cmd = sat( Kp * e - Kd * v_tcp , Fmax )
      Fz_cmd  = -F_press
      Selection vector: (T,T,T,F,F,F) — XYZ compliant, rotation locked.
    """

    def __init__(self,
                 r0_m: float = 0.0015,              # initial spiral radius (1.5mm — avoids peg-at-center stall)
                 pitch_m: float = 0.002,            # spiral pitch (per turn)
                 v_s_m_s: float = 0.005,            # tangential speed
                 R_max_m: float = 0.008,            # abort radius
                 Kp_xy_N_per_m: float = 350.0,
                 Kd_xy_N_per_m_per_s: float = 40.0,
                 Fmax_N: float = 3.0,
                 F_press_N: float = 9.0,
                 fz_gate_low_N: float = 3.0,
                 stall_progress_ratio: float = 0.15,
                 stall_window_s: float = 1.0,
                 lag_pause_thresh_m: float = 0.002,
                 near_miss_fz_thresh_N: float = 4.0):
        self.r0 = float(r0_m)
        self.pitch = float(pitch_m)
        self.v_s = float(v_s_m_s)
        self.R_max = float(R_max_m)
        self.Kp = float(Kp_xy_N_per_m)
        self.Kd = float(Kd_xy_N_per_m_per_s)
        self.Fmax = float(Fmax_N)
        self.F_press = float(F_press_N)
        self.fz_gate = float(fz_gate_low_N)
        self.stall_progress_ratio = float(stall_progress_ratio)
        self.stall_window_s = float(stall_window_s)
        self.lag_pause_thresh_m = float(lag_pause_thresh_m)
        self.near_miss_fz_thresh_N = float(near_miss_fz_thresh_N)

        # Reset state
        self._center_xy: Optional[tuple[float, float]] = None
        self._theta: float = 0.0
        self._spiral_path_len: float = 0.0
        self._last_t: Optional[float] = None
        self._tcp_buf: deque = deque(maxlen=10)  # last 10 samples for velocity (~100ms at 100Hz)
        # |fz| history for gradient-following (chamfer-deepening direction):
        # when |fz| drops, the peg is moving from rim into chamfer — keep going
        # in the same direction instead of letting spiral pull peg back to rim.
        self._fz_buf: deque = deque()  # (t, |fz|) over a 200ms window

        # Stall-tracker: actual peg progress vs spiral arc-length progress
        self._t_stall_start: Optional[float] = None
        self._spiral_arc_at_stall_start: float = 0.0
        self._tcp_pos_at_stall_start: Optional[tuple[float, float]] = None

        # Telemetry
        self.last_radius_m: float = 0.0
        self.last_p_ref: Optional[tuple[float, float]] = None
        self.last_e_m: tuple[float, float] = (0.0, 0.0)
        self.last_v_tcp: tuple[float, float] = (0.0, 0.0)
        self.last_cmd_base: tuple[float, float, float] = (0.0, 0.0, 0.0)
        self.last_mode: str = "uninit"
        self.abort_reason: Optional[str] = None  # set when stall/radius abort

    def set_center(self, center_xy_base: tuple[float, float]) -> None:
        """FSM calls this on first SEARCH tick with current TCP xy as spiral origin."""
        self._center_xy = (float(center_xy_base[0]), float(center_xy_base[1]))
        self._theta = 0.0
        self._spiral_path_len = 0.0
        self._t_stall_start = None
        self._spiral_arc_at_stall_start = 0.0
        self._tcp_pos_at_stall_start = None
        self._fz_buf.clear()

    def _estimate_velocity(self, t: float, tcp_x: float, tcp_y: float
                            ) -> tuple[float, float]:
        """Backward finite difference over the buffered window."""
        self._tcp_buf.append((t, tcp_x, tcp_y))
        if len(self._tcp_buf) < 3:
            return 0.0, 0.0
        t0, x0, y0 = self._tcp_buf[0]
        dt = t - t0
        if dt <= 1e-4:
            return 0.0, 0.0
        return (tcp_x - x0) / dt, (tcp_y - y0) / dt

    def update(self, t: float, fz_smoothed_base: float,
               tcp_x_base: float, tcp_y_base: float
               ) -> tuple[float, float, float]:
        """Returns (Fx_cmd, Fy_cmd, Fz_cmd) in base_link frame, or sets
        self.abort_reason if the search should be aborted."""
        Fz_cmd = -self.F_press
        self.abort_reason = None

        # Center must be set by FSM at first tick; if not, default to current.
        if self._center_xy is None:
            self.set_center((tcp_x_base, tcp_y_base))

        # First-tick init of last_t
        if self._last_t is None:
            self._last_t = t
            self._tcp_buf.append((t, tcp_x_base, tcp_y_base))
            self.last_mode = "spiral_init"
            self.last_cmd_base = (0.0, 0.0, Fz_cmd)
            return 0.0, 0.0, Fz_cmd

        dt = t - self._last_t
        self._last_t = t
        if dt <= 0:
            self.last_mode = "spiral_zero_dt"
            return self.last_cmd_base[0], self.last_cmd_base[1], Fz_cmd

        # If peg is off-rim (small fz), the v4 detector handles transition.
        # Director still commands Fz_press to keep peg pressed; lateral
        # command is reduced (peg may already be falling into chamfer).
        if abs(fz_smoothed_base) < self.fz_gate:
            self.last_mode = "spiral_off_rim"
            self.last_cmd_base = (0.0, 0.0, Fz_cmd)
            # Don't advance theta when off rim (peg doesn't need to spiral)
            return 0.0, 0.0, Fz_cmd

        # Advance spiral parameter ONLY when peg is keeping up with ref.
        # If position error > lag_pause_thresh, hold theta until peg catches up.
        # This makes the peg control the spiral expansion rate — guarantees
        # peg actually traces the spiral disk, not a smaller-radius shadow of it.
        # Critical when spiral center is offset from actual hole — peg has to
        # physically reach the radius where the hole is, not just spiral ref.
        cur_r = self.r0 + (self.pitch / (2 * math.pi)) * self._theta
        cur_r_safe = max(cur_r, self.r0)

        # Compute current peg-to-ref error magnitude (before advancing)
        prev_p_ref_x = self._center_xy[0] + cur_r * math.cos(self._theta)
        prev_p_ref_y = self._center_xy[1] + cur_r * math.sin(self._theta)
        prev_e_mag = math.hypot(prev_p_ref_x - tcp_x_base, prev_p_ref_y - tcp_y_base)

        spiral_advanced = False
        if prev_e_mag <= self.lag_pause_thresh_m:
            # Peg is at ref → advance spiral
            theta_dot = self.v_s / cur_r_safe
            self._theta += theta_dot * dt
            spiral_advanced = True
        # else: peg is lagging — hold theta until peg catches up.

        new_r = self.r0 + (self.pitch / (2 * math.pi)) * self._theta
        self.last_radius_m = new_r

        # Abort: spiral exhausted
        if new_r > self.R_max:
            self.abort_reason = f"spiral_exhausted (r={1000*new_r:.1f}mm > R_max={1000*self.R_max:.1f}mm)"
            self.last_mode = "abort_spiral_exhausted"
            self.last_cmd_base = (0.0, 0.0, Fz_cmd)
            return 0.0, 0.0, Fz_cmd

        # Reference position
        cx, cy = self._center_xy
        p_ref_x = cx + new_r * math.cos(self._theta)
        p_ref_y = cy + new_r * math.sin(self._theta)
        self.last_p_ref = (p_ref_x, p_ref_y)

        # Position error
        e_x = p_ref_x - tcp_x_base
        e_y = p_ref_y - tcp_y_base
        self.last_e_m = (e_x, e_y)
        e_mag = math.hypot(e_x, e_y)

        # TCP velocity (for telemetry / damping option)
        v_x, v_y = self._estimate_velocity(t, tcp_x_base, tcp_y_base)
        self.last_v_tcp = (v_x, v_y)

        # Maintain |fz| history for gradient-following.
        abs_fz = abs(fz_smoothed_base)
        self._fz_buf.append((t, abs_fz))
        while self._fz_buf and (t - self._fz_buf[0][0]) > 0.2:
            self._fz_buf.popleft()
        d_fz_dt = 0.0
        if len(self._fz_buf) >= 3:
            t0, fz0 = self._fz_buf[0]
            window_dt = t - t0
            if window_dt > 1e-3:
                d_fz_dt = (abs_fz - fz0) / window_dt  # N/s

        # GRADIENT-FOLLOWING OVERRIDE: when |fz| is dropping AND already low
        # (peg moving from rim into chamfer), continue in the direction of
        # recent peg velocity instead of chasing the spiral ref. This pushes
        # peg deeper into the chamfer where it dropped instead of letting the
        # spiral yank it back onto the rim. Active control kicks in as the
        # peg approaches the chamfer edge.
        v_mag = math.hypot(v_x, v_y)
        gradient_active = (d_fz_dt < -3.0           # fz dropping > 3 N/s
                           and abs_fz < 6.0          # already at/near chamfer threshold
                           and v_mag > 5e-4)         # peg actually moving (>0.5mm/s)

        if gradient_active:
            # Continue in current peg-velocity direction at full Fmax.
            # Sign-flip: cmd opposite of desired motion direction.
            v_dir_x = v_x / v_mag
            v_dir_y = v_y / v_mag
            Fx_cmd = -self.Fmax * v_dir_x
            Fy_cmd = -self.Fmax * v_dir_y
        elif e_mag > 1e-6:
            ux_e = e_x / e_mag
            uy_e = e_y / e_mag
            # CONSTANT-FORCE tracking (replaces PD): always command Fmax magnitude
            # in the direction of the reference position. PD with default Kp=350 N/m
            # gave only ~1.3N of commanded force at typical position errors (0.5–4 mm),
            # well below stiction threshold → immediate stall. Constant-force always
            # engages Fmax — guarantees stiction break, spiral reference pulls peg
            # along path. Damping subtracted to prevent overshoot when peg is moving.
            #
            # IMPORTANT (iter 7 finding): peg moves OPPOSITE direction of commanded
            # F_lat in base_link. To make peg approach p_ref, command F_lat in the
            # OPPOSITE of (p_ref - p_tcp).
            #
            # NEGATIVE sign: move-toward-target requires opposite-direction commanded F_lat
            Fx_cmd = -self.Fmax * ux_e - self.Kd * v_x
            Fy_cmd = -self.Fmax * uy_e - self.Kd * v_y
            # Re-saturate at 1.25*Fmax (allows damping headroom)
            F_mag = math.hypot(Fx_cmd, Fy_cmd)
            cap = self.Fmax * 1.25
            if F_mag > cap:
                scale = cap / F_mag
                Fx_cmd *= scale
                Fy_cmd *= scale
        else:
            Fx_cmd = 0.0
            Fy_cmd = 0.0

        # Spiral arc-length tracking for stall detection.
        # Only count when spiral was allowed to advance (peg keeping up). When
        # peg is lagging and theta is held, the stall comparison is meaningless.
        if spiral_advanced:
            self._spiral_path_len += self.v_s * dt

        # Stall detector: actual TCP progress vs spiral path length
        if self._t_stall_start is None:
            self._t_stall_start = t
            self._spiral_arc_at_stall_start = self._spiral_path_len
            self._tcp_pos_at_stall_start = (tcp_x_base, tcp_y_base)
        elif (t - self._t_stall_start) >= self.stall_window_s:
            arc_elapsed = self._spiral_path_len - self._spiral_arc_at_stall_start
            tcp_progress = math.hypot(
                tcp_x_base - self._tcp_pos_at_stall_start[0],
                tcp_y_base - self._tcp_pos_at_stall_start[1])
            if arc_elapsed > 1e-4 and tcp_progress / arc_elapsed < self.stall_progress_ratio:
                self.abort_reason = (
                    f"lateral_stall (tcp_progress={1000*tcp_progress:.2f}mm vs "
                    f"spiral_arc={1000*arc_elapsed:.2f}mm in {self.stall_window_s:.2f}s)"
                )
                self.last_mode = "abort_stall"
            # Reset window
            self._t_stall_start = t
            self._spiral_arc_at_stall_start = self._spiral_path_len
            self._tcp_pos_at_stall_start = (tcp_x_base, tcp_y_base)

        self.last_cmd_base = (Fx_cmd, Fy_cmd, Fz_cmd)
        self.last_mode = f"spiral r={1000*new_r:.2f}mm θ={math.degrees(self._theta):.0f}°"
        return Fx_cmd, Fy_cmd, Fz_cmd


class FoundHoleDetector:
    """v4 rim-cross state-transition predicate.

    Fires when |fz_smoothed| has been below `rim_low_thresh` for at least
    `off_sustain_s` seconds, AND was above `rim_high_thresh` at any point
    in the previous `recent_window_s` seconds.

    Time-based windows (not sample counts), so robust to tick rate variation.
    Sticky single-shot — `update()` returns True only on the firing tick.

    Derived from cross-demo analysis of 10 GUIDED demos in 3 directions
    (analysis/CONTROL_LAW.md). Default thresholds verified 10/10 + robust
    across 94/96 parameter combos in the sweep.
    """

    def __init__(self, rim_high_thresh: float = 4.0,
                 rim_low_thresh: float = 3.0,
                 off_sustain_s: float = 0.30,
                 recent_window_s: float = 2.5):
        self.rim_high = float(rim_high_thresh)
        self.rim_low = float(rim_low_thresh)
        self.off_sustain_s = float(off_sustain_s)
        self.recent_window_s = float(recent_window_s)
        self._on_rim_history: deque = deque()  # (t, was_on_rim)
        self._off_run_start: Optional[float] = None
        self._fired = False

    def reset(self):
        self._on_rim_history.clear()
        self._off_run_start = None
        self._fired = False

    @property
    def fired(self) -> bool:
        return self._fired

    def update(self, t: float, fz_smoothed: float) -> bool:
        """Returns True on the tick the predicate first fires; False after."""
        if self._fired:
            return False
        abs_fz = abs(fz_smoothed)
        on_rim_now = abs_fz > self.rim_high
        off_rim_now = abs_fz < self.rim_low

        self._on_rim_history.append((t, on_rim_now))
        while (self._on_rim_history
               and (t - self._on_rim_history[0][0]) > self.recent_window_s):
            self._on_rim_history.popleft()

        if off_rim_now:
            if self._off_run_start is None:
                self._off_run_start = t
        else:
            self._off_run_start = None

        if (self._off_run_start is not None
            and (t - self._off_run_start) >= self.off_sustain_s
            and any(on_rim for _, on_rim in self._on_rim_history)):
            self._fired = True
            return True
        return False


class ContactSearchFSM:
    APPROACH        = "APPROACH"
    FIND_HOLE       = "FIND_HOLE"
    ENTRY_SETTLE    = "ENTRY_SETTLE"
    WEDGE_RECOVERY  = "WEDGE_RECOVERY"   # counter-torque on Tx to lift stuck prong
    INSERT          = "INSERT"
    DONE            = "DONE"
    SAFETY_RETRACT  = "SAFETY_RETRACT"
    GUIDED          = "GUIDED"           # operator-drag mode for GOLD data collection
    SEARCH          = "SEARCH"           # autonomous F/T-driven search (replaces operator drag)
    INSERT_DESCENT  = "INSERT_DESCENT"   # post-GUIDED Z-descent at captured hole_xy

    def __init__(self, cfg: Optional[dict] = None,
                 spiral_origin_override: Optional[tuple[float, float]] = None,
                 predicted_tcp_xy: Optional[tuple[float, float]] = None,
                 predicted_tcp_z: Optional[float] = None,
                 r_grasp_to_partcenter_world: tuple[float, float, float] = (0.0, 0.0, 0.0),
                 object_origin_in_EE: Optional[tuple[float, float, float]] = None):
        self.cfg = cfg or {}
        self.state = self.APPROACH
        # Optional spiral origin override (e.g. hole_xy_prior from successful
        # prior attempt). When provided, FIND_HOLE spiral expands from THIS
        # xy instead of from peg's contact xy.
        self.spiral_origin_override = spiral_origin_override
        # Predicted TCP xy at hole (from CAD chain). Used as a GLOBAL
        # PLAUSIBILITY ANCHOR — separate concept from spiral_center_xy.
        self.predicted_tcp_xy = predicted_tcp_xy
        # Predicted TCP z at seat (from CAD chain). Used as the absolute
        # reference for the "At Target" marker. Absolute is correct because
        # multi-contact insertions (peg encountering features along the way)
        # would re-latch surface_z and erase relative-descent metrics, but
        # absolute target-z stays valid throughout.
        self.predicted_tcp_z = predicted_tcp_z
        # Vector from TCP to part_center (object origin) in WORLD frame.
        # Computed from CAD predicted_tcp_at_seat - T_world_object_seat.
        # Used for Fz→T lever-arm compensation: pure Fz at TCP creates
        # torque T = r × F at part center (since grasp is offset from
        # part center). To get pure Fz behavior at part center, we add
        # -compensating-torque to commanded TCP wrench. This addresses the
        # operator's geometric insight that v75 confirmed: passive Fz=-9N
        # alone caused tilt 0→4° in 10s due to uncompensated lever-arm
        # moment around the offset grasp point.
        self._r_partcenter = r_grasp_to_partcenter_world
        # object_origin in EE frame = constant offset of part center from TCP
        # in EE-frame coords. Used to recompute r_world dynamically each tick
        # by applying the current TCP quat rotation. If not provided, dynamic
        # update is disabled and static _r_partcenter is used.
        self._object_origin_in_EE = object_origin_in_EE

        # Measured-not-assumed quantities populated on state transitions
        self.surface_z: Optional[float] = None       # set when APPROACH exits
        self.surface_t: Optional[float] = None
        self.hole_xy: Optional[tuple[float, float]] = None  # set when FIND_HOLE exits
        self.hole_z: Optional[float] = None
        self.hole_t: Optional[float] = None
        self.insert_start_t: Optional[float] = None

        # GUIDED-mode state (operator-drag data collection):
        self.guided_mode: bool = bool(self.cfg.get("guided_mode", False))
        # Operator signals "above hole now" by setting this flag (the wrapper
        # sets it on receiving SIGUSR1). FSM checks each tick during GUIDED
        # state — when True, captures TCP and transitions to INSERT_DESCENT.
        self._guided_hole_marked: bool = False
        # Captured at the moment of the operator's signal (= GUIDED→INSERT_DESCENT transition):
        self.hole_observed_xy: Optional[tuple[float, float]] = None
        self.hole_observed_z:  Optional[float] = None
        self.hole_observed_t:  Optional[float] = None
        # GUIDED-state defaults (loose for easy operator manipulation, matching
        # what we validated in 2026-05-06 gimbal_mode_test.py):
        self.guided_gain    = float(self.cfg.get("guided_gain",    2.0))
        self.guided_damping = float(self.cfg.get("guided_damping", 0.05))
        self._guided_entered: bool = False
        self._insert_descent_entered: bool = False
        self.insert_start_z: Optional[float] = None

        # v4 Found Hole predicate detector (analysis/CONTROL_LAW.md). Runs
        # alongside operator SIGUSR1 in GUIDED state. If `v4_autofire` is True,
        # a v4 fire ALSO triggers GUIDED → INSERT_DESCENT (stage 3b). If False
        # (stage 3a / default), v4 fire is logged but the operator's SIGUSR1
        # remains the only trigger.
        self.v4_autofire: bool = bool(self.cfg.get("v4_autofire", False))
        self.found_hole_detector = FoundHoleDetector(
            rim_high_thresh = float(self.cfg.get("v4_rim_high_thresh", 4.0)),
            rim_low_thresh  = float(self.cfg.get("v4_rim_low_thresh",  3.0)),
            off_sustain_s   = float(self.cfg.get("v4_off_sustain_s",   0.30)),
            recent_window_s = float(self.cfg.get("v4_recent_window_s", 2.5)),
        )
        # Set when v4 fires (None until then). Persisted to meta on exit.
        self.v4_predicate_fire: Optional[dict] = None

        # Autonomous SEARCH (replaces operator-drag GUIDED). When True,
        # APPROACH→Contact routes to SEARCH instead of GUIDED. SEARCH uses
        # SearchDirector to actively command F_lat in the −r_cop direction
        # (analysis/SEARCH_CONTROL_LAW.md). v4 detector continues to run and
        # triggers SEARCH → INSERT_DESCENT.
        self.autonomous_search: bool = bool(self.cfg.get("autonomous_search", False))
        # SearchDirector v3: Archimedean spiral PD director (analysis/SEARCH_CONTROL_LAW.md).
        # Parameters from GPT-5 analysis (gpt-temaxw-74fd8c) backed by Chhatpar/Branicky 2001
        # and Jasim/Plapper/Voos 2014. Spiral parameters tuned for chamfer-capture not
        # fine-clearance — paired with v4 detector for transition.
        self.search_director = SearchDirector(
            r0_m              = float(self.cfg.get("search_r0_m",        0.0015)),  # 1.5mm — avoids peg-at-center stall
            pitch_m           = float(self.cfg.get("search_pitch_m",     0.002)),   # 2mm
            v_s_m_s           = float(self.cfg.get("search_v_s_m_s",     0.005)),   # 5mm/s
            R_max_m           = float(self.cfg.get("search_R_max_m",     0.008)),   # 8mm
            Kp_xy_N_per_m     = float(self.cfg.get("search_Kp_N_m",      350.0)),
            Kd_xy_N_per_m_per_s = float(self.cfg.get("search_Kd_N_s_m",  40.0)),
            Fmax_N            = float(self.cfg.get("search_Fmax_N",      3.0)),
            F_press_N         = float(self.cfg.get("search_F_press_N",   9.0)),
            fz_gate_low_N     = float(self.cfg.get("search_fz_gate_N",   3.0)),
        )
        # Seeded by FSM at SEARCH entry (predicted_tcp_xy from CAD prior).
        # SearchDirector's bootstrap mode drives toward this xy until |F_lat|
        # picks up the chamfer signal.
        self.search_max_duration_s: float = float(self.cfg.get("search_max_duration_s", 15.0))
        self._search_entered: bool = False
        self._search_start_t: Optional[float] = None
        self._search_last_cmd_t: Optional[float] = None
        self._search_cmd_period_s: float = float(self.cfg.get("search_cmd_period_s", 0.10))

        # FIND_HOLE spiral state
        self._spiral_origin_xy: Optional[tuple[float, float]] = None
        self._spiral_theta: float = 0.0
        self._spiral_last_t: Optional[float] = None
        self._latest_T_lat_mag: float = 0.0
        self._engaged_first_t: Optional[float] = None    # tracks sustain
        self._F_lat_smooth_buffer: deque = deque()       # for engagement check
        self._xy_buffer: deque = deque()                 # for v_xy estimation
        self._entry_settle_start_t: Optional[float] = None
        self._entry_settle_z_at_start: Optional[float] = None
        self._wedge_start_t: Optional[float] = None
        self._wedge_attempts: int = 0
        self._tx_lp_buffer: deque = deque()              # for low-pass filtering Tx
        self._ty_lp_buffer: deque = deque()              # for low-pass filtering Ty
        self._T_partcenter: tuple[float, float] = (0.0, 0.0)  # sensed T about part center (filled per tick)
        self._tilt_err_world: tuple[float, float, float] = (0.0, 0.0, 0.0)  # EE-Z cross world-Z; tilt vector for orientation-feedback control
        self._global_seat_first_t: Optional[float] = None  # state-independent seat-sustain timer

        # INSERT motion-stopped detection
        self._z_buffer: deque = deque()    # of (t, z)
        self._fz_buffer: deque = deque()
        self._F_lat_buffer: deque = deque()  # of (t, F_lat_mag)
        self._motion_stopped_first_t: Optional[float] = None

        # APPROACH timeout (don't fall forever)
        self._approach_start_t: Optional[float] = None
        self._find_hole_start_t: Optional[float] = None

        # Diagnostic throttle
        self._last_diag_t: float = 0.0

        # === Config knobs (all overrideable from defaults.yaml) ===
        # APPROACH: descend under light Fz until contact
        self.approach_fz_N            = float(self.cfg.get("approach_fz_N",            6.0))
        self.approach_max_duration_s  = float(self.cfg.get("approach_max_duration_s", 30.0))
        # Grace period at start of APPROACH before contact detection arms.
        # Force_mode_controller startup transient + sensor noise can cause
        # smoothed fz to briefly exceed contact_threshold_N right at ACTIVE
        # entry, before peg has descended at all. 1s grace lets the controller
        # settle. Verified empirically 2026-05-06 session 7 (raw fz oscillating
        # ±5N in air during first ~0.5s of force-mode active).
        self.approach_grace_period_s  = float(self.cfg.get("approach_grace_period_s", 1.0))
        self.contact_threshold_N      = float(self.cfg.get("contact_threshold_N",     6.0))
        self.contact_sustain_s        = float(self.cfg.get("contact_sustain_s",       0.1))
        self.fz_smooth_window_s       = float(self.cfg.get("fz_smooth_window_s",      0.3))

        # FIND_HOLE: spiral on surface
        self.find_hole_fz_N           = float(self.cfg.get("find_hole_fz_N",           4.0))
        # Drop threshold = minimum z-descent that confirms peg moved past rim.
        # NOT a hard "engagement depth" — combined with force/torque signal
        # below to detect FULL engagement (no rim contact remaining).
        self.find_hole_drop_thresh_m  = float(self.cfg.get("find_hole_drop_thresh_m", 0.0008))
        # Engagement signal thresholds — peg is FULLY engaged in slot when all
        # are satisfied (checked DURING ENTRY_SETTLE, not FIND_HOLE):
        #   F_lat low      no rim wall pushing peg sideways
        #   |T_lat| low    peg sitting flush with slot walls (no tilt)
        #   |v_xy| low     peg quasi-static (not still drifting)
        # These are GEOMETRY-AGNOSTIC: works for round, U-shape, L-shape pegs.
        self.engaged_F_lat_max_N      = float(self.cfg.get("engaged_F_lat_max_N",      3.0))
        self.engaged_T_lat_max_Nm     = float(self.cfg.get("engaged_T_lat_max_Nm",     0.15))
        # T about PART CENTER threshold (operator's principled fix). Larger
        # than TCP-frame threshold because the long peg-tip→TCP lever arm
        # multiplies F into bigger moments at part center. Initial guess
        # 0.50 Nm; will iterate from log data.
        self.engaged_T_lat_max_Nm_partcenter = float(self.cfg.get("engaged_T_lat_max_Nm_partcenter", 0.50))
        self.engaged_v_xy_max_m_s     = float(self.cfg.get("engaged_v_xy_max_m_s",     0.001))  # 1mm/s
        self.engaged_sustain_s        = float(self.cfg.get("engaged_sustain_s",        0.5))

        # ENTRY_SETTLE — between FIND_HOLE and INSERT (per GPT review).
        # Stops spiral, holds gentle Fz, waits for transients, then verifies
        # quasi-static low F_lat / T_lat. Without this, we can't distinguish
        # "true engagement" from "spiral-commanded force showing up as F_lat".
        self.entry_settle_fz_N        = float(self.cfg.get("entry_settle_fz_N",        4.0))
        self.entry_settle_wait_s      = float(self.cfg.get("entry_settle_wait_s",      0.2))   # 200ms transient die-down
        self.entry_settle_verify_s    = float(self.cfg.get("entry_settle_verify_s",    0.2))   # 200ms sustain
        self.entry_settle_max_s       = float(self.cfg.get("entry_settle_max_s",       1.5))   # max time before giving up
        self.entry_settle_gain        = float(self.cfg.get("entry_settle_gain",        1.0))
        self.entry_settle_damping     = float(self.cfg.get("entry_settle_damping",     0.7))
        self.find_hole_max_radius_m   = float(self.cfg.get("find_hole_max_radius_m",  0.012))    # 12mm
        self.find_hole_max_duration_s = float(self.cfg.get("find_hole_max_duration_s", 60.0))
        # INITIAL_PRESS phase: cross-run analysis shows successes have ~5N
        # sustained F_lat at t=1-1.5s. Direction = base_link angle (degrees).
        # Operator demo walk direction was -67° world → +113° base_link.
        # Easier to think in base_link: 90° = +Y push (= -Y world walk).
        self.find_hole_press_duration_s = float(self.cfg.get("find_hole_press_duration_s", 1.5))
        self.find_hole_press_F_N         = float(self.cfg.get("find_hole_press_F_N",         5.0))
        self.find_hole_press_fz_N        = float(self.cfg.get("find_hole_press_fz_N",        8.0))
        self.find_hole_press_dir_deg     = float(self.cfg.get("find_hole_press_dir_deg",   90.0))
        # Aggressive stuck detector: operator demos descend 10mm+ in 5s.
        # If our run shows < this z_drop after this window post-contact,
        # peg is genuinely stuck on rim — abort early.
        self.find_hole_stuck_window_s = float(self.cfg.get("find_hole_stuck_window_s", 15.0))
        self.find_hole_stuck_z_drop_m = float(self.cfg.get("find_hole_stuck_z_drop_m", 0.001))   # 1mm
        # Stankowski/Chhatpar Archimedean spiral
        self.spiral_v_m_s             = float(self.cfg.get("spiral_v_m_s",             0.0015))
        self.spiral_pitch_m           = float(self.cfg.get("spiral_pitch_m",           0.0006))
        self.spiral_kp_N_per_m        = float(self.cfg.get("spiral_kp_N_per_m",        1500.0))
        self.spiral_F_max_N           = float(self.cfg.get("spiral_F_max_N",           6.0))
        # H101: directed sweep — replaces spiral with sustained directed push
        self.find_hole_use_directed_sweep    = bool(self.cfg.get("find_hole_use_directed_sweep",    False))
        self.find_hole_directed_F_N          = float(self.cfg.get("find_hole_directed_F_N",          5.0))
        self.find_hole_directed_dwell_s      = float(self.cfg.get("find_hole_directed_dwell_s",      1.5))
        self.find_hole_directed_rotate_deg   = float(self.cfg.get("find_hole_directed_rotate_deg",  60.0))
        self.find_hole_directed_n_directions = int(self.cfg.get("find_hole_directed_n_directions",   6))
        self.find_hole_directed_latch_F_N    = float(self.cfg.get("find_hole_directed_latch_F_N",    2.0))
        # Iter-2: distance-scaled P-controller params
        self.find_hole_directed_near_dwell_mm    = float(self.cfg.get("find_hole_directed_near_dwell_mm",    3.0))
        self.find_hole_directed_scaling_zone_mm  = float(self.cfg.get("find_hole_directed_scaling_zone_mm", 10.0))
        self.find_hole_directed_dir_deadband_mm  = float(self.cfg.get("find_hole_directed_dir_deadband_mm", 0.5))
        # Iter-7: pulse pattern (push/release cycle) to emulate operator's grip-release
        # which produces T_lat relaxation moments observable in GOLD at close-pass
        self.find_hole_directed_pulse_period_s   = float(self.cfg.get("find_hole_directed_pulse_period_s",   1.5))
        self.find_hole_directed_pulse_release_s  = float(self.cfg.get("find_hole_directed_pulse_release_s",  0.3))
        # Iter-9: synchronize Fz pulse with F_lat pulse — drop cmd_fz during release window
        # so peg's pinning force on rim drops briefly → can shift into slot via gravity
        self.find_hole_directed_release_fz_N     = float(self.cfg.get("find_hole_directed_release_fz_N",    1.0))
        # Iter-10 (A1+A2+A3): orientation-feedback steering + online seat refinement
        self.find_hole_tilt_tolerance_deg     = float(self.cfg.get("find_hole_tilt_tolerance_deg",     3.33))
        self.find_hole_tilt_relax_min_deg     = float(self.cfg.get("find_hole_tilt_relax_min_deg",     0.40))
        self.find_hole_tilt_relax_sustain_s   = float(self.cfg.get("find_hole_tilt_relax_sustain_s",   0.30))
        self.find_hole_tilt_high_sustain_s    = float(self.cfg.get("find_hole_tilt_high_sustain_s",    0.30))
        self.find_hole_tilt_history_window_s  = float(self.cfg.get("find_hole_tilt_history_window_s",  1.0))
        # Iter-10 state
        self._tilt_history: list[tuple[float, float]] = []  # (t, tilt_deg) rolling
        self._tilt_high_first_t: float | None = None        # when tilt first crossed tolerance
        self._tilt_running_peak_deg: float = 0.0
        self._tilt_running_peak_t: float | None = None
        self._tilt_relax_first_t: float | None = None       # when sustained drop from peak began
        self._chamfer_engaged: bool = False                 # latched: peg engaged, transition to ENTRY_SETTLE
        self._seat_xy_refined: tuple[float, float] | None = None  # online-refined seat estimate
        self._last_F_lat_baselink: tuple[float, float] = (0.0, 0.0)
        self._last_T_lat_baselink: tuple[float, float] = (0.0, 0.0)
        self._last_stable_unit: tuple[float, float] = (0.0, 0.0)
        self._find_hole_first_t: float | None = None
        # Optional tilt rocking during search (Tx/Ty oscillation)
        self.find_hole_tilt_T_Nm      = float(self.cfg.get("find_hole_tilt_T_Nm",      0.0))
        self.find_hole_tilt_freq_hz   = float(self.cfg.get("find_hole_tilt_freq_hz",   1.5))
        # Tz wiggle for U-shape yaw alignment during search
        self.find_hole_tz_T_Nm        = float(self.cfg.get("find_hole_tz_T_Nm",        0.0))
        self.find_hole_tz_freq_hz     = float(self.cfg.get("find_hole_tz_freq_hz",     0.5))

        # INSERT: push down with full compliance to seat
        self.insert_fz_N              = float(self.cfg.get("insert_fz_N",              9.0))
        self.insert_tilt_T_Nm         = float(self.cfg.get("insert_tilt_T_Nm",         0.20))
        self.insert_tilt_freq_hz      = float(self.cfg.get("insert_tilt_freq_hz",      1.0))
        # Tz wiggle for U-shape yaw alignment when one prong is stuck on rim
        self.insert_tz_T_Nm           = float(self.cfg.get("insert_tz_T_Nm",           0.30))
        self.insert_tz_freq_hz        = float(self.cfg.get("insert_tz_freq_hz",        0.7))
        self.insert_motion_window_s   = float(self.cfg.get("insert_motion_window_s",   1.5))
        self.insert_motion_thresh_m_s = float(self.cfg.get("insert_motion_thresh_m_s", 0.0005))  # 0.5mm/s
        self.insert_max_duration_s    = float(self.cfg.get("insert_max_duration_s",    30.0))
        self.insert_min_descent_m     = float(self.cfg.get("insert_min_descent_m",     0.005))   # 5mm
        # Lateral-force safety abort: smoothed over window to ignore
        # impulsive collisions (e.g. prong catching rim edge for 11ms).
        # Operator demos peaked at 14N briefly; sustained 30N+ means real jam.
        self.abort_F_lat_N            = float(self.cfg.get("abort_F_lat_N",            30.0))
        self.abort_F_lat_window_s     = float(self.cfg.get("abort_F_lat_window_s",     0.10))

        # Predicted-pose recovery leash (FIND_HOLE) and engagement gate
        # (ENTRY_SETTLE). GPT-recommended starting values.
        self.recovery_r_free_m        = float(self.cfg.get("recovery_r_free_m",        0.006))   # 6mm
        self.recovery_r_full_m        = float(self.cfg.get("recovery_r_full_m",        0.012))   # 12mm
        self.recovery_F_max_N         = float(self.cfg.get("recovery_F_max_N",         2.0))     # 20-40% of spiral_F_max
        self.engagement_dist_thresh_m = float(self.cfg.get("engagement_dist_thresh_m", 0.006))   # 6mm
        # Z-drop dominance shortcut: if peg has dropped > this much below
        # surface_z, consider it unambiguously engaged regardless of CAD-xy
        # distance. Operator-demo full insertion is ~25-30mm; CAD-xy can be
        # off by 5-20mm so the dist gate alone wrongly rejects real seats.
        self.engaged_z_drop_dominant_m = float(self.cfg.get("engaged_z_drop_dominant_m", 0.020))   # 20mm
        # Absolute "At Target" tolerance: |tcp_z - predicted_tcp_z| < this.
        # Used by the global seat detector when predicted_tcp_z is available
        # (CAD chain). 5mm catches both peg-slightly-above-predicted (operator
        # marked hole near edge, peg seated at slot lip) and peg-slightly-below
        # (slot deeper than CAD; verified 1.6mm in 2026-05-06 first GUIDED demo).
        self.at_target_z_tol_m = float(self.cfg.get("at_target_z_tol_m", 0.005))
        # Tilt thresholds (deg from canonical face-down). Operator-demo data:
        #   u_orange: max 1.36° during successful descent
        #   u_brown:  max 5.56°
        #   line_green: max 4.07°
        #   inverted_u: max 6.23°
        # Gate at ~2× u_orange's max (most stringent object) → 3°.
        # INSERT abort threshold ~5° — past which lever-arm runaway dominates.
        self.engaged_tilt_max_deg     = float(self.cfg.get("engaged_tilt_max_deg",     3.0))
        self.insert_tilt_abort_deg    = float(self.cfg.get("insert_tilt_abort_deg",    5.0))
        self._tilt_deg                = 0.0

        # WEDGE_RECOVERY (counter-torque on Tx). Verified from 60 operator
        # demos: sensed Tx has consistent sign during wedge (50-96% sign
        # persistence), decays as peg unwedges. Counter-torque (opposite
        # sign) rotates peg to lower stuck prong. GPT-recommended conservative
        # gains; ~30% of operator-demo peak (1 Nm).
        self.wedge_min_Tx_Nm          = float(self.cfg.get("wedge_min_Tx_Nm",          0.05))   # only act if |Tx| > this
        self.wedge_dominance_ratio    = float(self.cfg.get("wedge_dominance_ratio",    1.5))    # |Tx| > ratio * |Ty|
        self.wedge_kp_torque          = float(self.cfg.get("wedge_kp_torque",          0.7))    # commanded = -kp * Tx_filt
        self.wedge_cmd_Tx_max_Nm      = float(self.cfg.get("wedge_cmd_Tx_max_Nm",      0.4))    # cap commanded torque
        self.wedge_filter_window_s    = float(self.cfg.get("wedge_filter_window_s",    0.1)) # 100ms low-pass
        self.wedge_fz_N               = float(self.cfg.get("wedge_fz_N",               2.5))    # REDUCED Fz during recovery (avoid pinning)
        self.wedge_duration_s         = float(self.cfg.get("wedge_duration_s",         1.5))    # then re-evaluate via ENTRY_SETTLE
        self.wedge_max_attempts       = int(self.cfg.get("wedge_max_attempts",         3))      # before retract
        self.wedge_retract_fz_N       = float(self.cfg.get("wedge_retract_fz_N",       6.0))    # +Fz to lift peg off rim
        # Orientation-feedback gain: torque commanded ∝ -kp * tilt_err vector
        # (drives EE to canonical face-down). Used in WEDGE_RECOVERY +
        # RETRACT phases. tilt_err magnitude ≈ tilt_rad so kp=2 means
        # 1° tilt → 0.035 Nm command, 5° → 0.17 Nm. Cap via wedge_cmd_Tx_max_Nm.
        self.relevel_kp_torque        = float(self.cfg.get("relevel_kp_torque",        2.0))

        # Default gain/damping per state (UR force_mode parameters)
        self.gain_approach            = float(self.cfg.get("gain_approach",   1.0))
        self.damping_approach         = float(self.cfg.get("damping_approach", 0.7))
        self.gain_find_hole           = float(self.cfg.get("gain_find_hole",  0.7))
        self.damping_find_hole        = float(self.cfg.get("damping_find_hole", 0.5))
        self.gain_insert              = float(self.cfg.get("gain_insert",     1.0))
        self.damping_insert           = float(self.cfg.get("damping_insert",  0.7))

    # ---------- Helpers --------------------------------------------------
    def _smooth_fz(self, t: float, fz: float) -> float:
        self._fz_buffer.append((t, fz))
        while self._fz_buffer and (t - self._fz_buffer[0][0]) > self.fz_smooth_window_s:
            self._fz_buffer.popleft()
        if not self._fz_buffer:
            return fz
        return sum(v for _, v in self._fz_buffer) / len(self._fz_buffer)

    def _z_velocity(self, t: float, z: float, window_s: float) -> Optional[float]:
        self._z_buffer.append((t, z))
        while self._z_buffer and (t - self._z_buffer[0][0]) > window_s:
            self._z_buffer.popleft()
        if len(self._z_buffer) < 5:
            return None
        t0, z0 = self._z_buffer[0]
        dt = t - t0
        if dt <= 0:
            return None
        return (z - z0) / dt    # +ve = ascending; for descent we want negative

    def _z_velocity_xy(self, t: float, x: float, y: float, window_s: float) -> Optional[tuple[float, float]]:
        """Returns (vx, vy) over window_s, or None if not enough samples."""
        self._xy_buffer.append((t, x, y))
        while self._xy_buffer and (t - self._xy_buffer[0][0]) > window_s:
            self._xy_buffer.popleft()
        if len(self._xy_buffer) < 5:
            return None
        t0, x0, y0 = self._xy_buffer[0]
        dt = t - t0
        if dt <= 0:
            return None
        return ((x - x0) / dt, (y - y0) / dt)

    def _approach_wrench(self) -> tuple[tuple, float, float, tuple]:
        """APPROACH wrench: light downward push.
        Selection vector: X+Y LOCKED, Z compliant, rotation LOCKED.
        2026-05-06: changed XY from compliant→locked. Without this, force-mode
        yields to any sensed lateral force during descent (gravity moments,
        payload mismatch residuals, encoder noise) → 0.83mm XY drift over 19s.
        With XY locked, peg descends in a perfectly straight Z line from hover.
        Contact happens at exactly hover_xy. Required for clean GUIDED-mode
        data collection (we want contact_xy = hover_xy = (offset_xy + slot_xy)).
        Rotation already locked from the prior fix.
        """
        wrench = (0.0, 0.0, -self.approach_fz_N, 0.0, 0.0, 0.0)
        sel = (False, False, True, False, False, False)  # XY LOCKED, Z compliant, rotation LOCKED
        return wrench, self.gain_approach, self.damping_approach, sel

    def _partcenter_torque_correction(self, Fx: float, Fy: float, Fz: float) -> tuple[float, float, float]:
        """[Disabled — kept for reference] Compute (Tx, Ty, Tz) correction so
        wrench at TCP has same effect at PART CENTER. Tested in v77/v78 and
        produced destabilizing cross-coupling. Not currently called."""
        rx_bl = -self._r_partcenter[0]
        ry_bl = -self._r_partcenter[1]
        rz_bl =  self._r_partcenter[2]
        Tx = ry_bl * Fz - rz_bl * Fy
        Ty = rz_bl * Fx - rx_bl * Fz
        Tz = rx_bl * Fy - ry_bl * Fx
        return Tx, Ty, Tz

    def sensed_T_at_partcenter(self, F_sensed_base: tuple[float, float, float],
                                T_sensed_base: tuple[float, float, float]) -> tuple[float, float, float]:
        """Transform sensed wrench (F, T) at TCP to torque about part center.

        Operator's principled-fix interpretation: when peg contacts rim, the
        rim reaction force F creates a moment about TCP that the sensor
        reports as T_TCP. The MEANINGFUL torque for analysis is T about the
        part center (where the part actually pivots), not T about TCP.

        Math: T_partcenter = T_TCP + (TCP - PartCenter) × F = T_TCP + (-r) × F
        where r = PartCenter - TCP.

        Frame: F, T sensed in BASE frame. r in WORLD = BASE here. Both
        compatible. Returns T in base frame.
        """
        rx, ry, rz = self._r_partcenter
        Fx, Fy, Fz = F_sensed_base
        Tx_T, Ty_T, Tz_T = T_sensed_base
        # (-r) × F in base
        T_pc_x = Tx_T + (-ry * Fz - (-rz) * Fy)
        T_pc_y = Ty_T + (-rz * Fx - (-rx) * Fz)
        T_pc_z = Tz_T + (-rx * Fy - (-ry) * Fx)
        return T_pc_x, T_pc_y, T_pc_z

    def _entry_settle_wrench(self, t: float) -> tuple[tuple, float, float, tuple]:
        """ENTRY_SETTLE wrench: gentle Fz down, all force-controlled."""
        wrench = (0.0, 0.0, -self.entry_settle_fz_N, 0.0, 0.0, 0.0)
        sel = (True, True, True, False, False, False)  # rotation LOCKED — prismatic peg, no rotational compliance needed (2026-05-06)
        return wrench, self.entry_settle_gain, self.entry_settle_damping, sel

    def _insert_wrench(self, t: float) -> tuple[tuple, float, float, tuple]:
        """INSERT wrench: Fz down + optional rocking, all force-controlled."""
        if self.insert_start_t is None:
            self.insert_start_t = t
        t_in = t - self.insert_start_t
        Tx = 0.0
        Ty = 0.0
        Tz = 0.0
        if self.insert_tilt_T_Nm > 0.0:
            theta = 2.0 * math.pi * self.insert_tilt_freq_hz * t_in
            Tx = self.insert_tilt_T_Nm * math.cos(theta)
            Ty = self.insert_tilt_T_Nm * math.sin(theta)
        if self.insert_tz_T_Nm > 0.0:
            tz_theta = 2.0 * math.pi * self.insert_tz_freq_hz * t_in
            Tz = self.insert_tz_T_Nm * math.sin(tz_theta)
        wrench = (0.0, 0.0, -self.insert_fz_N, Tx, Ty, Tz)
        sel = (True, True, True, False, False, False)  # rotation LOCKED — prismatic peg, no rotational compliance needed (2026-05-06)
        return wrench, self.gain_insert, self.damping_insert, sel

    def _find_hole_directed_wrench(self, t: float, tcp_xy: tuple[float, float]) -> tuple[tuple, float, float, tuple, dict]:
        """Iter-2 (post-H101): distance-scaled P-controller toward seat with near-seat dwell.

        Replaces H101's rotating-fixed-magnitude design after empirical observation
        (insert_u_orange_20260505_201537):
          - 5N rotating: TCP overshot seat (got within 6.6mm at t+7.7s, then drifted to 17.7mm)
          - GOLD operator R=0.88 (fixed direction); rotation pushed peg AWAY after first approach
          - GOLD median cmd_F_lat = 1.76N, p75 = 5.00N — magnitude scaled with distance, not constant

        New control law:
          - dist > scaling_zone_mm:   F = F_max * unit(seat-tcp)             # full push toward seat
          - near_dwell < dist < scaling_zone:  F = F_max * unit * (dist/scaling) * 0.7   # P-scaled approach
          - dist < near_dwell_mm:     F = 0  → let cmd_fz drive peg through chamfer (operator-style settle)

        Backing invariants: I001, I003, I004, I005 plus the new GOLD-vs-H101 diff.
        """
        # Iter-10: prefer online-refined seat estimate if we've latched one (chamfer engaged at xy=...)
        if self._seat_xy_refined is not None:
            seat_xy = self._seat_xy_refined
        elif self.spiral_origin_override is not None:
            seat_xy = (float(self.spiral_origin_override[0]),
                       float(self.spiral_origin_override[1]))
        elif self.predicted_tcp_xy is not None:
            seat_xy = (float(self.predicted_tcp_xy[0]),
                       float(self.predicted_tcp_xy[1]))
        else:
            return self._find_hole_wrench_spiral(t, tcp_xy)

        # Iter-10 (A2+A3): orientation feedback — read tilt + tilt direction
        # self._tilt_deg is computed in update() at the start of each tick;
        # self._tilt_err_world = (-ee_zy, +ee_zx, 0.0) is the tilt axis in world (xy)
        tilt_deg = getattr(self, "_tilt_deg", 0.0)
        tilt_err = getattr(self, "_tilt_err_world", (0.0, 0.0, 0.0))
        tilt_axis_xy_mag = math.hypot(tilt_err[0], tilt_err[1])

        # Maintain rolling tilt history (last find_hole_tilt_history_window_s seconds)
        self._tilt_history.append((t, tilt_deg))
        cutoff = t - self.find_hole_tilt_history_window_s
        self._tilt_history = [(tt, td) for (tt, td) in self._tilt_history if tt >= cutoff]
        # Track running peak
        if tilt_deg > self._tilt_running_peak_deg:
            self._tilt_running_peak_deg = tilt_deg
            self._tilt_running_peak_t = t

        # Detect "high tilt" sustained state — peg wedged at rim edge
        is_high_tilt = tilt_deg > self.find_hole_tilt_tolerance_deg
        if is_high_tilt:
            if self._tilt_high_first_t is None:
                self._tilt_high_first_t = t
        else:
            self._tilt_high_first_t = None

        # Detect chamfer-engagement (sustained drop from peak)
        relaxation = self._tilt_running_peak_deg - tilt_deg
        if (self._tilt_running_peak_deg > self.find_hole_tilt_tolerance_deg - 0.5
            and relaxation >= self.find_hole_tilt_relax_min_deg):
            if self._tilt_relax_first_t is None:
                self._tilt_relax_first_t = t
            elif (t - self._tilt_relax_first_t) >= self.find_hole_tilt_relax_sustain_s:
                # Latch: chamfer engaged at this xy — refine the seat estimate
                if not self._chamfer_engaged:
                    self._chamfer_engaged = True
                    self._seat_xy_refined = (float(tcp_xy[0]), float(tcp_xy[1]))
        else:
            self._tilt_relax_first_t = None

        # Distance to seat
        dx = seat_xy[0] - tcp_xy[0]
        dy = seat_xy[1] - tcp_xy[1]
        dist_m = math.hypot(dx, dy)

        near_dwell_m = self.find_hole_directed_near_dwell_mm * 1e-3
        scaling_m   = self.find_hole_directed_scaling_zone_mm * 1e-3
        F_max = self.find_hole_directed_F_N

        # Iter-7 structural change: pulse pattern (push/release cycle) to emulate
        # operator's natural grip-release. From iter-6 telemetry diff vs GOLD:
        #   - GOLD |T_lat| relaxed at close-pass: 0.184 → 0.032 Nm (operator's grip eased)
        #   - FAIL iter-6 |T_lat| stayed at 0.1 Nm — peg never released to engage chamfer
        #   - GOLD's fz_t went NEGATIVE at t+10s (peg fell into slot)
        #   - FAIL iter-6 fz_t stayed at +8.5N — peg stayed pinned on rim
        # Mechanism: brief release of lateral force lets peg's potential energy (cmd_fz=-9N)
        # drive it through chamfer geometry. Continuous push keeps peg pinned against rim.
        deadband_m = self.find_hole_directed_dir_deadband_mm * 1e-3

        # Initialize cycle-start time (set ONCE when first entering FIND_HOLE)
        if self._find_hole_first_t is None:
            self._find_hole_first_t = t
        t_in_phase = t - self._find_hole_first_t

        period = max(self.find_hole_directed_pulse_period_s, 0.1)
        release_s = max(self.find_hole_directed_pulse_release_s, 0.0)
        push_s = period - release_s
        in_release = (t_in_phase % period) > push_s

        # Compute direction (deadband-latched) regardless of pulse phase
        if dist_m < deadband_m:
            if self._last_stable_unit == (0.0, 0.0):
                self._last_stable_unit = (0.0, -1.0)
            ux, uy = self._last_stable_unit
        else:
            ux, uy = dx / dist_m, dy / dist_m
            self._last_stable_unit = (ux, uy)

        # Iter-10: tilt-direction OVERRIDE.
        # When tilt > tolerance sustained → peg is wedged at rim edge → tilt direction
        # IS the position-error signal (tcp_top tilts +X means peg bottom held back at -X
        # → move TCP +X to slide peg past the pinning edge).
        # When tilt is in normal envelope → push toward seat estimate (with deadband).
        tilt_override_active = (
            self._tilt_high_first_t is not None
            and (t - self._tilt_high_first_t) >= self.find_hole_tilt_high_sustain_s
            and tilt_axis_xy_mag > 1e-4
        )

        if in_release:
            Fx, Fy, zone = 0.0, 0.0, "PULSE_RELEASE"
            fz_cmd = -self.find_hole_directed_release_fz_N
        elif tilt_override_active:
            # Push in the tilt direction to slide peg past the pinning rim edge
            txn = tilt_err[0] / tilt_axis_xy_mag
            tyn = tilt_err[1] / tilt_axis_xy_mag
            # Negate per the wrapper's sign convention (Fx_b = -kp * err_x_base)
            Fx, Fy, zone = -F_max * txn, -F_max * tyn, f"TILT_OVERRIDE_{tilt_deg:.1f}deg"
            fz_cmd = -self.find_hole_fz_N
        else:
            Fx, Fy, zone = -F_max * ux, -F_max * uy, "PULSE_PUSH"
            fz_cmd = -self.find_hole_fz_N

        # Compat: keep _spiral_last_t / _spiral_origin_xy initialized for ENTRY_SETTLE checks
        if self._spiral_last_t is None:
            self._spiral_last_t = t
            self._spiral_origin_xy = (float(tcp_xy[0]), float(tcp_xy[1]))

        wrench = (Fx, Fy, fz_cmd, 0.0, 0.0, 0.0)
        sel = (True, True, True, False, False, False)  # rotation LOCKED — prismatic peg, no rotational compliance needed (2026-05-06)
        info = {
            "spiral_radius_m": 0.0,
            "spiral_theta_rad": 0.0,
            "spiral_target_xy_m": seat_xy,
            "err_xy_base_mm": (dx * 1000.0, dy * 1000.0),
            "F_lat_cmd_xy_N": (Fx, Fy),
            "directed_seat_xy_m": seat_xy,
            "directed_dist_to_seat_mm": dist_m * 1000.0,
            "directed_zone": zone,
            "directed_fz_cmd": fz_cmd,
        }
        return wrench, self.gain_find_hole, self.damping_find_hole, sel, info

    def _find_hole_wrench_spiral(self, t: float, tcp_xy: tuple[float, float]) -> tuple[tuple, float, float, tuple, dict]:
        """Original spiral PD implementation — kept as a fallback path when directed sweep has no seat prior."""
        return self._find_hole_wrench_impl(t, tcp_xy)

    def _find_hole_wrench(self, t: float, tcp_xy: tuple[float, float]) -> tuple[tuple, float, float, tuple, dict]:
        """Dispatcher: directed sweep (H101) if enabled, else original spiral."""
        if self.find_hole_use_directed_sweep:
            return self._find_hole_directed_wrench(t, tcp_xy)
        return self._find_hole_wrench_impl(t, tcp_xy)

    def _find_hole_wrench_impl(self, t: float, tcp_xy: tuple[float, float]) -> tuple[tuple, float, float, tuple, dict]:
        """FIND_HOLE wrench: PD-derived spiral xy + Fz. Stankowski Archimedean."""
        # Initialize spiral state on first tick
        if self._spiral_origin_xy is None:
            if self.spiral_origin_override is not None:
                # Use known-good xy (e.g. hole_xy_prior) as spiral center.
                # This gives the spiral a head start vs starting from random
                # peg-contact xy.
                self._spiral_origin_xy = (float(self.spiral_origin_override[0]),
                                          float(self.spiral_origin_override[1]))
            else:
                self._spiral_origin_xy = (float(tcp_xy[0]), float(tcp_xy[1]))
            self._spiral_theta = 0.0
            self._spiral_last_t = t

        # Advance theta
        dt = max(t - (self._spiral_last_t or t), 1e-3)
        self._spiral_last_t = t
        theta_dot = (2.0 * math.pi * self.spiral_v_m_s) / max(
            self.spiral_pitch_m * math.sqrt(1.0 + self._spiral_theta ** 2), 1e-6)
        self._spiral_theta += theta_dot * dt
        radius = (self.spiral_pitch_m / (2.0 * math.pi)) * self._spiral_theta

        # Compute desired EE xy from origin + spiral offset
        desired_x = self._spiral_origin_xy[0] + radius * math.cos(self._spiral_theta)
        desired_y = self._spiral_origin_xy[1] + radius * math.sin(self._spiral_theta)

        # PD-derived lateral force (in BASE_LINK convention — wrapper's flip
        # handles base ↔ base_link). tcp_xy from /tcp_pose_broadcaster is in
        # `base`; we negate err to express in base_link.
        err_x_baselink = -(desired_x - tcp_xy[0])
        err_y_baselink = -(desired_y - tcp_xy[1])
        Fx_pd = self.spiral_kp_N_per_m * err_x_baselink
        Fy_pd = self.spiral_kp_N_per_m * err_y_baselink
        F_pd_mag = math.hypot(Fx_pd, Fy_pd)
        if F_pd_mag > self.spiral_F_max_N:
            Fx_pd = Fx_pd * self.spiral_F_max_N / F_pd_mag
            Fy_pd = Fy_pd * self.spiral_F_max_N / F_pd_mag

        # === Recovery leash toward predicted_tcp_xy (per GPT review) ===
        # If peg drifts past r_free from predicted hole xy (CAD-derived),
        # add a soft pull-back force. Smooth ramp r_free → r_full, capped
        # at F_max. ZERO inside r_free so legitimate spiral search isn't
        # penalized. NOT applied during ENTRY_SETTLE (per GPT rec —
        # would fight chamfer self-alignment).
        Fx_rec = 0.0
        Fy_rec = 0.0
        recovery_dist_mm = 0.0
        if self.predicted_tcp_xy is not None:
            dx_pred_base = self.predicted_tcp_xy[0] - tcp_xy[0]
            dy_pred_base = self.predicted_tcp_xy[1] - tcp_xy[1]
            dist_to_pred = math.hypot(dx_pred_base, dy_pred_base)
            recovery_dist_mm = dist_to_pred * 1000
            if dist_to_pred > self.recovery_r_free_m:
                excess = dist_to_pred - self.recovery_r_free_m
                ramp = min(1.0, excess / max(
                    self.recovery_r_full_m - self.recovery_r_free_m, 1e-6))
                rec_mag = self.recovery_F_max_N * ramp
                rec_dir_x_base = dx_pred_base / dist_to_pred
                rec_dir_y_base = dy_pred_base / dist_to_pred
                # Convert to base_link convention (X/Y sign flip)
                Fx_rec = rec_mag * (-rec_dir_x_base)
                Fy_rec = rec_mag * (-rec_dir_y_base)

        # Compose: spiral PD + recovery leash
        Fx = Fx_pd + Fx_rec
        Fy = Fy_pd + Fy_rec

        # Optional tilt rocking + Tz wiggle for U-shape yaw alignment.
        # The Tz oscillation rotates the EE about the grasp point, which
        # SWEEPS the peg-tip through different yaw positions (since peg is
        # offset from grasp). For a U-shape peg, this is critical: spiral
        # alone covers xy; Tz wiggle covers the rotational DOF.
        Tx, Ty, Tz = 0.0, 0.0, 0.0
        if self._find_hole_start_t is not None:
            t_in = t - self._find_hole_start_t
            if self.find_hole_tilt_T_Nm > 0.0:
                tilt_theta = 2.0 * math.pi * self.find_hole_tilt_freq_hz * t_in
                Tx = self.find_hole_tilt_T_Nm * math.cos(tilt_theta)
                Ty = self.find_hole_tilt_T_Nm * math.sin(tilt_theta)
            if self.find_hole_tz_T_Nm > 0.0:
                tz_theta = 2.0 * math.pi * self.find_hole_tz_freq_hz * t_in
                Tz = self.find_hole_tz_T_Nm * math.sin(tz_theta)

        # All force-controlled (compliant in all DOFs, like operator hand).
        # Spiral Fx/Fy is intentionally GENTLE (kp=1500, F_max=10) so the
        # peg can self-align via chamfer geometry under admittance.
        wrench = (Fx, Fy, -self.find_hole_fz_N, Tx, Ty, Tz)
        sel = (True, True, True, False, False, False)  # rotation LOCKED — prismatic peg, no rotational compliance needed (2026-05-06)
        info = {
            "spiral_radius_m": radius,
            "spiral_theta_rad": self._spiral_theta,
            "spiral_target_xy_m": (desired_x, desired_y),
            "err_xy_base_mm": ((desired_x - tcp_xy[0]) * 1000, (desired_y - tcp_xy[1]) * 1000),
        }
        return wrench, self.gain_find_hole, self.damping_find_hole, sel, info

    # ---------- Main update ----------------------------------------------
    def update(self, t: float, tcp_x: float, tcp_y: float, tcp_z: float,
               fz_baselink: float,
               F_lat_baselink: tuple[float, float] = (0.0, 0.0),
               T_lat_baselink: tuple[float, float] = (0.0, 0.0),
               tcp_quat_xyzw: tuple[float, float, float, float] = (0.0, -1.0, 0.0, 0.0)) -> FSMAction:
        """Advance FSM one tick. Returns the action to take this tick.

        Signals:
            tcp_z          current EE z (used for surface_z + drop detection)
            fz_baselink    smoothed for contact + sustained-push checks
            F_lat_baselink (Fx,Fy) — for partial-engagement detection
                           (high F_lat = peg pushing on rim wall)
            T_lat_baselink (Tx,Ty) — for tilt detection
                           (high |T| = peg not flush with slot walls)
            tcp_quat_xyzw  EE orientation. Used to detect "one prong in / one
                           on rim" tilt that pure F/T cannot — the part is
                           geometrically tilted because grasp is offset from
                           part center, so partial engagement shows as EE quat
                           drift from canonical face-down. Operator-demo data
                           (60 successful inserts) shows max tilt during
                           descent ≤ 1.4° for u_orange, ≤ 6° for u_brown.
                           Gates use 3° threshold (~2× u_orange demo max).
        """
        action = FSMAction()
        # H101: cache F_lat_baselink so directed-sweep can read it without changing _find_hole_wrench signature
        self._last_F_lat_baselink = (float(F_lat_baselink[0]), float(F_lat_baselink[1]))
        # SearchDirector reads base-frame torques to compute r_cop
        self._last_T_lat_baselink = (float(T_lat_baselink[0]), float(T_lat_baselink[1]))
        # === Compute EE Z-axis in world (yaw-invariant) for orientation feedback ===
        # ee_z_world = (2*qx*qz + 2*qy*qw, 2*qy*qz - 2*qx*qw, 1 - 2*(qx² + qy²))
        # For canonical face-down: ee_z = (0, 0, -1).
        # The tilt error vector = ee_z × desired_ee_z (= world -Z).
        # Used for orientation-feedback control (re-leveling during RETRACT and
        # WEDGE_RECOVERY).
        _qx, _qy, _qz, _qw = tcp_quat_xyzw
        ee_zx = 2 * (_qx*_qz + _qy*_qw)
        ee_zy = 2 * (_qy*_qz - _qx*_qw)
        # Tilt error in world frame: cross(ee_z, world_-Z) = (ee_zy * -1 - 0, 0 - ee_zx * -1, 0)
        # = (-ee_zy, +ee_zx, 0). For small tilt, magnitude ≈ sin(tilt) ≈ tilt_rad.
        self._tilt_err_world = (-ee_zy, +ee_zx, 0.0)

        # === Update DYNAMIC r vector (TCP→part_center) from current EE pose ===
        # The held part stays at fixed orientation in EE frame (gripper holds
        # rigidly). So object_origin_in_EE is constant. But r in world frame
        # = R(EE_in_world) × object_origin_in_EE rotates with EE drift. For
        # 5° tilt the X-component of r can swing ±21mm (vs canonical -27mm).
        # Dynamic r is essential when controlling actively against tilt.
        if self._object_origin_in_EE is not None:
            qx, qy, qz, qw = tcp_quat_xyzw
            ox, oy, oz = self._object_origin_in_EE
            # Rotate vector by quaternion: v' = q * v * q^-1, computed via:
            # v' = v + 2*qw*(q_vec × v) + 2*q_vec × (q_vec × v)
            # qvec = (qx, qy, qz)
            # q_vec × v
            cx = qy*oz - qz*oy
            cy = qz*ox - qx*oz
            cz = qx*oy - qy*ox
            # q_vec × (q_vec × v)
            ccx = qy*cz - qz*cy
            ccy = qz*cx - qx*cz
            ccz = qx*cy - qy*cx
            self._r_partcenter = (
                ox + 2*qw*cx + 2*ccx,
                oy + 2*qw*cy + 2*ccy,
                oz + 2*qw*cz + 2*ccz,
            )

        # === Transform sensed wrench T from TCP to PART CENTER ===
        # Operator's principled-fix interpretation: the sensor reports T at
        # TCP (= grasp point). The MEANINGFUL torque for analyzing peg tilt
        # is T about the part center. T_partcenter = T_TCP + (TCP - PartCenter)
        # × F = T_TCP + (-r) × F, where r = PartCenter - TCP (now dynamic).
        rx_pc, ry_pc, rz_pc = self._r_partcenter
        Fx_b, Fy_b, Fz_b = F_lat_baselink[0], F_lat_baselink[1], fz_baselink
        T_pc_x = T_lat_baselink[0] + (-ry_pc * Fz_b - (-rz_pc) * Fy_b)
        T_pc_y = T_lat_baselink[1] + (-rz_pc * Fx_b - (-rx_pc) * Fz_b)
        self._T_partcenter = (T_pc_x, T_pc_y)

        # Compute YAW-INVARIANT tilt of peg axis from world -Z (face-down).
        # GPT review caught: 2*acos(|q·q_ref|) (full quat distance) counts yaw,
        # so the tilt gate would trip on pure Tz wiggle. The geometric
        # quantity we actually want is the angle between the EE's local
        # +Z axis and world -Z (face-down direction). For quat (qx,qy,qz,qw):
        #   ee_z_in_world.z = 1 - 2*(qx² + qy²)        (yaw-invariant)
        # tilt_deg = acos(-ee_z_in_world.z)  in degrees
        # (canonical face-down: qx=qy=0 → ee_z.z=-1 → tilt=0; pure yaw rotates
        #  qz/qw but keeps qx=qy=0, so ee_z.z stays -1, tilt stays 0.)
        _qx, _qy, _qz, _qw = tcp_quat_xyzw
        _ee_zz = 1.0 - 2.0 * (_qx*_qx + _qy*_qy)
        _ee_zz = max(-1.0, min(1.0, _ee_zz))
        self._tilt_deg = math.degrees(math.acos(-_ee_zz))
        # Track for engagement-detection (used by FIND_HOLE → INSERT transition).
        # When peg is FULLY engaged in slot: F_lat low (no rim push), |T_lat|
        # low (no tilt). When PARTIALLY engaged: either is high.
        T_lat_inst = math.hypot(T_lat_baselink[0], T_lat_baselink[1])
        self._latest_T_lat_mag = T_lat_inst

        fz_smoothed = self._smooth_fz(t, fz_baselink)
        F_lat_inst = math.hypot(F_lat_baselink[0], F_lat_baselink[1])

        # Buffer + smooth F_lat for safety check. Impulsive collisions (peg-rim
        # prong catching for 11ms) can spike to 30N+ instantaneously without
        # being a real jam state. Only abort if force is sustained over the
        # smoothing window — that's what indicates the algorithm is actually
        # pushing peg into a wall.
        self._F_lat_buffer.append((t, F_lat_inst))
        while self._F_lat_buffer and (t - self._F_lat_buffer[0][0]) > self.abort_F_lat_window_s:
            self._F_lat_buffer.popleft()
        F_lat_smoothed = (sum(v for _, v in self._F_lat_buffer) /
                          len(self._F_lat_buffer)) if self._F_lat_buffer else F_lat_inst
        if F_lat_smoothed > self.abort_F_lat_N:
            action.kind = "exit_abort"
            action.abort_reason = (
                f"F_lat sustained {F_lat_smoothed:.1f}N > {self.abort_F_lat_N:.0f}N over "
                f"{self.abort_F_lat_window_s*1000:.0f}ms in {self.state} (inst peak {F_lat_inst:.1f}N)"
            )
            return action

        # === STATE-INDEPENDENT SEAT DETECTOR ===
        # Runs every tick regardless of FSM sub-state. Two paths:
        #   (1) ABSOLUTE (preferred): tcp_z near predicted_tcp_z (CAD seat z),
        #       motion stopped, tilt low, sustained. Robust to multi-contact
        #       insertions because it doesn't depend on surface_z (which
        #       re-latches on every new Contact event).
        #   (2) RELATIVE (fallback when predicted_tcp_z is None): z_drop from
        #       FIRST contact ≥ engaged_z_drop_dominant_m. Original behavior;
        #       only correct for single-contact insertions.
        # Include SEARCH and APPROACH so a peg that drops straight through
        # the chamfer without needing lateral search is detected as seated
        # immediately (verified case 2026-05-06 u_brown: peg fell to slot
        # bottom during APPROACH, then briefly bumped fz>3 entering SEARCH,
        # then aborted on lateral_stall — but tcp_z was 1.7mm below predicted
        # seat with motion stopped throughout).
        if self.state in (self.APPROACH, self.FIND_HOLE, self.ENTRY_SETTLE,
                          self.WEDGE_RECOVERY, self.INSERT,
                          self.SEARCH,
                          self.INSERT_DESCENT):
            v_z = self._z_velocity(t, tcp_z, self.insert_motion_window_s)
            speed_z = abs(v_z) if v_z is not None else float('inf')
            motion_ok = (speed_z < self.insert_motion_thresh_m_s and
                          self._tilt_deg < self.insert_tilt_abort_deg)

            # Path (1): absolute z near predicted seat
            absolute_at_target = False
            if self.predicted_tcp_z is not None:
                dz_abs = abs(tcp_z - self.predicted_tcp_z)
                absolute_at_target = dz_abs <= self.at_target_z_tol_m

            # Path (2): relative descent fallback
            relative_seated = False
            z_drop_global = float('inf')
            if self.surface_z is not None:
                z_drop_global = self.surface_z - tcp_z
                relative_seated = z_drop_global >= self.engaged_z_drop_dominant_m

            if (absolute_at_target or relative_seated) and motion_ok:
                if self._global_seat_first_t is None:
                    self._global_seat_first_t = t
                elif (t - self._global_seat_first_t) >= 1.0:
                    self.state = self.DONE
                    action.kind = "exit_done"
                    action.transitioned = True
                    action.new_state = self.DONE
                    if self.predicted_tcp_z is not None:
                        dz = (tcp_z - self.predicted_tcp_z) * 1000
                        criterion = (f"ABSOLUTE: tcp_z={tcp_z:.4f} predicted={self.predicted_tcp_z:.4f} "
                                     f"Δ={dz:+.2f}mm (tol={self.at_target_z_tol_m*1000:.0f}mm)")
                    else:
                        criterion = (f"RELATIVE z_drop: {z_drop_global*1000:.1f}mm "
                                     f"(tol={self.engaged_z_drop_dominant_m*1000:.0f}mm)")
                    action.transition_msg = (
                        f"GLOBAL SEAT detector: {criterion}, "
                        f"|dz/dt|={speed_z*1000:.3f}mm/s tilt={self._tilt_deg:.2f}° "
                        f"sustained 1.0s (state was {self.state})"
                    )
                    return action
            else:
                self._global_seat_first_t = None

        # State logic
        if self.state == self.APPROACH:
            return self._tick_approach(t, tcp_x, tcp_y, tcp_z, fz_smoothed, action)
        elif self.state == self.FIND_HOLE:
            return self._tick_find_hole(t, tcp_x, tcp_y, tcp_z, fz_smoothed,
                                         F_lat_baselink, action)
        elif self.state == self.ENTRY_SETTLE:
            return self._tick_entry_settle(t, tcp_x, tcp_y, tcp_z, fz_smoothed,
                                            F_lat_baselink, T_lat_baselink, action)
        elif self.state == self.WEDGE_RECOVERY:
            return self._tick_wedge_recovery(t, tcp_x, tcp_y, tcp_z, fz_smoothed,
                                              F_lat_baselink, T_lat_baselink, action)
        elif self.state == self.INSERT:
            return self._tick_insert(t, tcp_x, tcp_y, tcp_z, fz_smoothed, action)
        elif self.state == self.GUIDED:
            return self._tick_guided(t, tcp_x, tcp_y, tcp_z, fz_smoothed, action)
        elif self.state == self.SEARCH:
            return self._tick_search(t, tcp_x, tcp_y, tcp_z, fz_smoothed, action)
        elif self.state == self.INSERT_DESCENT:
            return self._tick_insert_descent(t, tcp_x, tcp_y, tcp_z, fz_smoothed, action)
        elif self.state == self.DONE:
            action.kind = "exit_done"
            return action
        elif self.state == self.SAFETY_RETRACT:
            action.kind = "exit_abort"
            action.abort_reason = "safety_retract reached"
            return action
        else:
            action.kind = "exit_abort"
            action.abort_reason = f"unknown state {self.state}"
            return action

    # ---------- State ticks ----------------------------------------------
    def _tick_guided(self, t, tcp_x, tcp_y, tcp_z, fz_smoothed, action):
        """GUIDED state: operator drags EE laterally on rim until peg above hole.
        sel=(T,T,F,F,F,F) — XY compliant, Z LOCKED, rotation LOCKED.
        Wrench=(0,0,0,0,0,0) — no commanded force, pure compliance.
        Loose gain+damping (operator-friendly) per gimbal_mode_test validation.
        Exit: wrapper sets self._guided_hole_marked=True on SIGUSR1 → capture
        TCP as hole_observed, transition to INSERT_DESCENT.
        """
        # First-tick entry: arm gimbal force-mode wrench
        if not self._guided_entered:
            self._guided_entered = True
            wrench = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
            sel = (True, True, False, False, False, False)
            action.new_wrench = True
            action.wrench_baselink = wrench
            action.selection_vector = sel
            action.gain = self.guided_gain
            action.damping = self.guided_damping
            action.state = self.GUIDED
            action.msg = (f"GUIDED active: drag peg laterally; Z+orientation "
                          f"locked. Send SIGUSR1 when peg above hole.")
            return action

        # v4 Found Hole predicate (parallel to SIGUSR1). Logs on first fire;
        # if v4_autofire enabled (stage 3b), also marks hole.
        v4_fired_now = self.found_hole_detector.update(t, fz_smoothed)
        if v4_fired_now:
            self.v4_predicate_fire = {
                "t_s": float(t),
                "xy_m": [float(tcp_x), float(tcp_y)],
                "z_m": float(tcp_z),
                "fz_smoothed_at_fire_N": float(fz_smoothed),
                "abs_fz_smoothed_at_fire_N": float(abs(fz_smoothed)),
                "thresholds": {
                    "rim_high_N": self.found_hole_detector.rim_high,
                    "rim_low_N":  self.found_hole_detector.rim_low,
                    "off_sustain_s":   self.found_hole_detector.off_sustain_s,
                    "recent_window_s": self.found_hole_detector.recent_window_s,
                },
                "autofired": bool(self.v4_autofire),
            }
            action.msg = (f"v4 Found-Hole predicate FIRED at t={t:.2f}s "
                          f"xy=({tcp_x:+.4f},{tcp_y:+.4f}) "
                          f"|fz|={abs(fz_smoothed):.2f}N "
                          f"(autofire={self.v4_autofire})")
            if self.v4_autofire and not self._guided_hole_marked:
                self._guided_hole_marked = True

        # Wait for hole-mark signal (SIGUSR1 OR v4 autofire)
        if self._guided_hole_marked:
            self.hole_observed_xy = (float(tcp_x), float(tcp_y))
            self.hole_observed_z  = float(tcp_z)
            self.hole_observed_t  = float(t)
            self.state = self.INSERT_DESCENT
            # Source attribution: v4 if it fired and autofire is on AND v4
            # fired at or before this tick; SIGUSR1 otherwise.
            v4_was_trigger = (self.v4_autofire
                              and self.v4_predicate_fire is not None
                              and self.v4_predicate_fire["t_s"] <= t)
            trigger_label = "v4 predicate" if v4_was_trigger else "SIGUSR1"
            action.transitioned = True
            action.new_state = self.INSERT_DESCENT
            action.transition_msg = (
                f"hole_observed_xy=({self.hole_observed_xy[0]:+.4f},"
                f"{self.hole_observed_xy[1]:+.4f}) at t={t:.2f}s "
                f"({trigger_label}) → INSERT_DESCENT"
            )
            action.state = self.INSERT_DESCENT
            return action

        action.state = self.GUIDED
        return action

    def _tick_search(self, t, tcp_x, tcp_y, tcp_z, fz_smoothed, action):
        """SEARCH state: autonomous F/T-driven director replaces operator drag.

        Each tick:
          1. SearchDirector computes (Fx, Fy, Fz) from (fz_smoothed, Tx_base,
             Ty_base) using the −r_cop control law.
          2. Issue new wrench (rate-limited to search_cmd_period_s).
          3. Run v4 detector; on fire → SEARCH → INSERT_DESCENT.
          4. Timeout: ABORT after search_max_duration_s.

        Same selection vector as APPROACH/INSERT_DESCENT: X,Y,Z compliant,
        rotation locked.
        """
        if not self._search_entered:
            self._search_entered = True
            self._search_start_t = t
            # Seed spiral center at CAD-predicted seat. As of 2026-05-06,
            # primitives/shared/config.py:DEFAULT_BASE_POSITION incorporates
            # the empirical calibration. Bias correction is no longer applied
            # in the FSM (the prior u_orange-derived bias was contaminated by
            # off-center grasp, which does not generalize across objects).
            # Per-grasp offset (the actual unknown at runtime) is what the
            # spiral search must discover by sweeping outward from the predicted
            # seat. Verified against u_brown centered-grasp demo bias = (+1.5, -4) mm.
            if self.predicted_tcp_xy is not None:
                self.search_director.set_center(self.predicted_tcp_xy)
            else:
                self.search_director.set_center((tcp_x, tcp_y))
            Fx, Fy, Fz = self.search_director.update(t, fz_smoothed, tcp_x, tcp_y)
            wrench = (Fx, Fy, Fz, 0.0, 0.0, 0.0)
            sel = (True, True, True, False, False, False)
            action.new_wrench = True
            action.wrench_baselink = wrench
            action.selection_vector = sel
            action.gain = 0.5
            action.damping = 0.7
            action.state = self.SEARCH
            action.msg = (f"SEARCH active (spiral): "
                          f"center=({1000*tcp_x:+.1f},{1000*tcp_y:+.1f})mm "
                          f"r0={1000*self.search_director.r0:.1f}mm "
                          f"pitch={1000*self.search_director.pitch:.1f}mm "
                          f"v_s={1000*self.search_director.v_s:.1f}mm/s "
                          f"R_max={1000*self.search_director.R_max:.1f}mm "
                          f"Fmax={self.search_director.Fmax:.1f}N "
                          f"first_cmd=({Fx:+.2f},{Fy:+.2f},{Fz:+.2f})N")
            self._search_last_cmd_t = t
            return action

        # Timeout safety
        if t - self._search_start_t > self.search_max_duration_s:
            action.kind = "exit_abort"
            action.abort_reason = (
                f"SEARCH timeout {self.search_max_duration_s}s — peg never crossed rim "
                f"(v4 didn't fire). Likely r_cop direction reversed or peg stuck on rim corner.")
            return action

        # v4 detector — fires on rim-cross. SEARCH always autofires (v4 is the
        # only transition trigger out of SEARCH; SIGUSR1 is not used in
        # autonomous mode by convention).
        v4_fired_now = self.found_hole_detector.update(t, fz_smoothed)
        if v4_fired_now:
            self.v4_predicate_fire = {
                "t_s": float(t),
                "xy_m": [float(tcp_x), float(tcp_y)],
                "z_m": float(tcp_z),
                "fz_smoothed_at_fire_N": float(fz_smoothed),
                "abs_fz_smoothed_at_fire_N": float(abs(fz_smoothed)),
                "thresholds": {
                    "rim_high_N": self.found_hole_detector.rim_high,
                    "rim_low_N":  self.found_hole_detector.rim_low,
                    "off_sustain_s":   self.found_hole_detector.off_sustain_s,
                    "recent_window_s": self.found_hole_detector.recent_window_s,
                },
                "autofired": True,
                "from_state": self.SEARCH,
            }
            self.hole_observed_xy = (float(tcp_x), float(tcp_y))
            self.hole_observed_z  = float(tcp_z)
            self.hole_observed_t  = float(t)
            self.state = self.INSERT_DESCENT
            action.transitioned = True
            action.new_state = self.INSERT_DESCENT
            action.transition_msg = (
                f"v4 predicate FIRED in SEARCH at t={t:.2f}s "
                f"xy=({tcp_x:+.4f},{tcp_y:+.4f}) "
                f"|fz|={abs(fz_smoothed):.2f}N "
                f"after {t-self._search_start_t:.2f}s of search → INSERT_DESCENT"
            )
            action.state = self.INSERT_DESCENT
            return action

        # Per-tick spiral update (high-rate inner loop on the PD position-tracker)
        Fx, Fy, Fz = self.search_director.update(t, fz_smoothed, tcp_x, tcp_y)
        # Director self-aborted (spiral exhausted or stall)?
        if self.search_director.abort_reason is not None:
            action.kind = "exit_abort"
            action.abort_reason = (
                f"SEARCH director: {self.search_director.abort_reason} "
                f"after {t - self._search_start_t:.2f}s; tcp=({1000*tcp_x:+.1f},"
                f"{1000*tcp_y:+.1f})mm radius={1000*self.search_director.last_radius_m:.2f}mm"
            )
            return action

        # Rate-limited issue of wrench (every search_cmd_period_s)
        if (self._search_last_cmd_t is None
                or (t - self._search_last_cmd_t) >= self._search_cmd_period_s):
            wrench = (Fx, Fy, Fz, 0.0, 0.0, 0.0)
            sel = (True, True, True, False, False, False)
            action.new_wrench = True
            action.wrench_baselink = wrench
            action.selection_vector = sel
            action.gain = 0.5
            action.damping = 0.7
            self._search_last_cmd_t = t
            # Diagnostic per cmd cycle
            action.msg = (f"SEARCH t+{t-self._search_start_t:.1f}s "
                          f"r={1000*self.search_director.last_radius_m:.2f}mm "
                          f"cmd=({Fx:+.2f},{Fy:+.2f})N "
                          f"tcp=({1000*tcp_x:+.1f},{1000*tcp_y:+.1f})mm")

        action.state = self.SEARCH
        return action

    def _tick_insert_descent(self, t, tcp_x, tcp_y, tcp_z, fz_smoothed, action):
        """Post-GUIDED Z-descent at the operator-marked hole_xy.
        Same selection vector as APPROACH (XY locked, Z compliant, rotation
        locked). Robot descends straight down from current TCP into the slot.
        Termination: existing seat predicate (motion_stopped + tcp_z near
        predicted + descent_post_contact > insert_min_descent_m) — declared
        via the global seat detector that runs every tick before state dispatch.
        Or: SAFETY_RETRACT if F_lat overload, or operator SIGINT for manual end.
        """
        if not self._insert_descent_entered:
            self._insert_descent_entered = True
            wrench = (0.0, 0.0, -self.insert_fz_N, 0.0, 0.0, 0.0)
            sel = (False, False, True, False, False, False)  # XY locked, Z compliant, rotation locked
            action.new_wrench = True
            action.wrench_baselink = wrench
            action.selection_vector = sel
            action.gain = self.gain_insert
            action.damping = self.damping_insert
            action.state = self.INSERT_DESCENT
            action.msg = (f"INSERT_DESCENT: descending Z at xy="
                          f"({tcp_x:+.4f},{tcp_y:+.4f})")
            return action

        action.state = self.INSERT_DESCENT
        return action

    def mark_hole(self) -> None:
        """Wrapper calls this from its SIGUSR1 handler. Triggers GUIDED→INSERT_DESCENT
        on the next tick if currently in GUIDED state. No-op otherwise."""
        if self.state == self.GUIDED:
            self._guided_hole_marked = True

    def _tick_approach(self, t, tcp_x, tcp_y, tcp_z, fz_smoothed, action):
        if self._approach_start_t is None:
            self._approach_start_t = t

        # Approach timeout safety
        if t - self._approach_start_t > self.approach_max_duration_s:
            action.kind = "exit_abort"
            action.abort_reason = f"APPROACH timeout {self.approach_max_duration_s}s — never made contact"
            return action

        # Grace period: ignore contact-detection during the first
        # approach_grace_period_s of APPROACH. Lets force_mode_controller
        # startup transient + sensor noise settle before we start looking
        # for contact. Without this, ±5N noise during force-mode startup
        # can falsely trip the 3N contact threshold at hover height.
        in_grace = (t - self._approach_start_t) < self.approach_grace_period_s

        # Contact check: smoothed fz above threshold
        # In base frame, peg pushed UP by reaction = +Fz (positive)
        if not in_grace and fz_smoothed > self.contact_threshold_N:
            # === Sign-sanity gate (added 2026-05-05 after fold-mirror Stage A bug) ===
            # If override and predicted disagree by >5mm, the override is likely the
            # mirrored fold-equivalent and will steer peg AWAY from the actual slot.
            # Detect at contact moment when we have both signals available.
            if (self.spiral_origin_override is not None
                    and self.predicted_tcp_xy is not None):
                dx_mm = (self.spiral_origin_override[0] - self.predicted_tcp_xy[0]) * 1000.0
                dy_mm = (self.spiral_origin_override[1] - self.predicted_tcp_xy[1]) * 1000.0
                gap_mm = (dx_mm * dx_mm + dy_mm * dy_mm) ** 0.5
                if gap_mm > 5.0:
                    action.override_vs_predicted_gap_mm = gap_mm
                    action.override_vs_predicted_warning = (
                        f"hole_xy_prior override and CAD predicted_tcp_xy disagree by "
                        f"{gap_mm:.1f}mm (override=({self.spiral_origin_override[0]:+.4f},"
                        f"{self.spiral_origin_override[1]:+.4f}) vs predicted=("
                        f"{self.predicted_tcp_xy[0]:+.4f},{self.predicted_tcp_xy[1]:+.4f})). "
                        f"This is the fold-mirror signature — override likely points to the "
                        f"WRONG fold equivalent for this run's held_quat. Steering will go AWAY "
                        f"from the actual slot. See SKILL.md section 11."
                    )
            # Capture surface_z + surface_t at the contact moment (used by all paths)
            self.surface_z = float(tcp_z)
            self.surface_t = float(t)
            warning = getattr(action, 'override_vs_predicted_warning', None)

            # Routing options on Contact:
            #   1. autonomous_search=True  → SEARCH (active F/T-driven director)
            #   2. guided_mode=True        → GUIDED (operator-drag data collection)
            #   3. default                 → FIND_HOLE (legacy spiral, deprecated)
            if self.autonomous_search:
                self.state = self.SEARCH
                action.transitioned = True
                action.new_state = self.SEARCH
                action.transition_msg = (
                    f"surface_z={self.surface_z:.4f}m at t={t:.2f}s "
                    f"(fz_smoothed={fz_smoothed:.2f}N > {self.contact_threshold_N}) "
                    f"→ SEARCH (spiral PD: r0={1000*self.search_director.r0:.1f}mm "
                    f"pitch={1000*self.search_director.pitch:.1f}mm v_s={1000*self.search_director.v_s:.1f}mm/s "
                    f"R_max={1000*self.search_director.R_max:.1f}mm Fmax={self.search_director.Fmax:.1f}N "
                    f"F_press={self.search_director.F_press:.1f}N timeout={self.search_max_duration_s}s)"
                )
                action.state = self.SEARCH
                return action

            if self.guided_mode:
                self.state = self.GUIDED
                action.transitioned = True
                action.new_state = self.GUIDED
                action.transition_msg = (
                    f"surface_z={self.surface_z:.4f}m at t={t:.2f}s "
                    f"(fz_smoothed={fz_smoothed:.2f}N > {self.contact_threshold_N}) "
                    f"→ GUIDED (operator-drag); send SIGUSR1 to wrapper when peg is above hole"
                )
                # Trigger initial GUIDED wrench (set in _tick_guided on first entry)
                action.state = self.GUIDED
                return action

            # Default: APPROACH → FIND_HOLE (autonomous insertion path)
            self.state = self.FIND_HOLE
            self._find_hole_start_t = t
            action.transitioned = True
            action.new_state = self.FIND_HOLE
            action.transition_msg = (f"surface_z={self.surface_z:.4f}m at t={t:.2f}s "
                                     f"(fz_smoothed={fz_smoothed:.2f}N > {self.contact_threshold_N})")
            if warning:
                action.transition_msg += f" || SIGN-SANITY WARNING: {warning}"
            # H101: F_lat_baselink already cached on self at start of update() (line ~585)
            # Start FIND_HOLE wrench immediately (will fall through to next call)
            wrench, gain, damping, sel, _ = self._find_hole_wrench(t, (tcp_x, tcp_y))
            action.new_wrench = True
            action.wrench_baselink = wrench
            action.selection_vector = sel
            action.gain = gain
            action.damping = damping
            action.state = self.FIND_HOLE
            return action

        # Still descending — keep approach wrench
        wrench, gain, damping, sel = self._approach_wrench()
        # Only re-issue if not yet started
        if self._approach_start_t == t:
            action.new_wrench = True
            action.wrench_baselink = wrench
            action.selection_vector = sel
            action.gain = gain
            action.damping = damping
        action.state = self.APPROACH
        action.msg = f"approach: tcp_z={tcp_z:.4f} fz_smoothed={fz_smoothed:+.2f}N"
        return action

    def _tick_wedge_recovery(self, t, tcp_x, tcp_y, tcp_z, fz_smoothed,
                              F_lat_baselink, T_lat_baselink, action):
        """WEDGE_RECOVERY: counter-torque on full (Tx, Ty) vector.

        Per operator's directional-info insight: sensed (Tx, Ty) is a 2D vector
        encoding the tilt axis (not just side). Tx-only counter-torque can't
        flatten arbitrary tilts. Use full vector: cmd = -k * (Tx_filt, Ty_filt).

        First N attempts apply counter-torque with reduced Fz. If those fail
        (tilt persists), final attempt RETRACTS the EE +z to break the wedge
        contact, before returning to FIND_HOLE.
        """
        if self._wedge_start_t is None:
            self._wedge_start_t = t
        t_in_state = t - self._wedge_start_t

        # === ORIENTATION FEEDBACK CONTROL ===
        # Operator's principled insight: the right way to drive the part flat is
        # to command torque proportional to the actual tilt error vector, not
        # to react to sensed contact T. Tilt error vector = ee_z × world_-Z =
        # (-ee_zy, +ee_zx, 0) in world frame. To express in base_link, negate
        # x, y. Then T_cmd = -kp * tilt_err drives EE toward face-down.
        # tilt_err magnitude ≈ sin(tilt) ≈ tilt_rad for small angles.
        terr_world_x, terr_world_y, _ = self._tilt_err_world
        # Convert to base_link (X, Y inverted from world)
        terr_bl_x = -terr_world_x
        terr_bl_y = -terr_world_y
        # Counter-torque to re-level
        Tx_cmd_relevel = -self.relevel_kp_torque * terr_bl_x
        Ty_cmd_relevel = -self.relevel_kp_torque * terr_bl_y
        # Cap
        Tx_cmd_relevel = max(-self.wedge_cmd_Tx_max_Nm,
                             min(self.wedge_cmd_Tx_max_Nm, Tx_cmd_relevel))
        Ty_cmd_relevel = max(-self.wedge_cmd_Tx_max_Nm,
                             min(self.wedge_cmd_Tx_max_Nm, Ty_cmd_relevel))

        # If wedge attempts exhausted: RETRACT mode. Lift EE +z AND apply the
        # re-level torque (so peg comes back up FACE-DOWN, not still tilted).
        retract_mode = (self._wedge_attempts >= self.wedge_max_attempts)

        if retract_mode:
            # +Fz lift + corrective torque so we land back at canonical face-down
            wrench = (0.0, 0.0, self.wedge_retract_fz_N,
                      Tx_cmd_relevel, Ty_cmd_relevel, 0.0)
            cmd_msg = (f"RETRACT +Fz={self.wedge_retract_fz_N:.1f}N + relevel "
                       f"T=({Tx_cmd_relevel:+.3f},{Ty_cmd_relevel:+.3f})Nm")
        else:
            # Pure orientation feedback: drive tilt to zero with reduced Fz
            wrench = (0.0, 0.0, -self.wedge_fz_N,
                      Tx_cmd_relevel, Ty_cmd_relevel, 0.0)
            cmd_msg = (f"cmd_T=({Tx_cmd_relevel:+.3f},{Ty_cmd_relevel:+.3f})Nm "
                       f"Fz={self.wedge_fz_N}N")
        sel = (True, True, True, False, False, False)  # rotation LOCKED — prismatic peg, no rotational compliance needed (2026-05-06)

        # Successful unwedge: tilt below engagement gate (peg level enough).
        unwedged = (self._tilt_deg <= self.engaged_tilt_max_deg)

        # Retract finishes when peg is back above surface
        if retract_mode and self.surface_z is not None:
            if tcp_z >= self.surface_z - 0.0005:  # within 0.5mm of surface
                # Reset all engagement state — fresh start from above
                self.state = self.FIND_HOLE
                self._wedge_start_t = None
                self._wedge_attempts = 0  # reset for next cycle
                self._tx_lp_buffer.clear()
                self._ty_lp_buffer.clear()
                self._engaged_first_t = None
                self._F_lat_smooth_buffer.clear()
                self.hole_z = None
                self.hole_xy = None
                self._motion_stopped_first_t = None
                # Reset spiral state cleanly: theta + last_t (else dt blows up
                # on next tick) + start_t (else FIND_HOLE timeout fires
                # immediately with old elapsed time)
                self._spiral_theta = 0.0
                self._spiral_last_t = None
                self._find_hole_start_t = None
                self._entry_settle_start_t = None
                action.transitioned = True
                action.new_state = self.FIND_HOLE
                action.transition_msg = (
                    f"WEDGE_RECOVERY RETRACTED: tcp_z back to surface "
                    f"({tcp_z:.4f}m, surface_z={self.surface_z:.4f}). "
                    f"Wedge attempts reset; resuming spiral search."
                )
                # Re-issue find_hole wrench
                w_fh, g_fh, d_fh, s_fh, _ = self._find_hole_wrench(t, (tcp_x, tcp_y))
                action.new_wrench = True
                action.wrench_baselink = w_fh
                action.selection_vector = s_fh
                action.gain = g_fh
                action.damping = d_fh
                action.state = self.FIND_HOLE
                return action

        if unwedged or (not retract_mode and t_in_state >= self.wedge_duration_s):
            # Counter-torque attempt finished. Re-enter ENTRY_SETTLE.
            self.state = self.ENTRY_SETTLE
            self._entry_settle_start_t = t
            self._wedge_start_t = None
            self._tx_lp_buffer.clear()
            self._ty_lp_buffer.clear()
            self._engaged_first_t = None
            self._F_lat_smooth_buffer.clear()
            action.transitioned = True
            action.new_state = self.ENTRY_SETTLE
            z_drop_mm = (self.surface_z-tcp_z)*1000 if self.surface_z else 0
            action.transition_msg = (
                f"wedge_recovery {'CLEARED' if unwedged else 'TIMEOUT'}: "
                f"tilt={self._tilt_deg:.2f}° "
                f"z_drop={z_drop_mm:.2f}mm. Re-checking engagement."
            )
            wrench_es, gain_es, damping_es, sel_es = self._entry_settle_wrench(t)
            action.new_wrench = True
            action.wrench_baselink = wrench_es
            action.selection_vector = sel_es
            action.gain = gain_es
            action.damping = damping_es
            action.state = self.ENTRY_SETTLE
            return action

        # Continue (either counter-torque or retract)
        action.new_wrench = True
        action.wrench_baselink = wrench
        action.selection_vector = sel
        action.gain = 0.7
        action.damping = 0.5
        action.state = self.WEDGE_RECOVERY
        z_drop_mm = (self.surface_z-tcp_z)*1000 if self.surface_z else 0
        action.msg = (f"wedge_recovery [{'RETRACT' if retract_mode else 'RELEVEL'}]: "
                      f"{cmd_msg} tilt={self._tilt_deg:.2f}° "
                      f"z_drop={z_drop_mm:.2f}mm "
                      f"t={t_in_state:.2f}/{self.wedge_duration_s:.1f}s")
        return action

    def _tick_entry_settle(self, t, tcp_x, tcp_y, tcp_z, fz_smoothed,
                           F_lat_baselink, T_lat_baselink, action):
        """ENTRY_SETTLE: spiral stopped, settle then verify quasi-static low F/T.

        Per GPT review: this is the missing state from the prior architecture.
        Without it, FIND_HOLE → INSERT could fire when peg is wedged at first
        prong because we couldn't distinguish commanded-spiral-force from
        true contact-force. Here, with no commanded XY, F_lat/T_lat reflect
        ONLY contact reaction.

        Sequence:
          1. Wait `entry_settle_wait_s` (transient die-down — chamfer impact)
          2. Check sustained F_lat low + |T_lat| low + |v_xy| low
          3. If all hold for `entry_settle_verify_s` → INSERT
          4. If timeout exceeded without pass → BACK to FIND_HOLE
             (the candidate entry was a false positive: peg slid across rim
             noise, not a real chamfer entry)
        """
        if self._entry_settle_start_t is None:
            self._entry_settle_start_t = t
        t_in_state = t - self._entry_settle_start_t

        # Waiting period — let chamfer-impact transients die down
        if t_in_state < self.entry_settle_wait_s:
            action.state = self.ENTRY_SETTLE
            action.msg = (f"entry_settle: waiting transients ({t_in_state:.2f}s/"
                          f"{self.entry_settle_wait_s:.2f}s) tcp_z={tcp_z:.4f}")
            return action

        # Verify quasi-static engagement
        F_lat_inst = math.hypot(F_lat_baselink[0], F_lat_baselink[1])
        # Use T about PART CENTER (not TCP) per operator's principled-fix
        # insight. The MEANINGFUL torque for tilt analysis is moment of contact
        # force about part center, not about grasp/TCP. Part center is the
        # geometric pivot that determines whether peg is level or tilted.
        T_pc_x, T_pc_y = self._T_partcenter
        T_lat_inst = math.hypot(T_pc_x, T_pc_y)

        # Track XY velocity (peg motion in last 100ms)
        v_xy = self._z_velocity_xy(t, tcp_x, tcp_y, 0.1)
        v_xy_mag = math.hypot(v_xy[0], v_xy[1]) if v_xy is not None else float('inf')

        F_lat_low = F_lat_inst < self.engaged_F_lat_max_N
        T_lat_low = T_lat_inst < self.engaged_T_lat_max_Nm_partcenter
        quasi_static = v_xy_mag < self.engaged_v_xy_max_m_s

        # Engagement plausibility gate (per GPT review): even if F/T look
        # locally good, peg can be in a metastable LOW-FORCE state at xy
        # FAR from the predicted hole. Reject if so. Distance gate uses
        # CAD-predicted xy as global plausibility anchor (CAD has mm-scale
        # error, gate is 6mm — looser than acceptable variance, tighter
        # than observed false-metastable drift of 20mm).
        dist_to_pred_mm = float('inf')
        dist_ok = True   # default-pass when no predicted_xy provided
        if self.predicted_tcp_xy is not None:
            dx = self.predicted_tcp_xy[0] - tcp_x
            dy = self.predicted_tcp_xy[1] - tcp_y
            dist_to_pred = math.hypot(dx, dy)
            dist_to_pred_mm = dist_to_pred * 1000
            dist_ok = dist_to_pred <= self.engagement_dist_thresh_m

        # === HIGH-Z-DROP SHORTCUT (2026-05-04 PM) ===
        # If peg has dropped substantially below surface (e.g. >15mm), it's
        # unambiguously inside the slot regardless of xy distance. The
        # CAD-predicted xy gate has ~few-mm error; a peg that's dropped
        # 20-30mm has CLEARLY found a real hole, just in a slightly
        # different xy than CAD predicted.
        z_drop_dominant = False
        if self.surface_z is not None:
            z_drop = self.surface_z - tcp_z
            if z_drop >= self.engaged_z_drop_dominant_m:
                z_drop_dominant = True

        # === TILT GATE (operator's geometric insight, 2026-05-04 PM) ===
        # The U-part is held by gripper at a grasp point ~28mm offset from the
        # part's geometric center. "One prong in slot, other prong on rim" is
        # geometrically a TILTED part — and is invisible to F/T because the
        # contact forces can balance. ONLY EE quat shows it. Operator-demo
        # data: 60 successful inserts have max tilt ≤ 1.4° (u_orange) up to
        # ~6° (u_brown). v74 attempt-1 reached 4.4° at engagement candidate
        # (gate let it through), then ran away to 10.4° during INSERT.
        tilt_ok = self._tilt_deg <= self.engaged_tilt_max_deg

        all_pass = (F_lat_low and T_lat_low and quasi_static and tilt_ok and
                    (dist_ok or z_drop_dominant))
        if all_pass:
            if self._engaged_first_t is None:
                self._engaged_first_t = t
            if (t - self._engaged_first_t) >= self.entry_settle_verify_s:
                # CONFIRMED ENGAGEMENT — promote to INSERT
                self.hole_xy = (float(tcp_x), float(tcp_y))
                self.hole_z  = float(tcp_z)
                self.hole_t  = float(t)
                self.state = self.INSERT
                self.insert_start_t = t
                self.insert_start_z = float(tcp_z)
                action.transitioned = True
                action.new_state = self.INSERT
                action.transition_msg = (
                    f"engagement confirmed — F_lat={F_lat_inst:.2f}N (<{self.engaged_F_lat_max_N}) "
                    f"|T_lat|={T_lat_inst:.3f}Nm (<{self.engaged_T_lat_max_Nm}) "
                    f"|v_xy|={v_xy_mag*1000:.2f}mm/s (<{self.engaged_v_xy_max_m_s*1000}) "
                    f"sustained {self.entry_settle_verify_s:.2f}s. Hole at ({tcp_x:.4f},{tcp_y:.4f}) z={tcp_z:.4f}"
                )
                wrench, gain, damping, sel = self._insert_wrench(t)
                action.new_wrench = True
                action.wrench_baselink = wrench
                action.selection_vector = sel
                action.gain = gain
                action.damping = damping
                action.state = self.INSERT
                return action
        else:
            self._engaged_first_t = None

        # Timeout — engagement check failed. Decide between WEDGE_RECOVERY
        # (if Tx signal is dominant, indicating side-specific wedge) or
        # FIND_HOLE (no clear wedge signature, return to search).
        if t_in_state > (self.entry_settle_wait_s + self.entry_settle_max_s):
            # Check WEDGE dominance using T about PART CENTER (operator's
            # principled fix). The peg-on-rim wedge produces small T at TCP
            # but significant T at part center (long lever arm).
            Tx_inst = self._T_partcenter[0]
            Ty_inst = self._T_partcenter[1]
            tx_dominant = (
                abs(Tx_inst) > self.wedge_min_Tx_Nm and
                abs(Tx_inst) > self.wedge_dominance_ratio * abs(Ty_inst)
            )
            if tx_dominant and self._wedge_attempts < self.wedge_max_attempts:
                self._wedge_attempts += 1
                self.state = self.WEDGE_RECOVERY
                self._wedge_start_t = None
                self._tx_lp_buffer.clear()
                self._engaged_first_t = None
                self._F_lat_smooth_buffer.clear()
                self._entry_settle_start_t = None
                action.transitioned = True
                action.new_state = self.WEDGE_RECOVERY
                action.transition_msg = (
                    f"WEDGE detected (attempt {self._wedge_attempts}/{self.wedge_max_attempts}): "
                    f"Tx={Tx_inst:+.4f}Nm dominant over Ty={Ty_inst:+.4f}Nm. "
                    f"Applying counter-torque cmd_Tx={-self.wedge_kp_torque*Tx_inst:+.3f}Nm "
                    f"with reduced Fz={self.wedge_fz_N}N."
                )
                wrench_wr = (0.0, 0.0, -self.wedge_fz_N,
                             max(-self.wedge_cmd_Tx_max_Nm,
                                 min(self.wedge_cmd_Tx_max_Nm,
                                     -self.wedge_kp_torque * Tx_inst)),
                             0.0, 0.0)
                action.new_wrench = True
                action.wrench_baselink = wrench_wr
                action.selection_vector = (True, True, True, False, False, False)  # rotation LOCKED
                action.gain = 0.7
                action.damping = 0.5
                action.state = self.WEDGE_RECOVERY
                return action

            # Otherwise: back to FIND_HOLE
            self.state = self.FIND_HOLE
            self._entry_settle_start_t = None
            self._engaged_first_t = None
            self._F_lat_smooth_buffer.clear()
            # Per GPT: on ENTRY_SETTLE fail with bad distance, RESET the
            # spiral center toward predicted_xy (clamped near r_free/2).
            # Avoids retracing same outward excursion. If dist was OK
            # (engagement failed for F/T reasons only), keep current center.
            if (self.predicted_tcp_xy is not None and
                    dist_to_pred_mm > (self.engagement_dist_thresh_m * 1000)):
                # Recenter: predicted + clamped offset toward current
                pred_x, pred_y = self.predicted_tcp_xy
                dx = tcp_x - pred_x
                dy = tcp_y - pred_y
                d  = math.hypot(dx, dy)
                clamp_r = self.recovery_r_free_m / 2  # 3mm by default
                if d > clamp_r and d > 1e-6:
                    dx = dx * clamp_r / d
                    dy = dy * clamp_r / d
                self._spiral_origin_xy = (pred_x + dx, pred_y + dy)
                self._spiral_theta = 0.0
                recenter_msg = (f" RECENTERED spiral to "
                                f"({self._spiral_origin_xy[0]:+.4f},"
                                f"{self._spiral_origin_xy[1]:+.4f})")
            else:
                recenter_msg = ""
            action.transitioned = True
            action.new_state = self.FIND_HOLE
            action.transition_msg = (
                f"entry_settle TIMEOUT — F_lat={F_lat_inst:.2f}N "
                f"|T_lat|={T_lat_inst:.3f}Nm |v_xy|={v_xy_mag*1000:.2f}mm/s "
                f"tilt={self._tilt_deg:.2f}° (gate={self.engaged_tilt_max_deg:.1f}°) "
                f"dist_to_pred={dist_to_pred_mm:.1f}mm (gate={self.engagement_dist_thresh_m*1000:.0f}mm). "
                f"Back to FIND_HOLE.{recenter_msg}"
            )
            action.state = self.FIND_HOLE
            return action

        # Continue settling — re-issue settle wrench every tick (force-mode keeps it active)
        wrench, gain, damping, sel = self._entry_settle_wrench(t)
        # Only re-issue if first tick of state — otherwise leave (no XY drift)
        if t_in_state < 0.05:
            action.new_wrench = True
            action.wrench_baselink = wrench
            action.selection_vector = sel
            action.gain = gain
            action.damping = damping
        action.state = self.ENTRY_SETTLE
        action.msg = (f"entry_settle: F_lat={F_lat_inst:.2f}N(low={F_lat_low}) "
                      f"|T_lat|={T_lat_inst:.3f}Nm(low={T_lat_low}) "
                      f"|v_xy|={v_xy_mag*1000:.2f}mm/s(static={quasi_static}) "
                      f"all_pass={all_pass} sustain="
                      f"{(t-self._engaged_first_t) if self._engaged_first_t else 0:.2f}/"
                      f"{self.entry_settle_verify_s:.2f}s")
        return action

    def _tick_find_hole(self, t, tcp_x, tcp_y, tcp_z, fz_smoothed, F_lat_baselink, action):
        if self._find_hole_start_t is None:
            self._find_hole_start_t = t

        # CANDIDATE ENTRY: z dropped past rim — peg moved into chamfer/slot.
        # Per GPT review (2026-05-04 PM): we cannot use F_lat/T_lat to verify
        # FULL engagement while spiral is still running, because commanded
        # lateral spiral force shows up in measured F_lat (UR force-mode
        # design). So FIND_HOLE → ENTRY_SETTLE on the FIRST z drop, then
        # ENTRY_SETTLE stops the spiral, settles, and only THEN checks
        # force/torque to distinguish full vs partial engagement.
        #
        # Iter-10 (A3): ALSO transition on tilt-relaxation event (chamfer engaged but
        # peg hasn't dropped 0.8mm yet — common for tight-clearance pegs like u_orange).
        # The directed wrench function sets self._chamfer_engaged when sustained tilt
        # relaxation is detected. Fires earlier than the z_drop gate.
        # Iter-10 (bouncing-trap fix): require TCP to be NEAR the seat estimate when
        # firing the z_drop trigger. Otherwise random rim depressions (~1mm dips off-slot)
        # cause infinite FIND_HOLE↔ENTRY_SETTLE bouncing. iter-10 telemetry showed this:
        # peg drifted 35mm laterally over 280s of bouncing on a false 1mm dip.
        z_drop_raw_gate = self.surface_z is not None and tcp_z < (self.surface_z - self.find_hole_drop_thresh_m)
        seat_est = self._seat_xy_refined or self.spiral_origin_override or self.predicted_tcp_xy
        if z_drop_raw_gate and seat_est is not None:
            dist_to_seat_est = math.hypot(seat_est[0] - tcp_x, seat_est[1] - tcp_y)
            # Only allow z_drop trigger if peg is plausibly at the seat (≤10mm from estimate).
            # This is wider than ENTRY_SETTLE's own dist gate (6mm) so that a borderline
            # case doesn't bounce; tight enough to reject far-off rim depressions.
            z_drop_gate = dist_to_seat_est < 0.010
        else:
            z_drop_gate = z_drop_raw_gate  # fall back if no seat estimate
        chamfer_relax_gate = bool(getattr(self, "_chamfer_engaged", False))
        if z_drop_gate or chamfer_relax_gate:
            self.state = self.ENTRY_SETTLE
            self._entry_settle_start_t = t
            self._entry_settle_z_at_start = float(tcp_z)
            action.transitioned = True
            action.new_state = self.ENTRY_SETTLE
            trigger = "z_drop" if z_drop_gate else "tilt_relaxation"
            z_drop_mm = (self.surface_z - tcp_z) * 1000 if self.surface_z else 0
            action.transition_msg = (
                f"candidate entry [{trigger}] — z_drop={z_drop_mm:.2f}mm "
                f"tilt={getattr(self, '_tilt_deg', 0.0):.2f}° "
                f"peak={self._tilt_running_peak_deg:.2f}° "
                f"at xy=({tcp_x:.4f},{tcp_y:.4f}); stopping spiral, verifying engagement"
            )
            # Issue settle wrench (Fz only, no XY spiral)
            wrench, gain, damping, sel = self._entry_settle_wrench(t)
            action.new_wrench = True
            action.wrench_baselink = wrench
            action.selection_vector = sel
            action.gain = gain
            action.damping = damping
            action.state = self.ENTRY_SETTLE
            return action

        # Spiral budget exhausted (max radius or max time) → ABORT
        in_burst_t = t - self._find_hole_start_t
        if in_burst_t > self.find_hole_max_duration_s:
            action.kind = "exit_abort"
            action.abort_reason = (f"FIND_HOLE timeout {self.find_hole_max_duration_s}s "
                                   f"— no z drop detected. spiral_radius="
                                   f"{(self.spiral_pitch_m/(2.0*math.pi))*self._spiral_theta*1000:.2f}mm")
            return action

        # PROXIMITY-AWARE STUCK DETECTOR (iter-8):
        # GOLD operator timeline (insert_u_orange_20260505_193645):
        #   t+5s: at close-pass (dist=0.87mm), z_drop=0.17mm  ← old detector aborts here
        #   t+7s: bounced to 1.12mm (chamfer engagement)
        #   t+10s: z_drop=1.45mm (peg dropping in)
        # The 15s/<1mm rule is right when TCP is FAR from seat (peg never aligning),
        # but wrong when TCP is NEAR seat (peg needs time to engage chamfer). Splitting:
        #   - If TCP > proximity_thresh: 15s timeout (fast-fail when not approaching)
        #   - If TCP < proximity_thresh: extended timeout (give chamfer engagement time)
        if self.surface_t is not None and self.surface_z is not None:
            t_since_contact = t - self.surface_t
            z_drop_now = self.surface_z - tcp_z
            # Compute distance to seat (use spiral_origin_override = hole_xy_prior or
            # predicted_tcp_xy as best-known seat estimate)
            seat_xy_for_stuck = None
            if self.spiral_origin_override is not None:
                seat_xy_for_stuck = self.spiral_origin_override
            elif self.predicted_tcp_xy is not None:
                seat_xy_for_stuck = self.predicted_tcp_xy
            dist_to_seat_m = float("inf")
            if seat_xy_for_stuck is not None:
                dist_to_seat_m = math.hypot(seat_xy_for_stuck[0] - tcp_x,
                                            seat_xy_for_stuck[1] - tcp_y)
            # Proximity-scaled stuck window: if peg is near seat, give chamfer engagement time
            proximity_threshold_m = 0.002  # 2mm
            extended_window_s = 30.0
            window_s = extended_window_s if dist_to_seat_m < proximity_threshold_m else self.find_hole_stuck_window_s
            if t_since_contact >= window_s and z_drop_now < self.find_hole_stuck_z_drop_m:
                action.kind = "exit_abort"
                action.abort_reason = (
                    f"FIND_HOLE STUCK: z_drop={z_drop_now*1000:.2f}mm < "
                    f"{self.find_hole_stuck_z_drop_m*1000:.1f}mm after "
                    f"{t_since_contact:.1f}s (proximity {dist_to_seat_m*1000:.1f}mm, "
                    f"window={window_s:.0f}s). Aborting; no progress."
                )
                return action
        radius = (self.spiral_pitch_m / (2.0 * math.pi)) * self._spiral_theta
        if radius > self.find_hole_max_radius_m:
            action.kind = "exit_abort"
            action.abort_reason = (f"FIND_HOLE max radius {self.find_hole_max_radius_m*1000:.0f}mm hit "
                                   f"— no z drop detected. theta={self._spiral_theta:.2f} rad")
            return action

        # === INITIAL DIRECTED PRESS (data-driven fixed-direction push) ===
        # Cross-run analysis (success vs failure at each timepoint):
        # successful inserts have F_lat=5N sustained at t=1-1.5s causing 2mm
        # xy excursion, then peg drops at t=2s. Failed inserts have F_lat=0.6N
        # and 0.3mm excursion. The DIRECTION of the push isn't toward
        # predicted_xy (peg is already there); it's a FIXED nudge that gets
        # the peg WALKING — operator demo's mean walk direction was ~-67°
        # in xy = mostly -Y in world = +Y in base_link (sign flip).
        in_burst_t = t - self._find_hole_start_t
        if in_burst_t < self.find_hole_press_duration_s:
            # Fixed direction in base_link frame from config
            angle_rad = math.radians(self.find_hole_press_dir_deg)
            Fmag = self.find_hole_press_F_N
            Fx_cmd = Fmag * math.cos(angle_rad)
            Fy_cmd = Fmag * math.sin(angle_rad)
            wrench = (Fx_cmd, Fy_cmd, -self.find_hole_press_fz_N, 0.0, 0.0, 0.0)
            sel = (True, True, True, False, False, False)  # rotation LOCKED — prismatic peg, no rotational compliance needed (2026-05-06)
            action.new_wrench = True
            action.wrench_baselink = wrench
            action.selection_vector = sel
            action.gain = self.gain_find_hole
            action.damping = self.damping_find_hole
            action.state = self.FIND_HOLE
            action.msg = (f"find_hole [INITIAL_PRESS]: t={in_burst_t:.2f}s/"
                          f"{self.find_hole_press_duration_s:.1f}s dir="
                          f"{self.find_hole_press_dir_deg:.0f}° "
                          f"F=({Fx_cmd:+.1f},{Fy_cmd:+.1f},-{self.find_hole_press_fz_N:.0f})N "
                          f"drop_so_far={(self.surface_z-tcp_z)*1000:.2f}mm")
            return action

        # Continue spiral
        wrench, gain, damping, sel, info = self._find_hole_wrench(t, (tcp_x, tcp_y))
        action.new_wrench = True   # re-issue every tick (PD setpoint changes)
        action.wrench_baselink = wrench
        action.selection_vector = sel
        action.gain = gain
        action.damping = damping
        action.state = self.FIND_HOLE
        action.msg = (f"find_hole: r={info['spiral_radius_m']*1000:.2f}mm "
                      f"θ={math.degrees(info['spiral_theta_rad']) % 360:.0f}° "
                      f"err=({info['err_xy_base_mm'][0]:+.1f},{info['err_xy_base_mm'][1]:+.1f})mm "
                      f"F=({wrench[0]:+.1f},{wrench[1]:+.1f})N "
                      f"surface_z={self.surface_z:.4f} tcp_z={tcp_z:.4f} "
                      f"(drop_so_far={(self.surface_z-tcp_z)*1000:.2f}mm)")
        return action

    def _tick_insert(self, t, tcp_x, tcp_y, tcp_z, fz_smoothed, action):
        # Track z over motion window
        in_state_t = t - (self.insert_start_t or t)

        # Compute z-descent rate over window
        descent_rate = self._z_velocity(t, tcp_z, self.insert_motion_window_s)
        # descent_rate < 0 = peg descending (z decreasing); > 0 = ascending; ~0 = stopped

        # Check motion-stopped: |dz/dt| < threshold sustained
        if descent_rate is not None:
            speed = abs(descent_rate)
            if speed < self.insert_motion_thresh_m_s:
                # SEAT criterion: either of (a) descended ≥5mm SINCE entering
                # INSERT, OR (b) total descent from surface ≥25mm (peg was
                # already seated when ENTRY_SETTLE committed via z_drop_dominant
                # shortcut; further descent during INSERT not needed).
                descended_post_hole = (self.hole_z - tcp_z) if self.hole_z is not None else 0.0
                descended_from_surface = (self.surface_z - tcp_z) if self.surface_z is not None else 0.0
                seated_via_post_hole = descended_post_hole >= self.insert_min_descent_m
                seated_via_surface  = descended_from_surface >= self.engaged_z_drop_dominant_m
                if seated_via_post_hole or seated_via_surface:
                    if self._motion_stopped_first_t is None:
                        self._motion_stopped_first_t = t
                    # Sustain for half the motion window
                    if (t - self._motion_stopped_first_t) >= (self.insert_motion_window_s * 0.5):
                        # SEAT confirmed — peg bottomed out
                        self.state = self.DONE
                        action.kind = "exit_done"
                        action.transitioned = True
                        action.new_state = self.DONE
                        criterion = ("post-hole" if seated_via_post_hole else "z-drop-dominant")
                        action.transition_msg = (f"INSERT seated [{criterion}]: descended_post_hole="
                                                 f"{descended_post_hole*1000:.2f}mm, "
                                                 f"descended_from_surface={descended_from_surface*1000:.2f}mm, "
                                                 f"motion stopped (|dz/dt|={speed*1000:.3f}mm/s) "
                                                 f"sustained {self.insert_motion_window_s*0.5:.1f}s")
                        return action
                else:
                    self._motion_stopped_first_t = None
            else:
                self._motion_stopped_first_t = None

        # Insert timeout — peg never bottomed
        if in_state_t > self.insert_max_duration_s:
            action.kind = "exit_abort"
            action.abort_reason = (f"INSERT timeout {self.insert_max_duration_s}s — peg never bottomed. "
                                   f"descent_post_hole={(self.hole_z-tcp_z)*1000 if self.hole_z else 0:.2f}mm")
            return action

        # Tilt runaway → WEDGE_RECOVERY (per GPT review): going back to
        # FIND_HOLE would bounce-loop because peg is below surface_z and the
        # find_hole z-drop trigger would immediately re-fire ENTRY_SETTLE.
        # WEDGE_RECOVERY applies counter-torque to flatten the part, then
        # re-enters ENTRY_SETTLE for re-validation.
        if (self._tilt_deg > self.insert_tilt_abort_deg and
                self._wedge_attempts < self.wedge_max_attempts):
            self._wedge_attempts += 1
            self.state = self.WEDGE_RECOVERY
            self._wedge_start_t = None  # reset; _tick_wedge_recovery sets on entry
            action.transitioned = True
            action.new_state = self.WEDGE_RECOVERY
            action.transition_msg = (
                f"INSERT → WEDGE_RECOVERY: tilt runaway {self._tilt_deg:.2f}° > "
                f"{self.insert_tilt_abort_deg:.1f}° (lever-arm pumping peg out "
                f"of partial engagement; attempt {self._wedge_attempts}/"
                f"{self.wedge_max_attempts})"
            )
            action.state = self.WEDGE_RECOVERY
            return action

        # Continue insert
        wrench, gain, damping, sel = self._insert_wrench(t)
        # Re-issue every tick if tilt is on (varies per tick); else only at start
        if self.insert_tilt_T_Nm > 0.0 or in_state_t < 0.05:
            action.new_wrench = True
        action.wrench_baselink = wrench
        action.selection_vector = sel
        action.gain = gain
        action.damping = damping
        action.state = self.INSERT
        descended_post_hole = (self.hole_z - tcp_z) if self.hole_z is not None else 0.0
        action.msg = (f"insert: tcp_z={tcp_z:.4f} descent_post_hole={descended_post_hole*1000:.2f}mm "
                      f"dz/dt={(descent_rate*1000) if descent_rate is not None else float('nan'):+.2f}mm/s "
                      f"fz_smoothed={fz_smoothed:+.2f}N")
        return action

    # ---------- Diagnostic helper ----------------------------------------
    def diag_string(self, t: float, tcp_z: float, fz_smoothed: float, period_s: float = 1.0) -> Optional[str]:
        if (t - self._last_diag_t) < period_s:
            return None
        self._last_diag_t = t
        return (f"FSM[{self.state}]: tcp_z={tcp_z:.4f} fz_smooth={fz_smoothed:+.2f} "
                f"surface_z={self.surface_z}")
