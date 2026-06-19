"""
OnRobot RG2 arc model — CAD-grounded width <-> joint-angle <-> gap <-> depth.

Single source of truth for BOTH backends (sim = Isaac joint; real = modbus rGWD),
so command -> motion is identical. Replaces the linear width->joint viz-fake that
husarion / tonydle / our old bridge all used.

Geometry: manufacturer RG2_v2.STEP (aruco-runner). Parallelogram 4-bar, arm R=55.0.
  alpha(theta)      = alpha_zero + theta            (alpha = moment-arm angle)
  gap_rubber(alpha) = 2*R*sin(alpha) + 2*c          (rubber pad-to-pad CONTACT gap; predicate number)
  raw(alpha)        = gap_rubber + 2*offset          (getWidth = structural/aluminum gap)
  depth(alpha)      = R*cos(alpha) + depth_offset    (fingertip depth from flange; width<->depth coupling)

Params come from config/rg2_gripper.yaml (loaded once). The ~1mm fingertip_offset /
raw-0 anchor are PARAMS pending the real device's getWidth/getWidthWithOffset readback.
"""
from __future__ import annotations
import math
from dataclasses import dataclass


@dataclass
class Rg2ArcParams:
    arm_radius_mm: float = 55.0
    gap_rubber_c_mm: float = -1.98          # per-side; gap = 2*(R*sin a + c) = 110 sin a - 3.96
    depth_offset_mm: float = 142.9
    alpha_zero_deg: float = 27.95
    fingertip_offset_mm: float = 4.9        # firmware REG 1031 (structural plate); CAD geom = 4.45
    width_min_mm: float = 0.0
    width_max_mm: float = 110.0
    theta_closed_rad: float = -0.452
    theta_open_rad: float = 0.782

    @classmethod
    def from_yaml(cls, path: str) -> "Rg2ArcParams":
        import yaml
        with open(path) as f:
            d = yaml.safe_load(f)["rg2"]
        keep = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in d.items() if k in keep})


class Rg2Arc:
    def __init__(self, p: Rg2ArcParams | None = None):
        self.p = p or Rg2ArcParams()

    # --- angle <-> arc ---
    def alpha(self, theta: float) -> float:
        return math.radians(self.p.alpha_zero_deg) + theta

    def gap_rubber_mm(self, theta: float) -> float:
        return 2.0 * (self.p.arm_radius_mm * math.sin(self.alpha(theta)) + self.p.gap_rubber_c_mm)

    def raw_width_mm(self, theta: float) -> float:
        """getWidth() — structural/aluminum gap = rubber + 2*offset."""
        return self.gap_rubber_mm(theta) + 2.0 * self.p.fingertip_offset_mm

    def depth_mm(self, theta: float) -> float:
        """Fingertip depth from flange (table-clearance; couples to width)."""
        return self.p.arm_radius_mm * math.cos(self.alpha(theta)) + self.p.depth_offset_mm

    # --- derived open limit (GPT must-fix #1): theta_open must track the ACTIVE offset,
    #     not the YAML constant (which is the CAD-offset value). Else open over-shoots ~1mm. ---
    def theta_at_raw_max(self) -> float:
        s = (self.p.width_max_mm - 2.0 * self.p.fingertip_offset_mm - 2.0 * self.p.gap_rubber_c_mm) / (2.0 * self.p.arm_radius_mm)
        s = max(-1.0, min(1.0, s))
        return math.asin(s) - math.radians(self.p.alpha_zero_deg)

    # --- command (raw width) -> joint angle ---
    def theta_of_raw_width(self, raw_mm: float) -> float:
        raw = max(self.p.width_min_mm, min(self.p.width_max_mm, raw_mm))
        s = (raw - 2.0 * self.p.fingertip_offset_mm - 2.0 * self.p.gap_rubber_c_mm) / (2.0 * self.p.arm_radius_mm)
        s = max(-1.0, min(1.0, s))
        theta = math.asin(s) - math.radians(self.p.alpha_zero_deg)
        # clamp to [closed, DERIVED open] — derived so the open limit tracks the chosen offset
        return max(self.p.theta_closed_rad, min(self.theta_at_raw_max(), theta))

    # --- width-protocol helpers (mirror getWidthWithOffset) ---
    def rubber_of_raw(self, raw_mm: float) -> float:
        return max(0.0, raw_mm - 2.0 * self.p.fingertip_offset_mm)

    def raw_of_rubber(self, rubber_mm: float) -> float:
        return rubber_mm + 2.0 * self.p.fingertip_offset_mm

    # --- contact-gap (rubber) is the CONSUMER-FACING width (GPT review must-fix #3) ---
    def max_contact_mm(self) -> float:
        """Max achievable rubber contact gap = raw_max - 2*offset (raw can't exceed width_max)."""
        return self.p.width_max_mm - 2.0 * self.p.fingertip_offset_mm

    def contact_of_theta(self, theta: float) -> float:
        """Rubber pad-to-pad CONTACT gap from joint theta. Clamped nonnegative (must-fix nit)."""
        return max(0.0, self.gap_rubber_mm(theta))

    def theta_of_contact(self, contact_mm: float) -> float:
        """Commanded CONTACT gap -> joint theta. Contact clamped to [0, max_contact]
        BEFORE adding the offset (so a 110mm 'open' request saturates at the real max, not over-opens)."""
        c = max(0.0, min(self.max_contact_mm(), contact_mm))
        return self.theta_of_raw_width(c + 2.0 * self.p.fingertip_offset_mm)

    # explicit-name aliases so command code never drifts into raw/rubber ambiguity (nit)
    def contact_of_raw(self, raw_mm: float) -> float:
        return self.rubber_of_raw(raw_mm)

    def raw_of_contact(self, contact_mm: float) -> float:
        return self.raw_of_rubber(contact_mm)


if __name__ == "__main__":
    # Self-test vs CAD-GROUND-TRUTH.md (firmware offset 4.9).
    arc = Rg2Arc()
    print("theta  alpha   gap_rubber  raw    depth   | CAD gap_rubber=110sin(a)-3.96")
    for th in (-0.452, -0.2, 0.0, 0.3, 0.6, 0.782):
        a = math.degrees(arc.alpha(th))
        cad = 110 * math.sin(math.radians(a)) - 3.96
        print(f"{th:+.3f} {a:5.1f}  {arc.gap_rubber_mm(th):7.2f}   {arc.raw_width_mm(th):6.1f} "
              f"{arc.depth_mm(th):6.1f}  | {cad:7.2f}")
    print("\nraw -> theta -> rubber (vs CAD table):")
    for raw in (0, 30, 55, 80, 110):
        th = arc.theta_of_raw_width(raw)
        print(f"  raw {raw:3d} -> theta {th:+.3f} -> rubber {arc.rubber_of_raw(raw):5.1f}")
    print(f"depth swing close->open = {arc.depth_mm(-0.452) - arc.depth_mm(0.782):.1f} mm (CAD 38.7)")
    print(f"\nmax contact = {arc.max_contact_mm():.1f} mm (= width_max - 2*offset)")
    print("CONTACT cmd -> theta -> contact (round-trip; 110 must SATURATE at max_contact):")
    for c in (0, 35, 55, 100, 110):
        th = arc.theta_of_contact(c)
        print(f"  cmd {c:3d} -> theta {th:+.3f} -> contact {arc.contact_of_theta(th):5.1f}")
