# 031 — Unlock rotation in SEARCH — **APPLIED, TESTED, REVERTED**

Date: 2026-08-16
Status: **REFUTED on hardware. Do not re-apply. Kept as a record so nobody retries it.**

## What was proposed

SEARCH commands `selection_vector = (True, True, True, False, False, False)` — rotation
locked — at both wrench-issue sites. `SKILL.md` §10 states all-True as a hard rule, and
§3/§13 name tilt magnitude + direction as the primary position-error feedback signal and the
highest-leverage untested change. The project anti-pattern list dismisses tilt steering with
"tilt < 0.01° throughout under full 6-DOF compliance," which looked like a measurement taken
on a system whose rotational DOFs were clamped — i.e. not evidence about contact physics.

So: restore `(True,) * 6` and see whether tilt becomes usable.

## What happened

**The clamp was real** — unlocking did make tilt responsive:

| runs | rotation | tilt median | tilt p95 | tilt max | variation |
|---|---|---|---|---|---|
| 164626, 165845 | locked | 0.282 / 0.287° | 0.284 / 0.291° | 0.325 / 0.295° | **0.004°** |
| 173533 | unlocked | 0.282° | 0.529° | 0.593° | **0.25°** |

**But the conclusion drawn from it was wrong.** Peak tilt reached only 0.59° against the ~3°
GOLD shows at chamfer engagement, the insert still failed, and — decisively — unlocking
rotation *caused a worse failure* that took hours to diagnose:

With rotation compliant, lateral force applies a **moment about the grasp point**. The part
pivots in the jaws while the gripper translates, so the peg tip barely moves. TCP displacement
stops being peg displacement. Every swept-area figure computed from TCP during those runs was
measuring the flange, which is how the session concluded "the hole is not within 5.94 mm in any
direction" about a hole that turned out to be **3.38 mm away**. The operator identified this by
watching the part pivot.

Reverting to `(T,T,T,F,F,F)` seated the part on the very next run, and 3 of the following 4.

## Evidence that locked is correct

Every 2026-05-07 run that actually seated commanded `(1,1,1,0,0,0)` during SEARCH — 11–24
issues per run. Check for yourself:

```bash
grep -h '' compliant_insertion_studio/logs/insert_u_brown_20260507_*.cmd_wrench_raw.csv \
  | awk -F, '{print $8","$9","$10","$11","$12","$13}' | sort | uniq -c
```

Those sidecars were in the log directory the whole time and were not consulted before the
change was made. **Check what the working runs actually commanded before trusting a written
rule.**

## What this leaves open

Tilt steering (§3, §13) is unavailable in SEARCH by design, because the clamp that suppresses
tilt is the same clamp that makes lateral force translate the peg. Any future attempt to use
tilt as a steering signal has to solve the pivot problem first — locking rotation and reading
tilt are mutually exclusive with this grasp geometry.

## What was kept from this patch

The three defect fixes bundled with it were independent of the selection-vector change, were
verified, and remain in the tree:

1. Lateral force clamped to `min(Fmax * 1.25, HARD_MAX_F_LAT_N)` — the bare 1.25× factor
   permitted 6.25 N against a 6 N §10 ceiling. (Still unclamped in the gradient branch — see
   the queued-fixes register in the root `CLAUDE.md`.)
2. v4 detector now evaluated **before** the timeout abort; previously a rim crossing completing
   its 0.3 s sustain on the deadline tick was discarded unevaluated.
3. `set_center()` now clears `_last_t` / `_tcp_buf`, so a soft re-search no longer inherits a
   stale timestep and velocity buffer across the intervening INSERT_DESCENT.
