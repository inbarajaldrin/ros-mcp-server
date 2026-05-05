# raw_fsm_logs — what these are and what they are NOT

## What's here

- `loop_v_series/` — 81 files. Stdout/stderr from `loop_iterate.py` for each FSM version (v3 through v97) during the 2026-05-03/04 iteration session. Total ~3 MB.
- `iter_series/` — 16 files. Stdout from individual `iterate_insert.py` runs from earlier the same day.

The matching per-attempt logs (`loop_attempt_NNN.log`) and `summary.json` for each iteration are at `.planning/phases/05-algorithm-derivation/loop_u_orange_<unix_ts>/` (87 directories) — those stay where they are; this folder collects the higher-level loop-orchestrator output that lived in /tmp.

## ⚠ Critical caveat — DO NOT use as ground truth

These logs contain the FSM's **own report** of what it observed and decided:
- "engagement confirmed" — the FSM's guess based on its (often-wrong) z_drop / xy-distance gates
- "STUCK: z_drop=0.05mm < 1.0mm after 15.0s" — the FSM's claim, sometimes contradicted by the CSV
- "GLOBAL SEAT" — the v87 detector's verdict, validated offline but not field-validated
- "INITIAL_PRESS direction: +Y baselink" — the FSM's commanded direction (config-level, this is reliable)
- "VERDICT: success/abort" — derived from the FSM's predicates, which were the buggy bit

**The FSM was wrong about whether the peg seated, on multiple iterations.** Its stdout reflects the wrong threshold, not the physics. The CSV is the only source of truth for what actually happened.

## What these logs ARE useful for

1. **FSM CONFIG per attempt** — startup-time parameter prints are reliable: `spiral_F_max=10`, `find_hole_fz=8`, `engaged_tilt_max_deg=3.5`, `hole_xy_prior=(...)`. These are deterministic settings applied to the controller. **Use these to bin CSVs by config.**

2. **Version banner** — each log starts with the FSM version label (`Phase 5 v3 FSM` plus the version-specific change list). Maps to entries in `../v82_v97_iteration_history.json`.

3. **State-transition timestamps** — `[t=23.45s] FSM → ENTRY_SETTLE` is reliable as a wall-clock event marker, useful for aligning the loop log to the CSV's `t_s`. The transition itself happened; whether the FSM was *right* to transition is a separate question.

4. **Detection-bug forensics** — comparing what the log claims (`STUCK z_drop=0.05mm`) against what the CSV shows at the same wall-clock time exposes where the FSM's detection logic was wrong. This is a **discovery target**, not a label source.

## How the discovery loop should treat these

✅ **DO** parse for FSM config + version + state transition timestamps
✅ **DO** cross-check FSM-claimed events against CSV ground truth — disagreements ARE the discovery signal
❌ **DO NOT** use FSM-claimed `engagement_confirmed` / `STUCK` / `seat_detected` as labels for training a classifier
❌ **DO NOT** use the FSM's `outcome=success/abort` as truth — use the CSV's `final_z_drop_mm ≥ 20mm` or our `durable_collapse_rate` predicate instead

## Provenance

- `loop_v_series/loop_v3*.log` (May 4 04:00–07:00) — earlier "v3" series, FSM redesign experiments
- `loop_v_series/loop_v82.log` through `loop_v97.log` (May 4 evening) — the 16-iteration session with stagnant outcomes
- `loop_v_series/loop_demo_v83.log` — operator-assisted demo collected during v83 iteration
- `iter_series/iter*.log` (May 4 01:00–05:00) — individual `iterate_insert.py` runs during early-day exploration

These are session-specific and not regenerable. They live under version control so a future agent inspecting the same FSM versions can match logs to CSVs without filesystem archaeology.
