# Assembly state machine (replay_real_assembly.py) — arm positions, reads, and where the time goes

Mapped 2026-10-21 from the code (line numbers at e27ccb2) and the 18 replay timing logs of 2026-10-20/21
(`ablations/logs/replay_*.timing.jsonl`). Operator, 16:30: "the amount of motion front and back to home and
unnecessary reads … have you even mapped out the assembly state machine to optimize". This is that map.
The one full cycle that reached the per-part loop (10:27) had the arm moving 23 % of 596 s.

## States (arm → what is read)

| # | state | arm | read / why | guard | replay_real_assembly.py |
|---|---|---|---|---|---|
| S0 | startup | home, gripper open | – | not `--skip-startup` | 2992-3008 |
| S1 | derive sequence | where the arm is | each part, one marker read, 10 s timeout | `--derive-sequence` | 3039 |
| S2 | parts home can't see | staging view (clear area with `--part-at-staging`) → **home** | same read, 15 s timeout | a part unseen in S1 | 3050-3064 |
| S3 | base view | 0.56 m over the base (tracker → region) | 15 marker samples + depth outline + ICP (`choose_base_read`) | `--base-from-camera` | 3104-3116 |
| S4 | back | **home** | – | | 3117-3125 |
| S5 | spacing check | home | `observe_parts` all | base seen, no `--part-at-staging` | 3140-3178 |
| L0 | between parts | 0.56 m over base → **home** | `reread_base`, `verify_seated` (ledger-only when certified) | not the first part | 3191-3213 |
| L1 | line_green only | 0.56 m over base → **home** | notch depth (U leg gaps), marker hover fallback | `FOLLOWS_SEATED` | 3216 → 1158-1209 |
| L2 | find the part | home miss 5 s → 0.52 m over last-seen → staging view | marker read | not seen here | 3237 → 736 |
| L3 | pick room | in place | `observe_parts` + other bases | | 3249 → 643 |
| P1 | unwind wrist_3 | – | – | | 2314 |
| P2 | depth-pick view | 0.52 m over the part (often the pose L2 just left) | part CAD/depth fit (~8 s, untimed) | `PICK_FROM=depth` | 2437 → 1732 |
| P3-5 | grasp, close, safe height, rotate | – | width guard | | 2591 |
| P6 | pre-insert | in place | held-yaw read 6 s | always | 2582-2583 |
| P7 | insert (drive_mate) | hover → align funnel → descent | wrist turn at hover | `SEAMS_DRIVE=1` | 1663; drive_mate_server.py:248; translate_object.py:236-268 |
| P7b | retry | lift → insert (2-3) | base "re-read" is a no-op with `PRE_INSERT_READ=off` | | 2718-2737 |
| P7c | recovery | clear area → base re-read → clear area → regrasp | | | 2739-2786 |
| C1 | certify | tilted seat view; sides outer/inner/along± | ExecuteMate: own ICP base + depth + marker (~20 s) | `SEAMS_CERTIFY=1` | 2815 → 1697; seat_check.py:223; seams/vision_ops.py:145 |
| E | end | base re-read, verify, safe height, home | | | 3285-3322 |

## Avoidable moves and reads (measured), ranked

1. **Base re-read after certify** — ~20 s/part (~80 s/cycle). ExecuteMate already fits the base by ICP at the seat
   view (vision_ops.py:151-158); the replay re-reads it at 0.56 m (move 5.6 + markers 6.0 + ~8.5 untimed). Return
   the certify fit, use it as `observed_base`, skip 3202; drop the final re-read at 3288. Check the tilted-view ICP
   against the top-down read for one cycle first.
2. **Timed-out single-part observes in planning** — 24 × 10.8 s from home + 12 × 15.8 s at the staging view across
   18 runs (16:26 run: 32 s; 13:01: 53 s). One `observe_many` per view; plan unseen parts from part_tracker.json.
3. **Between-step overhead** — median 2.24 s over 85 gaps: two fresh python+rclpy `snapshot_robot_state` processes
   and a 0.5 s sleep per step. One in-process subscriber: ~40 s/cycle.
4. **Certify view order** — u_brown's inner view rejected 2/2 (end below the seat = impossible reading), ~26 s per
   u_brown. Reorder `VIEW_SIDES` (outer, along±, inner) or remember the side that worked per part.
5. **line_green's notch trip** — home → 0.56 m → home (~12-16 s). Take the notch capture at the L0 pose.
6. **Finding staged parts via home** — home cannot see the staging region: per part home 4.0 + miss 5.8 + 0.9 +
   0.52 m view 7.3 + a second move to the same 0.52 m depth-pick pose 3.7 + 1.5 s sleep. Go from the base view
   straight to the depth-pick view; merge `_part_view` with `depth_pick_target`: ~15 s/part.
7. **Home stops between views at startup** — staging → home → base view (3.9 s); base view → home → staging part
   view (4.0 + 5.8 s miss); `--part-at-staging` clear area → home → base view → staging (3.9 s).
8. **Held-yaw read on the U parts** — 0 samples in 6.8 s every time (u_brown 3/3, u_orange 2/2): ~14 s/cycle.
9. **Base read cost under `BASE_FROM=icp`** — 15 marker samples to seed ICP and an outline thrown away: ~5 s × ~5.
10. **Spacing check then pick-room check from the same pose** — 4.6 s + 9.4 s gap; reuse.

Items 1-8: ~4-5 min per FMB1 cycle.

## Correctness risks found on the way

- No base read does not stop the run (3127: DEFAULT_BASE_POSITION, then pick and rotate); drive_mate refuses only
  at the insert with the part in hand (16:26 run). Fail closed when `SEAMS_DRIVE=1`.
- Retries don't re-read the base with `PRE_INSERT_READ=off`: attempts 2-3 aim at the same base attempt 1 may have
  pushed (14:02 line_green: three identical targets).
- The timing log misses a lot: `spacing_check` imports a second copy of the replay module (the script runs as
  `__main__`), so its `observe_many` is never timed and it keeps its own RUN_BASE / LAST_SEEN_XY. ICP, outline,
  part depth fit (8.2 s median) and notch depth are untimed: 10-25 s/part shows only as gaps.
- A part unseen by home and the staging view falls back to the recorded sequence (13:01: line_green, yellow).

Timeline helper used for the numbers: one row per step (duration, moving time, gap before) from any
`replay_*.timing.jsonl` + `.motion.jsonl`.
