# Debugging one part: the per-part iteration loop (2026-10-19)

Use this when one part's pick, place or insert misbehaves. Do **not** iterate through
`fmb1_loop.py` with `--layout staging`: the generated layouts and the tipping are the *evaluation*
start state and add minutes of motion and new failure modes per try. Iterate on the one part, from
the simplest start state, and read the data before changing anything.

## 1. Set the start state once

All parts on the table (plain spots, upright, nothing tipped) — from an assembled board:

```bash
python3 -u ablations/disassemble.py --assembly-json x                 # free spots seen from home
```

Only the part under test out, the rest seated (fastest for an insert problem):

```bash
python3 -u ablations/disassemble.py --assembly-json x --only PART --to-staging   # pulls PART with the insert
#   grasp, sets it down in the clear area (-0.32, -0.50), stays above it
```

If the board is empty and you need the other parts seated first:

```bash
J=ablations/ground_truth_resources/Assembly_fmb_assembly_1_results.json
python3 -u ablations/replay_real_assembly.py --assembly-json $J --only u_brown
python3 -u ablations/replay_real_assembly.py --assembly-json $J --only u_orange   --skip-startup
python3 -u ablations/replay_real_assembly.py --assembly-json $J --only line_green --skip-startup
```

## 2. One try

```bash
# part in the clear area (after --to-staging or a failed insert's recovery):
python3 -u ablations/replay_real_assembly.py --assembly-json $J --only PART --skip-startup --part-at-staging
# part on the table where the camera sees it from home (or it looks for it: staging view, above its last position):
python3 -u ablations/replay_real_assembly.py --assembly-json $J --only PART --skip-startup
# still held after a stop:
python3 -u ablations/disassemble.py --assembly-json x --place-held PART      # -> clear area, then the first line
```

A seated part that must come out again: `disassemble.py --assembly-json x --only PART --to-staging`.
A part too close to another: `disassemble.py --assembly-json x --relocate PART --grasp-id auto [--spot-staging]`.

## 3. Read the data before touching code

| question | where |
|---|---|
| what each step did, in order | `ablations/logs/replay_*.log` (or `loop_<ts>_assemble.log`); `>>> label: cmd` lines |
| the insert's physics | `compliant_insertion_studio/logs/insert_<part>_<ts>.csv` + `.wrench_raw.csv` + `meta.json` — trust these, not the FSM's labels |
| where it stalled vs CAD | the `Contact candidates (CAD-derived): SEAT=.. BASE_RIM=.. OBJ:..` line; stall height − SEAT tells the stage |
| how far off the aim | `[VERIFY] result=.. pos_dev=.. ori_dev=..` (prismatic) / `Δ=` on the GLOBAL SEAT line (compliant) |
| grasp vs aim | a regrasp that fixes it = grasp offset; the same miss after a regrasp = aim (base read, seat bias) |
| what the camera saw | `python3 -u ablations/spacing_check.py` (parts + base from home); `ablations/seat_check.py --object line_green` |
| the clearances | CAD, not guesses: `ablations/grasp_depth.py` style mesh queries (`load_mesh`, seat poses from `cad_lookup`) |

Rules that have cost runs when skipped: settle ≥ 1.5 s before any force baseline; no force on parts
above 5 N; never shorten a primitive's duration without knowing the joint distance; test new motion
with nothing gripped first.

## 4. When it works twice in a row

Put it back in the loop: `python3 -u ablations/fmb1_loop.py --cycles 3 --layout staging` is the
evaluation, with recording (`compliant_insertion_studio/scripts/record_demo_bag.sh`).
