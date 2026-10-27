# M4 dispatch pack - a planner build (harbor authoring) for FMB1 on the elevated bench, in simulation

Written 2026-10-26 on the cell PC (OptiPlex), branch `dev`. Everything here was derived from a **read-only** look at
`dual-a4500:~/prompt2assembly` (plan-anything `refs/pa/published` @ `538bbeee`, 2026-10-26 11:40 -0700) and the real
cell's measurements (`../realcell_env/`). Nothing was written on dual-a4500. The operator dispatches; this pack is
what the dispatch needs.

**What it is for.** M4 (`docs/ARCHITECTURE_SEAMS.md:69`, `docs/LADDER.json` rung M4) is "the delivered FMB1 planner
runs on the real cell with one substitution (`_correct_seat` -> drive-mate)". The planner is BUILT in simulation by a
prompt2assembly harbor authoring job. For the build to be worth carrying to the cell, the simulated bench and task
must state what the real cell is: the robot base 0.080 m above the table, a base plate that is free and sits where
it really sits, parts starting where and how they really start, and a seam the substitution can be made at.

**TL;DR (dual-a4500, in order):** stage `environments/dimelab_elevated` -> static gate -> `publish_environment` ->
stage `tasks/fmb1_assemble_ur5e_rg2_realcell_gen` -> exercise the layout engine -> `publish_task` -> seal/promote
-> fill three fields in `requests/authoring-fmb1-realcell-gen.yaml` -> `harbor_ready.py` -> `pa dispatch harbor`.
Steps in §3; every command is quoted from the skill or the script it comes from, with the place it was read.

---

## 1. What is in the pack

| path | what | state |
|---|---|---|
| `environments/dimelab_elevated/declaration.yaml` | the published `dimelab` declaration (`a10d8eba…`, verified identical on dual-a4500 2026-10-26) with TWO changes: `name: dimelab_elevated` and `robots[0].base_pose.xyz` z `0.0 -> 0.08` (+ a comment block saying why) | unpublished, ungated |
| `environments/dimelab_elevated/sim-binding.yaml` | dimelab's `sim-binding.yaml` (`dc3d5e00…`) with its `name`/`environment` lines renamed; the hop to prims is unchanged | " |
| `environments/dimelab_elevated/assets/rig_manifest.json` | dimelab's, byte-identical (`a982d39d…`) | " |
| `tasks/fmb1_assemble_ur5e_rg2_realcell_gen/task.yaml` | the task: a generated sibling of the control `fmb1_assemble`, built from `fmb1_assemble_ur5e_rg2_flat_gen` @ 538bbeee with the real-cell placements (§2) | unpublished, ungated, `environment.receipt_sha256: PUBLISH_FIRST` |
| `requests/authoring-fmb1-realcell-gen.yaml` | the `pa.dispatch-request/2` harbor-authoring request: a copy of the request that built fmb1flat x Opus 5.5 attempt 12 (`.pa/runs/02f640fb…`, 2026-10-13), task/id/series changed; three `<...>` fields to fill | ready after §3 step 4 |
| `fallback/` | A: Body1 static at the measured pose (if the engine refuses a free reference part); B: the untipped (standing) start | only if needed |

Not in the pack (the publisher writes them): `gate.json`, `receipt.json`, `occurrence-interface.json`, `task_regimes/`,
the task's `physics:` block.

## 2. What the real cell showed, and what the task declares

Source of every number: `../realcell_env/MEASURED_CELL.yaml` (keys cited in `task.yaml`). Frames: twin world = the
table-top plane under `base_link`; from UR `base`: `p = (-x, -y, z + 0.080)`, yaw + 180 deg.

| real cell (measured) | twin control / flat_gen | this task |
|---|---|---|
| base_link 0.080 m above the table (config.py:3; Vention design 80.48 mm; operator 2026-10-23: 0.080) | base_link AT the table plane (dimelab, ur5e_rg2_bench: z 0.0) | bench `dimelab_elevated`, z 0.08 |
| base1 untaped; pushed to a new pose every cycle; creeps 1-4 mm per part; re-measured before every insert | Body1 `static` at (0.005, 0.48), yaw 0 | Body1 **free, drawn** in `base_start`, yaw free |
| base1 sat, over 114 reads, inside world x [-0.268, 0.020], y [0.327, 0.629] (footprint union + 5 mm) | assembly region x [-0.15, 0.16], y [0.36, 0.60] | `assembly` and `base_start` = the measured rectangle |
| staging region UR x [0.19, 0.62], y [-0.62, -0.25] = world x [-0.62, -0.19], y [0.25, 0.62]; parts kept >= 25 mm apart | staging x [-0.51, -0.19], y [0.22, 0.54] | `staging` = the cell's, its +x edge 25 mm clear of `base_start` (x <= -0.293) |
| one clear area for regrasps at UR (-0.320, -0.500) = world (0.320, 0.500) | reorient x [0.19, 0.51], y [0.22, 0.54] | `reorient` = that point +- 56.4 mm (half the longest part) |
| untipped cycles (runs 93-108): every part in its seated orientation up to yaw; tipped cycles (run 109 on): U parts / yellow knocked flat; line_green always on a long face | control: authored (standing) frame; flat_gen: laid flat, flips free | flat_gen's domains (tipped). Fallback B = standing |
| perception: wrist camera, in view only, ~2.6 Hz, held part occluded, base re-read per mate | `/objects/poses` 30 Hz, everything, always | cannot be declared in a sim task - the `prompt.task_layer` tells the planner to structure for it (one `insert_mate` function; base from the stream per mate; held pose chained) |
| insertion: force-controlled search (`/ur5e/drive_mate`) | the planner's own position descent | the planner's own descent, isolated behind `insert_mate(mate_id, reference_pose, held_pose)` - the M4 substitution point |

What is NOT carried, on purpose: the real cell's referee limits (its certifier judges z only), its 40 N fixed grip,
wall-clock time, the Vention slider. Those are the real cell's and live in `../realcell_env/` for the cell-side
adapter (`OWED.md` §2), not in a simulated task.

## 3. On dual-a4500, in order

Conventions below: `W=~/prompt2assembly`, `PA=$W/lane-publication/.venv/bin/pa`, `export PA_WORKSPACE=$W`.
Skills: `$W/plan-anything/.claude/skills/{onboard-environment,author-task,dispatch-job}/SKILL.md` (line numbers
as of plan-anything `4b4bbc545`).

### 0. Get the pack

```bash
git clone -b dev --depth 1 https://github.com/inbarajaldrin/ros-mcp-server.git /tmp/rms-dev   # or rsync from the cell PC
cp -r /tmp/rms-dev/compliant_insertion_studio/m4_dispatch ~/m4_dispatch
```

### 1. Open a data lane (author-task SKILL.md:1690-1760 "The lane, end to end")

```bash
$PA lane create --kind data m4-dimelab-elevated-20261026      # reads the CURRENT generation as the base
```

Both entries are NEW (not in the published base), so there is nothing to `declare_data_paths` for them: create
`<lane>/staging/environments/dimelab_elevated/` and `<lane>/staging/tasks/fmb1_assemble_ur5e_rg2_realcell_gen/`
and copy the pack's files in. (If the lane manager insists on a declared path, declare `environments/dimelab` and
`tasks/fmb1_assemble_ur5e_rg2_flat_gen` to seed the directories, then rename.) Edit inside `staging/`, never in the
checkout (author-task SKILL.md:1690: "Every edit under data/ reaches the generation ONLY through a data lane").

### 2. Publish the environment (onboard-environment SKILL.md:455-600, the staged-declaration route)

`--copy-of dimelab` is **refused** here: `publish_environment.py:1404-1411` refuses `ENVIRONMENT_COPY_DIVERGES`
unless the copy equals the source "at every path but the name", and `base_pose` differs. So this is a FULL
publication of a new bench, which needs its own gate:

```bash
cd $W/robot-integration
# the four environment facts (SKILL.md:470-500): the Isaac binary validate_home's lock names TODAY (5.0.0 lock until
# the operator's PHASE 2 promote; after it the record's platform install, `sim-instrumentation/scripts/pa-box-path
# isaac_python`), PA_KIT_MCP_PATH, BOTH launch packages, PA_WORKSPACE. Read SKILL.md:470-500 before running.
SIMCELL_ISAACSIM_BIN=<per the lock> PA_KIT_MCP_PATH=$HOME/Documents/pa-footprint/omni-kit-mcp \
PYTHONPATH=src/pa_robot_contract python3 -m pa_robot_contract \
  --composite-repo $HOME/Documents/pa-footprint/composite-root \
  gate <lane>/staging/environments/dimelab_elevated/declaration.yaml \
  --launch-package src/pa_robot_bringup --launch-package src/pa_rg2_adapter --json \
  > <lane>/staging/environments/dimelab_elevated/gate.json        # INSIDE the lane (SKILL.md:548-550)
cd $W/plan-anything
python3 scripts/publish_environment.py \
  --declaration <lane>/staging/environments/dimelab_elevated/declaration.yaml \
  --binding     <lane>/staging/environments/dimelab_elevated/sim-binding.yaml \
  --gate        <lane>/staging/environments/dimelab_elevated/gate.json \
  --publisher "/onboard-environment, dimelab_elevated (M4, real-cell base height)" --host dual-a4500 \
  --module $W/robot-integration
```

Expect and handle (all from the skill): a sidecar/composite pin refusal ("hashes to X where the declaration pins
Y") -> `--repin-robots <ENV_DIR>` first (SKILL.md:451-460; dimelab's pins are `5a2d9d59…`/`d91b5a68…` and were current
on 2026-10-23); `ENVIRONMENT_DECLARES_DEVICE` cannot occur (dimelab's declaration carries no device lines, and this
is the same file). **Honesty note (SKILL.md:506-535):** a static gate certifies names and assets; the live-cell
certificate comes from a bring-up. dimelab itself carries a `regate_owed` stamp today (`gate_predates_sidecar_attestation`,
copied into `ur5e_rg2_bench` on 10-23), so `dimelab_elevated` will carry the same maturity, stamped by the publisher.
The honest discharge - a dispatched sim-only spike `purpose: environment-regate` on the lane's overlay, then a full
publish with that gate - is OWED and is the same debt dimelab has.

### 3. Publish the task (author-task SKILL.md:811-840 path 2, :1085-1112 PUBLISH)

First exercise the engine on the staged home - this is what admits a drawn Body1 or refuses it:

```bash
cd $W/sim-instrumentation
for s in $(seq 0 49); do
  python -m sim_instrumentation.bundle_layout --home <lane candidate data home> \
     --task fmb1_assemble_ur5e_rg2_realcell_gen --seed $s --out /tmp/m4draws/draw$s.json 2>/tmp/m4draws/draw$s.err \
     && echo "seed $s admitted" || { echo "seed $s REFUSED: $(grep PA_REFUSAL /tmp/m4draws/draw$s.err)"; }
done
```

(`bundle_layout.py:434-482`: `--home --task --seed --out`; a refusal prints `PA_REFUSAL <code>` and exits 2.)
Read the `engine` digest the first admitted draw prints: if it is not `778eed5a…`, re-pin `layouts.generator.sha256`
in the staged `task.yaml`. Record the reviewed `{seed, record_sha256}` pairs under
`admission.evidence.author_engine_exercise` (SKILL.md:539-545 - never under `layouts:`). Expected refusals and what
they mean:

| refusal | meaning | do |
|---|---|---|
| anything naming `occurrence:00` (`LAYOUT_DOMAIN_UNDECLARED`, a seat/rest refusal) | the engine will not draw the reference part | `fallback/README.md` A (Body1 static at the measured pose) |
| `LAYOUT_DRAW_UNADMITTED` on some seeds | parts crowd `staging` at 25 mm (flat_gen saw 3 of 50) | a typed refusal, acceptable; many -> widen `staging` toward -x (the cell's region runs to x -0.62 already) |
| `LAYOUT_REGION_ANCHOR_*` | the region anchor is not `work_table` with a world pose | should not happen: anchors are flat_gen's |

Then publish:

```bash
cd $W/plan-anything
python3 scripts/publish_task.py <lane>/staging/tasks/fmb1_assemble_ur5e_rg2_realcell_gen
```

It re-pins `environment.receipt_sha256` from the lane's post-image (the `PUBLISH_FIRST` placeholder is why step 2
runs first - "robot -> environment -> task -> assembly LAST", SKILL.md:567-577), derives `physics:` and writes
`gate.json`/`receipt.json`. The admission is `asserted` (a publishable pilot that flips to `proven` on the first
clear, SKILL.md:762-803); the request in step 5 is that first clear's chance.

### 4. Seal and promote

```bash
$PA lane seal m4-dimelab-elevated-20261026
$PA lane promote --dry-run m4-dimelab-elevated-20261026     # readers run first (SKILL.md:1745-1749); fix red before promoting
$PA lane promote m4-dimelab-elevated-20261026               # the operator's step
git -C $W/plan-anything for-each-ref refs/pa/generations --sort=-committerdate --format='%(refname:short)' | head -1
```

The last line is the `generation_digest` for step 5.

### 5. Fill and walk the request

`requests/authoring-fmb1-realcell-gen.yaml`: set `generation_digest`, `harbor.seat`, and (if not Opus 5.5)
`harbor.model` + `series.name` + `id`. Then the readiness walk (dispatch-job SKILL.md:16-45):

```bash
cd $W
python3 plan-anything/scripts/harbor_ready.py --request ~/m4_dispatch/requests/authoring-fmb1-realcell-gen.yaml --box dual-a4500
```

Exit 0 = every rung passed. Read a remedy sentence as a diagnosis against the thing it names.

### 6. Fire and watch (dispatch-job SKILL.md:1478-1492, :1562-1576)

The dispatching shell declares the relay upstreams the enclosure needs - for Claude authoring `PA_MODEL_UPSTREAM`
and `PA_MIRROR_UPSTREAM`, "resolved at dispatch time (never pasted)"; missing ones refuse `ENCLOSURE_NETWORKS_UNRAISED`
(the operator's shell has done this for every October run; the 10-13 requests used
`env $(python3 <ws>/.pa/relay_env.py <request>)`). Nothing about pacing or render in the shell.

```bash
$PA dispatch harbor ~/m4_dispatch/requests/authoring-fmb1-realcell-gen.yaml
ID=m4--fmb1_assemble_ur5e_rg2_realcell_gen--claude-opus-5-5--attempt-01
ROOTS=$(grep -rlx "id: $ID" $PA_WORKSPACE/.pa/runs/*/dispatch-request.yaml); test "$(echo "$ROOTS" | wc -l)" -eq 1
python3 plan-anything/scripts/list_runs.py --show <job>          # the dossier; ends with the trial dir
python3 plan-anything/scripts/watch_agent.py <trial-dir> > /tmp/m4_agent.txt   # the agent's live stream
```

Poll the dispatch receipt's `queue.terminal_state == complete` and `leases.released`. A `[quota]` line is the seat's
meter; an exhausted seat ends the run as a PAUSE, resumed by its seat event (SKILL.md:55-100), never by hand.

### 7. After the build

- The delivery is named by digest: the receipt's `authoring.admitted_delivery` (`pa delivery show <digest>`).
- **Scoring it later needs a seed panel**: a generated task with no `layout_panels.<task>` in
  `data/benchmarks/b1/protocol.yaml` refuses `LAYOUT_PANEL_UNDECLARED` (author-task SKILL.md:1028-1041). Draw nine
  once, in a data lane: `python -m lane_publication.dispatch.layout_panel draw 9`, then
  `python -m lane_publication.dispatch.layout_panel check plan-anything/data` must print `"ok": true`. Authoring
  does not need it.
- **M4 on the cell** (the part this pack does not do): carry the delivery to the cell PC, replace `insert_mate` by a
  call to `/ur5e/drive_mate` (`seams/client.py`; goal `{mate_id, target VisionPose(base), held VisionPose(tool0)}`),
  and run it against the surface in `../realcell_env/INTERFACE.md`. The twin-surface adapter (`/ur5e_0/...` names,
  `/objects/poses` in twin world) is still OWED (`../realcell_env/OWED.md` §2); until it exists the planner's topic
  names are substituted by hand too.

## 4. Decisions made here (the operator may overturn any of them; each is one line to change)

1. **A new bench `dimelab_elevated`, not dimelab edited in place.** dimelab is pinned by five published tasks
   (beam_assemble, fmb1_assemble, jenga x2, machinery_conformance_pose); `ur5e_rg2_bench` (dimelab copied on 10-23
   for the paper, base at z 0) is pinned by the live paper task `fmb1_assemble_ur5e_rg2_flat_gen`. Raising either
   re-pins and re-gates work the paper is mid-way through. A new name pins nothing else. The operator's own precedent
   is the 10-23 copy ("literally just the name changes"); this one changes the name and one number.
2. **Body1 free and drawn** (the M4 behaviour), with the static-at-measured-pose fallback ready (`fallback/` A).
3. **Tipped (laid-flat) domains**, flat_gen's, because the cell's phase-2 cycles start that way and a planner that
   handles a flat part handles a standing one; `fallback/` B is the standing start.
4. **`admission: asserted`** with the real-cell runs as the recorded evidence, not `proven`: nothing has cleared this
   task in simulation. An asserted task is dispatchable for authoring - flat_gen's own `gate.json` reads
   `dispatchability.authoring: false` while twelve authoring runs were dispatched against it, and the dispatcher
   (`lane-publication/lane_publication/dispatch/`) carries no admission-class gate (grepped 2026-10-26).
5. **Request = the 10-13 Opus 5.5 request** (`evaluate_planner_road: in_cell`, `seat_swap: true`, `render: off`),
   host dual-a4500 named by the operator. Model is the operator's call; Opus 5.5 is the pair that cleared flat_gen.
6. **`prompt.task_layer` carries the seam instruction** (four sentences). author-task SKILL.md:75-80 says the layer
   is for the exceptional run-specific instruction only; the M4 substitution is exactly that.

## 5. Open items (not settled here; each says who/what settles it)

| item | settles it |
|---|---|
| Can the engine draw the reference part (free Body1)? | step 3's exercise on dual-a4500; else fallback A |
| The cell->graph frame map for Body1 (Rz(-90), yaw + 90 deg) is translation-checked only | compare one admitted draw's Body1 against the real photo of base1 at the same yaw; `MEASURED_CELL.yaml parts.frame_map_cell_to_graph` |
| The static gate's maturity (`regate_owed`) | a dispatched sim-only regate spike on the lane overlay (onboard-environment SKILL.md:506-535) - the same debt dimelab carries |
| The generator engine digest at head | the exercise prints it; re-pin if it moved from `778eed5a…` |
| The task name (the convention has no variant slot) | the operator (author-task SKILL.md:823-826) |
| The 0.48 mm between config (0.080) and the Vention design (80.48 mm) | not used anywhere on the cell; the twin carries 0.080 |
| Scoring panel | step 7, only when the delivery is to be scored |

## 6. Provenance of the copied bytes

| bytes | where from | sha256 at copy |
|---|---|---|
| dimelab declaration | `dual-a4500:~/prompt2assembly/plan-anything/data/environments/dimelab/declaration.yaml` (verified 2026-10-26 23:2x) | `a10d8ebab67069d71746db54dc192db3f32208ff076651d014b478b73bd7146c` |
| dimelab sim-binding | same dir | `dc3d5e00f9660b3ed30956a4d4bf188a8ac420a771be927c60004c0a17072185` |
| dimelab rig_manifest | same dir, `assets/` | `a982d39d27162e70e188f0aea6c28e1008a0dc2907c22cb36b8b1bd91bfe1476` |
| flat_gen task.yaml (the sibling template) | `data/tasks/fmb1_assemble_ur5e_rg2_flat_gen/task.yaml` @ 538bbeee | `f6bb8aa54131045fb0bfed925a519c1ca6165c98304c5e8339753397c4bda275` |
| prompt pins | `data/prompts/{platform,assembly}.md` @ 538bbeee | `de728a1d…` / `d6be4fa5…` (unchanged since the flat_gen pins) |
| the authoring request shape | `.pa/runs/02f640fb61b3df43fbce0867d1b99048f1cf03b2dc19a5bd06382ff8f00c3b41/dispatch-request.yaml` | (not hashed; copied field by field) |
| real-cell numbers | `../realcell_env/MEASURED_CELL.yaml` (commit 00380ea) | - |

The local mirror with the read-only copies: `~/Documents/dimelab_env_elevated/` on the cell PC (`MANIFEST.md`;
`reference/current-20261026/` holds the 538bbeee copies of the two FMB1 tasks, `ur5e_rg2_bench` and the protocol).
