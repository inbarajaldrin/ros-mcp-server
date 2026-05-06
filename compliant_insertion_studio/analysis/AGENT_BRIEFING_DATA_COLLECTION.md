# Briefing: regime-decoding data collection (next-session agent)

You are taking over a single, narrow task: **drive the operator through 15 guided u_orange insertion demos and verify each demo's data was captured.** Do NOT read CLAUDE.md or pre-load any other project context — your only job is to run the collection script and confirm outputs. Treat anything else as out of scope.

## What's already true (don't re-derive)

1. The robot bringup is up. `launch_robot.sh real` was run earlier and now automatically calls `set_payload` with the calibrated mass + CoG. Force-mode gravity comp is in place. **TCP drift during APPROACH is ~0.83mm** (was 7.85mm before the fix earlier today).
2. The operator is at the robot. Pendant is in Local mode. Camera perception (aruco) is publishing both topics. Part u_orange is currently held by the gripper.
3. The collection script does everything per-demo: rotate_object, hover (with offset injection), force-mode entry, telemetry logging. The operator guides each insertion manually and Ctrl+Cs when done.
4. The chained held_quat preserves R_grasp's fold info; rotate_object converges EE to canonical from any starting pose. **Don't propose changes to held_quat handling.** The default in the script is the validated -X-fold value.

## What you do

### Step 1 — confirm the operator is ready

Just say: "Pendant green and ready? I'll launch the collection script when you say so." Wait for confirmation.

### Step 2 — launch the script

Run:

```bash
python3 -m compliant_insertion_studio.scripts.collect_regime_data \
  --object u_orange --base base1 --grasp-id 1
```

The script prints what's coming next and waits for the operator's [Enter]. It runs each demo as a synchronous subprocess. Don't wrap it in `run_in_background` — the operator interacts with the subprocess via the terminal.

### Step 3 — between demos

After each demo completes, the script prints:

- `✓ basename=insert_u_orange_<ts>  outcome=<done|abort|...>`
- Or `✗` if a CSV wasn't produced

Just observe. Don't intervene unless the operator asks. Don't propose fixes between demos. The script handles cleanup.

If a demo fails (no CSV, or wrapper aborts):

- The script asks "Press [Enter] to launch this demo, or 'skip' to skip, 'quit' to stop."
- You can re-run the same variation by just pressing Enter again. The script's manifest tracks which variations have ≥3 successful demos.

### Step 4 — when all 15 are done

The script prints a summary and writes the manifest to `compliant_insertion_studio/logs/regime_collection_<ts>.json`. Print that path back to the operator, plus the per-variation count.

### Step 5 — verify the data

Run a quick sanity check that all 15 demos have full sidecar bundles:

```bash
python3 -c "
import json, glob, os
manifest_path = sorted(glob.glob('compliant_insertion_studio/logs/regime_collection_*.json'))[-1]
m = json.load(open(manifest_path))
print(f'Session: {manifest_path}')
print(f'Demos: {len([d for d in m[\"demos\"] if d.get(\"action\")==\"ran\"])}')
for d in m['demos']:
    if d.get('action') != 'ran': continue
    bn = d['basename']
    sidecars = [f'compliant_insertion_studio/logs/{bn}.{s}.csv' for s in ('joints_raw','wrench_raw','cmd_wrench_raw','fm_events')]
    missing = [s for s in sidecars if not os.path.exists(s)]
    flag = '✓' if not missing else '✗'
    print(f'  {flag} {d[\"label\"]:<20s} {bn}  outcome={d[\"outcome\"]}  missing={missing or \"none\"}')
"
```

Report any rows with `✗` to the operator.

### Step 6 — stop

Tell the operator: "All N demos captured. Manifest at <path>. Ready to hand off to the regime-analysis session."

Do NOT propose analysis here. Do NOT segment the data. Do NOT compare to GOLD. Your context will be cleared after this — the regime analysis happens in a separate fresh agent run.

## Anti-patterns

- **Do not** read or apply anything from `CLAUDE.md`, the skill files, `REGIME_DECODING.md`, or any of the iteration history. They're not relevant to this narrow task and the context budget is reserved for the analysis session that comes after.
- **Do not** propose code changes to the wrapper, FSM, or rotate_object. Everything you need is already in place.
- **Do not** run analysis scripts (`30_segment_regimes.py` etc) — that's the next agent's job.
- **Do not** add new variations to the script unless the operator explicitly asks. The 5×3=15 design is set.
- **Do not** hold the part with `--bootstrap-from-tcp` — that flag was removed because it flipped the fold.

## If the operator changes scope

If the operator asks for something beyond running the script + reporting outcomes (e.g., "also run the analysis", "check this other thing", "compare to GOLD"), say:

> That's part of the regime-analysis task and will be handled in a separate session with full project context. For now I'm just running the collection.

Then ask them to confirm before doing anything outside collection.

## Files referenced

- Collection script: `compliant_insertion_studio/scripts/collect_regime_data.py`
- Per-session manifest output: `compliant_insertion_studio/logs/regime_collection_<ts>.json`
- Per-demo CSVs + sidecars: `compliant_insertion_studio/logs/insert_u_orange_<ts>.{csv, joints_raw.csv, wrench_raw.csv, cmd_wrench_raw.csv, fm_events.csv, meta.json}`
