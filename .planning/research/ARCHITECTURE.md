# Architecture Research

**Domain:** Compliant peg-in-hole insertion subsystem inside an existing ROS2 MCP server (ros-mcp-server). Greenfield subsystem inside a brownfield repo.
**Researched:** 2026-05-01
**Confidence:** HIGH for component boundaries / file layout / build order (validated against existing repo conventions and the working `compliant_insert.py` placeholder). MEDIUM for the runtime classifier question (intentionally deferred per PROJECT.md — depends on what the data shows).

---

## Standard Architecture

### System Overview

```
┌──────────────────────────────────────────────────────────────────────────┐
│  CALLER LAYER (orchestration — already exists, do not modify shape)      │
│                                                                          │
│   server_p3.py:translate_object("insert", base_name=..., grasp_id=...)   │
│                              │                                           │
│                              ▼                                           │
│   primitives/translate_object.py main() --insert --mode real             │
│      ├─ translate_for_target_real(...)        ← hover above base         │
│      └─ run_perform_insert_real(args)         ← THE INTEGRATION SEAM     │
│                              │                                           │
└──────────────────────────────┼───────────────────────────────────────────┘
                               │  subprocess.Popen(...)
                               ▼
┌──────────────────────────────────────────────────────────────────────────┐
│  DISPATCHER (new, thin)                                                  │
│                                                                          │
│   primitives/compliant_insert_episode.py                                 │
│      ├─ resolve_config(object_name)  ──►  insert_configs/<obj>.yaml      │
│      │     fallback chain: <obj>.yaml → defaults.yaml → MANUAL_GUIDED   │
│      ├─ pick mode: COLLECTING (--collect) | RUNTIME (default)            │
│      └─ delegate to compliant_insert.py with resolved params + meta      │
└──────────────────────────────┬───────────────────────────────────────────┘
                               │  in-process (same module) OR subprocess
                               ▼
┌──────────────────────────────────────────────────────────────────────────┐
│  EPISODE WRAPPER (new — owns lifecycle)                                  │
│                                                                          │
│   primitives/compliant_insert.py  (already a placeholder ~250 LOC)       │
│      PRE → HOVER → ZERO → ACTIVE → DONE/ABORT → SAFE-HEIGHT-THEN-HOME    │
│                                                                          │
│   Subscriptions: /tcp_pose_broadcaster/pose                              │
│                  /force_torque_sensor_broadcaster/wrench                 │
│                  /gripper_width                                          │
│   Services:      /force_mode_controller/start_force_mode                 │
│                  /force_mode_controller/stop_force_mode                  │
│                  /io_and_status_controller/zero_ftsensor                 │
│                  /controller_manager/switch_controller (via CLI)         │
│   Signals:       SIGTERM=success, SIGUSR1=event_marker toggle,           │
│                  SIGUSR2=re-zero, custom=abort                           │
│                                                                          │
│   Outputs (per episode):                                                 │
│      logs/insert_<obj>_<YYYYMMDD_HHMMSS>.csv                             │
│      logs/insert_<obj>_<YYYYMMDD_HHMMSS>.meta.json                       │
└──────────────────────────────┬───────────────────────────────────────────┘
                               │ writes telemetry
                               ▼
┌──────────────────────────────────────────────────────────────────────────┐
│  STATIC DATA STORE (filesystem; no DB, no server)                        │
│                                                                          │
│   logs/                            (raw + meta, .gitignored)             │
│      insert_*.csv                  100 Hz pose+wrench+phase+marker      │
│      insert_*.meta.json            target pose, params used, outcome    │
│                                                                          │
│   primitives/insert_configs/       (committed; small YAML diffs)         │
│      defaults.yaml                 universal floors / fallbacks         │
│      u_brown.yaml                  per-object overrides                 │
│      u_orange.yaml                                                       │
│      line_green.yaml                                                     │
│      inverted_u_yellow.yaml                                              │
└──────────────────────────────┬───────────────────────────────────────────┘
                               │ read offline
                               ▼
┌──────────────────────────────────────────────────────────────────────────┐
│  ANALYZER (new — offline, single-page web app)                           │
│                                                                          │
│   tools/analyze_inserts.html                                             │
│      Plotly.js (CDN) + vanilla JS, no build step, no server.             │
│      User drag-drops the logs/ folder OR uses File System Access API.    │
│      Auto-pairs CSV ↔ meta JSON by filename stem.                        │
│      Views: per-episode plots, cross-episode overlay,                    │
│             per-object signature card (stats → config values).           │
│                                                                          │
│   Output: human-derived YAML edits committed under insert_configs/       │
│   (no automated config writer in PoC — operator types the values).       │
└──────────────────────────────────────────────────────────────────────────┘
```

### Component Responsibilities

| Component | Responsibility | Implementation |
|-----------|----------------|----------------|
| `translate_object.py:run_perform_insert_real` | Existing function; **only change**: swap script_path target | Edit `:1085` to point at new dispatcher script |
| `compliant_insert_episode.py` (new dispatcher) | Resolve YAML → params; decide mode (collect vs runtime); spawn/import wrapper; surface fallback to operator | Thin script (≤150 LOC); no ROS state itself |
| `compliant_insert.py` (existing, evolves) | Episode lifecycle, ROS subscriptions, force-mode RPCs, CSV/meta writing, signal handling, safe-height exit | Already exists as scaffold; add HOVER/PRE phases, meta JSON, parametric `force_mode_params` |
| `insert_configs/defaults.yaml` + `<obj>.yaml` | Per-object parameter overrides for one universal algorithm | Plain YAML, deep-merge: defaults <- per-object |
| `logs/insert_*.csv` + `*.meta.json` | Raw telemetry + episode metadata; sole input to analyzer | Filesystem only, `.gitignore` entry |
| `tools/analyze_inserts.html` | Offline visualization → human-derived parameter values | Static HTML, Plotly.js CDN, FileReader / drag-drop |
| `primitives/shared/insert_config.py` (new helper) | YAML load + deep-merge + validation; importable by dispatcher *and* by ablation tooling | ≤80 LOC; no ROS deps |

**Explicit non-components (over-engineering avoided):**

- **No runtime classifier.** Family selection is offline-only in PoC. Object name → YAML lookup is the entire "decide which family this is" path at runtime. PROJECT.md flags statistical classification as a research carve-out *after data collection*, not a Phase 1 deliverable.
- **No config DB.** YAML on disk + `git diff` is the audit trail.
- **No ros2 launch / lifecycle node.** A plain `subprocess.Popen` from `translate_object.py` (matching how `move_to_clear_area`, `move_down`, etc. are already invoked) is sufficient and matches existing repo convention.
- **No telemetry bus / message types.** CSV + sidecar JSON beats a custom ROS message; the analyzer is offline anyway.

---

## Recommended Project Structure

```
ros-mcp-server/
├── primitives/
│   ├── compliant_insert.py              # ← evolves: episode wrapper (lifecycle owner)
│   ├── compliant_insert_episode.py      # ← NEW: dispatcher / config resolver / collect-mode
│   ├── insert_configs/                  # ← NEW: per-object YAML
│   │   ├── defaults.yaml
│   │   ├── u_brown.yaml
│   │   ├── u_orange.yaml
│   │   ├── line_green.yaml
│   │   └── inverted_u_yellow.yaml
│   ├── shared/
│   │   └── insert_config.py             # ← NEW: load + deep-merge helper (no ROS deps)
│   ├── translate_object.py              # ← edit ONE line (:1085) to retarget dispatcher
│   ├── core/                            # unchanged
│   ├── _real_mode_stash/                # untouched, reference-only
│   └── ... (existing primitives unchanged)
├── tools/                               # ← NEW directory (does not exist today)
│   └── analyze_inserts.html             # ← NEW: single-page Plotly dashboard
├── logs/                                # exists; add .gitignore entry for insert_*
│   ├── insert_<obj>_<ts>.csv
│   └── insert_<obj>_<ts>.meta.json
└── server_p3.py                         # unchanged (translate_object tool stays)
```

### Structure Rationale

- **`primitives/compliant_insert*.py` (two files, not one).** Keeping `compliant_insert.py` as the lifecycle/ROS-node module and `compliant_insert_episode.py` as the dispatcher is worth the extra file because:
  1. The wrapper holds rclpy state and is the thing under test in real-mode debugging — operators will run it directly with `--object-name X --fz N` for tuning sessions. Stuffing dispatcher logic into it bloats the part you reload most often.
  2. Matches the stash's prior split (the stash had `prismatic_peg_insertion.py` calling out to `peg_in_hole_insert.py` urscript — same intuition).
  3. The dispatcher has no ROS imports; it can be unit-tested without a live robot.
  4. Subcommand conflation (one file, `--mode collect|run`) muddies argparse and makes the operator-facing flag set noisier than necessary.

  **Counter-trap:** do **not** also split out a `force_mode_client.py` — the SetForceMode RPC is small enough (~30 LOC) to live inline in the wrapper, and a third file just creates more import bookkeeping.

- **`primitives/insert_configs/` (committed) vs `logs/` (gitignored).** Configs are source of truth (small, version-controlled, diff-readable). Logs are raw telemetry (binary-ish in volume, regenerable from a demo session). Putting configs *under primitives/* (vs at repo root) keeps the per-object parameter set co-located with the code that consumes it — matches how `primitives/shared/config.py` holds the global constants today.

- **`tools/` at repo root.** Not under `primitives/` because the analyzer is not a robot primitive — it's an offline operator tool. New top-level dir is justified (none exists today; the dashboard is the first member; future analyzers go here too).

- **Reuse `logs/`** at repo root, not a new `data/` dir. Logs already exist there (`grasp_publisher.log`, the existing `insert_u_brown_*.csv` files from the placeholder). `.gitignore` already covers `debug_logs/`; add `logs/insert_*.csv` and `logs/insert_*.meta.json` to it.

---

## Architectural Patterns

### Pattern 1: Subprocess Boundary Between Orchestrator and Primitive

**What:** `translate_object.py` calls insertion via `subprocess.Popen` with `__RESULT_JSON__` sentinel parsing for return values. Same pattern as `run_move_to_clear_area`, `run_move_down`.

**When to use:** Always, in this repo. It's the established convention and gives free process isolation (a force-mode controller leak in the wrapper does not poison the parent's rclpy context).

**Trade-offs:** Slower than in-process call (~200 ms subprocess startup), but irrelevant against multi-second insertions. JSON sentinel parsing is brittle but already battle-tested across the repo.

**Example:**
```python
# In translate_object.py:run_perform_insert_real (edit only the script_path line)
script_path = os.path.join(os.path.dirname(__file__), 'compliant_insert_episode.py')
cmd_args = ['--object-name', args.object_name,
            '--base-name', args.base_name,
            '--grasp-id', str(args.grasp_id),
            '--final-base-pos', *map(str, args.final_base_pos),
            '--final-base-orientation', *map(str, args.final_base_orientation),
            '--current-object-orientation', *map(str, args.current_object_orientation)]
return run_subprocess(script_path, cmd_args)
```

### Pattern 2: Defaults + Per-Object Override Deep-Merge

**What:** Single `defaults.yaml` holds universal values; `<object>.yaml` holds only the keys that differ. Resolver does a recursive dict merge.

**When to use:** Whenever you have N objects sharing 80%+ of their parameter set. Adding a part is then a 10–20-line YAML diff, not a full file copy.

**Trade-offs:** Operators must understand merge semantics ("missing key = inherit from defaults"). Mitigation: validation step that prints the *resolved* effective config so the operator sees what's actually in effect.

**Example:**
```yaml
# insert_configs/defaults.yaml
hover_offset_z: 0.05
fz_target: 3.0
force_mode:
  selection: [1, 1, 1, 1, 1, 1]
  gain: 1.0
  damping: 0.005
  lin_speed: 0.02
  ang_speed: 0.20
termination:
  z_reached: { delta_m: 0.015, weight: 1.0 }
  motion_stopped: { window_s: 1.5, max_dz_m: 0.0008, weight: 1.0 }
  force_absorbed: { fz_min_n: 0.5, window_s: 1.0, weight: 1.0 }
  combine: any   # any|all|weighted
retry: { max_retries: 1, retract_m: 0.005 }

# insert_configs/u_brown.yaml  (overrides only)
fz_target: 3.0
force_mode:
  selection: [1, 1, 1, 0, 0, 0]   # XYZ compliant, RxRyRz locked for u_brown
termination:
  combine: weighted
```

### Pattern 3: Filesystem-as-Telemetry-Bus

**What:** Episode wrapper writes a `.csv` + `.meta.json` pair per run. Analyzer reads those files; nothing else does. No ROS bag, no DB, no message broker.

**When to use:** Single-operator, single-instance data collection where total dataset is sub-GB and only humans (and one HTML page) consume it.

**Trade-offs:** Naive vs. ROS bag — you lose perfect timestamp fidelity and replay-into-ROS capability. Acceptable given the analyzer is the only downstream consumer and rebag-from-CSV is unnecessary for parameter derivation.

### Pattern 4: Offline Classification, Online Lookup

**What:** "Which family is this object?" is decided once, by a human staring at the dashboard, and the answer is committed as a YAML file. At runtime, the dispatcher does a literal `object_name → file lookup`. No model loading, no inference path on the robot.

**When to use:** Always, for any system where parameters are stable per-object and the object identity is already known from upstream (here: the LLM/agent passes `object_name` explicitly).

**Trade-offs:** Cannot generalize to a new object without operator intervention. **This is the right trade-off** for PoC — the manual-guidance fallback (Active requirement: "Behavior on unknown object") is the planned escape hatch, and it doubles as the data-collection path for the new part. Avoids "silently using wrong defaults on a part the system has never seen."

---

## Data Flow

### Episode Collection Flow (operator-driven, ~5 demos per object)

```
operator runs:
  python3 primitives/compliant_insert_episode.py \
      --object-name u_brown --base-name fmb1_base \
      --grasp-id 0 --collect --user-notes "guided in by hand, slight CCW"
       │
       ▼
[dispatcher] resolves config:
   primitives/insert_configs/u_brown.yaml ∪ defaults.yaml
   if u_brown.yaml missing → MANUAL_GUIDED stub config (full 6-DOF, no auto-terminate)
       │
       ▼
[wrapper compliant_insert.py] PRE → HOVER → ZERO → ACTIVE
   /tcp_pose + /wrench → 100 Hz CSV writer (with phase + event_marker columns)
   operator pushes part by hand; presses Ctrl-Z to send SIGUSR1 (toggle marker)
       │
       ▼
operator presses Ctrl-C (or sends SIGTERM) → DONE phase
   wrapper:
     1. stop_force_mode
     2. switch back to scaled_joint_trajectory_controller
     3. move_to_safe_height (REUSE existing primitive)
     4. flush CSV, write .meta.json
     5. prompt for free-text notes (interactive via stdin)
       │
       ▼
artifacts on disk:
   logs/insert_u_brown_20260501_184432.csv      (100 Hz, ~10–60 s of data)
   logs/insert_u_brown_20260501_184432.meta.json (params, target, outcome, notes)
```

### Offline Analysis Flow (human-in-the-loop, ~30 min per object)

```
operator opens tools/analyze_inserts.html in Firefox/Chrome
       │
       ▼
[dashboard] drag-drop logs/ folder
   - parse all *.csv → in-memory time-series
   - parse all *.meta.json → episode metadata
   - pair by filename stem
       │
       ▼
operator inspects:
   - per-episode: F/T traces, Z(t), F-vs-Z phase plot, 3D path
   - cross-episode overlay: filter by object + outcome → spot signature shape
   - signature card: median Fz at success, |Tx|/|Ty| peak distributions, etc.
       │
       ▼
operator types values into primitives/insert_configs/u_brown.yaml
   (no automated YAML writer in PoC — explicit human decision)
       │
       ▼
git commit primitives/insert_configs/u_brown.yaml
   (audit trail = git log over insert_configs/)
```

### Runtime Flow (autonomous, post-tuning)

```
LLM agent → server_p3.py:translate_object("insert", base_name="fmb1", grasp_id=0)
       │
       ▼
[server_p3] auto-injects current_object_orientation from session state
       │
       ▼
primitives/translate_object.py --insert --mode real ...
   ├─ translate_for_target_real(...)         ← hover above base (existing code)
   └─ run_perform_insert_real(args)          ← edited to spawn new dispatcher
       │
       ▼
primitives/compliant_insert_episode.py
   ├─ resolve config: u_brown.yaml ∪ defaults.yaml
   │     IF NOT FOUND → emit __RESULT_JSON__ with
   │        {"result":"failure","error":"no config for object 'X'",
   │         "hint":"run --collect to gather demos and create insert_configs/X.yaml"}
   │     bubble through translate_object → server → operator/agent
   └─ spawn wrapper with resolved force_mode_params + termination policy
       │
       ▼
[wrapper] PRE → HOVER → ZERO → ACTIVE → AUTO-TERMINATE per termination policy
   on success: DONE → safe-height → home → __RESULT_JSON__ {result:success,...}
   on failure: ABORT → retract per retry policy → re-attempt OR final failure
```

### Key Data Flows

1. **Telemetry write path:** wrapper subscribes to `/tcp_pose` + `/wrench` at sensor rate, writes 100 Hz row to CSV → flushed on episode end.
2. **Config read path:** dispatcher loads `defaults.yaml` once + `<obj>.yaml` once at startup, deep-merges, passes resolved dict (or CLI flags) to wrapper. **Eager-load only the requested object** (lazy-load by name, not all-load) — eager-load-all is wasteful and surfaces YAML parse errors for objects unrelated to the current run.
3. **Result return path:** wrapper prints `__RESULT_JSON__\n{...}\n__END_RESULT_JSON__` to stdout → captured by `run_subprocess` in `translate_object.py` → parsed by `extract_json_from_output` → surfaced through MCP tool result.

---

## Build Order (Phase Implications for Roadmap)

This is the dependency DAG. Phases must respect it.

```
[1] insert_configs/defaults.yaml + shared/insert_config.py loader
       │ (anything below needs to load YAML)
       ▼
[2] compliant_insert.py wrapper extension
       (already exists as scaffold; add HOVER, meta JSON write, parametric
        force_mode_params consumption, safe-height-then-home exit)
       │ (need data flowing into logs/ before dashboard has anything to show)
       ▼
[3] FMB1 data collection sessions (5 demos × 4 objects = 20 episodes)
       (uses [2] in --collect mode with MANUAL_GUIDED fallback config)
       │ (dashboard needs real CSV/meta to develop against)
       ▼
[4] tools/analyze_inserts.html
       (build against the real 20-episode dataset, not synthetic data)
       │ (configs need data + dashboard to derive parameter values)
       ▼
[5] Per-object YAMLs (u_brown, u_orange, line_green, inverted_u_yellow)
       (operator authors from dashboard signature cards)
       │
       ▼
[6] compliant_insert_episode.py dispatcher
       (resolves YAML, dispatches to wrapper, handles unknown-object fallback)
       │
       ▼
[7] translate_object.py:1085 integration edit (one line + arg passthrough)
       │
       ▼
[8] End-to-end ablation validation
       (recording_dryrun_real_u_brown.yaml completes insert leg autonomously)
       │
       ▼
[9] Generalization validation: one part from a second assembly,
       config-only, no algorithm change
```

**Critical sequencing notes:**

- **[1] before [2]:** the wrapper must consume parametric inputs from day one, even if the parameters are CLI flags (later promoted to YAML). Do **not** hard-code values in the wrapper expecting to "make it parametric later" — that is the path to two parallel implementations.
- **[3] before [4]:** building the dashboard against synthetic data is a trap; it produces a dashboard that visualizes a fiction. Collect demos first, even with placeholder analysis tooling (Excel, jupyter scratchpad), then build the dashboard against the real shapes you saw.
- **[6] before [7]:** the integration edit is the *last* step. Until the dispatcher works standalone (operator can run `compliant_insert_episode.py --object-name u_brown` from a shell and get a successful insert), do not redirect `translate_object.py` to it — otherwise a broken dispatcher silently breaks the whole `--insert` flow including ablations.
- **[2] and [3] can interleave heavily.** Each demo session reveals wrapper bugs (missed event_marker rows, bad CSV flush on SIGTERM, etc.); fixing them is part of the wrapper phase, not a separate "polish" pass.

---

## Anti-Patterns

### Anti-Pattern 1: Single-Script Conflation

**What people do:** Put dispatcher logic, lifecycle logic, config parsing, and dashboard generation all into one ~1500-LOC `compliant_insert.py`.
**Why it's wrong:** The wrapper is the thing operators run hundreds of times during tuning; making it import YAML schemas, asyncio, jinja2, etc. inflates startup time and makes "does the robot work?" debugging harder. Also, the dispatcher logic should be unit-testable without a live UR5e — impossible if it's entangled with rclpy initialization.
**Do this instead:** Two files (`compliant_insert.py` for lifecycle, `compliant_insert_episode.py` for dispatch). Resist the urge to also split into 5 files — two is the right number for this scope.

### Anti-Pattern 2: Eager-Load-All Configs

**What people do:** On dispatcher startup, glob `insert_configs/*.yaml`, parse them all, build an in-memory registry.
**Why it's wrong:** A YAML parse error in `inverted_u_yellow.yaml` will fail an insert of `u_brown`. Surfaces unrelated failures and slows startup linearly with object count.
**Do this instead:** Lazy-load by exact filename: `insert_configs/<object_name>.yaml`. If it doesn't exist, fall back to `defaults.yaml + MANUAL_GUIDED override`. Validate on read, not on startup.

### Anti-Pattern 3: "Config Defaults Are Safe" Silent Fallback

**What people do:** If `<object>.yaml` is missing, use `defaults.yaml` as-is and run autonomously.
**Why it's wrong:** Defaults that are sane for u_brown (a tall thin peg) may slam line_green (a flat oblong) into the table. Silent default-config gambling damages parts. PROJECT.md explicitly calls this out as the wrong move.
**Do this instead:** Missing `<object>.yaml` → switch the wrapper into `MANUAL_GUIDED` mode (full 6-DOF compliance, no autonomous termination, operator-terminated only). Surface a clear message: "No config for X — running guided demo. Run dashboard, write `insert_configs/X.yaml`, retry."

### Anti-Pattern 4: ROS Bag Instead of CSV

**What people do:** Record `/wrench`, `/tcp_pose`, etc. into a ROS bag for "fidelity."
**Why it's wrong:** Bags require ROS to read; the dashboard is a static HTML page. Adds a `bag → CSV` conversion step on every analysis pass. Loses none of the information that matters at 100 Hz.
**Do this instead:** CSV at write-time. Add a `phase` and `event_marker` column the bag wouldn't have anyway.

### Anti-Pattern 5: Mid-Insert Re-Classification

**What people do:** During ACTIVE phase, monitor F/T signature, decide "this looks like a yellow part," switch parameters mid-flight.
**Why it's wrong:** The agent already passed `object_name`. Mid-insert family switches add a control loop that can oscillate, and PROJECT.md scopes this out ("Multi-strategy retry chains" — explicitly out of scope, max 1–2 retries). Runtime classification is also explicitly deferred.
**Do this instead:** One config per object, picked once, used throughout the episode. Retry = retract + re-approach with the *same* config, not a different family.

### Anti-Pattern 6: Adding a "Standardize the Stash" Phase

**What people do:** Try to fix import paths in `_real_mode_stash/` and reuse it.
**Why it's wrong:** Stash is documented (Key Decisions: "reference only, not foundation") as overcomplicated for this scope and using stale module paths. Re-deriving cleanly using stash patterns as guidance is faster than untangling stash code.
**Do this instead:** Treat stash as a textbook. Read it for force-mode RPC patterns, settle-time values, retract behavior. Write fresh in `compliant_insert.py`.

---

## Integration Points

### Internal Boundaries

| Boundary | Communication | Notes |
|----------|---------------|-------|
| `translate_object.py` ↔ dispatcher | `subprocess.Popen` + `__RESULT_JSON__` stdout sentinel | Single line edit at `:1085` (script_path); arg passthrough at `:1093–1107` already covers all needed fields except `--user-notes` (collect-mode only, not exposed via translate_object) |
| dispatcher ↔ wrapper | In-process function call **or** `subprocess.Popen` (TBD; either is acceptable) | If subprocess: same `__RESULT_JSON__` convention. If in-process: dispatcher imports `compliant_insert.run_episode(resolved_config)`. **Recommendation: in-process** — avoids double subprocess overhead and the dispatcher has no ROS state to lose, but acceptable to start with subprocess and inline later if launch latency matters. |
| wrapper ↔ ROS2 graph | rclpy node, standard service/topic clients | Already established in placeholder |
| wrapper ↔ `move_to_safe_height.py` | `subprocess.run` (matches existing pattern in `move_to_clear_area`) | Wrapper invokes safe-height as a subprocess from cleanup() — do not re-implement safe-height inline |
| analyzer ↔ filesystem | Browser File System Access API (Chromium) or drag-drop FileReader (universal) | No server; opens `file://`. Modern browsers gate FSA behind user gesture — fine. |
| dispatcher ↔ `insert_configs/` | `pathlib` + `yaml.safe_load` via `shared/insert_config.py` | New file, ≤80 LOC, no ROS deps, importable from anywhere |

### Concrete Integration Edits

| File | Line(s) | Edit |
|------|---------|------|
| `primitives/translate_object.py` | `:1085` | `script_path = ... 'compliant_insert_episode.py'` (was `'prismatic_peg_insertion.py'`) |
| `primitives/translate_object.py` | `:1093–1107` | Arg passthrough already covers `object_name`, `base_name`, `grasp_id`, `final_base_pos`, `final_base_orientation`, `use_default_base_position`, `current_object_orientation`. **No change needed.** |
| `primitives/translate_object.py` | `:1157` | Optional: drop `--insertion-type {prismatic,legacy}` once dispatcher is the only path; keep for one milestone as escape hatch |
| `.gitignore` | append | `logs/insert_*.csv`<br>`logs/insert_*.meta.json` |
| `server_p3.py` | none | **No change.** The MCP tool `translate_object("insert", ...)` already abstracts the implementation; swapping the script behind it is invisible to the agent. |

### Operator-Facing Failure Surface

When dispatcher hits "no config for object X":

```json
{
  "result": "failure",
  "error": "no insert config for object 'screw_blue'",
  "hint": "run: python3 primitives/compliant_insert_episode.py --object-name screw_blue --base-name <base> --collect  (5 times), then open tools/analyze_inserts.html and author primitives/insert_configs/screw_blue.yaml"
}
```

This bubbles up through `run_perform_insert_real` → `extract_json_from_output` → `node.error_message` → MCP tool result. The agent (LLM) sees the `hint` field and can either prompt the operator or refuse the action. Critical: the hint must mention the **exact command** to run — the LLM uses this verbatim in its operator-facing message.

---

## Scaling Considerations (right-sized for this domain)

| Scale | Architecture Adjustments |
|-------|--------------------------|
| 4 objects, 1 assembly (PoC target) | Current design as specified — no changes |
| 8 objects, 2 assemblies (next milestone) | No structural change; one more YAML per new object. `defaults.yaml` may need a `family:` key if two genuinely-different geometries (e.g., square pegs vs round) emerge — split into `defaults_round.yaml` / `defaults_square.yaml` and add a `defaults_ref:` field in per-object configs. **Defer until needed.** |
| 30+ objects | Consider a config index file (`insert_configs/index.yaml` listing known objects) to enable dispatcher startup validation. Likely also want an automated config writer in the dashboard (operator clicks "save signature → config"). Out of PoC scope. |
| Parallel data collection (multi-robot) | Out of scope per PROJECT.md ("Single-instance, no parallel data collection"). Would require namespacing in `logs/` (e.g., `logs/<robot_id>/insert_*.csv`) and conflict-resolution in shared `insert_configs/`. Don't design for it now. |

---

## Open Architectural Questions (Forwarded to Roadmap)

1. **In-process vs subprocess between dispatcher and wrapper.** Both are viable. Recommendation: in-process (simpler, faster), but the integration phase should empirically pick whichever debugs more cleanly. If wrapper crashes during testing leak rclpy state into the dispatcher, fall back to subprocess.
2. **Termination criterion combine logic** (`any` / `all` / `weighted`). Schema supports all three; PROJECT.md says termination is itself a deliverable. Build the wrapper to handle all three, decide per-object during the analysis phase.
3. **Whether `--insertion-type {prismatic,legacy}` survives.** Suggest deprecating once the new path is validated (one milestone), since the dispatcher subsumes the family/parameter selection that flag was attempting.
4. **Statistical classifier vs hand-rules** for parameter derivation. Explicitly deferred to a post-data-collection phase; affects only the analysis tooling, not the runtime architecture.

---

*Architecture research for: Compliant Insertion Studio (PoC milestone, FMB1 + UR5e + OnRobot RG2)*
*Researched: 2026-05-01*
