# Project Conventions — Compliant Insertion Studio

**Single source of truth for project working patterns.** Auto-summarized into `CLAUDE.md` by `gsd-tools generate-claude-md` (which keeps headings, bullets, and tables; drops prose). The "Key Rules" section is structured to survive that summarizer intact.

## Key Rules — every agent must follow

- **Research before code**: spend 5–15 min on `WebSearch` + `WebFetch` before any non-trivial deliverable. Find existing tools, papers, vendor docs, and reference implementations. Don't reinvent.
- **Clone references locally**: useful repos go to `_references/repos/` via `git clone --depth 1`; useful articles get saved as markdown to `_references/articles/`. Both directories are gitignored.
- **Honesty over confidence**: if you don't know something, say so. Do not write SOPs or specs based on shallow grounding. Pause and research.
- **Per-piece copy/modify/write-fresh decision**: when borrowing from a reference, decide explicitly per piece — not whole-repo. Credit sources in code comments.
- **Inline-default working pattern**: do work in the main conversation so the operator can see and intervene. Subagents only when work is genuinely independent + parallel + would pollute main context (the 4-researcher init was the right use case; routine code/planning is not).
- **Phase boundaries are guidance, not gates**: if requirements from later phases are coupled to current work, finish them together and mark them complete now. Update REQUIREMENTS.md traceability to record where each requirement actually completed.
- **All project deliverables under `compliant_insertion_studio/`**: the entire subsystem is a self-contained folder droppable into other robotics projects. New code defaults to that folder unless modifying an existing host-repo primitive.
- **F/T calibration is three layers**: foundational payload calibration (per-mount, one-time, sets `set_target_payload`), session-level smoke test (per-session, confirms sensor health), per-pose `zero_ftsensor` (immediately before force mode). Per-pose alone does not substitute for foundational.
- **Pendant in Local mode**: do not write code requiring Remote mode or `dashboard_client/recover`. Recovery from protective stops is manual.
- **Force-mode wrench ≤ 5 N default**: higher only with explicit per-task override and operator awareness.
- **SIGTERM cleanup must be idempotent and reliable**: wrapper must reach safe-state DONE exit even if force mode is already stopped or controller switch fails partway.
- **Hands-off window during F/T zero**: operator confirmation gate before zero, +1 s post-zero drift check, no operator load during baseline windows.
- **Safe height before move_home when holding a part**: direct `move_home` plans straight-line trajectories that ignore inserted bases.
- **Don't commit unless explicitly approved**, no `Co-Authored-By` lines (operator global rule).
- **`_references/` and `compliant_insertion_studio/logs/` are gitignored**: never commit reference repos or telemetry.
- **Ask the operator before**: adding a new top-level dependency, writing > 200 LOC without checkpoint, performing any robot motion, modifying primitives outside `compliant_insertion_studio/`, departing from a documented decision in PROJECT/REQUIREMENTS/ROADMAP.
- **Two execution tracks — away-from-robot and at-robot**: every requirement is tagged either `[N]` (no-robot — can be done from anywhere with the codebase) or `[R]` (robot-required — needs the physical UR5e + bringup). When the operator is away from the robot, work the `[N]` track. When the operator is at the robot, work the `[R]` track and any `[N]` items needed to unblock it. **`.planning/TRACKS.md` is the live list of what's ready in each track right now.** Update it whenever a task transitions states (ready → in-progress → done, or new requirements get tagged).

## Anti-patterns — explicit "don't"

- Writing 100-line SOPs based on extrapolation rather than documented procedures
- Confusing similar-sounding concepts (e.g., "calibration" vs "bias offset" — they are different things on a UR5e)
- Citing thresholds that you guessed at as if they came from research
- Shipping a deliverable without distinguishing "verified empirically" from "research-backed" from "extrapolated"
- Spawning a subagent for routine work
- Splitting coupled work across phase boundaries to honor the convention rather than because the work is actually separable
- Putting project deliverables outside `compliant_insertion_studio/`
- Treating `zero_ftsensor` as a substitute for correct payload identification
- Code that assumes Remote pendant mode or dashboard recovery automation

## Decision matrix — copy / modify / write-fresh

| Decision | When to use |
|---|---|
| **Copy (lift file, attribute source)** | Code is in our language + framework + license is compatible + fits our architecture as-is |
| **Modify after copying** | Mostly fits, needs only surface tweaks (paths, message types, function names) |
| **Write fresh from algorithm/pattern** | Reference is in different language/framework/era, but the algorithm or pattern is sound — translate the *idea*, not the lines |
| **Skip** | Reference is well-known but doesn't fit our stack/scope (e.g., a node requiring an accelerometer we don't have) |

## When to use which calibration layer

| Layer | Frequency | What it does | Trigger |
|---|---|---|---|
| **Foundational** payload calibration | Per gripper mount (one-time) | Recovers mass + CoG + bias via Kubus 2007 LSQ; outputs `set_target_payload(mass, cog)` for bringup launch | New gripper / new jig / sensor remount / orientation-dependent bias observed |
| **Session** F/T smoke test | Per session | Zero + 5 s hold + bias verification in known neutral pose; pass/fail per axis | Start of session / after protective stop / after physical bump / when force-mode misbehaves |
| **Per-pose** `zero_ftsensor` | Immediately before each force-mode entry | Single-pose bias subtraction | Inside the wrapper's PRE phase, after smoke passes |

---

## Detailed rationale (humans only — auto-summarizer drops most of what follows)

### Research before code — procedure

- Search 2–4 query variations (vendor docs, GitHub, papers, community forums)
- Identify candidate repos / articles
- Clone repos into `_references/repos/`, save articles to `_references/articles/`
- Read enough to make an honest copy/modify/write-fresh decision per piece
- Only then write code

This applies to every substantive deliverable, not just the "interesting" ones. F/T calibration was the trigger that made the operator surface this principle explicitly; the rule is general.

### Phase boundaries — finish coupled work together (rationale)

- Project-defined deviation from conventional GSD
- If requirements from later phases are tightly coupled to current work, finish them together and mark them complete now, regardless of which phase they nominally belong to
- Update REQUIREMENTS.md traceability when this happens to record where each requirement actually completed
- ROADMAP.md phase-completion still requires *all* of that phase's owned requirements to be done — but those done early count
- Do **not** intentionally pull future work forward to "save time" — only collapse when the coupling is real and finishing-now is materially cheaper than finishing-later
- This avoids the conventional-GSD waste of revisiting touched files in a later phase to make small additions that could have been done in one pass

### Folder structure rationale

- All Compliant Insertion Studio code lives under `compliant_insertion_studio/` (see REQUIREMENTS.md → "Project Layout" for full tree)
- The entire subsystem is droppable into other robotics projects with a one-line edit at the host's `translate_object` equivalent
- Implication: when adding new code for this project, default to placing it under `compliant_insertion_studio/`. Only put code in `primitives/` (or elsewhere) if you're modifying *existing* primitives or fixing a bug in the host repo unrelated to this project's deliverables

### Calibration hierarchy detail

- **Foundational** (per-mount): payload mass + CoG via `set_target_payload()` after running calibration script (or URCap Measure wizard). If this is wrong, nothing downstream is right — orientation-dependent bias persists no matter how often you `zero_ftsensor`
- **Session-level**: F/T smoke test confirms the sensor's bias is close to zero in a known pose. If this fails, foundational is suspect
- **Per-pose**: `zero_ftsensor` before force mode subtracts residual bias. Fast, cheap, but does NOT substitute for foundational

### Inline default rationale

- The GSD `workflow.research / plan_check / verifier / code_review / pattern_mapper` toggles are all `false` in `.planning/config.json` to enforce inline-default
- If you find yourself wanting to spawn an agent for routine work, ask first
- The 4-researcher init at project start was right because the 4 surveys were genuinely independent (each researcher knew nothing about the others), parallel (5 min instead of 20), and produced artifacts (the .md files) rather than intermediate chatter that would have polluted the main context

### Safety conventions detail

- Operator's pendant preference is Local mode (manual control retained)
- Don't call `dashboard_client/recover`, `dashboard_client/play`, or other Remote-mode services
- Recovery from protective stops is manual on the pendant
- After protective stop, operator clears it; only then can your code resume
- Force-mode wrench limits exist because gear/part/fixture damage is the binding constraint, not benchmark performance
- SIGTERM cleanup is operator trust. If the wrapper's cleanup is unreliable, operators will hesitate to use SIGTERM and instead try to interrupt other ways (which may leave the robot in force mode)

### Commit discipline detail

- Pre-existing-codebase fixes (bug fixes outside the project's scope) get their own commit, separate from project deliverables
- Project commits should reference requirement IDs they complete (e.g., `feat(WRAP-01..05): episode wrapper PRE/HOVER/ZERO phases`)
- The `_references/` folder is gitignored — never committed
- Telemetry logs are gitignored — never committed

---

*Defined: 2026-05-01 during project initialization, after multiple operator clarifications.*
*Update mechanism: edit this file → run `gsd-tools generate-claude-md` → commit both this file and CLAUDE.md.*
