# Phase 4 — Analyzer Dashboard ✅ SHIPPED 2026-05-04

## Goal

Build a static HTML dashboard that auto-loads the 60 CSV+meta+signals episodes from Phase 3, surfaces per-object insertion signatures, and lets the operator visually identify the cleanest segmentation method before formalizing the insertion algorithm in Phase 5.

## Outcome

**Two-stage analyzer system delivered**, exceeding the original 9-requirement scope (DASH-01..09):

- **Stage A (preprocessor)** `compliant_insertion_studio/analyzer/preprocess.py` — Python, ~700 LOC. Computes per-shape clean baselines from quiet windows mined across all 60 episodes (iterative refinement, 3-shape pooling: u_brown+u_orange→u_shape), runs **5 segmentation methods** on every episode, computes drift signatures and per-episode feature summaries, writes `<basename>.signals.json` sidecars.
- **Stage B (viewer)** `compliant_insertion_studio/analyzer/analyze_inserts.html` — single-page HTML, Plotly + PapaParse from CDN, ~1100 LOC. Two tabs: **Single Episode** (6 time-series with method overlays + Fz-vs-Z phase plot + TCP top-down + raw column explorer) and **Cross-Episode** (per-episode feature scatter across 18 dimensions with shape/assist filters + two-episode contact-aligned compare across 6 channels).
- **Launcher** `analyzer/serve.sh` + `_build_manifest.py` — one-command "open dashboard with all 60 episodes pre-loaded" via `python3 -m http.server`.

## Five Segmentation Methods (vs the originally-planned 1)

The operator wanted **3+ different ways to interpret the data** so the cleanest split could be picked visually before formalization. Five orthogonal lenses delivered:

| Method | Signal source | Threshold | What it catches |
|---|---|---|---|
| **M1 — Force-domain** | F_lat / T_lat / Tz / v_lat per-channel | baseline p99 | sample-level statistical anomaly |
| **M2 — Kinematic** | TCP velocity derivatives only (force-blind) | 1.5× v_lat p99, floor 4 mm/s | strong lateral motion regardless of force |
| **M3 — Energetic** | Signed dot product `F_base · v` (lateral mechanical power) | 3× power_lat p99, floor 5 mW | external work injection into lateral DOFs |
| **M5 — Torque-motion** | Torque × angular velocity coupling | median + 5×MAD (robust) | rotation-driven nudges with weak lateral force |
| **M7 — Object slip** | `|d(obj_xy)/dt − d(tcp_xy)/dt|` | median + 5×MAD | above-wrist forces invisible to F/T sensor |

## Critical fixes applied during build (from GPT adversarial review)

| Severity | Issue | Fix |
|---|---|---|
| CRITICAL | M3 was multiplying tool-frame `|F_lat|` by base-frame `|v_lat|` and calling it physical work | Transform wrench to base_link via TCP quat per row; use signed dot product |
| HIGH | Baseline pool was 6 traces × ~3.5k samples; some "autonomous" traces self-trigger M1 | Iterative quiet-window refinement; pool grew to 60 traces × 60–80k samples (10–25× larger); 95% of samples are quiet on convergence |
| MEDIUM | Viewer showed baseline p99 reference lines, but methods use 1.5×p99 / 3×p99 | Pull actual threshold from each method object in `signals.methods` |
| MEDIUM | Raw column picker `Number()`-coerced string columns (`phase`, `wrench_frame_id`) | Filter to numeric columns only; expose derived JS columns |
| LOW | First-contact scan `range(start, len(df) - sustain)` excluded last valid window | Off-by-one fix |
| LOW | Drift display truthiness rejected true 0.0 mm | `!= null` check |
| docs | SCHEMA.md said wrench was transformed to base_link by wrapper | Corrected: wrench is logged in `tool0_controller`; `wrench_frame_id` column documents this; transform is now done in analyzer per row |

## Key empirical findings

- **`assist_level` is a noisy label.** Multiple "assisted" episodes are indistinguishable from autonomous in F/T/motion (e.g., `u_brown_081329`). Telemetry-derived "effective autonomy" via M1 quiet-window mask is more useful than the binary tag.
- **Most operator nudges are sub-threshold in lateral force.** Max |F_lat| during autonomous traces hits ~14 N (friction transients at contact); a sustained-force nudge detector needs >14 N to be specific. Most assists show up in `|T_lat|` or `power_lat` instead — which is exactly what M3 + M5 catch.
- **Per-shape baselines were heavily inflated** before the iterative refinement. `inverted_u_yellow` p99 of `power_lat` dropped from 0.0145 W → 0.008 W after quiet-window pooling.
- **u_orange has 0 native autonomous traces.** Resolved by physical-shape pooling: u_brown + u_orange → u_shape, contributing 2 autonomous traces between them.

## Requirements traceability (DASH-01..09)

| Req | Description | Status |
|---|---|---|
| DASH-01 | Single static HTML, Plotly 3.5.1 + PapaParse 5.5.3 from CDN, opens directly in browser | ✅ |
| DASH-02 | File picker for CSV + meta JSON triples; auto-pair by basename | ✅ + manifest auto-load mode (`http.server` flow) |
| DASH-03 | Single-episode view: F-vs-t (3 traces) + T-vs-t (3 traces) + Z-vs-t with synced cursors, sidecar metadata panel, event-marker vertical lines, phase-band background coloring | ✅ |
| DASH-04 | F-vs-Z phase plot (peg-in-hole signature) | ✅ — time-colored Viridis |
| DASH-05 | 3D trajectory view with target marker | partial — TCP top-down xy plane delivered; full 3D deferred (operator reviewed; xy plane is the diagnostic axis) |
| DASH-06 | Cross-episode overlay view: filter by object + outcome, time-aligned on first-contact | ✅ — Two-episode compare with `t since first_contact` alignment in Cross-Episode tab |
| DASH-07 | Per-object signature card: median Fz, |Tx|/|Ty| peaks, lateral travel, descent duration | ✅ — encoded as 12-feature scatter axes + per-episode summary in `signals.json` |
| DASH-08 | Decimation for large datasets | ✅ — `scattergl` everywhere |
| DASH-09 | Functional UI only, no styling polish | ✅ |

## Deliverables shipped

- `compliant_insertion_studio/analyzer/preprocess.py` — Stage A preprocessor (60 sidecar files generated)
- `compliant_insertion_studio/analyzer/analyze_inserts.html` — Stage B viewer
- `compliant_insertion_studio/analyzer/manifest.json` — 60-episode index with feature summary
- `compliant_insertion_studio/analyzer/_build_manifest.py` — manifest regenerator
- `compliant_insertion_studio/analyzer/serve.sh` — launcher (`bash …/serve.sh 8766`)
- `compliant_insertion_studio/logs/insert_*.signals.json` × 60 — per-episode segmentation outputs
- `compliant_insertion_studio/docs/SCHEMA.md` — corrected wrench-frame description + tcp_to_object_transform convention block

## What Phase 5 inherits

- 5 candidate segmentation methods with empirical baselines locked
- Per-shape `power_lat` p99 thresholds (u_shape=0.008 W, line_green=0.010 W, inverted_u_yellow=0.008 W) — natural floor for "force-absorbed" termination predicate
- Per-episode feature summary in `signals.json.features` — feeds Phase 5 termination derivation
- Cross-episode scatter UI to visually pick which feature axes separate clean from assisted

## Deferred to Phase 5+ or future iteration

- M4 (multivariate change-point CUSUM/PELT on residual stack)
- M6 (low-frequency human-energy ratio FFT)
- Live JS threshold sliders that re-run M1/M2/M3 client-side (currently thresholds are baked into sidecar)
- Small-multiples grid (60 mini-plots contact-aligned)
- Keyboard ←/→ stepping between episodes
- Synced cursors across the 6 time-series plots in Single Episode tab
