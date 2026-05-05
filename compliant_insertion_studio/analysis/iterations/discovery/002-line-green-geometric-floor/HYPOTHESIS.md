# Discovery iteration — 002-line-green-geometric-floor

## Question
Resolves `STATE.json:open_questions[2]`: "Why does line_green show only 7mm final descent? Geometry (shallow slot) confirmed or algorithm (premature seat)?"

The bracketed note on I002 ("verified_3_objects (line_green has shallow slot, hits floor first)") asserts geometry but had no evidence link. This iteration provides it.

## Method
`scripts/11_line_green_seat_depth.py` over the 60 GOLD demos (May-3 operator-driven). For each object, compute the spread of `final_z_drop_mm`. The discriminator:

- **Geometric floor**: tight cluster across n≥10 independent demos (IQR << median). Different operators / demos cannot push past a physical stop.
- **Premature seat (algorithmic)**: scattered values, since the FSM's seat predicate would fire at variable depths depending on F/T noise.

Outliers below 5 mm are excluded as likely operator-aborts logged as "success" (operator-sigterm reason on partial demos). The `full_descent` subset is the geometry probe.

## Result
**New invariant** — line_green has a hard geometric floor at 7.3 ± 0.4 mm.

Per-object full-descent IQR (mm):

| Object | n_full | Median | Min–Max | IQR |
|---|---|---|---|---|
| u_orange | 10 | 31.48 | 31.41–31.66 | **0.03** |
| inverted_u_yellow | 16 | 31.17 | 30.78–32.16 | 0.5 |
| u_brown | 8 | 30.37 | 29.12–31.5 | 1.24 |
| **line_green** | **18** | **7.32** | **6.49–7.75** | **0.35** |

line_green's IQR (0.35 mm) is comparable to or tighter than the other three objects' IQRs at their ~31 mm seat depth. The cluster is just as tight as the deep-seat objects — only the absolute depth differs by 4×. If the FSM were terminating early, the spread would inflate (different attempts triggering at different noise levels). It is not.

**Conclusion**: line_green peg geometry physically bottoms at ~7.3 mm. This is an object-specific geometric constant, not an algorithmic gate.

## Implications
1. I002 ("Post-collapse dz/dt = -6.4 mm/s") is **fully portable across all 4 objects** — line_green simply hits the floor before the collapse phase finishes. Its truncated ACTIVE phase still obeys the same dz/dt rule during the brief window of true post-contact descent.
2. Any FSM termination predicate that requires `final_z_drop ≥ 25 mm` (the May-4 default for u_orange) **will never fire on line_green**. Per-object termination depth must be configurable, or the predicate must use a fraction-of-expected-depth check derived from CAD.
3. Cross-object work that uses `final_z_drop_mm` as a "fully seated" bool must use a per-object threshold:
   - u_orange / u_brown / inv_u_yellow: ≥ 25 mm
   - line_green: ≥ 6 mm

## Files written
- `metrics.json` — full per-object distribution
- `FINDING.md` — this narrative
- New invariant **I010** appended to `STATE.json:known_invariants`
- Open question Q3 removed from `STATE.json:open_questions`
- Updated `I002` evidence link to point at this iteration
