# M2: aim error over base position (FMB1, 10-23)

    python3 ablations/m2_report.py /tmp/fmb1_55.log /tmp/fmb1_56.log ... [--md compliant_insertion_studio/docs/M2_ERROR_MAP.md]

Per cycle and part, from the replay's console log and the drive-align logs it names:
  base     drive-mate's measured base for the mate (x, y mm; yaw from the align record)
  r0       the first servo reading (U parts: marker servo; line_green / yellow: depth servo) - what the aim missed by
           before any correction, i.e. the vision-chain error at that base pose
  moved    the servo's total aim correction = the aim error at that base pose (r0 is after the servo prior)
  slide    the insert's measured seat minus its prediction (x, y, z mm) - what the probes / channel still had to find
  depth    ExecuteMate's certified seat depth (mm)
Sigmas: per part, the spread of r0, slide and depth over the cycles, and of the base read (drive-mate's per-mate reads
of the same cycle's base: it does not move between mates unless an insert pushed it).

| cycle | base set (x, y mm, yaw) | part | measured base | r0 (mm) | servo moved | slide (x, y, z) | depth |
|---|---|---|---|---|---|---|---|
| fmb1_56 | [112.4, -483.4, 23.4] | u_brown | [113.2, -487.8] | [-0.5, 2.5] | [-3.5, -9.4] | [0.45, 0.48, 0.15] | 0.1 |
| fmb1_56 | [112.4, -483.4, 23.4] | u_orange | [113.2, -487.8] | [3.9, -2.0] | None | [-0.44, -1.1, 0.06] | -0.41 |
| fmb1_56 | [112.4, -483.4, 23.4] | line_green | [113.2, -487.8] | None | None | [-2.34, -0.96, -1.99] | 1.28 |
| fmb1_56 | [112.4, -483.4, 23.4] | inverted_u_yellow | [113.2, -487.8] | [3.4, 1.1] | None | [-1.48, -1.31, 1.07] | -0.67 |
| fmb1_57 | [137.6, -448.3, 14.65] | u_brown | [138.7, -451.8] | [-0.9, 3.5] | [-4.5, -10.0] | [0.74, -0.03, 0.5] | -0.25 |
| fmb1_57 | [137.6, -448.3, 14.65] | u_orange | [138.7, -451.8] | [-3.1, -0.6] | [-1.1, -2.6] | [-0.13, -0.58, 1.21] | -0.31 |
| fmb1_57 | [137.6, -448.3, 14.65] | line_green | [138.7, -451.8] | [3.5, -2.5] | None | [-2.9, -1.06, 0.89] | 1.09 |
| fmb1_57 | [137.6, -448.3, 14.65] | inverted_u_yellow | [138.7, -451.8] | [2.9, 1.2] | None | [-1.51, -0.74, 1.65] | -0.55 |
| fmb1_58 | [131.6, -500.6, 21.76] | u_brown | [132.2, -502.5] | [0.0, -0.2] | [-4.5, -7.5] | [0.31, 0.48, -0.04] | -0.03 |
| fmb1_58 | [131.6, -500.6, 21.76] | u_orange | [132.2, -502.5] | [7.7, -1.9] | None | [-0.46, -1.11, 0.06] | -0.36 |
| fmb1_58 | [131.6, -500.6, 21.76] | line_green | [132.2, -502.5] | [4.4, -2.6] | None | [-2.43, -1.02, -1.99] | 1.39 | servo misread -> held-part aim
| fmb1_58 | [131.6, -500.6, 21.76] | inverted_u_yellow | [132.2, -502.5] | [3.9, -0.1] | None | [-1.54, -0.75, 1.59] | -0.75 |

| part | sigma aim error = servo moved (mm) | sigma slide (mm) | sigma depth (mm) | n |
|---|---|---|---|---|
| u_brown | [0.58, 1.31] | [0.22, 0.29, 0.27] | [0.18] | 3 |
| u_orange | None | [0.19, 0.3, 0.66] | [0.05] | 3 |
| line_green | None | [0.3, 0.05, 1.66] | [0.15] | 3 |
| inverted_u_yellow | None | [0.03, 0.33, 0.32] | [0.1] | 3 |

## Reading it (10-23, runs 56-58 = the 3 qualifying cycles)

- **Cleared:** 3 consecutive unattended FMB1 cycles (reset + base pushed by the arm to a new pose + assembly), rc 0,
  every mate certified: run 56 (112, -483) 23.4 deg 638 s; run 57 (138, -448) 14.7 deg 638 s; run 58 (132, -501)
  21.8 deg 734 s (line_green wedged an end +8.1 mm with the held-part aim (-4.2, +2.7) mm; the automatic nudge freed it,
  taps read -1.32..-1.42 mm, certified +1.39 mm).
- **Aim error over base position:** u_brown's total servo correction (the vision chain's error at that pose) was
  (-3.5, -9.4), (-4.5, -10.0), (-4.5, -7.5) mm at the three poses, against (-3.6..-4.2, -7.3..-8.3) at the layout
  pose: it moves ~1.3 mm (sigma) over ~50 mm of base travel - the >2.5 mm / 20 mm of 10-22 did not reappear with the
  same-frame servo, the in-hand CAD base fit and the servo prior. The servo closes it every cycle (converged in 1-2 rounds).
- **After the servo:** the inserts still slid 0.2-0.3 mm (sigma) laterally; certified depths repeat to 0.05-0.18 mm.
- **Open:** line_green's wedge in run 58 came with a good aim (the slide 2.4 mm is its usual); the insert trace is the
  next thing to read. u_orange's servo measures from the clear height, whose total correction is not parsed here yet.
