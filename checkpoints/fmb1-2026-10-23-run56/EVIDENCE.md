# Evidence for fmb1-2026-10-23-run56

run timing: `ablations/logs/replay_20261023_141513.timing.jsonl`

```
run: wall 638 s, arm moving 224 s (35%)
  u_brown              wall    136 s  moving    52 s  still    84 s  (in-process    5 s, 12 blocks)
  u_orange             wall    117 s  moving    44 s  still    73 s  (in-process    5 s, 11 blocks)
  line_green           wall    121 s  moving    37 s  still    83 s  (in-process    5 s, 12 blocks)
  inverted_u_yellow    wall    149 s  moving    43 s  still   106 s  (in-process    5 s, 11 blocks)

blocks by time with the arm still (top 15):
  block                                                        n    wall  moving   still
  seam: drive --mate fmb1/mate:03                              1    81.0    25.9    55.1
  seam: drive --mate fmb1/mate:00                              1    71.8    32.9    38.9
  seam: drive --mate fmb1/mate:01                              1    58.3    25.8    32.5
  seam: drive --mate fmb1/mate:02                              1    47.6    18.9    28.7
  subprocess: rotate_object                                    4    24.2     0.0    24.2
  subprocess: move_to_grasp                                    4    43.3    26.2    17.1
  subprocess: control_gripper                                  8    15.9     0.0    15.9
  seam: execute --mate fmb1/mate:03                            1    13.6     0.0    13.6
  subprocess: move_to_safe_height                              8    32.6    20.3    12.3
  depth_pick: inverted_u_yellow                                1    14.2     2.2    12.0
  seam: execute --mate fmb1/mate:00                            1    10.6     0.0    10.6
  seam: execute --mate fmb1/mate:02                            1     9.9     0.0     9.9
  depth_pick: u_brown                                          1    13.2     3.3     9.9
  seam: execute --mate fmb1/mate:01                            1     9.8     0.0     9.8
  seat_check: line_green                                       1    13.5     4.1     9.4
```

## Certification
```
  [execute_mate] fmb1/mate:00: placed=True refusal=''  depth=0.0001
  [execute_mate] fmb1/mate:01: placed=True refusal=''  depth=-0.00041
  [execute_mate] fmb1/mate:02: placed=True refusal=''  depth=0.00128
  [execute_mate] fmb1/mate:03: placed=True refusal=''  depth=-0.00067
  Completed: ['u_brown', 'u_orange', 'line_green', 'inverted_u_yellow']
RUN EXIT 0 14:25:51
```
