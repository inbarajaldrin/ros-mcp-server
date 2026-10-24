# Evidence for fmb1-2026-10-23-run74c

run timing: `ablations/logs/replay_20261023_210100.timing.jsonl`

```
run: wall 534 s, arm moving 197 s (37%)
  u_brown              wall    101 s  moving    42 s  still    60 s  (in-process    4 s, 12 blocks)
  u_orange             wall    113 s  moving    45 s  still    68 s  (in-process    4 s, 12 blocks)
  line_green           wall    101 s  moving    37 s  still    63 s  (in-process    5 s, 12 blocks)
  inverted_u_yellow    wall    110 s  moving    36 s  still    74 s  (in-process    5 s, 12 blocks)

blocks by time with the arm still (top 15):
  block                                                        n    wall  moving   still
  seam: drive --mate fmb1/mate:03                              1    47.7    19.2    28.6
  seam: drive --mate fmb1/mate:01                              1    51.2    24.4    26.8
  seam: drive --mate fmb1/mate:00                              1    47.7    24.9    22.8
  seam: drive --mate fmb1/mate:02                              1    39.7    18.5    21.2
  subprocess: control_gripper                                  8    16.7     0.0    16.7
  subprocess: rotate_object                                    4    18.9     3.3    15.6
  depth_pick: inverted_u_yellow                                1    14.6     0.0    14.6
  subprocess: move_to_safe_height                              8    32.3    20.3    12.0
  subprocess: move_to_grasp                                    4    37.7    26.2    11.6
  depth_pick: u_orange                                         1    15.0     4.0    11.0
  seam: execute --mate fmb1/mate:02                            1    10.3     0.0    10.3
  depth_pick: line_green                                       1    12.0     1.8    10.2
  seam: execute --mate fmb1/mate:03                            1    10.0     0.0    10.0
  seam: execute --mate fmb1/mate:01                            1     9.3     0.0     9.3
  seam: execute --mate fmb1/mate:00                            1     9.0     0.0     9.0
```

## Certification
```
  [execute_mate] fmb1/mate:00: placed=True refusal=''  depth=0.00011
  [execute_mate] fmb1/mate:01: placed=True refusal=''  depth=1e-05
  [execute_mate] fmb1/mate:02: placed=True refusal=''  depth=0.00085
  [execute_mate] fmb1/mate:03: placed=True refusal=''  depth=-0.0004
  Completed: ['u_brown', 'u_orange', 'line_green', 'inverted_u_yellow']
RUN EXIT 0 21:09:55
```
