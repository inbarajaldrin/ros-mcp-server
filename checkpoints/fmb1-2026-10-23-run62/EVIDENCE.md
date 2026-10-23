# Evidence for fmb1-2026-10-23-run62

run timing: `ablations/logs/replay_20261023_163849.timing.jsonl`

```
run: wall 588 s, arm moving 226 s (38%)
  u_brown              wall    134 s  moving    55 s  still    78 s  (in-process    5 s, 12 blocks)
  u_orange             wall    111 s  moving    44 s  still    66 s  (in-process    4 s, 11 blocks)
  line_green           wall    114 s  moving    39 s  still    74 s  (in-process    5 s, 12 blocks)
  inverted_u_yellow    wall    116 s  moving    42 s  still    75 s  (in-process    5 s, 11 blocks)

blocks by time with the arm still (top 15):
  block                                                        n    wall  moving   still
  seam: drive --mate fmb1/mate:00                              1    71.3    35.5    35.9
  seam: drive --mate fmb1/mate:03                              1    54.8    22.2    32.6
  seam: drive --mate fmb1/mate:01                              1    51.7    25.4    26.3
  seam: drive --mate fmb1/mate:02                              1    39.8    18.5    21.3
  subprocess: rotate_object                                    4    24.6     6.0    18.5
  subprocess: move_to_grasp                                    4    43.3    26.2    17.1
  subprocess: control_gripper                                  8    15.4     0.0    15.4
  subprocess: move_to_safe_height                              8    32.5    20.3    12.2
  seam: execute --mate fmb1/mate:03                            1    11.2     0.0    11.2
  seam: execute --mate fmb1/mate:01                            1     9.9     0.0     9.9
  seat_check: line_green                                       1    14.0     4.1     9.9
  seam: execute --mate fmb1/mate:02                            1     9.8     0.0     9.8
  depth_pick: line_green                                       1    11.8     2.2     9.6
  seam: execute --mate fmb1/mate:00                            1     9.2     0.0     9.2
  depth_pick: u_orange                                         1    11.1     1.9     9.2
```

## Certification
```
  [execute_mate] fmb1/mate:00: placed=True refusal=''  depth=-0.00015
  [execute_mate] fmb1/mate:01: placed=True refusal=''  depth=-0.00037
  [execute_mate] fmb1/mate:02: placed=True refusal=''  depth=0.00146
  [execute_mate] fmb1/mate:03: placed=True refusal=''  depth=-0.00063
  Completed: ['u_brown', 'u_orange', 'line_green', 'inverted_u_yellow']
RUN EXIT 0 16:48:38
```
