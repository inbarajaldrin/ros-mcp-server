# Evidence for fmb1-2026-10-23-run48

run timing: `ablations/logs/replay_20261023_104012.timing.jsonl`

```
run: wall 772 s, arm moving 286 s (37%)
  u_brown              wall    137 s  moving    50 s  still    88 s  (in-process    5 s, 12 blocks)
  u_orange             wall    134 s  moving    44 s  still    90 s  (in-process    5 s, 11 blocks)
  line_green           wall    229 s  moving    96 s  still   133 s  (in-process   55 s, 14 blocks)
  inverted_u_yellow    wall    156 s  moving    48 s  still   108 s  (in-process    5 s, 11 blocks)

blocks by time with the arm still (top 15):
  block                                                        n    wall  moving   still
  seam: drive --mate fmb1/mate:03                              1    93.5    32.9    60.5
  seam: drive --mate fmb1/mate:02                              1    86.0    36.8    49.2
  seam: drive --mate fmb1/mate:01                              1    73.1    26.7    46.4
  seam: drive --mate fmb1/mate:00                              1    70.0    30.7    39.3
  subprocess: rotate_object                                    4    25.3     0.0    25.3
  seam: execute --mate fmb1/mate:02                            2    22.6     0.0    22.6
  subprocess: move_to_grasp                                    4    45.2    26.2    19.0
  subprocess: control_gripper                                  8    16.1     0.0    16.1
  subprocess: move_to_safe_height                              8    33.9    20.4    13.5
  seam: execute --mate fmb1/mate:00                            1    11.8     0.0    11.8
  seam: execute --mate fmb1/mate:03                            1    11.1     0.0    11.1
  seam: execute --mate fmb1/mate:01                            1    10.8     0.0    10.8
  depth_pick: inverted_u_yellow                                1    10.5     0.0    10.5
  depth_pick: u_brown                                          1    13.6     3.4    10.2
  seat_check: line_green                                       1    13.9     4.1     9.7
```

## Certification
```
  [execute_mate] fmb1/mate:00: placed=True refusal=''  depth=0.0001
  [execute_mate] fmb1/mate:01: placed=True refusal=''  depth=-0.00038
  [execute_mate] fmb1/mate:02: placed=False refusal='not_at_seat' outside (xy implied by seat depth): an end at +2.39 mm outside +-2.0 mm (the part can wedge here) depth=0.00126
  [execute_mate] fmb1/mate:02: placed=True refusal=''  depth=0.00125
  [execute_mate] fmb1/mate:03: placed=True refusal=''  depth=4e-05
  Completed: ['u_brown', 'u_orange', 'line_green', 'inverted_u_yellow']
RUN EXIT 0 10:53:05
```
