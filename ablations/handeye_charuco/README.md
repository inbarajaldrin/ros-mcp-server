# ChArUco hand-eye calibration (recovered 2026-11-18)

The wrist camera's mount (tool -> colour optical frame) solved from views of a ChArUco board on the table, plus base1's top-face
markers at a second, known height. Two planes 35 mm apart separate "focal length off" from "camera sits lower". On 2026-10-13
it reached ~0.9 mm held-out (board on spot A predicting spots B and C) for the old mount. The scripts lived only in a session
scratchpad and were recovered from that session's transcript (ea57148e, lines 7215 and 7382).

- Board: `cv2.aruco.CharucoBoard((8, 6), 0.030, 0.022, DICT_7X7_50)`: 8x6 squares, 30 mm squares, 22 mm markers.
- Views: `python3 -m primitives.scan_workspace --survey --wide --target-xyz X Y Z --image-dir DIR --out survey.json --mode real`
  (`--plan-only` prints the viewpoints without moving). It aims the TOOL z axis at the target, which suited the old straight-down
  camera; with the v3.1 mount (30 deg tilt, 89 mm off-axis) check that the board stays in frame before moving.
- `detect_cache.py out.npz LOC:survey.json[,...]` detects the board and base1 corners once (CPU, multiprocess).
- `final_solve.py LOC:survey.json[,...]` solves X (tool -> camera), an optional focal scale, each board's pose with its plane height
  pinned, base1's pose with its top face pinned, and the base1 marker edge.
- 10-13 raw views: `~/Documents/calib_bags/charuco_C_20261013_225529` (4.9 GB mcap) on the cell PC.
