#!/usr/bin/env python3
"""Checkpoint a known-good version of the cell (operator 10-23: "checkpoint a successful iteration ... if I have a demo to
show we have a successful version, including dependencies").

    python3 ablations/checkpoint.py NAME --run ablations/logs/replay_<ts>.timing.jsonl [--log /tmp/fmb1_NN.log]

Records, under checkpoints/NAME/ (committed) and git tag ckpt/NAME (on HEAD):
  MANIFEST.json   repo commit; every ~/Desktop/ros2_ws/src package (git HEAD + its uncommitted diff, or a tarball when it is
                  not under git); pip freeze; the installed ros-humble ur / realsense / rclpy package versions
  configs/        the localizer calibration (robot_config.yaml, filter_config.yaml, sticker_calibration.json), the F/T
                  calibration and defaults, the cell layout
  ws/             <pkg>.patch for uncommitted changes, <pkg>.tar.gz for packages not under git (sources only)
  EVIDENCE.md     the run's cycle_time summary and its certify lines
  RUN.md          how to bring the stack up and run one cycle from this checkpoint, and how to restore it
A checkpoint is promoted only from a full uninterrupted cycle (reset + assembly, every mate certified) on exactly HEAD.
"""
import argparse
import datetime
import json
import os
import shutil
import subprocess
import sys
import tarfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
WS = Path.home() / 'Desktop' / 'ros2_ws'
LOCALIZER_CFG = WS / 'src' / 'aruco_camera_localizer' / 'config'


def sh(cmd, cwd=None):
    p = subprocess.run(cmd, cwd=cwd, shell=isinstance(cmd, str), capture_output=True, text=True)
    return p.stdout.strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('name')
    ap.add_argument('--run', required=True, help="the qualifying run's replay_<ts>.timing.jsonl")
    ap.add_argument('--log', help="the cycle's console log (for the certify lines)")
    a = ap.parse_args()
    if sh(['git', 'status', '--porcelain', '--untracked-files=no'], REPO):
        print('refused: the repo has uncommitted tracked changes - commit them so the checkpoint is exactly HEAD'); return 1
    out = REPO / 'checkpoints' / a.name
    if out.exists():
        print(f'refused: {out} exists'); return 1
    (out / 'configs').mkdir(parents=True); (out / 'ws').mkdir()
    head = sh(['git', 'rev-parse', 'HEAD'], REPO)
    man = {'name': a.name, 'created': datetime.datetime.now().isoformat(timespec='seconds'),
           'repo': {'path': str(REPO), 'branch': sh(['git', 'rev-parse', '--abbrev-ref', 'HEAD'], REPO), 'commit': head},
           'ros2_ws': {}, 'run': a.run}
    for d in sorted((WS / 'src').iterdir()):
        if not d.is_dir():
            continue
        if sh(['git', 'rev-parse', '--git-dir'], d):
            diff = sh(['git', 'diff', 'HEAD', '--', '.', ':(exclude)*.pyc', ':(exclude)*__pycache__*'], d)
            ent = {'git': sh(['git', 'rev-parse', 'HEAD'], d), 'remote': sh(['git', 'remote', 'get-url', 'origin'], d)}
            if diff:
                (out / 'ws' / f'{d.name}.patch').write_text(diff + '\n'); ent['patch'] = f'ws/{d.name}.patch'
            man['ros2_ws'][d.name] = ent
        else:
            tgz = out / 'ws' / f'{d.name}.tar.gz'
            with tarfile.open(tgz, 'w:gz') as t:
                t.add(d, arcname=d.name, filter=lambda ti: None if any(x in ti.name for x in ('__pycache__', '.pyc', '/build/', '/.git/')) else ti)
            man['ros2_ws'][d.name] = {'tarball': f'ws/{d.name}.tar.gz'}
    for f in ('robot_config.yaml', 'filter_config.yaml', 'sticker_calibration.json'):
        if (LOCALIZER_CFG / f).exists():
            shutil.copy2(LOCALIZER_CFG / f, out / 'configs' / f'localizer_{f}')
    for f in (REPO / 'compliant_insertion_studio' / 'configs').glob('*'):
        if f.is_file():
            shutil.copy2(f, out / 'configs' / f.name)
    shutil.copy2(REPO / 'primitives' / 'shared' / 'cell_layout.yaml', out / 'configs' / 'cell_layout.yaml')
    (out / 'pip_freeze.txt').write_text(sh([sys.executable, '-m', 'pip', 'freeze']) + '\n')
    (out / 'ros_debs.txt').write_text(sh("dpkg -l | awk '/ros-humble-(ur|realsense|rclpy|controller|ros2-control)/{print $2, $3}'") + '\n')
    ev = [f"# Evidence for {a.name}", '', f"run timing: `{a.run}`", '', '```',
          sh([sys.executable, str(REPO / 'ablations' / 'cycle_time.py'), a.run], REPO), '```']
    if a.log and os.path.exists(a.log):
        lines = [l.rstrip() for l in open(a.log) if '[execute_mate] fmb' in l or 'RUN EXIT' in l or 'Completed:' in l]
        ev += ['', '## Certification', '```', *lines, '```']
    (out / 'EVIDENCE.md').write_text('\n'.join(ev) + '\n')
    (out / 'MANIFEST.json').write_text(json.dumps(man, indent=1) + '\n')
    (out / 'RUN.md').write_text(RUN_MD.format(name=a.name, commit=head[:10]))
    subprocess.run(['git', 'add', str(out)], cwd=REPO, check=True)
    subprocess.run(['git', '-c', 'user.name=Aldrin Inbaraj', '-c', 'user.email=125698539+inbarajaldrin@users.noreply.github.com',
                    'commit', '-q', '-m', f'Checkpoint {a.name}: known-good cycle on {head[:10]} (configs, ros2_ws versions, evidence)'],
                   cwd=REPO, check=True)
    subprocess.run(['git', 'tag', f'ckpt/{a.name}'], cwd=REPO, check=True)
    print(f'checkpoint {a.name}: tag ckpt/{a.name}, {out}')
    return 0


RUN_MD = """# Checkpoint {name}

A full FMB1 cycle (reset + assembly, every mate certified) ran on commit `{commit}` with the dependencies in
MANIFEST.json. Tag: `ckpt/{name}`.

## Demo from this checkpoint (leaves the development branch alone)
```bash
cd ~/Documents/ros-mcp-server-verified
git worktree add ~/fmb1_demo_{name} ckpt/{name}          # a frozen copy of the code at this checkpoint
cd ~/fmb1_demo_{name}
bash compliant_insertion_studio/scripts/stack_up.sh      # driver, gripper, camera, grasp points (no arm motion)
# seam servers, from the checkpoint copy:
for s in vision_server grip_adapter drive_mate_server execute_mate_server; do nohup python3 -u -m seams.$s > /tmp/$s.log 2>&1 & done
export FMB_LAYOUT=fmb1_home_2026_10_21 BASE_FROM=icp NO_AIM_BIAS=1 PRE_INSERT_READ=off FOLLOW_YAW=0
J=ablations/ground_truth_resources/Assembly_fmb_assembly_1_results.json
# parts staged (u_brown at the clear area, the others on the table) -> one assembly:
SEAMS_DRIVE=1 SEAMS_CERTIFY=1 python3 -u ablations/replay_real_assembly.py --assembly-json $J --part-at-staging
```
The full cycle with the reset is in compliant_insertion_studio/docs/HANDOFF_NEXT_AGENT.md (2026-10-22/23 section).

## Restore the dependencies to this checkpoint
- ros2_ws packages: for each entry in MANIFEST.json `ros2_ws`, `git -C ~/Desktop/ros2_ws/src/<pkg> checkout <git>` and
  `git apply checkpoints/{name}/<patch>` where a patch is listed; packages saved as a tarball: extract over src/.
  Then `colcon build` in ~/Desktop/ros2_ws.
- calibration: copy configs/localizer_*.yaml|json back to ~/Desktop/ros2_ws/src/aruco_camera_localizer/config/
  (drop the localizer_ prefix); the F/T calibration and cell layout are in the repo at this commit already.
- Python packages: pip_freeze.txt; ROS debs: ros_debs.txt.
"""

if __name__ == '__main__':
    sys.exit(main())
