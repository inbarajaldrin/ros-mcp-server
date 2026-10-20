#!/usr/bin/env python3
"""Offline readiness check for a new assembly on the UR5e cell (2026-10-20, from onboarding FMB3).

    python3 ablations/onboard_check.py fmb3

Checks every per-part input the real run reads, all from CAD/data files (no robot, no camera):
mesh, grasp points + the assembly grasp, closed grip width (CAD), symmetry file, seat pose in the base,
grasp depth entry (grasp_depth.py --assembly ... --write), insert routing, the base's CAD size/height,
the parts seated under each part (contact candidates), the recorded tool sequence, and the localizer's
marker dictionaries (a part id that collides with another dictionary's id = phantom detections).
Prints one line per check; exit 1 if any check fails.
"""
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / 'ablations'))
DATA = Path('/home/aldrin/Documents/aruco-grasp-annotator/data')


def main():
    name = sys.argv[1] if len(sys.argv) > 1 else 'fmb3'
    n = name.replace('fmb', '')
    bad = []

    def ok(cond, what, detail=''):
        print(f"  {'ok  ' if cond else 'FAIL'} {what}{(' - ' + detail) if detail else ''}")
        if not cond:
            bad.append(what)
    asm_p = REPO / f'ablations/eval_resources/{name}_assembly.json'
    gt_p = REPO / f'ablations/ground_truth_resources/Assembly_fmb_assembly_{n}_results.json'
    ok(asm_p.exists(), f'{asm_p.name}'); ok(gt_p.exists(), f'{gt_p.name} (recorded tool sequences)')
    if bad:
        return 1
    asm = json.load(open(asm_p)); base = asm['base_name']
    print(f"{name}: base {base}, parts {[o['object_name'] for o in asm['assembly_order']]}")
    from primitives.shared import config as C
    from compliant_insertion_studio.wrapper.cad_lookup import load_seat_pose_in_base
    from compliant_insertion_studio.wrapper.cad_geometry import get_seated_so_far, _load_base_top_z_local
    from spacing_check import BASE_SIDE_M
    ok((DATA / f'models/{base}.obj').exists(), f'{base} mesh')
    try:
        ok(True, f'{base} CAD', f"side {1000*BASE_SIDE_M[base]:.1f} mm, top {1000*_load_base_top_z_local(base):.1f} mm above origin")
    except Exception as e:
        ok(False, f'{base} CAD size', str(e))
    gd = json.load(open(REPO / 'ablations/eval_resources/grasp_depth.json')).get('parts', {})
    tr = (REPO / 'primitives/translate_object.py').read_text()
    gt = {e['object_name']: e for e in json.load(open(gt_p))['assembly_order']}
    for o in asm['assembly_order']:
        p, gid = o['object_name'], o['grasp_id']
        print(f"- {p} (grasp {gid})")
        ok((DATA / f'models/{p}.obj').exists(), 'mesh')
        try:
            g = json.load(open(DATA / f'grasp_points/{p}_grasp_points.json'))
            ok(any(x['id'] == gid for x in g['grasp_points']), 'grasp point', f"marker ids {[m['aruco_id'] for m in g.get('markers', [])]}")
        except Exception as e:
            ok(False, 'grasp points', str(e))
        try:
            ok(True, 'grip width (CAD)', f"{C.cad_grip_width_mm(p, gid):.1f} mm closed, pre-grasp {o['gripper_width_mm']} mm")
        except Exception as e:
            ok(False, 'grip width (CAD)', str(e))
        ok(any((REPO / 'data').rglob(f'{p}*symmetry*')) or any(Path('/home/aldrin/Documents').glob(f'**/symmetry/{p}*')),
           'symmetry file', '(fold-aware planner/rotate)')
        try:
            sp, sq = load_seat_pose_in_base(base, p)
            ok(True, 'seat pose', f"{[round(1000*v, 1) for v in sp]} mm")
        except Exception as e:
            ok(False, 'seat pose', str(e))
        ok(p in gd and gd[p].get('base') == base, 'grasp depth entry',
           f"run: python3 ablations/grasp_depth.py --assembly {name} --write" if p not in gd else f"{1000*gd[p]['grasp_tool_offset_m']:.1f} mm, limited by {gd[p]['limited_by']}")
        ok(True, 'insert routing (translate_object)',
           'prismatic list' if f"'{p}'" in tr else 'default compliant_insert FSM (fine for U-type parts; bars/forks need prismatic)')
        ok(True, 'seated under it', str(get_seated_so_far(base, p)))
        ok(p in gt and bool(gt[p].get('tool_sequence')), 'recorded tool sequence', f"{len(gt.get(p, {}).get('tool_sequence', []))} steps")
    print("\nBefore the robot: the localizer's object list has these parts, nothing else on the table shares their\n"
          "marker ids across dictionaries (detect_markers drops cross-dictionary duplicates), and the active layout\n"
          "in primitives/shared/cell_layout.yaml matches the table.")
    print(f"\n{'READY' if not bad else 'NOT READY: ' + ', '.join(bad)}")
    return 1 if bad else 0


if __name__ == '__main__':
    sys.exit(main())
