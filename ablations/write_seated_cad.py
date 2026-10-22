#!/usr/bin/env python3
"""Write the SEATED CAD of an assembly: fmb_assemblyN_seated.json beside the nominal file (each part dropped along the
base z onto the base and the parts seated before it - cad_lookup._settle_drop) and a combined OBJ of the seated assembly
for viewing. The nominal file keeps its design pose (clearances built in); the seated file is what the cell measures
against (operator 10-22: "refer to a seated cad, a cad is made with tolerances").

    python3 ablations/write_seated_cad.py base1 [base3 ...]
"""
import json
import sys
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation as R

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO)]
from compliant_insertion_studio.wrapper import cad_lookup as cl


def write(base):
    src = cl._resolve_assembly_path(base)
    d = json.load(open(src))
    out_v, out_f, n0 = [], [], 0
    for c in d['components']:
        name = c['name']
        if name != base:
            drop = float(cl._settle_drop(base, name))
            c['position']['z'] = round(float(c['position']['z']) - drop, 6)
            c['seated_drop_mm'] = round(1000 * drop, 3)
        p = np.array([c['position'][k] for k in 'xyz'], float)
        q = [c['rotation']['quaternion'][k] for k in 'xyzw']
        v, f = [], []
        for line in open(cl.MODELS_DIR / f"{name}.obj"):
            if line.startswith('v '):
                v.append([float(x) / 100.0 for x in line.split()[1:4]])
            elif line.startswith('f '):
                f.append([int(t.split('/')[0]) for t in line.split()[1:]])
        out_v.append((name, R.from_quat(q).apply(np.array(v)) + p))
        out_f.append([[i + n0 for i in ff] for ff in f])
        n0 += len(v)
    d['seated'] = {'from': src.name, 'by': 'ablations/write_seated_cad.py',
                   'how': 'each part dropped along the base z onto the base + parts seated before it (CAD, 1.5 mm height '
                          'maps); seated_drop_mm per part'}
    out = src.with_name(src.stem + '_seated.json')
    json.dump(d, open(out, 'w'), indent=2)
    obj = src.with_name(src.stem + '_seated.obj')
    with open(obj, 'w') as fh:
        fh.write(f"# {base} seated assembly (m), from {src.name}\n")
        for (name, V), F in zip(out_v, out_f):
            fh.write(f"o {name}\n")
            fh.writelines(f"v {x:.6f} {y:.6f} {z:.6f}\n" for x, y, z in V)
            fh.writelines("f " + " ".join(map(str, ff)) + "\n" for ff in F)
    print(out)
    print(obj)
    for c in d['components']:
        if 'seated_drop_mm' in c:
            print(f"  {c['name']:18s} z {1000 * c['position']['z']:7.2f} mm (dropped {c['seated_drop_mm']:.2f})")


if __name__ == '__main__':
    for b in sys.argv[1:] or ['base1']:
        write(b)
