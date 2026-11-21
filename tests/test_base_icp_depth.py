"""base_icp.read_icp converts depth through config.depth_to_m (the measured D415 correction), like every depth reader.

2026-11-21: it used raw / 1000, so a base read sat ~4.5 mm low and the table-height gate refused base1 on the live cell
(raw fit z -67.0 mm, corrected -61.9 mm against -62.5)."""
import ast
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / 'ablations' / 'base_icp.py'


def _read_icp_src():
    tree = ast.parse(SRC.read_text())
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'read_icp')
    return ast.get_source_segment(SRC.read_text(), fn)


def test_read_icp_converts_depth_through_depth_to_m():
    src = _read_icp_src()
    assert 'depth_to_m(' in src
    assert '/ 1000.0' not in src.split('depth_to_m(')[0].split('capture(')[-1]


def test_depth_to_m_undoes_the_measured_correction():
    import numpy as np
    from primitives.shared import config as C
    true = 0.5
    raw_mm = (C.DEPTH_SCALE * true + C.DEPTH_OFFSET_M) * 1000.0
    assert abs(float(C.depth_to_m(np.array([raw_mm]))[0]) - true) < 1e-9
    assert float(C.depth_to_m(np.array([0.0]))[0]) == 0.0
