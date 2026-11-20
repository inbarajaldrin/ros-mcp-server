"""Subprocess budgets shared by every caller of a primitive (lead 2026-11-20, review r2 P3-8).

ROTATE_TIMEOUT_S is rotate_object's subprocess timeout AND the --budget-s it is given: rotate_object refuses
(ROTATE_OUTLASTS_CALLER) before any motion that its caller's timeout would cut, since a trajectory the controller has
accepted keeps running after its client is killed.  90 s: the cell PC's projected worst reorient was 46.7 s
(lab mujoco/evidence/w5b-rotate-timing), before the tier-2 mesh check.
"""
ROTATE_TIMEOUT_S = 90
