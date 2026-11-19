import numpy as np

from primitives.shared.collision import UR5E_LINK_RADII, check_self_collision

# The r22 u_brown hover branch: 126.5 mm between the base and forearm DH segments.
R22_HOVER = np.array([-0.3706677152601609, -1.6457364199526756, -2.8426231411865426,
                      -1.8204146172954576, -0.33883765477109506, -3.121392096863145])


def test_self_collision_keeps_the_conservative_radii():
    """One geometry for the real arm and the simulator; never shrunk to admit a stance."""
    assert UR5E_LINK_RADII == (0.075, 0.065, 0.055, 0.045, 0.045, 0.040)


def test_r22_hover_branch_is_refused():
    """Base 75 mm + forearm 55 mm + 10 mm margin need 140 mm; the branch has 126.5 mm."""
    assert check_self_collision(R22_HOVER)
