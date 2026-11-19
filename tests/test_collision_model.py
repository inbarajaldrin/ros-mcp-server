import numpy as np

from primitives.shared.collision import UR5E_LINK_RADII, check_self_collision


def test_measured_ur5e_radii_accept_exact_r22_hover_branch():
    """The exact u_brown hover branch has 126.5 mm base/forearm clearance.

    The vendored measured radii plus the checker's 10 mm safety margin need
    110 mm. The former approximate radii incorrectly required 140 mm.
    """
    joints = np.array([
        -0.3706677152601609,
        -1.6457364199526756,
        -2.8426231411865426,
        -1.8204146172954576,
        -0.33883765477109506,
        -3.121392096863145,
    ])

    assert UR5E_LINK_RADII[:3] == (0.060, 0.054, 0.040)
    assert not check_self_collision(joints)
