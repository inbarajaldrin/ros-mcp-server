from seams import seat_learn as SL


def _rows(tmp_path, obj, rows):
    p = tmp_path / 'h.json'
    for applied, end, ok in rows:
        SL.record_insert(obj, applied, end, True, path=p)
        SL.mark_certified(obj, ok, path=p)
    return p


def test_contact_ended_inserts_learn_where_they_bottomed(tmp_path):
    # u_brown 10-23: aimed at table dz -1.24 mm, ended +0.60 mm over the aim on contact -> seat at -0.64 mm
    p = _rows(tmp_path, 'u_brown', [(-0.00124, 0.0006, True)] * 4)
    dz, n = SL.learned_dz('u_brown', path=p)
    assert n == 4 and abs(dz - (-0.00064)) < 1e-6


def test_floor_ended_inserts_keep_the_floor_where_it_seated(tmp_path):
    # line_green: aimed at +5.81 mm, every clean push ran to the -2 mm floor -> learned +5.81 (no walk down)
    p = _rows(tmp_path, 'line_green', [(0.00581, -0.00199, True)] * 5)
    dz, _ = SL.learned_dz('line_green', path=p)
    assert abs(dz - 0.00582) < 2e-5


def test_inserts_that_needed_a_recovery_do_not_count(tmp_path):
    p = _rows(tmp_path, 'line_green', [(0.00581, -0.00199, True), (0.00581, -0.00199, False), (0.00581, -0.00199, False)])
    dz, n = SL.learned_dz('line_green', path=p)
    assert dz is None and n == 1


def test_disagreeing_inserts_give_no_value(tmp_path):
    p = _rows(tmp_path, 'u_orange', [(0.0, 0.000, True), (0.0, 0.004, True), (0.0, -0.004, True), (0.0, 0.006, True)])
    assert SL.learned_dz('u_orange', path=p)[0] is None
