from ablations.replay_sim_assembly import _command_for_step, _result_from_output, _staging_view_command


def test_commands_preserve_recorded_sim_calls():
    cmd, timeout = _command_for_step(
        "ros-mcp-server__translate_object(action='insert', object_name='u_brown', "
        "base_name='base1', mode='sim')"
    )
    assert cmd[-5:] == ["ablations.sim_regular_insert", "--object-name", "u_brown",
                        "--base-name", "base1"]
    assert timeout == 30


def test_non_sim_step_is_refused():
    try:
        _command_for_step("ros-mcp-server__move_home(mode='real')")
    except ValueError as error:
        assert "refuses non-sim" in str(error)
    else:
        raise AssertionError("real step was not refused")


def test_move_home_uses_sim_primitive():
    command, timeout = _command_for_step("ros-mcp-server__move_home(mode='sim')")
    assert command[-3:] == ["primitives.move_home", "--mode", "sim"]
    assert timeout == 120


def test_last_result_envelope_wins():
    text = "__RESULT_JSON__\n{\"result\":\"failure\"}\n__END_RESULT_JSON__\n" \
           "__RESULT_JSON__\n{\"result\":\"success\"}\n__END_RESULT_JSON__\n"
    assert _result_from_output(text) == {"result": "success"}


def test_fov_view_uses_production_camera_aim():
    cmd, timeout = _staging_view_command()
    assert cmd[3] == "primitives.move_cartesian"
    assert "--look-at-z" in cmd
    assert "--quat" in cmd
    assert timeout == 120
