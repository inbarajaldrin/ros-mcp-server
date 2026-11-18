from ablations.replay_sim_assembly import _command_for_step, _result_from_output


def test_commands_preserve_recorded_sim_calls():
    cmd, timeout = _command_for_step(
        "ros-mcp-server__translate_object(action='insert', object_name='u_brown', "
        "base_name='base1', mode='sim')"
    )
    assert cmd[-7:] == ["--mode", "sim", "--object-name", "u_brown",
                        "--base-name", "base1", "--insert"]
    assert timeout == 360


def test_non_sim_step_is_refused():
    try:
        _command_for_step("ros-mcp-server__move_home(mode='real')")
    except ValueError as error:
        assert "refuses non-sim" in str(error)
    else:
        raise AssertionError("real step was not refused")


def test_last_result_envelope_wins():
    text = "__RESULT_JSON__\n{\"result\":\"failure\"}\n__END_RESULT_JSON__\n" \
           "__RESULT_JSON__\n{\"result\":\"success\"}\n__END_RESULT_JSON__\n"
    assert _result_from_output(text) == {"result": "success"}
