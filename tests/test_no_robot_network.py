"""The autouse guard is active in every test (review r5 P3-4)."""
import socket

import pytest

from ablations import cell_io


def test_the_dashboard_is_stubbed_without_a_per_test_patch(no_robot_network):
    assert cell_io.dashboard_stop() is False and cell_io.robot_safety_status() == "NORMAL"
    assert no_robot_network == {"dashboard_stop": 1, "robot_safety_status": 1}


@pytest.mark.parametrize("address", [("192.168.1.111", 29999), ("192.168.1.111", 30002), ("8.8.8.8", 53)])
def test_no_connection_leaves_this_machine(address):
    with pytest.raises(ConnectionRefusedError, match="no connection to anything but loopback"):
        socket.create_connection(address, timeout=0.5)
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        with pytest.raises(ConnectionRefusedError, match="no connection to anything but loopback"):
            s.connect(address)
    finally:
        s.close()


def test_stop_group_cannot_reach_the_dashboard_even_unpatched(tmp_path):
    """_stop_group's escalation calls the module-level dashboard_stop: the stub answers, the robot never does."""
    import subprocess
    import sys
    proc = subprocess.Popen([sys.executable, "-c", "import signal, time\n"
                             "signal.signal(signal.SIGINT, signal.SIG_IGN)\nsignal.signal(signal.SIGTERM, signal.SIG_IGN)\n"
                             "print('ready', flush=True)\ntime.sleep(30)"], stdout=subprocess.PIPE, start_new_session=True)
    proc.stdout.readline()
    try:
        assert cell_io._stop_group(proc, 0.5, "guard test").startswith("stuck")   # the stub answered False
    finally:
        proc.kill()
        proc.wait()
