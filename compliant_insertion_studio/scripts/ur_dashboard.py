#!/usr/bin/env python3
# Rewritten from protocol (not copied) for prismatic-manipulation.
# Origin of the idea: isaac-sim-mcp/scripts/ur_dashboard.py.
# Protocol per UR e-Series Dashboard Server docs + GPT parity + live URSim 5.25.1
# probe (2026-06-19). All command strings / response semantics live in dashboard_spec.
#   https://www.universal-robots.com/articles/ur/dashboard-server-e-series-port-29999/
"""Raw-TCP client for the UR / URSim Dashboard Server (port 29999).

Speaks the dashboard wire protocol directly — deliberately NOT via the
ur_robot_driver's ROS2 services (ursim_cli.py does that). The dashboard server is
part of URSim itself and answers the moment the container boots, BEFORE any ROS
driver exists. That breaks the chicken-and-egg of bring-up: you can't power on the
robot if the only thing that powers it on comes up with the driver.

ZERO ROS / third-party dependency — stdlib + dashboard_spec only.

Usage:
  ur_dashboard.py [--host H] [--port 29999] [--timeout S] [--wait S] CMD [CMD ...]

Queries:  mode safety safetymode state running saved loaded remote
          version polyscope serial model opmode
Actions:  power_on power_off brake_release stop play pause shutdown
          unlock close_popup close_safety_popup restart_safety clear_opmode
          load <prog.urp>   load_installation <name>   popup <text>
          log <message>     set_opmode <manual|automatic>
          flight_report [controller|software|system]   support_file <dir>
Combos:   power_up                  (power on -> wait -> brake release -> RUNNING)
          start_external [urp]      (stop -> load -> play -> confirm running)
          status                    (mode | safety | loaded | opmode)
          wait_dashboard            (block until the server answers)
          selftest                  (read-only validation of every query)
          selftest --cycle --confirm-cycle
                                    (adds a power round-trip that restores the exact
                                     initial state; refuses without --confirm-cycle and
                                     always refuses with --real)
          raw <text>                (send an arbitrary line verbatim)

Exit code is non-zero if any step fails. Designed for shell orchestration
(see scripts/launch_ur.sh).
"""

import argparse
import os
import socket
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # sibling import
import dashboard_spec as spec

DEFAULT_URP = "/ursim/programs/external_control.urp"

# CLI token -> canonical spec name (simple direct-send commands).
ALIAS = {
    # queries
    "mode": "robot_mode", "safety": "safety_status", "safetymode": "safety_mode",
    "state": "program_state", "running": "program_running", "saved": "program_saved",
    "loaded": "loaded_program", "remote": "remote_control",
    "version": "version", "polyscope": "polyscope_version", "serial": "serial_number",
    "model": "robot_model", "opmode": "operational_mode",
    # actions (no arg)
    "power_on": "power_on", "power_off": "power_off", "brake_release": "brake_release",
    "stop": "stop", "play": "play", "pause": "pause", "shutdown": "shutdown",
    "unlock": "unlock_protective_stop", "close_popup": "close_popup",
    "close_safety_popup": "close_safety_popup", "restart_safety": "restart_safety",
    "clear_opmode": "clear_operational_mode", "quit": "quit",
    # actions (one trailing arg)
    "load": "load", "load_installation": "load_installation", "popup": "popup",
    "log": "add_to_log", "set_opmode": "set_operational_mode",
    "flight_report": "generate_flight_report", "support_file": "generate_support_file",
}
COMBOS = {"power_up", "start_external", "status", "wait_dashboard", "selftest", "raw"}


class DashboardClient:
    def __init__(self, host="127.0.0.1", port=29999, timeout=5.0):
        self.host, self.port, self.timeout = host, port, timeout
        self.sock = None

    def connect(self):
        self.sock = socket.create_connection((self.host, self.port), timeout=self.timeout)
        self.sock.settimeout(self.timeout)
        return self._recv()  # greeting line

    def _recv(self):
        """Read one newline-terminated response."""
        buf = b""
        while b"\n" not in buf:
            try:
                chunk = self.sock.recv(4096)
            except socket.timeout:
                break
            if not chunk:
                break
            buf += chunk
        return buf.decode(errors="replace").strip()

    def send(self, line):
        self.sock.sendall((line + "\n").encode())
        return self._recv()

    def close(self):
        if self.sock:
            try:
                self.sock.sendall(b"quit\n")
            except OSError:
                pass
            self.sock.close()
            self.sock = None

    # --- helpers --------------------------------------------------------------
    def send_named(self, name, arg=None):
        """Send a canonical command; return (ok, response)."""
        resp = self.send(spec.wire(name, arg))
        return spec.succeeded(name, resp), resp

    def poll(self, line, predicate, attempts=20, interval=1.0):
        last = ""
        for _ in range(attempts):
            last = self.send(line)
            if predicate(last):
                return True, last
            time.sleep(interval)
        return False, last

    # --- combos ---------------------------------------------------------------
    def power_up(self, attempts=45, interval=1.0):
        """Drive POWER_OFF -> RUNNING as a state machine.

        A freshly-booted controller silently drops an early 'power on', so we
        re-issue the right command for the CURRENT mode each tick rather than
        firing once: 'power on' while POWER_OFF, 'brake release' once IDLE,
        succeed at RUNNING.
        """
        last = ""
        for _ in range(attempts):
            last = self.send(spec.wire("robot_mode"))
            mode = spec.parse_robot_mode(last)
            if mode == "RUNNING":
                return True, "robot powered on, brakes released (RUNNING)"
            if mode == "IDLE":
                r = self.send(spec.wire("brake_release"))
                if spec.is_denied(r):
                    return False, f"brake release denied (pendant in Local mode?): {r}"
            elif mode == "POWER_OFF":
                r = self.send(spec.wire("power_on"))
                if spec.is_denied(r):
                    return False, f"power on denied (pendant in Local mode?): {r}"
            # BOOTING / POWER_ON / CONFIRM_SAFETY: wait for the next tick
            time.sleep(interval)
        return False, f"robot never reached RUNNING (last: {last})"

    def start_external(self, urp=DEFAULT_URP):
        """stop -> load external_control -> play -> confirm running."""
        # 'stop' is a best-effort pre-clear (it legitimately "fails" when nothing is
        # running), but a Local-mode DENIAL is a real blocker — abort with the true cause.
        _, r = self.send_named("stop")
        if spec.is_denied(r):
            return False, f"stop denied (pendant in Local mode?): {r}"
        time.sleep(0.5)
        ok, r = self.send_named("load", urp)
        if not ok:
            return False, f"load failed: {r}"
        time.sleep(0.5)
        ok, r = self.send_named("play")
        if not ok:
            return False, f"play failed: {r}"
        ok, last = self.poll(spec.wire("program_running"),
                             lambda x: spec.parse_running(x) is True, attempts=10)
        if not ok:
            return False, f"program not running after play (last: {last})"
        return True, "external_control.urp playing"

    def status(self):
        return True, " | ".join(self.send(spec.wire(n))
                                for n in ("robot_mode", "safety_status",
                                          "loaded_program", "operational_mode"))


def wait_dashboard(host, port, timeout_s):
    """Block until the dashboard server accepts a connection and greets us."""
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        try:
            c = DashboardClient(host, port, timeout=2.0)
            greet = c.connect()
            c.close()
            if "Dashboard Server" in greet:
                return True
        except OSError:
            pass
        time.sleep(1.0)
    return False


# --- selftest ----------------------------------------------------------------
def selftest(client, cycle=False, confirm=False, real=False):
    """Read-only validation of every query; --cycle adds a self-restoring round-trip.

    --cycle snapshots the initial loaded program + power/run state, power-cycles the
    robot, then RESTORES that exact state and verifies it. It is destructive, so it
    refuses unless `confirm` is set, and always refuses when `real` is set.

    Returns (all_passed, list_of_(label, ok, detail)).
    """
    results = []

    def check(label, ok, detail):
        results.append((label, ok, detail))

    # 1. greeting
    check("greeting", spec.GREETING in (client.connect() or ""), "")

    # 2. every read-only query: must be recognised + (where applicable) in-enum
    enum_for = {
        "robot_mode": (spec.parse_robot_mode, spec.ROBOT_MODES),
        "safety_status": (spec.parse_safety, spec.SAFETY_STATUSES),
        "safety_mode": (spec.parse_safety, spec.SAFETY_STATUSES),
        "operational_mode": (lambda r: r.strip(), spec.OPERATIONAL_MODES),
        "robot_model": (lambda r: r.strip(), spec.ROBOT_MODELS),
    }
    for name in sorted(spec.READ_ONLY):
        resp = client.send(spec.wire(name))
        if spec.is_unrecognized(resp):
            check(name, False, f"UNSUPPORTED on this version: {resp!r}")
            continue
        if spec.is_denied(resp):
            check(name, False, f"denied: {resp!r}")
            continue
        ok, detail = True, resp
        if name in enum_for:
            parse, allowed = enum_for[name]
            val = parse(resp)
            ok = val in allowed
            detail = f"{resp!r} -> {val}" + ("" if ok else f" NOT IN {allowed}")
        check(name, ok, detail)

    if not cycle:
        return all(ok for _, ok, _ in results), results

    # 3. --cycle: power round-trip that RESTORES the exact initial state.
    if real:
        check("cycle", False, "refused: --real set (never power-cycle real hardware)")
        return False, results
    if not confirm:
        check("cycle", False,
              "refused: --cycle is destructive — pass --confirm-cycle to proceed")
        return False, results

    # snapshot the EXACT initial state so we can put it back
    init_mode = spec.parse_robot_mode(client.send(spec.wire("robot_mode")))
    init_loaded = spec.parse_loaded_program(client.send(spec.wire("loaded_program")))
    init_running = spec.parse_running(client.send(spec.wire("program_running")))
    check("cycle:snapshot", init_mode is not None and init_loaded is not None,
          f"mode={init_mode} loaded={init_loaded} running={init_running}")
    if init_loaded is None:
        check("cycle", False, "refused: no program loaded to restore")
        return False, results

    # benign reversible actions
    ok, r = client.send_named("add_to_log", "prismatic selftest --cycle"); check("cycle:add_to_log", ok, r)
    ok, r = client.send_named("popup", "prismatic selftest"); check("cycle:popup", ok, r)
    ok, r = client.send_named("close_popup"); check("cycle:close_popup", ok, r)

    # power round-trip
    _, r = client.send_named("stop"); check("cycle:stop", not spec.is_denied(r), r)
    time.sleep(0.5)
    ok, r = client.send_named("power_off"); check("cycle:power_off", ok, r)
    ok, msg = client.power_up(); check("cycle:power_up", ok, msg)

    # RESTORE: reload the exact program that was loaded; replay it only if it was running
    if init_running:
        ok, msg = client.start_external(init_loaded); check("cycle:restore_play", ok, msg)
    else:
        ok, r = client.send_named("load", init_loaded); check("cycle:restore_load", ok, r)

    # verify EXACT restoration (mode + loaded program + run state)
    final_mode = spec.parse_robot_mode(client.send(spec.wire("robot_mode")))
    final_loaded = spec.parse_loaded_program(client.send(spec.wire("loaded_program")))
    final_running = spec.parse_running(client.send(spec.wire("program_running")))
    check("cycle:restored",
          final_mode == init_mode and final_loaded == init_loaded and final_running == init_running,
          f"mode={final_mode} loaded={final_loaded} running={final_running} "
          f"(expected mode={init_mode} loaded={init_loaded} running={init_running})")
    return all(ok for _, ok, _ in results), results


def _run_selftest(args):
    cycle = "--cycle" in args.commands
    confirm = "--confirm-cycle" in args.commands
    client = DashboardClient(args.host, args.port, args.timeout)
    try:
        passed, results = selftest(client, cycle=cycle, confirm=confirm, real=args.real)
    finally:
        client.close()
    width = max(len(label) for label, _, _ in results)
    for label, ok, detail in results:
        mark = "PASS" if ok else "FAIL"
        print(f"  [{mark}] {label:<{width}}  {detail}")
    print(f"  selftest {'PASSED' if passed else 'FAILED'} "
          f"({sum(ok for _, ok, _ in results)}/{len(results)})")
    return 0 if passed else 1


def main():
    ap = argparse.ArgumentParser(description="UR/URSim Dashboard Server raw-TCP client")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=29999)
    ap.add_argument("--timeout", type=float, default=5.0)
    ap.add_argument("--wait", type=float, default=0.0, metavar="SECONDS",
                    help="wait up to SECONDS for the dashboard server before running commands")
    ap.add_argument("--real", action="store_true",
                    help="treat the target as real hardware (makes 'selftest --cycle' refuse)")
    ap.add_argument("commands", nargs=argparse.REMAINDER,
                    help="dashboard commands / combos (see module docstring)")
    args = ap.parse_args()

    # --wait / wait_dashboard: block for the server to come up.
    if args.wait > 0 or (args.commands and args.commands[0] == "wait_dashboard"):
        secs = args.wait if args.wait > 0 else 90.0
        if not wait_dashboard(args.host, args.port, secs):
            print(f"  x dashboard server not up at {args.host}:{args.port} after {secs}s",
                  file=sys.stderr)
            sys.exit(1)
        print(f"  - dashboard server up at {args.host}:{args.port}")
        if args.commands and args.commands[0] == "wait_dashboard":
            args.commands = args.commands[1:]

    if not args.commands:
        return

    # selftest owns its own connection lifecycle.
    if args.commands[0] == "selftest":
        sys.exit(_run_selftest(args))

    client = DashboardClient(args.host, args.port, args.timeout)
    try:
        client.connect()
    except OSError as e:
        print(f"  x cannot connect to dashboard {args.host}:{args.port}: {e}", file=sys.stderr)
        sys.exit(1)

    rc = 0
    cmds = args.commands
    i = 0
    try:
        while i < len(cmds):
            c = cmds[i]
            if c == "power_up":
                ok, msg = client.power_up()
            elif c == "start_external":
                urp = DEFAULT_URP
                if i + 1 < len(cmds) and not _is_keyword(cmds[i + 1]):
                    urp = cmds[i + 1]; i += 1
                ok, msg = client.start_external(urp)
            elif c == "status":
                ok, msg = client.status()
            elif c == "raw":
                if i + 1 >= len(cmds):
                    ok, msg = False, "raw: missing argument"
                else:
                    i += 1; msg = client.send(cmds[i])
                    ok = bool(msg) and not spec.is_unrecognized(msg) and not spec.is_denied(msg)
            elif c in ALIAS:
                name = ALIAS[c]
                arg = None
                if name in spec.TAKES_ARG:
                    if i + 1 >= len(cmds):
                        ok, msg = False, f"{c}: missing argument"
                        print(f"  x {c}: {msg}", file=sys.stderr); rc = 1; break
                    i += 1; arg = cmds[i]
                ok, msg = client.send_named(name, arg)
            else:
                ok, msg = False, f"unknown command: {c}"
            mark = "-" if ok else "x"
            print(f"  {mark} {c}: {msg}")
            if not ok:
                rc = 1
                break
            i += 1
    finally:
        client.close()
    sys.exit(rc)


def _is_keyword(tok):
    return tok in ALIAS or tok in COMBOS


if __name__ == "__main__":
    main()
