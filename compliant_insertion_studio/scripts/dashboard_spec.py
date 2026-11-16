#!/usr/bin/env python3
# Ported into prismatic-manipulation from isaac-sim-mcp/scripts/ur_dashboard.py +
# ros-mcp-server/utils/ursim_cli.py (single source of truth for the command set).
# Protocol pinned 2026-06-19 three ways:
#   - UR e-Series Dashboard Server docs (port 29999):
#       https://www.universal-robots.com/articles/ur/dashboard-server-e-series-port-29999/
#       https://www.universal-robots.com/manuals/EN/HTML/SW5_25/Content/prod-dashboard/Dashboard_table.htm
#   - GPT independent parity enumeration (codex session 019edf43-9426-7602-88a8-a5a3c64e8388)
#   - LIVE probe of URSim 5.25.1 (where docs and probe conflict, the live probe wins)
"""Single source of truth for the UR / URSim Dashboard Server (port 29999) protocol.

PURE DATA — no imports (not even stdlib). This is intentional: the raw-TCP client
(ur_dashboard.py) must stay zero-dependency so it can run BEFORE the ROS driver /
rclpy context exists. Both transports (raw TCP and the ROS /dashboard_client/*
services) import this module so the command vocabulary and the response semantics
live in exactly one place.

Layout:
  WIRE                  canonical_name -> exact wire command string
  READ_ONLY / ACTION    classification of every canonical_name
  LOCAL_MODE_BLOCKED    actions the real robot rejects when the pendant is in Local mode
  SUCCESS               canonical_name -> predicate(response_str) for the raw-TCP path
  ROBOT_MODES / SAFETY_STATUSES / PROGRAM_STATES / OPERATIONAL_MODES   enum value sets
  GREETING / UNRECOGNIZED / REMOTE_DENIAL   sentinel response fragments
  parse_*()             pure helpers to normalise wire responses to semantic values
"""

# --- sentinel response fragments (live-verified on URSim 5.25.1) --------------
GREETING = "Connected: Universal Robots Dashboard Server"
# This version's "not a command" reply (NOT documented by UR; pinned by live probe on
# URSim 5.25.1). Version-pinned: other PolyScope versions MAY word this differently, so
# is_unrecognized() checks a small set of known forms rather than this one literal.
UNRECOGNIZED = "could not understand:"
UNRECOGNIZED_FORMS = ("could not understand:", "unknown command", "not understood",
                      "is not a recognized")
# Real-robot denial when the pendant is in Local mode (documented + ursim_cli-verified):
REMOTE_DENIAL = "Command is not allowed due to safety reasons"

# --- command vocabulary: canonical_name -> exact wire string ------------------
# Commands that take an argument are marked in TAKES_ARG; the wire string here is
# the bare verb (the arg is appended by the caller, e.g. "load " + path).
WIRE = {
    # --- read-only queries ---
    "robot_mode":        "robotmode",
    "safety_status":     "safetystatus",
    "safety_mode":       "safetymode",            # deprecated (use safety_status); still answers
    "program_state":     "programState",
    "program_running":   "running",
    "program_saved":     "isProgramSaved",
    "loaded_program":    "get loaded program",
    "remote_control":    "is in remote control",
    "polyscope_version": "PolyscopeVersion",
    "version":           "version",               # exists from PolyScope 5.13.0+
    "serial_number":     "get serial number",
    "robot_model":       "get robot model",
    "operational_mode":  "get operational mode",
    # --- state-changing actions ---
    "power_on":          "power on",
    "power_off":         "power off",
    "brake_release":     "brake release",
    "stop":              "stop",
    "play":              "play",
    "pause":             "pause",
    "load":              "load",                  # + " <program.urp>"
    "load_installation": "load installation",     # + " <name.installation>"
    "unlock_protective_stop": "unlock protective stop",
    "close_popup":       "close popup",
    "close_safety_popup": "close safety popup",
    "popup":             "popup",                 # + " <text>"
    "add_to_log":        "addToLog",              # + " <message>"
    "restart_safety":    "restart safety",
    "set_operational_mode": "set operational mode",  # + " manual|automatic"
    "clear_operational_mode": "clear operational mode",
    "generate_flight_report": "generate flight report",  # + " controller|software|system"
    "generate_support_file":  "generate support file",   # + " <dir>"
    "shutdown":          "shutdown",
    "quit":              "quit",
}

# Commands that require an appended argument.
TAKES_ARG = frozenset({
    "load", "load_installation", "popup", "add_to_log",
    "set_operational_mode", "generate_flight_report", "generate_support_file",
})

# --- classification -----------------------------------------------------------
READ_ONLY = frozenset({
    "robot_mode", "safety_status", "safety_mode", "program_state",
    "program_running", "program_saved", "loaded_program", "remote_control",
    "polyscope_version", "version", "serial_number", "robot_model",
    "operational_mode",
})
ACTION = frozenset(WIRE) - READ_ONLY - {"quit"}  # quit is connection mgmt, neither

# Real robot rejects these in Local mode (see REMOTE_DENIAL). URSim ignores this and
# executes them anyway — so this set is only consulted in real mode.
LOCAL_MODE_BLOCKED = frozenset({
    "power_on", "power_off", "brake_release", "stop", "play", "pause",
    "load", "load_installation", "unlock_protective_stop", "restart_safety",
    "close_safety_popup", "set_operational_mode", "shutdown",
})

# --- enum value sets (for interpreting / validating responses) ----------------
ROBOT_MODES = ("NO_CONTROLLER", "DISCONNECTED", "CONFIRM_SAFETY", "BOOTING",
               "POWER_OFF", "POWER_ON", "IDLE", "BACKDRIVE", "RUNNING")
SAFETY_STATUSES = ("NORMAL", "REDUCED", "PROTECTIVE_STOP", "RECOVERY",
                   "SAFEGUARD_STOP", "SYSTEM_EMERGENCY_STOP", "ROBOT_EMERGENCY_STOP",
                   "VIOLATION", "FAULT", "AUTOMATIC_MODE_SAFEGUARD_STOP",
                   "SYSTEM_THREE_POSITION_ENABLING_STOP")
PROGRAM_STATES = ("STOPPED", "PLAYING", "PAUSED")
OPERATIONAL_MODES = ("MANUAL", "AUTOMATIC", "NONE")
ROBOT_MODELS = ("UR3", "UR5", "UR10", "UR16", "UR20", "UR30")

# --- raw-TCP success predicates ----------------------------------------------
# A response is a failure if it is the unrecognized-command reply or the Local-mode
# denial; those are checked centrally (is_unrecognized / is_denied). Beyond that,
# each action has an expected success fragment. Queries always "succeed" (the
# response IS the datum) as long as it is recognised and not denied.
SUCCESS = {
    "power_on":          lambda r: r.startswith("Powering on"),
    "power_off":         lambda r: r.startswith("Powering off"),
    "brake_release":     lambda r: r.startswith("Brake releasing"),
    "stop":              lambda r: r.startswith("Stopped"),
    "play":              lambda r: r.startswith("Starting program"),
    "pause":             lambda r: r.startswith("Pausing program"),
    # load success: "Loading program: <p>"  (NOT "Loaded" — that string never appears)
    "load":              lambda r: r.startswith("Loading program:"),
    "load_installation": lambda r: r.startswith("Loading installation:"),
    "unlock_protective_stop": lambda r: r.startswith("Protective stop releasing"),
    "close_popup":       lambda r: "closing popup" in r.lower(),
    "close_safety_popup": lambda r: "closing safety popup" in r.lower(),
    "popup":             lambda r: "showing popup" in r.lower(),
    "add_to_log":        lambda r: r.startswith("Added log message"),
    "restart_safety":    lambda r: r.startswith("Restarting safety"),
    "set_operational_mode": lambda r: r.startswith("Setting operational mode:"),
    "clear_operational_mode": lambda r: "no longer controlled" in r.lower(),
    # [UNVERIFIED format] UR docs don't pin the success string; best-effort: non-empty,
    # no error/fail wording. Treat the verdict as version-specific, not a hard contract.
    "generate_flight_report": lambda r: bool(r) and not any(
        w in r.lower() for w in ("error", "fail", "could not")),
    "generate_support_file":  lambda r: r.startswith("Completed successfully:"),
    "shutdown":          lambda r: r.startswith("Shutting down"),
    "quit":              lambda r: r.startswith("Disconnected"),
}


# --- pure response helpers ----------------------------------------------------
def is_unrecognized(resp):
    """True if the server didn't recognise the command.

    Checks a set of known wordings (UNRECOGNIZED_FORMS) so a PolyScope version that
    rephrases the reply doesn't make an unknown command look successful.
    """
    low = (resp or "").lower()
    return any(form in low for form in UNRECOGNIZED_FORMS)


def is_denied(resp):
    """True if the command was rejected because the pendant is in Local mode."""
    return REMOTE_DENIAL in (resp or "")


def succeeded(name, resp):
    """Generic success verdict for the raw-TCP path.

    Failure if unrecognised or denied. Actions must also match their SUCCESS
    predicate; queries succeed on any recognised, non-denied response.
    """
    if is_unrecognized(resp) or is_denied(resp):
        return False
    pred = SUCCESS.get(name)
    if pred is not None:
        return pred(resp)
    return True  # queries / commands without an explicit predicate


def parse_robot_mode(resp):
    """'Robotmode: RUNNING' -> 'RUNNING' (None if unparseable)."""
    if resp and ":" in resp:
        return resp.split(":", 1)[1].strip()
    return None


def parse_safety(resp):
    """'Safetystatus: NORMAL' / 'Safetymode: NORMAL' -> 'NORMAL'."""
    if resp and ":" in resp:
        return resp.split(":", 1)[1].strip()
    return None


def parse_program_state(resp):
    """'PLAYING external_control.urp' -> ('PLAYING', 'external_control.urp').

    Live URSim 5.25.1 appends the program name after the state; older docs show a
    bare state. Returns (state, program_or_None).
    """
    if not resp:
        return (None, None)
    parts = resp.split(None, 1)
    state = parts[0].strip() if parts else None
    program = parts[1].strip() if len(parts) > 1 else None
    return (state, program)


def parse_loaded_program(resp):
    """'Loaded program: /ursim/programs/external_control.urp' -> that path.

    Returns None for 'No program loaded' or an unparseable reply.
    """
    if resp and resp.startswith("Loaded program:"):
        return resp.split(":", 1)[1].strip()
    return None


def parse_running(resp):
    """'Program running: true' -> True / False / None."""
    if not resp:
        return None
    low = resp.lower()
    if "true" in low:
        return True
    if "false" in low:
        return False
    return None


def wire(name, arg=None):
    """Build the exact bytes-minus-newline for a canonical command name."""
    base = WIRE[name]
    if name in TAKES_ARG and arg:
        return f"{base} {arg}"
    return base
