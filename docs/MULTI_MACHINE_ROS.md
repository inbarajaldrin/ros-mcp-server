# Multi-machine ROS 2: hardware-near nodes at the robot, compute on a GPU box

Working notes, to be cleaned up into a DroidStack setup method.
The same shape serves this UR5e cell, RoboSort's JETANK, or a Raspberry Pi / Jetson at any robot.
Status: **design notes only. Nothing here is deployed on the cell yet.** Each fact carries the command that measured it and the time.

## Where it goes in DroidStack (plan47, 2026-11-23, platform dev b2246d6f)

- **The real deployment of a cell, never the fleet.** The fleet (sim boxes) runs localhost-only by construction: domain 0,
  `ROS_LOCALHOST_ONLY=1`, discovery range LOCALHOST. That isolation is a safety control on a lab network with live robots.
- It has two halves:
  - **Per-machine transport fact:** the interface, peers, multicast off and the domain. It sits beside the machine's ROS road record
    (`~/.droidstack/machine.yaml` `deployment.ros_setup` / `deployment.ros_rmw`). The RMW is always recorded, never implicit.
  - **Per-cell node placement:** which node runs on which machine, and why. It belongs to the cell's REAL binding (a separately
    qualified real binding, `docs/bring-your-own-robot.md`). No real-hardware road is built yet.
- **Later:** one DroidStack doc, proposed as `docs/real-deployment-multi-machine.md`, as a plan47 lane once the operator says go. This file
  is its source.
- **Reuse:**
  - `ros_transport.with_participant_capacity`: it appends a MaxAutoParticipantIndex fragment to any `CYCLONEDDS_URI`.
  - the domain bounds 0..232 and the port arithmetic (`dispatch/constants.py`).
  - the recorded-RMW rule (`ros_road.py`).
- **Does not exist in DroidStack:** Tailscale host facts, a peer list, a domain registry.

## Domain rule

- **Use a NON-ZERO domain for the real cell, unique per cell.** Sim attempts on dual/a4500 are localhost-only on domain 0. A hand-run
  graph on domain 0 without the localhost range could otherwise see the cell.
- **No sim graph on any box may use the cell's domain.**
- **Participant capacity:** with multicast off, Cyclone finds unicast peers by participant index. Its default admits only 9
  processes per domain on one machine, DroidStack declares 32, and the grip-force live run counted 13 on one cell. Set
  MaxAutoParticipantIndex ≥ 32.
- **`use_sim_time` false** on the real cell.

## The split

| stays on the machine at the robot | moves to the GPU box |
|---|---|
| UR driver (500 Hz RTDE loop; the robot connects back to this machine's IP) | ArUco localizer (~3 Hz on the cell PC, CPU-bound) |
| gripper bridge (RG2 over the UR tool serial) | grasp-point publisher, seams servers |
| USB cameras (D415 on USB 3, D455) | the board program's planning (16.5 of the 19.6 min before the first pick on 11-23) |
| force-mode insert (tight loop on the wrench) | rendering / pose fitting |
| `stop_run.sh` (the dashboard `stop` must not depend on the link) | |

Rule: anything with a hard real-time loop or a physical connection stays local. Only services, poses and compressed images cross the network.

## Transport (proven in RoboSort, `github.com/inbarajaldrin/RoboSort`, `dds_config/`)

- `RMW_IMPLEMENTATION=rmw_cyclonedds_cpp` on every machine. The cell already runs Cyclone: it is the only RMW installed, but it has no XML.
- The same `ROS_DOMAIN_ID` on every machine. `ROS_LOCALHOST_ONLY` must also be identical everywhere: a mixed 0/1 setup gives one-way discovery.
- `CYCLONEDDS_URI=file:///…/cyclonedds.xml`, pinned to the Tailscale interface, multicast off, every machine listed as a peer:

```xml
<CycloneDDS xmlns="https://cdds.io/config">
  <Domain id="<same on all machines>">
    <General>
      <NetworkInterfaceAddress>tailscale0</NetworkInterfaceAddress>
      <AllowMulticast>false</AllowMulticast>
      <EnableMulticastLoopback>false</EnableMulticastLoopback>
    </General>
    <Discovery>
      <Peers>
        <Peer address="<robot-side machine tailscale IP>"/>
        <Peer address="<gpu box tailscale IP>"/>
      </Peers>
      <ParticipantIndex>auto</ParticipantIndex>
      <MaxAutoParticipantIndex>250</MaxAutoParticipantIndex>
    </Discovery>
  </Domain>
</CycloneDDS>
```

- After any change: `ros2 daemon stop && ros2 daemon start` on every machine.

## Measured (2026-11-23)

- Tailscale cell PC ↔ a4500 and ↔ dual-a4500: a direct path, 6 ms round trip (`tailscale ping`).
- Not yet measured: bandwidth with depth streaming across. Raw D415 depth at 1280×720×15 Hz is ~28 MB/s, so send compressed images or only the crops the checks need.

## Still to measure (each with its command and time)

| what | how |
|---|---|
| clock offset between the machines | chrony on both (`chronyc tracking`) |
| per-topic rate and bandwidth across the link, depth and MJPEG included | `ros2 topic hz` / `ros2 topic bw` on the FAR side |
| participant count on each machine | `ros2 node list` / Cyclone participant count |
| round-trip latency | `tailscale ping` (6 ms, above) |

## Open items before it carries a robot run

1. **THE STOP PATH.**
   - It lives on the cell PC side: the UR driver's own stale-command behaviour, or a watchdog there. Never on the GPU box. The arm
     side never waits forever for a plan, and the dashboard stop stays local.
   - It is proven by actually dropping the link (`tailscale down`) mid-motion and recording what the arm and gripper did. That test
     moves the robot, so it needs the operator's go.
2. **Bring-up and teardown on two machines** (`stack_up.sh` / `stack_down.sh` / `stop_run.sh`). The seams servers do not inherit
   env, so the DDS env goes in their unit files.
3. **First test without arm motion:** run the localizer on a4500 against the cell's cameras. Measure its rate, its latency, the
   bandwidth, and how well its poses agree with the local localizer's.
