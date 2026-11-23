# Multi-machine ROS 2: hardware-near nodes at the robot, compute on a GPU box

Working notes, to be cleaned up into a DroidStack setup method (plan47 asked where it belongs, 2026-11-23).
The same shape serves this UR5e cell, RoboSort's JETANK, or a Raspberry Pi / Jetson at any robot.
Status: **design notes only. Nothing here is deployed on the cell yet.**

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

## Open items before it carries a robot run

1. **Clock sync** (chrony on both machines). The "read only after the arm is still ≥ 0.25 s" rule compares camera stamps with arm-state stamps from the other machine.
2. **Link-drop behaviour.** A lost peer mid-run must end in a clean stop: the arm side never waits forever for a plan, and the dashboard stop stays local.
3. **Bring-up and teardown on two machines** (`stack_up.sh` / `stack_down.sh` / `stop_run.sh`). The seams servers do not inherit env, so the DDS env goes in their unit files.
4. **First test without arm motion:** the localizer on a4500 against the cell's cameras. Measure its rate, its latency, the bandwidth, and the pose agreement with the local localizer.
