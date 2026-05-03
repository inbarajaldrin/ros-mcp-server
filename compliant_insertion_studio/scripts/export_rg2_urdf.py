#!/usr/bin/env python3
# Reference: https://github.com/isaac-sim/IsaacSim/tree/main/source/extensions/isaacsim.asset.exporter.urdf
"""
Export the OnRobot RG2 USD asset to URDF for Phase 7 (Gripper URDF + RViz Visualization).

Drives Isaac Sim's `isaacsim.asset.exporter.urdf` extension headlessly via the
`run_headless_code()` helper from the Isaac Sim CLI harness. The exporter's
heavy lifting is done by `nvidia.srl.from_usd.to_urdf.UsdToUrdf`, which is
bundled in the extension's pip_prebundle.

Usage (drive from outside Isaac Sim):
    ~/env_isaaclab/bin/python compliant_insertion_studio/scripts/export_rg2_urdf.py

Output:
    compliant_insertion_studio/urdf/rg2/rg2.urdf      (the URDF)
    compliant_insertion_studio/urdf/rg2/meshes/...    (mesh referenced by URDF)
"""
import os
import sys
from pathlib import Path

# Make the IsaacSim CLI harness importable.
HARNESS = Path("/home/aaugus11/Documents/cli/IsaacSim/agent-harness")
if str(HARNESS) not in sys.path:
    sys.path.insert(0, str(HARNESS))

from cli_anything.isaacsim.utils.isaacsim_backend import run_headless_code  # noqa: E402

REPO = Path("/home/aaugus11/Documents/ros-mcp-server")
RG2_USD = "/home/aaugus11/Documents/isaac-sim-mcp/exts/ur5e-dt/assets/gripper/RG2.usd"
OUT_DIR = REPO / "compliant_insertion_studio" / "urdf" / "rg2"
OUT_DIR.mkdir(parents=True, exist_ok=True)

EXPORT_CODE = f'''
import json
import omni.kit.app
from isaacsim.core.utils.stage import open_stage

# Enable the URDF exporter extension (brings nvidia.srl.from_usd into the
# Python path via its pip_prebundle).
manager = omni.kit.app.get_app().get_extension_manager()
manager.set_extension_enabled_immediate("isaacsim.asset.exporter.urdf", True)

# Pump a few update ticks so the extension finishes loading.
for _ in range(10):
    app.update()

# Open the RG2 USD.
RG2_USD = r"{RG2_USD}"
open_stage(RG2_USD)
for _ in range(5):
    app.update()

import omni.usd
from pxr import Usd, UsdPhysics, Gf, Sdf

stage = omni.usd.get_context().get_stage()
default_prim = stage.GetDefaultPrim()
print("CLI_RESULT:" + json.dumps({{"default_prim": str(default_prim.GetPath()) if default_prim else None}}))

# Collect inertia data into a temp anonymous layer (mirrors the exporter UI flow).
import nvidia.srl.usd.prim_helper as prim_helper
import nvidia.srl.tools.logger as srl_logger
from nvidia.srl.from_usd.to_urdf import UsdToUrdf

inertia_temp_layer = Sdf.Layer.CreateAnonymous("inertia_temp.usda")
root_layer = stage.GetRootLayer()
root_layer.subLayerPaths.append(inertia_temp_layer.identifier)
stage.SetEditTarget(Usd.EditTarget(inertia_temp_layer))

# Default inertias for prims that have MassAPI / RigidBodyAPI but no authored
# values — UsdToUrdf needs all three (mass / com / inertia_diag) to emit a
# URDF inertial block.
inertia_prims = prim_helper.get_prims(stage, has_apis=[UsdPhysics.MassAPI, UsdPhysics.RigidBodyAPI])
print("CLI_RESULT:" + json.dumps({{"inertia_prim_count": len(inertia_prims)}}))

for prim in inertia_prims:
    mass_api = UsdPhysics.MassAPI(prim)
    if not (mass_api.GetMassAttr().IsValid() and mass_api.GetMassAttr().HasAuthoredValue()):
        mass_api.CreateMassAttr(0.05)  # 50 g placeholder per finger / link
    if not (mass_api.GetCenterOfMassAttr().IsValid() and mass_api.GetCenterOfMassAttr().HasAuthoredValue()):
        mass_api.CreateCenterOfMassAttr(Gf.Vec3f(0.0, 0.0, 0.0))
    if not (mass_api.GetDiagonalInertiaAttr().IsValid() and mass_api.GetDiagonalInertiaAttr().HasAuthoredValue()):
        mass_api.CreateDiagonalInertiaAttr(Gf.Vec3f(1e-4, 1e-4, 1e-4))
    if not (mass_api.GetPrincipalAxesAttr().IsValid() and mass_api.GetPrincipalAxesAttr().HasAuthoredValue()):
        mass_api.CreatePrincipalAxesAttr(Gf.Quatf(1.0, (0.0, 0.0, 0.0)))

OUT_DIR = r"{OUT_DIR}"
EXPORT_FILENAME = "rg2"
MESH_DIR = os.path.join(OUT_DIR, "meshes")
EXPORT_PATH = os.path.join(OUT_DIR, EXPORT_FILENAME + ".urdf")

usd_to_urdf = UsdToUrdf(
    stage,
    root=str(default_prim.GetPath()) if default_prim else None,
    log_level=srl_logger.level_from_name("ERROR"),
)

# Use file:// prefix → URDF mesh paths will be absolute file URIs into MESH_DIR.
# (We will rewrite to package-relative paths after export.)
output_path = usd_to_urdf.save_to_file(
    urdf_output_path=EXPORT_PATH,
    visualize_collision_meshes=False,
    mesh_dir=MESH_DIR + ("" if MESH_DIR.endswith("/") else "/"),
    mesh_path_prefix="file://",
    use_uri_file_prefix=True,
)
print("CLI_RESULT:" + json.dumps({{"urdf_path": output_path}}))

# Revert temp layer
root_layer.subLayerPaths.remove(inertia_temp_layer.identifier)
stage.SetEditTarget(stage.GetRootLayer())

# Sanity inspection of the output URDF.
with open(output_path, "r") as fh:
    urdf = fh.read()
link_count = urdf.count("<link ")
joint_count = urdf.count("<joint ")
print("CLI_RESULT:" + json.dumps({{"link_count": link_count, "joint_count": joint_count, "size_bytes": len(urdf)}}))
'''


def main():
    print("[export_rg2_urdf] Driving Isaac Sim headless export…")
    print(f"  source USD : {RG2_USD}")
    print(f"  output dir : {OUT_DIR}")
    print()
    result = run_headless_code(EXPORT_CODE, timeout=300)
    print("--- stdout ---")
    print(result["stdout"])
    print("--- stderr ---")
    print(result["stderr"])
    print("--- returncode ---")
    print(result["returncode"])
    print("--- parsed CLI_RESULT lines ---")
    for r in result["result"]:
        print(f"  {r}")
    if result["returncode"] != 0:
        sys.exit(result["returncode"])


if __name__ == "__main__":
    main()
