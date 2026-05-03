#!/usr/bin/env python3
# Reference: https://github.com/isaac-sim/IsaacSim/tree/main/source/extensions/isaacsim.asset.exporter.urdf
"""
Standalone Isaac Sim USD→URDF exporter for the OnRobot RG2.

Run this with the env_isaaclab venv Python (the symlink, NOT its resolved
target — the harness in cli_anything/isaacsim resolves the symlink which
breaks `import isaacsim`).

Usage:
    ~/env_isaaclab/bin/python compliant_insertion_studio/scripts/export_rg2_urdf_standalone.py
"""
import json
import os

# Boot SimulationApp before any omni / isaacsim imports.
from isaacsim import SimulationApp

app = SimulationApp({"headless": True})

import omni.kit.app
from isaacsim.core.utils.stage import open_stage  # noqa: E402

# Enable the URDF exporter extension; this brings nvidia.srl.from_usd into the
# Python path via the extension's pip_prebundle.
manager = omni.kit.app.get_app().get_extension_manager()
manager.set_extension_enabled_immediate("isaacsim.asset.exporter.urdf", True)
for _ in range(20):
    app.update()

import omni.usd  # noqa: E402
from pxr import Usd, UsdPhysics, Gf, Sdf  # noqa: E402

RG2_USD = "/home/aaugus11/Documents/isaac-sim-mcp/exts/ur5e-dt/assets/gripper/RG2.usd"
OUT_DIR = "/home/aaugus11/Documents/ros-mcp-server/compliant_insertion_studio/urdf/rg2"
os.makedirs(OUT_DIR, exist_ok=True)
MESH_DIR = os.path.join(OUT_DIR, "meshes")
EXPORT_PATH = os.path.join(OUT_DIR, "rg2.urdf")

open_stage(RG2_USD)
for _ in range(10):
    app.update()

stage = omni.usd.get_context().get_stage()
default_prim = stage.GetDefaultPrim()
print("CLI_RESULT:" + json.dumps({"default_prim": str(default_prim.GetPath()) if default_prim else None}))

# Imports that depend on the urdf-exporter extension being enabled.
import nvidia.srl.usd.prim_helper as prim_helper  # noqa: E402
import nvidia.srl.tools.logger as srl_logger  # noqa: E402
from nvidia.srl.from_usd.to_urdf import UsdToUrdf  # noqa: E402

# Anonymous overlay layer to author missing inertia data without polluting the
# source RG2.usd.
inertia_layer = Sdf.Layer.CreateAnonymous("inertia_temp.usda")
root_layer = stage.GetRootLayer()
root_layer.subLayerPaths.append(inertia_layer.identifier)
stage.SetEditTarget(Usd.EditTarget(inertia_layer))

inertia_prims = prim_helper.get_prims(stage, has_apis=[UsdPhysics.MassAPI, UsdPhysics.RigidBodyAPI])
print("CLI_RESULT:" + json.dumps({"inertia_prim_count": len(inertia_prims)}))

for prim in inertia_prims:
    mass_api = UsdPhysics.MassAPI(prim)
    if not (mass_api.GetMassAttr().IsValid() and mass_api.GetMassAttr().HasAuthoredValue()):
        mass_api.CreateMassAttr(0.05)
    if not (mass_api.GetCenterOfMassAttr().IsValid() and mass_api.GetCenterOfMassAttr().HasAuthoredValue()):
        mass_api.CreateCenterOfMassAttr(Gf.Vec3f(0.0, 0.0, 0.0))
    if not (mass_api.GetDiagonalInertiaAttr().IsValid() and mass_api.GetDiagonalInertiaAttr().HasAuthoredValue()):
        mass_api.CreateDiagonalInertiaAttr(Gf.Vec3f(1e-4, 1e-4, 1e-4))
    if not (mass_api.GetPrincipalAxesAttr().IsValid() and mass_api.GetPrincipalAxesAttr().HasAuthoredValue()):
        mass_api.CreatePrincipalAxesAttr(Gf.Quatf(1.0, (0.0, 0.0, 0.0)))

usd_to_urdf = UsdToUrdf(
    stage,
    root=str(default_prim.GetPath()) if default_prim else None,
    log_level=srl_logger.level_from_name("ERROR"),
)
output_path = usd_to_urdf.save_to_file(
    urdf_output_path=EXPORT_PATH,
    visualize_collision_meshes=False,
    mesh_dir=MESH_DIR + "/",
    mesh_path_prefix="file://",
    use_uri_file_prefix=True,
)
print("CLI_RESULT:" + json.dumps({"urdf_path": output_path}))

root_layer.subLayerPaths.remove(inertia_layer.identifier)
stage.SetEditTarget(stage.GetRootLayer())

with open(output_path, "r") as fh:
    urdf = fh.read()
link_count = urdf.count("<link ")
joint_count = urdf.count("<joint ")
print("CLI_RESULT:" + json.dumps({
    "link_count": link_count,
    "joint_count": joint_count,
    "size_bytes": len(urdf),
}))

app.close()
