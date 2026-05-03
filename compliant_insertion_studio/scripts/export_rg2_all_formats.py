#!/usr/bin/env python3
# Reference: https://docs.isaacsim.omniverse.nvidia.com/latest/reference_python_api.html
"""
Export the OnRobot RG2 USD asset to every format Isaac Sim's asset_converter
supports, so we can choose the best one for RViz mesh fidelity.

Output: /tmp/rg2_exports/<format>/rg2.<ext> + any byproducts (MTL, textures, .bin).

Run with the env_isaaclab venv Python (the symlink — NOT its resolved target).
Documented Isaac Sim quirk: the harness symlink points to the venv where
`isaacsim` is installed; resolving the symlink escapes the venv.

Usage:
    /home/aaugus11/env_isaaclab/bin/python compliant_insertion_studio/scripts/export_rg2_all_formats.py
"""
import asyncio
import os
import sys
import json

from isaacsim import SimulationApp

app = SimulationApp({"headless": True})

import omni.kit.app  # noqa: E402
import omni.usd  # noqa: E402

manager = omni.kit.app.get_app().get_extension_manager()
manager.set_extension_enabled_immediate("omni.kit.tool.asset_exporter", True)
manager.set_extension_enabled_immediate("omni.kit.asset_converter", True)
for _ in range(20):
    app.update()

import omni.kit.asset_converter as converter  # noqa: E402
from isaacsim.core.utils.stage import open_stage  # noqa: E402

RG2_USD = "/home/aaugus11/Documents/isaac-sim-mcp/exts/ur5e-dt/assets/gripper/RG2.usd"
OUT_ROOT = "/tmp/rg2_exports"

open_stage(RG2_USD)
for _ in range(10):
    app.update()


async def export_one(ext: str) -> dict:
    out_dir = os.path.join(OUT_ROOT, ext)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"rg2.{ext}")

    ctx = converter.AssetConverterContext()
    # Turn on every "include materials / embed textures" toggle if it exists.
    for fld in (
        "ignore_materials",
        "ignore_animations",
        "ignore_camera",
        "ignore_light",
        "single_mesh",
        "smooth_normals",
        "preview_surface",
        "support_point_instancer",
        "embed_mdl_in_usd",
        "embed_textures",
        "convert_fbx_to_y_up",
        "convert_fbx_to_z_up",
        "keep_all_materials",
        "merge_all_meshes",
        "use_meter_as_world_unit",
        "create_world_as_default_root_prim",
        "export_baked_mdl",
        "export_separate_gltf",
        "export_animations",
        "export_lights",
        "export_cameras",
        "export_visible_only",
        "export_materials",
        "export_mdl_gltf_extension",
    ):
        if hasattr(ctx, fld):
            # Set "include / preserve" toggles to True.
            if fld.startswith("ignore"):
                setattr(ctx, fld, False)
            else:
                setattr(ctx, fld, True)

    task = converter.get_instance().create_converter_task(
        import_path=RG2_USD,
        output_path=out_path,
        asset_converter_context=ctx,
    )

    # Wait until done — this is a coroutine in standalone mode (no re-entrance).
    success = await task.wait_until_finished()
    listing = sorted(os.listdir(out_dir))
    return {
        "format": ext,
        "wait_success": bool(success),
        "status": str(task.get_status()),
        "error": task.get_error_message() if hasattr(task, "get_error_message") else None,
        "files": listing,
        "byproducts": [f for f in listing if not f.endswith(f".{ext}")],
    }


async def main():
    results = []
    for ext in ("obj", "gltf", "glb", "fbx", "stl", "usdz"):
        try:
            r = await export_one(ext)
        except Exception as e:
            import traceback
            r = {"format": ext, "exception": str(e), "trace": traceback.format_exc()[:1000]}
        results.append(r)
        # Pump update between exports
        for _ in range(5):
            app.update()
    return results


loop = asyncio.get_event_loop()
all_results = loop.run_until_complete(main())

print()
print("=" * 60)
print("EXPORT RESULTS")
print("=" * 60)
print(json.dumps(all_results, indent=2))

app.close()
