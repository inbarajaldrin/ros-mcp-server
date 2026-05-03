#!/usr/bin/env python3
# Reference: https://docs.isaacsim.omniverse.nvidia.com/latest/reference_python_api.html
"""
Build the canonical OBJ+MTL pipeline for the OnRobot RG2 mesh in our URDF.

This combines two facts discovered during Phase 7:

1. Isaac Sim's `omni.kit.tool.asset_exporter` converts the RG2 USD to a SINGLE
   OBJ file with each per-link mesh as a `g mesh` group, and each group is
   correctly tagged via `usemtl material_<name>`. This is exactly the layout
   RViz/Assimp wants for per-mesh-group coloring inside one URDF <visual>.

2. The exporter HOWEVER drops the USD's `diffuse_color_constant` shader input
   during conversion — every newmtl in the produced .mtl gets `Kd 1 1 1`
   (white). So the geometry + structure is right, but the color is missing.

The fix: copy the asset_converter's OBJ as-is, then re-author the MTL using
the actual diffuse_color_constant values read from the source USD's
UsdShade.Material shader inputs.

The resulting compliant_insertion_studio/urdf/rg2/rg2.urdf is a single-link,
single-visual URDF. The single visual references the single OBJ; the OBJ
references the MTL; the MTL has correct per-material colors. RViz's Assimp
loader picks all this up natively (this is the same pipeline the UR5e
description package uses with DAE).

Pre-requisite (run once before this script):
    /home/aaugus11/env_isaaclab/bin/python compliant_insertion_studio/scripts/export_rg2_all_formats.py

That produces /tmp/rg2_exports/obj/{rg2.obj, rg2.mtl}.

Usage:
    python3 compliant_insertion_studio/scripts/build_rg2_obj_pipeline.py
"""
import math
import re
import shutil
import sys
from pathlib import Path
from xml.etree import ElementTree as ET

from pxr import Usd, UsdShade, Gf, UsdGeom

REPO = Path("/home/aaugus11/Documents/ros-mcp-server")
RG2_USD = "/home/aaugus11/Documents/isaac-sim-mcp/exts/ur5e-dt/assets/gripper/RG2.usd"
SRC_OBJ = Path("/tmp/rg2_exports/obj/rg2.obj")
SRC_MTL = Path("/tmp/rg2_exports/obj/rg2.mtl")
OUT_DIR = REPO / "compliant_insertion_studio" / "urdf" / "rg2"
OUT_OBJ = OUT_DIR / "rg2.obj"
OUT_MTL = OUT_DIR / "rg2.mtl"
OUT_URDF = OUT_DIR / "rg2.urdf"


def read_usd_material_colors(usd_path: str) -> dict[str, tuple[float, float, float]]:
    """Read every UsdShade.Material under the USD and extract its
    diffuse_color_constant. Returns a {material_name: (r, g, b)} dict.
    """
    stage = Usd.Stage.Open(usd_path)
    out: dict[str, tuple[float, float, float]] = {}
    for prim in stage.Traverse():
        if not prim.IsA(UsdShade.Material):
            continue
        # Material name in MTL = material's last path component (e.g.
        # "/khi_rs080n/Looks/material_CCCCCC" -> "material_CCCCCC")
        name = prim.GetName()
        for child in prim.GetChildren():
            if not child.IsA(UsdShade.Shader):
                continue
            shader = UsdShade.Shader(child)
            for input_name in (
                "diffuse_color_constant",
                "diffuseColor",
                "diffuse_color",
                "baseColor",
                "albedo",
            ):
                inp = shader.GetInput(input_name)
                if inp and inp.Get() is not None:
                    c = inp.Get()
                    out[name] = (float(c[0]), float(c[1]), float(c[2]))
                    break
            if name in out:
                break
    return out


def patch_mtl(src_mtl: Path, dst_mtl: Path, colors: dict[str, tuple[float, float, float]]) -> dict[str, dict]:
    """Read the asset_converter-emitted MTL and rewrite each material's Kd
    from the USD-resolved colors. Returns a per-material report.
    """
    text = src_mtl.read_text()
    blocks = re.split(r"(?m)^newmtl\s+", text)
    header = blocks[0]
    out_lines = [header.rstrip()]
    report: dict[str, dict] = {}
    for blk in blocks[1:]:
        first_nl = blk.find("\n")
        mat_name = blk[:first_nl].strip()
        body = blk[first_nl + 1:]
        # Strip any existing Kd/Ka/Ks/illum lines so we author cleanly
        body = re.sub(r"(?m)^(Kd|Ka|Ks|Ke|Ns|d|Tr|illum)\b.*$\n?", "", body).strip()
        rgba = colors.get(mat_name)
        if rgba is None:
            report[mat_name] = {"warning": "no USD diffuse_color_constant; defaulting to grey 0.5"}
            r, g, b = 0.5, 0.5, 0.5
        else:
            r, g, b = rgba
            report[mat_name] = {"diffuse": rgba, "source": "usd"}
        # Author Kd + a small Ka so RViz lighting has something to add
        ka_r, ka_g, ka_b = r * 0.25, g * 0.25, b * 0.25
        ks = 0.05  # small specular so highlights aren't completely flat
        out_lines.append(f"newmtl {mat_name}")
        out_lines.append(f"Kd {r:.6f} {g:.6f} {b:.6f}")
        out_lines.append(f"Ka {ka_r:.6f} {ka_g:.6f} {ka_b:.6f}")
        out_lines.append(f"Ks {ks:.6f} {ks:.6f} {ks:.6f}")
        out_lines.append("Ns 10")
        out_lines.append("illum 2")
        if body:
            out_lines.append(body)
        out_lines.append("")
    dst_mtl.write_text("\n".join(out_lines).rstrip() + "\n")
    return report


def determine_obj_root_xform(obj_path: Path) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    """Looking at our previous extract_rg2_urdf.py findings: in the source USD,
    every link prim has the same `-90° about Z` (or `+90°` for right-side links)
    rotation relative to /khi_rs080n. The asset_converter's OBJ flattens every
    transform into world coordinates, so the points should already be in the
    /khi_rs080n root frame. The URDF visual <origin> can be identity.
    """
    return (0.0, 0.0, 0.0), (0.0, 0.0, 0.0)


def build_urdf(obj_filename: str, urdf_path: Path) -> None:
    """Single-link, single-visual URDF that references the OBJ. The OBJ
    references the MTL (mtllib rg2.mtl) which RViz/Assimp loads automatically.
    """
    robot = ET.Element("robot", {"name": "rg2"})
    link = ET.SubElement(robot, "link", {"name": "rg2_base_link"})

    inertial = ET.SubElement(link, "inertial")
    ET.SubElement(inertial, "origin", {"xyz": "0 0 0.1", "rpy": "0 0 0"})
    ET.SubElement(inertial, "mass", {"value": "0.78"})  # OnRobot RG2 ≈ 780 g
    ET.SubElement(
        inertial,
        "inertia",
        {"ixx": "1e-3", "ixy": "0", "ixz": "0", "iyy": "1e-3", "iyz": "0", "izz": "1e-3"},
    )

    xyz, rpy = determine_obj_root_xform(OUT_OBJ)

    visual = ET.SubElement(link, "visual")
    ET.SubElement(
        visual,
        "origin",
        {"xyz": f"{xyz[0]:.6f} {xyz[1]:.6f} {xyz[2]:.6f}",
         "rpy": f"{rpy[0]:.6f} {rpy[1]:.6f} {rpy[2]:.6f}"},
    )
    g = ET.SubElement(visual, "geometry")
    ET.SubElement(
        g,
        "mesh",
        {"filename": f"package://compliant_insertion_studio/urdf/rg2/{obj_filename}",
         "scale": "1 1 1"},
    )
    # No <material> block — colors come from the OBJ's MTL.

    # Collision: keep the high-fidelity convex per-part STL meshes from the
    # previous pipeline (they're known good).
    for fname, xyz_c, rpy_c in (
        ("base_collision.stl",        (0.000000, -0.000000, 0.000000), (0.0, -0.0, -1.570797)),
        ("lo_knuckle_collision.stl",  (-0.017180, -0.000000, 0.125800), (0.0, -0.0, -1.570797)),
        ("li_knuckle_collision.stl",  (-0.007680, -0.000000, 0.142300), (0.0, -0.0, -1.570797)),
        ("l_finger_collision.stl",    (-0.056770,  0.000000, 0.163970), (0.0, -0.0, -1.570797)),
        ("ro_knuckle_collision.stl",  ( 0.017180, -0.000000, 0.125800), (0.0,  0.0,  1.570797)),
        ("ri_knuckle_collision.stl",  ( 0.007680, -0.000000, 0.142300), (0.0,  0.0,  1.570796)),
        ("r_finger_collision.stl",    ( 0.056770,  0.000000, 0.163970), (0.0,  0.0,  1.570797)),
    ):
        c = ET.SubElement(link, "collision")
        ET.SubElement(
            c, "origin",
            {"xyz": f"{xyz_c[0]:.6f} {xyz_c[1]:.6f} {xyz_c[2]:.6f}",
             "rpy": f"{rpy_c[0]:.6f} {rpy_c[1]:.6f} {rpy_c[2]:.6f}"}
        )
        gc = ET.SubElement(c, "geometry")
        ET.SubElement(
            gc, "mesh",
            {"filename": f"package://compliant_insertion_studio/urdf/rg2/meshes/{fname}",
             "scale": "1 1 1"}
        )

    ET.indent(robot, space="  ")
    ET.ElementTree(robot).write(urdf_path, encoding="utf-8", xml_declaration=True)


def main() -> None:
    if not SRC_OBJ.is_file():
        print(f"ERROR: {SRC_OBJ} not found. Run export_rg2_all_formats.py first.", file=sys.stderr)
        sys.exit(1)

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Copy the OBJ verbatim (its mtllib reference is `rg2.mtl`, in the same dir).
    shutil.copy(SRC_OBJ, OUT_OBJ)
    print(f"Copied OBJ: {OUT_OBJ.relative_to(REPO)} ({OUT_OBJ.stat().st_size:,} bytes)")

    # 2. Read USD material colors and patch the MTL.
    colors = read_usd_material_colors(RG2_USD)
    print(f"USD-resolved material colors:")
    for name, c in colors.items():
        print(f"  {name}: {c}")
    report = patch_mtl(SRC_MTL, OUT_MTL, colors)
    print(f"\nWrote MTL: {OUT_MTL.relative_to(REPO)}")
    for mat, info in report.items():
        print(f"  {mat}: {info}")

    # 3. Write the URDF.
    build_urdf(OUT_OBJ.name, OUT_URDF)
    print(f"\nWrote URDF: {OUT_URDF.relative_to(REPO)}")


if __name__ == "__main__":
    main()
