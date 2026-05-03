#!/usr/bin/env python3
"""
Convert the Isaac Sim asset_converter's OBJ output to DAE using pyassimp,
which calls libassimp directly (no apt install needed since libassimp5 is
already on the system).

Pipeline:
1. Read /tmp/rg2_exports/obj/rg2.obj (multiple `g mesh` groups, each with
   `usemtl material_<name>`).
2. Patch the OBJ's MTL with USD-canonical diffuse_color_constant values
   (the asset_converter ships `Kd 1 1 1` for everything — wrong).
3. pyassimp.load() with smoothing post-processing → smoothed vertex normals.
4. pyassimp.export() to DAE.
5. Update the URDF to reference rg2.dae.

Run with system Python 3:
    pip install --user pyassimp pycollada    # already installed
    python3 compliant_insertion_studio/scripts/build_rg2_dae_via_assimp.py
"""
import re
import shutil
import sys
from pathlib import Path
from xml.etree import ElementTree as ET

import pyassimp
import pyassimp.postprocess as pp
from pxr import Usd, UsdShade

REPO = Path("/home/aaugus11/Documents/ros-mcp-server")
RG2_USD = "/home/aaugus11/Documents/isaac-sim-mcp/exts/ur5e-dt/assets/gripper/RG2.usd"
SRC_OBJ = Path("/tmp/rg2_exports/obj/rg2.obj")
SRC_MTL = Path("/tmp/rg2_exports/obj/rg2.mtl")
WORK_DIR = Path("/tmp/rg2_assimp_work")
WORK_OBJ = WORK_DIR / "rg2.obj"
WORK_MTL = WORK_DIR / "rg2.mtl"
OUT_DIR = REPO / "compliant_insertion_studio" / "urdf" / "rg2"
OUT_DAE = OUT_DIR / "rg2.dae"
OUT_URDF = OUT_DIR / "rg2.urdf"


def read_usd_diffuses(usd_path: str) -> dict[str, tuple[float, float, float]]:
    stage = Usd.Stage.Open(usd_path)
    out: dict[str, tuple[float, float, float]] = {}
    for prim in stage.Traverse():
        if not prim.IsA(UsdShade.Material):
            continue
        name = prim.GetName()
        for child in prim.GetChildren():
            if not child.IsA(UsdShade.Shader):
                continue
            sh = UsdShade.Shader(child)
            for inp_name in (
                "diffuse_color_constant", "diffuseColor", "diffuse_color",
                "baseColor", "albedo",
            ):
                inp = sh.GetInput(inp_name)
                if inp and inp.Get() is not None:
                    c = inp.Get()
                    out[name] = (float(c[0]), float(c[1]), float(c[2]))
                    break
            if name in out:
                break
    return out


def patch_mtl(src: Path, dst: Path, colors: dict[str, tuple[float, float, float]]) -> None:
    text = src.read_text()
    # Split into header + per-material blocks
    blocks = re.split(r"(?m)^newmtl\s+", text)
    out_lines = [blocks[0].rstrip()]
    for blk in blocks[1:]:
        nl = blk.find("\n")
        name = blk[:nl].strip()
        body = re.sub(r"(?m)^(Kd|Ka|Ks|Ke|Ns|d|Tr|illum)\b.*$\n?", "", blk[nl + 1:]).strip()
        rgb = colors.get(name, (0.5, 0.5, 0.5))
        r, g, b = rgb
        out_lines.append(f"newmtl {name}")
        out_lines.append(f"Kd {r:.6f} {g:.6f} {b:.6f}")
        out_lines.append(f"Ka {r * 0.25:.6f} {g * 0.25:.6f} {b * 0.25:.6f}")
        out_lines.append("Ks 0.05 0.05 0.05")
        out_lines.append("Ns 10")
        out_lines.append("illum 2")
        if body:
            out_lines.append(body)
        out_lines.append("")
    dst.write_text("\n".join(out_lines).rstrip() + "\n")


def build_urdf(dae_filename: str, urdf_path: Path) -> None:
    robot = ET.Element("robot", {"name": "rg2"})
    link = ET.SubElement(robot, "link", {"name": "rg2_base_link"})

    inertial = ET.SubElement(link, "inertial")
    ET.SubElement(inertial, "origin", {"xyz": "0 0 0.1", "rpy": "0 0 0"})
    ET.SubElement(inertial, "mass", {"value": "0.78"})
    ET.SubElement(
        inertial, "inertia",
        {"ixx": "1e-3", "ixy": "0", "ixz": "0", "iyy": "1e-3", "iyz": "0", "izz": "1e-3"},
    )

    visual = ET.SubElement(link, "visual")
    ET.SubElement(visual, "origin", {"xyz": "0 0 0", "rpy": "0 0 0"})
    g = ET.SubElement(visual, "geometry")
    ET.SubElement(
        g, "mesh",
        {"filename": f"package://compliant_insertion_studio/urdf/rg2/{dae_filename}",
         "scale": "1 1 1"},
    )

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
        ET.SubElement(c, "origin", {
            "xyz": f"{xyz_c[0]:.6f} {xyz_c[1]:.6f} {xyz_c[2]:.6f}",
            "rpy": f"{rpy_c[0]:.6f} {rpy_c[1]:.6f} {rpy_c[2]:.6f}",
        })
        gc = ET.SubElement(c, "geometry")
        ET.SubElement(gc, "mesh", {
            "filename": f"package://compliant_insertion_studio/urdf/rg2/meshes/{fname}",
            "scale": "1 1 1",
        })

    ET.indent(robot, space="  ")
    ET.ElementTree(robot).write(urdf_path, encoding="utf-8", xml_declaration=True)


def main() -> None:
    if not SRC_OBJ.is_file():
        print(f"ERROR: {SRC_OBJ} not found. Run export_rg2_all_formats.py first.", file=sys.stderr)
        sys.exit(1)

    WORK_DIR.mkdir(parents=True, exist_ok=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Copy OBJ + patch MTL.
    shutil.copy(SRC_OBJ, WORK_OBJ)
    colors = read_usd_diffuses(RG2_USD)
    print(f"USD-resolved material colors:")
    for k, v in colors.items():
        print(f"  {k}: {v}")
    patch_mtl(SRC_MTL, WORK_MTL, colors)
    print(f"\nPatched MTL: {WORK_MTL}")

    # 2. Load OBJ via pyassimp with smoothing + normal-generation flags.
    flags = (
        pp.aiProcess_Triangulate
        | pp.aiProcess_GenSmoothNormals     # ← fixes the faceting GPT noted
        | pp.aiProcess_JoinIdenticalVertices
        | pp.aiProcess_ImproveCacheLocality
        | pp.aiProcess_FixInfacingNormals
    )
    print(f"\nLoading OBJ via pyassimp (with smoothing + normal generation)...")
    with pyassimp.load(str(WORK_OBJ), processing=flags) as scene:
        print(f"  meshes: {len(scene.meshes)}")
        print(f"  materials: {len(scene.materials)}")
        for i, m in enumerate(scene.materials):
            try:
                props = m.properties if isinstance(m.properties, dict) else {}
                name = props.get("name", f"<unnamed_{i}>")
                kd = props.get("diffuse", "?")
                print(f"    [{i}] name={name}  diffuse={kd}")
            except Exception as e:
                print(f"    [{i}] (could not introspect: {e})")

        # 3. Export to DAE
        print(f"\nExporting to DAE...")
        pyassimp.export(scene, str(OUT_DAE), file_type="collada")
        print(f"  → {OUT_DAE} ({OUT_DAE.stat().st_size:,} bytes)")

    # 4. Build URDF
    build_urdf(OUT_DAE.name, OUT_URDF)
    print(f"Wrote URDF: {OUT_URDF.relative_to(REPO)}")


if __name__ == "__main__":
    main()
