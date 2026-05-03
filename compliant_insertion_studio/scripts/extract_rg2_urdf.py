#!/usr/bin/env python3
"""
Hand-extract the RG2 USD into a single-link URDF (fixed-joint everywhere).

The Isaac Sim USD→URDF exporter (`isaacsim.asset.exporter.urdf`, which uses
`nvidia.srl.from_usd.to_urdf.UsdToUrdf` under the hood) can't process the
RG2.usd because the OnRobot RG2's 4-bar parallel-finger linkage has kinematic
loops. Phase 7 only needs visualization (D-7 in 07-CONTEXT.md: "Fixed-joint
everywhere — visual blob"), so we sidestep the loop problem entirely:

  - One URDF link `rg2_base_link`, no joints.
  - All RG2 sub-meshes (base + knuckles + fingers) are dropped into that
    one link as separate <visual> + <collision> blocks, each at its
    USD-authored local pose.

Output:
  compliant_insertion_studio/urdf/rg2/rg2.urdf
  compliant_insertion_studio/urdf/rg2/meshes/<linkname>.obj   (one per Mesh)

This script uses only `pxr` (system Python; no Isaac Sim required).
"""
import os
import sys
from pathlib import Path
from xml.etree import ElementTree as ET

from pxr import Usd, UsdGeom, UsdShade, Gf

RG2_USD = "/home/aaugus11/Documents/isaac-sim-mcp/exts/ur5e-dt/assets/gripper/RG2.usd"
OUT_ROOT = Path("/home/aaugus11/Documents/ros-mcp-server/compliant_insertion_studio/urdf/rg2")
MESH_DIR = OUT_ROOT / "meshes"
URDF_PATH = OUT_ROOT / "rg2.urdf"

# Each entry: (link_subprim_under_default_prim, output_mesh_basename).
# Material colors are read live from the USD's material bindings.
LINKS = [
    ("onrobot_rg2_base_link", "base"),
    ("left_outer_knuckle",    "lo_knuckle"),
    ("left_inner_knuckle",    "li_knuckle"),
    ("left_inner_finger",     "l_finger"),
    ("right_outer_knuckle",   "ro_knuckle"),
    ("right_inner_knuckle",   "ri_knuckle"),
    ("right_inner_finger",    "r_finger"),
]


def read_diffuse_rgba(stage, mesh_prim, fallback=(0.5, 0.5, 0.5, 1.0)):
    """Resolve material:binding on a Mesh prim, then read the bound
    UsdShade.Material's child Shader's diffuse_color_constant input.
    Falls back if anything is missing.
    """
    rel = mesh_prim.GetRelationship("material:binding")
    if not rel:
        return fallback
    targets = rel.GetTargets()
    if not targets:
        return fallback
    mat_prim = stage.GetPrimAtPath(targets[0])
    if not mat_prim or not mat_prim.IsValid() or not mat_prim.IsA(UsdShade.Material):
        return fallback
    for child in mat_prim.GetChildren():
        if child.IsA(UsdShade.Shader):
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
                    return (float(c[0]), float(c[1]), float(c[2]), 1.0)
    return fallback


def extract_mesh_to_stl(mesh_prim, out_path: Path, transform_local_to_link: Gf.Matrix4d):
    """Read a UsdGeom.Mesh and write a binary STL file. STL contains only
    triangles (no material data) so RViz uses the URDF <material><color>
    instead of the mesh's intrinsic material — this is what we want.
    """
    import struct
    mesh = UsdGeom.Mesh(mesh_prim)
    points = mesh.GetPointsAttr().Get()
    face_vertex_counts = mesh.GetFaceVertexCountsAttr().Get()
    face_vertex_indices = mesh.GetFaceVertexIndicesAttr().Get()
    if points is None or face_vertex_counts is None or face_vertex_indices is None:
        return False, 0, 0

    # Transform points into the link's frame.
    transformed = []
    for p in points:
        v = transform_local_to_link.Transform(Gf.Vec3d(p[0], p[1], p[2]))
        transformed.append((float(v[0]), float(v[1]), float(v[2])))

    # Build triangles via fan triangulation of each face.
    triangles = []
    idx = 0
    for fvc in face_vertex_counts:
        verts_idx = list(face_vertex_indices[idx : idx + fvc])
        for i in range(1, fvc - 1):
            v0 = transformed[verts_idx[0]]
            v1 = transformed[verts_idx[i]]
            v2 = transformed[verts_idx[i + 1]]
            # Compute face normal (RViz/Assimp tolerates zero normals but a
            # real one renders better).
            ax = v1[0] - v0[0]; ay = v1[1] - v0[1]; az = v1[2] - v0[2]
            bx = v2[0] - v0[0]; by = v2[1] - v0[1]; bz = v2[2] - v0[2]
            nx = ay * bz - az * by
            ny = az * bx - ax * bz
            nz = ax * by - ay * bx
            n_len = (nx * nx + ny * ny + nz * nz) ** 0.5
            if n_len > 1e-12:
                nx /= n_len; ny /= n_len; nz /= n_len
            triangles.append((nx, ny, nz, v0, v1, v2))
        idx += fvc

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("wb") as fh:
        header = b"Extracted from " + str(mesh_prim.GetPath()).encode()
        header = header[:80].ljust(80, b" ")
        fh.write(header)
        fh.write(struct.pack("<I", len(triangles)))
        for nx, ny, nz, v0, v1, v2 in triangles:
            fh.write(struct.pack("<fff", nx, ny, nz))
            fh.write(struct.pack("<fff", v0[0], v0[1], v0[2]))
            fh.write(struct.pack("<fff", v1[0], v1[1], v1[2]))
            fh.write(struct.pack("<fff", v2[0], v2[1], v2[2]))
            fh.write(struct.pack("<H", 0))  # attribute byte count
    return True, len(transformed), len(triangles)


def quat_to_rpy(q_real: float, q_i: float, q_j: float, q_k: float) -> tuple[float, float, float]:
    """Convert quaternion (re, i, j, k) to roll/pitch/yaw (xyz Euler, radians).
    Standard formula from Wikipedia.
    """
    import math

    w, x, y, z = q_real, q_i, q_j, q_k
    # roll (x-axis)
    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    roll = math.atan2(sinr_cosp, cosr_cosp)
    # pitch (y-axis)
    sinp = 2.0 * (w * y - z * x)
    if abs(sinp) >= 1.0:
        pitch = math.copysign(math.pi / 2, sinp)
    else:
        pitch = math.asin(sinp)
    # yaw (z-axis)
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    yaw = math.atan2(siny_cosp, cosy_cosp)
    return roll, pitch, yaw


def main():
    stage = Usd.Stage.Open(RG2_USD)
    if stage is None:
        print(f"Could not open {RG2_USD}", file=sys.stderr)
        sys.exit(1)

    default_prim = stage.GetDefaultPrim()
    print(f"defaultPrim: {default_prim.GetPath()}")

    # `/khi_rs080n` itself is the default prim (root), and each LINKS entry is
    # a child Xform under it. Each entry's local transform is what we want as
    # the <visual><origin/> in the URDF link.
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    MESH_DIR.mkdir(parents=True, exist_ok=True)

    visual_blocks = []
    collision_blocks = []

    for link_name, mesh_basename in LINKS:
        link_prim_path = f"/{default_prim.GetName()}/{link_name}"
        link_prim = stage.GetPrimAtPath(link_prim_path)
        if not link_prim or not link_prim.IsValid():
            print(f"  SKIP: {link_prim_path} not found")
            continue

        link_xform = UsdGeom.Xformable(link_prim)
        link_local_m = link_xform.GetLocalTransformation()
        link_xlate = link_local_m.ExtractTranslation()
        link_rot = link_local_m.ExtractRotation().GetQuat()
        rpy = quat_to_rpy(
            link_rot.GetReal(),
            link_rot.GetImaginary()[0],
            link_rot.GetImaginary()[1],
            link_rot.GetImaginary()[2],
        )

        # Visuals
        visuals_prim = stage.GetPrimAtPath(f"{link_prim_path}/visuals")
        if visuals_prim and visuals_prim.IsValid() and visuals_prim.IsA(UsdGeom.Mesh):
            obj_path = MESH_DIR / f"{mesh_basename}_visual.stl"
            # The visuals mesh's transform is identity relative to its parent
            # link prim (verified empirically — Mesh prims are direct children
            # without their own Xform ops). Keep transform_local_to_link as
            # identity so the OBJ stays in the link's own local frame.
            ok, vcount, fcount = extract_mesh_to_stl(
                visuals_prim, obj_path, Gf.Matrix4d(1)
            )
            if ok:
                rgba = read_diffuse_rgba(stage, visuals_prim)
                print(f"  visual {link_name:30s} → {obj_path.name}  v={vcount} f={fcount}  rgba={rgba}")
                visual_blocks.append({
                    "obj": obj_path.name,
                    "xyz": (link_xlate[0], link_xlate[1], link_xlate[2]),
                    "rpy": rpy,
                    "rgba": rgba,
                })

        # Collisions
        collisions_prim = stage.GetPrimAtPath(f"{link_prim_path}/collisions")
        if collisions_prim and collisions_prim.IsValid() and collisions_prim.IsA(UsdGeom.Mesh):
            obj_path = MESH_DIR / f"{mesh_basename}_collision.stl"
            ok, vcount, fcount = extract_mesh_to_stl(
                collisions_prim, obj_path, Gf.Matrix4d(1)
            )
            if ok:
                print(f"  collis {link_name:30s} → {obj_path.name}  v={vcount} f={fcount}")
                collision_blocks.append({
                    "obj": obj_path.name,
                    "xyz": (link_xlate[0], link_xlate[1], link_xlate[2]),
                    "rpy": rpy,
                })

    # Build the URDF document.
    robot = ET.Element("robot", {"name": "rg2"})

    # MATERIALS AT TOP LEVEL (not inline per visual).
    # Multi-visual-per-link URDFs hit a documented RViz2 bug where only the
    # FIRST visual's inline <material><color> is honored; subsequent ones get
    # collapsed to default grey. The well-known workaround is to define each
    # unique color as a top-level <material name="..."> and reference it by
    # name from each <visual>.
    #   ros2/rviz#1293 — multiple visuals: first material used for both
    #   ros-visualization/rviz#843 — multiple visuals get first material
    #   ros-answers/question/154816 — color issue with URDF + multiple visuals
    unique_rgbas = {}
    for blk in visual_blocks:
        unique_rgbas[blk["rgba"]] = unique_rgbas.get(blk["rgba"], 0) + 1
    rgba_to_name = {}
    for i, (rgba, _count) in enumerate(unique_rgbas.items()):
        # Stable, descriptive name keyed on the rgba value
        name = f"rg2_color_{i}_{int(rgba[0]*255):03d}_{int(rgba[1]*255):03d}_{int(rgba[2]*255):03d}"
        rgba_to_name[rgba] = name
        m = ET.SubElement(robot, "material", {"name": name})
        ET.SubElement(m, "color", {"rgba": " ".join(f"{c:.3f}" for c in rgba)})

    rg2_link = ET.SubElement(robot, "link", {"name": "rg2_base_link"})

    # Inertial — placeholder; the real value would come from an OnRobot datasheet.
    # Phase 7 is visualization-only (D-8: GRIPPER_CENTER_TOOL_OFFSET stays
    # untouched; no force-mode integration). We only need a valid URDF.
    inertial = ET.SubElement(rg2_link, "inertial")
    ET.SubElement(inertial, "origin", {"xyz": "0 0 0.1", "rpy": "0 0 0"})
    ET.SubElement(inertial, "mass", {"value": "0.78"})  # OnRobot RG2 ≈ 780 g
    ET.SubElement(
        inertial,
        "inertia",
        {
            "ixx": "1e-3", "ixy": "0", "ixz": "0",
            "iyy": "1e-3", "iyz": "0",
            "izz": "1e-3",
        },
    )

    for blk in visual_blocks:
        v = ET.SubElement(rg2_link, "visual")
        ET.SubElement(
            v,
            "origin",
            {
                "xyz": f"{blk['xyz'][0]:.6f} {blk['xyz'][1]:.6f} {blk['xyz'][2]:.6f}",
                "rpy": f"{blk['rpy'][0]:.6f} {blk['rpy'][1]:.6f} {blk['rpy'][2]:.6f}",
            },
        )
        g = ET.SubElement(v, "geometry")
        ET.SubElement(
            g,
            "mesh",
            {
                "filename": f"package://compliant_insertion_studio/urdf/rg2/meshes/{blk['obj']}",
                "scale": "1 1 1",
            },
        )
        # Reference the top-level material by NAME ONLY (no inline <color>).
        # RViz2 looks up the rgba from the named top-level <material> and
        # applies it correctly per-visual.
        ET.SubElement(v, "material", {"name": rgba_to_name[blk["rgba"]]})

    for blk in collision_blocks:
        c = ET.SubElement(rg2_link, "collision")
        ET.SubElement(
            c,
            "origin",
            {
                "xyz": f"{blk['xyz'][0]:.6f} {blk['xyz'][1]:.6f} {blk['xyz'][2]:.6f}",
                "rpy": f"{blk['rpy'][0]:.6f} {blk['rpy'][1]:.6f} {blk['rpy'][2]:.6f}",
            },
        )
        g = ET.SubElement(c, "geometry")
        ET.SubElement(
            g,
            "mesh",
            {
                "filename": f"package://compliant_insertion_studio/urdf/rg2/meshes/{blk['obj']}",
                "scale": "1 1 1",
            },
        )

    # Pretty-print the XML
    ET.indent(robot, space="  ")
    tree = ET.ElementTree(robot)
    tree.write(URDF_PATH, encoding="utf-8", xml_declaration=True)
    print(f"\nWrote {URDF_PATH}")
    print(f"  visual_blocks   : {len(visual_blocks)}")
    print(f"  collision_blocks: {len(collision_blocks)}")


if __name__ == "__main__":
    main()
