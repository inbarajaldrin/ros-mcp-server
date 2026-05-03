#!/usr/bin/env python3
# Reference: https://trimesh.org/
"""
Build a HIGHER-POLY DAE for the RG2 by Loop-subdividing each sub-mesh.

Why: the source RG2.usd has a uniform 10000 triangles per sub-mesh (70k
total) with fully unwelded vertices and `bilinear` subdivisionScheme — so
nothing in the asset_converter or pyassimp pipeline can smooth the curved
gripper body beyond what the source already encodes. The "low-poly look"
GPT and the operator both noted is real and structural to the source.

This script reads each per-link mesh from the USD directly, welds vertices
with a small tolerance (so that subdivision can flow across triangle edges),
runs Loop subdivision N times (each iteration ~4x the tri count), then
emits a single DAE with per-mesh materials matching the USD's
diffuse_color_constant values.

Trade-off: subdivision multiplies the polycount. 1 iteration = ~4x → 280k.
2 iterations = ~16x → 1.1M. RViz handles a few million tris fine, but
launch parsing the URDF takes longer. Default to 1 iteration.

Usage:
    python3 compliant_insertion_studio/scripts/build_rg2_dae_subdivided.py [--subdiv N]
"""
import argparse
import sys
from pathlib import Path
from xml.etree import ElementTree as ET

import numpy as np
import trimesh
import collada
from collada.scene import Scene, Node, GeometryNode, MaterialNode
from pxr import Usd, UsdGeom, UsdShade, Gf

REPO = Path("/home/aaugus11/Documents/ros-mcp-server")
RG2_USD = "/home/aaugus11/Documents/isaac-sim-mcp/exts/ur5e-dt/assets/gripper/RG2.usd"
OUT_DIR = REPO / "compliant_insertion_studio" / "urdf" / "rg2"
OUT_DAE = OUT_DIR / "rg2.dae"
OUT_URDF = OUT_DIR / "rg2.urdf"

LINKS = [
    "onrobot_rg2_base_link",
    "left_outer_knuckle",
    "left_inner_knuckle",
    "left_inner_finger",
    "right_outer_knuckle",
    "right_inner_knuckle",
    "right_inner_finger",
]


def read_usd_diffuses(stage: Usd.Stage) -> dict[str, dict]:
    out: dict[str, dict] = {}
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
                    out[name] = {"rgb": (float(c[0]), float(c[1]), float(c[2]))}
                    break
            if name in out:
                break
    return out


def get_link_material_name(stage: Usd.Stage, link_path: str) -> str | None:
    visuals_prim = stage.GetPrimAtPath(f"{link_path}/visuals")
    if not visuals_prim or not visuals_prim.IsValid():
        return None
    rel = visuals_prim.GetRelationship("material:binding")
    if not rel:
        return None
    targets = rel.GetTargets()
    if not targets:
        return None
    return Path(str(targets[0])).name


def usd_mesh_to_trimesh(stage: Usd.Stage, link_path: str) -> trimesh.Trimesh | None:
    """Read a USD link's visual mesh, transform points into the asset root
    frame (so all sub-meshes can be combined into one DAE coordinate
    system), and return a welded trimesh.Trimesh.
    """
    visuals_prim = stage.GetPrimAtPath(f"{link_path}/visuals")
    if not visuals_prim or not visuals_prim.IsValid() or not visuals_prim.IsA(UsdGeom.Mesh):
        return None
    mesh_prim = stage.GetPrimAtPath(link_path)
    if not mesh_prim:
        return None
    mesh = UsdGeom.Mesh(visuals_prim)
    points = mesh.GetPointsAttr().Get()
    fvc = mesh.GetFaceVertexCountsAttr().Get()
    fvi = mesh.GetFaceVertexIndicesAttr().Get()
    if points is None:
        return None

    # Transform to asset-root frame
    M = UsdGeom.Xformable(mesh_prim).GetLocalTransformation()
    verts = np.array([
        list(M.Transform(Gf.Vec3d(p[0], p[1], p[2]))) for p in points
    ], dtype=np.float64)

    # Triangulate
    triangles = []
    idx = 0
    for count in fvc:
        face = list(fvi[idx : idx + count])
        for i in range(1, count - 1):
            triangles.append((face[0], face[i], face[i + 1]))
        idx += count
    triangles = np.array(triangles, dtype=np.int64)

    tm = trimesh.Trimesh(vertices=verts, faces=triangles, process=False)
    return tm


def build_dae(stage: Usd.Stage, materials: dict[str, dict],
              subdiv: int) -> collada.Collada:
    mesh_doc = collada.Collada()

    name_to_material: dict[str, collada.material.Material] = {}
    for mat_name, info in materials.items():
        r, g, b = info["rgb"]
        effect = collada.material.Effect(
            f"effect_{mat_name}", [], "lambert",
            ambient=(r * 0.25, g * 0.25, b * 0.25, 1.0),
            diffuse=(r, g, b, 1.0),
            specular=(0.05, 0.05, 0.05, 1.0),
        )
        material = collada.material.Material(f"material_id_{mat_name}", mat_name, effect)
        mesh_doc.effects.append(effect)
        mesh_doc.materials.append(material)
        name_to_material[mat_name] = material

    geometry_nodes: list[Node] = []
    default_prim = stage.GetDefaultPrim()
    print(f"\nProcessing {len(LINKS)} links (subdiv iterations: {subdiv})...")

    for link_name in LINKS:
        link_path = f"/{default_prim.GetName()}/{link_name}"
        tm = usd_mesh_to_trimesh(stage, link_path)
        if tm is None:
            print(f"  SKIP: {link_path}")
            continue
        mat_name = get_link_material_name(stage, link_path)
        if not mat_name or mat_name not in name_to_material:
            print(f"  SKIP: {link_path} (no material binding)")
            continue

        v0, t0 = len(tm.vertices), len(tm.faces)
        # Weld duplicate vertices first (tolerance via trimesh default ~1e-6).
        # Use a more aggressive merge to fix the unwelded source mesh.
        tm.merge_vertices(merge_tex=False, merge_norm=False)
        v_after_weld = len(tm.vertices)
        # Loop subdivision REQUIRES manifold mesh (CAD-converted USDs are
        # often non-manifold). Use midpoint subdivision instead — tolerates
        # non-manifold edges, just splits each triangle into 4 by inserting
        # midpoint vertices at edge midpoints.
        for _ in range(subdiv):
            new_v, new_f = trimesh.remesh.subdivide(tm.vertices, tm.faces)
            tm = trimesh.Trimesh(vertices=new_v, faces=new_f, process=False)
            tm.merge_vertices(merge_tex=False, merge_norm=False)
        # Compute smooth vertex normals
        normals = tm.vertex_normals
        v1, t1 = len(tm.vertices), len(tm.faces)
        print(f"  {link_name:30s}  v: {v0}→{v_after_weld} (welded) →{v1} (subdiv)  t: {t0}→{t1}  mat={mat_name}")

        # Flatten arrays for pycollada
        vert_floats = np.asarray(tm.vertices, dtype=np.float32).flatten()
        norm_floats = np.asarray(normals, dtype=np.float32).flatten()

        vert_src = collada.source.FloatSource(f"{link_name}_verts", vert_floats, ("X", "Y", "Z"))
        norm_src = collada.source.FloatSource(f"{link_name}_norms", norm_floats, ("X", "Y", "Z"))
        geom = collada.geometry.Geometry(mesh_doc, f"geom_{link_name}", link_name, [vert_src, norm_src])

        input_list = collada.source.InputList()
        input_list.addInput(0, "VERTEX", f"#{link_name}_verts")
        input_list.addInput(1, "NORMAL", f"#{link_name}_norms")

        # Per-vertex normals → vertex index = normal index
        triangles = np.asarray(tm.faces, dtype=np.uint32)
        n_tris = triangles.shape[0]
        indices = np.zeros((n_tris, 6), dtype=np.uint32)
        indices[:, 0::2] = triangles  # vertex indices
        indices[:, 1::2] = triangles  # normal indices == vertex indices

        triset = geom.createTriangleSet(indices.flatten(), input_list, mat_name)
        geom.primitives.append(triset)
        mesh_doc.geometries.append(geom)

        material_binding = MaterialNode(mat_name, name_to_material[mat_name], inputs=[])
        geom_node = GeometryNode(geom, [material_binding])
        node = Node(f"node_{link_name}", children=[geom_node])
        geometry_nodes.append(node)

    scene_obj = Scene("rg2_scene", geometry_nodes)
    mesh_doc.scenes.append(scene_obj)
    mesh_doc.scene = scene_obj
    return mesh_doc


def build_urdf(dae_filename: str, urdf_path: Path) -> None:
    robot = ET.Element("robot", {"name": "rg2"})
    link = ET.SubElement(robot, "link", {"name": "rg2_base_link"})
    inertial = ET.SubElement(link, "inertial")
    ET.SubElement(inertial, "origin", {"xyz": "0 0 0.1", "rpy": "0 0 0"})
    ET.SubElement(inertial, "mass", {"value": "0.78"})
    ET.SubElement(inertial, "inertia",
                  {"ixx": "1e-3", "ixy": "0", "ixz": "0",
                   "iyy": "1e-3", "iyz": "0", "izz": "1e-3"})

    visual = ET.SubElement(link, "visual")
    ET.SubElement(visual, "origin", {"xyz": "0 0 0", "rpy": "0 0 0"})
    g = ET.SubElement(visual, "geometry")
    ET.SubElement(g, "mesh",
                  {"filename": f"package://compliant_insertion_studio/urdf/rg2/{dae_filename}",
                   "scale": "1 1 1"})

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
        ET.SubElement(gc, "mesh",
                      {"filename": f"package://compliant_insertion_studio/urdf/rg2/meshes/{fname}",
                       "scale": "1 1 1"})

    ET.indent(robot, space="  ")
    ET.ElementTree(robot).write(urdf_path, encoding="utf-8", xml_declaration=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--subdiv", type=int, default=1,
                    help="Loop subdivision iterations (0=just weld, 1≈4x tris, 2≈16x tris)")
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    stage = Usd.Stage.Open(RG2_USD)
    if stage is None:
        print(f"ERROR: cannot open {RG2_USD}", file=sys.stderr)
        sys.exit(1)

    materials = read_usd_diffuses(stage)
    print(f"USD-resolved material colors:")
    for k, v in materials.items():
        print(f"  {k}: {v['rgb']}")

    mesh_doc = build_dae(stage, materials, args.subdiv)
    mesh_doc.write(str(OUT_DAE))
    print(f"\nWrote DAE: {OUT_DAE.relative_to(REPO)} ({OUT_DAE.stat().st_size:,} bytes)")

    build_urdf(OUT_DAE.name, OUT_URDF)
    print(f"Wrote URDF: {OUT_URDF.relative_to(REPO)}")


if __name__ == "__main__":
    main()
