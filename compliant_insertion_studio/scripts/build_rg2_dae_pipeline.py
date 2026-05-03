#!/usr/bin/env python3
# Reference: https://pycollada.readthedocs.io/
"""
Build a high-fidelity RG2 visual mesh in Collada (.dae) format using pycollada,
sourcing geometry + material colors directly from the OnRobot RG2 USD asset.

Why DAE: ROS RViz uses Assimp under the hood. Assimp's DAE loader honors
embedded diffuse + ambient + specular per-mesh, which gives the same shaded
material quality the UR5e arm uses (UR description ships .dae visuals).
Isaac Sim's asset_exporter does NOT support DAE natively (only OBJ / FBX /
glTF / USDZ / STL), so we go USD → DAE directly via pxr + pycollada.

Output:
  compliant_insertion_studio/urdf/rg2/rg2.dae   (single DAE, all 7 sub-meshes
                                                  as separate <geometry>s
                                                  with per-mesh materials)
  compliant_insertion_studio/urdf/rg2/rg2.urdf  (single visual referencing
                                                  rg2.dae)

Run with system Python 3:
    pip install --user pycollada     # if not present
    python3 compliant_insertion_studio/scripts/build_rg2_dae_pipeline.py
"""
import sys
from pathlib import Path
from xml.etree import ElementTree as ET

import numpy as np
from pxr import Usd, UsdGeom, UsdShade, Gf

import collada
from collada.scene import (
    Scene,
    Node,
    GeometryNode,
    MaterialNode,
)

REPO = Path("/home/aaugus11/Documents/ros-mcp-server")
RG2_USD = "/home/aaugus11/Documents/isaac-sim-mcp/exts/ur5e-dt/assets/gripper/RG2.usd"
OUT_DIR = REPO / "compliant_insertion_studio" / "urdf" / "rg2"
OUT_DAE = OUT_DIR / "rg2.dae"
OUT_URDF = OUT_DIR / "rg2.urdf"

# Per-link mesh path in the USD. Same set as the previous STL extractor.
LINKS = [
    "onrobot_rg2_base_link",
    "left_outer_knuckle",
    "left_inner_knuckle",
    "left_inner_finger",
    "right_outer_knuckle",
    "right_inner_knuckle",
    "right_inner_finger",
]


def read_usd_materials(stage: Usd.Stage) -> dict[str, dict]:
    """Read every UsdShade.Material's diffuse_color_constant and return
    {material_name: {"rgb": (r, g, b)}}."""
    out: dict[str, dict] = {}
    for prim in stage.Traverse():
        if not prim.IsA(UsdShade.Material):
            continue
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
                    out[name] = {"rgb": (float(c[0]), float(c[1]), float(c[2]))}
                    break
            if name in out:
                break
    return out


def get_link_material_name(stage: Usd.Stage, link_path: str) -> str | None:
    """Look up the material:binding on this link's `visuals` Mesh prim."""
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


def extract_link_mesh(stage: Usd.Stage, link_path: str):
    """Return (vertices_world, triangles, normals_per_tri) where vertices are
    expressed in the asset root frame (i.e. /khi_rs080n's frame), so we can
    bake all sub-meshes into one DAE coordinate system without per-mesh node
    transforms.
    """
    link_prim = stage.GetPrimAtPath(link_path)
    if not link_prim or not link_prim.IsValid():
        return None
    visuals_prim = stage.GetPrimAtPath(f"{link_path}/visuals")
    if not visuals_prim or not visuals_prim.IsValid() or not visuals_prim.IsA(UsdGeom.Mesh):
        return None

    mesh = UsdGeom.Mesh(visuals_prim)
    points = mesh.GetPointsAttr().Get()
    fvc = mesh.GetFaceVertexCountsAttr().Get()
    fvi = mesh.GetFaceVertexIndicesAttr().Get()
    if points is None or fvc is None or fvi is None:
        return None

    # Local-to-asset-root transform = link prim's xformable transformation.
    # (USD's GetLocalTransformation gives the prim's transform in its parent's
    # frame. Since /khi_rs080n is identity, this == in-asset-root frame.)
    link_xform = UsdGeom.Xformable(link_prim)
    M = link_xform.GetLocalTransformation()

    # Apply the transform to each point.
    verts = []
    for p in points:
        v = M.Transform(Gf.Vec3d(p[0], p[1], p[2]))
        verts.append((float(v[0]), float(v[1]), float(v[2])))
    verts = np.array(verts, dtype=np.float32)

    # Fan-triangulate each face.
    triangles = []
    idx = 0
    for count in fvc:
        face_idx = list(fvi[idx : idx + count])
        for i in range(1, count - 1):
            triangles.append((face_idx[0], face_idx[i], face_idx[i + 1]))
        idx += count
    triangles = np.array(triangles, dtype=np.uint32)

    # Compute per-triangle normals.
    v0 = verts[triangles[:, 0]]
    v1 = verts[triangles[:, 1]]
    v2 = verts[triangles[:, 2]]
    a = v1 - v0
    b = v2 - v0
    n = np.cross(a, b)
    nl = np.linalg.norm(n, axis=1, keepdims=True)
    nl[nl < 1e-12] = 1.0
    n = (n / nl).astype(np.float32)

    return verts, triangles, n


def build_dae(usd_stage: Usd.Stage, materials: dict[str, dict]) -> collada.Collada:
    mesh_doc = collada.Collada()

    # Effects + materials per unique USD material.
    name_to_material: dict[str, collada.material.Material] = {}
    for mat_name, info in materials.items():
        r, g, b = info["rgb"]
        # Diffuse + small ambient + small specular for shaded look (mirrors
        # what UR's DAEs do with Phong/Lambert).
        effect = collada.material.Effect(
            f"effect_{mat_name}",
            [],  # no surface params
            "lambert",
            ambient=(r * 0.25, g * 0.25, b * 0.25, 1.0),
            diffuse=(r, g, b, 1.0),
            specular=(0.05, 0.05, 0.05, 1.0),
        )
        material = collada.material.Material(
            f"material_id_{mat_name}", mat_name, effect
        )
        mesh_doc.effects.append(effect)
        mesh_doc.materials.append(material)
        name_to_material[mat_name] = material

    # Per-link Geometry + node, each bound to the link's USD material.
    geometry_nodes: list[Node] = []
    default_prim_name = usd_stage.GetDefaultPrim().GetName()

    for link_name in LINKS:
        link_path = f"/{default_prim_name}/{link_name}"
        result = extract_link_mesh(usd_stage, link_path)
        if result is None:
            print(f"  SKIP: {link_path} has no extractable visual mesh")
            continue
        verts, triangles, normals = result
        mat_name = get_link_material_name(usd_stage, link_path)
        if mat_name is None or mat_name not in name_to_material:
            print(f"  SKIP: {link_path} has no resolvable material binding")
            continue

        # Flatten arrays for pycollada Source format
        vert_floats = verts.flatten()
        # Build flat normal array — one normal entry per triangle, repeated 3x
        # so each vertex of the triangle uses the same per-face normal.
        norm_floats = np.repeat(normals, 3, axis=0).flatten()

        vert_src = collada.source.FloatSource(
            f"{link_name}_verts", vert_floats, ("X", "Y", "Z")
        )
        norm_src = collada.source.FloatSource(
            f"{link_name}_norms", norm_floats, ("X", "Y", "Z")
        )
        geom = collada.geometry.Geometry(
            mesh_doc, f"geom_{link_name}", link_name, [vert_src, norm_src]
        )

        # Triangle-set with 2 inputs: VERTEX (uses verts source) and NORMAL.
        input_list = collada.source.InputList()
        input_list.addInput(0, "VERTEX", f"#{link_name}_verts")
        input_list.addInput(1, "NORMAL", f"#{link_name}_norms")

        # Triangle indices array: for each tri, [v0, n0, v1, n1, v2, n2].
        # vertex indices come from `triangles`; normal index = global triangle
        # index (since normals are per-triangle, repeated 3x in the flat array,
        # so normal-index i for triangle T gives 3T+i, mapped via the flat
        # index list).
        n_tris = triangles.shape[0]
        indices = np.zeros((n_tris, 6), dtype=np.uint32)
        indices[:, 0] = triangles[:, 0]  # v0
        indices[:, 2] = triangles[:, 1]  # v1
        indices[:, 4] = triangles[:, 2]  # v2
        # Per-tri normal indices: triangle T uses normal entries 3T, 3T+1, 3T+2
        # (all the same value since normals are per-face — but we authored
        # 3 copies, one per vertex of the tri).
        tri_idx = np.arange(n_tris, dtype=np.uint32)
        indices[:, 1] = tri_idx * 3
        indices[:, 3] = tri_idx * 3 + 1
        indices[:, 5] = tri_idx * 3 + 2
        triset = geom.createTriangleSet(
            indices.flatten(), input_list, mat_name
        )
        geom.primitives.append(triset)
        mesh_doc.geometries.append(geom)

        # Bind material in scene.
        material_binding = MaterialNode(mat_name, name_to_material[mat_name], inputs=[])
        geom_node = GeometryNode(geom, [material_binding])
        node = Node(f"node_{link_name}", children=[geom_node])
        geometry_nodes.append(node)
        print(f"  added geometry: {link_name}  v={verts.shape[0]} t={triangles.shape[0]} mat={mat_name}")

    scene = Scene("rg2_scene", geometry_nodes)
    mesh_doc.scenes.append(scene)
    mesh_doc.scene = scene

    return mesh_doc


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

    # Single visual referencing the DAE.
    visual = ET.SubElement(link, "visual")
    ET.SubElement(visual, "origin", {"xyz": "0 0 0", "rpy": "0 0 0"})
    g = ET.SubElement(visual, "geometry")
    ET.SubElement(
        g, "mesh",
        {"filename": f"package://compliant_insertion_studio/urdf/rg2/{dae_filename}",
         "scale": "1 1 1"}
    )
    # No <material> block — DAE provides materials.

    # Collisions: keep the per-part STL pipeline (already known good).
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
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    stage = Usd.Stage.Open(RG2_USD)
    if stage is None:
        print(f"ERROR: cannot open {RG2_USD}", file=sys.stderr)
        sys.exit(1)

    print(f"USD opened. Default prim: {stage.GetDefaultPrim().GetPath()}")
    materials = read_usd_materials(stage)
    print("USD-resolved material colors:")
    for k, v in materials.items():
        print(f"  {k}: {v['rgb']}")

    print("\nBuilding DAE...")
    mesh_doc = build_dae(stage, materials)
    mesh_doc.write(str(OUT_DAE))
    print(f"\nWrote DAE: {OUT_DAE.relative_to(REPO)} ({OUT_DAE.stat().st_size:,} bytes)")

    build_urdf(OUT_DAE.name, OUT_URDF)
    print(f"Wrote URDF: {OUT_URDF.relative_to(REPO)}")


if __name__ == "__main__":
    main()
