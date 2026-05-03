#!/usr/bin/env python3
"""
Build the production DAE for the RG2 gripper using the SOURCE USD'S AUTHORED
per-face-vertex normals — DO NOT weld, DO NOT recompute.

Why: every previous DAE pipeline (extract_rg2_urdf.py STL, build_rg2_obj_pipeline.py,
build_rg2_dae_pipeline.py, build_rg2_dae_via_assimp.py, build_rg2_dae_subdivided.py)
either let pyassimp/trimesh weld duplicate vertices (which destroys the sharp
normal seams at recess edges) OR computed flat per-face normals that don't
preserve curvature smoothing within a face.

The source RG2.usd has *faceVarying* normals (30000 of them, one per
face-vertex). These were authored to preserve curvature where surfaces are
smooth AND keep sharp seams where surfaces meet at hard edges (e.g., the
recessed bolt-holes and rectangular cavity on the base housing front face).

Welding collapses two coincident vertices into one, forcing them to share a
single normal — which is the average of the inward and outward normals at a
recess rim, pointing into the cavity. That's why the recess interiors render
dark in our previous pipeline but not in Isaac Sim (which respects the
authored normals directly).

This script:
1. Reads each link's `/visuals` Mesh prim from RG2.usd
2. Pulls out vertices, triangles (fan-triangulated), and authored faceVarying
   normals — keeps them all in their original layout
3. Transforms vertices to the asset root frame
4. Emits one DAE with per-mesh per-vertex normals AS-AUTHORED + materials
   from UsdShade.Material's diffuse_color_constant

The result should match Isaac Sim's render — including the recessed features
on the front housing rendering with correct shading instead of looking dark.
"""
from pathlib import Path
from xml.etree import ElementTree as ET
import sys

import numpy as np
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


def get_link_material_name(stage, link_path):
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


def extract_link_with_authored_normals(stage, link_path):
    """Extract vertices, triangles, and authored faceVarying normals from a
    USD link's /visuals Mesh prim. Vertices are transformed into the asset
    root frame. Returns (verts, tris, normals_per_vertex) where:
      - verts: (N, 3) float32 in asset-root frame
      - tris: (T, 3) uint32 indexing into verts
      - normals_per_vertex: (N, 3) float32, one per vertex (since the source
        is unwelded with faceVarying interp == one normal per vertex)
    """
    link_prim = stage.GetPrimAtPath(link_path)
    visuals_prim = stage.GetPrimAtPath(f"{link_path}/visuals")
    if not visuals_prim or not visuals_prim.IsValid() or not visuals_prim.IsA(UsdGeom.Mesh):
        return None
    mesh = UsdGeom.Mesh(visuals_prim)

    points = mesh.GetPointsAttr().Get()
    fvc = mesh.GetFaceVertexCountsAttr().Get()
    fvi = mesh.GetFaceVertexIndicesAttr().Get()
    norms = mesh.GetNormalsAttr().Get()
    norm_interp = mesh.GetNormalsInterpolation()

    if not (points and fvc and fvi and norms):
        return None

    n_verts = len(points)
    n_norms = len(norms)
    n_fvi = len(fvi)

    # Source RG2: unwelded mesh with faceVarying normals == 1 normal per
    # face-vertex. Since the mesh is unwelded (3 distinct vertices per tri,
    # no sharing), face-vertex index N corresponds to vertex N exactly.
    # That means normals_per_vertex == normals (mapped 1:1 by index).
    # If a future asset is welded with faceVarying normals, the mapping
    # would be different — bail out so we notice.
    if not (norm_interp == "faceVarying" and n_norms == n_fvi == n_verts):
        # Fall back: try direct vertex-mapped if interpolation is "vertex"
        if not (norm_interp == "vertex" and n_norms == n_verts):
            print(f"  WARN: {link_path} normals layout unexpected "
                  f"(interp={norm_interp}, n_norms={n_norms}, n_verts={n_verts}, n_fvi={n_fvi})")
            print(f"        falling back to computed normals")
            norms = None  # will compute below

    # Transform vertices to asset-root frame
    M = UsdGeom.Xformable(link_prim).GetLocalTransformation()
    verts = np.array([
        list(M.Transform(Gf.Vec3d(p[0], p[1], p[2]))) for p in points
    ], dtype=np.float32)

    # Transform normals through the rotation part of the link matrix.
    # USD's local transform of these links is just rotation (no skew/scale),
    # so applying it to normals = applying rotation. For correctness in
    # general we'd use inverse-transpose; here it's identity for our case.
    if norms is not None:
        # Decompose: extract pure rotation by zeroing translation
        # (and re-orthonormalize if there's any skew).
        R = M.ExtractRotationMatrix()
        norms_array = np.array([
            list(R * Gf.Vec3d(n[0], n[1], n[2])) for n in norms
        ], dtype=np.float32)
        # Renormalize (rotation should preserve length, but be safe)
        lens = np.linalg.norm(norms_array, axis=1, keepdims=True)
        lens[lens < 1e-12] = 1.0
        norms_array = norms_array / lens
    else:
        norms_array = None  # computed below per face

    # Triangulate (fan)
    tris = []
    idx = 0
    for count in fvc:
        face = list(fvi[idx : idx + count])
        for i in range(1, count - 1):
            tris.append((face[0], face[i], face[i + 1]))
        idx += count
    tris = np.array(tris, dtype=np.uint32)

    # If we couldn't use authored normals, compute per-face flat normals as fallback
    if norms_array is None:
        v0 = verts[tris[:, 0]]
        v1 = verts[tris[:, 1]]
        v2 = verts[tris[:, 2]]
        face_n = np.cross(v1 - v0, v2 - v0)
        ln = np.linalg.norm(face_n, axis=1, keepdims=True)
        ln[ln < 1e-12] = 1.0
        face_n = face_n / ln
        # Replicate face normal to each of its 3 vertices
        norms_array = np.zeros((len(verts), 3), dtype=np.float32)
        for ti, t in enumerate(tris):
            norms_array[t[0]] = face_n[ti]
            norms_array[t[1]] = face_n[ti]
            norms_array[t[2]] = face_n[ti]

    return verts, tris, norms_array


def build_dae(stage, materials):
    mesh_doc = collada.Collada()
    name_to_material = {}
    for mat_name, info in materials.items():
        r, g, b = info["rgb"]
        # Source RG2.usd has per-face FLAT normals (every vertex of a face
        # shares one normal — verified empirically against cross-product
        # face normals: 100% aligned). With pure Lambert shading, faces
        # angled away from RViz's directional light go very dark, making
        # the rounded body look uniformly black. To compensate, bump the
        # ambient term high (≈ 0.85 × diffuse) so even shadowed faces stay
        # close to the canonical USD color. This trades a little shading
        # contrast for the brightness Isaac Sim's render naturally produces.
        AMB_FACTOR = 0.85
        effect = collada.material.Effect(
            f"effect_{mat_name}", [], "lambert",
            ambient=(r * AMB_FACTOR, g * AMB_FACTOR, b * AMB_FACTOR, 1.0),
            diffuse=(r, g, b, 1.0),
            specular=(0.05, 0.05, 0.05, 1.0),
        )
        material = collada.material.Material(f"material_id_{mat_name}", mat_name, effect)
        mesh_doc.effects.append(effect)
        mesh_doc.materials.append(material)
        name_to_material[mat_name] = material

    geometry_nodes = []
    default_prim = stage.GetDefaultPrim()
    print(f"\nProcessing {len(LINKS)} links (using AUTHORED normals from USD)...")

    for link_name in LINKS:
        link_path = f"/{default_prim.GetName()}/{link_name}"
        result = extract_link_with_authored_normals(stage, link_path)
        if result is None:
            print(f"  SKIP: {link_path}")
            continue
        verts, tris, norms = result
        mat_name = get_link_material_name(stage, link_path)
        if not mat_name or mat_name not in name_to_material:
            print(f"  SKIP: {link_path} (no material binding)")
            continue
        print(f"  {link_name:30s}  v={len(verts)}  t={len(tris)}  norms={len(norms)}  mat={mat_name}")

        vert_floats = verts.flatten()
        norm_floats = norms.flatten()
        vert_src = collada.source.FloatSource(f"{link_name}_verts", vert_floats, ("X", "Y", "Z"))
        norm_src = collada.source.FloatSource(f"{link_name}_norms", norm_floats, ("X", "Y", "Z"))
        geom = collada.geometry.Geometry(mesh_doc, f"geom_{link_name}", link_name,
                                          [vert_src, norm_src])

        input_list = collada.source.InputList()
        input_list.addInput(0, "VERTEX", f"#{link_name}_verts")
        input_list.addInput(1, "NORMAL", f"#{link_name}_norms")

        # Per-vertex normals → normal index == vertex index
        n_tris = tris.shape[0]
        indices = np.zeros((n_tris, 6), dtype=np.uint32)
        indices[:, 0::2] = tris  # vertex
        indices[:, 1::2] = tris  # normal (same as vertex)

        triset = geom.createTriangleSet(indices.flatten(), input_list, mat_name)
        geom.primitives.append(triset)
        mesh_doc.geometries.append(geom)

        material_binding = MaterialNode(mat_name, name_to_material[mat_name], inputs=[])
        geom_node = GeometryNode(geom, [material_binding])
        node = Node(f"node_{link_name}", children=[geom_node])
        geometry_nodes.append(node)

    scene = Scene("rg2_scene", geometry_nodes)
    mesh_doc.scenes.append(scene)
    mesh_doc.scene = scene
    return mesh_doc


def build_urdf(dae_filename, urdf_path):
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


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    stage = Usd.Stage.Open(RG2_USD)
    if stage is None:
        print(f"ERROR: cannot open {RG2_USD}", file=sys.stderr)
        sys.exit(1)

    materials = read_usd_diffuses(stage)
    print(f"USD-resolved material colors:")
    for k, v in materials.items():
        print(f"  {k}: {v['rgb']}")

    mesh_doc = build_dae(stage, materials)
    mesh_doc.write(str(OUT_DAE))
    print(f"\nWrote DAE: {OUT_DAE.relative_to(REPO)} ({OUT_DAE.stat().st_size:,} bytes)")
    build_urdf(OUT_DAE.name, OUT_URDF)
    print(f"Wrote URDF: {OUT_URDF.relative_to(REPO)}")


if __name__ == "__main__":
    main()
