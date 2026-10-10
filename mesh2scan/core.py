# SPDX-License-Identifier: GPL-3.0-or-later
"""Topology-independent acquisition -> point volume -> triangle reconstruction.

Blender data are accessed only on the main thread. All expensive reconstruction,
ray casting, nearest-point fields and decimation use native Blender operations.
NumPy is bundled with Blender; no pip installs or downloaded executables are used.
"""

import json
import math
import time
from dataclasses import dataclass, replace

import bmesh
import bpy
import numpy as np
from mathutils import Matrix, Vector
from mathutils.bvhtree import BVHTree
from .features import (
    curved_reference,
    recover_boundaries,
    add_diagnostic_colors,
    boundary_corner_count,
)
from .guided import guided_mesh
from .planning import is_guided, fit_plan


class ScanError(RuntimeError):
    pass


@dataclass
class ScanOptions:
    mesher: str = "GUIDED"
    memory_policy: str = "AUTO"
    observation_cap: int = 2_000_000
    observed_fit: float = 0.25
    fairing_iterations: int = 2
    surface_mode: str = "SMOOTH"
    smooth_levels: int = 3
    smooth_strength: float = 0.7
    boundary_recovery: bool = True
    retained_gb: float = 0.0
    preplanned: bool = False
    component_id: int = 0
    acquisition: str = "RAYS"
    target_triangles: int = 1_000_000
    resolution: str = "TARGET"
    voxel_world: float = 0.001
    samples_per_triangle: float = 3.0
    adaptive: float = 0.65
    detail_boost: float = 2.0
    crease_angle: float = math.radians(25.0)
    snap_radius: float = 0.2
    radius_voxels: float = 2.5
    standoff_voxels: float = 4.0
    output: str = "SURFACE"
    fidelity: str = "SOURCE"
    keep_cloud: bool = True
    hide_source: bool = True
    use_modifiers: bool = True
    clean_source: bool = True
    marked_edges: bool = False
    memory_gb: float = 8.0
    seed: int = 42


def mesh_coords(mesh):
    co = np.empty((len(mesh.vertices), 3), dtype=np.float32)
    mesh.vertices.foreach_get("co", co.ravel())
    return co


def triangle_data(mesh):
    mesh.calc_loop_triangles()
    tri = np.empty((len(mesh.loop_triangles), 3), dtype=np.int32)
    mesh.loop_triangles.foreach_get("vertices", tri.ravel())
    polygon = np.empty(len(tri), dtype=np.int32)
    mesh.loop_triangles.foreach_get("polygon_index", polygon)
    return tri, polygon


def point_mesh(name, points, normals=None, triangle_ids=None):
    mesh = bpy.data.meshes.new(name)
    mesh.vertices.add(len(points))
    mesh.vertices.foreach_set("co", np.asarray(points, np.float32).ravel())
    if normals is not None:
        attr = mesh.attributes.new("scan_normal", "FLOAT_VECTOR", "POINT")
        attr.data.foreach_set("vector", np.asarray(normals, np.float32).ravel())
    if triangle_ids is not None:
        attr = mesh.attributes.new("source_triangle", "INT", "POINT")
        attr.data.foreach_set("value", np.asarray(triangle_ids, np.int32))
    mesh.update()
    return mesh


def named(nodes, name, data_type="FLOAT_VECTOR"):
    node = nodes.new("GeometryNodeInputNamedAttribute")
    node.data_type = data_type
    node.inputs["Name"].default_value = name
    return node.outputs["Attribute"]


def math_node(nodes, links, op, a, b=None):
    n = nodes.new("ShaderNodeMath")
    n.operation = op
    if isinstance(a, (int, float)):
        n.inputs[0].default_value = a
    else:
        links.new(a, n.inputs[0])
    if b is not None:
        if isinstance(b, (int, float)):
            n.inputs[1].default_value = b
        else:
            links.new(b, n.inputs[1])
    return n.outputs[0]


def vector_node(nodes, links, op, a, b=None):
    n = nodes.new("ShaderNodeVectorMath")
    n.operation = op
    links.new(a, n.inputs[0])
    if b is not None:
        socket = n.inputs[3] if op == "SCALE" else n.inputs[1]
        if isinstance(b, (int, float)):
            socket.default_value = b
        else:
            links.new(b, socket)
    return (
        n.outputs["Value"] if op in {"DOT_PRODUCT", "DISTANCE", "LENGTH"} else n.outputs["Vector"]
    )


def object_geometry(nodes, obj):
    n = nodes.new("GeometryNodeObjectInfo")
    n.transform_space = "RELATIVE"
    n.inputs["Object"].default_value = obj
    n.inputs["As Instance"].default_value = False
    return n.outputs["Geometry"]


def store_attr(nodes, links, geometry, name, value, data_type="FLOAT_VECTOR"):
    node = nodes.new("GeometryNodeStoreNamedAttribute")
    node.data_type = data_type
    node.domain = "POINT"
    node.inputs["Name"].default_value = name
    links.new(geometry, node.inputs["Geometry"])
    links.new(value, node.inputs["Value"])
    return node.outputs["Geometry"]


def node_tree(name):
    tree = bpy.data.node_groups.new(name, "GeometryNodeTree")
    tree.interface.new_socket(name="Geometry", in_out="INPUT", socket_type="NodeSocketGeometry")
    tree.interface.new_socket(name="Geometry", in_out="OUTPUT", socket_type="NodeSocketGeometry")
    entry = tree.nodes.new("NodeGroupInput")
    out = tree.nodes.new("NodeGroupOutput")
    return tree, entry.outputs["Geometry"], out.inputs["Geometry"]


def bake(obj, modifier, context):
    """Evaluate and copy native output without relying on area/operator context."""
    context.view_layer.update()
    evaluated = obj.evaluated_get(context.evaluated_depsgraph_get())
    mesh = bpy.data.meshes.new_from_object(
        evaluated, preserve_all_data_layers=True, depsgraph=context.evaluated_depsgraph_get()
    )
    old = obj.data
    obj.modifiers.remove(modifier)
    obj.data = mesh
    if old.users == 0:
        bpy.data.meshes.remove(old)
    return mesh


def bake_nodes(obj, tree, context):
    modifier = obj.modifiers.new("Virtual acquisition", "NODES")
    modifier_name = modifier.name
    modifier.node_group = tree
    try:
        return bake(obj, modifier, context)
    finally:
        remaining = obj.modifiers.get(modifier_name)
        if remaining is not None:
            obj.modifiers.remove(remaining)
        if tree.users == 0:
            bpy.data.node_groups.remove(tree)


def ray_acquire(cloud, source, h, options, context):
    """Teleport a scanner just off each proposed region and record actual BVH hits.

    This is the local, orientation-aware impossible scanner described in the
    reference concept, rather than a simulated fixed-camera LiDAR instrument.
    Misses are removed; there is no invented hit or uniform-sampling fallback.
    """
    tree, geo, output = node_tree("M2S - orientation-aware ray acquisition")
    nodes, links = tree.nodes, tree.links
    pos = nodes.new("GeometryNodeInputPosition").outputs["Position"]
    normal = named(nodes, "scan_normal")
    offset = vector_node(nodes, links, "SCALE", normal, h * options.standoff_voxels)
    origin = vector_node(nodes, links, "ADD", pos, offset)
    direction = vector_node(nodes, links, "SCALE", normal, -1.0)
    ray = nodes.new("GeometryNodeRaycast")
    ray.data_type = "INT"
    # Newer Blender versions expose interpolation as a menu input socket.
    interpolation = ray.inputs.get("Interpolation")
    if interpolation is not None:
        interpolation.default_value = "Interpolated"
    else:
        ray.mapping = "INTERPOLATED"
    links.new(object_geometry(nodes, source), ray.inputs["Target Geometry"])
    links.new(named(nodes, "source_triangle", "INT"), ray.inputs["Attribute"])
    links.new(origin, ray.inputs["Source Position"])
    links.new(direction, ray.inputs["Ray Direction"])
    ray.inputs["Ray Length"].default_value = h * options.standoff_voxels * 2.1
    geo = store_attr(nodes, links, geo, "scanner_position", origin)
    geo = store_attr(nodes, links, geo, "ray_direction", direction)
    geo = store_attr(nodes, links, geo, "_scan_hit", ray.outputs["Is Hit"], "BOOLEAN")
    geo = store_attr(nodes, links, geo, "_scan_hit_position", ray.outputs["Hit Position"])
    # Capture every ray field before replacing the normal used to cast that ray.
    geo = store_attr(nodes, links, geo, "_scan_hit_normal", ray.outputs["Hit Normal"])
    geo = store_attr(nodes, links, geo, "_scan_hit_triangle", ray.outputs["Attribute"], "INT")
    delete = nodes.new("GeometryNodeDeleteGeometry")
    delete.domain = "POINT"
    links.new(geo, delete.inputs["Geometry"])
    links.new(
        math_node(nodes, links, "LESS_THAN", named(nodes, "_scan_hit", "BOOLEAN"), 0.5),
        delete.inputs["Selection"],
    )
    geo = store_attr(
        nodes,
        links,
        delete.outputs["Geometry"],
        "source_triangle",
        named(nodes, "_scan_hit_triangle", "INT"),
        "INT",
    )
    geo = store_attr(nodes, links, geo, "scan_normal", named(nodes, "_scan_hit_normal"))
    setpos = nodes.new("GeometryNodeSetPosition")
    links.new(geo, setpos.inputs["Geometry"])
    links.new(named(nodes, "_scan_hit_position"), setpos.inputs["Position"])
    links.new(setpos.outputs["Geometry"], output)
    bake_nodes(cloud, tree, context)
    for attr in list(cloud.data.attributes):
        if attr.name.startswith("_scan_hit"):
            cloud.data.attributes.remove(attr)


def volume_reconstruct(cloud, output_obj, h, options, context, source=None, open_panel=False):
    """The volume is built from acquired POINTS, never the source mesh."""
    tree, _, output = node_tree("M2S - observation volume reconstruction")
    nodes, links = tree.nodes, tree.links
    to_points = nodes.new("GeometryNodeMeshToPoints")
    to_points.mode = "VERTICES"
    links.new(object_geometry(nodes, cloud), to_points.inputs["Mesh"])
    volume = nodes.new("GeometryNodePointsToVolume")
    resolution = volume.inputs.get("Resolution Mode")
    if resolution is not None:
        resolution.default_value = "Size"
    else:
        volume.resolution_mode = "VOXEL_SIZE"
    volume.inputs["Voxel Size"].default_value = h
    volume.inputs["Radius"].default_value = h * options.radius_voxels
    volume.inputs["Density"].default_value = 1.0
    links.new(to_points.outputs["Points"], volume.inputs["Points"])
    surface = nodes.new("GeometryNodeVolumeToMesh")
    resolution = surface.inputs.get("Resolution Mode")
    if resolution is not None:
        resolution.default_value = "Grid"
    else:
        surface.resolution_mode = "GRID"
    surface.inputs["Threshold"].default_value = 0.5
    # Adaptive reduction is a separate, feature-weighted stage. Turning on
    # OpenVDB adaptivity here can reduce a planar panel to a few giant faces.
    surface.inputs["Adaptivity"].default_value = 0.0
    links.new(volume.outputs["Volume"], surface.inputs["Volume"])
    geo = surface.outputs["Mesh"]
    if options.output == "SURFACE":
        cloud_geo = object_geometry(nodes, cloud)
        nearest = nodes.new("GeometryNodeSampleNearest")
        nearest.domain = "POINT"
        links.new(cloud_geo, nearest.inputs["Geometry"])
        sample_pos = nodes.new("GeometryNodeSampleIndex")
        sample_pos.data_type = "FLOAT_VECTOR"
        sample_pos.domain = "POINT"
        sample_normal = nodes.new("GeometryNodeSampleIndex")
        sample_normal.data_type = "FLOAT_VECTOR"
        sample_normal.domain = "POINT"
        for node in (sample_pos, sample_normal):
            links.new(cloud_geo, node.inputs["Geometry"])
            links.new(nearest.outputs["Index"], node.inputs["Index"])
        pos = nodes.new("GeometryNodeInputPosition").outputs["Position"]
        links.new(pos, sample_pos.inputs["Value"])
        links.new(named(nodes, "scan_normal"), sample_normal.inputs["Value"])
        measured_position = sample_pos.outputs["Value"]
        measured_normal = sample_normal.outputs["Value"]
        if options.fidelity == "SOURCE" and source is not None:
            # Source-fidelity mode also uses the exact reference to disambiguate
            # the two envelope skins at a crease. The volume itself still comes
            # exclusively from observations. Cloud-only mode has no such lookup.
            reference = object_geometry(nodes, source)
            proximity = nodes.new("GeometryNodeProximity")
            proximity.target_element = "FACES"
            links.new(reference, proximity.inputs["Geometry"])
            normal_at_surface = nodes.new("GeometryNodeSampleNearestSurface")
            normal_at_surface.data_type = "FLOAT_VECTOR"
            links.new(reference, normal_at_surface.inputs["Mesh"])
            links.new(
                nodes.new("GeometryNodeInputNormal").outputs["Normal"],
                normal_at_surface.inputs["Value"],
            )
            measured_position = proximity.outputs["Position"]
            measured_normal = normal_at_surface.outputs["Value"]
        delta = vector_node(nodes, links, "SUBTRACT", pos, measured_position)
        measured_normal = vector_node(nodes, links, "NORMALIZE", measured_normal)
        signed = vector_node(nodes, links, "DOT_PRODUCT", delta, measured_normal)
        delete = nodes.new("GeometryNodeDeleteGeometry")
        delete.domain = "FACE"
        delete.mode = "ALL"
        links.new(geo, delete.inputs["Geometry"])
        reject = math_node(nodes, links, "LESS_THAN", signed, 0.0)
        if open_panel and options.fidelity == "SOURCE":
            # Cut the volumetric overhang around an open perimeter BEFORE
            # projecting it onto a zero-thickness panel. Otherwise an entire
            # rim folds onto the same boundary and creates overlapping fans.
            normal_delta = vector_node(nodes, links, "SCALE", measured_normal, signed)
            tangent_delta = vector_node(nodes, links, "SUBTRACT", delta, normal_delta)
            tangent_length = vector_node(nodes, links, "LENGTH", tangent_delta)
            overhang = math_node(nodes, links, "GREATER_THAN", tangent_length, h * 0.45)
            reject = math_node(nodes, links, "MAXIMUM", reject, overhang)
        links.new(reject, delete.inputs["Selection"])
        geo = delete.outputs["Geometry"]
        if options.fidelity == "SURFELS":
            # Orthogonal projection onto the nearest measured tangent plane.
            # This is an approximate cloud-only fit, not CAD or Poisson fitting.
            displacement = vector_node(
                nodes,
                links,
                "SCALE",
                sample_normal.outputs["Value"],
                math_node(nodes, links, "MULTIPLY", signed, -1.0),
            )
            setpos = nodes.new("GeometryNodeSetPosition")
            links.new(geo, setpos.inputs["Geometry"])
            links.new(displacement, setpos.inputs["Offset"])
            geo = setpos.outputs["Geometry"]
    links.new(geo, output)
    bake_nodes(output_obj, tree, context)


def fidelity_and_detail(obj, source, features, h, options, context, fit=True):
    tree, geo, output = node_tree("M2S - measured shape and feature recovery")
    nodes, links = tree.nodes, tree.links
    pos = nodes.new("GeometryNodeInputPosition").outputs["Position"]
    reference = object_geometry(nodes, source)
    sample = nodes.new("GeometryNodeSampleNearestSurface")
    sample.data_type = "FLOAT"
    links.new(reference, sample.inputs["Mesh"])
    links.new(named(nodes, "reference_curvature", "FLOAT"), sample.inputs["Value"])
    curvature = sample.outputs["Value"]
    geo = store_attr(nodes, links, geo, "scan_curvature", curvature, "FLOAT")
    weight = math_node(nodes, links, "MULTIPLY", curvature, 0.7)
    if fit and options.output == "SURFACE" and options.fidelity == "SOURCE":
        proximity = nodes.new("GeometryNodeProximity")
        proximity.target_element = "FACES"
        links.new(object_geometry(nodes, source), proximity.inputs["Geometry"])
        setpos = nodes.new("GeometryNodeSetPosition")
        links.new(geo, setpos.inputs["Geometry"])
        links.new(proximity.outputs["Position"], setpos.inputs["Position"])
        geo = setpos.outputs["Geometry"]
    if features is not None:
        proximity = nodes.new("GeometryNodeProximity")
        proximity.target_element = "EDGES"
        links.new(object_geometry(nodes, features), proximity.inputs["Geometry"])
        # Save weights before moving vertices. Fields in subsequent nodes are
        # otherwise reevaluated on modified geometry by Geometry Nodes.
        distance = proximity.outputs["Distance"]
        weight = math_node(
            nodes,
            links,
            "MAXIMUM",
            math_node(
                nodes, links, "SUBTRACT", 1.0, math_node(nodes, links, "DIVIDE", distance, h * 8.0)
            ),
            0.0,
        )
        weight = math_node(
            nodes, links, "MAXIMUM", weight, math_node(nodes, links, "MULTIPLY", curvature, 0.7)
        )
        geo = store_attr(nodes, links, geo, "scan_detail", weight, "FLOAT")
        if (
            fit
            and options.output == "SURFACE"
            and options.fidelity != "NONE"
            and options.snap_radius > 0
        ):
            setpos = nodes.new("GeometryNodeSetPosition")
            links.new(geo, setpos.inputs["Geometry"])
            links.new(
                math_node(nodes, links, "LESS_THAN", distance, h * options.snap_radius),
                setpos.inputs["Selection"],
            )
            links.new(proximity.outputs["Position"], setpos.inputs["Position"])
            geo = setpos.outputs["Geometry"]
    geo = store_attr(nodes, links, geo, "scan_detail", weight, "FLOAT")
    links.new(geo, output)
    bake_nodes(obj, tree, context)


def guided_observation_fit(obj, cloud, h, options, context):
    if options.observed_fit <= 0:
        return
    tree, geo, output = node_tree("mesh2scan - bounded observation fit")
    nodes, links = tree.nodes, tree.links
    target = object_geometry(nodes, cloud)
    nearest = nodes.new("GeometryNodeSampleNearest")
    nearest.domain = "POINT"
    links.new(target, nearest.inputs["Geometry"])
    pos = nodes.new("GeometryNodeInputPosition").outputs["Position"]
    sampled = []
    for value in (pos, named(nodes, "scan_normal")):
        node = nodes.new("GeometryNodeSampleIndex")
        node.data_type = "FLOAT_VECTOR"
        node.domain = "POINT"
        links.new(target, node.inputs["Geometry"])
        links.new(nearest.outputs["Index"], node.inputs["Index"])
        links.new(value, node.inputs["Value"])
        sampled.append(node.outputs["Value"])
    delta = vector_node(nodes, links, "SUBTRACT", pos, sampled[0])
    normal = vector_node(nodes, links, "NORMALIZE", sampled[1])
    signed = vector_node(nodes, links, "DOT_PRODUCT", delta, normal)
    bounded = math_node(
        nodes, links, "MINIMUM", math_node(nodes, links, "MAXIMUM", signed, -h * 0.15), h * 0.15
    )
    factor = math_node(nodes, links, "MULTIPLY", bounded, -options.observed_fit)
    offset = vector_node(nodes, links, "SCALE", normal, factor)
    shade_normal = nodes.new("GeometryNodeInputNormal").outputs["Normal"]
    agreement = math_node(
        nodes,
        links,
        "GREATER_THAN",
        vector_node(nodes, links, "DOT_PRODUCT", shade_normal, normal),
        0.95,
    )
    interior = math_node(nodes, links, "LESS_THAN", named(nodes, "scan_detail", "FLOAT"), 0.5)
    nearby = math_node(nodes, links, "LESS_THAN", vector_node(nodes, links, "LENGTH", delta), h * 3)
    selection = math_node(
        nodes, links, "MULTIPLY", math_node(nodes, links, "MULTIPLY", agreement, interior), nearby
    )
    setpos = nodes.new("GeometryNodeSetPosition")
    links.new(geo, setpos.inputs["Geometry"])
    links.new(selection, setpos.inputs["Selection"])
    links.new(offset, setpos.inputs["Offset"])
    links.new(setpos.outputs["Geometry"], output)
    bake_nodes(obj, tree, context)


def clean_triangles(obj, h, options, context, source=None):
    weld = obj.modifiers.new("Merge coincident scan vertices", "WELD")
    weld.mode = "ALL"
    weld.merge_threshold = max(h * 0.001, 1e-9)
    bake(obj, weld, context)
    tri = obj.modifiers.new("Triangle scan", "TRIANGULATE")
    tri.quad_method = "BEAUTY"
    tri.ngon_method = "BEAUTY"
    bake(obj, tri, context)
    tree, geo, output = node_tree("M2S - remove collapsed samples")
    nodes, links = tree.nodes, tree.links
    area = nodes.new("GeometryNodeInputMeshFaceArea").outputs["Area"]
    delete = nodes.new("GeometryNodeDeleteGeometry")
    delete.domain = "FACE"
    delete.mode = "ALL"
    links.new(geo, delete.inputs["Geometry"])
    invalid = math_node(nodes, links, "LESS_THAN", area, max(h * h * 1e-6, 1e-14))
    if options.output == "SURFACE" and options.fidelity == "SOURCE" and source is not None:
        reference = object_geometry(nodes, source)
        normal = nodes.new("GeometryNodeInputNormal").outputs["Normal"]
        measured = nodes.new("GeometryNodeSampleNearestSurface")
        measured.data_type = "FLOAT_VECTOR"
        links.new(reference, measured.inputs["Mesh"])
        links.new(normal, measured.inputs["Value"])
        agreement = vector_node(nodes, links, "DOT_PRODUCT", normal, measured.outputs["Value"])
        reversed_face = math_node(nodes, links, "LESS_THAN", agreement, 0.001)
        invalid = math_node(nodes, links, "MAXIMUM", invalid, reversed_face)
    links.new(invalid, delete.inputs["Selection"])
    neighbors = nodes.new("GeometryNodeInputMeshVertexNeighbors")
    loose = nodes.new("GeometryNodeDeleteGeometry")
    loose.domain = "POINT"
    links.new(delete.outputs["Geometry"], loose.inputs["Geometry"])
    links.new(
        math_node(nodes, links, "LESS_THAN", neighbors.outputs["Face Count"], 0.5),
        loose.inputs["Selection"],
    )
    links.new(loose.outputs["Geometry"], output)
    bake_nodes(obj, tree, context)


def adaptive_reduce(obj, target, options, context):
    count = len(obj.data.polygons)
    if not count or (options.resolution == "VOXEL" and options.adaptive <= 0):
        return
    if options.resolution == "TARGET":
        # Target is a budget, not a command to add uninformative subdivisions.
        ratio = min(1.0, target / count)
    else:
        ratio = 1.0 - options.adaptive * 0.65
    if ratio >= 0.995:
        return
    group = obj.vertex_groups.new(name="Scan detail protection")
    attr = obj.data.attributes.get("scan_detail")
    if attr is not None:
        weights = np.empty(len(obj.data.vertices), np.float32)
        attr.data.foreach_get("value", weights)
        weights = np.clip(weights, 0, 1)
        bins = np.rint(weights * 15).astype(np.uint8)
        for level in range(1, 16):
            ids = np.flatnonzero(bins == level)
            if len(ids):
                group.add(ids.tolist(), level / 15.0, "REPLACE")
    modifier = obj.modifiers.new("Adaptive scan density", "DECIMATE")
    modifier.decimate_type = "COLLAPSE"
    modifier.ratio = ratio
    modifier.use_collapse_triangulate = True
    modifier.vertex_group = group.name
    # Blender interprets large vertex weights as regions to decimate. Our
    # attribute marks regions to protect, so its influence must be inverted.
    modifier.invert_vertex_group = True
    group_name = group.name
    modifier.vertex_group_factor = options.adaptive * 60.0
    bake(obj, modifier, context)
    remaining = obj.vertex_groups.get(group_name)
    if remaining is not None:
        obj.vertex_groups.remove(remaining)


def smooth_shading(obj, options, context):
    tree, geo, output = node_tree("M2S - crease-aware display normals")
    nodes, links = tree.nodes, tree.links
    faces = nodes.new("GeometryNodeSetShadeSmooth")
    faces.domain = "FACE"
    faces.inputs["Shade Smooth"].default_value = True
    links.new(geo, faces.inputs["Geometry"])
    edges = nodes.new("GeometryNodeSetShadeSmooth")
    edges.domain = "EDGE"
    angle = nodes.new("GeometryNodeInputMeshEdgeAngle").outputs["Unsigned Angle"]
    links.new(faces.outputs["Geometry"], edges.inputs["Geometry"])
    links.new(
        math_node(nodes, links, "LESS_THAN", angle, options.crease_angle),
        edges.inputs["Shade Smooth"],
    )
    links.new(edges.outputs["Geometry"], output)
    bake_nodes(obj, tree, context)


def remove_overlapping_topology(obj, context):
    """Remove excess fitted faces on edges shared by more than two triangles.

    Projection of a volume rim onto an open panel can overlap small face fans.
    Keep the two largest incident faces, instead of shipping a nonmanifold fan.
    This can leave an open fitting crack, which the capture report exposes.
    It is not a general self-intersection or manufacturing repair.
    """
    tri, _ = triangle_data(obj.data)
    if not len(tri):
        return 0
    n = len(tri)
    edges = np.concatenate((tri[:, [0, 1]], tri[:, [1, 2]], tri[:, [2, 0]]))
    edges.sort(axis=1)
    keys = edges[:, 0].astype(np.uint64) * np.uint64(len(obj.data.vertices)) + edges[:, 1]
    del edges
    unique, counts = np.unique(keys, return_counts=True)
    bad = unique[counts > 2]
    if not len(bad):
        return 0
    ids = np.flatnonzero(np.isin(keys, bad))
    order = np.argsort(keys[ids])
    ids = ids[order]
    selected_keys = keys[ids]
    starts = np.r_[0, np.flatnonzero(selected_keys[1:] != selected_keys[:-1]) + 1]
    ends = np.r_[starts[1:], len(ids)]
    co = mesh_coords(obj.data)
    p = co[tri]
    areas = np.linalg.norm(np.cross(p[:, 1] - p[:, 0], p[:, 2] - p[:, 0]), axis=1)
    drop = np.zeros(n, bool)
    for start, end in zip(starts, ends):
        faces = ids[start:end] % n
        faces = faces[~drop[faces]]
        if len(faces) > 2:
            drop[faces[np.argsort(areas[faces])[: len(faces) - 2]]] = True
    number = int(drop.sum())
    if not number:
        return 0
    attr = obj.data.attributes.new("_scan_drop_overlap", "BOOLEAN", "FACE")
    attr.data.foreach_set("value", drop)
    tree, geo, output = node_tree("M2S - discard overlapping fitted face fans")
    delete = tree.nodes.new("GeometryNodeDeleteGeometry")
    delete.domain = "FACE"
    delete.mode = "ALL"
    tree.links.new(geo, delete.inputs["Geometry"])
    tree.links.new(named(tree.nodes, "_scan_drop_overlap", "BOOLEAN"), delete.inputs["Selection"])
    tree.links.new(delete.outputs["Geometry"], output)
    bake_nodes(obj, tree, context)
    attr = obj.data.attributes.get("_scan_drop_overlap")
    if attr is not None:
        obj.data.attributes.remove(attr)
    return number


def geometry_stats(mesh):
    tri, _ = triangle_data(mesh)
    if not len(tri):
        return {
            "vertices": len(mesh.vertices),
            "triangles": 0,
            "boundary_edges": 0,
            "nonmanifold_edges": 0,
            "degenerate_triangles": 0,
        }
    # Vectorized edge incidence is substantially cheaper than a second BMesh
    # for a multi-million-triangle output. Exact edge topology, not a print audit.
    edges = np.concatenate((tri[:, [0, 1]], tri[:, [1, 2]], tri[:, [2, 0]]))
    edges.sort(axis=1)
    encoded = edges[:, 0].astype(np.uint64) * np.uint64(len(mesh.vertices)) + edges[:, 1]
    del edges
    _, counts = np.unique(encoded, return_counts=True)
    co = mesh_coords(mesh)
    p = co[tri]
    twice_area = np.linalg.norm(np.cross(p[:, 1] - p[:, 0], p[:, 2] - p[:, 0]), axis=1)
    return {
        "vertices": len(co),
        "triangles": len(tri),
        "boundary_edges": int(np.count_nonzero(counts == 1)),
        "nonmanifold_edges": int(np.count_nonzero(counts > 2)),
        "degenerate_triangles": int(np.count_nonzero(twice_area <= 1e-14)),
    }


class ScanJob:
    def __init__(self, context, source, options):
        self.context = context
        self.source = source
        self.options = options
        self.temporary = []
        self.persistent = []
        self.result = None
        self.cloud = None
        self.report = {}
        self.started = time.perf_counter()
        self.view_layer = context.view_layer
        self.was_hidden = source.hide_get(view_layer=self.view_layer)
        self.completed = False

    def new_object(self, name, mesh):
        obj = bpy.data.objects.new(name, mesh)
        self.context.scene.collection.objects.link(obj)
        self.temporary.append(obj)
        return obj

    def cleanup(self):
        for obj in reversed(self.temporary):
            if obj in self.persistent:
                continue
            try:
                mesh = obj.data
                bpy.data.objects.remove(obj, do_unlink=True)
                if mesh.users == 0:
                    bpy.data.meshes.remove(mesh)
            except (ReferenceError, RuntimeError):
                pass
        self.temporary.clear()
        if not self.completed:
            try:
                self.source.hide_set(self.was_hidden, view_layer=self.view_layer)
            except (ReferenceError, RuntimeError):
                pass

    def stages(self):
        try:
            yield from self._stages()
        finally:
            self.cleanup()

    def _stages(self):
        ctx, original, opts = self.context, self.source, self.options
        if original.type != "MESH" or not len(original.data.polygons):
            raise ScanError("Select a mesh with surface faces in Object Mode.")
        if not is_guided(opts) and not bpy.app.build_options.openvdb:
            raise ScanError(
                "This Blender build has no OpenVDB support. Use an official Blender build."
            )
        if abs(original.matrix_world.determinant()) < 1e-30:
            raise ScanError("The object has a zero scale axis. Restore a nonzero scale first.")
        yield (0.02, "Reading the component in its original coordinate system")
        # Hidden originals must be evaluated as visible while taking a new
        # snapshot, so a dense capture after Preview sees the current modifiers.
        if opts.resolution == "TARGET" and not opts.preplanned:
            from .batch import measure_component, allocate_budget

            measured = measure_component(ctx, original, opts)
            plan = fit_plan([measured], opts.target_triangles, opts, allocate_budget)
            if not plan["fits"]:
                raise ScanError(
                    f"Estimated capture RAM {plan['estimated_peak_gib']:.1f} GiB exceeds the safe allowance {plan['memory_limit_gib']:.1f} GiB. "
                    "Choose Fit available RAM, Guided surface, or a smaller target. This is RAM, not disk space."
                )
            opts = replace(opts, target_triangles=plan["effective_triangles"])
            self.options = opts
            self.plan = plan
            yield (
                0.04,
                f"RAM plan: {plan['effective_triangles']:,} triangles, about {plan['estimated_peak_gib']:.1f} GiB",
            )
        original.hide_set(False)
        ctx.view_layer.update()
        depsgraph = ctx.evaluated_depsgraph_get()
        if opts.use_modifiers:
            mesh = bpy.data.meshes.new_from_object(
                original.evaluated_get(depsgraph),
                preserve_all_data_layers=True,
                depsgraph=depsgraph,
            )
        else:
            mesh = original.data.copy()
        snapshot = self.new_object("_M2S reference", mesh)
        # Normalize large/small imported coordinates without changing proportions.
        co = mesh_coords(mesh).astype(np.float64)
        world = np.array(original.matrix_world, np.float64)
        co = co @ world[:3, :3].T + world[:3, 3]
        low, high = co.min(axis=0), co.max(axis=0)
        center, scale = (low + high) * 0.5, float((high - low).max())
        if not np.isfinite(co).all() or scale <= 1e-20:
            raise ScanError("The mesh has invalid coordinates or no measurable size.")
        co = (co - center) / scale
        mesh.vertices.foreach_set("co", co.astype(np.float32).ravel())
        if original.matrix_world.determinant() < 0:
            mesh.flip_normals()
        mesh.update()
        frame = Matrix.Translation(Vector(center)) @ Matrix.Scale(scale, 4)
        if len(mesh.polygons) > 3_000_000:
            raise ScanError(
                "Isolate a component or simplify the reference below 3 million faces before capture."
            )
        bm = bmesh.new()
        bm.from_mesh(mesh)
        if opts.clean_source:
            bmesh.ops.remove_doubles(bm, verts=list(bm.verts), dist=1e-7)
            bmesh.ops.dissolve_degenerate(bm, edges=list(bm.edges), dist=1e-10)
            loose = [v for v in bm.verts if not v.link_faces]
            if loose:
                bmesh.ops.delete(bm, geom=loose, context="VERTS")
        bmesh.ops.recalc_face_normals(bm, faces=list(bm.faces))
        bm.verts.index_update()
        bm.faces.index_update()
        boundary = sum(e.is_boundary for e in bm.edges)
        nonmanifold = sum(len(e.link_faces) > 2 for e in bm.edges)
        if not boundary and bm.calc_volume(signed=True) < 0:
            bmesh.ops.reverse_faces(bm, faces=list(bm.faces))
            bm.normal_update()
        # Feature measurements are sampled from geometric creases/boundaries;
        # original rendering triangle edges are deliberately not all preserved.
        feature_edges, feature_normals, perimeter_edges = [], [], []
        curvature = np.zeros(len(bm.faces), np.float32)
        for edge in bm.edges:
            if edge.is_boundary:
                perimeter_edges.append([edge.verts[0].index, edge.verts[1].index])
            angle = edge.calc_face_angle(0.0) if len(edge.link_faces) == 2 else 0.0
            # Sharp creases get a local distance band, not a high score over
            # an entire flat source polygon. Continuous curvature stays broad.
            if angle < opts.crease_angle:
                for face in edge.link_faces:
                    curvature[face.index] = max(curvature[face.index], angle)
            if (
                edge.is_boundary
                or angle >= opts.crease_angle
                or (opts.marked_edges and not edge.smooth)
            ):
                feature_edges.append([edge.verts[0].index, edge.verts[1].index])
                n = sum((face.normal for face in edge.link_faces), Vector())
                if n.length < 1e-9 and edge.link_faces:
                    n = edge.link_faces[0].normal.copy()
                feature_normals.append(tuple(n.normalized()))
        bm.to_mesh(mesh)
        bm.free()
        feature_co = mesh_coords(mesh)
        original_triangles, _ = triangle_data(mesh)
        input_triangles = len(original_triangles)
        raw_bvh = (
            BVHTree.FromPolygons(
                feature_co.tolist(), original_triangles.tolist(), all_triangles=True
            )
            if opts.surface_mode == "SMOOTH"
            else None
        )
        yield (0.07, "Preparing a smooth reference with protected creases and boundaries")
        reference_mesh, smooth_report = curved_reference(mesh, feature_edges, curvature, opts)
        if reference_mesh != mesh:
            snapshot.data = reference_mesh
            if mesh.users == 0:
                bpy.data.meshes.remove(mesh)
            mesh = reference_mesh
        protected = mesh.attributes.get("reference_protected")
        if (
            protected
            and opts.surface_mode == "SMOOTH"
            and opts.fairing_iterations > 0
            and smooth_report["used_levels"] > 0
        ):
            flags = np.empty(len(mesh.vertices), bool)
            protected.data.foreach_get("value", flags)
            group = snapshot.vertex_groups.new(name="Interior fairing")
            ids = np.flatnonzero(~flags)
            if len(ids):
                group.add(ids.tolist(), 1.0, "REPLACE")
                modifier = snapshot.modifiers.new("Guarded curve fairing", "SMOOTH")
                modifier.factor = 0.18
                modifier.iterations = opts.fairing_iterations
                modifier.vertex_group = group.name
                mesh = bake(snapshot, modifier, ctx)
            snapshot.vertex_groups.clear()
            smooth_report["fairing_iterations"] = opts.fairing_iterations
        co = mesh_coords(mesh)
        reference_curvature = np.empty(len(mesh.polygons), np.float32)
        mesh.attributes["reference_curvature"].data.foreach_get("value", reference_curvature)
        triangles, polygons = triangle_data(mesh)
        if not len(triangles):
            raise ScanError("No nondegenerate surface remains after cleanup.")
        p = co[triangles]
        cross = np.cross(p[:, 1] - p[:, 0], p[:, 2] - p[:, 0])
        twice_area = np.linalg.norm(cross, axis=1)
        valid = twice_area > 1e-14
        triangles, polygons, p, cross, twice_area = (
            x[valid] for x in (triangles, polygons, p, cross, twice_area)
        )
        area = twice_area * 0.5
        total_area = float(area.sum())
        normals = cross / twice_area[:, None]
        target = max(500, int(opts.target_triangles))
        # Native voxel cells on arbitrary slopes need ~2.8 triangles per h² of
        # surface area. Capture above the desired budget, then simplify adaptively.
        reserve = 1.0 + opts.adaptive * 1.8
        h = (
            math.sqrt(2.8 * total_area / (target * reserve))
            if opts.resolution == "TARGET"
            else opts.voxel_world / scale
        )
        if not math.isfinite(h) or h <= 1e-7:
            raise ScanError("Voxel size is too small. Increase it or isolate a smaller component.")
        expected = 2.8 * total_area / (h * h)
        sample_count = max(1_000, int(expected * opts.samples_per_triangle))
        if is_guided(opts):
            sample_count = min(sample_count, opts.observation_cap)
        if sample_count > 100_000_000:
            raise ScanError(
                "This component requires over 100 million observations. Split it into panels or lower the shared budget."
            )
        # Include input and worst-case two-sided envelope, not only the final mesh.
        estimate_gb = (
            (
                (target * 680 + sample_count * 100 + len(triangles) * 600)
                if is_guided(opts)
                else (sample_count * 100 + expected * 1900 + len(triangles) * 800)
            )
            / 1024**3
            + opts.retained_gb
            + 0.25
        )
        if estimate_gb > opts.memory_gb:
            raise ScanError(
                f"Estimated peak memory {estimate_gb:.1f} GiB exceeds your {opts.memory_gb:.1f} GiB budget. "
                "Use Preview, fewer triangles, or a larger memory budget."
            )
        features = None
        if feature_edges:
            fm = bpy.data.meshes.new("_M2S measured feature lines")
            fm.from_pydata(feature_co.tolist(), feature_edges, [])
            fm.update()
            features = self.new_object("_M2S features", fm)
        self.report = {
            "version": "0.0.2",
            "blender": bpy.app.version_string,
            "source": original.name,
            "acquisition": opts.acquisition,
            "mesher": "GUIDED_SURFACE" if is_guided(opts) else "OBSERVATION_VOLUME",
            "plan": getattr(self, "plan", None),
            "output": opts.output,
            "fidelity": opts.fidelity,
            "target_triangles": target,
            "voxel_size_world": h * scale,
            "world_units_per_metre": 1.0 / max(ctx.scene.unit_settings.scale_length, 1e-30),
            "input_triangles": input_triangles,
            "scanner_reference": smooth_report,
            "smooth_max_offset_world": smooth_report["max_vertex_offset_normalized"] * scale,
            "component_id": opts.component_id,
            "input_boundary_edges": boundary,
            "input_nonmanifold_edges": nonmanifold,
            "feature_segments": len(feature_edges),
            "estimated_peak_gib": round(estimate_gb, 2),
            "original_matrix_world": world.tolist(),
            "notes": [
                "Synthetic measurements of source geometry; no additional real-world accuracy.",
                "Surface mode can have open boundaries; watertightness is reported, not promised.",
            ],
        }
        yield (0.12, f"Acquiring {sample_count:,} surface observations")
        rng = np.random.default_rng(opts.seed)
        tri_weight = 1.0 + opts.detail_boost * np.minimum(reference_curvature[polygons], 1.0)
        weights = area * tri_weight
        cdf = np.cumsum(weights, dtype=np.float64)
        # Minimum per-face observations provide coverage for very small parts.
        coverage_count = min(len(triangles), sample_count // 3)
        coverage_ids = (
            np.arange(len(triangles), dtype=np.int32)
            if coverage_count == len(triangles)
            else np.linspace(0, len(triangles) - 1, coverage_count).astype(np.int32)
        )
        random_count = sample_count - coverage_count
        ids = np.searchsorted(cdf, rng.random(random_count) * cdf[-1]).astype(np.int32)
        r = rng.random((random_count, 2), dtype=np.float32)
        u = np.sqrt(r[:, 0])
        a, b, c = 1 - u, u * (1 - r[:, 1]), u * r[:, 1]
        points = p[ids, 0] * a[:, None] + p[ids, 1] * b[:, None] + p[ids, 2] * c[:, None]
        points = np.concatenate((points, p[coverage_ids].mean(axis=1)))
        sample_normals = np.concatenate((normals[ids], normals[coverage_ids]))
        sample_ids = np.concatenate((ids, coverage_ids))
        # Explicit feature observations strengthen coverage at open perimeters
        # and crease intersections without preserving the source tessellation.
        if feature_edges:
            fe = np.asarray(feature_edges, np.int32)
            seg = feature_co[fe]
            lengths = np.linalg.norm(seg[:, 1] - seg[:, 0], axis=1)
            counts = np.maximum(2, np.ceil(lengths / (h * 0.75)).astype(np.int32) + 1)
            if int(counts.sum()) > 5_000_000:
                raise ScanError(
                    "Too many sharp edges at this resolution. Increase the crease angle or voxel size."
                )
            eids = np.repeat(np.arange(len(fe)), counts)
            start = np.repeat(np.cumsum(counts) - counts, counts)
            t = (np.arange(len(eids)) - start) / (counts[eids] - 1)
            ep = seg[eids, 0] * (1 - t[:, None]) + seg[eids, 1] * t[:, None]
            en = np.asarray(feature_normals, np.float32)[eids]
            points = np.concatenate((points, ep.astype(np.float32)))
            sample_normals = np.concatenate((sample_normals, en))
            # -1 explicitly identifies a crease/boundary observation in uniform
            # mode; ray mode replaces it with the actually hit triangle index.
            sample_ids = np.concatenate((sample_ids, np.full(len(ep), -1, np.int32)))
        proposed = len(points)
        cloud = self.new_object(
            original.name + " - Scan Points",
            point_mesh("Virtual observations", points, sample_normals, sample_ids),
        )
        self.cloud = cloud
        del points, sample_normals, sample_ids, p, cross, ids, r
        if opts.acquisition == "RAYS":
            yield (0.23, "Ray-casting from virtual local scanner positions")
            # Ray attribute lookup uses a triangulated reference and FACE IDs.
            tri_mod = snapshot.modifiers.new("Scanner reference triangles", "TRIANGULATE")
            bake(snapshot, tri_mod, ctx)
            attr = snapshot.data.attributes.get("source_triangle") or snapshot.data.attributes.new(
                "source_triangle", "INT", "FACE"
            )
            attr.data.foreach_set("value", np.arange(len(snapshot.data.polygons), dtype=np.int32))
            ray_acquire(cloud, snapshot, h, opts, ctx)
        if len(cloud.data.vertices) < 100:
            raise ScanError("Too few ray hits. Check normals or reduce scanner standoff.")
        cloud["vgc_cloud"] = True
        cloud["vgc_component_id"] = opts.component_id
        self.report["proposed_observations"] = proposed
        self.report["acquired_observations"] = len(cloud.data.vertices)
        self.report["ray_hit_fraction"] = (
            len(cloud.data.vertices) / proposed if opts.acquisition == "RAYS" else None
        )
        if is_guided(opts):
            yield (0.38, "Preparing connected boundaries and feature density")
            seed = self.new_object("_M2S guided seed", snapshot.data.copy())
            fidelity_and_detail(seed, snapshot, features, h, opts, ctx, fit=False)
            result = self.new_object(
                original.name + " - mesh2scan", bpy.data.meshes.new("Guided scan")
            )
            self.result = result
            for count, iteration, output_mesh in guided_mesh(seed, target, opts):
                if output_mesh is not None:
                    old_mesh = result.data
                    result.data = output_mesh
                    if old_mesh.users == 0:
                        bpy.data.meshes.remove(old_mesh)
                else:
                    yield (
                        0.4 + 0.35 * min(count / max(target, 1), 1),
                        f"Conforming detail refinement: {count:,} triangles",
                    )
            guided_observation_fit(result, cloud, h, opts, ctx)
            self.report["overlapping_faces_removed"] = 0
            self.report["volume_faces_before_fitting"] = None
            self.report["notes"].append(
                "Guided mode retains the reference connectivity and uses observations for a bounded tangent fit; it is not cloud-only volume reconstruction."
            )
            self.report["boundary_recovery"] = {
                "method": "CONFORMING_EDGE_REFINEMENT",
                "corners_inserted": 0,
            }
        else:
            yield (0.38, "Fusing acquired points into a sparse OpenVDB volume")
            empty = bpy.data.meshes.new("Virtual captured surface")
            result = self.new_object(original.name + " - mesh2scan", empty)
            self.result = result
            volume_reconstruct(
                cloud, result, h, opts, ctx, source=snapshot, open_panel=bool(boundary)
            )
            if not len(result.data.polygons):
                raise ScanError("No surface reconstructed. Increase sampling or capture radius.")
            self.report["volume_faces_before_fitting"] = len(result.data.polygons)
            yield (0.61, "Recovering measured shape, creases and panel boundaries")
            fidelity_and_detail(result, snapshot, features, h, opts, ctx)
            clean_triangles(result, h, opts, ctx, source=snapshot)
            self.report["overlapping_faces_removed"] = remove_overlapping_topology(result, ctx)
            yield (0.75, "Allocating triangle density to geometric detail")
            corner_reserve = (
                boundary_corner_count(feature_co, perimeter_edges)
                if opts.boundary_recovery and opts.output == "SURFACE" and opts.fidelity == "SOURCE"
                else 0
            )
            reduction_target = max(500, target - corner_reserve)
            adaptive_reduce(result, reduction_target, opts, ctx)
            clean_triangles(result, h, opts, ctx, source=snapshot)
            if (
                opts.boundary_recovery
                and opts.output == "SURFACE"
                and opts.fidelity == "SOURCE"
                and perimeter_edges
            ):
                self.report["boundary_recovery"] = recover_boundaries(
                    result, feature_co, perimeter_edges, h
                )
                clean_triangles(result, h, opts, ctx, source=snapshot)
            self.report["overlapping_faces_removed"] += remove_overlapping_topology(result, ctx)
            # Boundary fitting can add a few corners. Refresh fields after topology
            # edits without snapping interior vertices or fitting again.
        fidelity_and_detail(result, snapshot, features, h, opts, ctx, fit=False)
        add_diagnostic_colors(result.data)
        if not len(result.data.polygons):
            raise ScanError(
                "All scan faces collapsed. Try a finer voxel size or disable edge snapping."
            )
        yield (0.86, "Checking triangulation, open edges and sampled deviation")
        self.report.update(geometry_stats(result.data))
        self.report["topologically_closed"] = (
            self.report["boundary_edges"] == 0 and self.report["nonmanifold_edges"] == 0
        )
        self.report["self_intersections_checked"] = False
        # Face-centre deviations reveal chamfered edges that vertex-only tests
        # miss. This is sampled evidence, not a bound on every point of the mesh.
        source_bvh = BVHTree.FromObject(snapshot, ctx.evaluated_depsgraph_get())
        out_tri, _ = triangle_data(result.data)
        out_co = mesh_coords(result.data)
        indices = np.linspace(0, len(out_tri) - 1, min(4096, len(out_tri))).astype(np.int32)
        centers = out_co[out_tri[indices]].mean(axis=1)
        detail_attr = result.data.attributes.get("scan_detail")
        if detail_attr is not None:
            detail = np.empty(len(out_co), np.float32)
            detail_attr.data.foreach_get("value", detail)
            sampled_detail = detail[out_tri[indices]].mean(axis=1)
            pp = out_co[out_tri[indices]]
            sampled_area = (
                np.linalg.norm(np.cross(pp[:, 1] - pp[:, 0], pp[:, 2] - pp[:, 0]), axis=1)
                * 0.5
                * scale**2
            )
            self.report["sampled_density_allocation"] = {
                "samples": len(indices),
                "low_detail_mean_area_world_squared": float(
                    sampled_area[sampled_detail < 0.25].mean()
                )
                if np.any(sampled_detail < 0.25)
                else None,
                "high_detail_mean_area_world_squared": float(
                    sampled_area[sampled_detail > 0.75].mean()
                )
                if np.any(sampled_detail > 0.75)
                else None,
                "meaning": "Smaller mean triangle area means denser tessellation; sampled diagnostic, not a guarantee.",
            }
        deviation = (
            np.asarray([source_bvh.find_nearest(Vector(v))[3] for v in centers], np.float64) * scale
        )
        self.report["sampled_face_deviation_world"] = {
            "samples": len(deviation),
            "mean": float(deviation.mean()),
            "rms": float(np.sqrt(np.mean(deviation**2))),
            "p95": float(np.percentile(deviation, 95)),
            "maximum": float(deviation.max()),
        }
        if raw_bvh is not None:
            raw_deviation = (
                np.array([raw_bvh.find_nearest(Vector(v))[3] for v in centers], np.float64) * scale
            )
            self.report["sampled_original_mesh_deviation_world"] = {
                "samples": len(raw_deviation),
                "rms": float(np.sqrt(np.mean(raw_deviation**2))),
                "maximum": float(raw_deviation.max()),
            }
        self.report["final_triangle_budget_ratio"] = len(out_tri) / target
        del out_tri, out_co, centers, triangles, polygons, co, area, normals
        raw_bvh = None
        smooth_shading(result, opts, ctx)
        # Keep the captured pose. Blender decomposes matrix_world into transform
        # channels; compute coordinates against the resulting actual matrix so
        # rounding or a sheared parent cannot move the captured surface.
        result.matrix_world = original.matrix_world.copy()
        ctx.view_layer.update()
        local = result.matrix_world.inverted() @ frame
        result.data.transform(local)
        if local.determinant() < 0:
            result.data.flip_normals()
        result.data.update()
        cloud.matrix_world = frame
        self.report["seconds"] = round(time.perf_counter() - self.started, 3)
        if not is_guided(opts) and opts.output == "SURFACE" and boundary:
            self.report["notes"].append(
                "Open source panel: the negative observation envelope was removed. No print wall was added."
            )
        if opts.fidelity == "SOURCE" and opts.output == "SURFACE":
            self.report["notes"].append(
                "Fitted to the scanner reference. Smooth PN geometry is inferred; it is not a certified Class A or CAD surface."
            )
        if opts.output == "ENVELOPE":
            self.report["notes"].append(
                "Unsigned capture envelope adds resolution-dependent offsets around observations."
            )
        result["vgc_scan"] = True
        result["vgc_reference"] = original
        result["vgc_report"] = json.dumps(self.report)
        result["vgc_cloud_name"] = cloud.name if opts.keep_cloud else ""
        cloud["vgc_report"] = json.dumps(self.report)
        self.persistent.append(result)
        if opts.keep_cloud:
            self.persistent.append(cloud)
            cloud.hide_set(True)
            cloud.hide_render = True
        for obj in ctx.selected_objects:
            obj.select_set(False)
        result.select_set(True)
        ctx.view_layer.objects.active = result
        original.hide_set(opts.hide_source)
        self.completed = True
        yield (1.0, f"Complete: {len(result.data.polygons):,} triangles")


def export_cloud_ply(obj, path, scale_factor=1.0):
    """Little-endian, float64 XYZ / float32 normals with component and hit IDs.

    World coordinates retain component registration. A multiplier of 1000 can
    convert metre-valued coordinates to millimetres without changing ratios.
    """
    mesh = obj.data
    co = mesh_coords(mesh).astype(np.float64)
    matrix = np.array(obj.matrix_world, np.float64)
    co = (co @ matrix[:3, :3].T + matrix[:3, 3]) * scale_factor
    normals = np.zeros((len(co), 3), np.float32)
    attr = mesh.attributes.get("scan_normal")
    if attr:
        attr.data.foreach_get("vector", normals.ravel())
        normal_matrix = np.linalg.inv(matrix[:3, :3]).T
        normals = normals.astype(np.float64) @ normal_matrix.T
        lengths = np.linalg.norm(normals, axis=1)
        normals /= np.maximum(lengths[:, None], 1e-30)
    ids = np.full(len(co), -1, np.int32)
    attr = mesh.attributes.get("source_triangle")
    if attr:
        attr.data.foreach_get("value", ids)
    ray_data = all(mesh.attributes.get(name) for name in ("scanner_position", "ray_direction"))
    fields = [(n, "<f8") for n in ("x", "y", "z")] + [(n, "<f4") for n in ("nx", "ny", "nz")]
    fields += [("component_id", "<i4"), ("source_triangle", "<i4")]
    if ray_data:
        fields += [(n, "<f8") for n in ("scanner_x", "scanner_y", "scanner_z")]
        fields += [(n, "<f4") for n in ("ray_dx", "ray_dy", "ray_dz", "incidence_angle")]
    data = np.empty(len(co), dtype=np.dtype(fields))
    for i, n in enumerate(("x", "y", "z")):
        data[n] = co[:, i]
    for i, n in enumerate(("nx", "ny", "nz")):
        data[n] = normals[:, i]
    data["component_id"] = obj.get("vgc_component_id", 0)
    data["source_triangle"] = ids
    if ray_data:
        scanner = np.empty((len(co), 3), np.float32)
        direction = np.empty_like(scanner)
        mesh.attributes["scanner_position"].data.foreach_get("vector", scanner.ravel())
        mesh.attributes["ray_direction"].data.foreach_get("vector", direction.ravel())
        scanner = (scanner.astype(np.float64) @ matrix[:3, :3].T + matrix[:3, 3]) * scale_factor
        direction = direction.astype(np.float64) @ matrix[:3, :3].T
        direction /= np.maximum(np.linalg.norm(direction, axis=1)[:, None], 1e-30)
        for i, n in enumerate(("scanner_x", "scanner_y", "scanner_z")):
            data[n] = scanner[:, i]
        for i, n in enumerate(("ray_dx", "ray_dy", "ray_dz")):
            data[n] = direction[:, i]
        data["incidence_angle"] = np.arccos(
            np.clip(np.abs(np.sum(direction * normals, axis=1)), 0, 1)
        )
    ply_types = {"f8": "double", "f4": "float", "i4": "int"}
    lines = [
        "ply",
        "format binary_little_endian 1.0",
        "comment Synthetic geometry acquisition, not a physical scan",
        "comment Coordinates are world coordinates; all normals are world normals",
        "comment source_triangle -1 denotes an explicit boundary or crease observation",
        "comment incidence_angle is radians when present",
        f"element vertex {len(co)}",
    ]
    for name, dtype in fields:
        lines.append(f"property {ply_types[dtype[1:]]} {name}")
    lines += ["end_header", ""]
    with open(path, "wb") as f:
        f.write("\n".join(lines).encode("ascii"))
        data.tofile(f)
    return len(co)
