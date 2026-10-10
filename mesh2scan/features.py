# SPDX-License-Identifier: GPL-3.0-or-later
"""Curved reference geometry and explicit boundary diagnostics/recovery.

PN positions follow Vlachos et al., Curved PN Triangles (2001). Creases and
open boundaries use straight cubic edge controls; smooth interior edges share
their controls. This interpolates geometry, not a normal-map height field.
"""

import math
import bpy
import numpy as np
from mathutils import Vector
from mathutils.kdtree import KDTree


def triangle_mesh(name, coords, triangles):
    mesh = bpy.data.meshes.new(name)
    mesh.vertices.add(len(coords))
    mesh.vertices.foreach_set("co", np.asarray(coords, np.float32).ravel())
    mesh.loops.add(len(triangles) * 3)
    mesh.loops.foreach_set("vertex_index", np.asarray(triangles, np.int32).ravel())
    mesh.polygons.add(len(triangles))
    mesh.polygons.foreach_set("loop_start", np.arange(len(triangles), dtype=np.int32) * 3)
    mesh.polygons.foreach_set("loop_total", np.full(len(triangles), 3, np.int32))
    mesh.update(calc_edges=True)
    return mesh


def curved_reference(mesh, feature_edges, curvature, options):
    """Return a welded, crease-aware PN reference and a bounded work report."""
    mesh.calc_loop_triangles()
    tri = np.empty((len(mesh.loop_triangles), 3), np.int32)
    loops = np.empty_like(tri)
    poly = np.empty(len(tri), np.int32)
    mesh.loop_triangles.foreach_get("vertices", tri.ravel())
    mesh.loop_triangles.foreach_get("loops", loops.ravel())
    mesh.loop_triangles.foreach_get("polygon_index", poly)
    co = np.empty((len(mesh.vertices), 3), np.float32)
    mesh.vertices.foreach_get("co", co.ravel())
    levels = int(options.smooth_levels) if options.surface_mode == "SMOOTH" else 0
    # Reference refinement is independent of the scan triangle budget. Keep
    # it bounded before allocating the arrays, including the user's RAM limit.
    limit = min(2_000_000, max(500, int(options.memory_gb * 1024**3 / 1400)))
    if options.mesher == "GUIDED":
        limit = min(limit, max(500, int(options.target_triangles * 0.45)))
    while levels and len(tri) * 4**levels > limit:
        levels -= 1
    if not levels or options.smooth_strength <= 0:
        attr = mesh.attributes.get("reference_curvature") or mesh.attributes.new(
            "reference_curvature", "FLOAT", "FACE"
        )
        attr.data.foreach_set("value", np.clip(curvature / options.crease_angle, 0, 1))
        return mesh, {
            "method": "ORIGINAL",
            "requested_levels": options.smooth_levels,
            "used_levels": 0,
            "triangles": len(tri),
            "max_vertex_offset_normalized": 0.0,
        }
    # Imported custom normals may encode bevel shading rather than shape, and
    # may be in the pre-normalization frame. Recompute geometric smooth fans.
    if mesh.has_custom_normals:
        mesh.normals_split_custom_set([(0.0, 0.0, 0.0)] * len(mesh.loops))
    mesh.polygons.foreach_set("use_smooth", np.ones(len(mesh.polygons), bool))
    mesh.set_sharp_from_angle(angle=options.crease_angle)
    nv = len(co)
    protected = np.asarray(feature_edges, np.int32).reshape(-1, 2)
    protected.sort(axis=1)
    protected_keys = protected[:, 0].astype(np.int64) * nv + protected[:, 1]
    edges = np.empty((len(mesh.edges), 2), np.int32)
    mesh.edges.foreach_get("vertices", edges.ravel())
    edges.sort(axis=1)
    sharp = mesh.attributes.get("sharp_edge") or mesh.attributes.new(
        "sharp_edge", "BOOLEAN", "EDGE"
    )
    sharp.data.foreach_set(
        "value", np.isin(edges[:, 0].astype(np.int64) * nv + edges[:, 1], protected_keys)
    )
    mesh.update()
    cn = np.empty((len(mesh.corner_normals), 3), np.float32)
    mesh.corner_normals.foreach_get("vector", cn.ravel())
    normals = cn[loops]
    p = co[tri]
    all_edges = np.stack((tri[:, [0, 1]], tri[:, [1, 2]], tri[:, [2, 0]]), axis=1)
    all_edges.sort(axis=2)
    all_keys = all_edges[:, :, 0].astype(np.int64) * nv + all_edges[:, :, 1]
    edge_keys, edge_inverse = np.unique(all_keys, return_inverse=True)
    side = np.linalg.norm(co[all_edges[:, :, 1]] - co[all_edges[:, :, 0]], axis=2)
    twice_area = np.linalg.norm(np.cross(p[:, 1] - p[:, 0], p[:, 2] - p[:, 0]), axis=1)
    altitude = twice_area / np.maximum(side.max(axis=1), 1e-20)
    limits = np.full(len(edge_keys), np.inf, np.float32)
    np.minimum.at(limits, edge_inverse, np.repeat(altitude * 0.3, 3))
    controls = [p[:, 0], p[:, 1], p[:, 2]]
    exponents = [(3, 0, 0), (0, 3, 0), (0, 0, 3)]
    edge_controls = []
    for i, j in ((0, 1), (1, 0), (1, 2), (2, 1), (2, 0), (0, 2)):
        delta = p[:, j] - p[:, i]
        e = np.sort(tri[:, [i, j]], axis=1)
        hard = np.isin(e[:, 0].astype(np.int64) * nv + e[:, 1], protected_keys)
        correction = np.sum(delta * normals[:, i], axis=1)[:, None] * normals[:, i]
        limit = np.minimum(
            limits[np.searchsorted(edge_keys, e[:, 0].astype(np.int64) * nv + e[:, 1])],
            np.linalg.norm(delta, axis=1) * 0.25,
        )
        magnitude = np.linalg.norm(correction, axis=1)
        correction *= np.minimum(1, limit / np.maximum(magnitude, 1e-20))[:, None]
        correction[hard] = 0
        control = (2 * p[:, i] + p[:, j] - options.smooth_strength * correction) / 3
        controls.append(control)
        edge_controls.append(control)
        exp = [0, 0, 0]
        exp[i], exp[j] = 2, 1
        exponents.append(tuple(exp))
    e = np.mean(edge_controls, axis=0)
    controls.append(e + (e - p.mean(axis=1)) * 0.5)
    exponents.append((1, 1, 1))
    n = 2**levels
    grid = [(i, j) for i in range(n + 1) for j in range(n + 1 - i)]
    lookup = {ij: k for k, ij in enumerate(grid)}
    local_tri = []
    for i in range(n):
        for j in range(n - i):
            local_tri.append((lookup[i, j], lookup[i + 1, j], lookup[i, j + 1]))
            if i + j < n - 1:
                local_tri.append((lookup[i + 1, j], lookup[i + 1, j + 1], lookup[i, j + 1]))
    # Grid barycentrics have p0 at (0,0), p1 at (n,0), p2 at (0,n).
    bary = np.array([(1 - (i + j) / n, i / n, j / n) for i, j in grid], np.float32)
    values = np.zeros((len(tri), len(grid), 3), np.float32)
    for control, powers in zip(controls, exponents):
        coefficient = math.factorial(3) / math.prod(math.factorial(k) for k in powers)
        basis = coefficient * np.prod(bary ** np.array(powers), axis=1)
        values += control[:, None, :] * basis[None, :, None]
    linear = np.einsum("gc,tcd->tgd", bary, p)
    offset = float(np.linalg.norm(values - linear, axis=2).max())
    # Share vertices by source topology, not rounded position. Floating point
    # differences between patch evaluations must never open a closed surface.
    flat = values.reshape(-1, 3)
    keys = np.zeros((len(tri), len(grid), 4), np.int64)
    for g, (i, j) in enumerate(grid):
        bc = np.array([n - i - j, i, j])
        active = np.flatnonzero(bc)
        if len(active) == 1:
            keys[:, g, 0] = 0
            keys[:, g, 1] = tri[:, active[0]]
        elif len(active) == 2:
            a, b = active
            keys[:, g, 0] = 1
            keys[:, g, 1] = np.minimum(tri[:, a], tri[:, b])
            keys[:, g, 2] = np.maximum(tri[:, a], tri[:, b])
            keys[:, g, 3] = np.where(tri[:, a] < tri[:, b], bc[b], bc[a])
        else:
            keys[:, g, 0] = 2
            keys[:, g, 1] = np.arange(len(tri))
            keys[:, g, 2] = g
    unique_points, first, inverse = np.unique(
        keys.reshape(-1, 4), axis=0, return_index=True, return_inverse=True
    )
    indices = (
        np.arange(len(tri), dtype=np.int32)[:, None, None] * len(grid)
        + np.array(local_tri, np.int32)[None, :, :]
    ).reshape(-1, 3)
    new_tri = inverse[indices].astype(np.int32)
    result = triangle_mesh("Curved scanner reference", flat[first], new_tri)
    fixed = np.zeros(len(unique_points), bool)
    corner_mask = unique_points[:, 0] == 0
    fixed[corner_mask] = np.isin(unique_points[corner_mask, 1], protected.ravel())
    edge_mask = unique_points[:, 0] == 1
    fixed[edge_mask] = np.isin(
        unique_points[edge_mask, 1] * nv + unique_points[edge_mask, 2], protected_keys
    )
    lock = result.attributes.new("reference_protected", "BOOLEAN", "POINT")
    lock.data.foreach_set("value", fixed)
    attr = result.attributes.new("reference_curvature", "FLOAT", "FACE")
    attr.data.foreach_set(
        "value", np.repeat(np.clip(curvature[poly] / options.crease_angle, 0, 1), n * n)
    )
    return result, {
        "method": "CREASE_AWARE_PN",
        "requested_levels": options.smooth_levels,
        "used_levels": levels,
        "triangles": len(new_tri),
        "max_vertex_offset_normalized": offset,
        "normal_source": "geometric smooth fans; shader normal maps excluded",
        "reference_cap_reached": levels < options.smooth_levels,
    }


def boundary_edges(mesh):
    mesh.calc_loop_triangles()
    tri = np.empty((len(mesh.loop_triangles), 3), np.int32)
    mesh.loop_triangles.foreach_get("vertices", tri.ravel())
    edges = np.concatenate((tri[:, [0, 1]], tri[:, [1, 2]], tri[:, [2, 0]]))
    ordered = np.sort(edges, axis=1)
    keys = ordered[:, 0].astype(np.int64) * len(mesh.vertices) + ordered[:, 1]
    _, first, count = np.unique(keys, return_index=True, return_counts=True)
    first = first[count == 1]
    return tri, edges[first], first % max(1, len(tri))


def boundary_corner_count(coords, edges):
    neighbors = {}
    for a, b in edges:
        neighbors.setdefault(a, []).append(b)
        neighbors.setdefault(b, []).append(a)
    count = 0
    for vertex, other in neighbors.items():
        if len(other) != 2:
            continue
        a, b = coords[other[0]] - coords[vertex], coords[other[1]] - coords[vertex]
        bend = np.linalg.norm(np.cross(a, b)) / max(np.linalg.norm(a) * np.linalg.norm(b), 1e-30)
        count += bend >= math.sin(math.radians(1))
    return count


def recover_boundaries(obj, reference_coords, source_edges, h, radius=3.0):
    """Fit only real scan perimeters; insert crossed source corners explicitly.

    Do not fill holes, bridge neighbouring sheets, or move interior cracks to a
    distant perimeter. Large unresolved gaps remain visible in the report.
    """
    if not source_edges:
        return {"vertices_fitted": 0, "corners_inserted": 0}
    mesh = obj.data
    tri, edges, faces = boundary_edges(mesh)
    if not len(edges):
        return {"vertices_fitted": 0, "corners_inserted": 0}
    co = np.empty((len(mesh.vertices), 3), np.float32)
    mesh.vertices.foreach_get("co", co.ravel())
    segments = reference_coords[np.array(source_edges, np.int32)]
    delta = segments[:, 1] - segments[:, 0]
    lengths = np.linalg.norm(delta, axis=1)
    counts = np.maximum(2, np.ceil(lengths / max(h, 1e-8)).astype(np.int32) + 1)
    count = int(counts.sum())
    if count > 2_000_000:
        return {"vertices_fitted": 0, "corners_inserted": 0, "skipped": "boundary index too large"}
    kd = KDTree(count)
    sid = []
    for i, (segment, steps) in enumerate(zip(segments, counts)):
        for t in np.linspace(0, 1, steps):
            kd.insert(Vector(segment[0] + (segment[1] - segment[0]) * t), len(sid))
            sid.append(i)
    kd.balance()
    ids = np.unique(edges)
    mapping = {}
    fitted = 0
    for vi in ids:
        candidates = {sid[index] for _, index, _ in kd.find_n(Vector(co[vi]), min(12, count))}
        chosen, best, location = None, float("inf"), None
        for si in candidates:
            t = np.clip(
                np.dot(co[vi] - segments[si, 0], delta[si]) / max(lengths[si] ** 2, 1e-30), 0, 1
            )
            pos = segments[si, 0] + t * delta[si]
            d = float(np.linalg.norm(co[vi] - pos))
            if d < best:
                chosen, best, location = si, d, pos
        if best <= radius * h:
            co[vi] = location
            mapping[int(vi)] = chosen
            fitted += 1
    # If an output perimeter edge crosses a genuine source corner, split its
    # incident triangle through that corner. A nearest-line snap alone cuts the
    # corner diagonally and reproduces the user's ragged wireframe outline.
    source_edges = np.asarray(source_edges, np.int32)
    new_coords, replacements = [], {}
    corners = set()
    for (a, b), fi in zip(edges, faces):
        sa, sb = mapping.get(int(a)), mapping.get(int(b))
        if sa is None or sb is None or sa == sb or int(fi) in replacements:
            continue
        shared = set(source_edges[sa]).intersection(source_edges[sb])
        if len(shared) != 1:
            continue
        corner_id = next(iter(shared))
        corner = reference_coords[corner_id]
        oa = next(int(v) for v in source_edges[sa] if v != corner_id)
        ob = next(int(v) for v in source_edges[sb] if v != corner_id)
        va, vb = reference_coords[oa] - corner, reference_coords[ob] - corner
        bend = np.linalg.norm(np.cross(va, vb)) / max(
            np.linalg.norm(va) * np.linalg.norm(vb), 1e-30
        )
        if bend < math.sin(math.radians(1)) or tuple(corner) in corners:
            continue
        if (
            np.linalg.norm(corner - co[a]) < h * 0.01
            or np.linalg.norm(corner - co[b]) < h * 0.01
            or np.linalg.norm(corner - co[a]) + np.linalg.norm(corner - co[b]) > h * 6
        ):
            continue
        # Directed boundary edge order follows the face winding.
        face = tri[fi]
        opposite = next(int(v) for v in face if v != a and v != b)
        ci = len(co) + len(new_coords)
        new_coords.append(corner)
        replacements[int(fi)] = ((int(a), ci, opposite), (ci, int(b), opposite))
        corners.add(tuple(corner))
    mesh.vertices.foreach_set("co", co.ravel())
    mesh.update()
    if replacements:
        keep = np.ones(len(tri), bool)
        keep[list(replacements)] = False
        triangles = np.concatenate(
            (tri[keep], np.asarray([f for pair in replacements.values() for f in pair], np.int32))
        )
        coords = np.concatenate((co, np.array(new_coords, np.float32)))
        new = triangle_mesh("Boundary recovered scan", coords, triangles)
        # Density attributes are sampled again after this topology change.
        obj.data = new
        if mesh.users == 0:
            bpy.data.meshes.remove(mesh)
    return {
        "vertices_fitted": fitted,
        "corners_inserted": len(replacements),
        "distinct_corners": len(corners),
    }


def add_diagnostic_colors(mesh):
    """Curvature and refinement-demand colors, independent of shader normals."""
    for scalar, name in (("scan_curvature", "Scan Curvature"), ("scan_detail", "Scan Detail")):
        attr = mesh.attributes.get(scalar)
        if not attr:
            continue
        values = np.empty(len(mesh.vertices), np.float32)
        attr.data.foreach_get("value", values)
        v = np.clip(values, 0, 1)
        # Blue -> cyan -> yellow -> red. The diagnostic is dimensionless;
        # it is a feature score, not a principal-curvature CAD classifier.
        colors = np.ones((len(v), 4), np.float32)
        stops = np.array([0.0, 0.33, 0.66, 1.0])
        palette = np.array([[0.05, 0.15, 1], [0, 1, 1], [1, 1, 0], [1, 0.05, 0]])
        for channel in range(3):
            colors[:, channel] = np.interp(v, stops, palette[:, channel])
        color = mesh.color_attributes.get(name) or mesh.color_attributes.new(
            name, "BYTE_COLOR", "POINT"
        )
        color.data.foreach_set("color", colors.ravel())
    mesh.color_attributes.active_color_name = "Scan Detail"
