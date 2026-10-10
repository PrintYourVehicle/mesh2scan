# SPDX-License-Identifier: GPL-3.0-or-later
"""Conforming adaptive reference-guided scan tessellation.

This is deliberately a hybrid, not point-cloud-only volume extraction. Known
surface connectivity prevents cracks at panel boundaries and sharp folds. New
vertices are shared by edge ID and refinement prioritizes long/detail edges.
Captured observations supply a bounded tangent fit; protected edges stay fixed.
"""

import numpy as np
from .features import triangle_mesh


def refine_arrays(coords, triangles, detail, curvature, target, strength):
    """Yield conforming rounds. Splitting an edge costs its incident face count."""
    co = np.asarray(coords, np.float32)
    tri = np.asarray(triangles, np.int32)
    weights = np.asarray(detail, np.float32)
    cv = np.asarray(curvature, np.float32)
    for iteration in range(30):
        remaining = target - len(tri)
        if remaining <= 0:
            break
        # Triangle-major layout keeps inverse incidence in an (F, 3) array.
        edges = np.stack((tri[:, [0, 1]], tri[:, [1, 2]], tri[:, [2, 0]]), axis=1).reshape(-1, 2)
        edges.sort(axis=1)
        nv = len(co)
        encoded = edges[:, 0].astype(np.int64) * nv + edges[:, 1]
        unique, inverse, costs = np.unique(encoded, return_inverse=True, return_counts=True)
        edge = np.stack((unique // nv, unique % nv), axis=1).astype(np.int32)
        delta = co[edge[:, 1]] - co[edge[:, 0]]
        scores = np.sum(delta * delta, axis=1) * (1 + strength * 5 * weights[edge].mean(axis=1))
        # Refine only the upper half in a round to avoid a nearly uniform 4x
        # jump. Later rounds return to those edges as their length warrants.
        selected_count = min(len(edge), max(1, len(edge) // 2))
        best = np.argpartition(scores, len(scores) - selected_count)[-selected_count:]
        best = best[np.argsort(-scores[best], kind="stable")]
        cumulative = np.cumsum(costs[best], dtype=np.int64)
        selected = best[cumulative <= remaining]
        if not len(selected):
            # Odd remaining counts can be filled on a boundary edge; a closed
            # triangular manifold necessarily has an even face count.
            affordable = np.flatnonzero(costs <= remaining)
            if not len(affordable):
                break
            selected = affordable[np.argmax(scores[affordable])][None]
        mids = np.full(len(edge), -1, np.int32)
        mids[selected] = np.arange(len(selected), dtype=np.int32) + nv
        mid_per_face = mids[inverse].reshape(-1, 3)
        mask = np.sum((mid_per_face >= 0) * np.array([1, 2, 4], np.int32), axis=1)
        output = [tri[mask == 0]]
        for pattern in (1, 2, 4):
            ids = np.flatnonzero(mask == pattern)
            if not len(ids):
                continue
            rotation = {1: 0, 2: 1, 4: 2}[pattern]
            abc = np.roll(tri[ids], -rotation, axis=1)
            a, b, c = abc.T
            m = mid_per_face[ids, rotation]
            output.extend((np.stack((a, m, c), axis=1), np.stack((m, b, c), axis=1)))
        for pattern in (3, 6, 5):
            ids = np.flatnonzero(mask == pattern)
            if not len(ids):
                continue
            rotation = {3: 0, 6: 1, 5: 2}[pattern]
            a, b, c = np.roll(tri[ids], -rotation, axis=1).T
            ab = mid_per_face[ids, rotation]
            bc = mid_per_face[ids, (rotation + 1) % 3]
            output.append(np.stack((b, bc, ab), axis=1))
            # Triangulate the remaining quad using its shorter diagonal.
            pa = (co[edge[selected, 0]] + co[edge[selected, 1]]) * 0.5
            # Midpoint lookup includes new positions without a global Blender
            # mesh allocation at every round.
            full = np.concatenate((co, pa))
            choose = np.sum((full[a] - full[bc]) ** 2, axis=1) <= np.sum(
                (full[ab] - full[c]) ** 2, axis=1
            )
            t1 = np.where(
                choose[:, None], np.stack((a, ab, bc), axis=1), np.stack((a, ab, c), axis=1)
            )
            t2 = np.where(
                choose[:, None], np.stack((a, bc, c), axis=1), np.stack((ab, bc, c), axis=1)
            )
            output.extend((t1, t2))
            del full
        ids = np.flatnonzero(mask == 7)
        if len(ids):
            a, b, c = tri[ids].T
            ab, bc, ca = mid_per_face[ids].T
            output.extend(
                (
                    np.stack((a, ab, ca), axis=1),
                    np.stack((ab, b, bc), axis=1),
                    np.stack((ca, bc, c), axis=1),
                    np.stack((ab, bc, ca), axis=1),
                )
            )
        # All neighboring triangles refer to the same midpoint. No T-junction
        # repair, nearest-edge snapping, or destructive decimation is involved.
        co = np.concatenate((co, (co[edge[selected, 0]] + co[edge[selected, 1]]) * 0.5))
        weights = np.concatenate((weights, weights[edge[selected]].mean(axis=1)))
        cv = np.concatenate((cv, cv[edge[selected]].mean(axis=1)))
        tri = np.concatenate(output).astype(np.int32, copy=False)
        yield co, tri, weights, cv, iteration + 1
    return co, tri, weights, cv


def guided_mesh(reference, target, options):
    mesh = reference.data
    mesh.calc_loop_triangles()
    tri = np.empty((len(mesh.loop_triangles), 3), np.int32)
    mesh.loop_triangles.foreach_get("vertices", tri.ravel())
    co = np.empty((len(mesh.vertices), 3), np.float32)
    mesh.vertices.foreach_get("co", co.ravel())
    detail = np.zeros(len(co), np.float32)
    curvature = np.zeros(len(co), np.float32)
    for name, array in (("scan_detail", detail), ("scan_curvature", curvature)):
        attr = mesh.attributes.get(name)
        if attr:
            attr.data.foreach_get("value", array)
    result = co, tri, detail, curvature
    for co, tri, detail, curvature, iteration in refine_arrays(
        co, tri, detail, curvature, target, options.adaptive
    ):
        result = co, tri, detail, curvature
        yield len(tri), iteration, None
    co, tri, detail, curvature = result
    output = triangle_mesh("mesh2scan guided surface", co, tri)
    for name, values in (("scan_detail", detail), ("scan_curvature", curvature)):
        attr = output.attributes.new(name, "FLOAT", "POINT")
        attr.data.foreach_set("value", values)
    yield len(tri), 0, output
