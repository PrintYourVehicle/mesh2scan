# SPDX-License-Identifier: GPL-3.0-or-later
"""World-area/feature allocation and sequential captures with one total budget."""

import json
import math
import time
from dataclasses import replace
import bpy
import numpy as np
from .core import ScanError, ScanJob, mesh_coords, triangle_data
from .planning import fit_plan


def measure_component(context, obj, options):
    if abs(obj.matrix_world.determinant()) < 1e-30:
        raise ScanError(f"{obj.name}: restore its zero scale axis before capture.")
    hidden = obj.hide_get(view_layer=context.view_layer)
    mesh = None
    try:
        obj.hide_set(False, view_layer=context.view_layer)
        context.view_layer.update()
        dg = context.evaluated_depsgraph_get()
        mesh = (
            bpy.data.meshes.new_from_object(
                obj.evaluated_get(dg), preserve_all_data_layers=True, depsgraph=dg
            )
            if options.use_modifiers
            else obj.data.copy()
        )
        if len(mesh.polygons) > 3_000_000:
            raise ScanError(
                f"{obj.name}: simplify the input below 3 million faces or split it into components."
            )
        co = mesh_coords(mesh).astype(np.float64)
        m = np.array(obj.matrix_world, np.float64)
        co = co @ m[:3, :3].T + m[:3, 3]
        tri, _ = triangle_data(mesh)
        if not len(tri) or not np.isfinite(co).all():
            raise ScanError(f"{obj.name}: no finite mesh surface to capture.")
        p = co[tri]
        cross = np.cross(p[:, 1] - p[:, 0], p[:, 2] - p[:, 0])
        area = np.linalg.norm(cross, axis=1) * 0.5
        normals = cross / np.maximum(area[:, None] * 2, 1e-30)
        edges = np.concatenate((tri[:, [0, 1]], tri[:, [1, 2]], tri[:, [2, 0]]))
        edges.sort(axis=1)
        key = edges[:, 0].astype(np.int64) * len(co) + edges[:, 1]
        order = np.argsort(key)
        skey = key[order]
        starts = np.r_[0, np.flatnonzero(skey[1:] != skey[:-1]) + 1]
        ends = np.r_[starts[1:], len(order)]
        counts = ends - starts
        paired = starts[counts == 2]
        a, b = order[paired] % len(tri), order[paired + 1] % len(tri)
        angles = np.arccos(np.clip(np.sum(normals[a] * normals[b], axis=1), -1, 1))
        scores = np.zeros(len(tri), np.float64)
        np.maximum.at(scores, a, angles)
        np.maximum.at(scores, b, angles)
        total = float(area.sum())
        if total <= 1e-25:
            raise ScanError(f"{obj.name}: surface has zero area.")
        curvature = float(np.sum(area * np.clip(scores / options.crease_angle, 0, 1)) / total)
        be = edges[order[starts[counts == 1]]]
        length = (
            float(np.linalg.norm(co[be[:, 0]] - co[be[:, 1]], axis=1).sum()) if len(be) else 0.0
        )
        # Dimensionless boundary score is capped: a thin trim should get more
        # sampling per area, but must not consume the entire vehicle budget.
        boundary = min(1.0, length / max(math.sqrt(total) * 12, 1e-30))
        feature = min(1.0, 0.8 * curvature + 0.2 * boundary)
        importance = float(
            getattr(obj, "mesh2scan_importance", obj.get("mesh2scan_importance", 1.0))
        )
        if not math.isfinite(importance) or importance <= 0:
            importance = 1.0
        importance = min(8.0, max(0.125, importance))
        return {
            "source": obj.name,
            "input_triangles": len(tri),
            "area_world_squared": total,
            "curvature_score": curvature,
            "boundary_length_world": length,
            "importance": importance,
            "allocation_weight": total * (1 + options.detail_boost * feature) * importance,
        }
    finally:
        if mesh is not None:
            bpy.data.meshes.remove(mesh)
        obj.hide_set(hidden, view_layer=context.view_layer)


def allocate_budget(measurements, total, minimum=500, preserve_input=False):
    """Exact integer apportionment; the floor is inside the shared budget."""
    n = len(measurements)
    floors = np.array(
        [max(minimum, m["input_triangles"]) if preserve_input else minimum for m in measurements],
        np.int64,
    )
    floor = int(floors.sum())
    if total < floor:
        raise ScanError(
            f"Allow at least {floor:,} triangles to preserve these parts, or simplify the source first."
        )
    weights = np.array([m["allocation_weight"] for m in measurements], np.float64)
    exact = weights / weights.sum() * (total - floor)
    result = np.floor(exact).astype(np.int64) + floors
    left = total - int(result.sum())
    result[np.argsort(-(exact - np.floor(exact)), kind="stable")[:left]] += 1
    return result.tolist()


def collect_sources(context, settings):
    selected = (
        list(context.selected_objects) if settings.scope == "SELECTED" else [context.active_object]
    )
    result, seen = [], set()
    for obj in selected:
        if obj is None or obj.type != "MESH" or obj.get("vgc_cloud"):
            continue
        if settings.rescan_reference and obj.get("vgc_scan"):
            ref = obj.get("vgc_reference")
            if ref is not None and isinstance(ref, bpy.types.Object) and ref.type == "MESH":
                obj = ref
        if obj.as_pointer() not in seen:
            seen.add(obj.as_pointer())
            result.append(obj)
    if not result:
        raise ScanError("Select the exterior mesh parts you want to scan in Object Mode.")
    return sorted(result, key=lambda obj: obj.name)


class BatchScanJob:
    def __init__(self, context, sources, options):
        self.context, self.sources, self.options = context, sources, options
        self.view_layer = context.view_layer
        self.report = {}
        self.result = self.cloud = None
        self.jobs = []
        self.completed = False
        self.hidden = [obj.hide_get(view_layer=self.view_layer) for obj in sources]
        self.previous_selected = list(context.selected_objects)
        self.previous_active = context.active_object
        self.started = time.perf_counter()
        self.batch_id = f"M2S-{time.time_ns()}"

    def stages(self):
        try:
            measurements = []
            n = len(self.sources)
            yield (0.01, f"Planning a shared scan budget for {n} parts")
            for i, obj in enumerate(self.sources):
                measurements.append(measure_component(self.context, obj, self.options))
                yield (
                    0.02 + 0.07 * (i + 1) / n,
                    f"Measuring surface area and features: {obj.name}",
                )
            plan = None
            if self.options.resolution == "TARGET":
                plan = fit_plan(
                    measurements, self.options.target_triangles, self.options, allocate_budget
                )
                if not plan["fits"]:
                    raise ScanError(
                        f"Estimated capture RAM {plan['estimated_peak_gib']:.1f} GiB exceeds the safe allowance {plan['memory_limit_gib']:.1f} GiB. "
                        "Use Fit available RAM, Guided surface, or a smaller target. This is RAM, not disk space."
                    )
                self.options = replace(self.options, target_triangles=plan["effective_triangles"])
                allocations = plan["allocations"]
                yield (
                    0.095,
                    f"RAM plan: {plan['effective_triangles']:,} triangles total; about {plan['estimated_peak_gib']:.1f} GiB",
                )
            else:
                allocations = [self.options.target_triangles] * n
            weights = np.array(allocations, np.float64)
            weights /= weights.sum()
            previous, retained = 0.0, 0.0
            for i, (source, target, fraction) in enumerate(zip(self.sources, allocations, weights)):
                options = replace(
                    self.options,
                    target_triangles=target,
                    retained_gb=retained,
                    component_id=i,
                    preplanned=True,
                )
                job = ScanJob(self.context, source, options)
                self.jobs.append(job)
                steps = job.stages()
                try:
                    for progress, text in steps:
                        yield (
                            min(0.99, 0.1 + 0.88 * (previous + float(fraction) * progress)),
                            f"Part {i + 1}/{n}: {source.name} — {text}",
                        )
                finally:
                    steps.close()
                job.result["vgc_batch_id"] = self.batch_id
                job.result["vgc_component_id"] = i
                if options.keep_cloud:
                    job.cloud["vgc_batch_id"] = self.batch_id
                self.result, self.cloud = job.result, job.cloud if options.keep_cloud else None
                measurements[i]["target_triangles"] = target
                measurements[i]["capture"] = job.report
                # Include already retained outputs/clouds in the next native
                # capture's RAM estimate rather than forgetting previous parts.
                retained += (
                    job.report["triangles"] * 220
                    + (job.report["acquired_observations"] * 100 if options.keep_cloud else 0)
                ) / 1024**3
                previous += float(fraction)
            reports = [job.report for job in self.jobs]
            self.report = {
                "version": "0.0.1",
                "blender": bpy.app.version_string,
                "batch_id": self.batch_id,
                "scope": "SELECTED",
                "component_count": n,
                "plan": plan,
                "target_triangles": self.options.target_triangles,
                "triangles": sum(r["triangles"] for r in reports),
                "acquired_observations": sum(r["acquired_observations"] for r in reports),
                "boundary_edges": sum(r["boundary_edges"] for r in reports),
                "nonmanifold_edges": sum(r["nonmanifold_edges"] for r in reports),
                "degenerate_triangles": sum(r["degenerate_triangles"] for r in reports),
                "sampled_face_deviation_world": {
                    "maximum": max(r["sampled_face_deviation_world"]["maximum"] for r in reports)
                },
                "retained_output_estimate_gib": retained,
                "seconds": round(time.perf_counter() - self.started, 3),
                "components": measurements,
                "notes": [
                    "One shared target budget, distributed by world area, features and importance.",
                    "Separate output parts retain placement. All results are rolled back if this batch is cancelled.",
                ],
            }
            self.report["final_triangle_budget_ratio"] = (
                self.report["triangles"] / self.options.target_triangles
            )
            text = json.dumps(self.report)
            for job in self.jobs:
                job.result["vgc_batch_report"] = text
                job.result.select_set(True)
            self.completed = True
            yield (1.0, f"Complete: {n} parts, {self.report['triangles']:,} triangles total")
        finally:
            if not self.completed:
                # Cancellation/errors never leave a partially hidden car or
                # incomplete batch pretending to be a successful total scan.
                for job in self.jobs:
                    for obj in (job.result, job.cloud):
                        try:
                            if obj is not None:
                                mesh = obj.data
                                bpy.data.objects.remove(obj, do_unlink=True)
                                if mesh.users == 0:
                                    bpy.data.meshes.remove(mesh)
                        except (ReferenceError, RuntimeError):
                            pass
                for obj, hidden in zip(self.sources, self.hidden):
                    try:
                        obj.hide_set(hidden, view_layer=self.view_layer)
                    except (ReferenceError, RuntimeError):
                        pass
                try:
                    for obj in self.view_layer.objects:
                        obj.select_set(False, view_layer=self.view_layer)
                    for obj in self.previous_selected:
                        obj.select_set(True, view_layer=self.view_layer)
                    self.view_layer.objects.active = self.previous_active
                except (ReferenceError, RuntimeError):
                    pass
