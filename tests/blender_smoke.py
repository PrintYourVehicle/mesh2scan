"""Run with Blender --background --factory-startup --python, or official bpy."""

import sys, json, math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import bpy
import numpy as np
import mesh2scan as addon
from mesh2scan.core import ScanJob, ScanOptions, mesh_coords
from mesh2scan.batch import BatchScanJob, fit_plan, allocate_budget

addon.register()
ctx = bpy.context
results = []


def scan(kind, opts):
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)
    if kind == "cube":
        bpy.ops.mesh.primitive_cube_add()
    elif kind == "sphere":
        bpy.ops.mesh.primitive_uv_sphere_add(segments=16, ring_count=10)
    else:
        # Folded grid with a sharp crease, shallow curvature and an intentional
        # square vent; matches the failure mechanisms in the screenshot.
        co = []
        for j in range(9):
            for i in range(13):
                x = (i - 6) / 6
                y = (j - 4) / 4
                z = 0.3 * abs(x) + 0.08 * y * y
                co.append((x, y, z))
        faces = []
        for j in range(8):
            for i in range(12):
                if 4 <= i <= 5 and 3 <= j <= 4:
                    continue
                a = j * 13 + i
                faces.append((a, a + 1, a + 14, a + 13))
        m = bpy.data.meshes.new(kind)
        m.from_pydata(co, [], faces)
        m.update()
        ob = bpy.data.objects.new(kind, m)
        ctx.scene.collection.objects.link(ob)
        ctx.view_layer.objects.active = ob
        ob.select_set(True)
    ob = ctx.object
    before = mesh_coords(ob.data).copy()
    job = ScanJob(ctx, ob, opts)
    for p, msg in job.stages():
        print(kind, round(p, 2), msg, flush=True)
    assert np.array_equal(before, mesh_coords(ob.data))
    assert job.report["nonmanifold_edges"] == 0 and job.report["degenerate_triangles"] == 0
    if kind in ("cube", "sphere"):
        assert job.report["boundary_edges"] == 0
    if kind == "fold":
        # One outer perimeter and one vent; no newly created internal cracks.
        from mesh2scan.features import boundary_edges

        _, edges, _ = boundary_edges(job.result.data)
        adjacent = {}
        for a, b in edges:
            adjacent.setdefault(int(a), []).append(int(b))
            adjacent.setdefault(int(b), []).append(int(a))
        assert all(len(v) == 2 for v in adjacent.values())
        unseen = set(adjacent)
        loops = 0
        while unseen:
            todo = [unseen.pop()]
            loops += 1
            while todo:
                for v in adjacent[todo.pop()]:
                    if v in unseen:
                        unseen.remove(v)
                        todo.append(v)
        assert loops == 2, loops
    assert "Scan Detail" in job.result.data.color_attributes
    results.append({"case": kind, "report": job.report})


opts = ScanOptions(
    target_triangles=25000, keep_cloud=False, observation_cap=30000, crease_angle=math.radians(25)
)
for kind in ("cube", "sphere", "fold"):
    scan(kind, opts)
# Actual memory scaling without allocating a 25M mesh.
options = ScanOptions(target_triangles=25_000_000, memory_gb=2, observation_cap=10000)
plan = fit_plan(
    [{"allocation_weight": 1, "input_triangles": 1000}], 25_000_000, options, allocate_budget
)
assert plan["fits"] and plan["adjusted"] and plan["effective_triangles"] < 25_000_000
strict = ScanOptions(target_triangles=25_000_000, memory_gb=2, memory_policy="STRICT")
assert not fit_plan(
    [{"allocation_weight": 1, "input_triangles": 1000}], 25_000_000, strict, allocate_budget
)["fits"]
results.append({"case": "memory_plan", "plan": plan})
# Real UI operators and custom profiles.
s = ctx.scene.mesh2scan_settings
s.scanner_profile = "PANEL"
assert s.mesher == "GUIDED"
s.triangles_million = 0.02
s.profile_name = "Smoke test profile"
assert bpy.ops.object.mesh2scan_profile(action="SAVE") == {"FINISHED"}
s.triangles_million = 0.01
assert bpy.ops.object.mesh2scan_profile(action="LOAD") == {"FINISHED"}
assert abs(s.triangles_million - 0.02) < 1e-6
assert bpy.ops.object.mesh2scan_profile(action="DELETE") == {"FINISHED"}
assert bpy.ops.object.mesh2scan_plan() == {"FINISHED"}
assert json.loads(s.plan_json)["fits"]
# Both methods remain callable.
scan(
    "cube",
    ScanOptions(mesher="VOLUME", target_triangles=8000, keep_cloud=False, observation_cap=10000),
)
reports = ROOT / "reports"
reports.mkdir(parents=True, exist_ok=True)
(
    reports
    / ("validation_" + str(bpy.app.version[0]) + "_" + str(bpy.app.version[1]) + ".json")
).write_text(json.dumps(results, indent=2))
addon.unregister()
print("SMOKE_PASSED", bpy.app.version_string, flush=True)
