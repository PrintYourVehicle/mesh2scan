"""Optional one-million-face scaling check in a fresh native Blender process."""

import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import bpy
from mesh2scan.core import ScanJob, ScanOptions
from mesh2scan.features import boundary_edges

bpy.ops.object.select_all(action="SELECT")
bpy.ops.object.delete(use_global=False)
coords = []
for j in range(9):
    for i in range(13):
        x, y = (i - 6) / 6, (j - 4) / 4
        coords.append((x, y, 0.3 * abs(x) + 0.08 * y * y))
faces = []
for j in range(8):
    for i in range(12):
        if 4 <= i <= 5 and 3 <= j <= 4:
            continue
        a = j * 13 + i
        faces.append((a, a + 1, a + 14, a + 13))
mesh = bpy.data.meshes.new("Dense folded panel")
mesh.from_pydata(coords, [], faces)
mesh.update()
source = bpy.data.objects.new("Dense folded panel", mesh)
bpy.context.scene.collection.objects.link(source)
bpy.context.view_layer.objects.active = source
source.select_set(True)
job = ScanJob(
    bpy.context,
    source,
    ScanOptions(
        target_triangles=1_000_000,
        observation_cap=100_000,
        keep_cloud=False,
        crease_angle=math.radians(25),
    ),
)
for progress, message in job.stages():
    print(round(progress, 2), message, flush=True)
assert job.report["triangles"] == 1_000_000
assert job.report["nonmanifold_edges"] == 0
assert job.report["degenerate_triangles"] == 0
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
assert loops == 2
report = dict(report=job.report, boundary_loops=loops)
try:
    import resource

    # Linux ru_maxrss is KiB; macOS reports bytes.
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    report["process_peak_rss_gib"] = value / (1024**3 if sys.platform == "darwin" else 1024**2)
except ImportError:
    pass
reports = ROOT / "reports"
reports.mkdir(parents=True, exist_ok=True)
(reports / "dense_capture_5_1.json").write_text(json.dumps(report, indent=2))
print(
    "DENSE_CAPTURE_PASSED", bpy.app.version_string, report.get("process_peak_rss_gib"), flush=True
)
