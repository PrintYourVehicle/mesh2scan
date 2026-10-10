"""Focused native installation/UI checks; no desktop automation required."""

import json
import re
import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import bpy

try:
    import _bpy_restrict_state as restrictions
except ImportError:
    import bpy_restrict_state as restrictions

facade = sys.modules["bpy"]
sys.modules["bpy"] = restrictions._bpy
with restrictions.RestrictBlend():
    import mesh2scan as addon

    assert not hasattr(restrictions._bpy.data, "scenes")
    addon.register()
    addon.unregister()
    addon.register()
sys.modules["bpy"] = facade
from mesh2scan.core import ScanOptions
from mesh2scan.batch import BatchScanJob

ctx = bpy.context
bpy.ops.object.select_all(action="SELECT")
bpy.ops.object.delete(use_global=False)
bpy.ops.mesh.primitive_grid_add(x_subdivisions=3, y_subdivisions=3)
a = ctx.object
bpy.ops.mesh.primitive_grid_add(x_subdivisions=3, y_subdivisions=3, location=(4, 0, 0))
b = ctx.object
a.select_set(True)
job = BatchScanJob(ctx, [a, b], ScanOptions(target_triangles=8000, keep_cloud=False))
steps = job.stages()
for p, message in steps:
    if "Complete:" in message and p < 1:
        break
steps.close()
assert not any(o.get("vgc_scan") for o in bpy.data.objects)
assert not any(o.name.startswith("_M2S") for o in bpy.data.objects)
assert not a.hide_get() and not b.hide_get()


class ReadOnly:
    def __init__(self, wrapped):
        object.__setattr__(self, "wrapped", wrapped)

    def __getattr__(self, key):
        return getattr(self.wrapped, key)

    def __setattr__(self, key, value):
        raise AssertionError("Draw wrote " + key)


class Layout:
    enabled = True

    def column(self, *args, **kwargs):
        return Layout()

    row = box = column

    def prop(self, *args, **kwargs):
        pass

    label = separator = progress = prop

    def operator(self, *args, **kwargs):
        return SimpleNamespace()


s = ctx.scene.mesh2scan_settings
s.busy = True
s.profile_tools = True
s.advanced = True
scene = SimpleNamespace(mesh2scan_settings=ReadOnly(s), as_pointer=ctx.scene.as_pointer)
context = SimpleNamespace(scene=scene, active_object=b, selected_objects=[a, b], mode="OBJECT")
panel = SimpleNamespace(layout=Layout())
for section in ("SHAPE", "CAPTURE", "MEMORY", "VOLUME"):
    s.advanced_section = section
    addon.M2S_PT_panel.draw(panel, context)
assert addon.M2S_OT_capture.poll(context)
addon._active_captures[scene.as_pointer()] = SimpleNamespace()
addon.M2S_PT_panel.draw(panel, context)
addon._active_captures.clear()

# Event has no .timer; unrelated input must still reach Blender's buttons.
fake = SimpleNamespace(
    _finished=False,
    _scene=ctx.scene,
    _job=SimpleNamespace(view_layer=ctx.view_layer),
    _step=lambda context: {"RUNNING_MODAL"},
)
assert addon.M2S_OT_capture.modal(fake, ctx, SimpleNamespace(type="TIMER")) == {"RUNNING_MODAL"}
assert addon.M2S_OT_capture.modal(fake, ctx, SimpleNamespace(type="LEFTMOUSE")) == {"PASS_THROUGH"}

# A strict oversized request returns a recoverable warning, clears running state,
# and leaves the inputs visible. It never attempts to allocate a 25M result.
s.busy = False
s.triangles_million = 25
s.memory_gb = 1
s.memory_policy = "STRICT"
assert bpy.ops.object.mesh2scan_capture() == {"CANCELLED"}
assert not s.busy and not addon._active_captures
assert s.last_error and not b.hide_get()

icons = set(bpy.types.UILayout.bl_rna.functions["label"].parameters["icon"].enum_items.keys())
used = set(re.findall(r"icon=[\"\']([A-Z_]+)", (ROOT / "mesh2scan" / "__init__.py").read_text()))
assert used <= icons, used - icons
assert "progress" in bpy.types.UILayout.bl_rna.functions
report = dict(
    blender=bpy.app.version_string,
    restricted_install=True,
    readonly_draw=True,
    batch_cancel_rollback=True,
    timer_without_event_timer=True,
    recoverable_memory_stop=True,
    native_ui_icons=True,
)
reports = ROOT / "reports"
reports.mkdir(parents=True, exist_ok=True)
(reports / f"lifecycle_{bpy.app.version[0]}_{bpy.app.version[1]}.json").write_text(
    json.dumps(report, indent=2)
)
addon.unregister()
print("LIFECYCLE_PASSED", bpy.app.version_string, flush=True)
