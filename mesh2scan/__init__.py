# SPDX-License-Identifier: GPL-3.0-or-later
"""Installable Blender 4.5+ extension, with a staged cancellable UI."""

import json
import math
from pathlib import Path

import bpy
from bpy.app.handlers import persistent
from bpy.props import (
    BoolProperty,
    EnumProperty,
    FloatProperty,
    IntProperty,
    PointerProperty,
    StringProperty,
)
from bpy_extras.io_utils import ExportHelper

from .core import ScanError, ScanJob, ScanOptions, export_cloud_ply
from .batch import BatchScanJob, collect_sources, measure_component, allocate_budget
from .planning import fit_plan
from . import profiles


_active_captures = {}


def _active_capture(scene):
    """Read-only lookup: draw and poll callbacks must never modify scene IDs."""
    return _active_captures.get(scene.as_pointer())


_applying_profile = False


def _scanner_profile(self, context):
    global _applying_profile
    _applying_profile = True
    try:
        profiles.apply_builtin(self)
    finally:
        _applying_profile = False


def _mark_custom(self, context):
    self.plan_json = ""
    if not _applying_profile and self.scanner_profile != "CUSTOM":
        self.scanner_profile = "CUSTOM"


class M2SSettings(bpy.types.PropertyGroup):
    scanner_profile: EnumProperty(
        name="Scanner profile",
        items=profiles._profile_items,
        default="PREVIEW",
        update=_scanner_profile,
    )
    mesher: EnumProperty(
        name="Surface method",
        items=[
            (
                "GUIDED",
                "Guided surface",
                "Preserve reference connectivity; adaptive edges and bounded observation fit",
            ),
            (
                "VOLUME",
                "Observation volume",
                "Point-cloud OpenVDB experiment; can leave fitting cracks",
            ),
        ],
        default="GUIDED",
        update=_mark_custom,
    )
    memory_policy: EnumProperty(
        name="When RAM is tight",
        items=[
            (
                "AUTO",
                "Fit available RAM",
                "Reduce the effective triangle target, report it, and continue",
            ),
            (
                "STRICT",
                "Keep requested target",
                "Stop before capture if the request exceeds the safe RAM allowance",
            ),
        ],
        default="AUTO",
        update=_mark_custom,
    )
    observation_cap: IntProperty(
        name="Guided observation cap",
        default=1_000_000,
        min=1000,
        max=10_000_000,
        update=_mark_custom,
    )
    observed_fit: FloatProperty(
        name="Observation fit strength", default=0.25, min=0, max=1, update=_mark_custom
    )
    fairing_iterations: IntProperty(
        name="Interior fairing passes", default=2, min=0, max=8, update=_mark_custom
    )
    plan_json: StringProperty(default="", options={"SKIP_SAVE"})
    advanced_section: EnumProperty(
        name="Advanced section",
        items=[
            ("SHAPE", "Shape & edges", ""),
            ("CAPTURE", "Acquisition", ""),
            ("MEMORY", "RAM & output", ""),
            ("VOLUME", "Volume experiment", ""),
        ],
        default="SHAPE",
    )
    scope: EnumProperty(
        name="Capture",
        items=[
            ("ACTIVE", "Active part", "Capture the active mesh"),
            ("SELECTED", "Selected exterior parts", "One shared budget across selected mesh parts"),
        ],
        default="ACTIVE",
        update=_mark_custom,
    )
    surface_mode: EnumProperty(
        name="Reference surface",
        items=[
            (
                "SMOOTH",
                "Smooth inferred geometry",
                "Curved PN geometry; protects creases and original boundary lines",
            ),
            (
                "ORIGINAL",
                "Original geometry",
                "Scan the actual source facets without inferring a curved surface",
            ),
        ],
        default="SMOOTH",
        update=_mark_custom,
    )
    smooth_levels: IntProperty(
        name="Smooth reference refinement",
        default=3,
        min=0,
        max=5,
        description="PN reference tessellation; capped separately from scan output density",
        update=_mark_custom,
    )
    smooth_strength: FloatProperty(
        name="Smooth strength",
        default=0.7,
        min=0,
        max=1,
        description="Interpolate toward curved geometry; zero scans the original planes",
        update=_mark_custom,
    )
    boundary_recovery: BoolProperty(
        name="Recover panel boundaries",
        default=True,
        description="Fit true perimeter vertices and restore crossed source corners",
        update=_mark_custom,
    )
    custom_profile: EnumProperty(name="Saved profile", items=profiles.custom_items)
    profile_name: StringProperty(name="Profile name", default="My scanner")
    profile_tools: BoolProperty(name="My profiles", default=False)
    color_overlay: EnumProperty(
        name="Colors",
        items=[
            ("DETAIL", "Refinement demand", "Blue: low; cyan/yellow/red: higher refinement demand"),
            (
                "CURVATURE",
                "Curvature signal",
                "Normalized source face-angle signal; not a CAD surface classifier",
            ),
        ],
        default="DETAIL",
    )
    acquisition: EnumProperty(
        name="Acquisition",
        items=[
            (
                "RAYS",
                "Virtual rays",
                "Actual hits from local scanner positions oriented to each region",
            ),
            (
                "UNIFORM",
                "Perfect surface sampling",
                "Zero-noise surface measurements without visibility filtering",
            ),
        ],
        default="RAYS",
        update=_mark_custom,
    )
    resolution: EnumProperty(
        name="Resolution",
        items=[
            (
                "TARGET",
                "Triangle budget",
                "Estimate voxel spacing from source area; adaptively reduce toward this budget",
            ),
            ("VOXEL", "Voxel size", "Set a measurable spatial resolution in scene distance units"),
        ],
        default="TARGET",
        update=_mark_custom,
    )
    triangles_million: FloatProperty(
        name="Total triangles (M)",
        default=0.08,
        min=0.01,
        max=100.0,
        precision=2,
        description="Approximate budget; a target does not guarantee a particular face count",
        update=_mark_custom,
    )
    voxel_size: FloatProperty(
        name="Voxel size",
        default=0.001,
        min=1e-9,
        soft_max=1.0,
        subtype="DISTANCE",
        unit="LENGTH",
        precision=6,
        description="World-space voxel side length; not changed by object scale",
        update=_mark_custom,
    )
    samples: FloatProperty(
        name="Observation density",
        default=2.0,
        min=1.0,
        max=6.0,
        precision=1,
        description="Samples per estimated capture triangle; higher values reduce holes at fine features",
        update=_mark_custom,
    )
    adaptive: FloatProperty(
        name="Adaptive detail",
        default=0.65,
        min=0,
        max=1.0,
        precision=2,
        description="Capture extra geometry, then simplify flat regions while protecting features. Zero disables reduction",
        update=_mark_custom,
    )
    detail_boost: FloatProperty(
        name="Curvature sampling boost",
        default=3.0,
        min=0,
        max=5.0,
        description="Extra observations on faces near curvature transitions; zero gives the uniform baseline",
        update=_mark_custom,
    )
    crease_angle: FloatProperty(
        name="Crease angle",
        default=math.radians(35),
        min=math.radians(1),
        max=math.pi,
        subtype="ANGLE",
        description="Identify real changes in face normal, rather than every triangle edge",
        update=_mark_custom,
    )
    snap_radius: FloatProperty(
        name="Edge recovery (voxels)",
        default=0.0,
        min=0,
        max=1.2,
        description="Snap nearby recovered surface vertices onto observed crease and boundary lines. Zero disables snapping",
        update=_mark_custom,
    )
    radius_voxels: FloatProperty(
        name="Fusion radius (voxels)",
        default=2.5,
        min=1.2,
        max=4.0,
        description="Observation support radius. Large values can fuse narrow gaps and neighbouring sheets",
        update=_mark_custom,
    )
    standoff_voxels: FloatProperty(
        name="Scanner standoff (voxels)",
        default=4.0,
        min=0.5,
        max=30.0,
        description="Virtual local scanner distance. Reduce for densely overlapping surfaces",
        update=_mark_custom,
    )
    output: EnumProperty(
        name="Reconstruction",
        items=[
            (
                "SURFACE",
                "Measured surface",
                "Keep the positive observed skin; preserve open panel boundaries where resolution permits",
            ),
            (
                "ENVELOPE",
                "Capture envelope",
                "Keep the complete two-sided unsigned volume; adds an offset and is not a print wall",
            ),
        ],
        default="SURFACE",
        update=_mark_custom,
    )
    fidelity: EnumProperty(
        name="Surface fitting",
        items=[
            (
                "SOURCE",
                "Source fidelity",
                "Fit new topology to the chosen original or curved scanner reference",
            ),
            (
                "SURFELS",
                "Measured tangent planes",
                "Fit using acquired points and normals only; approximate experimental surface fitting",
            ),
            ("NONE", "Raw volume surface", "Leave the observed outer volume offset intact"),
        ],
        default="SOURCE",
        update=_mark_custom,
    )
    keep_cloud: BoolProperty(
        name="Keep observation cloud",
        default=False,
        description="Keep a hidden vertex-only object for PLY export and comparison",
        update=_mark_custom,
    )
    rescan_reference: BoolProperty(
        name="Rescan original source",
        default=True,
        description="When a generated scan is active, capture its original reference again",
        update=_mark_custom,
    )
    hide_source: BoolProperty(
        name="Hide source after capture",
        default=True,
        description="Only hide the original in the viewport; geometry is not modified",
        update=_mark_custom,
    )
    use_modifiers: BoolProperty(
        name="Capture evaluated modifiers", default=True, update=_mark_custom
    )
    clean_source: BoolProperty(
        name="Clean reference copy",
        default=True,
        description="Weld coincident vertices and repair normals on a temporary copy",
        update=_mark_custom,
    )
    marked_edges: BoolProperty(
        name="Include marked sharp edges",
        default=False,
        description="May capture unwanted triangulation if every imported edge is marked sharp",
        update=_mark_custom,
    )
    memory_gb: FloatProperty(
        name="Memory budget (GiB)",
        default=8.0,
        min=1.0,
        max=128.0,
        description="Conservative estimated peak limit, not an operating-system RAM reservation",
        update=_mark_custom,
    )
    seed: IntProperty(name="Sampling seed", default=42, min=0, update=_mark_custom)
    advanced: BoolProperty(name="Advanced settings", default=False)
    busy: BoolProperty(default=False, options={"SKIP_SAVE"})
    progress: FloatProperty(default=0, min=0, max=1, options={"SKIP_SAVE"})
    status: StringProperty(
        default="Select one component and start with Preview", options={"SKIP_SAVE"}
    )
    last_error: StringProperty(default="", options={"SKIP_SAVE"})
    last_report: StringProperty(default="")
    last_cloud: PointerProperty(type=bpy.types.Object)

    def options(self):
        return ScanOptions(
            mesher=self.mesher,
            memory_policy=self.memory_policy,
            observation_cap=self.observation_cap,
            observed_fit=self.observed_fit,
            fairing_iterations=self.fairing_iterations,
            surface_mode=self.surface_mode,
            smooth_levels=self.smooth_levels,
            smooth_strength=self.smooth_strength,
            boundary_recovery=self.boundary_recovery,
            acquisition=self.acquisition,
            target_triangles=round(self.triangles_million * 1_000_000),
            resolution=("TARGET" if self.mesher == "GUIDED" else self.resolution),
            voxel_world=self.voxel_size,
            samples_per_triangle=self.samples,
            adaptive=self.adaptive,
            detail_boost=self.detail_boost,
            crease_angle=self.crease_angle,
            snap_radius=self.snap_radius,
            radius_voxels=self.radius_voxels,
            standoff_voxels=self.standoff_voxels,
            output=("SURFACE" if self.mesher == "GUIDED" else self.output),
            fidelity=("SOURCE" if self.mesher == "GUIDED" else self.fidelity),
            keep_cloud=self.keep_cloud,
            hide_source=self.hide_source,
            use_modifiers=self.use_modifiers,
            clean_source=self.clean_source,
            marked_edges=self.marked_edges,
            memory_gb=self.memory_gb,
            seed=self.seed,
        )


class M2S_OT_capture(bpy.types.Operator):
    bl_idname = "object.mesh2scan_capture"
    bl_label = "Capture mesh2scan"
    bl_description = "Acquire new observations and reconstruct a separate triangular scan mesh"
    bl_options = {"REGISTER", "UNDO"}

    @classmethod
    def poll(cls, context):
        s = context.scene.mesh2scan_settings
        objects = context.selected_objects if s.scope == "SELECTED" else [context.active_object]
        return (
            context.mode == "OBJECT"
            and any(
                obj is not None and obj.type == "MESH" and not obj.get("vgc_cloud")
                for obj in objects
            )
            and _active_capture(context.scene) is None
        )

    def _start(self, context):
        self._scene = context.scene
        self._scene_key = self._scene.as_pointer()
        self._wm = context.window_manager
        self._timer = None
        self._steps = None
        self._job = None
        self._progress_started = False
        self._finished = False
        self._owns_capture = False
        if _active_capture(self._scene) is not None:
            raise ScanError("A capture is already running in this scene.")
        _active_captures[self._scene_key] = self
        self._owns_capture = True
        settings = self._scene.mesh2scan_settings
        settings.last_error = ""
        settings.progress = 0
        settings.status = "Planning capture"
        settings.busy = True
        sources = collect_sources(context, settings)
        self._job = (
            BatchScanJob(context, sources, settings.options())
            if settings.scope == "SELECTED"
            else ScanJob(context, sources[0], settings.options())
        )
        self._steps = self._job.stages()
        self._progress_started = True
        self._wm.progress_begin(0, 100)

    def _redraw(self, context):
        screen = getattr(context, "screen", None)
        if screen:
            for area in screen.areas:
                if area.type == "VIEW_3D":
                    area.tag_redraw()

    def _finish(self, context, failed=False):
        if getattr(self, "_finished", False):
            return
        self._finished = True
        cleanup_errors = []
        steps, self._steps = getattr(self, "_steps", None), None
        if steps is not None:
            try:
                steps.close()
            except Exception as error:
                cleanup_errors.append(str(error))
        timer, self._timer = getattr(self, "_timer", None), None
        if timer is not None:
            try:
                self._wm.event_timer_remove(timer)
            except Exception as error:
                cleanup_errors.append(str(error))
        if getattr(self, "_progress_started", False):
            self._progress_started = False
            try:
                self._wm.progress_end()
            except Exception as error:
                cleanup_errors.append(str(error))
        settings = None
        if getattr(self, "_owns_capture", False):
            self._owns_capture = False
            if _active_captures.get(self._scene_key) is self:
                _active_captures.pop(self._scene_key)
            try:
                settings = self._scene.mesh2scan_settings
                settings.busy = False
                if failed:
                    settings.progress = 0
            except ReferenceError:
                pass
        if not failed and settings is not None:
            settings.last_report = json.dumps(self._job.report)
            settings.last_cloud = self._job.cloud if self._job.options.keep_cloud else None
            triangles = self._job.report["triangles"]
            self.report({"INFO"}, f"Virtual scan created: {triangles:,} triangles")
        if cleanup_errors:
            message = "Capture cleanup: " + "; ".join(cleanup_errors)
            if settings is not None:
                settings.last_error = message
            self.report({"WARNING"}, message)
        self._redraw(context)

    def _advance(self, context):
        settings = self._scene.mesh2scan_settings
        progress, label = next(self._steps)
        settings.progress = progress
        settings.status = label
        self._wm.progress_update(progress * 100)
        self._redraw(context)
        return progress

    def _step(self, context):
        try:
            if self._advance(context) >= 1:
                self._finish(context)
                return {"FINISHED"}
        except StopIteration:
            self._finish(context)
            return {"FINISHED"}
        except Exception as error:
            return self._fail(context, error)
        return {"RUNNING_MODAL"}

    def _fail(self, context, error):
        if getattr(self, "_owns_capture", False):
            try:
                settings = self._scene.mesh2scan_settings
                settings.last_error = str(error)
                settings.status = "Capture stopped"
            except ReferenceError:
                pass
        self._finish(context, failed=True)
        self.report({"WARNING"}, str(error))
        return {"CANCELLED"}

    def execute(self, context):
        # Synchronous path for scripts and headless integration tests.
        try:
            self._start(context)
            while True:
                outcome = self._step(context)
                if outcome != {"RUNNING_MODAL"}:
                    return outcome
        except Exception as error:
            return self._fail(context, error)

    def invoke(self, context, event):
        if context.window is None:
            return self.execute(context)
        try:
            self._start(context)
            # Validate the source and display the first stage immediately.
            outcome = self._step(context)
            if outcome != {"RUNNING_MODAL"}:
                return outcome
            self._timer = self._wm.event_timer_add(0.15, window=context.window)
            if self._wm.modal_handler_add(self) is False:
                raise ScanError(
                    "Blender could not start the capture event handler. Try again in the 3D View."
                )
            self._redraw(context)
            return {"RUNNING_MODAL"}
        except Exception as error:
            return self._fail(context, error)

    def cancel(self, context):
        if not getattr(self, "_finished", False):
            try:
                self._scene.mesh2scan_settings.status = "Capture cancelled; source retained"
            except ReferenceError:
                pass
            self._finish(context, failed=True)

    def modal(self, context, event):
        try:
            if self._finished:
                return {"CANCELLED"}
            if context.scene != self._scene or context.view_layer != self._job.view_layer:
                self.cancel(context)
                return {"CANCELLED"}
            if event.type == "ESC":
                self.cancel(context)
                return {"CANCELLED"}
            # Blender Event exposes the event type, not an originating Timer.
            if event.type == "TIMER":
                return self._step(context)
            # Let Blender receive mouse events so the Cancel button is clickable.
            return {"PASS_THROUGH"}
        except Exception as error:
            return self._fail(context, error)


class M2S_OT_cancel_capture(bpy.types.Operator):
    bl_idname = "object.mesh2scan_cancel"
    bl_label = "Cancel Capture"
    bl_description = "Cancel between capture stages and restore the controls"

    def execute(self, context):
        capture = _active_capture(context.scene)
        if capture is not None:
            capture.cancel(context)
        else:
            settings = context.scene.mesh2scan_settings
            settings.busy = False
            settings.progress = 0
            settings.status = "Capture reset; start with Preview"
            settings.last_error = ""
        return {"FINISHED"}


def _cloud(context):
    obj = context.active_object
    if obj is not None and obj.get("vgc_cloud"):
        return obj
    if obj is not None and obj.get("vgc_scan"):
        return bpy.data.objects.get(obj.get("vgc_cloud_name", ""))
    return context.scene.mesh2scan_settings.last_cloud


class M2S_OT_export_cloud(bpy.types.Operator, ExportHelper):
    bl_idname = "export_mesh.mesh2scan_points"
    bl_label = "Export Observation Cloud (PLY)"
    filename_ext = ".ply"
    filter_glob: StringProperty(default="*.ply", options={"HIDDEN"})
    scale_factor: FloatProperty(
        name="Coordinate multiplier",
        default=1.0,
        min=1e-6,
        max=1e6,
        description="1 retains world values; 1000 converts metre-valued coordinates to millimetres",
    )

    @classmethod
    def poll(cls, context):
        return _active_capture(context.scene) is None and _cloud(context) is not None

    def invoke(self, context, event):
        cloud = _cloud(context)
        if cloud and not self.filepath:
            self.filepath = bpy.path.clean_name(cloud.name) + ".ply"
        return ExportHelper.invoke(self, context, event)

    def execute(self, context):
        obj = _cloud(context)
        if obj is None:
            self.report({"ERROR"}, "Capture with Keep observation cloud enabled first")
            return {"CANCELLED"}
        try:
            count = export_cloud_ply(obj, self.filepath, self.scale_factor)
            self.report({"INFO"}, f"Exported {count:,} synthetic observations")
            return {"FINISHED"}
        except Exception as error:
            self.report({"ERROR"}, str(error))
            return {"CANCELLED"}


class M2S_OT_export_report(bpy.types.Operator, ExportHelper):
    bl_idname = "export_mesh.mesh2scan_report"
    bl_label = "Export Capture Report (JSON)"
    filename_ext = ".json"
    filter_glob: StringProperty(default="*.json", options={"HIDDEN"})

    @classmethod
    def poll(cls, context):
        return (
            bool(context.scene.mesh2scan_settings.last_report)
            and _active_capture(context.scene) is None
        )

    def execute(self, context):
        obj = context.active_object
        text = (obj.get("vgc_batch_report") or obj.get("vgc_report")) if obj is not None else None
        text = text or context.scene.mesh2scan_settings.last_report
        try:
            Path(self.filepath).write_text(json.dumps(json.loads(text), indent=2), encoding="utf-8")
            self.report({"INFO"}, "Capture report exported")
            return {"FINISHED"}
        except Exception as error:
            self.report({"ERROR"}, str(error))
            return {"CANCELLED"}


class M2S_OT_plan(bpy.types.Operator):
    bl_idname = "object.mesh2scan_plan"
    bl_label = "Check RAM & Target"
    bl_description = "Show the affordable target before acquiring any observations"

    @classmethod
    def poll(cls, context):
        return M2S_OT_capture.poll(context)

    def execute(self, context):
        settings = context.scene.mesh2scan_settings
        try:
            options = settings.options()
            if options.resolution != "TARGET":
                raise ScanError(
                    "Budget planning uses Triangle budget mode. Switch resolution in Advanced."
                )
            measurements = [
                measure_component(context, obj, options)
                for obj in collect_sources(context, settings)
            ]
            plan = fit_plan(measurements, options.target_triangles, options, allocate_budget)
            settings.plan_json = json.dumps(plan)
            self.report(
                {"INFO"},
                f"Plan: {plan['effective_triangles']:,} triangles, about {plan['estimated_peak_gib']:.1f} GiB RAM",
            )
            return {"FINISHED"}
        except Exception as error:
            self.report({"WARNING"}, str(error))
            return {"CANCELLED"}


class M2S_OT_profile(bpy.types.Operator):
    bl_idname = "object.mesh2scan_profile"
    bl_label = "Scanner Profile"
    action: EnumProperty(items=[(a, a.title(), "") for a in ("SAVE", "LOAD", "DELETE", "REFRESH")])

    @classmethod
    def poll(cls, context):
        return _active_capture(context.scene) is None

    def execute(self, context):
        s = context.scene.mesh2scan_settings
        try:
            data = profiles.refresh_profiles()
            if self.action == "SAVE":
                name = s.profile_name.strip()
                if not name or name == "NONE" or len(name) > 80:
                    raise ValueError("Choose a profile name of 1–80 characters, other than NONE.")
                data[name] = profiles.snapshot(s)
                profiles.write_profiles(data)
                s.custom_profile = name
            elif self.action == "LOAD":
                profiles.load_profile(s, s.custom_profile)
            elif self.action == "DELETE":
                if s.custom_profile not in data:
                    raise ValueError("Choose a saved profile to delete.")
                del data[s.custom_profile]
                profiles.write_profiles(data)
                s.custom_profile = next(iter(data), "NONE")
            else:
                s.custom_profile = next(iter(data), "NONE")
            self.report({"INFO"}, "Scanner profiles updated")
            return {"FINISHED"}
        except Exception as error:
            self.report({"ERROR"}, str(error))
            return {"CANCELLED"}


class M2S_OT_colors(bpy.types.Operator):
    bl_idname = "object.mesh2scan_colors"
    bl_label = "Show Scan Colors"
    restore: BoolProperty(default=False)

    @classmethod
    def poll(cls, context):
        obj = context.active_object
        return (
            obj is not None
            and bool(obj.get("vgc_scan"))
            and context.area is not None
            and context.area.type == "VIEW_3D"
            and _active_capture(context.scene) is None
        )

    def execute(self, context):
        shading = context.space_data.shading
        shading.type = "SOLID"
        if self.restore:
            shading.color_type = "MATERIAL"
        else:
            name = (
                "Scan Curvature"
                if context.scene.mesh2scan_settings.color_overlay == "CURVATURE"
                else "Scan Detail"
            )
            objects = [
                obj
                for obj in context.selected_objects
                if obj.type == "MESH" and obj.get("vgc_scan")
            ]
            for obj in objects:
                if name in obj.data.color_attributes:
                    obj.data.color_attributes.active_color_name = name
            shading.color_type = "VERTEX"
        return {"FINISHED"}


class M2S_PT_importance(bpy.types.Panel):
    bl_label = "mesh2scan Importance"
    bl_idname = "M2S_PT_importance"
    bl_space_type = "PROPERTIES"
    bl_region_type = "WINDOW"
    bl_context = "object"

    @classmethod
    def poll(cls, context):
        return context.object is not None and context.object.type == "MESH"

    def draw(self, context):
        self.layout.prop(context.object, "mesh2scan_importance")
        self.layout.label(text="1 = normal; higher gets more of the shared budget")


class M2S_PT_panel(bpy.types.Panel):
    bl_label = "mesh2scan"
    bl_idname = "M2S_PT_panel"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "mesh2scan"

    def draw(self, context):
        layout, s = self.layout, context.scene.mesh2scan_settings
        busy = _active_capture(context.scene) is not None
        obj = context.active_object
        layout.label(text="mesh2scan", icon="MESH_DATA")
        layout.label(text="an experimental run.")
        controls = layout.column()
        controls.enabled = not busy
        box = controls.box()
        box.label(text="1 · Choose a capture", icon="PREFERENCES")
        box.prop(s, "scanner_profile", text="Profile")
        box.prop(s, "scope", text="Parts")
        box.prop(s, "mesher", text="Method")
        box.prop(s, "surface_mode", text="Shape")
        if s.mesher == "GUIDED":
            box.label(text="Connected boundaries · adaptive triangles", icon="CHECKMARK")
        else:
            box.label(text="Volume experiment: inspect crease cracks", icon="INFO")
        box = controls.box()
        box.label(text="2 · Set the total budget", icon="MOD_TRIANGULATE")
        box.prop(
            s,
            "triangles_million"
            if s.mesher == "GUIDED" or s.resolution == "TARGET"
            else "voxel_size",
        )
        if s.scope == "SELECTED":
            count = sum(
                o.type == "MESH" and not o.get("vgc_cloud") for o in context.selected_objects
            )
            box.label(text=f"{count} selected parts share this total")
        box.prop(s, "memory_policy", text="RAM policy")
        box.operator(M2S_OT_plan.bl_idname, icon="VIEWZOOM")
        if s.plan_json:
            plan = json.loads(s.plan_json)
            box.label(text=f"Requested: {plan['requested_triangles']:,} triangles")
            box.label(
                text=f"Planned: {plan['effective_triangles']:,} triangles",
                icon="CHECKMARK" if plan["fits"] else "ERROR",
            )
            box.label(
                text=f"RAM estimate: {plan['estimated_peak_gib']:.1f} / {plan['memory_limit_gib']:.1f} GiB"
            )
            if plan["adjusted"]:
                box.label(text="Target adjusted; requested value stays editable", icon="INFO")
            if not plan["fits"]:
                box.label(text="Lower target or use Fit available RAM")
        if busy:
            box = layout.box()
            box.label(text="Capture in progress", icon="TIME")
            box.progress(factor=s.progress, type="BAR", text=f"{s.progress:.0%}")
            import textwrap

            for line in textwrap.wrap(s.status, 44):
                box.label(text=line)
            box.operator(M2S_OT_cancel_capture.bl_idname, icon="CANCEL")
            box.label(text="Cancel takes effect between native stages")
        else:
            row = controls.row()
            row.scale_y = 1.6
            row.operator(M2S_OT_capture.bl_idname, text="3 · Capture mesh2scan", icon="MOD_REMESH")
            if s.busy:
                controls.operator(
                    M2S_OT_cancel_capture.bl_idname,
                    text="Reset Interrupted Capture",
                    icon="FILE_REFRESH",
                )
            if not M2S_OT_capture.poll(context):
                layout.label(text="Select mesh parts in Object Mode", icon="INFO")
            if s.last_error:
                box = layout.box()
                box.label(text="Capture needs attention", icon="INFO")
                import textwrap

                for line in textwrap.wrap(s.last_error, 44):
                    box.label(text=line)
            elif s.last_report:
                report = json.loads(s.last_report)
                box = layout.box()
                box.label(text=f"{report['triangles']:,} triangles", icon="CHECKMARK")
                box.label(
                    text=f"{report.get('component_count', 1)} parts · {report['seconds']:.1f}s"
                )
                box.label(
                    text=f"Open edges: {report['boundary_edges']:,} · nonmanifold: {report['nonmanifold_edges']:,}"
                )
                plan = report.get("plan")
                if plan and plan.get("adjusted"):
                    box.label(
                        text=f"RAM-adjusted target: {plan['effective_triangles']:,}", icon="INFO"
                    )
                if obj is not None and obj.get("vgc_scan"):
                    box.prop(s, "color_overlay", text="Inspect")
                    row = box.row(align=True)
                    row.operator(M2S_OT_colors.bl_idname, text="Show Colors", icon="SHADING_SOLID")
                    row.operator(M2S_OT_colors.bl_idname, text="Materials").restore = True
                row = box.row(align=True)
                row.operator(M2S_OT_export_report.bl_idname, text="Report", icon="TEXT")
                row.operator(M2S_OT_export_cloud.bl_idname, text="Points", icon="EXPORT")
        controls.prop(s, "advanced", icon="TRIA_DOWN" if s.advanced else "TRIA_RIGHT", emboss=False)
        if s.advanced:
            box = controls.box()
            box.prop(s, "advanced_section", text="")
            fields = {
                "SHAPE": (
                    "smooth_levels",
                    "smooth_strength",
                    "fairing_iterations",
                    "adaptive",
                    "detail_boost",
                    "crease_angle",
                    "marked_edges",
                    "observed_fit",
                    "use_modifiers",
                    "clean_source",
                ),
                "CAPTURE": ("acquisition", "samples", "observation_cap", "standoff_voxels", "seed"),
                "MEMORY": ("memory_gb", "keep_cloud", "hide_source", "rescan_reference"),
                "VOLUME": (
                    "resolution",
                    "output",
                    "fidelity",
                    "radius_voxels",
                    "snap_radius",
                    "boundary_recovery",
                ),
            }
            if s.advanced_section == "VOLUME" and s.mesher == "GUIDED":
                box.label(text="Select Observation volume to enable these")
                box.enabled = False
            for prop in fields[s.advanced_section]:
                box.prop(s, prop)
            if s.advanced_section == "MEMORY":
                box.label(text="Limit is RAM, not disk space")
            if s.advanced_section == "SHAPE":
                box.label(text="Per-part importance: Object Properties")
        controls.prop(
            s, "profile_tools", icon="TRIA_DOWN" if s.profile_tools else "TRIA_RIGHT", emboss=False
        )
        if s.profile_tools:
            box = controls.box()
            box.prop(s, "custom_profile")
            row = box.row(align=True)
            for action, label in (("LOAD", "Load"), ("REFRESH", "Refresh"), ("DELETE", "Delete")):
                row.operator(M2S_OT_profile.bl_idname, text=label).action = action
            box.prop(s, "profile_name")
            box.operator(
                M2S_OT_profile.bl_idname, text="Save Current Profile", icon="FILE_TICK"
            ).action = "SAVE"
        layout.separator()
        layout.label(text="0.0.1 · synthetic geometry · experimental")


@persistent
def _stop_captures(_dummy):
    for capture in tuple(_active_captures.values()):
        capture.cancel(bpy.context)


_handlers = (
    (bpy.app.handlers.load_pre, _stop_captures),
    (bpy.app.handlers.undo_pre, _stop_captures),
    (bpy.app.handlers.redo_pre, _stop_captures),
)

_classes = (
    M2SSettings,
    M2S_OT_capture,
    M2S_OT_cancel_capture,
    M2S_OT_export_cloud,
    M2S_OT_export_report,
    M2S_OT_plan,
    M2S_OT_profile,
    M2S_OT_colors,
    M2S_PT_importance,
    M2S_PT_panel,
)


def register():
    for cls in _classes:
        bpy.utils.register_class(cls)
    bpy.types.Scene.mesh2scan_settings = PointerProperty(type=M2SSettings)
    bpy.types.Object.mesh2scan_importance = FloatProperty(
        name="Scan importance", default=1.0, min=0.125, max=8.0
    )
    for handlers, callback in _handlers:
        if callback not in handlers:
            handlers.append(callback)
    # Extension registration runs under RestrictBlend: scene data is unavailable.
    # Draw/poll only read state. Resetting stale flags is an operator action.


def unregister():
    _stop_captures(None)
    for handlers, callback in _handlers:
        if callback in handlers:
            handlers.remove(callback)
    if hasattr(bpy.types.Object, "mesh2scan_importance"):
        del bpy.types.Object.mesh2scan_importance
    if hasattr(bpy.types.Scene, "mesh2scan_settings"):
        del bpy.types.Scene.mesh2scan_settings
    for cls in reversed(_classes):
        bpy.utils.unregister_class(cls)
