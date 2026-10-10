# SPDX-License-Identifier: GPL-3.0-or-later
"""Built-in scanner configurations and portable user JSON profiles."""

import json
from pathlib import Path
import bpy

FIELDS = (
    "mesher",
    "memory_policy",
    "observation_cap",
    "observed_fit",
    "fairing_iterations",
    "scope",
    "surface_mode",
    "smooth_levels",
    "smooth_strength",
    "boundary_recovery",
    "acquisition",
    "resolution",
    "triangles_million",
    "voxel_size",
    "samples",
    "adaptive",
    "detail_boost",
    "crease_angle",
    "snap_radius",
    "radius_voxels",
    "standoff_voxels",
    "output",
    "fidelity",
    "keep_cloud",
    "rescan_reference",
    "hide_source",
    "use_modifiers",
    "clean_source",
    "marked_edges",
    "memory_gb",
    "seed",
)

BASE = dict(
    scope="ACTIVE",
    surface_mode="SMOOTH",
    smooth_levels=3,
    smooth_strength=0.7,
    mesher="GUIDED",
    memory_policy="AUTO",
    observation_cap=1_000_000,
    observed_fit=0.25,
    fairing_iterations=2,
    boundary_recovery=True,
    acquisition="RAYS",
    resolution="TARGET",
    triangles_million=0.08,
    samples=2.0,
    adaptive=0.65,
    detail_boost=3.0,
    crease_angle=0.6108652382,
    snap_radius=0.0,
    radius_voxels=2.5,
    standoff_voxels=4.0,
    output="SURFACE",
    fidelity="SOURCE",
    keep_cloud=False,
    rescan_reference=True,
    hide_source=True,
    use_modifiers=True,
    clean_source=True,
    marked_edges=False,
    memory_gb=8.0,
    seed=42,
    voxel_size=0.001,
)
PROFILES = {
    "PREVIEW": ("Quick check · 80k", "Small guided capture; automatically fits available RAM", {}),
    "PANEL": (
        "Body panel · 500k",
        "Conservative smoothing and connected borders",
        {"triangles_million": 0.5},
    ),
    "DETAIL": (
        "Edges & trim · 1M",
        "Stronger adaptive density without crease snapping",
        {
            "triangles_million": 1.0,
            "adaptive": 0.8,
            "detail_boost": 5.0,
            "crease_angle": 0.4363323130,
        },
    ),
    "EXTERIOR5": (
        "Exterior · 5M total",
        "Selected parts share 5M, adjusted to available RAM",
        {"scope": "SELECTED", "triangles_million": 5.0, "memory_gb": 12.0},
    ),
    "EXTERIOR10": (
        "Exterior · 10M total",
        "Selected parts share 10M, adjusted to available RAM",
        {"scope": "SELECTED", "triangles_million": 10.0, "memory_gb": 16.0},
    ),
    "EXTERIOR25": (
        "Exterior · 25M requested",
        "Request 25M; the planner shows an affordable effective target",
        {"scope": "SELECTED", "triangles_million": 25.0, "memory_gb": 32.0},
    ),
    "VOLUME": (
        "Volume experiment · 80k",
        "Cloud-to-OpenVDB extraction; may leave crease cracks",
        {"mesher": "VOLUME", "smooth_strength": 0.5, "keep_cloud": True, "observed_fit": 0.0},
    ),
    "RAW": (
        "Original shape · 80k",
        "Guided capture without inferred smoothing",
        {"surface_mode": "ORIGINAL", "smooth_levels": 0, "fairing_iterations": 0},
    ),
}
_profile_items = [(key, label, description) for key, (label, description, _) in PROFILES.items()]
_profile_items.append(("CUSTOM", "Custom / edited", "Use current settings or save a user profile"))
_custom_items = [("NONE", "No saved profiles", "Save your current configuration first")]
_custom_profiles = {}


def apply_builtin(settings):
    if settings.scanner_profile in PROFILES:
        values = {**BASE, **PROFILES[settings.scanner_profile][2]}
        for field, value in values.items():
            setattr(settings, field, value)


def profile_directory():
    # Access user configuration only from explicit operators, never register/draw.
    root = bpy.utils.user_resource("CONFIG", path="mesh2scan", create=True)
    if not root:
        raise RuntimeError("Blender could not create a user profile directory.")
    return Path(root)


def refresh_profiles():
    global _custom_items, _custom_profiles
    path = profile_directory() / "scanner_profiles.json"
    data = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    if not isinstance(data, dict):
        raise ValueError("User profile file must contain a JSON object.")
    profiles = {
        key: value
        for key, value in data.items()
        if isinstance(key, str) and isinstance(value, dict)
    }
    items = [(name, name, "Saved scanner configuration") for name in sorted(profiles)]
    _custom_profiles = profiles
    # Keep the returned strings alive (Blender's dynamic EnumProperty rule).
    # Retain earlier lists while the module is registered too.
    _enum_history.append(_custom_items)
    _custom_items = items or [
        ("NONE", "No saved profiles", "Save your current configuration first")
    ]
    return profiles


_enum_history = []


def custom_items(_self, _context):
    return _custom_items


def snapshot(settings):
    return {name: getattr(settings, name) for name in FIELDS}


def write_profiles(profiles):
    path = profile_directory() / "scanner_profiles.json"
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(profiles, indent=2, ensure_ascii=False), encoding="utf-8")
    temporary.replace(path)
    refresh_profiles()


def load_profile(settings, name):
    profiles = refresh_profiles()
    if name not in profiles:
        raise ValueError("Choose a saved profile; use Refresh if the list is empty.")
    values = profiles[name]
    unknown = set(values) - set(FIELDS)
    if unknown:
        raise ValueError("Profile contains unsupported settings: " + ", ".join(sorted(unknown)))
    # Validate every field before applying so a corrupt profile cannot leave a
    # partially loaded configuration. RNA enforces supported enums/ranges.
    for field, value in values.items():
        prop = settings.bl_rna.properties[field]
        if prop.type == "ENUM" and value not in {item.identifier for item in prop.enum_items}:
            raise ValueError(f"Unsupported value for {field}: {value}")
        if prop.type in {"FLOAT", "INT"}:
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"Invalid numeric value for {field}")
            if prop.type == "INT" and not isinstance(value, int):
                raise ValueError(f"Invalid integer value for {field}")
            if not prop.hard_min <= value <= prop.hard_max:
                raise ValueError(f"Out-of-range value for {field}")
        elif prop.type == "BOOLEAN" and not isinstance(value, bool):
            raise ValueError(f"Invalid boolean value for {field}")
    settings.scanner_profile = "CUSTOM"
    for field, value in values.items():
        setattr(settings, field, value)
