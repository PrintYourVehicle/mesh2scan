# SPDX-License-Identifier: GPL-3.0-or-later
"""Read-only RAM estimates and an explicit affordable target."""

import ctypes
import os


def available_memory_gib():
    try:
        if os.name == "nt":

            class MemoryStatus(ctypes.Structure):
                _fields_ = [("length", ctypes.c_ulong), ("load", ctypes.c_ulong)] + [
                    (name, ctypes.c_ulonglong)
                    for name in (
                        "total_phys",
                        "avail_phys",
                        "total_page",
                        "avail_page",
                        "total_virtual",
                        "avail_virtual",
                        "extended",
                    )
                ]

            status = MemoryStatus()
            status.length = ctypes.sizeof(status)
            if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
                return status.avail_phys / 1024**3
        elif os.path.exists("/proc/meminfo"):
            with open("/proc/meminfo", encoding="ascii") as handle:
                for line in handle:
                    if line.startswith("MemAvailable:"):
                        available = float(line.split()[1]) * 1024
                        # Container limits may be much lower than host RAM.
                        try:
                            with open("/sys/fs/cgroup/memory.max") as f:
                                limit = f.read().strip()
                            with open("/sys/fs/cgroup/memory.current") as f:
                                used = int(f.read())
                            if limit != "max":
                                available = min(available, max(0, int(limit) - used))
                        except (OSError, ValueError):
                            pass
                        return available / 1024**3
        else:
            pages = os.sysconf("SC_AVPHYS_PAGES")
            size = os.sysconf("SC_PAGE_SIZE")
            return pages * size / 1024**3
    except (OSError, ValueError, AttributeError):
        pass
    return None


def is_guided(options):
    return options.mesher == "GUIDED" and options.output == "SURFACE"


def estimate_peak(target, input_triangles, options, retained=0.0):
    levels = options.smooth_levels if options.surface_mode == "SMOOTH" else 0
    limit = min(2_000_000, max(500, int(target * 0.45))) if is_guided(options) else 2_000_000
    while levels and input_triangles * 4**levels > limit:
        levels -= 1
    reference = input_triangles * 4**levels
    if is_guided(options):
        # NumPy incidence/split arrays, Blender meshes, diagnostics and bounded
        # observation cloud coexist. This remains an estimate, not a guarantee.
        observations = min(
            options.observation_cap, max(1000, int(target * options.samples_per_triangle))
        )
        peak = target * 680 + observations * 100 + reference * 600
    else:
        expected = target * (1 + options.adaptive * 1.8)
        observations = max(1000, int(expected * options.samples_per_triangle))
        peak = expected * 1900 + observations * 100 + reference * 800
    return peak / 1024**3 + retained + 0.25


def fit_plan(measurements, requested, options, allocate):
    available = available_memory_gib()
    limit = min(options.memory_gb, available * 0.70) if available else options.memory_gb
    limit = max(0.25, limit)
    user_requested = int(requested)
    minimum = sum(
        max(500, m["input_triangles"]) if is_guided(options) else 500 for m in measurements
    )

    def check(total):
        allocations = allocate(measurements, total, preserve_input=is_guided(options))
        peak, retained = 0.0, options.retained_gb
        for m, target in zip(measurements, allocations):
            peak = max(peak, estimate_peak(target, m["input_triangles"], options, retained))
            observations = (
                min(options.observation_cap, int(target * options.samples_per_triangle))
                if is_guided(options)
                else int(target * (1 + options.adaptive * 1.8) * options.samples_per_triangle)
            )
            retained += (target * 220 + (observations * 100 if options.keep_cloud else 0)) / 1024**3
        return peak, allocations

    requested = max(minimum, int(requested))
    peak, allocations = check(requested)
    effective = requested
    if peak > limit and options.memory_policy == "AUTO":
        low, high = minimum, requested
        if check(low)[0] <= limit:
            for _ in range(25):
                mid = (low + high + 1) // 2
                if check(mid)[0] <= limit:
                    low = mid
                else:
                    high = mid - 1
            effective = low
            peak, allocations = check(effective)
    return {
        "requested_triangles": user_requested,
        "effective_triangles": effective,
        "allocations": allocations,
        "estimated_peak_gib": peak,
        "memory_limit_gib": limit,
        "available_ram_gib": available,
        "fits": peak <= limit,
        "adjusted": effective != user_requested,
        "reason": "SOURCE_MINIMUM"
        if effective > user_requested
        else ("RAM_LIMIT" if effective < user_requested else "UNCHANGED"),
        "mesher": options.mesher,
    }
