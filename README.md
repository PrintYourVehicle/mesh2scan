# mesh2scan

A Blender extension that performs a virtual 3D scan of a mesh you supply. It ray-casts from virtual scanner positions to acquire real surface observations, then reconstructs a separate, densely triangulated scan mesh from those measurements. The original object is never modified.

> Experimental, pre-release (`0.0.1`). It scans the geometry you give it, so it measures the shape that is actually there — it does not invent detail that a game mesh only fakes with normal maps. Results can have open boundaries, are not watertight-guaranteed, and are not CAD/Class-A surfaces.

<img width="1672" height="941" alt="Car Scan_ From Metal to Mesh" src="https://github.com/user-attachments/assets/2be1aa1b-69fd-4de4-b120-80dcbd0a9cce" />


## Features

- **Virtual scanning**: casts rays from virtual scanner standoff positions and records actual hit points and normals (misses are discarded), or samples the surface directly for a zero-noise baseline. The original mesh stays untouched; the scan is a new object.
- **Guided surface** (default): keeps the reference connectivity and adaptively splits shared edges by length and detail toward a triangle budget, then applies a bounded observation fit. Preserves creases and open borders.
- **Observation volume** (experimental): fuses only the acquired points into a native OpenVDB volume and extracts an independent isosurface, then fits and simplifies it.
- **Smooth inferred geometry**: optional crease- and boundary-protected curved (PN-triangle) reference, so coarse source facets scan as a smooth surface. Switch to **Original shape** to scan the actual facets.
- **Shared triangle budget** across selected parts for whole-exterior captures, weighted by surface area, curvature, and per-part importance.
- **RAM preflight** that estimates peak memory and either auto-reduces the target or stops before capture.
- **Diagnostics**: refinement/curvature vertex colors, a per-capture JSON report (triangle count, open/nonmanifold edges, ray-hit fraction, sampled deviation from the source), and PLY export of the observation cloud.
- Presets from an 80k quick check up to a 25M shared exterior.

## Requirements

- Blender **4.5+** (checked on 4.5.4 LTS and 5.1.0).
- No extra runtime packages or platform-specific binaries.

## Install

1. Download `mesh2scan-0.0.1.zip` (the build output, **not** the source ZIP).
2. In Blender: **Preferences → Get Extensions → menu → Install from Disk**, pick the ZIP, and enable mesh2scan.
3. Open the 3D View sidebar (`N`) → **mesh2scan**.

## Usage

1. Select a mesh in Object Mode.
2. Pick **Quick check · 80k**, **Guided surface**, and **Smooth inferred geometry**.
3. Click **Check RAM & Target**, then **Capture mesh2scan**.
4. Inspect the result, colors, and report before raising the budget. Use **Original shape** to compare without inferred smoothing.

For a whole exterior, select the exterior parts and choose an Exterior preset. The triangle count is a shared total, not per part. Set per-part share in **Object Properties → mesh2scan Importance**.

## Build

Requires Python 3.11+ (standard library only).

```sh
python tools/build.py            # writes dist/mesh2scan-0.0.1.zip
python tools/build.py --source   # also writes the repository archive
```

## Develop

```sh
python -m venv .venv
# activate .venv, then:
python -m pip install bpy==5.1.0 -r requirements-dev.txt
ruff check .
python tests/blender_smoke.py
python tests/lifecycle.py
```

Use Python 3.11 with `bpy==4.5.4` for the Blender 4.5 line in a separate environment. Run the smoke/lifecycle checks in a fresh scene: they create and delete temporary objects.

## License

GPL-3.0-or-later. See [LICENSE](LICENSE).
