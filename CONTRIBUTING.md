# Contributing

Keep Blender data access on the main thread. Registration must work under RestrictBlend; panel draw and operator poll must remain read-only. Do not derive running state from a saved busy flag. Preserve cancellation cleanup and source visibility/selection, including when changing scenes or disabling the extension.

Geometry changes need a small native Blender example reproducing the defect. For guided refinement, check boundary-loop count as well as nonmanifold and zero-area triangles: a mesh with holes can still have zero nonmanifold edges. Keep curve, feature, and topology claims separate. No broad test suite is needed for a label-only change.

Use `python tools/build.py` for an install ZIP. Run `ruff check .`, `python tests/blender_smoke.py` and `python tests/lifecycle.py` before contributing geometry. Use GPL-3.0-or-later compatible source and retain author/reference attribution. Never commit personal scanner profiles, game assets, virtual environments, or large scan outputs. Include a concise problem, behaviour change and validation in the pull request description.
