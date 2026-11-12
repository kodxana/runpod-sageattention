"""Every builder and runtime image pin agrees with matrix.json.

Each digest is written in four places: matrix.json, build.yml's
workflow_dispatch and workflow_call defaults, and release.yml's dispatch
defaults (which release.yml forwards to build.yml).  The workflows build and
test with the input values, so bumping matrix.json alone would silently build
a release on the old builder, or validate a wheel on an image it will not ship
in.  The workflow checks only prove each value is an @sha256 reference.
"""

from __future__ import annotations

import json
from pathlib import Path
import re
import unittest


ROOT = Path(__file__).resolve().parents[1]
WORKFLOWS = ROOT / ".github" / "workflows"
MATRIX = json.loads((ROOT / "matrix.json").read_text(encoding="utf-8"))

# matrix.json field -> workflow input prefix
FIELDS = {"builder_image": "builder_image", "comfyui_runtime_image": "runtime_image"}


def _section(text: str, trigger: str) -> str:
    """The inputs block of one `on:` trigger."""

    body = text.split(f"\n  {trigger}:\n", 1)[1]
    end = re.search(r"\n  \S", body)
    return body[: end.start()] if end else body


def _defaults(section: str) -> dict[str, str]:
    """input name -> default, for the 6-space-indented inputs of a section."""

    defaults = {}
    for match in re.finditer(
        r"(?m)^      (?P<name>[a-z0-9_]+):\n(?P<body>(?:        .*\n)+)", section
    ):
        default = re.search(r'(?m)^        default: "?(?P<value>[^"\n]*)"?$', match["body"])
        if default:
            defaults[match["name"]] = default["value"]
    return defaults


def _cuda_suffix(build: dict) -> str:
    return build["id"].rsplit("-", 1)[1]  # cp312-torch2.10.0-cu128 -> cu128


class ImagePinsAgreeWithMatrixTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        build = (WORKFLOWS / "build.yml").read_text(encoding="utf-8")
        release = (WORKFLOWS / "release.yml").read_text(encoding="utf-8")
        cls.sources = {
            "build.yml workflow_dispatch": _defaults(_section(build, "workflow_dispatch")),
            "build.yml workflow_call": _defaults(_section(build, "workflow_call")),
            "release.yml workflow_dispatch": _defaults(_section(release, "workflow_dispatch")),
        }

    def test_every_pin_matches_matrix(self) -> None:
        for build in MATRIX["builds"]:
            suffix = _cuda_suffix(build)
            for field, prefix in FIELDS.items():
                expected = build[field]
                for source, defaults in self.sources.items():
                    with self.subTest(build=build["id"], input=f"{prefix}_{suffix}", source=source):
                        self.assertIn(f"{prefix}_{suffix}", defaults)
                        self.assertEqual(defaults[f"{prefix}_{suffix}"], expected)

    def test_workflows_have_no_pins_for_builds_matrix_dropped(self) -> None:
        suffixes = {_cuda_suffix(build) for build in MATRIX["builds"]}
        for source, defaults in self.sources.items():
            for name in defaults:
                match = re.fullmatch(r"(?:builder|runtime)_image_(cu\d+)", name)
                if match:
                    with self.subTest(source=source, input=name):
                        self.assertIn(match[1], suffixes)

    def test_parser_reads_the_inputs_it_compares(self) -> None:
        # Guard against the regexes quietly matching nothing.
        for source, defaults in self.sources.items():
            with self.subTest(source=source):
                self.assertGreaterEqual(
                    sum(name.startswith(("builder_image_", "runtime_image_")) for name in defaults),
                    2 * len(MATRIX["builds"]),
                )


if __name__ == "__main__":
    unittest.main()
