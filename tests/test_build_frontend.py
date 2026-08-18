from __future__ import annotations

import ast
import json
from pathlib import Path
import re
import unittest


ROOT = Path(__file__).resolve().parents[1]


class BuildFrontendContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.matrix = json.loads(
            (ROOT / "matrix.json").read_text(encoding="utf-8")
        )
        cls.build_script = (ROOT / "scripts" / "build-wheel.sh").read_text(
            encoding="utf-8"
        )
        cls.dockerfile = (ROOT / "docker" / "Dockerfile.builder").read_text(
            encoding="utf-8"
        )
        cls.bake = (ROOT / "docker" / "docker-bake.hcl").read_text(
            encoding="utf-8"
        )
        cls.patch = (
            ROOT / "patches" / "sageattention" / "2.2.0" / "setup.py.patch"
        ).read_text(encoding="utf-8")

    def test_matrix_matches_exact_builder_frontend_pins(self) -> None:
        expected = {
            "build": "1.2.2.post1",
            "packaging": "25.0",
            "setuptools": "80.9.0",
            "wheel": "0.45.1",
        }
        self.assertEqual(self.matrix["build_frontend"], expected)
        build_args = {
            "build": "BUILD_VERSION",
            "packaging": "PACKAGING_VERSION",
            "setuptools": "SETUPTOOLS_VERSION",
            "wheel": "WHEEL_VERSION",
        }
        for distribution, version in expected.items():
            argument = build_args[distribution]
            self.assertRegex(
                self.bake,
                rf'(?m)^\s*{argument}\s*=\s*"{re.escape(version)}"\s*$',
            )
            self.assertIn(
                f'"{distribution}==${{{argument}}}"',
                self.dockerfile,
            )

    def test_downstream_pyproject_requires_reviewed_exact_versions(self) -> None:
        self.assertIn("diff --git a/pyproject.toml b/pyproject.toml", self.patch)
        for distribution in ("packaging", "setuptools", "wheel"):
            version = self.matrix["build_frontend"][distribution]
            self.assertIn(f'+  "{distribution}=={version}"', self.patch)

        added_lines = {
            line[1:]
            for line in self.patch.splitlines()
            if line.startswith("+") and not line.startswith("+++")
        }
        self.assertFalse(
            any(
                requirement in line
                for line in added_lines
                for requirement in ("setuptools<", "wheel<", "packaging<")
            )
        )

    def test_build_checks_matrix_pins_without_bypassing_pep517(self) -> None:
        self.assertIn('build_frontend = matrix["build_frontend"]', self.build_script)
        for distribution in ("build", "packaging", "setuptools", "wheel"):
            self.assertIn(
                f'build_frontend["{distribution}"]',
                self.build_script,
            )
        self.assertIn(
            "actual = importlib.metadata.version(distribution)",
            self.build_script,
        )
        invocation = "python3.12 -m build --wheel --no-isolation"
        self.assertIn(invocation, self.build_script)
        self.assertNotIn("--skip-dependency-check", self.build_script)
        self.assertLess(
            self.build_script.index("expected_frontend = {"),
            self.build_script.index(invocation),
        )

    def test_packaging_version_is_recorded_in_build_evidence(self) -> None:
        self.assertIn(
            '"packaging": importlib.metadata.version("packaging")',
            self.build_script,
        )

    def test_embedded_build_python_is_syntactically_valid(self) -> None:
        lines = self.build_script.splitlines()
        block_count = 0
        index = 0
        while index < len(lines):
            if "<<'PY'" not in lines[index]:
                index += 1
                continue
            body: list[str] = []
            index += 1
            while index < len(lines) and lines[index] != "PY":
                body.append(lines[index])
                index += 1
            self.assertLess(index, len(lines), "unterminated Python heredoc")
            ast.parse("\n".join(body))
            block_count += 1
            index += 1
        self.assertGreaterEqual(block_count, 4)


if __name__ == "__main__":
    unittest.main()
