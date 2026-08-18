from __future__ import annotations

import ast
from copy import deepcopy
import json
import math
from pathlib import Path
import textwrap
import unittest


ROOT = Path(__file__).resolve().parents[1]
MATRIX = json.loads((ROOT / "matrix.json").read_text(encoding="utf-8"))
RELEASE_WORKFLOW = (ROOT / ".github" / "workflows" / "release.yml").read_text(
    encoding="utf-8"
)


def embedded_python_blocks(workflow: str) -> list[str]:
    blocks: list[str] = []
    lines = workflow.splitlines()
    index = 0
    while index < len(lines):
        if "python3.12 - <<'PY'" not in lines[index]:
            index += 1
            continue
        body: list[str] = []
        index += 1
        while index < len(lines) and lines[index].strip() != "PY":
            body.append(lines[index])
            index += 1
        if index == len(lines):
            raise AssertionError("unterminated Python heredoc in release workflow")
        blocks.append(textwrap.dedent("\n".join(body)))
        index += 1
    return blocks


RUNTIME_GATE_BLOCK = next(
    block
    for block in embedded_python_blocks(RELEASE_WORKFLOW)
    if "def validate_runtime_report(" in block
)
RUNTIME_GATE_TREE = ast.parse(RUNTIME_GATE_BLOCK)
RUNTIME_GATE_FUNCTION = next(
    node
    for node in RUNTIME_GATE_TREE.body
    if isinstance(node, ast.FunctionDef) and node.name == "validate_runtime_report"
)
RUNTIME_GATE_NAMESPACE = {"json": json, "math": math}
exec(
    compile(
        ast.Module(body=[RUNTIME_GATE_FUNCTION], type_ignores=[]),
        filename="release.yml:validate_runtime_report",
        mode="exec",
    ),
    RUNTIME_GATE_NAMESPACE,
)
VALIDATE_RUNTIME_REPORT = RUNTIME_GATE_NAMESPACE["validate_runtime_report"]


class ReleaseRuntimeGateTests(unittest.TestCase):
    capability = "9.0"
    build = MATRIX["builds"][0]
    numeric_policy = MATRIX["validation"]["runtime_numeric"]
    required_modules = set(MATRIX["validation"]["required_extensions"])
    wheel = {"asset": "sageattention.whl", "sha256": "a" * 64}
    runtime_image_ref = build["comfyui_runtime_image"]
    report_path = Path("runtime-sm90.json")

    def valid_report(self) -> dict:
        canonical_case = self.numeric_policy["canonical_case"]
        shape = [
            canonical_case["batch_size"],
            canonical_case["query_heads"],
            canonical_case["sequence_length"],
            canonical_case["head_dimension"],
        ]
        dtype = canonical_case["dtype"]
        results = [
            {
                "causal": causal,
                "cosine_similarity": 1.0,
                "expected_output_dtype": dtype,
                "expected_output_shape": shape,
                "implementation": implementation,
                "output_dtype": dtype,
                "output_shape": shape,
                "relative_l2": 0.0,
            }
            for implementation in self.numeric_policy["implementations_by_capability"][
                self.capability
            ]
            for causal in self.numeric_policy["canonical_case"]["causal_modes"]
        ]
        return {
            "actual_compute_capability": self.capability,
            "build_id": self.build["id"],
            "compiled_modules": {
                module: f"/opt/sageattention/{module.rsplit('.', 1)[-1]}.so"
                for module in self.required_modules
            },
            "cuda_device_name": "NVIDIA H100 80GB HBM3",
            "expected_compute_capability": self.capability,
            "expected_runtime_image": self.build["comfyui_runtime_image"],
            "results": results,
            "runtime_image_ref": self.runtime_image_ref,
            "runtime_numeric_policy": deepcopy(self.numeric_policy),
            "sageattention_version": self.build["wheel_version"],
            "schema_version": 1,
            "status": "pass",
            "torch_cuda_version": self.build["torch_cuda_version"],
            "torch_version": self.build["torch_version"],
            "wheel_asset": self.wheel["asset"],
            "wheel_sha256": self.wheel["sha256"],
        }

    def validate(self, report: dict) -> None:
        VALIDATE_RUNTIME_REPORT(
            report,
            self.report_path,
            self.build,
            self.capability,
            self.wheel,
            self.numeric_policy,
            self.required_modules,
            self.runtime_image_ref,
        )

    def assert_rejected(self, report: dict, message: str) -> None:
        with self.assertRaisesRegex(SystemExit, message):
            self.validate(report)

    def test_valid_report_passes_and_embedded_block_is_syntactically_valid(self) -> None:
        self.validate(self.valid_report())
        ast.parse(RUNTIME_GATE_BLOCK)

    def test_schema_modules_device_and_policy_are_exactly_bound(self) -> None:
        cases = []
        for value in (None, True, 2):
            report = self.valid_report()
            report["schema_version"] = value
            cases.append((f"schema-{value!r}", report, "schema mismatch"))

        report = self.valid_report()
        report["compiled_modules"].pop(next(iter(self.required_modules)))
        cases.append(("missing-module", report, "compiled module mismatch"))
        report = self.valid_report()
        report["compiled_modules"]["sageattention._unexpected"] = "/tmp/unexpected.so"
        cases.append(("extra-module", report, "compiled module mismatch"))
        report = self.valid_report()
        report["compiled_modules"][next(iter(self.required_modules))] = "   "
        cases.append(("blank-module-path", report, "compiled module mismatch"))
        report = self.valid_report()
        report["compiled_modules"] = []
        cases.append(("modules-not-object", report, "compiled module mismatch"))

        for value in (None, "", "   "):
            report = self.valid_report()
            report["cuda_device_name"] = value
            cases.append((f"device-{value!r}", report, "no CUDA device name"))

        report = self.valid_report()
        report["runtime_numeric_policy"]["minimum_cosine_similarity"] = 0.9
        cases.append(("policy-value", report, "numeric policy mismatch"))
        report = self.valid_report()
        report["runtime_numeric_policy"]["canonical_case"]["causal_modes"] = [0, 1]
        cases.append(("policy-types", report, "numeric policy mismatch"))

        for name, report, message in cases:
            with self.subTest(name=name):
                self.assert_rejected(report, message)

    def test_pass_reports_cannot_carry_failure_diagnostics(self) -> None:
        for value in ([], [{"message": "kernel failed"}]):
            report = self.valid_report()
            report["failures"] = value
            with self.subTest(top_level=value):
                self.assert_rejected(report, "contains failures")

        for value in ([], [{"message": "bad metric"}]):
            report = self.valid_report()
            report["results"][0]["errors"] = value
            with self.subTest(result=value):
                self.assert_rejected(report, "contains errors")

    def test_metrics_are_finite_non_boolean_numbers_and_thresholded(self) -> None:
        invalid_metrics = (
            ("cosine_similarity", True, "invalid cosine metric"),
            ("cosine_similarity", math.nan, "invalid cosine metric"),
            ("cosine_similarity", math.inf, "invalid cosine metric"),
            ("cosine_similarity", "1.0", "invalid cosine metric"),
            ("relative_l2", False, "invalid relative-L2 metric"),
            ("relative_l2", math.nan, "invalid relative-L2 metric"),
            ("relative_l2", math.inf, "invalid relative-L2 metric"),
            ("relative_l2", "0.0", "invalid relative-L2 metric"),
        )
        for field, value, message in invalid_metrics:
            report = self.valid_report()
            report["results"][0][field] = value
            with self.subTest(field=field, value=value):
                self.assert_rejected(report, message)

        report = self.valid_report()
        report["results"][0]["cosine_similarity"] = (
            self.numeric_policy["minimum_cosine_similarity"] - 0.0001
        )
        self.assert_rejected(report, "cosine threshold failure")
        report = self.valid_report()
        report["results"][0]["relative_l2"] = (
            self.numeric_policy["maximum_relative_l2"] + 0.0001
        )
        self.assert_rejected(report, "relative-L2 threshold failure")

    def test_coverage_shape_and_dtype_gates_remain_fail_closed(self) -> None:
        report = self.valid_report()
        report["results"].pop()
        self.assert_rejected(report, "implementation coverage mismatch")

        report = self.valid_report()
        report["results"][0]["causal"] = 1
        self.assert_rejected(report, "implementation coverage mismatch")

        for field, value in (
            ("output_shape", [1, 8, 256, 64]),
            ("expected_output_shape", [1, 8, 256, 64]),
            ("output_dtype", "float32"),
            ("expected_output_dtype", "float32"),
        ):
            report = self.valid_report()
            report["results"][0][field] = value
            with self.subTest(field=field):
                self.assert_rejected(report, "runtime output tensor mismatch")


if __name__ == "__main__":
    unittest.main()
