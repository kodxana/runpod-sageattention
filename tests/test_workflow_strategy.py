from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import textwrap
import unittest


ROOT = Path(__file__).resolve().parents[1]
WORKFLOW_PATH = ROOT / ".github" / "workflows" / "build.yml"


def plan_script(workflow: str) -> str:
    marker = "- name: Generate build and representative-GPU plans"
    section = workflow[workflow.index(marker):]
    lines = section.splitlines()
    start = next(
        index for index, line in enumerate(lines) if "python3.12 - <<'PY'" in line
    )
    body: list[str] = []
    for line in lines[start + 1:]:
        if line.strip() == "PY":
            return textwrap.dedent("\n".join(body))
        body.append(line)
    raise AssertionError("unterminated workflow plan heredoc")


class WorkflowStrategyTests(unittest.TestCase):
    def test_plan_emits_json_typed_matrix_and_parallelism_outputs(self) -> None:
        workflow = WORKFLOW_PATH.read_text(encoding="utf-8")
        script = plan_script(workflow)
        environment = {
            **os.environ,
            "BUILDER_CU128": "builder/cu128@sha256:" + "1" * 64,
            "BUILDER_CU130": "builder/cu130@sha256:" + "2" * 64,
            "BUILD_BACKEND": "GPU",
            "BUILD_GPU_ID": "NVIDIA A100 80GB PCIe",
            "RUNTIME_CU128": "runtime/cu128@sha256:" + "3" * 64,
            "RUNTIME_CU130": "runtime/cu130@sha256:" + "4" * 64,
            "GPU_SM80": "NVIDIA A100 80GB PCIe",
            "GPU_SM86": "NVIDIA A40",
            "GPU_SM89": "NVIDIA L40S",
            "GPU_SM90": "NVIDIA H100 PCIe",
            "GPU_SM120": "NVIDIA GeForce RTX 5090",
            "ENFORCE_DIGESTS": "true",
            "CONFIRM_PAID_PODS": "true",
            "CONTAINER_DISK_GB": "80",
            "CPU_FLAVOR_IDS": "cpu3g",
            "CPU_VCPU_COUNT": "16",
            "CPU_MIN_MEMORY_GB": "32",
            "TIMEOUT_SECONDS": "14400",
            "GPU_MAX_PARALLEL": "2",
        }
        with tempfile.TemporaryDirectory() as directory:
            output_path = Path(directory) / "github-output"
            environment["GITHUB_OUTPUT"] = str(output_path)
            subprocess.run(
                [sys.executable, "-c", script],
                cwd=ROOT,
                env=environment,
                check=True,
                capture_output=True,
                text=True,
            )
            outputs = dict(
                line.split("=", 1)
                for line in output_path.read_text(encoding="utf-8").splitlines()
            )

        self.assertEqual(set(outputs), {"builds", "gpu_max_parallel", "gpu-tests"})
        builds = json.loads(outputs["builds"])
        gpu_tests = json.loads(outputs["gpu-tests"])
        max_parallel = json.loads(outputs["gpu_max_parallel"])
        self.assertEqual(set(builds), {"include"})
        self.assertEqual(set(gpu_tests), {"include"})
        self.assertEqual(len(builds["include"]), 2)
        self.assertEqual(len(gpu_tests["include"]), 10)
        self.assertIs(type(max_parallel), int)
        self.assertEqual(max_parallel, 2)

    def test_gpu_strategy_decodes_plan_output_as_number(self) -> None:
        workflow = WORKFLOW_PATH.read_text(encoding="utf-8")
        self.assertIn(
            "gpu_max_parallel: ${{ steps.plan.outputs.gpu_max_parallel }}",
            workflow,
        )
        self.assertIn(
            "max-parallel: ${{ fromJSON(needs.plan.outputs.gpu_max_parallel) }}",
            workflow,
        )
        self.assertNotIn("max-parallel: ${{ inputs.gpu_max_parallel }}", workflow)
        self.assertIn(
            "matrix: ${{ fromJSON(needs.plan.outputs.gpu-tests) }}",
            workflow,
        )


if __name__ == "__main__":
    unittest.main()
