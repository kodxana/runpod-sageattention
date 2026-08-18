from __future__ import annotations

from pathlib import Path
import re
import unittest


ROOT = Path(__file__).resolve().parents[1]
BUILD_WORKFLOW = ROOT / ".github" / "workflows" / "build.yml"
RELEASE_WORKFLOW = ROOT / ".github" / "workflows" / "release.yml"


class WorkflowSecretContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.build = BUILD_WORKFLOW.read_text(encoding="utf-8")
        cls.release = RELEASE_WORKFLOW.read_text(encoding="utf-8")

    def test_environment_secrets_are_optional_at_reusable_call_boundary(self) -> None:
        workflow_call = self.build.split("  workflow_call:\n", 1)[1].split(
            "\npermissions:\n", 1
        )[0]
        for name in ("RUNPOD_API_KEY", "RUNPOD_SSH_PRIVATE_KEY"):
            with self.subTest(secret=name):
                match = re.search(
                    rf"(?ms)^      {re.escape(name)}:\n(?P<body>.*?)(?=^      \S|\Z)",
                    workflow_call,
                )
                self.assertIsNotNone(match)
                declaration = match.group("body")
                self.assertIn("required: false", declaration)
                self.assertNotIn("required: true", declaration)

    def test_release_forwards_only_optional_runpod_secret_fallbacks(self) -> None:
        factory = self.release.split("  factory:\n", 1)[1].split(
            "\n  publish:\n", 1
        )[0]
        self.assertNotIn("secrets: inherit", factory)
        self.assertIn(
            "RUNPOD_API_KEY: ${{ secrets.RUNPOD_API_KEY }}", factory
        )
        self.assertIn(
            "RUNPOD_SSH_PRIVATE_KEY: ${{ secrets.RUNPOD_SSH_PRIVATE_KEY }}",
            factory,
        )

    def test_paid_jobs_fail_before_work_when_credentials_are_unavailable(self) -> None:
        self.assertEqual(
            self.build.count("- name: Verify protected Runpod credentials"), 2
        )
        self.assertEqual(
            self.build.count(
                "RUNPOD_API_KEY is missing from repository secrets and the "
                "runpod-paid environment"
            ),
            2,
        )
        self.assertEqual(
            self.build.count(
                "RUNPOD_SSH_PRIVATE_KEY is missing from repository secrets and the "
                "runpod-paid environment"
            ),
            2,
        )

        build_job = self.build.split("  build:\n", 1)[1].split(
            "\n  gpu-test:\n", 1
        )[0]
        gpu_test_job = self.build.split("  gpu-test:\n", 1)[1]
        for job in (build_job, gpu_test_job):
            with self.subTest(job=job.splitlines()[0] if job else "unknown"):
                self.assertLess(
                    job.index("- name: Verify protected Runpod credentials"),
                    job.index("- name: Install checksum-pinned runpodctl"),
                )


if __name__ == "__main__":
    unittest.main()
