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

    def test_release_call_uses_supported_expression_contexts_and_types(self) -> None:
        factory = self.release.split("  factory:\n", 1)[1].split(
            "\n  publish:\n", 1
        )[0]
        call_inputs = factory.split("    with:\n", 1)[1].split(
            "\n    secrets:\n", 1
        )[0]

        # GitHub permits only github and needs in jobs.<job_id>.with.<input_id>
        # for reusable-workflow calls. The typed inputs context is not legal here.
        self.assertNotIn("${{ inputs.", call_inputs)
        self.assertIn(
            "source_ref: ${{ needs.preflight.outputs.source_sha }}", call_inputs
        )

        numeric_or_boolean = {
            "container_disk_gb",
            "cpu_vcpu_count",
            "cpu_min_memory_gb",
            "timeout_seconds",
            "gpu_max_parallel",
            "confirm_paid_pods",
        }
        string_inputs = {
            "build_backend",
            "build_gpu_id",
            "builder_image_cu128",
            "builder_image_cu130",
            "runtime_image_cu128",
            "runtime_image_cu130",
            "gpu_id_sm80",
            "gpu_id_sm86",
            "gpu_id_sm89",
            "gpu_id_sm90",
            "gpu_id_sm120",
            "cloud_type",
            "registry_auth_id",
            "cpu_flavor_ids",
        }
        for name in numeric_or_boolean:
            with self.subTest(typed_input=name):
                self.assertIn(
                    f"{name}: ${{{{ fromJSON(github.event.inputs.{name}) }}}}",
                    call_inputs,
                )
        for name in string_inputs:
            with self.subTest(string_input=name):
                self.assertIn(
                    f"{name}: ${{{{ github.event.inputs.{name} }}}}", call_inputs
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

    def test_release_publishes_only_a_missing_or_safe_existing_draft(self) -> None:
        publish = self.release.split(
            "      - name: Create or publish immutable GitHub release\n", 1
        )[1]
        self.assertIn("gh api graphql", publish)
        self.assertIn('if ! response="$(gh api graphql', publish)
        self.assertIn("GitHub release lookup failed", publish)
        self.assertIn("release(tagName: $tag)", publish)
        self.assertIn(".data.repository != null", publish)
        self.assertIn(".isDraft == false", publish)
        self.assertIn("is already published; refusing to overwrite it", publish)
        self.assertIn(".releaseAssets.totalCount == 0", publish)
        self.assertIn("existing release is not an empty, non-prerelease draft", publish)
        self.assertIn('gh release create "${RELEASE_TAG}"', publish)
        self.assertIn("--draft", publish)
        self.assertIn("--verify-tag", publish)
        self.assertIn(
            'gh release upload "${RELEASE_TAG}" "${release_asset_paths[@]}"',
            publish,
        )
        self.assertNotIn("--clobber", publish)
        self.assertIn("uploaded asset set is inconsistent; refusing to publish", publish)
        self.assertIn("hashlib.file_digest(stream, \"sha256\")", publish)
        self.assertIn('"digest": f"sha256:{digest}"', publish)
        self.assertIn("[.releaseAssets.nodes[] | {name, digest, size}]", publish)
        self.assertIn('gh release edit "${RELEASE_TAG}"', publish)
        self.assertIn("--prerelease=false", publish)
        self.assertIn("--draft=false", publish)
        self.assertIn("published release postcondition failed", publish)
        self.assertLess(
            publish.index('gh release upload "${RELEASE_TAG}"'),
            publish.index('gh release edit "${RELEASE_TAG}"'),
        )


if __name__ == "__main__":
    unittest.main()
