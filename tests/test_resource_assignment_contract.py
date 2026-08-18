from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class ResourceAssignmentContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.script = (ROOT / "scripts" / "build-wheel.sh").read_text(
            encoding="utf-8"
        )
        cls.helper = (ROOT / "tools" / "pod_resources.py").read_text(
            encoding="utf-8"
        )

    def test_build_prefers_uploaded_resource_helper_for_old_images(self) -> None:
        repo_helper = 'if [[ -f "${REPO_ROOT}/tools/pod_resources.py" ]]'
        installed_helper = "elif command -v pod-resources"
        self.assertIn(repo_helper, self.script)
        self.assertIn(installed_helper, self.script)
        self.assertLess(
            self.script.index(repo_helper), self.script.index(installed_helper)
        )

    def test_assignment_capacity_and_peak_contract_is_fail_closed(self) -> None:
        self.assertIn('capacity_source == "runpod-api-assignment"', self.script)
        self.assertIn('snapshot.get("schema_version") != 2', self.script)
        self.assertIn(
            "a positive receipt-backed Runpod API memory assignment is required",
            self.script,
        )
        self.assertIn('memory.get("usage_peak_eligible") is not True', self.script)
        self.assertIn('memory.get("usage_source")', self.script)
        self.assertIn(
            "untrusted Pod memory usage forces MAX_JOBS=1", self.script
        )
        self.assertIn(
            "untrusted Pod memory usage forces EXT_PARALLEL=1", self.script
        )
        self.assertIn(
            'if [[ "${FORCED_SINGLE_JOB}" == "1" && "${EXT_PARALLEL}" != "1" ]]',
            self.script,
        )
        self.assertIn(
            "A positive cgroup-membership memory peak is required", self.script
        )
        self.assertIn("CGROUP_PEAK_END < CGROUP_PEAK_START", self.script)
        self.assertIn("CGROUP_PEAK_END > MEMORY_CAPACITY_BYTES", self.script)
        self.assertIn('"memory_policy": {', self.script)
        self.assertIn('peak_evidence_mode == "process-group-rss"', self.script)
        self.assertIn('--monitor-process-group "${evidence_file}"', self.script)
        self.assertIn('"mode": "process-group-rss"', self.script)
        self.assertIn('"covers_other_pod_processes": False', self.script)
        self.assertIn('"includes_file_cache": False', self.script)
        self.assertIn(
            '"sampled_peak_plus_reserve_within_assignment"', self.script
        )
        self.assertIn('"whole_pod_enforced": sample["whole_pod_enforced"]', self.script)
        self.assertIn("start_new_session=True", self.helper)
        self.assertIn("signal.SIGHUP", self.helper)
        self.assertIn("_kill_process_group_bounded", self.helper)

    def test_live_receipt_requires_root_ownership_but_fixtures_are_mappable(
        self,
    ) -> None:
        self.assertIn('live_filesystem = probe.filesystem_root == Path("/")', self.helper)
        self.assertIn("if live_filesystem and (", self.helper)
        self.assertIn("must be owned by root:root", self.helper)
        self.assertIn("stat.S_IMODE(parent_stat.st_mode) != 0o755", self.helper)
        self.assertIn("stat.S_IMODE(receipt_stat.st_mode) != 0o444", self.helper)

    def test_low_resource_escape_hatch_cannot_mask_cpu_or_disk(self) -> None:
        cpu_failure = 'hard_failures.append(f"effective CPU'
        escape_hatch = 'if os.environ.get("ALLOW_LOW_RESOURCES") == "1"'
        self.assertIn(cpu_failure, self.script)
        self.assertIn("filesystem free disk", self.script)
        self.assertLess(
            self.script.index(cpu_failure), self.script.rindex(escape_hatch)
        )
        self.assertLess(
            self.script.index("filesystem free disk"),
            self.script.rindex(escape_hatch),
        )
        # The override is reached only after hard failures have already exited.
        self.assertLess(
            self.script.index("if hard_failures:"),
            self.script.rindex(escape_hatch),
        )

    def test_near_capacity_ambiguous_usage_fails_headroom_preflight(self) -> None:
        marker = 'RESOURCE_SNAPSHOT_JSON="${RESOURCE_SNAPSHOT_JSON}" python3.12 - <<\'PY\'\n'
        preflight = self.script.split(marker, 1)[1].split("\nPY\n", 1)[0]
        gib = 1024**3
        snapshot = {
            "schema_version": 2,
            "cgroup": {"version": 2},
            "memory": {
                "limited": False,
                "limit_bytes": 64 * gib,
                "capacity_bytes": 32 * gib,
                "capacity_source": "runpod-api-assignment",
                "capacity_is_hard_limit": False,
                "assigned_capacity_bytes": 32 * gib,
                "available_bytes": 1 * gib,
                "usage_source": "cgroup-v2:/sys/fs/cgroup",
                "usage_trustworthy": False,
                "usage_peak_eligible": True,
                "usage_scope": "ambiguous-cgroup-root",
                "peak_evidence_mode": "cgroup",
            },
            "cpu": {"effective_count": 4},
            "build": {
                "reserve_bytes": 5 * gib,
                "memory_per_job_bytes": 8 * gib,
                "forced_single_job": True,
                "suggested_jobs": 1,
            },
        }
        with tempfile.TemporaryDirectory() as directory:
            env = dict(os.environ)
            env.update(
                {
                    "ALLOW_LOW_RESOURCES": "0",
                    "MIN_CPUS": "4",
                    "MIN_DISK_GIB": "0",
                    "MIN_MEMORY_GIB": "32",
                    "OUTPUT_DIR": directory,
                    "RECOMMENDED_MEMORY_GIB": "64",
                    "RESOURCE_SNAPSHOT_JSON": json.dumps(snapshot),
                    "WORK_PARENT": directory,
                }
            )
            completed = subprocess.run(
                [sys.executable, "-c", preflight],
                capture_output=True,
                check=False,
                env=env,
                text=True,
            )
        self.assertNotEqual(completed.returncode, 0)
        self.assertIn("reserve plus one compiler", completed.stderr)

    def test_no_counter_rss_fallback_requires_recommended_capacity(self) -> None:
        marker = 'RESOURCE_SNAPSHOT_JSON="${RESOURCE_SNAPSHOT_JSON}" python3.12 - <<\'PY\'\n'
        preflight = self.script.split(marker, 1)[1].split("\nPY\n", 1)[0]
        gib = 1024**3

        def run(
            capacity_gib: int, **overrides: str
        ) -> subprocess.CompletedProcess[str]:
            snapshot = {
                "schema_version": 2,
                "cgroup": {"version": None},
                "memory": {
                    "limited": False,
                    "limit_bytes": 128 * gib,
                    "capacity_bytes": capacity_gib * gib,
                    "capacity_source": "runpod-api-assignment",
                    "capacity_is_hard_limit": False,
                    "assigned_capacity_bytes": capacity_gib * gib,
                    "available_bytes": 0,
                    "usage_current_bytes": None,
                    "usage_source": "",
                    "usage_trustworthy": False,
                    "usage_peak_eligible": False,
                    "usage_scope": "unavailable",
                    "peak_evidence_mode": "process-group-rss",
                },
                "cpu": {"effective_count": 4},
                "build": {
                    "reserve_bytes": 10 * gib,
                    "memory_per_job_bytes": 8 * gib,
                    "forced_single_job": True,
                    "suggested_jobs": 1,
                },
            }
            with tempfile.TemporaryDirectory() as directory:
                env = dict(os.environ)
                env.update(
                    {
                        "ALLOW_LOW_RESOURCES": "0",
                        "ALLOW_UNSAFE_PARALLELISM": "0",
                        "MIN_CPUS": "4",
                        "MIN_DISK_GIB": "0",
                        "MIN_MEMORY_GIB": "32",
                        "OUTPUT_DIR": directory,
                        "RECOMMENDED_MEMORY_GIB": "64",
                        "RESOURCE_SNAPSHOT_JSON": json.dumps(snapshot),
                        "WORK_PARENT": directory,
                    }
                )
                env.update(overrides)
                return subprocess.run(
                    [sys.executable, "-c", preflight],
                    capture_output=True,
                    check=False,
                    env=env,
                    text=True,
                )

        accepted = run(64)
        self.assertEqual(accepted.returncode, 0, accepted.stderr)
        self.assertIn("peak_evidence_mode=process-group-rss", accepted.stdout)
        self.assertIn("memory_headroom=unmeasured", accepted.stdout)

        below_recommended = run(63)
        self.assertNotEqual(below_recommended.returncode, 0)
        self.assertIn(
            "requires the recommended memory capacity", below_recommended.stderr
        )

        for override in ("ALLOW_LOW_RESOURCES", "ALLOW_UNSAFE_PARALLELISM"):
            with self.subTest(override=override):
                rejected = run(64, **{override: "1"})
                self.assertNotEqual(rejected.returncode, 0)
                self.assertIn(
                    "forbidden with process-group RSS", rejected.stderr
                )

    def test_no_counter_fallback_rejects_conflicting_readable_usage(self) -> None:
        self.assertIn(
            'and memory.get("usage_current_bytes") is None', self.script
        )
        self.assertIn('and usage_source == ""', self.script)
        self.assertIn('and usage_scope == "unavailable"', self.script)
        self.assertIn(
            'if rss_fallback_eligible and current is None and not usage_source',
            self.helper,
        )

    @unittest.skipUnless(
        os.name == "posix" and Path("/bin/bash").is_file(), "Linux Bash"
    )
    def test_shell_boundary_relays_term_and_waits_for_supervisor_cleanup(self) -> None:
        begin = "# BEGIN RSS SUPERVISOR SHELL\n"
        end = "# END RSS SUPERVISOR SHELL"
        function_source = self.script.split(begin, 1)[1].split(end, 1)[0]
        shell_program = (
            function_source
            + "\nRESOURCE_HELPER=(\"$1\" \"$2\")\n"
            + "run_rss_supervised_command \"$3\" \"$1\" -c "
            + "'import time; time.sleep(30)'\n"
        )
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "rss.json"
            shell = subprocess.Popen(
                [
                    "/bin/bash",
                    "-c",
                    shell_program,
                    "rss-shell-test",
                    sys.executable,
                    str(ROOT / "tools" / "pod_resources.py"),
                    str(output),
                ]
            )
            try:
                for _attempt in range(100):
                    if output.is_file():
                        break
                    if shell.poll() is not None:
                        self.fail("shell exited before supervisor evidence appeared")
                    time.sleep(0.02)
                else:
                    self.fail("supervisor did not write initial evidence")
                shell.send_signal(signal.SIGTERM)
                self.assertEqual(shell.wait(timeout=5), 143)
            finally:
                if shell.poll() is None:
                    shell.kill()
                    shell.wait(timeout=5)

            payload = json.loads(output.read_text(encoding="utf-8"))
            self.assertTrue(payload["complete"])
            self.assertEqual(payload["forwarded_signal"], "SIGTERM")
            self.assertFalse(payload["forced_kill"])
            with self.assertRaises(ProcessLookupError):
                os.killpg(payload["process_group_id"], 0)


if __name__ == "__main__":
    unittest.main()
