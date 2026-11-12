from __future__ import annotations

from datetime import datetime, timedelta, timezone
import unittest

from tools.reap_pods import POD_NAME, RunLookupError, reap, stale_reason
from tools.runpod_job import RestApiError


NOW = datetime(2026, 11, 12, 12, 0, tzinfo=timezone.utc)


def started(minutes_ago: int) -> str:
    return (NOW - timedelta(minutes=minutes_ago)).isoformat().replace("+00:00", "Z")


class FakeCtl:
    def __init__(self, pods, failing=()):
        self.pods = pods
        self.failing = set(failing)
        self.deleted: list[str] = []

    def list_pods(self):
        return self.pods

    def terminate_pod(self, pod_id):
        if pod_id in self.failing:
            raise RestApiError("DELETE", 500, "boom")
        self.deleted.append(pod_id)


class PodNameTests(unittest.TestCase):
    def test_matches_the_names_build_yml_creates(self) -> None:
        for name, run in (
            ("sage-build-cp312-torch2.10.0-cu128-19000000001-1", ("19000000001", "1")),
            ("sage-test-cp312-torch2.13.0-cu132-sm90-19000000001-2", ("19000000001", "2")),
        ):
            with self.subTest(name=name):
                match = POD_NAME.match(name)
                self.assertIsNotNone(match)
                self.assertEqual(match.groups(), run)

    def test_ignores_every_other_pod(self) -> None:
        for name in ("my-dev-pod", "smoketest-1-2-3", "sage-build-x", "sage-test-a-b-c-d"):
            with self.subTest(name=name):
                self.assertIsNone(POD_NAME.match(name))


class StaleReasonTests(unittest.TestCase):
    def test_completed_or_missing_run_is_stale(self) -> None:
        self.assertIn("completed", stale_reason(
            {"status": "completed", "conclusion": "cancelled"}, now=NOW, max_age_minutes=360,
        ))
        self.assertIn("no longer exists", stale_reason(None, now=NOW, max_age_minutes=360))

    def test_running_attempt_is_kept_until_the_cap(self) -> None:
        running = {"status": "in_progress", "run_started_at": started(30)}
        self.assertIsNone(stale_reason(running, now=NOW, max_age_minutes=360))
        old = {"status": "in_progress", "run_started_at": started(400)}
        self.assertIn("cap 360", stale_reason(old, now=NOW, max_age_minutes=360))

    def test_running_attempt_without_a_start_time_is_kept(self) -> None:
        self.assertIsNone(stale_reason({"status": "queued"}, now=NOW, max_age_minutes=1))


class ReapTests(unittest.TestCase):
    PODS = [
        {"id": "done", "name": "sage-test-cp312-torch2.10.0-cu128-sm80-11-1"},
        {"id": "live", "name": "sage-build-cp312-torch2.10.0-cu128-22-1"},
        {"id": "gone", "name": "sage-test-cp312-torch2.13.0-cu132-sm120-33-1"},
        {"id": "human", "name": "bohdan-debug"},
    ]
    RUNS = {
        (11, 1): {"status": "completed", "conclusion": "failure"},
        (22, 1): {"status": "in_progress", "run_started_at": started(20)},
        (33, 1): None,
    }

    def _reap(self, ctl, *, delete=True, lookup=None):
        lines: list[str] = []
        failed = reap(
            ctl, lookup or (lambda run, attempt: self.RUNS[(run, attempt)]),
            delete=delete, max_age_minutes=360, now=NOW, log=lines.append,
        )
        return failed, lines

    def test_deletes_only_factory_pods_of_finished_runs(self) -> None:
        ctl = FakeCtl(self.PODS)
        failed, _ = self._reap(ctl)
        self.assertEqual(failed, 0)
        self.assertEqual(ctl.deleted, ["done", "gone"])

    def test_dry_run_deletes_nothing(self) -> None:
        ctl = FakeCtl(self.PODS)
        failed, lines = self._reap(ctl, delete=False)
        self.assertEqual((failed, ctl.deleted), (0, []))
        self.assertEqual(sum("[dry-run] would delete" in line for line in lines), 2)

    def test_failed_delete_is_counted(self) -> None:
        ctl = FakeCtl(self.PODS, failing={"done"})
        failed, _ = self._reap(ctl)
        self.assertEqual((failed, ctl.deleted), (1, ["gone"]))

    def test_unknown_run_state_keeps_the_pod(self) -> None:
        def lookup(run, attempt):
            raise RunLookupError("HTTP 502")

        ctl = FakeCtl(self.PODS)
        failed, _ = self._reap(ctl, lookup=lookup)
        self.assertEqual((failed, ctl.deleted), (0, []))

    def test_each_run_attempt_is_looked_up_once(self) -> None:
        calls = []
        pods = [
            {"id": f"p{sm}", "name": f"sage-test-cp312-torch2.10.0-cu128-sm{sm}-11-1"}
            for sm in (80, 86, 89)
        ]

        def lookup(run, attempt):
            calls.append((run, attempt))
            return {"status": "completed"}

        failed, _ = self._reap(FakeCtl(pods), lookup=lookup)
        self.assertEqual((failed, calls), (0, [(11, 1)]))


if __name__ == "__main__":
    unittest.main()
