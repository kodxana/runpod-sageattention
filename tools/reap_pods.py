#!/usr/bin/env python3
"""Delete factory Pods whose GitHub Actions run is over.

``runpod_job.py`` deletes its Pod from a ``finally`` path, but that path dies
with the runner: a SIGKILLed, cancelled or lost runner leaves the Pod billing.
Builder Pods also arm an in-Pod watchdog; validation Pods run the ComfyUI
runtime image and have nothing.  REST v2 has no platform-side termination
deadline either, so this sweep (``.github/workflows/reap-pods.yml``, hourly)
is the backstop for every factory Pod.

Only Pods named by ``build.yml`` are considered::

    sage-build-<build id>-<run id>-<run attempt>
    sage-test-<build id>-<sm>-<run id>-<run attempt>

A Pod is deleted when the run attempt that created it has completed (its
``finally`` would have run by then), when that run no longer exists, or when
the attempt started more than ``--max-age-minutes`` ago.  Anything else is
kept, and every other Pod on the account is never touched.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import sys
from typing import Callable, Mapping, Sequence
from urllib import error as urllib_error
from urllib import request as urllib_request

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.runpod_job import USER_AGENT, CommandError, RunpodApi  # noqa: E402


# build.yml's names; run id and attempt are always the last two fields.
POD_NAME = re.compile(r"^sage-(?:build|test)-[A-Za-z0-9._-]+-(\d+)-(\d+)$")
# Longest a factory Pod may legitimately live: the build job's 300-minute
# GitHub timeout plus the placement waits it already includes, with margin.
DEFAULT_MAX_AGE_MINUTES = 360


class RunLookupError(RuntimeError):
    """GitHub could not say whether a run is still going."""


# (run_id, attempt) -> {"status": ..., "run_started_at": ...}, or None when
# GitHub has no such run.
RunLookup = Callable[[int, int], "Mapping[str, object] | None"]


def github_run_lookup(
    repository: str, token: str, *, api_url: str = "https://api.github.com"
) -> RunLookup:
    def lookup(run_id: int, attempt: int) -> Mapping[str, object] | None:
        url = f"{api_url}/repos/{repository}/actions/runs/{run_id}/attempts/{attempt}"
        request = urllib_request.Request(
            url,
            headers={
                "Accept": "application/vnd.github+json",
                "Authorization": f"Bearer {token}",
                "User-Agent": USER_AGENT,
                "X-GitHub-Api-Version": "2022-11-28",
            },
        )
        try:
            with urllib_request.urlopen(request, timeout=30) as response:
                payload = json.loads(response.read())
        except urllib_error.HTTPError as exc:
            if exc.code == 404:
                return None
            raise RunLookupError(f"GitHub run {run_id}/{attempt}: HTTP {exc.code}") from exc
        except (urllib_error.URLError, TimeoutError, OSError, ValueError) as exc:
            raise RunLookupError(f"GitHub run {run_id}/{attempt}: {exc}") from exc
        if not isinstance(payload, dict):
            raise RunLookupError(f"GitHub run {run_id}/{attempt}: not an object")
        return payload

    return lookup


def _started(run: Mapping[str, object]) -> datetime | None:
    value = run.get("run_started_at") or run.get("created_at")
    if not isinstance(value, str):
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None


def stale_reason(
    run: Mapping[str, object] | None, *, now: datetime, max_age_minutes: int
) -> str | None:
    """Why a factory Pod of this run attempt should go, or None to keep it."""

    if run is None:
        return "its workflow run no longer exists"
    status = run.get("status")
    if status == "completed":
        return f"its run attempt completed ({run.get('conclusion') or 'no conclusion'})"
    started = _started(run)
    if started is None:
        return None
    age_minutes = (now - started).total_seconds() / 60
    if age_minutes > max_age_minutes:
        return (
            f"its run attempt is still {status or 'running'} after "
            f"{age_minutes:.0f} min (cap {max_age_minutes})"
        )
    return None


def reap(
    ctl: RunpodApi,
    lookup: RunLookup,
    *,
    delete: bool,
    max_age_minutes: int,
    now: datetime | None = None,
    log: Callable[[str], None] = print,
) -> int:
    """Sweep once; returns the number of Pods that failed to delete."""

    now = now or datetime.now(timezone.utc)
    scanned = reaped = kept = failed = 0
    runs: dict[tuple[int, int], Mapping[str, object] | None] = {}
    for pod in ctl.list_pods():
        name = str(pod.get("name") or "")
        pod_id = str(pod.get("id") or "")
        match = POD_NAME.match(name)
        if not match:
            continue
        scanned += 1
        key = (int(match.group(1)), int(match.group(2)))
        if key not in runs:
            try:
                runs[key] = lookup(*key)
            except RunLookupError as exc:
                log(f"keep {name} ({pod_id}): {exc}")
                kept += 1
                continue
        reason = stale_reason(runs[key], now=now, max_age_minutes=max_age_minutes)
        if reason is None:
            log(f"keep {name} ({pod_id}): its run attempt is in progress")
            kept += 1
            continue
        if not pod_id:
            log(f"::warning::{name} is stale ({reason}) but has no id; skipping")
            kept += 1
            continue
        if not delete:
            log(f"[dry-run] would delete {name} ({pod_id}): {reason}")
            reaped += 1
            continue
        try:
            ctl.terminate_pod(pod_id)
        except CommandError as exc:
            log(f"::warning::failed to delete {name} ({pod_id}): {exc}")
            failed += 1
            continue
        log(f"deleted {name} ({pod_id}): {reason}")
        reaped += 1
    log(
        f"factory Pods scanned: {scanned} | "
        f"{'reaped' if delete else 'would reap'}: {reaped} | kept: {kept} | "
        f"failed: {failed}"
    )
    return failed


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--delete", action="store_true", help="actually delete")
    parser.add_argument(
        "--max-age-minutes", type=int, default=DEFAULT_MAX_AGE_MINUTES,
        help="delete a Pod whose run attempt is still going after this long",
    )
    args = parser.parse_args(argv)
    if args.max_age_minutes < 1:
        parser.error("--max-age-minutes must be at least 1")
    repository = os.environ.get("GITHUB_REPOSITORY", "").strip()
    token = os.environ.get("GITHUB_TOKEN", "").strip()
    if not repository or not token:
        parser.error("GITHUB_REPOSITORY and GITHUB_TOKEN are required")
    ctl = RunpodApi()
    try:
        failed = reap(
            ctl,
            github_run_lookup(repository, token),
            delete=args.delete,
            max_age_minutes=args.max_age_minutes,
        )
    except CommandError as exc:
        # A failed listing must not look like a clean account.
        print(f"::error::{exc}", file=sys.stderr)
        return 1
    if failed:
        print(f"::error::{failed} Pod deletion(s) failed", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
