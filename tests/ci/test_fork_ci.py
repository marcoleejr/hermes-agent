"""A fork's pull request never queues on a runner only the organisation has.

A job whose label no runner serves waits 24 h and is then cancelled, so the pull request stays red
with nothing tested. Replays the job gates of every unfiltered ``pull_request`` workflow for a fork
and for the upstream repository.
"""
from pathlib import Path

import pytest
from ruamel.yaml import YAML

from tests.ci import _gha_expr as gha

WORKFLOWS = Path(__file__).resolve().parents[2] / ".github/workflows"
UPSTREAM = "NousResearch/hermes-agent"
FORK = "someone/hermes-agent"


def _pull_request_workflows():
    for path in sorted(WORKFLOWS.glob("*.y*ml")):
        doc = YAML(typ="safe").load(path.read_text(encoding="utf-8"))
        triggers = doc.get("on") or doc.get(True) or {}
        if isinstance(triggers, dict) and "pull_request" in triggers and not triggers["pull_request"]:
            yield path.name, doc["jobs"]


def _runners(repository: str) -> dict[str, str]:
    """``workflow:job`` -> label for each job that starts on a PR before any other job finished."""
    context = {"github": {"repository": repository, "event_name": "pull_request"}, "inputs": {},
               "__status__": {"always": True, "success": True, "failure": False, "cancelled": False}}
    return {f"{name}:{job}": body["runs-on"]
            for name, jobs in _pull_request_workflows() for job, body in jobs.items()
            if "runs-on" in body and (not body.get("needs") or "always()" in str(body.get("if")))
            and gha.condition(body.get("if", "success()"), context)}


def test_a_fork_pull_request_only_asks_for_standard_runners():
    runners = _runners(FORK)
    assert {job: label for job, label in runners.items() if label != "ubuntu-latest"} == {}
    assert {"fork-ci.yml:lint", "fork-ci.yml:test"} <= runners.keys()


@pytest.mark.parametrize("job", ["ci.yaml:detect", "ci.yaml:all-checks-pass", "docker.yml:mode", "nix.yml:detect"])
def test_upstream_keeps_its_entry_jobs_and_not_the_fork_ones(job):
    runners = _runners(UPSTREAM)
    assert job in runners
    assert not [name for name in runners if name.startswith("fork-ci.yml")]
