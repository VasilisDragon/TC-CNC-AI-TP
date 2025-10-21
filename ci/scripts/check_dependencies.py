#!/usr/bin/env python3
"""Dependency policy checks for CNCTC.

Runs pip-audit on configured requirement sets, checks pinned Python
versions against PyPI release dates, and verifies the vcpkg baseline age.
Produces a Markdown summary (stdout + GitHub step summary) and exits
non-zero if policy violations are detected.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import requests


SEVERITY_ORDER = {
    "UNKNOWN": 0,
    "LOW": 1,
    "MEDIUM": 2,
    "MODERATE": 2,  # pip-audit historically used MODERATE for MEDIUM
    "HIGH": 3,
    "CRITICAL": 4,
}

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = REPO_ROOT / "ci" / "deps_policy.json"
DEFAULT_CONFIG = {
    "python": {
        "pip_audit_min_severity": "CRITICAL",
        "max_age_warning_days": 180,
        "max_age_fail_days": 540,
        "requirements": [],
    },
    "vcpkg": {
        "manifest": "vcpkg.json",
        "max_baseline_age_days": 120,
    },
    "summary": {
        "success_message": "Dependency health policy satisfied.",
        "actions": {},
    },
}


@dataclass
class PythonVulnerability:
    requirement_name: str
    package: str
    installed_version: str
    advisory_id: str
    severity: str
    fix_versions: List[str]


@dataclass
class PythonRelease:
    requirement_name: str
    package: str
    version: str
    released: datetime
    age_days: int
    threshold_days: int
    severity: str


def load_config() -> Dict:
    if not CONFIG_PATH.exists():
        return DEFAULT_CONFIG

    with CONFIG_PATH.open("r", encoding="utf-8") as handle:
        loaded = json.load(handle)

    # Merge defaults where fields are missing to keep access straightforward.
    def merge(base: Dict, override: Dict) -> Dict:
        merged: Dict = dict(base)
        for key, value in override.items():
            if isinstance(value, dict) and isinstance(base.get(key), dict):
                merged[key] = merge(base[key], value)
            else:
                merged[key] = value
        return merged

    return merge(DEFAULT_CONFIG, loaded)


def run_pip_audit(
    requirement_name: str,
    requirements_path: Path,
    min_severity: str,
) -> Tuple[List[PythonVulnerability], List[str]]:
    cmd = [
        sys.executable,
        "-m",
        "pip_audit",
        "-r",
        str(requirements_path),
        "--format",
        "json",
        "--progress-spinner",
        "off",
    ]
    proc = subprocess.run(
        cmd,
        cwd=REPO_ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )

    stderr_lines = [line for line in proc.stderr.strip().splitlines() if line]

    if proc.returncode not in (0, 1):
        raise RuntimeError(
            f"pip-audit failed for {requirements_path} (exit code {proc.returncode}):\n"
            + "\n".join(stderr_lines)
        )

    try:
        audit_payload = json.loads(proc.stdout or "[]")
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            f"pip-audit returned invalid JSON for {requirements_path}: {exc}"
        ) from exc

    if isinstance(audit_payload, dict):
        if "dependencies" in audit_payload:
            audit_entries = audit_payload.get("dependencies", [])
        elif "results" in audit_payload:
            audit_entries = audit_payload.get("results", [])
        else:
            audit_entries = [audit_payload]
    else:
        audit_entries = audit_payload

    min_threshold = SEVERITY_ORDER.get(min_severity.upper(), SEVERITY_ORDER["CRITICAL"])
    vulnerabilities: List[PythonVulnerability] = []

    for finding in audit_entries:
        package = finding.get("name", "<unknown>")
        version = finding.get("version", "<unknown>")
        vulns = finding.get("vulns") or []
        for advisory in vulns:
            severity = (advisory.get("severity") or "UNKNOWN").upper()
            severity_value = SEVERITY_ORDER.get(severity, 0)
            if severity_value < min_threshold:
                continue
            vulnerabilities.append(
                PythonVulnerability(
                    requirement_name=requirement_name,
                    package=package,
                    installed_version=version,
                    advisory_id=advisory.get("id") or advisory.get("aliases", ["?"])[0],
                    severity=severity,
                    fix_versions=advisory.get("fix_versions") or [],
                )
            )

    return vulnerabilities, stderr_lines


def parse_requirements_file(path: Path) -> List[Tuple[str, str]]:
    pinned: List[Tuple[str, str]] = []
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        stripped = raw_line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        stripped = stripped.split(";", 1)[0].strip()
        if not stripped:
            continue
        if "==" not in stripped:
            continue  # Skip unpinned entries for now.
        package, version = stripped.split("==", maxsplit=1)
        package = package.split("[", 1)[0].strip()
        version = version.strip()
        if package and version:
            pinned.append((package, version))
    return pinned


def fetch_pypi_release_date(package: str, version: str) -> datetime:
    url = f"https://pypi.org/pypi/{package}/{version}/json"
    response = requests.get(url, timeout=20)
    if response.status_code != 200:
        raise RuntimeError(
            f"PyPI returned {response.status_code} for {package}=={version}"
        )

    data = response.json()
    release_candidates = data.get("releases", {}).get(version)
    if not release_candidates:
        urls = data.get("urls", [])
        release_candidates = [
            {
                "upload_time_iso_8601": url.get("upload_time_iso_8601"),
                "upload_time": url.get("upload_time"),
            }
            for url in urls
            if url.get("upload_time_iso_8601") or url.get("upload_time")
        ]
    if not release_candidates:
        # Best effort: fall back to project info's upload time if provided.
        upload_time = data.get("info", {}).get("upload_time_iso_8601")
        if not upload_time:
            raise RuntimeError(f"No release metadata for {package}=={version}")
        release_candidates = [{"upload_time_iso_8601": upload_time}]

    timestamps = []
    for candidate in release_candidates:
        ts = candidate.get("upload_time_iso_8601") or candidate.get("upload_time")
        if not ts:
            continue
        timestamps.append(_parse_iso_timestamp(ts))

    if not timestamps:
        raise RuntimeError(f"No usable release timestamps for {package}=={version}")

    return max(timestamps)


def _parse_iso_timestamp(value: str) -> datetime:
    normalized = value.replace("Z", "+00:00")
    return datetime.fromisoformat(normalized).astimezone(timezone.utc)


def check_python_release_ages(
    requirement_name: str,
    requirements_path: Path,
    warning_days: Optional[int],
    fail_days: Optional[int],
) -> Tuple[List[PythonRelease], List[PythonRelease]]:
    now = datetime.now(timezone.utc)
    warnings: List[PythonRelease] = []
    failures: List[PythonRelease] = []
    warn_threshold = int(warning_days) if warning_days is not None else None
    fail_threshold = int(fail_days) if fail_days is not None else None

    if warn_threshold is not None and warn_threshold <= 0:
        warn_threshold = None
    if fail_threshold is not None and fail_threshold <= 0:
        fail_threshold = None
    if (
        warn_threshold is not None
        and fail_threshold is not None
        and fail_threshold <= warn_threshold
    ):
        fail_threshold = None

    for package, version in parse_requirements_file(requirements_path):
        released = fetch_pypi_release_date(package, version)
        age_days = (now - released).days
        if fail_threshold is not None and age_days > fail_threshold:
            failures.append(
                PythonRelease(
                    requirement_name=requirement_name,
                    package=package,
                    version=version,
                    released=released,
                    age_days=age_days,
                    threshold_days=fail_threshold,
                    severity="failure",
                )
            )
        elif warn_threshold is not None and age_days > warn_threshold:
            warnings.append(
                PythonRelease(
                    requirement_name=requirement_name,
                    package=package,
                    version=version,
                    released=released,
                    age_days=age_days,
                    threshold_days=warn_threshold,
                    severity="warning",
                )
            )
    return warnings, failures


def check_vcpkg_baseline(manifest_path: Path, max_age_days: int) -> Optional[Tuple[str, int, datetime]]:
    if not manifest_path.exists():
        return None

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    baseline = manifest.get("builtin-baseline")
    if not baseline:
        return None

    url = f"https://api.github.com/repos/microsoft/vcpkg/commits/{baseline}"
    headers = {
        "Accept": "application/vnd.github+json",
        "User-Agent": "cnctc-dependency-audit",
    }
    response = requests.get(url, headers=headers, timeout=20)
    if response.status_code != 200:
        raise RuntimeError(
            f"GitHub API returned {response.status_code} for vcpkg baseline {baseline}"
        )

    payload = response.json()
    commit_info = payload.get("commit", {}).get("committer") or {}
    committed_at_raw = commit_info.get("date")
    if not committed_at_raw:
        raise RuntimeError(f"Missing commit date for vcpkg baseline {baseline}")
    committed_at = _parse_iso_timestamp(committed_at_raw)
    age_days = (datetime.now(timezone.utc) - committed_at).days
    if age_days > max_age_days:
        return baseline, age_days, committed_at
    return None


def write_summary(lines: List[str]) -> None:
    summary = "\n".join(lines).strip() + "\n"
    print(summary)
    summary_path = os.getenv("GITHUB_STEP_SUMMARY")
    if summary_path:
        Path(summary_path).write_text(summary, encoding="utf-8")


def main() -> int:
    config = load_config()
    python_cfg = config.get("python", {})
    vcpkg_cfg = config.get("vcpkg", {})
    summary_cfg = config.get("summary", {})

    requirement_sets = python_cfg.get("requirements", [])
    pip_audit_min_severity = python_cfg.get("pip_audit_min_severity", "CRITICAL")

    python_vulns: List[PythonVulnerability] = []
    python_vuln_logs: Dict[str, List[str]] = {}
    python_age_warnings: List[PythonRelease] = []
    python_age_failures: List[PythonRelease] = []
    actions: List[str] = []

    for requirement in requirement_sets:
        name = requirement.get("name") or requirement.get("path")
        path = REPO_ROOT / requirement["path"]
        if not path.exists():
            raise FileNotFoundError(f"Requirement file not found: {path}")

        req_min_severity = requirement.get(
            "pip_audit_min_severity", pip_audit_min_severity
        )

        vulns, stderr_lines = run_pip_audit(name, path, req_min_severity)
        if stderr_lines:
            python_vuln_logs[name] = stderr_lines
        python_vulns.extend(vulns)

        warn_days = requirement.get("max_age_warning_days")
        if warn_days is None:
            warn_days = requirement.get("max_age_days")
        if warn_days is None:
            warn_days = python_cfg.get("max_age_warning_days")
        if warn_days is None:
            warn_days = python_cfg.get("max_age_days")
        warn_days = int(warn_days) if warn_days is not None else None

        fail_days = requirement.get("max_age_fail_days")
        if fail_days is None:
            fail_days = python_cfg.get("max_age_fail_days")
        fail_days = int(fail_days) if fail_days is not None else None

        warnings, failures = check_python_release_ages(
            name, path, warn_days, fail_days
        )
        python_age_warnings.extend(warnings)
        python_age_failures.extend(failures)

    vcpkg_manifest = REPO_ROOT / vcpkg_cfg.get("manifest", "vcpkg.json")
    vcpkg_issue = None
    if vcpkg_cfg:
        vcpkg_max_age = int(vcpkg_cfg.get("max_baseline_age_days", 120))
        vcpkg_issue = check_vcpkg_baseline(vcpkg_manifest, vcpkg_max_age)

    summary_lines: List[str] = ["# Dependency Health Report"]

    if requirement_sets:
        summary_lines.append("")
        summary_lines.append("## Python vulnerability scan")
        if python_vulns:
            summary_lines.append(
                f"- Detected {len(python_vulns)} critical vulnerabilities across pip requirement sets."
            )
            for vuln in python_vulns:
                fix_hint = (
                    f" (fix versions: {', '.join(vuln.fix_versions)})"
                    if vuln.fix_versions
                    else ""
                )
                summary_lines.append(
                    f"  - `{vuln.requirement_name}`: {vuln.package}=={vuln.installed_version} "
                    f"{vuln.advisory_id} ({vuln.severity}){fix_hint}"
                )
            actions.append(
                summary_cfg.get("actions", {}).get(
                    "python_vuln",
                    "Update the affected packages and regenerate requirement locks.",
                )
            )
        else:
            summary_lines.append("- No critical vulnerabilities detected by pip-audit.")
        if python_vuln_logs:
            summary_lines.append("  - pip-audit warnings:")
            for name, lines in python_vuln_logs.items():
                for line in lines:
                    summary_lines.append(f"    - [{name}] {line}")

        summary_lines.append("")
        summary_lines.append("## Python release freshness")
        if python_age_failures:
            summary_lines.append(
                f"- Failure: {len(python_age_failures)} pinned releases exceeded the freshness limit."
            )
            for stale in python_age_failures:
                released_str = stale.released.strftime("%Y-%m-%d")
                summary_lines.append(
                    f"  - `{stale.requirement_name}`: {stale.package}=={stale.version} "
                    f"released {released_str} ({stale.age_days} days old, limit {stale.threshold_days})"
                )
            actions.append(
                summary_cfg.get("actions", {}).get(
                    "python_stale",
                    "Upgrade stale Python packages and regenerate requirement locks.",
                )
            )
        if python_age_warnings:
            summary_lines.append(
                f"- Advisory: {len(python_age_warnings)} pinned releases exceeded the warning window (non-blocking)."
            )
            for warn in python_age_warnings:
                released_str = warn.released.strftime("%Y-%m-%d")
                summary_lines.append(
                    f"  - `{warn.requirement_name}`: {warn.package}=={warn.version} "
                    f"released {released_str} ({warn.age_days} days old, warning {warn.threshold_days})"
                )
            actions.append(
                summary_cfg.get("actions", {}).get(
                    "python_warn",
                    "Plan upgrades for warning-level pins.",
                )
            )
        if not python_age_failures and not python_age_warnings:
            summary_lines.append("- All pinned Python packages are within freshness policy.")

    if vcpkg_cfg:
        summary_lines.append("")
        summary_lines.append("## vcpkg baseline age")
        if vcpkg_issue:
            baseline, age_days, committed_at = vcpkg_issue
            committed_str = committed_at.strftime("%Y-%m-%d")
            summary_lines.append(
                f"- Built-in baseline {baseline} is {age_days} days old (committed {committed_str})."
            )
            actions.append(
                summary_cfg.get("actions", {}).get(
                    "vcpkg_stale",
                    "Update the vcpkg builtin-baseline to a recent commit.",
                )
            )
        else:
            summary_lines.append("- vcpkg baseline age is within policy.")

    summary_lines.append("")
    summary_lines.append("## Next steps")
    if actions:
        unique_actions = []
        for action in actions:
            if action and action not in unique_actions:
                unique_actions.append(action)
        for idx, action in enumerate(unique_actions, start=1):
            summary_lines.append(f"- {idx}. {action}")
    else:
        summary_lines.append(
            f"- {summary_cfg.get('success_message', 'All checks passed.')}"
        )

    write_summary(summary_lines)

    has_failures = bool(python_vulns or python_age_failures or vcpkg_issue)
    return 1 if has_failures else 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as exc:  # pylint: disable=broad-except
        print(f"Dependency policy check failed: {exc}", file=sys.stderr)
        sys.exit(2)


