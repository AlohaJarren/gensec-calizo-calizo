from __future__ import annotations

import argparse
import gzip
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Ensure repo root is on sys.path when running "python hw7/app.py ..."
_THIS_FILE = Path(__file__).resolve()
_REPO_ROOT = _THIS_FILE.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def _run_cmd(args: List[str], timeout_s: int = 30, env: Optional[Dict[str, str]] = None) -> Tuple[int, str, str]:
    """Run a local command and return (returncode, stdout, stderr)."""
    full_env = os.environ.copy()
    if env:
        full_env.update(env)
    p = subprocess.run(
        args,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=timeout_s,
        env=full_env,
    )
    return p.returncode, p.stdout or "", p.stderr or ""


def list_running_services_raw(max_lines: int = 200) -> List[str]:
    """
    Raw lines from:
      systemctl list-units --type=service --state=running --no-pager --no-legend
    """
    rc, out, _ = _run_cmd(
        ["systemctl", "list-units", "--type=service", "--state=running", "--no-pager", "--no-legend"],
        timeout_s=20,
    )
    if rc != 0:
        return []
    lines = [ln.rstrip("\n") for ln in out.splitlines() if ln.strip()]
    return lines[:max_lines]


def list_running_service_units() -> List[str]:
    """Return only the unit names from running services output."""
    units: List[str] = []
    for line in list_running_services_raw():
        parts = line.split(None, 1)
        if parts:
            units.append(parts[0])
    return units


def list_installed_packages() -> Dict[str, str]:
    """
    Return {package: version} using:
      dpkg-query -W -f=${Package} ${Version}\\n
    """
    rc, out, _ = _run_cmd(["dpkg-query", "-W", "-f=${Package} ${Version}\n"], timeout_s=40)
    if rc != 0:
        return {}
    pkgs: Dict[str, str] = {}
    for line in out.splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split(None, 1)
        if len(parts) == 2:
            pkgs[parts[0]] = parts[1]
    return pkgs


def _apt_changelog_base() -> Optional[List[str]]:
    """Prefer apt-get changelog, else apt changelog."""
    if shutil.which("apt-get"):
        return ["apt-get", "changelog"]
    if shutil.which("apt"):
        return ["apt", "changelog"]
    return None


def _read_local_changelog(package_name: str) -> Optional[str]:
    """
    Fallback: read changelog shipped with the installed package, if present.
    Typical Debian-style locations:
      /usr/share/doc/<pkg>/changelog.Debian.gz
      /usr/share/doc/<pkg>/changelog.gz
      /usr/share/doc/<pkg>/changelog.Debian
      /usr/share/doc/<pkg>/changelog
    """
    docdir = Path("/usr/share/doc") / package_name
    candidates = [
        docdir / "changelog.Debian.gz",
        docdir / "changelog.gz",
        docdir / "changelog.Debian",
        docdir / "changelog",
    ]
    for p in candidates:
        if not p.exists():
            continue
        try:
            if p.suffix == ".gz":
                with gzip.open(p, "rt", encoding="utf-8", errors="replace") as f:
                    return f.read()
            return p.read_text(encoding="utf-8", errors="replace")
        except Exception:
            continue
    return None


def get_package_changelog(package_name: str) -> Tuple[int, str, str]:
    """
    Get changelog text for a package.

    Primary: apt-get/apt changelog (may fail on Kali with "Changelog unavailable").
    Fallback: /usr/share/doc/<pkg>/changelog* if installed.
    """
    base = _apt_changelog_base()
    if base:
        env = {"PAGER": "cat", "LESS": "FRX"}
        rc, out, err = _run_cmd(base + [package_name], timeout_s=90, env=env)
        if rc == 0 and out.strip():
            return rc, out, err

        local = _read_local_changelog(package_name)
        if local:
            return 0, local, (err or "").strip()
        return rc, out, err

    local = _read_local_changelog(package_name)
    if local:
        return 0, local, ""
    return 127, "", "Neither apt-get nor apt found on PATH, and no local changelog found."


_CVE_RE = re.compile(r"\bCVE-\d{4}-\d{4,7}\b", re.IGNORECASE)


def extract_cves(text: str) -> List[str]:
    """Extract unique CVE identifiers from text, preserving first-seen order."""
    found = _CVE_RE.findall(text or "")
    seen = set()
    out: List[str] = []
    for c in found:
        c_up = c.upper()
        if c_up not in seen:
            seen.add(c_up)
            out.append(c_up)
    return out


def extract_security_lines(text: str, max_lines: int = 40) -> List[str]:
    """Extract a small set of security-relevant lines from changelog text."""
    hits: List[str] = []
    keywords = ("cve-", "security", "vuln", "fix", "exploit")
    for raw in (text or "").splitlines():
        line = raw.strip()
        if not line:
            continue
        lower = line.lower()
        if any(k in lower for k in keywords):
            hits.append(line)
        if len(hits) >= max_lines:
            break
    return hits


def guess_ssh_targets(running_units: List[str], installed: Dict[str, str]) -> Tuple[Optional[str], List[str]]:
    """Best-effort guesses for SSH service unit and server package(s)."""
    unit: Optional[str] = None
    for u in ("ssh.service", "sshd.service", "dropbear.service"):
        if u in running_units:
            unit = u
            break
    if unit is None:
        for u in running_units:
            if "ssh" in u and u.endswith(".service"):
                unit = u
                break

    candidates: List[str] = []
    for pkg in ("openssh-server", "dropbear", "tinysshd", "lsh-server"):
        if pkg in installed:
            candidates.append(pkg)

    return unit, candidates


def _md_code_block(s: str) -> str:
    return "```\n" + (s or "").rstrip() + "\n```\n"


def render_report_md(report: Dict[str, object]) -> str:
    lines: List[str] = []
    lines.append("# HW7 Package Vulnerability Mini-Advisor Report")
    lines.append("")
    lines.append(f"- Target: {report.get('target', '')}")
    lines.append(f"- Service unit: {report.get('service_unit') or 'unknown'}")
    lines.append("")

    lines.append("## Candidate packages")
    lines.append("")
    pkgs = report.get("candidate_packages") or []
    if not pkgs:
        lines.append("- None detected (limitation: heuristic match only)")
    else:
        vers = report.get("package_versions") or {}
        for p in pkgs:  # type: ignore
            v = vers.get(p, "")
            lines.append(f"- {p} {v}".rstrip())
    lines.append("")

    lines.append("## Extracted CVE references (best-effort)")
    lines.append("")
    cves = report.get("cves") or []
    if not cves:
        lines.append("- None found in changelog text")
    else:
        for c in cves:  # type: ignore
            lines.append(f"- {c}")
    lines.append("")

    lines.append("## Security-relevant changelog lines (subset)")
    lines.append("")
    sec_lines = report.get("security_lines") or []
    if not sec_lines:
        lines.append("- None")
    else:
        lines.append(_md_code_block("\n".join(sec_lines)))  # type: ignore

    lines.append("## Remediation command plan (printed only, not executed)")
    lines.append("")
    cmds = report.get("remediation_commands") or []
    lines.append(_md_code_block("\n".join(cmds)))  # type: ignore

    lines.append("## Known limitations")
    lines.append("")
    for item in (report.get("limitations") or []):  # type: ignore
        lines.append(f"- {item}")
    lines.append("")
    return "\n".join(lines)


def write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def build_report(target: str = "ssh") -> Dict[str, object]:
    running_units = list_running_service_units()
    running_raw_subset = list_running_services_raw(max_lines=20)
    installed = list_installed_packages()

    service_unit: Optional[str] = None
    candidate_packages: List[str] = []

    if target.lower() == "ssh":
        service_unit, candidate_packages = guess_ssh_targets(running_units, installed)

    changelog_text = ""
    changelog_status: Dict[str, Dict[str, object]] = {}
    for pkg in candidate_packages:
        rc, out, err = get_package_changelog(pkg)
        changelog_status[pkg] = {
            "returncode": rc,
            "stdout_len": len(out),
            "stderr_len": len(err),
            "stderr_preview": (err.strip()[:160] if err else ""),
        }
        if out:
            changelog_text += "\n" + out

    cves = extract_cves(changelog_text)
    security_lines = extract_security_lines(changelog_text, max_lines=40)

    remediation: List[str] = []
    if candidate_packages:
        remediation.append("sudo apt-get update")
        for pkg in candidate_packages:
            remediation.append(f"sudo apt-get install --only-upgrade {pkg}")
        if service_unit:
            remediation.append(f"sudo systemctl restart {service_unit}")
    else:
        remediation.append("# No candidate package detected - nothing to upgrade automatically")

    limitations = [
        "Scan mode is deterministic-only (no LLM used).",
        "Package and service detection is heuristic (may miss non-standard SSH servers).",
        "Changelog parsing is best-effort and does not prove vulnerability presence or absence.",
        "No upgrades or restarts are executed (command plan only).",
        "On some distros (including Kali), apt changelog may be unavailable; local /usr/share/doc fallback is used.",
    ]

    versions = {p: installed.get(p, "") for p in candidate_packages}
    return {
        "target": target,
        "service_unit": service_unit,
        "candidate_packages": candidate_packages,
        "package_versions": versions,
        "running_services_raw_subset": running_raw_subset,
        "changelog_status": changelog_status,
        "cves": cves,
        "security_lines": security_lines,
        "remediation_commands": remediation,
        "limitations": limitations,
    }


def run_mcp_server() -> None:
    """
    Start MCP server exposing constrained inspection tools (stdio transport).
    """
    try:
        from mcp.server.fastmcp import FastMCP  # type: ignore
    except Exception:
        print("ERROR: Could not import MCP FastMCP.", file=sys.stderr)
        print("Fix: run with uv:", file=sys.stderr)
        print("  uv add \"mcp[cli]\"", file=sys.stderr)
        print("  uv run python hw7/app.py mcp-server", file=sys.stderr)
        sys.exit(2)

    mcp = FastMCP("HW7 Package Advisor")

    @mcp.tool()
    def list_running_services() -> List[str]:
        """Return running systemd service unit names."""
        return list_running_service_units()

    @mcp.tool()
    def list_installed_packages_tool() -> Dict[str, str]:
        """Return installed packages and versions (dpkg-query)."""
        return list_installed_packages()

    @mcp.tool()
    def get_package_changelog_tool(package_name: str) -> str:
        """Return changelog text for a package (apt changelog; fallback to /usr/share/doc)."""
        rc, out, err = get_package_changelog(package_name)
        if rc != 0 and not out.strip():
            return (err or "").strip()
        return (out or "").strip()

    mcp.run()


def main() -> None:
    parser = argparse.ArgumentParser(description="HW7 Package Vulnerability Mini-Advisor")
    sub = parser.add_subparsers(dest="cmd", required=True)

    scan = sub.add_parser("scan", help="Scan a target and write a report artifact")
    scan.add_argument("--target", default="ssh", help="Target to scan (default: ssh)")
    scan.add_argument("--out", default="hw7/out/report.md", help="Output report path")
    scan.add_argument("--json", action="store_true", help="Also print JSON to stdout")

    sub.add_parser("mcp-server", help="Run MCP server (stdio transport)")

    agent = sub.add_parser("agent", help="Run Gemini-driven MCP agent and write artifacts")
    agent.add_argument("question", nargs="+", help="Question to answer (wrap in quotes recommended)")
    agent.add_argument("--out-dir", default="hw7/out", help="Output directory for artifacts")

    args = parser.parse_args()

    if args.cmd == "mcp-server":
        run_mcp_server()
        return

    if args.cmd == "agent":
        question = " ".join(args.question).strip()

        # Import agent client (works both as package import and local import).
        try:
            from hw7.agent_client import run_agent  # type: ignore
        except ModuleNotFoundError:
            from agent_client import run_agent  # type: ignore

        answer, answer_md, trace_json = run_agent(user_question=question, out_dir=args.out_dir)
        print("Agent answer written to:", answer_md)
        print("Agent trace written to:", trace_json)
        print("")
        print(answer)
        return

    report = build_report(target=args.target)
    md = render_report_md(report)
    out_path = Path(args.out)
    write_text(out_path, md)

    print(f"Wrote report: {out_path}")
    print("")
    print("Remediation command plan (not executed):")
    for c in report.get("remediation_commands", []):  # type: ignore
        print(c)

    if args.json:
        print("")
        print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
