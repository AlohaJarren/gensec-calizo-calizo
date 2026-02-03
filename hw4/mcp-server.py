"""
Non-Human Identity (NHI) security MCP server
"""
from __future__ import annotations

import hashlib
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

from fastmcp import FastMCP

mcp = FastMCP("nhi_guard")


# -----------------------------
# Detection patterns (heuristic)
# -----------------------------
@dataclass(frozen=True)
class Pattern:
    name: str
    regex: re.Pattern
    weight: int


def _compile_patterns() -> List[Pattern]:
    # NOTE: Keep these broad and heuristic. We are demonstrating the concept,
    # not building a perfect secret scanner.
    raw: List[Tuple[str, str, int]] = [
        ("aws_access_key_id", r"\bAKIA[0-9A-Z]{16}\b", 90),
        ("github_pat", r"\bghp_[A-Za-z0-9]{36}\b", 85),
        ("slack_token", r"\bxox[baprs]-[A-Za-z0-9-]{10,}\b", 80),
        ("google_api_key", r"\bAIza[0-9A-Za-z\-_]{35}\b", 75),
        ("private_key_block", r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----", 100),
        # Generic "key=value" style secrets (low confidence).
        (
            "generic_secret_assignment",
            r"(?i)\b(api[_-]?key|access[_-]?key|secret|token|password)\b\s*[:=]\s*['\"]?([A-Za-z0-9_/\-]{12,})",
            55,
        ),
    ]
    return [Pattern(n, re.compile(rx), w) for (n, rx, w) in raw]


PATTERNS = _compile_patterns()

TEXT_FILE_EXT_ALLOWLIST = {
    ".txt",
    ".md",
    ".py",
    ".js",
    ".ts",
    ".json",
    ".yaml",
    ".yml",
    ".env",
    ".ini",
    ".cfg",
    ".toml",
    ".sh",
    ".bash",
    ".zsh",
    ".properties",
}


def _is_probably_text(path: Path, max_bytes: int = 1024) -> bool:
    """
    Heuristic text/binary check: if there are NUL bytes in the first chunk, treat as binary.
    """
    try:
        with path.open("rb") as f:
            chunk = f.read(max_bytes)
        return b"\x00" not in chunk
    except Exception:
        return False


def _sha256_hex(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8", errors="ignore")).hexdigest()


def _redact(value: str) -> str:
    """
    Return a safe preview: keep at most first 3 and last 3 chars.
    """
    v = value.strip()
    if len(v) <= 8:
        return "***"
    return f"{v[:3]}...{v[-3:]}"


def _walk_files(root: Path, include_hidden: bool) -> List[Path]:
    out: List[Path] = []
    for p in root.rglob("*"):
        if not p.is_file():
            continue
        if not include_hidden and any(part.startswith(".") for part in p.parts):
            continue
        out.append(p)
    return out


def _should_scan_file(path: Path, max_file_kb: int) -> bool:
    if path.suffix.lower() not in TEXT_FILE_EXT_ALLOWLIST and path.name not in {".env", "Dockerfile"}:
        return False
    try:
        if path.stat().st_size > max_file_kb * 1024:
            return False
    except Exception:
        return False
    return _is_probably_text(path)


def _extract_findings_from_text(text: str) -> List[Dict[str, Any]]:
    findings: List[Dict[str, Any]] = []

    for pat in PATTERNS:
        if pat.name == "generic_secret_assignment":
            for m in pat.regex.finditer(text):
                key_name = m.group(1)
                value = m.group(2)
                fp = _sha256_hex(value)
                findings.append(
                    {
                        "kind": pat.name,
                        "confidence": "low",
                        "weight": pat.weight,
                        "key_name": key_name,
                        "secret_fingerprint": fp,
                        "secret_preview": _redact(value),
                    }
                )
        elif pat.name == "private_key_block":
            for _m in pat.regex.finditer(text):
                findings.append(
                    {
                        "kind": pat.name,
                        "confidence": "high",
                        "weight": pat.weight,
                        "secret_fingerprint": _sha256_hex(pat.name),
                        "secret_preview": "-----BEGIN ... PRIVATE KEY-----",
                    }
                )
        else:
            for m in pat.regex.finditer(text):
                value = m.group(0)
                fp = _sha256_hex(value)
                findings.append(
                    {
                        "kind": pat.name,
                        "confidence": "medium",
                        "weight": pat.weight,
                        "secret_fingerprint": fp,
                        "secret_preview": _redact(value),
                    }
                )

    return findings


@mcp.tool()
async def scan_path(path: str, include_hidden: bool = False, max_file_kb: int = 256, max_results: int = 200) -> Dict[str, Any]:
    """
    Scan a local path for likely NHI artifacts (API keys, tokens, private keys, etc.).

    Args:
        path: A file or directory path to scan.
        include_hidden: If true, include dotfiles/directories.
        max_file_kb: Skip files larger than this size.
        max_results: Cap total findings returned.

    Returns:
        A JSON-serializable dict:
          {
            "scanned_root": "...",
            "files_scanned": N,
            "findings": [ { ... } ],
            "summary": { ... }
          }
    """
    root = Path(path).expanduser().resolve()
    if not root.exists():
        return {"error": f"path does not exist: {root}"}

    files: List[Path]
    if root.is_file():
        files = [root]
    else:
        files = _walk_files(root, include_hidden)

    findings_out: List[Dict[str, Any]] = []
    files_scanned = 0

    for fp in files:
        if not _should_scan_file(fp, max_file_kb):
            continue
        files_scanned += 1

        try:
            text = fp.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue

        fnds = _extract_findings_from_text(text)
        for f in fnds:
            f2 = dict(f)
            f2["file"] = str(fp)
            # We do not include the full line. Provide a stable location hint only.
            f2["location_hint"] = "n/a"
            findings_out.append(f2)
            if len(findings_out) >= max_results:
                break

        if len(findings_out) >= max_results:
            break

    summary: Dict[str, Any] = {}
    by_kind: Dict[str, int] = {}
    for f in findings_out:
        by_kind[f["kind"]] = by_kind.get(f["kind"], 0) + 1
    summary["by_kind"] = by_kind
    summary["total_findings"] = len(findings_out)

    return {
        "scanned_root": str(root),
        "files_scanned": files_scanned,
        "findings": findings_out,
        "summary": summary,
    }


@mcp.tool()
async def score_findings(findings: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Score findings (0-100) using a simple heuristic.

    Args:
        findings: The "findings" array returned by scan_path.

    Returns:
        {
          "max_score": int,
          "average_score": float,
          "scored": [ { finding + score fields } ]
        }
    """
    if not findings:
        return {"max_score": 0, "average_score": 0.0, "scored": []}

    scored: List[Dict[str, Any]] = []
    total = 0
    max_score = 0

    for f in findings:
        base = int(f.get("weight", 30))
        conf = f.get("confidence", "low")
        bump = 0
        if conf == "high":
            bump = 10
        elif conf == "medium":
            bump = 5
        score = min(100, base + bump)

        f2 = dict(f)
        f2["risk_score"] = score
        scored.append(f2)

        total += score
        max_score = max(max_score, score)

    avg = total / max(1, len(scored))
    return {"max_score": max_score, "average_score": avg, "scored": scored}


@mcp.tool()
async def remediation_plan(scored_findings: List[Dict[str, Any]], top_n: int = 10) -> Dict[str, Any]:
    """
    Generate remediation tasks for the highest-risk findings.

    Args:
        scored_findings: Output from score_findings["scored"].
        top_n: Number of tasks to return.

    Returns:
        {
          "tasks": [ { ... } ],
          "notes": [ ... ]
        }
    """
    if not scored_findings:
        return {"tasks": [], "notes": ["no findings provided"]}

    ordered = sorted(scored_findings, key=lambda x: int(x.get("risk_score", 0)), reverse=True)
    tasks: List[Dict[str, Any]] = []

    for f in ordered[:top_n]:
        kind = f.get("kind", "unknown")
        file_path = f.get("file", "unknown")
        fp = f.get("secret_fingerprint", "unknown")
        score = int(f.get("risk_score", 0))

        if kind in {"private_key_block"}:
            action = "rotate_or_revoke_key_material"
            detail = "Treat as key compromise risk. Rotate immediately and remove from repo history."
        elif kind in {"aws_access_key_id", "github_pat", "slack_token", "google_api_key"}:
            action = "rotate_secret_and_move_to_env"
            detail = "Rotate the credential and store it in a secret manager or env var injection, not in source."
        else:
            action = "review_and_scope_down"
            detail = "Confirm if this is a real secret. If yes, rotate and scope down permissions."

        tasks.append(
            {
                "task_id": fp[:12],
                "action": action,
                "priority": "P0" if score >= 90 else ("P1" if score >= 75 else "P2"),
                "file": file_path,
                "secret_fingerprint": fp,
                "rationale": f"kind={kind}, score={score}",
                "detail": detail,
            }
        )

    notes = [
        "Tasks are heuristic. Validate before rotating production credentials.",
        "This server never returns raw secrets. Only fingerprints/previews.",
    ]
    return {"tasks": tasks, "notes": notes}


def _run_stdio() -> None:
    mcp.run(transport="stdio")


def _run_http() -> None:
    # If you choose to deploy later, keep this path.
    mcp.run(transport="http", host="0.0.0.0", port=8080)


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "stdio"
    if mode == "stdio":
        _run_stdio()
    else:
        _run_http()
