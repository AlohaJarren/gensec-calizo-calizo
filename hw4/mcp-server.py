"""
Non-Human Identity (NHI) security MCP server
"""
from __future__ import annotations

import hashlib
import os
import re
import sys
from pathlib import Path
from typing import Any

from fastmcp import FastMCP


mcp = FastMCP("nhi-security")

DEFAULT_EXCLUDES = {".git", ".venv", "node_modules", "__pycache__", ".pytest_cache"}

# Demo-safe patterns (kept generic and minimal).
# The goal is to show the workflow, not to build a perfect scanner.
PATTERNS: list[tuple[str, str]] = [
    (r"AKIA[0-9A-Z]{16}", "aws_access_key_id_like"),
    (r"-----BEGIN (?:RSA|EC|OPENSSH) PRIVATE KEY-----", "private_key_block"),
    (r"(?i)\bokta[a-z0-9_\-]{10,}\b", "okta_token_like"),
    (r"(?i)\bapi[_-]?key\s*[:=]\s*[A-Za-z0-9_\-]{16,}\b", "api_key_assignment"),
]

OWNER_HINT = re.compile(
    r"(?i)\bowner\s*[:=]\s*([A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,})\b"
)

def _sha256_hex(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()

def _read_text_safely(p: Path, max_bytes: int) -> str:
    data = p.read_bytes()
    if len(data) > max_bytes:
        data = data[:max_bytes]
    return data.decode("utf-8", errors="ignore")

def _iter_files(paths: list[str]) -> list[Path]:
    out: list[Path] = []
    for raw in paths:
        root = Path(raw).expanduser().resolve()
        if root.is_file():
            out.append(root)
            continue
        if root.is_dir():
            for p in root.rglob("*"):
                if not p.is_file():
                    continue
                if any(part in DEFAULT_EXCLUDES for part in p.parts):
                    continue
                out.append(p)
    return out

@mcp.tool("scan_repo")
async def scan_repo(paths: list[str], max_file_kb: int = 256) -> list[dict[str, Any]]:
    """
    Scan provided filesystem paths for likely secrets / machine identities.

    Args:
      paths: List of file or directory paths to scan.
      max_file_kb: Maximum KB to read per file (prevents scanning huge files).

    Returns:
      A list of findings. Each finding includes:
        - fingerprint: sha256 of the matched string (not the raw match)
        - kind: pattern label
        - file: file path
        - owner_email: parsed owner hint if present
        - evidence: short snippet with the match redacted
    """
    max_bytes = int(max_file_kb) * 1024
    findings: list[dict[str, Any]] = []

    for p in _iter_files(paths):
        try:
            text = _read_text_safely(p, max_bytes=max_bytes)
        except Exception:
            continue

        owner_email = None
        m = OWNER_HINT.search(text)
        if m:
            owner_email = m.group(1)

        for pat, kind in PATTERNS:
            for hit in re.finditer(pat, text):
                raw = hit.group(0)
                fp = _sha256_hex(raw)

                start = max(0, hit.start() - 30)
                end = min(len(text), hit.end() + 30)
                snippet = text[start:end].replace(raw, "[REDACTED]")

                findings.append(
                    {
                        "fingerprint": fp,
                        "kind": kind,
                        "file": str(p),
                        "owner_email": owner_email,
                        "evidence": snippet,
                    }
                )

    return findings

@mcp.tool("risk_score")
async def risk_score(findings: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    Assign an explainable risk score (0-100) to each finding.

    Heuristic scoring, intended for demo and prioritization:
    - Private keys score highest
    - Token/key-like strings score medium-high
    - Unknown owners increase risk (harder remediation)
    - Certain file locations (config/.env) increase risk

    Returns:
      Findings enriched with:
        - score: int 0-100
        - reasons: list[str]
    """
    scored: list[dict[str, Any]] = []

    for f in findings:
        score = 10
        reasons: list[str] = []

        kind = (f.get("kind") or "").lower()
        file_path = (f.get("file") or "").lower()
        has_owner = bool(f.get("owner_email"))

        if "private_key" in kind:
            score += 75
            reasons.append("private key material detected")
        elif "token" in kind:
            score += 45
            reasons.append("token-like string detected")
        elif "api_key" in kind:
            score += 40
            reasons.append("api key-like assignment detected")
        else:
            score += 25
            reasons.append("credential-like pattern detected")

        if ".env" in file_path or "config" in file_path or "settings" in file_path:
            score += 10
            reasons.append("found in likely configuration file")

        if not has_owner:
            score += 10
            reasons.append("no owner hint found")

        score = max(0, min(100, score))

        f2 = dict(f)
        f2["score"] = score
        f2["reasons"] = reasons
        scored.append(f2)

    scored.sort(key=lambda x: int(x.get("score", 0)), reverse=True)
    return scored

@mcp.tool("build_identity_graph")
async def build_identity_graph(findings: list[dict[str, Any]]) -> dict[str, Any]:
    """
    Build a simple identity graph from findings.

    Graph format:
      - nodes: dict[node_id] = {"type": "...", "label": "..."}
      - edges: list[{"src": "...", "dst": "...", "rel": "..."}]

    Nodes:
      - secret fingerprints
      - owner emails (or unknown_owner)
      - file locations
    """
    nodes: dict[str, dict[str, str]] = {}
    edges: list[dict[str, str]] = []

    for f in findings:
        sec = str(f.get("fingerprint") or "")
        owner = str(f.get("owner_email") or "unknown_owner")
        loc = str(f.get("file") or "unknown_location")

        if sec:
            nodes.setdefault(sec, {"type": "secret", "label": sec[:12]})
        nodes.setdefault(owner, {"type": "owner", "label": owner})
        nodes.setdefault(loc, {"type": "location", "label": loc})

        if sec:
            edges.append({"src": sec, "dst": owner, "rel": "owned_by"})
            edges.append({"src": sec, "dst": loc, "rel": "found_in"})

    return {"nodes": nodes, "edges": edges}

@mcp.tool("explain_limitations")
async def explain_limitations() -> str:
    """
    Return known limitations for rubric + screencast.
    """
    return (
        "Limitations:\n"
        "- Regex-based scanning only (false positives/negatives).\n"
        "- Does not validate real permissions/scopes for found tokens.\n"
        "- No automatic remediation actions (planning only).\n"
        "- Owner detection is best-effort via 'owner: email' hints.\n"
    )

def _run_stdio() -> None:
    mcp.run(transport="stdio")

def _run_http() -> None:
    port = int(os.environ.get("PORT", "8080"))
    mcp.run(transport="http", host="0.0.0.0", port=port)

if __name__ == "__main__":
    # Support lab-like "stdio" argument, default to stdio.
    mode = "stdio"
    if len(sys.argv) >= 2:
        mode = sys.argv[1].strip().lower()

    if mode == "http":
        _run_http()
    else:
        _run_stdio()
