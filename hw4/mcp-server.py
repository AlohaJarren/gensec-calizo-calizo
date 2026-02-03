"""
Non-Human Identity (NHI) security MCP server
"""
from __future__ import annotations

import hashlib
import os
import re
import sys
import time
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
    # NOTE: These are heuristic patterns. Expect noise.
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
        # JWT-like tokens are sensitive, but not always a credential.
        ("jwt_like", r"\beyJ[a-zA-Z0-9_-]{10,}\.[a-zA-Z0-9_-]{10,}\.[a-zA-Z0-9_-]{10,}\b", 60),
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

OWNER_HINT = re.compile(
    r"(?i)\b(owner|contact|maintainer)\b\s*[:=]\s*([A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,})"
)


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


def _infer_environment(file_path: str) -> str:
    p = file_path.lower()
    if "prod" in p or "production" in p or "live" in p:
        return "prod"
    if "stage" in p or "staging" in p:
        return "stage"
    if "dev" in p or "local" in p or "sandbox" in p:
        return "dev"
    return "unknown"


def _infer_system(file_path: str) -> str:
    p = file_path.lower()
    if ".github/workflows" in p or "gitlab-ci" in p or "jenkins" in p:
        return "ci"
    if "k8s" in p or "kubernetes" in p or "helm" in p:
        return "k8s"
    if "terraform" in p or "pulumi" in p or "cloudformation" in p:
        return "iac"
    if "dockerfile" in p or "docker" in p or "container" in p:
        return "container"
    return "app"


def _blast_radius(kind: str, environment: str) -> str:
    base = "Medium"
    detail = "Credential exposure can widen quietly if it lives long enough."
    if kind == "private_key_block":
        base = "High"
        detail = "Key material can unlock broad access (SSH, signing, or service auth)."
    elif kind == "aws_access_key_id":
        base = "High"
        detail = "Cloud API access can pivot fast across services if permissions are wide."
    elif kind == "github_pat":
        base = "High"
        detail = "Repo access can lead to CI secrets, releases, and lateral movement."
    elif kind == "slack_token":
        base = "Medium"
        detail = "Bot tokens can expose channels, files, and automation hooks."
    elif kind == "google_api_key":
        base = "Medium"
        detail = "Impact depends on key restrictions (referrer, IP, API scope)."
    elif kind == "jwt_like":
        base = "Medium"
        detail = "JWTs can act like identity tickets if they are still valid."
    elif kind == "generic_secret_assignment":
        base = "Low"
        detail = "Could be a real secret or a false positive. Validate before rotating."

    if environment == "prod" and base != "High":
        base = "High"

    return f"{base} - {detail}"


def _extract_owner_email(text: str) -> str:
    m = OWNER_HINT.search(text)
    if not m:
        return ""
    return m.group(2).strip()


def _extract_findings_from_text(text: str, file_path: str, owner_email: str) -> List[Dict[str, Any]]:
    findings: List[Dict[str, Any]] = []
    env = _infer_environment(file_path)
    system = _infer_system(file_path)

    for pat in PATTERNS:
        if pat.name == "generic_secret_assignment":
            for m in pat.regex.finditer(text):
                key_name = m.group(1)
                value = m.group(2)
                fp = _sha256_hex(value)
                line_no = text.count("\n", 0, m.start()) + 1
                findings.append(
                    {
                        "kind": pat.name,
                        "confidence": "low",
                        "weight": pat.weight,
                        "key_name": key_name,
                        "secret_fingerprint": fp,
                        "secret_preview": _redact(value),
                        "line": line_no,
                        "owner_email": owner_email,
                        "environment": env,
                        "system": system,
                        "blast_radius": _blast_radius(pat.name, env),
                    }
                )
        elif pat.name == "private_key_block":
            for m in pat.regex.finditer(text):
                line_no = text.count("\n", 0, m.start()) + 1
                findings.append(
                    {
                        "kind": pat.name,
                        "confidence": "high",
                        "weight": pat.weight,
                        "secret_fingerprint": _sha256_hex(f"{file_path}:{line_no}:{pat.name}"),
                        "secret_preview": "-----BEGIN ... PRIVATE KEY-----",
                        "line": line_no,
                        "owner_email": owner_email,
                        "environment": env,
                        "system": system,
                        "blast_radius": _blast_radius(pat.name, env),
                    }
                )
        else:
            for m in pat.regex.finditer(text):
                value = m.group(0)
                fp = _sha256_hex(value)
                line_no = text.count("\n", 0, m.start()) + 1
                findings.append(
                    {
                        "kind": pat.name,
                        "confidence": "medium",
                        "weight": pat.weight,
                        "secret_fingerprint": fp,
                        "secret_preview": _redact(value),
                        "line": line_no,
                        "owner_email": owner_email,
                        "environment": env,
                        "system": system,
                        "blast_radius": _blast_radius(pat.name, env),
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
            "scan_id": "...",
            "scanned_root": "...",
            "files_scanned": N,
            "findings": [ { ... } ],
            "summary": { ... }
          }
    """
    root = Path(path).expanduser().resolve()
    if not root.exists():
        return {"error": f"path does not exist: {root}"}

    scan_id = time.strftime("%Y%m%d-%H%M%S", time.localtime())

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

        owner_email = _extract_owner_email(text)
        fnds = _extract_findings_from_text(text, str(fp), owner_email)

        for f in fnds:
            f2 = dict(f)
            f2["file"] = str(fp)
            f2["location_hint"] = f"line={f.get('line', 0)}"
            findings_out.append(f2)
            if len(findings_out) >= max_results:
                break

        if len(findings_out) >= max_results:
            break

    summary: Dict[str, Any] = {}
    by_kind: Dict[str, int] = {}
    by_owner: Dict[str, int] = {}
    for f in findings_out:
        by_kind[f["kind"]] = by_kind.get(f["kind"], 0) + 1
        o = f.get("owner_email") or ""
        if o:
            by_owner[o] = by_owner.get(o, 0) + 1

    summary["by_kind"] = by_kind
    summary["by_owner"] = by_owner
    summary["total_findings"] = len(findings_out)

    return {
        "scan_id": scan_id,
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
        env = f.get("environment", "unknown")
        owner = f.get("owner_email") or ""

        bump = 0
        if conf == "high":
            bump = 10
        elif conf == "medium":
            bump = 5

        context = 0
        if env == "prod":
            context += 10
        if not owner:
            context += 5

        score = min(100, base + bump + context)

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
        owner = f.get("owner_email") or ""

        if kind in {"private_key_block"}:
            action = "rotate_or_revoke_key_material"
            detail = "Treat as key compromise risk. Rotate immediately and remove from repo history."
        elif kind in {"aws_access_key_id", "github_pat", "slack_token", "google_api_key"}:
            action = "rotate_secret_and_move_to_env"
            detail = "Rotate the credential and store it in a secret manager or env var injection, not in source."
        elif kind in {"jwt_like"}:
            action = "validate_and_rotate_if_active"
            detail = "Confirm if it is active. If yes, rotate and shorten lifetime where possible."
        else:
            action = "review_and_scope_down"
            detail = "Confirm if this is a real secret. If yes, rotate and scope down permissions."

        if not owner:
            detail = detail + " Add an owner tag to the file or repo so this does not drift again."

        tasks.append(
            {
                "task_id": fp[:12],
                "action": action,
                "priority": "P0" if score >= 90 else ("P1" if score >= 75 else "P2"),
                "file": file_path,
                "secret_fingerprint": fp,
                "owner_email": owner,
                "environment": f.get("environment", "unknown"),
                "system": f.get("system", "unknown"),
                "blast_radius": f.get("blast_radius", ""),
                "rationale": f"kind={kind}, score={score}",
                "detail": detail,
            }
        )

    notes = [
        "Tasks are heuristic. Validate before rotating production credentials.",
        "This server never returns raw secrets. Only fingerprints/previews.",
    ]
    return {"tasks": tasks, "notes": notes}


@mcp.tool()
async def summarize_blast_radius(scored_findings: List[Dict[str, Any]], top_n: int = 8) -> Dict[str, Any]:
    """
    Summarize blast radius for the highest-risk findings.

    This is a short narrative tool: it helps you explain impact without showing secrets.
    """
    if not scored_findings:
        return {"summary": [], "notes": ["no findings provided"]}

    ordered = sorted(scored_findings, key=lambda x: int(x.get("risk_score", 0)), reverse=True)
    out: List[Dict[str, Any]] = []
    for f in ordered[:top_n]:
        out.append(
            {
                "risk_score": int(f.get("risk_score", 0)),
                "kind": f.get("kind", "unknown"),
                "file": f.get("file", "unknown"),
                "owner_email": f.get("owner_email") or "",
                "environment": f.get("environment", "unknown"),
                "system": f.get("system", "unknown"),
                "blast_radius": f.get("blast_radius", ""),
            }
        )

    notes = ["Impact is contextual. Treat this as a compass, not a courtroom."]
    return {"summary": out, "notes": notes}


def _suggest_scope_cuts(kind: str) -> List[str]:
    if kind == "aws_access_key_id":
        return [
            "Prefer a role with narrowly scoped IAM policy, not a long-lived access key.",
            "Restrict to required services and actions only.",
            "Add explicit resource ARNs where possible, avoid wildcard resources.",
            "Enable key rotation and reduce key lifetime if supported.",
        ]
    if kind == "github_pat":
        return [
            "Replace PAT with GitHub App or fine-grained token where possible.",
            "Limit repo access to the smallest set of repos required.",
            "Prefer read-only permissions unless writes are required.",
            "Scope tokens to a single automation purpose (CI, release, or bots), not all-in-one.",
        ]
    if kind == "google_api_key":
        return [
            "Restrict by HTTP referrer or IP address where applicable.",
            "Limit enabled APIs to only what is needed.",
            "Use separate keys per environment (dev, stage, prod).",
            "Rotate keys and remove unused keys quickly.",
        ]
    if kind == "slack_token":
        return [
            "Reduce bot scopes to the minimum set of Slack permissions.",
            "Avoid workspace-wide access unless it is essential.",
            "Rotate the token and limit its distribution to a single runtime path.",
        ]
    if kind == "private_key_block":
        return [
            "Replace raw private keys with short-lived credentials where possible.",
            "Restrict principals that can use the key material.",
            "Rotate immediately and remove from repo history.",
        ]
    if kind == "jwt_like":
        return [
            "Reduce token lifetime and enforce audience and issuer checks.",
            "Avoid storing bearer tokens in source or logs.",
            "Rotate signing keys if compromise is suspected.",
        ]
    return [
        "Scope down permissions to only what is needed.",
        "Prefer short-lived credentials over long-lived secrets.",
    ]


@mcp.tool()
async def propose_least_privilege_patch(scored_findings: List[Dict[str, Any]], rollout_window: str = "24h", max_items: int = 6) -> Dict[str, Any]:
    """
    Propose a least-privilege diff and rollout plan based on scored findings.

    This does not apply any changes. It gives you a concise plan you can hand to a human.
    """
    if not scored_findings:
        return {"plan": {}, "notes": ["no findings provided"]}

    ordered = sorted(scored_findings, key=lambda x: int(x.get("risk_score", 0)), reverse=True)
    top = ordered[:max_items]

    diffs: List[Dict[str, Any]] = []
    for f in top:
        kind = f.get("kind", "unknown")
        diffs.append(
            {
                "secret_fingerprint": f.get("secret_fingerprint", "unknown"),
                "kind": kind,
                "file": f.get("file", "unknown"),
                "environment": f.get("environment", "unknown"),
                "system": f.get("system", "unknown"),
                "owner_email": f.get("owner_email") or "",
                "risk_score": int(f.get("risk_score", 0)),
                "scope_cuts": _suggest_scope_cuts(kind),
            }
        )

    steps = [
        "Inventory: confirm which findings are real credentials and still active.",
        "Containment: rotate the highest-risk items first, starting with prod.",
        "Least privilege: apply scope reductions and remove wildcard permissions.",
        "Rollout: ship changes behind a small blast radius, then widen gradually.",
        "Verify: confirm services still function and logs show expected access only.",
        "Cleanup: delete unused credentials and close out owners and tickets.",
    ]

    rollback = [
        "Keep previous credential valid for a short overlap window where possible.",
        "If a service breaks, revert to the previous credential and re-scope with a smaller diff.",
        "Log the incident and annotate the plan with what failed and why.",
    ]

    plan = {
        "rollout_window": rollout_window,
        "diffs": diffs,
        "steps": steps,
        "rollback": rollback,
    }

    notes = [
        "This is a plan, not an enforcement engine. Use judgment before applying changes.",
        "Small changes, clean logs, steady hands.",
    ]
    return {"plan": plan, "notes": notes}

def _capability_sketch(kind: str, action: str, environment: str, system: str) -> Dict[str, Any]:
    """
    Give a bounded, non-exploitative sketch of what a credential might allow.

    This is intentionally conservative and descriptive. It does not try to execute
    anything and it does not claim certainty.
    """
    action = (action or "read").strip().lower()
    env = (environment or "unknown").strip().lower()
    sysname = (system or "unknown").strip().lower()

    guardrails = [
        "No real requests are made. This is a simulated analysis.",
        "Treat as a hypothesis until validated by logs and owners.",
        "Do not paste secrets into prompts or tickets.",
    ]

    if kind == "aws_access_key_id":
        capabilities = [
            "Potential to call AWS APIs within the key's IAM permissions.",
            "Common surfaces: S3, Secrets Manager, STS, IAM, CloudWatch.",
        ]
        risk = "High" if env == "prod" else "Medium"
        checks = [
            "Check CloudTrail for recent API calls tied to this access key.",
            "Confirm if the key is still active and which IAM principal owns it.",
            "Identify attached policies and any wildcard resources/actions.",
        ]
    elif kind == "github_pat":
        capabilities = [
            "Potential to read/write repositories depending on token scope.",
            "Possible access to CI secrets via workflows, actions, or repo settings.",
        ]
        risk = "High" if env == "prod" else "Medium"
        checks = [
            "Review token scopes and whether it is fine-grained or classic.",
            "Check audit logs for token use and unusual repo access.",
            "Confirm if token is used by CI and whether it can be replaced by a GitHub App.",
        ]
    elif kind == "google_api_key":
        capabilities = [
            "Potential to access enabled Google APIs permitted by the key.",
            "Impact depends on key restrictions (referrer, IP, API allowlist).",
        ]
        risk = "High" if env == "prod" else "Medium"
        checks = [
            "Check API key restrictions and enabled APIs.",
            "Check billing or quota anomalies around the key's usage.",
            "Split keys by environment and rotate if leaked.",
        ]
    elif kind == "slack_token":
        capabilities = [
            "Potential to call Slack Web API within bot/user token scopes.",
            "May access channels, messages, and automation hooks depending on scopes.",
        ]
        risk = "Medium" if env == "prod" else "Low"
        checks = [
            "Check Slack app scopes and which workspace it targets.",
            "Review app audit logs for suspicious calls or mass exports.",
            "Rotate token and reduce scopes where possible.",
        ]
    elif kind == "private_key_block":
        capabilities = [
            "Key material can unlock authentication flows depending on where it is used.",
            "Possible SSH, signing, or service-to-service auth.",
        ]
        risk = "High"
        checks = [
            "Identify where this key is trusted (authorized_keys, known hosts, signing keys).",
            "Rotate immediately and remove from repo history.",
            "Search for the public key or fingerprint across infra.",
        ]
    elif kind == "jwt_like":
        capabilities = [
            "Bearer token may grant access until expiry if still valid.",
            "Impact depends on audience/issuer and backend validation.",
        ]
        risk = "High" if env == "prod" else "Medium"
        checks = [
            "Confirm token validity window and intended audience.",
            "Check logs for token reuse from unusual IPs or agents.",
            "Reduce token lifetime and enforce strict audience/issuer checks.",
        ]
    else:
        capabilities = [
            "This looks like a credential or secret material, but confidence may be low.",
            "If real, it could provide access inside its unknown scope.",
        ]
        risk = "Medium" if env == "prod" else "Low"
        checks = [
            "Validate if it is a real secret and whether it is still in use.",
            "Identify owner and rotate if it maps to a live credential.",
        ]

    suggested_next = []
    if action in {"write", "delete", "admin"}:
        suggested_next.append("Assume worst-case until you confirm scopes. Reduce permissions first, then rotate.")
    else:
        suggested_next.append("Start with logs and ownership. Rotate only after you understand dependencies.")

    return {
        "inputs": {"kind": kind, "action": action, "environment": env, "system": sysname},
        "risk": risk,
        "capabilities": capabilities,
        "validation_checks": checks,
        "suggested_next_step": suggested_next,
        "guardrails": guardrails,
    }


@mcp.tool()
async def simulate_agent_action(scored_findings: List[Dict[str, Any]], secret_prefix: str, action: str = "read") -> Dict[str, Any]:
    """
    Simulate what an agent could do if it obtained a token/secret.

    Args:
        scored_findings: The scored findings list from score_findings["scored"].
        secret_prefix: The first 6-12 chars of secret_fingerprint, or the 12-char secret node suffix.
        action: A hint like "read", "write", "delete", "admin".

    Returns:
        {
          "match": { ... finding fields ... },
          "simulation": { ... capability sketch ... }
        }
    """
    if not scored_findings:
        return {"error": "no scored_findings provided"}

    prefix = (secret_prefix or "").strip().lower()
    if not prefix:
        return {"error": "secret_prefix is required"}

    match: Optional[Dict[str, Any]] = None
    for f in scored_findings:
        fp = (f.get("secret_fingerprint") or "").lower()
        if fp.startswith(prefix):
            match = f
            break

    if match is None:
        return {"error": f"no finding matched prefix={secret_prefix}"}

    kind = match.get("kind", "unknown")
    env = match.get("environment", "unknown")
    system = match.get("system", "unknown")

    sim = _capability_sketch(kind=kind, action=action, environment=env, system=system)
    return {"match": match, "simulation": sim}

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
