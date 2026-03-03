from __future__ import annotations

import asyncio
import ast
import json
import os
import random
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, TextIO, Tuple


@dataclass
class ToolCall:
    name: str
    arguments: Dict[str, Any]


def _get_api_key() -> str:
    key = (os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY") or "").strip()
    if not key:
        raise RuntimeError('Missing API key. Set GEMINI_API_KEY (or GOOGLE_API_KEY).')
    return key


def _get_model_name() -> str:
    return (os.getenv("GEMINI_MODEL") or os.getenv("GOOGLE_MODEL") or "gemini-2.0-flash").strip()


def _parse_retry_delay_seconds(body_text: str) -> Optional[float]:
    """
    Some 429 responses include google.rpc.RetryInfo.retryDelay like "1s".
    Best-effort parse.
    """
    try:
        obj = json.loads(body_text)
        details = obj.get("error", {}).get("details", [])
        for d in details:
            if isinstance(d, dict) and str(d.get("@type", "")).endswith("RetryInfo"):
                rd = str(d.get("retryDelay", "")).strip()
                if rd.endswith("s"):
                    return float(rd[:-1])
    except Exception:
        pass
    return None


def _http_post_json(url: str, payload: Dict[str, Any], timeout_s: int = 60) -> Dict[str, Any]:
    """
    POST JSON with retry for HTTP 429 using truncated exponential backoff + jitter.
    """
    max_retries = int(os.getenv("GEMINI_MAX_RETRIES", "5"))
    base_delay = float(os.getenv("GEMINI_BACKOFF_BASE", "2"))
    max_delay = float(os.getenv("GEMINI_BACKOFF_MAX", "60"))

    data = json.dumps(payload).encode("utf-8")

    for attempt in range(max_retries + 1):
        req = urllib.request.Request(
            url,
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        try:
            with urllib.request.urlopen(req, timeout=timeout_s) as resp:
                body = resp.read().decode("utf-8", errors="replace")
            return json.loads(body)

        except urllib.error.HTTPError as e:
            body = ""
            try:
                body = e.read().decode("utf-8", errors="replace")
            except Exception:
                pass

            if e.code == 429:
                retry_after = e.headers.get("Retry-After")
                delay: Optional[float] = None

                if retry_after:
                    try:
                        delay = float(retry_after)
                    except Exception:
                        delay = None

                if delay is None and body:
                    delay = _parse_retry_delay_seconds(body)

                if delay is None:
                    delay = min(max_delay, base_delay * (2 ** attempt))
                    delay = delay + random.uniform(0, 1.0)

                if attempt >= max_retries:
                    raise RuntimeError(
                        "Gemini API 429 (Too Many Requests / quota). Try again later or reduce request frequency."
                    ) from e

                time.sleep(delay)
                continue

            # Non-429: surface body to make debugging easier.
            if body:
                raise RuntimeError(f"Gemini HTTP {e.code}: {body[:1200]}") from e
            raise

        except urllib.error.URLError as e:
            if attempt >= max_retries:
                raise RuntimeError(f"Gemini request failed: {e}") from e
            delay = min(max_delay, base_delay * (2 ** attempt)) + random.uniform(0, 1.0)
            time.sleep(delay)
            continue

    raise RuntimeError("Gemini request failed after retries.")


def _extract_candidate_text(resp: Dict[str, Any]) -> str:
    try:
        return resp["candidates"][0]["content"]["parts"][0]["text"]
    except Exception:
        return json.dumps(resp, indent=2, sort_keys=True)[:2000]


def _question_needs_changelog(question: str) -> bool:
    q = (question or "").lower()
    keywords = ("vulnerab", "cve", "security fix", "security", "exploit", "patch")
    return any(k in q for k in keywords)


def _safe_parse_obj(raw_text: str) -> Optional[Any]:
    s = (raw_text or "").strip()
    if not s:
        return None
    try:
        return json.loads(s)
    except Exception:
        pass
    try:
        return ast.literal_eval(s)
    except Exception:
        return None


def gemini_plan_tool_calls(tools: List[Dict[str, Any]], user_question: str) -> List[ToolCall]:
    """
    Ask Gemini (REST) to produce a minimal tool call plan as JSON.

    Uses structured output fields:
      generationConfig.responseMimeType = "application/json"
      generationConfig.responseJsonSchema = { ... }
    """
    api_key = _get_api_key()
    model = _get_model_name()
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}"

    prompt = (
        "You are a security assistant. Decide the minimal MCP tool calls needed to answer the user.\n"
        "Rules:\n"
        "- Use only the tools provided.\n"
        "- Return JSON only (no extra text).\n"
        "- Each tool call must include name and arguments.\n"
        "- If a tool takes no args, arguments must be {}.\n"
        "- Use at most 3 tool calls.\n\n"
        f"User question:\n{user_question}\n\n"
        f"Available tools (JSON):\n{json.dumps(tools, indent=2, sort_keys=True)}\n"
    )

    response_json_schema = {
        "type": "object",
        "properties": {
            "tool_calls": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "arguments": {"type": "object"},
                    },
                    "required": ["name", "arguments"],
                    "additionalProperties": False,
                },
            }
        },
        "required": ["tool_calls"],
        "additionalProperties": False,
    }

    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {
            "temperature": 0.0,
            "responseMimeType": "application/json",
            "responseJsonSchema": response_json_schema,
        },
    }

    resp = _http_post_json(url, payload)
    text = _extract_candidate_text(resp)

    try:
        plan_obj = json.loads(text)
    except Exception as e:
        raise RuntimeError(f"Failed to parse Gemini JSON plan. Raw text:\n{text}") from e

    calls: List[ToolCall] = []
    for c in plan_obj.get("tool_calls", []):
        name = str(c.get("name", "")).strip()
        args = dict(c.get("arguments") or {})
        if name:
            calls.append(ToolCall(name=name, arguments=args))

    return calls[:3]


def gemini_summarize(user_question: str, tool_results: Dict[str, Any]) -> str:
    """
    Ask Gemini to produce a concise answer based ONLY on tool results.

    Guardrails:
    - If changelog results are NOT provided, do NOT say "no CVEs found".
    - Do NOT claim installed version is vulnerable or not vulnerable based only on changelog references.
    """
    api_key = _get_api_key()
    model = _get_model_name()
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}"

    has_changelog = "get_package_changelog_tool" in (tool_results or {})

    prompt = (
        "You are a package vulnerability mini-advisor.\n"
        "Answer the user question using ONLY the tool results provided.\n"
        "Be concise and security-focused.\n\n"
        "Rules:\n"
        "- If changelog results are NOT provided, do NOT say \"no CVEs were found\".\n"
        "- If changelog results ARE provided, you may list CVE references found.\n"
        "- Do NOT claim the installed version is vulnerable or not vulnerable unless the tool results explicitly say so.\n"
        "- Changelog references provide security context, not definitive exposure.\n\n"
        f"Changelog tool present: {has_changelog}\n\n"
        f"User question:\n{user_question}\n\n"
        f"Tool results (JSON):\n{json.dumps(tool_results, indent=2, sort_keys=True)[:12000]}\n"
    )

    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {"temperature": 0.2},
    }
    resp = _http_post_json(url, payload)
    return _extract_candidate_text(resp).strip()


def _mcp_content_to_text(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        out: List[str] = []
        for part in content:
            if isinstance(part, dict) and "text" in part:
                out.append(str(part["text"]))
            elif hasattr(part, "text"):
                out.append(str(getattr(part, "text")))
            else:
                out.append(str(part))
        return "\n".join(out).strip()
    return str(content).strip()


def _sanitize_tool_output(tool_name: str, raw_text: str) -> Any:
    raw_text = (raw_text or "").strip()

    if tool_name == "list_running_services":
        obj = _safe_parse_obj(raw_text)
        if isinstance(obj, list):
            lines = [str(x) for x in obj]
        else:
            lines = [ln for ln in raw_text.splitlines() if ln.strip()]
        ssh_lines = [ln for ln in lines if "ssh" in ln.lower()]
        return {"ssh_related": ssh_lines[:30], "sample": lines[:20]}

    if tool_name == "list_installed_packages_tool":
        obj = _safe_parse_obj(raw_text)
        if isinstance(obj, dict):
            keep = {k: v for k, v in obj.items() if ("ssh" in k.lower()) or (k in ("openssl", "openssh-server"))}
            return {"filtered_packages": keep}
        return {"text_preview": raw_text[:2000]}

    if tool_name == "get_package_changelog_tool":
        import re

        cve_re = re.compile(r"\bCVE-\d{4}-\d{4,7}\b", re.IGNORECASE)
        cves: List[str] = []
        seen = set()
        for c in cve_re.findall(raw_text):
            c_up = c.upper()
            if c_up not in seen:
                seen.add(c_up)
                cves.append(c_up)

        keywords = ("cve-", "security", "vuln", "fix", "exploit")
        sec_lines: List[str] = []
        for ln in raw_text.splitlines():
            s = ln.strip()
            if not s:
                continue
            lo = s.lower()
            if any(k in lo for k in keywords):
                sec_lines.append(s)
            if len(sec_lines) >= 60:
                break

        return {"cves": cves[:60], "security_lines": sec_lines, "note": "changelog truncated"}

    return {"text_preview": raw_text[:2000]}


def _choose_candidate_package(tool_results: Dict[str, Any]) -> str:
    pkgs_obj = tool_results.get("list_installed_packages_tool", {})
    filtered = {}
    if isinstance(pkgs_obj, dict):
        fp = pkgs_obj.get("filtered_packages", {})
        if isinstance(fp, dict):
            filtered = fp

    for preferred in ("openssh-server", "dropbear", "tinysshd", "lsh-server"):
        if preferred in filtered:
            return preferred
    if filtered:
        return next(iter(filtered.keys()))
    return "openssh-server"


def _extract_findings(tool_results: Dict[str, Any]) -> Dict[str, Any]:
    service_unit = ""
    sr = tool_results.get("list_running_services", {})
    if isinstance(sr, dict):
        ssh_related = sr.get("ssh_related", [])
        if isinstance(ssh_related, list) and ssh_related:
            # Server returns unit names; keep the first ssh-like unit.
            service_unit = str(ssh_related[0]).strip()

    package = ""
    version = ""
    ip = tool_results.get("list_installed_packages_tool", {})
    if isinstance(ip, dict):
        fp = ip.get("filtered_packages", {})
        if isinstance(fp, dict):
            if "openssh-server" in fp:
                package = "openssh-server"
                version = str(fp.get("openssh-server", "")).strip()
            elif fp:
                package = str(next(iter(fp.keys())))
                version = str(fp.get(package, "")).strip()

    cves: List[str] = []
    ch = tool_results.get("get_package_changelog_tool", {})
    if isinstance(ch, dict):
        cves_obj = ch.get("cves", [])
        if isinstance(cves_obj, list):
            cves = [str(x) for x in cves_obj]

    return {"service_unit": service_unit, "package": package, "version": version, "cves": cves}


def _remediation_plan(service_unit: str, package: str) -> List[str]:
    cmds = ["sudo apt-get update"]
    if package:
        cmds.append(f"sudo apt-get install --only-upgrade {package}")
    if service_unit:
        cmds.append(f"sudo systemctl restart {service_unit}")
    return cmds


def _format_agent_answer_md(question: str, llm_answer: str, trace: Dict[str, Any]) -> str:
    tool_results = trace.get("tool_results", {}) if isinstance(trace.get("tool_results", {}), dict) else {}
    findings = _extract_findings(tool_results)

    service_unit = findings.get("service_unit", "") or ""
    package = findings.get("package", "") or ""
    version = findings.get("version", "") or ""
    cves = findings.get("cves", []) if isinstance(findings.get("cves", []), list) else []

    started = trace.get("started_at_unix", "")
    finished = trace.get("finished_at_unix", "")
    mcp_log = trace.get("mcp_server_stderr_log", "hw7/out/mcp_server_stderr.log")
    err = trace.get("error", None)

    lines: List[str] = []
    lines.append("# HW7 Agent Answer")
    lines.append("")
    lines.append(f"- Started (unix): {started}")
    lines.append(f"- Finished (unix): {finished}")
    lines.append("")
    lines.append("## Question")
    lines.append("")
    lines.append(question.strip())
    lines.append("")

    if err:
        lines.append("## Error")
        lines.append("")
        lines.append(str(err).strip())
        lines.append("")

    lines.append("## Findings (from tool results)")
    lines.append("")
    lines.append(f"- Service: {service_unit or 'unknown'}")
    pkg_line = package if package else "unknown"
    if version:
        pkg_line += f" {version}"
    lines.append(f"- Package: {pkg_line}")

    if cves:
        lines.append(f"- CVE references (from changelog): {', '.join(cves)}")
    else:
        if "get_package_changelog_tool" in tool_results:
            lines.append("- CVE references (from changelog): none found in retrieved snippet")
        else:
            lines.append("- CVE references (from changelog): not retrieved for this question")
    lines.append("")

    lines.append("## Remediation command plan (not executed)")
    lines.append("")
    plan = _remediation_plan(service_unit=service_unit, package=package)
    lines.append("```")
    lines.extend(plan)
    lines.append("```")
    lines.append("")

    lines.append("## LLM Answer (generated)")
    lines.append("")
    lines.append(llm_answer.strip())
    lines.append("")

    lines.append("## Evidence artifacts")
    lines.append("")
    lines.append("- Trace JSON: hw7/out/agent_trace.json")
    lines.append(f"- MCP server stderr log: {mcp_log}")
    lines.append("")

    lines.append("## Limitations")
    lines.append("")
    lines.append("- Changelog CVE references provide security context, but do not prove the installed version is vulnerable.")
    lines.append("- Tool outputs are truncated for safety and readability.")
    lines.append("- This workflow does not apply updates or restart services automatically.")
    lines.append("")

    return "\n".join(lines)


def _fallback_summary(question: str, tool_results: Dict[str, Any], error_text: str) -> str:
    findings = _extract_findings(tool_results)
    service_unit = findings.get("service_unit", "") or "unknown"
    package = findings.get("package", "") or "unknown"
    version = findings.get("version", "") or ""
    cves = findings.get("cves", []) if isinstance(findings.get("cves", []), list) else []

    parts: List[str] = []
    parts.append(f"Agent limitation: {error_text.strip()}")
    parts.append("")
    parts.append(f"Service: {service_unit}")
    if version:
        parts.append(f"Package: {package} {version}")
    else:
        parts.append(f"Package: {package}")
    if _question_needs_changelog(question):
        if cves:
            parts.append("CVE references (from changelog snippet): " + ", ".join(cves))
        else:
            parts.append("CVE references: unavailable (changelog not retrieved or empty).")
    return "\n".join(parts).strip()


async def run_agent_once(user_question: str, out_dir: Path) -> Tuple[str, Dict[str, Any]]:
    """
    One agent run:
    - Spawn MCP server via stdio
    - List tools
    - Ask Gemini for tool plan
    - Execute tool calls (+ guardrail)
    - Ask Gemini for summary
    - Return answer + trace

    Server logs in stdio mode go to stderr; we capture them to a file.
    """
    from mcp import ClientSession, StdioServerParameters  # type: ignore
    from mcp.client.stdio import stdio_client  # type: ignore

    out_dir.mkdir(parents=True, exist_ok=True)
    mcp_err_path = out_dir / "mcp_server_stderr.log"

    server_script = (Path(__file__).resolve().parent / "app.py").as_posix()
    server_params = StdioServerParameters(
        command=sys.executable,
        args=[server_script, "mcp-server"],
        env=None,
    )

    trace: Dict[str, Any] = {
        "question": user_question,
        "started_at_unix": int(time.time()),
        "tool_plan": [],
        "tool_results": {},
        "guardrails": {},
        "error": None,
    }

    needs_changelog = _question_needs_changelog(user_question)
    trace["guardrails"]["needs_changelog"] = needs_changelog

    tool_results: Dict[str, Any] = {}

    err_fh: TextIO
    with mcp_err_path.open("w", encoding="utf-8") as err_fh:
        try:
            try:
                stdio_ctx = stdio_client(server_params, errlog=err_fh)
            except TypeError:
                stdio_ctx = stdio_client(server_params)

            async with stdio_ctx as (read, write):
                async with ClientSession(read, write) as session:
                    await session.initialize()

                    tools_resp = await session.list_tools()
                    tools: List[Dict[str, Any]] = []
                    tool_names = set()
                    for t in tools_resp.tools:
                        tools.append({"name": t.name, "description": t.description, "input_schema": t.inputSchema})
                        tool_names.add(t.name)

                    # Plan tool calls via Gemini (may rate-limit).
                    try:
                        plan = gemini_plan_tool_calls(tools=tools, user_question=user_question)
                    except Exception as e:
                        trace["error"] = f"Planning failed: {e}"
                        trace["tool_results"] = {}
                        trace["finished_at_unix"] = int(time.time())
                        trace["mcp_server_stderr_log"] = str(mcp_err_path)
                        return _fallback_summary(user_question, {}, str(e)), trace

                    # Ensure baseline tools exist in plan (stable answers).
                    def _ensure(call_name: str) -> None:
                        if call_name in tool_names and all(c.name != call_name for c in plan):
                            plan.insert(0, ToolCall(name=call_name, arguments={}))

                    _ensure("list_installed_packages_tool")
                    _ensure("list_running_services")

                    trace["tool_plan"] = [{"name": c.name, "arguments": c.arguments} for c in plan]

                    # Execute planned tool calls.
                    for call in plan:
                        result = await session.call_tool(call.name, call.arguments)
                        raw_text = _mcp_content_to_text(getattr(result, "content", result))
                        tool_results[call.name] = _sanitize_tool_output(call.name, raw_text)

                    # Guardrail: if vuln question, force changelog lookup even if model forgot.
                    if needs_changelog and "get_package_changelog_tool" in tool_names and "get_package_changelog_tool" not in tool_results:
                        pkg = _choose_candidate_package(tool_results)
                        trace["guardrails"]["forced_changelog_package"] = pkg
                        result = await session.call_tool("get_package_changelog_tool", {"package_name": pkg})
                        raw_text = _mcp_content_to_text(getattr(result, "content", result))
                        tool_results["get_package_changelog_tool"] = _sanitize_tool_output("get_package_changelog_tool", raw_text)

        except Exception as e:
            trace["error"] = f"MCP execution failed: {e}"
            trace["tool_results"] = tool_results
            trace["finished_at_unix"] = int(time.time())
            trace["mcp_server_stderr_log"] = str(mcp_err_path)
            return _fallback_summary(user_question, tool_results, str(e)), trace

    # Summarize with Gemini; if rate-limited, fall back to deterministic summary.
    try:
        answer = gemini_summarize(user_question=user_question, tool_results=tool_results)
    except Exception as e:
        trace["error"] = f"Summarization failed: {e}"
        answer = _fallback_summary(user_question, tool_results, str(e))

    trace["tool_results"] = tool_results
    trace["finished_at_unix"] = int(time.time())
    trace["answer_preview"] = answer[:600]
    trace["mcp_server_stderr_log"] = str(mcp_err_path)

    return answer, trace


def run_agent(user_question: str, out_dir: str = "hw7/out") -> Tuple[str, str, str]:
    out_path = Path(out_dir)
    answer, trace = asyncio.run(run_agent_once(user_question=user_question, out_dir=out_path))

    answer_md = out_path / "agent_answer.md"
    trace_json = out_path / "agent_trace.json"

    answer_md.write_text(_format_agent_answer_md(user_question, answer, trace), encoding="utf-8")
    trace_json.write_text(json.dumps(trace, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    return answer, str(answer_md), str(trace_json)