from __future__ import annotations

import asyncio
import json
import os
import sys
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


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


def _http_post_json(url: str, payload: Dict[str, Any], timeout_s: int = 60) -> Dict[str, Any]:
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout_s) as resp:
        body = resp.read().decode("utf-8", errors="replace")
    return json.loads(body)


def _extract_candidate_text(resp: Dict[str, Any]) -> str:
    try:
        return resp["candidates"][0]["content"]["parts"][0]["text"]
    except Exception:
        return json.dumps(resp, indent=2, sort_keys=True)[:2000]


def gemini_plan_tool_calls(tools: List[Dict[str, Any]], user_question: str) -> List[ToolCall]:
    """
    Ask Gemini to output a minimal sequence of MCP tool calls in JSON.

    Uses REST structured outputs:
      generationConfig.responseMimeType
      generationConfig.responseJsonSchema
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

    # Hard safety cap
    return calls[:3]


def gemini_summarize(user_question: str, tool_results: Dict[str, Any]) -> str:
    """
    Ask Gemini to summarize the results.
    """
    api_key = _get_api_key()
    model = _get_model_name()
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}"

    prompt = (
        "You are a package vulnerability mini-advisor.\n"
        "Answer the user question using ONLY the tool results provided.\n"
        "Be concise and security-focused.\n"
        "Include:\n"
        "- What SSH server/service is running (if asked)\n"
        "- What package/version is installed\n"
        "- Any CVE references found in changelog text\n"
        "- Clear limitations\n\n"
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
        lines = [ln for ln in raw_text.splitlines() if ln.strip()]
        ssh_lines = [ln for ln in lines if "ssh" in ln.lower()]
        return {"ssh_related": ssh_lines[:30], "sample": lines[:20]}

    if tool_name == "list_installed_packages_tool":
        # fastmcp often serializes dicts as JSON; try parsing first.
        try:
            obj = json.loads(raw_text)
            if isinstance(obj, dict):
                keep = {k: v for k, v in obj.items() if ("ssh" in k.lower()) or (k in ("openssl", "openssh-server"))}
                return {"filtered_packages": keep}
        except Exception:
            pass
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


async def run_agent_once(user_question: str, out_dir: Path) -> Tuple[str, Dict[str, Any]]:
    """
    One agent run:
    - Spawn MCP server via stdio
    - List tools
    - Ask Gemini for tool plan
    - Call tools
    - Ask Gemini for summary
    - Return answer + trace
    """
    from mcp import ClientSession, StdioServerParameters  # type: ignore
    from mcp.client.stdio import stdio_client  # type: ignore

    out_dir.mkdir(parents=True, exist_ok=True)

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
    }

    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            tools_resp = await session.list_tools()
            tools: List[Dict[str, Any]] = []
            for t in tools_resp.tools:
                tools.append(
                    {
                        "name": t.name,
                        "description": t.description,
                        "input_schema": t.inputSchema,
                    }
                )

            plan = gemini_plan_tool_calls(tools=tools, user_question=user_question)
            trace["tool_plan"] = [{"name": c.name, "arguments": c.arguments} for c in plan]

            tool_results: Dict[str, Any] = {}
            for call in plan:
                result = await session.call_tool(call.name, call.arguments)
                raw_text = _mcp_content_to_text(getattr(result, "content", result))
                tool_results[call.name] = _sanitize_tool_output(call.name, raw_text)

            trace["tool_results"] = tool_results

    answer = gemini_summarize(user_question=user_question, tool_results=trace["tool_results"])
    trace["finished_at_unix"] = int(time.time())
    trace["answer_preview"] = answer[:600]

    return answer, trace


def run_agent(user_question: str, out_dir: str = "hw7/out") -> Tuple[str, str, str]:
    out_path = Path(out_dir)
    answer, trace = asyncio.run(run_agent_once(user_question=user_question, out_dir=out_path))

    answer_md = out_path / "agent_answer.md"
    trace_json = out_path / "agent_trace.json"

    answer_md.write_text(answer.strip() + "\n", encoding="utf-8")
    trace_json.write_text(json.dumps(trace, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    return answer, str(answer_md), str(trace_json)
