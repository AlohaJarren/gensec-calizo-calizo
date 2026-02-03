"""
FastAgent-based MCP agent that talks to the custom NHI MCP server.
""" 
from __future__ import annotations

import asyncio
import json
import os
from contextlib import AsyncExitStack
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


def _safe_json(obj: Any) -> str:
    return json.dumps(obj, indent=2, sort_keys=True, ensure_ascii=True)


def _result_to_text(result: Any) -> str:
    """
    MCP CallToolResult.content is typically a list of content blocks.
    We attempt to extract text. If we cannot, fall back to str().
    """
    try:
        parts: List[str] = []
        for item in getattr(result, "content", []):
            text = getattr(item, "text", None)
            if text is not None:
                parts.append(text)
            else:
                parts.append(str(item))
        if parts:
            return "\n".join(parts)
    except Exception:
        pass
    return str(result)


def _try_parse_json(text: str) -> Optional[Any]:
    try:
        return json.loads(text)
    except Exception:
        return None


@dataclass
class AgentState:
    last_scan: Optional[Dict[str, Any]] = None
    last_scored: Optional[Dict[str, Any]] = None
    last_plan: Optional[Dict[str, Any]] = None


def _memory_docker_params(memory_dir: Path) -> StdioServerParameters:
    """
    Official Docker Hub guidance shows running the memory server via:
      docker run -i --rm -v /local-directory:/local-directory mcp/memory
    We additionally set MEMORY_FILE_PATH so the graph persists in your repo folder. :contentReference[oaicite:3]{index=3}
    """
    memory_dir.mkdir(parents=True, exist_ok=True)
    host_mount = str(memory_dir.resolve())
    memory_file = "/local-directory/hw4_memory.jsonl"

    return StdioServerParameters(
        command="docker",
        args=[
            "run",
            "-i",
            "--rm",
            "-v",
            f"{host_mount}:/local-directory",
            "-e",
            f"MEMORY_FILE_PATH={memory_file}",
            "mcp/memory",
        ],
    )


def _nhi_stdio_params() -> StdioServerParameters:
    return StdioServerParameters(
        command="python",
        args=["mcp-server.py", "stdio"],
    )


async def _list_tools(session: ClientSession, label: str) -> None:
    res = await session.list_tools()
    tools = getattr(res, "tools", [])
    print(f"\n[{label}] tools loaded: {len(tools)}")
    for t in tools:
        # Each tool typically has: name, description, inputSchema
        print(f"- {t.name}")


def _build_memory_payloads(scan: Dict[str, Any], scored: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Convert scan/scored output into memory server entities and relations.

    Memory schema (entities/relations) follows:
    - entity: { "name": str, "entityType": str, "observations": [str, ...] }
    - relation: { "from": str, "to": str, "relationType": str } :contentReference[oaicite:4]{index=4}
    """
    entities: List[Dict[str, Any]] = []
    relations: List[Dict[str, Any]] = []

    scanned_root = scan.get("scanned_root", "unknown_root")
    repo_name = f"repo:{Path(scanned_root).name}"
    entities.append(
        {
            "name": repo_name,
            "entityType": "repo",
            "observations": [f"scanned_root={scanned_root}"],
        }
    )

    scored_list = scored.get("scored", [])
    for f in scored_list:
        file_path = f.get("file", "unknown_file")
        file_node = f"file:{file_path}"
        fp = f.get("secret_fingerprint", "unknown_fp")
        secret_node = f"secret:{fp[:12]}"

        # File entity
        entities.append(
            {
                "name": file_node,
                "entityType": "file",
                "observations": [f"path={file_path}"],
            }
        )

        # Secret entity (no raw secret stored)
        entities.append(
            {
                "name": secret_node,
                "entityType": "secret",
                "observations": [
                    f"kind={f.get('kind', 'unknown')}",
                    f"confidence={f.get('confidence', 'unknown')}",
                    f"risk_score={f.get('risk_score', '0')}",
                    f"preview={f.get('secret_preview', '***')}",
                ],
            }
        )

        # Relations
        relations.append({"from": secret_node, "to": file_node, "relationType": "found_in"})
        relations.append({"from": file_node, "to": repo_name, "relationType": "belongs_to"})

    return entities, relations


async def _ainput(prompt: str) -> str:
    return await asyncio.to_thread(input, prompt)


async def run() -> None:
    state = AgentState()
    hw4_dir = Path(__file__).resolve().parent
    memory_dir = hw4_dir / "memory_data"

    mem_params = _memory_docker_params(memory_dir)
    nhi_params = _nhi_stdio_params()

    async with AsyncExitStack() as stack:
        # Memory session
        mem_rw = await stack.enter_async_context(stdio_client(mem_params))
        mem_read, mem_write = mem_rw
        mem_sess = await stack.enter_async_context(ClientSession(mem_read, mem_write))
        await mem_sess.initialize()

        # NHI session
        nhi_rw = await stack.enter_async_context(stdio_client(nhi_params))
        nhi_read, nhi_write = nhi_rw
        nhi_sess = await stack.enter_async_context(ClientSession(nhi_read, nhi_write))
        await nhi_sess.initialize()

        print("HW4 MCP Agent (deterministic)")
        print("Type 'help' for commands.\n")

        while True:
            line = (await _ainput("> ")).strip()
            if not line:
                continue

            if line in {"quit", "exit"}:
                break

            if line == "help":
                print(
                    "\nCommands:\n"
                    "  tools                 - list tools from both servers\n"
                    "  scan <path>           - scan a path for NHI artifacts\n"
                    "  score                 - score last scan findings\n"
                    "  plan                  - remediation plan for last scored\n"
                    "  store                 - store last scored findings into memory graph\n"
                    "  search <query>        - search memory graph\n"
                    "  open <name> [name..]  - open nodes by exact name\n"
                    "  graph                 - read entire memory graph\n"
                    "  quit                  - exit\n"
                )
                continue

            if line == "tools":
                await _list_tools(nhi_sess, "nhi_guard")
                await _list_tools(mem_sess, "memory")
                continue

            if line.startswith("scan "):
                target = line[len("scan ") :].strip()
                res = await nhi_sess.call_tool(
                    "scan_path",
                    {"path": target, "include_hidden": False, "max_file_kb": 256, "max_results": 200},
                )
                text = _result_to_text(res)
                parsed = _try_parse_json(text)
                state.last_scan = parsed if parsed is not None else {"raw": text}

                print("\nscan result:")
                print(_safe_json(state.last_scan))
                continue

            if line == "score":
                if not state.last_scan or "findings" not in state.last_scan:
                    print("No scan in state. Run: scan <path>")
                    continue
                res = await nhi_sess.call_tool("score_findings", {"findings": state.last_scan["findings"]})
                text = _result_to_text(res)
                parsed = _try_parse_json(text)
                state.last_scored = parsed if parsed is not None else {"raw": text}

                print("\nscore result:")
                print(_safe_json(state.last_scored))
                continue

            if line == "plan":
                if not state.last_scored or "scored" not in state.last_scored:
                    print("No scored findings in state. Run: score")
                    continue
                res = await nhi_sess.call_tool("remediation_plan", {"scored_findings": state.last_scored["scored"], "top_n": 10})
                text = _result_to_text(res)
                parsed = _try_parse_json(text)
                state.last_plan = parsed if parsed is not None else {"raw": text}

                print("\nremediation plan:")
                print(_safe_json(state.last_plan))
                continue

            if line == "store":
                if not state.last_scan or not state.last_scored:
                    print("Need scan + score first.")
                    continue
                entities, relations = _build_memory_payloads(state.last_scan, state.last_scored)

                # Create entities and relations
                await mem_sess.call_tool("create_entities", {"entities": entities})
                await mem_sess.call_tool("create_relations", {"relations": relations})

                print(f"Stored: {len(entities)} entities, {len(relations)} relations")
                continue

            if line.startswith("search "):
                q = line[len("search ") :].strip()
                res = await mem_sess.call_tool("search_nodes", {"query": q})
                text = _result_to_text(res)
                parsed = _try_parse_json(text)
                print("\nsearch result:")
                print(_safe_json(parsed if parsed is not None else {"raw": text}))
                continue

            if line.startswith("open "):
                names = [n.strip() for n in line[len("open ") :].split() if n.strip()]
                res = await mem_sess.call_tool("open_nodes", {"names": names})
                text = _result_to_text(res)
                parsed = _try_parse_json(text)
                print("\nopen result:")
                print(_safe_json(parsed if parsed is not None else {"raw": text}))
                continue

            if line == "graph":
                res = await mem_sess.call_tool("read_graph", {})
                text = _result_to_text(res)
                parsed = _try_parse_json(text)
                print("\nfull graph:")
                print(_safe_json(parsed if parsed is not None else {"raw": text}))
                continue

            print("Unknown command. Type: help")


if __name__ == "__main__":
    asyncio.run(run())
