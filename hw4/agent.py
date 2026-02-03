"""
FastAgent-based MCP agent that talks to the custom NHI MCP server.
""" 
from __future__ import annotations

import asyncio
import json
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
    last_blast: Optional[Dict[str, Any]] = None
    last_plan: Optional[Dict[str, Any]] = None
    last_patch: Optional[Dict[str, Any]] = None
    last_sim: Optional[Dict[str, Any]] = None

def _memory_docker_params(memory_dir: Path) -> StdioServerParameters:
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
        print(f"- {t.name}")


def _relpath_or_raw(scanned_root: str, file_path: str) -> str:
    try:
        root = Path(scanned_root).resolve()
        fp = Path(file_path).resolve()
        return str(fp.relative_to(root))
    except Exception:
        return file_path


def _build_memory_payloads(scan: Dict[str, Any], scored: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    entities: List[Dict[str, Any]] = []
    relations: List[Dict[str, Any]] = []

    scanned_root = scan.get("scanned_root", "unknown_root")
    scan_id = scan.get("scan_id", "unknown_scan")
    repo_name = f"repo:{Path(scanned_root).name}"
    scan_name = f"scan:{scan_id}"

    entities.append(
        {
            "name": repo_name,
            "entityType": "repo",
            "observations": [f"scanned_root={scanned_root}"],
        }
    )
    entities.append(
        {
            "name": scan_name,
            "entityType": "scan",
            "observations": [
                f"scan_id={scan_id}",
                f"files_scanned={scan.get('files_scanned', 0)}",
                f"total_findings={scan.get('summary', {}).get('total_findings', 0)}",
            ],
        }
    )
    relations.append({"from": scan_name, "to": repo_name, "relationType": "scanned"})

    scored_list = scored.get("scored", [])
    for f in scored_list:
        file_raw = f.get("file", "unknown_file")
        file_rel = _relpath_or_raw(scanned_root, file_raw)
        file_node = f"file:{file_rel}"
        fp = f.get("secret_fingerprint", "unknown_fp")
        secret_node = f"secret:{fp[:12]}"

        owner_email = (f.get("owner_email") or "").strip()
        owner_node = f"owner:{owner_email}" if owner_email else ""

        env = (f.get("environment") or "unknown").strip()
        env_node = f"env:{env}"

        system = (f.get("system") or "unknown").strip()
        system_node = f"system:{system}"

        entities.append(
            {
                "name": file_node,
                "entityType": "file",
                "observations": [f"path={file_rel}"],
            }
        )
        entities.append(
            {
                "name": secret_node,
                "entityType": "secret",
                "observations": [
                    f"kind={f.get('kind', 'unknown')}",
                    f"confidence={f.get('confidence', 'unknown')}",
                    f"risk_score={f.get('risk_score', '0')}",
                    f"preview={f.get('secret_preview', '***')}",
                    f"location_hint={f.get('location_hint', '')}",
                    f"blast_radius={f.get('blast_radius', '')}",
                ],
            }
        )

        entities.append(
            {
                "name": env_node,
                "entityType": "environment",
                "observations": [f"name={env}"],
            }
        )
        entities.append(
            {
                "name": system_node,
                "entityType": "system",
                "observations": [f"name={system}"],
            }
        )
        if owner_node:
            entities.append(
                {
                    "name": owner_node,
                    "entityType": "owner",
                    "observations": [f"email={owner_email}"],
                }
            )

        relations.append({"from": secret_node, "to": file_node, "relationType": "found_in"})
        relations.append({"from": file_node, "to": repo_name, "relationType": "belongs_to"})
        relations.append({"from": scan_name, "to": secret_node, "relationType": "reported"})

        relations.append({"from": secret_node, "to": env_node, "relationType": "scoped_to"})
        relations.append({"from": secret_node, "to": system_node, "relationType": "touches"})

        if owner_node:
            relations.append({"from": secret_node, "to": owner_node, "relationType": "owned_by"})

    return entities, relations


def _build_patch_payloads(scan: Dict[str, Any], scored: Dict[str, Any], patch: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    entities: List[Dict[str, Any]] = []
    relations: List[Dict[str, Any]] = []

    scan_id = scan.get("scan_id", "unknown_scan")
    scan_name = f"scan:{scan_id}"
    plan_name = f"plan:{scan_id}"

    plan = patch.get("plan", {})
    rollout_window = plan.get("rollout_window", "unknown")
    diffs = plan.get("diffs", [])

    entities.append(
        {
            "name": plan_name,
            "entityType": "plan",
            "observations": [
                f"rollout_window={rollout_window}",
                f"diff_count={len(diffs)}",
            ],
        }
    )
    relations.append({"from": plan_name, "to": scan_name, "relationType": "derived_from"})

    # Link plan to top secrets it references.
    for d in diffs:
        fp = d.get("secret_fingerprint", "unknown_fp")
        secret_node = f"secret:{fp[:12]}"
        relations.append({"from": plan_name, "to": secret_node, "relationType": "covers"})

    # Store a compact narrative so you can open the plan in memory.
    steps = plan.get("steps", [])
    rollback = plan.get("rollback", [])

    entities.append(
        {
            "name": f"plantext:{scan_id}",
            "entityType": "note",
            "observations": [
                "least_privilege_patch",
                f"steps={'; '.join(steps)}",
                f"rollback={'; '.join(rollback)}",
            ],
        }
    )
    relations.append({"from": f"plantext:{scan_id}", "to": plan_name, "relationType": "describes"})

    return entities, relations

def _build_sim_payloads(scan: Dict[str, Any], sim: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    entities: List[Dict[str, Any]] = []
    relations: List[Dict[str, Any]] = []

    scan_id = scan.get("scan_id", "unknown_scan")
    scan_name = f"scan:{scan_id}"

    match = sim.get("match", {})
    simulation = sim.get("simulation", {})

    fp = match.get("secret_fingerprint", "unknown_fp")
    secret_node = f"secret:{fp[:12]}"

    note_name = f"sim:{scan_id}:{fp[:12]}"

    entities.append(
        {
            "name": note_name,
            "entityType": "note",
            "observations": [
                "simulation",
                f"risk={simulation.get('risk', '')}",
                f"capabilities={'; '.join(simulation.get('capabilities', []))}",
                f"checks={'; '.join(simulation.get('validation_checks', []))}",
                f"next={'; '.join(simulation.get('suggested_next_step', []))}",
            ],
        }
    )

    relations.append({"from": note_name, "to": scan_name, "relationType": "derived_from"})
    relations.append({"from": note_name, "to": secret_node, "relationType": "about"})

    return entities, relations

async def _ainput(prompt: str) -> str:
    return await asyncio.to_thread(input, prompt)


def _print_blast_summary(blast: Dict[str, Any]) -> None:
    summary = blast.get("summary", [])
    notes = blast.get("notes", [])
    print("\nblast radius summary:")
    print(_safe_json({"summary": summary, "notes": notes}))

def _pick_latest_scan_id(graph: Dict[str, Any]) -> Optional[str]:
    """
    Try to find the latest scan:<timestamp> by lexicographic order.
    Assumes scan ids are like YYYYMMDD-HHMMSS which sort correctly as strings.
    """
    entities = graph.get("entities", []) or []
    scan_ids: List[str] = []
    for e in entities:
        name = (e.get("name") or "").strip()
        if name.startswith("scan:"):
            scan_ids.append(name[len("scan:") :])
    if not scan_ids:
        return None
    return sorted(scan_ids)[-1]


def _entity_map(graph: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    m: Dict[str, Dict[str, Any]] = {}
    for e in graph.get("entities", []) or []:
        name = e.get("name")
        if name:
            m[name] = e
    return m


def _relations_from(graph: Dict[str, Any], from_name: str) -> List[Dict[str, Any]]:
    rels = graph.get("relations", []) or []
    return [r for r in rels if (r.get("from") or "") == from_name]


def _relations_to(graph: Dict[str, Any], to_name: str) -> List[Dict[str, Any]]:
    rels = graph.get("relations", []) or []
    return [r for r in rels if (r.get("to") or "") == to_name]


def _obs_get(entity: Dict[str, Any], key_prefix: str) -> str:
    """
    Pull a single observation string that starts with key_prefix.
    Example: key_prefix="risk_score=" returns "95" if present.
    """
    obs = entity.get("observations", []) or []
    for o in obs:
        if isinstance(o, str) and o.startswith(key_prefix):
            return o[len(key_prefix) :].strip()
    return ""


def _render_report(graph: Dict[str, Any], scan_id: str, max_secrets: int = 5) -> str:
    emap = _entity_map(graph)

    scan_name = f"scan:{scan_id}"
    plan_name = f"plan:{scan_id}"

    # Secrets reported by scan.
    secrets: List[str] = []
    for r in _relations_from(graph, scan_name):
        if r.get("relationType") == "reported":
            to = r.get("to") or ""
            if to.startswith("secret:"):
                secrets.append(to)

    # Rank secrets by risk_score observation if present.
    def secret_score(secret_node: str) -> int:
        e = emap.get(secret_node, {})
        v = _obs_get(e, "risk_score=")
        try:
            return int(v)
        except Exception:
            return -1

    secrets_sorted = sorted(secrets, key=secret_score, reverse=True)

    lines: List[str] = []
    lines.append(f"report for {scan_name}")
    lines.append("-" * 60)

    # Scan summary.
    scan_ent = emap.get(scan_name, {})
    lines.append("scan:")
    lines.append(f"- files_scanned: {_obs_get(scan_ent, 'files_scanned=') or 'unknown'}")
    lines.append(f"- total_findings: {_obs_get(scan_ent, 'total_findings=') or 'unknown'}")
    lines.append("")

    # Top secrets.
    lines.append(f"top secrets (by risk_score, up to {max_secrets}):")
    if not secrets_sorted:
        lines.append("- none stored yet (run: store)")
    else:
        for s in secrets_sorted[:max_secrets]:
            e = emap.get(s, {})
            kind = _obs_get(e, "kind=") or "unknown"
            score = _obs_get(e, "risk_score=") or "?"
            preview = _obs_get(e, "preview=") or "***"
            blast = _obs_get(e, "blast_radius=") or ""
            lines.append(f"- {s} kind={kind} risk_score={score} preview={preview}")
            if blast:
                lines.append(f"  blast_radius: {blast}")
    lines.append("")

    # Plan presence.
    if plan_name in emap:
        plan_ent = emap.get(plan_name, {})
        rollout = _obs_get(plan_ent, "rollout_window=") or "unknown"
        diff_count = _obs_get(plan_ent, "diff_count=") or "unknown"
        lines.append("least-privilege plan:")
        lines.append(f"- node: {plan_name}")
        lines.append(f"- rollout_window: {rollout}")
        lines.append(f"- diff_count: {diff_count}")
    else:
        lines.append("least-privilege plan:")
        lines.append("- none stored yet (run: patch then storepatch)")
    lines.append("")

    # Sim notes about secrets for this scan.
    sim_notes: List[str] = []
    for e_name, e in emap.items():
        if e_name.startswith(f"sim:{scan_id}:"):
            sim_notes.append(e_name)

    lines.append("simulations:")
    if not sim_notes:
        lines.append("- none stored yet (run: simulate then storesim)")
    else:
        for n in sorted(sim_notes):
            ent = emap.get(n, {})
            risk = ""
            obs = ent.get("observations", []) or []
            for o in obs:
                if isinstance(o, str) and o.startswith("risk="):
                    risk = o
                    break
            lines.append(f"- {n} {risk}".strip())
    lines.append("-" * 60)

    return "\n".join(lines)

async def run() -> None:
    state = AgentState()
    hw4_dir = Path(__file__).resolve().parent
    memory_dir = hw4_dir / "memory_data"

    mem_params = _memory_docker_params(memory_dir)
    nhi_params = _nhi_stdio_params()

    async with AsyncExitStack() as stack:
        mem_rw = await stack.enter_async_context(stdio_client(mem_params))
        mem_read, mem_write = mem_rw
        mem_sess = await stack.enter_async_context(ClientSession(mem_read, mem_write))
        await mem_sess.initialize()

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
                    "  blast                 - summarize blast radius for last scored\n"
                    "  plan                  - remediation plan for last scored\n"
                    "  patch                 - least-privilege diff and rollout plan\n"
                    "  store                 - store last scored findings into memory graph\n"
                    "  storepatch            - store last patch plan into memory graph\n"
                    "  search <query>        - search memory graph\n"
                    "  report                - print a one-screen summary from memory\n"
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

            if line == "blast":
                if not state.last_scored or "scored" not in state.last_scored:
                    print("No scored findings in state. Run: score")
                    continue
                res = await nhi_sess.call_tool("summarize_blast_radius", {"scored_findings": state.last_scored["scored"], "top_n": 8})
                text = _result_to_text(res)
                parsed = _try_parse_json(text)
                state.last_blast = parsed if parsed is not None else {"raw": text}
                _print_blast_summary(state.last_blast if isinstance(state.last_blast, dict) else {})
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

            if line == "patch":
                if not state.last_scored or "scored" not in state.last_scored:
                    print("No scored findings in state. Run: score")
                    continue
                res = await nhi_sess.call_tool(
                    "propose_least_privilege_patch",
                    {"scored_findings": state.last_scored["scored"], "rollout_window": "24h", "max_items": 6},
                )
                text = _result_to_text(res)
                parsed = _try_parse_json(text)
                state.last_patch = parsed if parsed is not None else {"raw": text}

                print("\nleast-privilege patch:")
                print(_safe_json(state.last_patch))
                continue

            if line.startswith("simulate "):
                if not state.last_scored or "scored" not in state.last_scored:
                    print("No scored findings in state. Run: score")
                    continue

                parts = [p for p in line.split() if p.strip()]
                if len(parts) < 2:
                    print("Usage: simulate <secret_prefix> [action]")
                    continue

                secret_prefix = parts[1]
                action = parts[2] if len(parts) >= 3 else "read"

                res = await nhi_sess.call_tool(
                    "simulate_agent_action",
                    {"scored_findings": state.last_scored["scored"], "secret_prefix": secret_prefix, "action": action},
                )
                text = _result_to_text(res)
                parsed = _try_parse_json(text)
                state.last_sim = parsed if parsed is not None else {"raw": text}

                print("\nsimulation:")
                print(_safe_json(state.last_sim))
                continue

            if line == "store":
                if not state.last_scan or not state.last_scored:
                    print("Need scan + score first.")
                    continue
                entities, relations = _build_memory_payloads(state.last_scan, state.last_scored)

                await mem_sess.call_tool("create_entities", {"entities": entities})
                await mem_sess.call_tool("create_relations", {"relations": relations})

                print(f"Stored: {len(entities)} entities, {len(relations)} relations")
                continue

            if line == "storepatch":
                if not state.last_scan or not state.last_scored or not state.last_patch:
                    print("Need scan + score + patch first.")
                    continue
                e2, r2 = _build_patch_payloads(state.last_scan, state.last_scored, state.last_patch)
                await mem_sess.call_tool("create_entities", {"entities": e2})
                await mem_sess.call_tool("create_relations", {"relations": r2})
                print(f"Stored patch: {len(e2)} entities, {len(r2)} relations")
                continue

            if line == "storesim":
                if not state.last_scan or not state.last_sim:
                    print("Need scan + simulate first.")
                    continue
                e3, r3 = _build_sim_payloads(state.last_scan, state.last_sim)
                await mem_sess.call_tool("create_entities", {"entities": e3})
                await mem_sess.call_tool("create_relations", {"relations": r3})
                print(f"Stored simulation: {len(e3)} entities, {len(r3)} relations")
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

            if line == "report":
                res = await mem_sess.call_tool("read_graph", {})
                text = _result_to_text(res)
                parsed = _try_parse_json(text)
                if not isinstance(parsed, dict):
                    print("Could not read graph as JSON.")
                    continue

                scan_id = _pick_latest_scan_id(parsed)
                if not scan_id:
                    print("No scan nodes found in memory yet. Run: store (after scan + score).")
                    continue

                print("")
                print(_render_report(parsed, scan_id=scan_id, max_secrets=5))
                continue

            print("Unknown command. Type: help")


if __name__ == "__main__":
    asyncio.run(run())
