from __future__ import annotations

import argparse
import sys
from pathlib import Path


def _add_repo_root_to_syspath() -> None:
    """
    Ensure the repository root is on sys.path so we can import sibling folders.
    """
    repo_root = Path(__file__).resolve().parent.parent
    repo_root_str = str(repo_root)
    if repo_root_str not in sys.path:
        sys.path.insert(0, repo_root_str)


def _print_trace_from_message(msg) -> None:
    """
    Print a compact, human-readable trace line for tool activity.
    """
    cls = msg.__class__.__name__
    tool_calls = getattr(msg, "tool_calls", None)

    if tool_calls:
        names = []
        for tc in tool_calls:
            name = tc.get("name", "unknown") if isinstance(tc, dict) else "unknown"
            names.append(name)
        print(f"Calling tools: {names}")
        return

    if cls == "ToolMessage":
        content = getattr(msg, "content", "") or ""
        snippet = content.strip().replace("\n", " ")
        if len(snippet) > 260:
            snippet = snippet[:260] + "..."
        if snippet:
            print(f"Tool result: {snippet}")


def _run_brave_agent(query: str, count: int, max_tool_calls: int) -> str:
    """
    Run a minimal agent powered by Brave Search.
    """
    _add_repo_root_to_syspath()

    try:
        from hw2.app import get_llm, require_env  # type: ignore
    except Exception as e:
        print("ERROR: Failed to import hw2 helpers (get_llm/require_env).", file=sys.stderr)
        print(f"Import error: {e}", file=sys.stderr)
        return ""

    require_env("GOOGLE_API_KEY")
    brave_key = require_env("BRAVE_SEARCH_API_KEY")

    try:
        from langchain_community.tools import BraveSearch  # type: ignore
        from langchain.agents import create_agent  # type: ignore
    except Exception as e:
        print("ERROR: Missing LangChain dependencies for the agent/tool.", file=sys.stderr)
        print("Try: uv add langchain langchain-community", file=sys.stderr)
        print(f"Import error: {e}", file=sys.stderr)
        return ""

    llm = get_llm()
    search = BraveSearch.from_api_key(api_key=brave_key, search_kwargs={"count": count})
    tools = [search]

    instructions = (
        "You are a helpful assistant that can use Brave Search when you need fresh info. "
        f"Use at most {max_tool_calls} tool calls. "
        "If you are not confident, say what you do not know."
    )

    agent = create_agent(
        llm,
        tools=tools,
        system_prompt=instructions,
    )

    # Stream the agent state so the terminal shows tool activity while it runs.
    final_text = ""
    try:
        for chunk in agent.stream(
            {"messages": [{"role": "user", "content": query}]},
            stream_mode="values",
        ):
            msgs = chunk.get("messages", []) if isinstance(chunk, dict) else []
            if not msgs:
                continue

            last = msgs[-1]
            _print_trace_from_message(last)

            content = getattr(last, "content", None)
            if isinstance(content, str) and content.strip():
                final_text = content.strip()
    except Exception as e:
        print("ERROR: Agent execution failed.", file=sys.stderr)
        print(f"Runtime error: {e}", file=sys.stderr)
        return ""

    return final_text.strip()


def _agent_cli(argv: list[str]) -> int:
    """HW3 agent mode - Brave Search first, then we layer in more tools."""
    parser = argparse.ArgumentParser(
        prog="hw3 agent",
        description="HW3 agent mode (create_agent + Brave Search).",
    )
    parser.add_argument("query", nargs="?", help="One-shot query. If omitted, runs interactive mode.")
    parser.add_argument("--count", type=int, default=3, help="Number of Brave results to fetch.")
    parser.add_argument("--max-tool-calls", type=int, default=6, help="Upper bound for tool calls.")
    args = parser.parse_args(argv)

    if args.query:
        out = _run_brave_agent(args.query, count=args.count, max_tool_calls=args.max_tool_calls)
        if out:
            print(out)
        return 0

    print('Entering agent mode. Type "exit" to quit.\n')
    while True:
        try:
            q = input("agent> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not q:
            continue
        if q.lower() in ("exit", "quit"):
            break

        out = _run_brave_agent(q, count=args.count, max_tool_calls=args.max_tool_calls)
        if out:
            print(out)
            print()

    return 0


def main() -> int:
    """
    Delegate execution to HW2's CLI.
    """
    _add_repo_root_to_syspath()

    if len(sys.argv) >= 2 and sys.argv[1] == "agent":
        return int(_agent_cli(sys.argv[2:]))

    try:
        from hw2.app import main as hw2_main  # type: ignore
    except Exception as e:
        print("ERROR: Failed to import hw2.app. Make sure hw2/app.py exists.", file=sys.stderr)
        print(f"Import error: {e}", file=sys.stderr)
        return 1

    return int(hw2_main())


if __name__ == "__main__":
    raise SystemExit(main())
