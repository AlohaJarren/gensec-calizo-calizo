"""
CS 575 - HW3 Agent App

A small LangChain agent that blends:
- Brave Search for fresh web context
- A local Project Gutenberg RAG store (reusing hw2/app.py)
- A guarded Terminal tool with human approval
- The Gutenberg vector DB persists under hw3/.chromadb, so I can ingest once andask follow-up questions across separate runs.

Commands:
- agent (one-shot or interactive)
- all hw2 commands (passed through when not in agent mode)
### Credentials and settings are supplied via environment variables.
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
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
    """Print a compact trace line for tool activity."""
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


def _require_env_soft(var_name: str) -> str:
    """
    Return an env var if set, else return an empty string.

    In agent mode, a missing key is a limitation to surface, not a hard exit.
    """
    return os.getenv(var_name, "").strip()


def _get_hw3_rag_config():
    """Reuse HW2's vectorstore builder, but store HW3 data under hw3/.chromadb."""
    _add_repo_root_to_syspath()
    from hw2.app import AppConfig, get_config  # type: ignore

    base = get_config()
    persist_dir = Path(__file__).resolve().parent.joinpath(".chromadb")

    return AppConfig(
        persist_dir=persist_dir,
        collection_name="hw3_agent_rag",
        embedding_model=base.embedding_model,
    )


def _get_tool_decorator():
    """Import the @tool decorator across LangChain versions."""
    try:
        from langchain_core.tools import tool  # type: ignore
        return tool
    except Exception:
        from langchain.tools import tool  # type: ignore
        return tool


def _is_command_safe(cmd: str) -> bool:
    """
    A small guardrail: only allow read-style commands.
    """
    s = (cmd or "").strip().lower()
    if not s:
        return False

    banned = [
        "rm ",
        "rm\t",
        "sudo",
        "chmod",
        "chown",
        "mv ",
        "dd ",
        "mkfs",
        "shutdown",
        "reboot",
        "kill ",
        "pkill",
        "apt ",
        "apt-get",
        "yum ",
        "dnf ",
        "pacman",
        "service ",
        "systemctl ",
        ">",
        ">>",
        "| sh",
        "| bash",
        "curl ",
        "wget ",
    ]

    for b in banned:
        if b in s:
            return False

    allowed_prefixes = (
        "ls",
        "pwd",
        "whoami",
        "id",
        "date",
        "uname",
        "cat",
        "head",
        "tail",
        "grep",
        "find",
        "du",
        "df",
        "stat",
        "wc",
        "echo",
        "tree",
        "sed",
        "awk",
        "cut",
        "sort",
        "uniq",
        "tr",
    )

    parts = s.split()
    if not parts:
        return False

    return parts[0] in allowed_prefixes


def _run_shell_command(cmd: str) -> str:
    """Execute a shell command and return combined stdout/stderr."""
    try:
        p = subprocess.run(
            cmd,
            shell=True,
            text=True,
            capture_output=True,
        )
        out = (p.stdout or "") + (p.stderr or "")
        out = out.strip()
        if not out:
            return "(no output)"
        return out
    except Exception as e:
        return f"Command execution failed: {e}"


def _build_terminal_tool():
    """Terminal tool with a human approval gate."""
    tool = _get_tool_decorator()

    @tool("terminal")
    def terminal(command: str) -> str:
        """Execute a read-only shell command after human approval."""
        cmd = (command or "").strip()

        if not _is_command_safe(cmd):
            return (
                "Refused: command is outside my read-only boundary. "
                "Try a safer alternative like ls, find, grep, du, or cat."
            )

        print("\nProposed terminal command:")
        print(cmd)
        ans = input("\nRun it? (y/N): ").strip().lower()
        if ans != "y":
            return "User declined to run the command."

        return _run_shell_command(cmd)

    return terminal


def _build_gutenberg_tools():
    """Wrap the HW2 RAG logic as tools so the agent can drive it."""
    _add_repo_root_to_syspath()

    from hw2.app import (  # type: ignore
        build_rag_answer,
        build_vectorstore,
        get_db_sources,
        ingest_documents,
        load_gutenberg_docs,
        retrieve_docs,
    )

    tool = _get_tool_decorator()

    @tool("gutenberg_ingest")
    def gutenberg_ingest(url: str, chunk_size: int = 2000, chunk_overlap: int = 200) -> str:
        """
        Ingest a Project Gutenberg plain-text URL into the local vector database.
        """
        key = _require_env_soft("GOOGLE_API_KEY")
        if not key:
            return "Missing GOOGLE_API_KEY. Set it in your shell before using RAG."

        try:
            config = _get_hw3_rag_config()
            vectorstore = build_vectorstore(config)
            docs = load_gutenberg_docs(url)
            if not docs:
                return "No documents returned from GutenbergLoader."
            n = ingest_documents(vectorstore, docs, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
            return f"Inserted {n} chunks from: {url}"
        except Exception as e:
            return f"RAG ingest failed: {e}"

    @tool("gutenberg_ask")
    def gutenberg_ask(question: str, k: int = 4) -> str:
        """Ask a question using the local Gutenberg vector database."""
        key = _require_env_soft("GOOGLE_API_KEY")
        if not key:
            return "Missing GOOGLE_API_KEY. Set it in your shell before using RAG."

        try:
            config = _get_hw3_rag_config()
            vectorstore = build_vectorstore(config)
            retriever = vectorstore.as_retriever(search_kwargs={"k": k})

            docs = retrieve_docs(vectorstore, retriever, question, k)
            if not docs:
                return "No chunks retrieved. Ingest a Gutenberg URL first."

            answer = build_rag_answer(question, docs)

            sources = []
            for d in docs:
                src = (d.metadata or {}).get("source", "unknown")
                if src not in sources:
                    sources.append(src)

            lines = []
            lines.append("Answer:")
            lines.append(answer.strip())
            lines.append("")
            lines.append("Sources:")
            for s in sources:
                lines.append(f"- {s}")

            return "\n".join(lines)
        except Exception as e:
            return f"RAG ask failed: {e}"

    @tool("gutenberg_sources")
    def gutenberg_sources() -> str:
        """List the sources stored in the local Gutenberg vector database."""
        key = _require_env_soft("GOOGLE_API_KEY")
        if not key:
            return "Missing GOOGLE_API_KEY. Set it in your shell before using RAG."

        try:
            config = _get_hw3_rag_config()
            vectorstore = build_vectorstore(config)

            total, counts = get_db_sources(vectorstore)
            lines = []
            lines.append("DB sources:")
            if total is not None:
                lines.append(f"Total chunks: {total}")
            else:
                lines.append("Total chunks: (unavailable)")

            if not counts:
                lines.append("No source metadata found yet. Ingest a document first.")
                return "\n".join(lines)

            for src, n in sorted(counts.items(), key=lambda x: (-x[1], x[0])):
                lines.append(f"- {src} ({n} chunks)")

            return "\n".join(lines)
        except Exception as e:
            return f"RAG sources failed: {e}"

    return [gutenberg_ingest, gutenberg_ask, gutenberg_sources]


def _looks_like_url(s: str) -> bool:
    """Small helper for spotting URLs inside a user query."""
    return bool(re.search(r"https?://\S+", s or ""))

def _maybe_answer_from_persisted_gutenberg(query: str, k: int = 4) -> str | None:
    """
    If the user references an "ingested" book but provides no URL, try the persisted DB first.
    If the DB is empty, return None and let the agent handle it.
    """
    q = (query or "").strip()
    if not q:
        return None

    if _looks_like_url(q):
        return None

    ql = q.lower()
    triggers = (
        "ingested",
        "gutenberg",
        "from the ingested",
        "from the book",
        "summarize the beginning",
    )
    if not any(t in ql for t in triggers):
        return None

    try:
        _add_repo_root_to_syspath()
        from hw2.app import (  # type: ignore
            build_rag_answer,
            build_vectorstore,
            get_db_sources,
            retrieve_docs,
        )

        config = _get_hw3_rag_config()
        vectorstore = build_vectorstore(config)

        total, counts = get_db_sources(vectorstore)
        if not counts:
            return None

        print("Calling tools: ['gutenberg_sources']")

        retriever = vectorstore.as_retriever(search_kwargs={"k": k})
        docs = retrieve_docs(vectorstore, retriever, q, k)
        if not docs:
            return None

        print("Calling tools: ['gutenberg_ask']")

        answer = build_rag_answer(q, docs)

        sources = []
        for d in docs:
            src = (d.metadata or {}).get("source", "unknown")
            if src not in sources:
                sources.append(src)

        lines = []
        lines.append(answer.strip())
        lines.append("")
        lines.append("Sources:")
        for s in sources:
            lines.append(f"- {s}")

        return "\n".join(lines).strip()
    except Exception:
        return None

def _run_brave_agent(query: str, count: int, max_tool_calls: int) -> str:
    """
    Run an agent powered by Brave Search, local RAG tools, and a guarded Terminal tool.
    """
    _add_repo_root_to_syspath()

    try:
        from hw2.app import get_llm  # type: ignore
    except Exception as e:
        print("ERROR: Failed to import hw2 helpers (get_llm).", file=sys.stderr)
        print(f"Import error: {e}", file=sys.stderr)
        return ""

    if not _require_env_soft("GOOGLE_API_KEY"):
        print("ERROR: Missing GOOGLE_API_KEY.", file=sys.stderr)
        return ""

    auto = _maybe_answer_from_persisted_gutenberg(query, k=4)
    if auto:
        return auto

    brave_key = _require_env_soft("BRAVE_SEARCH_API_KEY")
    if not brave_key:
        print("ERROR: Missing BRAVE_SEARCH_API_KEY.", file=sys.stderr)
        return ""

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
    rag_tools = _build_gutenberg_tools()
    terminal_tool = _build_terminal_tool()

    tools = [search] + rag_tools + [terminal_tool]

    instructions = (
        "You are a helpful assistant with tools. "
        "Use Brave Search when you need fresh info. "
        "If the user provides a Project Gutenberg URL, ingest it with gutenberg_ingest before answering. "
        "The Gutenberg database persists locally across runs under hw3/.chromadb. "
        "If the user asks about the book but does not provide a URL, call gutenberg_sources first. "
        "If sources exist, proceed with gutenberg_ask and stay grounded in the retrieved chunks. "
        "If no sources exist, ask the user for a Gutenberg URL to ingest. "
        "Use the terminal tool only for read-only inspection commands. "
        f"Use at most {max_tool_calls} tool calls. "
        "If you are not confident, say what you do not know."
    )

    agent = create_agent(
        llm,
        tools=tools,
        system_prompt=instructions,
    )

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
    """HW3 agent mode - one-shot or interactive."""
    parser = argparse.ArgumentParser(
        prog="hw3 agent",
        description="HW3 agent mode (Brave Search + Gutenberg tools + Terminal).",
    )
    parser.add_argument("query", nargs="?", help="One-shot query. If omitted, runs interactive mode.")
    parser.add_argument("--count", type=int, default=3, help="Number of Brave results to fetch.")
    parser.add_argument("--max-tool-calls", type=int, default=8, help="Upper bound for tool calls.")
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
