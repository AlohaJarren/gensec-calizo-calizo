from __future__ import annotations

import argparse
import os
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class AppConfig:
    """Application configuration for local persistence and embeddings."""

    persist_dir: Path
    collection_name: str
    embedding_model: str


def get_repo_relative_path(*parts: str) -> Path:
    """
    Return a path relative to this file's directory.
    This avoids confusion when running from different working directories.
    """
    return Path(__file__).resolve().parent.joinpath(*parts)


def get_config() -> AppConfig:
    """
    Centralized config for the app.
    """
    return AppConfig(
        persist_dir=get_repo_relative_path(".chromadb"),
        collection_name="hw2_rag",
        embedding_model="models/embedding-001",
    )


def require_env(var_name: str) -> str:
    """
    Ensure an environment variable exists and is non-empty.
    Returns the value if present, otherwise exits with a helpful message.
    """
    value = os.getenv(var_name, "").strip()
    if not value:
        print(f"ERROR: Missing required environment variable: {var_name}", file=sys.stderr)
        print("Set it in your shell before running the app.", file=sys.stderr)
        print(f"Example:\n  export {var_name}='YOUR_VALUE_HERE'\n", file=sys.stderr)
        sys.exit(1)
    return value


def ensure_persist_dir(path: Path) -> None:
    """Create the persistence directory if it does not exist."""
    path.mkdir(parents=True, exist_ok=True)


def build_vectorstore(config: AppConfig):
    """
    Instantiate a persistent Chroma vectorstore using Google Gemini embeddings.
    """
    try:
        from langchain_community.vectorstores import Chroma
        from langchain_google_genai import GoogleGenerativeAIEmbeddings
    except ImportError as e:
        print("ERROR: Missing dependencies. Did you install requirements?", file=sys.stderr)
        print("Try running from your repo root:", file=sys.stderr)
        print("  uv init --bare", file=sys.stderr)
        print("  uv add -r requirements.txt", file=sys.stderr)
        print(f"\nImport error: {e}", file=sys.stderr)
        sys.exit(1)

    ensure_persist_dir(config.persist_dir)

    embeddings = GoogleGenerativeAIEmbeddings(
        model=config.embedding_model,
        task_type="retrieval_query",
    )

    vectorstore = Chroma(
        collection_name=config.collection_name,
        persist_directory=str(config.persist_dir),
        embedding_function=embeddings,
    )
    return vectorstore


def cmd_status(_: argparse.Namespace) -> int:
    """
    Print the app configuration and verify we can initialize the vectorstore.
    Smoke test for environment before adding RAG logic.
    """
    config = get_config()

    # Validate keys without printing them.
    require_env("GOOGLE_API_KEY")

    # USER_AGENT is used in later labs/URLs - not strictly required yet,
    # but printing whether it exists helps avoid confusion later.
    user_agent = os.getenv("USER_AGENT", "").strip()

    print("== HW RAG App Status ==")
    print(f"Persist dir: {config.persist_dir}")
    print(f"Collection:  {config.collection_name}")
    print(f"Embeddings:  {config.embedding_model}")
    print(f"USER_AGENT set: {'yes' if user_agent else 'no'}")

    # Try initializing the vectorstore (this validates imports + filesystem).
    _ = build_vectorstore(config)
    print("Vectorstore init: OK")
    return 0


def cmd_reset_db(_: argparse.Namespace) -> int:
    """
    Delete the local Chroma persistence directory.
    Useful when you want a clean rebuild of your vector DB.
    """
    config = get_config()
    if config.persist_dir.exists():
        shutil.rmtree(config.persist_dir)
        print(f"Deleted: {config.persist_dir}")
    else:
        print(f"No DB found at: {config.persist_dir}")
    return 0


def build_cli() -> argparse.ArgumentParser:
    """Create the command-line interface."""
    parser = argparse.ArgumentParser(
        prog="hw2-rag",
        description="Custom LangChain RAG application (incremental HW build).",
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    p_status = subparsers.add_parser("status", help="Validate env + init vectorstore.")
    p_status.set_defaults(func=cmd_status)

    p_reset = subparsers.add_parser("reset-db", help="Delete local Chroma DB folder.")
    p_reset.set_defaults(func=cmd_reset_db)

    return parser


def main() -> int:
    """Entry point."""
    parser = build_cli()
    args = parser.parse_args()
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
