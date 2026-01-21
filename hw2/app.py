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

    We keep the Chroma directory under hw2/.chromadb so it is:
    - persisted locally
    - ignored by git via .gitignore
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
        # Newer LangChain prefers langchain-chroma, but we can fall back.
        try:
            from langchain_chroma import Chroma  # type: ignore
        except ImportError:
            from langchain_community.vectorstores import Chroma  # type: ignore

        from langchain_google_genai import GoogleGenerativeAIEmbeddings  # type: ignore
    except ImportError as e:
        print("ERROR: Missing dependencies for LangChain/Chroma.", file=sys.stderr)
        print("Install them from your repo root with uv:", file=sys.stderr)
        print("  uv init --bare", file=sys.stderr)
        print("  uv add langchain langchain-community langchain-google-genai chromadb", file=sys.stderr)
        print("  uv add langchain-text-splitters langchain-chroma", file=sys.stderr)
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


def load_gutenberg_docs(url: str):
    """
    Load a Project Gutenberg plain-text URL into LangChain Documents.

    This keeps ingestion simple: one URL becomes a document, then we split it
    into chunks before inserting into Chroma.
    """
    try:
        from langchain_community.document_loaders import GutenbergLoader  # type: ignore
    except ImportError as e:
        print("ERROR: GutenbergLoader is not available in your environment.", file=sys.stderr)
        print("Make sure LangChain community loaders are installed.", file=sys.stderr)
        print(f"Import error: {e}", file=sys.stderr)
        sys.exit(1)

    if not url.startswith("http://") and not url.startswith("https://"):
        print("ERROR: Please provide a full URL (starting with http:// or https://).", file=sys.stderr)
        sys.exit(1)

    loader = GutenbergLoader(url)
    docs = loader.load()

    # Make the source obvious and stable when we later cite where something came from.
    for d in docs:
        d.metadata = d.metadata or {}
        d.metadata["source"] = url

    return docs


def split_docs(docs, chunk_size: int, chunk_overlap: int):
    """
    Split Documents into smaller chunks for embedding and retrieval.
    A modest overlap helps keep sentence boundaries from being sliced too sharply.
    """
    try:
        # Newer splitters live in langchain-text-splitters, but we can fall back.
        try:
            from langchain_text_splitters import RecursiveCharacterTextSplitter  # type: ignore
        except ImportError:
            from langchain.text_splitter import RecursiveCharacterTextSplitter  # type: ignore
    except ImportError as e:
        print("ERROR: Missing text splitter dependency.", file=sys.stderr)
        print(f"Import error: {e}", file=sys.stderr)
        sys.exit(1)

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    return text_splitter.split_documents(docs)


def ingest_documents(vectorstore, docs, chunk_size: int, chunk_overlap: int) -> int:
    """
    Ingest Documents into the vectorstore.
    Returns the number of chunks added.
    """
    splits = split_docs(docs, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    vectorstore.add_documents(documents=splits)

    # Some Chroma versions persist automatically, but calling persist() is harmless.
    if hasattr(vectorstore, "persist"):
        try:
            vectorstore.persist()
        except Exception:
            pass

    return len(splits)


def cmd_status(_: argparse.Namespace) -> int:
    """
    Print the app configuration and verify we can initialize the vectorstore.
    This is a safe 'smoke test' for your environment before adding RAG logic.
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


def cmd_add_gutenberg(args: argparse.Namespace) -> int:
    """
    Load a Gutenberg book from a URL, chunk it, and store it in Chroma.
    """
    config = get_config()
    require_env("GOOGLE_API_KEY")

    vectorstore = build_vectorstore(config)

    print("== Gutenberg ingestion ==")
    print(f"URL: {args.url}")
    print(f"Chunk size: {args.chunk_size}")
    print(f"Chunk overlap: {args.chunk_overlap}")

    docs = load_gutenberg_docs(args.url)
    if not docs:
        print("No documents returned from loader.", file=sys.stderr)
        return 1

    raw_len = len(docs[0].page_content or "")
    print(f"Loaded 1 document ({raw_len} chars)")

    chunks_added = ingest_documents(
        vectorstore,
        docs,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
    )
    print(f"Inserted {chunks_added} chunks into Chroma")
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

    p_add = subparsers.add_parser(
        "add-gutenberg",
        help="Ingest a Project Gutenberg plain-text URL into the vector DB.",
    )
    p_add.add_argument("url", help="Gutenberg plain-text URL (pgXXXX.txt).")
    p_add.add_argument("--chunk-size", type=int, default=2000, help="Chunk size in characters.")
    p_add.add_argument("--chunk-overlap", type=int, default=200, help="Chunk overlap in characters.")
    p_add.set_defaults(func=cmd_add_gutenberg)

    return parser


def main() -> int:
    """Entry point."""
    parser = build_cli()
    args = parser.parse_args()
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
