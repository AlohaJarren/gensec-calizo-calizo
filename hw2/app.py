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


def get_llm():
    """
    Instantiate the chat model used for answering questions.

    The model name can be overridden with GEMINI_MODEL if you want to experiment.
    """
    try:
        from langchain_google_genai import ChatGoogleGenerativeAI  # type: ignore
    except ImportError as e:
        print("ERROR: Missing langchain-google-genai dependency.", file=sys.stderr)
        print(f"Import error: {e}", file=sys.stderr)
        sys.exit(1)

    model_name = os.getenv("GEMINI_MODEL", "gemini-2.0-flash").strip()
    return ChatGoogleGenerativeAI(model=model_name, temperature=0)


def format_docs(docs) -> str:
    """Concatenate retrieved chunks into a single context block."""
    return "\n\n".join(d.page_content for d in docs if getattr(d, "page_content", None))


def build_rag_answer(question: str, docs) -> str:
    """
    Generate a grounded answer using only retrieved context.

    The prompt is intentionally strict. If the answer is not in the context,
    the model should say "I don't know."
    """
    try:
        try:
            from langchain_core.prompts import ChatPromptTemplate  # type: ignore
        except ImportError:
            from langchain.prompts import ChatPromptTemplate  # type: ignore

        try:
            from langchain_core.output_parsers import StrOutputParser  # type: ignore
        except ImportError:
            from langchain.schema.output_parser import StrOutputParser  # type: ignore
    except ImportError as e:
        print("ERROR: Missing LangChain core prompt/parser components.", file=sys.stderr)
        print(f"Import error: {e}", file=sys.stderr)
        sys.exit(1)

    llm = get_llm()
    context = format_docs(docs)

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are an assistant for question-answering tasks. Use the provided context to answer. "
                "If the answer is not in the context, say \"I don't know\". Use three sentences maximum.",
            ),
            ("human", "Question:\n{question}\n\nContext:\n{context}\n\nAnswer:"),
        ]
    )

    chain = prompt | llm | StrOutputParser()
    return chain.invoke({"question": question, "context": context})


def retrieve_docs(vectorstore, retriever, query: str, k: int):
    """
    Retrieve relevant chunks for a query.

    LangChain has shifted retrievers toward the Runnable interface, so the most
    reliable call path is retriever.invoke(query). If that is not available,
    we fall back to similarity_search().
    """
    try:
        if hasattr(retriever, "invoke"):
            docs = retriever.invoke(query)
        else:
            docs = vectorstore.similarity_search(query, k=k)
    except Exception:
        docs = vectorstore.similarity_search(query, k=k)

    return docs or []


def clean_snippet(text: str, limit: int = 220) -> str:
    """Return a compact one-line preview of a chunk."""
    s = (text or "").strip().replace("\n", " ")
    if len(s) > limit:
        return s[:limit] + "..."
    return s


def get_db_sources(vectorstore):
    """
    Return (total_chunks, {source: count}) if the backend supports it.
    """
    try:
        data = vectorstore.get(include=["metadatas"])
        metadatas = data.get("metadatas", []) or []
        total = len(metadatas)

        counts: dict[str, int] = {}
        for md in metadatas:
            if not md:
                continue
            src = md.get("source", "unknown")
            counts[src] = counts.get(src, 0) + 1

        return total, counts
    except Exception:
        return None, {}


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


def cmd_search(args: argparse.Namespace) -> int:
    """
    Search the vector database and print the closest matching chunks.
    """
    config = get_config()
    require_env("GOOGLE_API_KEY")

    vectorstore = build_vectorstore(config)
    docs = vectorstore.similarity_search(args.query, k=args.k)

    print("== Vector search ==")
    print(f"Query: {args.query}")
    print(f"Matches: {len(docs)}\n")

    for i, d in enumerate(docs, start=1):
        src = (d.metadata or {}).get("source", "unknown")
        print(f"{i}. Source: {src}")
        print(f"   Snippet: {clean_snippet(d.page_content)}\n")

    return 0


def cmd_sources(_: argparse.Namespace) -> int:
    """
    List the sources currently stored in the database.
    """
    config = get_config()
    require_env("GOOGLE_API_KEY")

    vectorstore = build_vectorstore(config)
    total, counts = get_db_sources(vectorstore)

    print("== DB sources ==")
    if total is not None:
        print(f"Total chunks: {total}")
    else:
        print("Total chunks: (unavailable in this backend)")

    if not counts:
        print("No source metadata found yet. Ingest a document first.")
        return 0

    for src, n in sorted(counts.items(), key=lambda x: (-x[1], x[0])):
        print(f"- {src} ({n} chunks)")

    return 0


def cmd_ask(args: argparse.Namespace) -> int:
    """
    Ask a question against the local vector database.

    This command retrieves the most relevant chunks and answers strictly from them.
    """
    config = get_config()
    require_env("GOOGLE_API_KEY")

    vectorstore = build_vectorstore(config)
    retriever = vectorstore.as_retriever(search_kwargs={"k": args.k})

    # Retrieving first keeps the model honest: it should answer from the book, not from memory.
    docs = retrieve_docs(vectorstore, retriever, args.question, args.k)

    print("== RAG query ==")
    print(f"Question: {args.question}")
    print(f"Retrieved chunks: {len(docs)}")

    if args.show_snippets and docs:
        print("\nContext preview:")
        for i, d in enumerate(docs, start=1):
            src = (d.metadata or {}).get("source", "unknown")
            print(f"{i}. {src}")
            print(f"   {clean_snippet(d.page_content)}")

    answer = build_rag_answer(args.question, docs)
    print("\nAnswer:")
    print(answer.strip())

    # A small breadcrumb trail so you can prove grounding in your screencast.
    sources = []
    for d in docs:
        src = (d.metadata or {}).get("source", "unknown")
        if src not in sources:
            sources.append(src)

    print("\nSources:")
    for s in sources:
        print(f"- {s}")

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

    p_search = subparsers.add_parser(
        "search",
        help="Search for the closest chunks in the vector DB.",
    )
    p_search.add_argument("query", help="Search query.")
    p_search.add_argument("--k", type=int, default=4, help="Number of matches to show.")
    p_search.set_defaults(func=cmd_search)

    p_sources = subparsers.add_parser(
        "sources",
        help="List sources currently stored in the vector DB.",
    )
    p_sources.set_defaults(func=cmd_sources)

    p_ask = subparsers.add_parser(
        "ask",
        help="Ask a question using retrieval-augmented generation (RAG).",
    )
    p_ask.add_argument("question", help="Question to ask against your ingested documents.")
    p_ask.add_argument("--k", type=int, default=4, help="Number of chunks to retrieve.")
    p_ask.add_argument(
        "--show-snippets",
        action="store_true",
        help="Print short previews of the retrieved chunks before answering.",
    )
    p_ask.set_defaults(func=cmd_ask)

    return parser


def main() -> int:
    """Entry point."""
    parser = build_cli()
    args = parser.parse_args()
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())