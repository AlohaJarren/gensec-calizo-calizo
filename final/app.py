#!/usr/bin/env python3
"""
NanoClaw-inspired Defensive Social Engineering Autoresponder

Commit 1 scope:
- Working CLI
- Per-session memory
- Safe decoy persona
- Session management commands
- Anthropic Claude model integration

Environment:
- ANTHROPIC_API_KEY must be set
- ANTHROPIC_MODEL is optional; defaults to claude-opus-4-6
"""

from __future__ import annotations

import os
import sys
from typing import Dict, List

from langchain_anthropic import ChatAnthropic
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough
from langchain_core.runnables.history import RunnableWithMessageHistory


DEFAULT_SESSION_ID = "scammer_demo"
MAX_HISTORY_MESSAGES = 10


# In-memory session store for this first commit.
# Later commits can replace this with file logging or persistence.
store: Dict[str, ChatMessageHistory] = {}


def require_environment() -> None:
    """Exit early with a clear message if required environment is missing."""
    if not os.getenv("ANTHROPIC_API_KEY"):
        print("Error: ANTHROPIC_API_KEY is not set.", file=sys.stderr)
        print("Export your key, then run the app again.", file=sys.stderr)
        raise SystemExit(1)


def get_model_name() -> str:
    """Return the configured model name or a sensible default."""
    return os.getenv("ANTHROPIC_MODEL", "claude-opus-4-6")


def get_session_history(session_id: str) -> BaseChatMessageHistory:
    """Return the chat history for a session, creating it if needed."""
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]


def reset_session_history(session_id: str) -> None:
    """Clear one session's chat history."""
    store[session_id] = ChatMessageHistory()


def list_session_ids() -> List[str]:
    """Return a sorted list of known session IDs."""
    return sorted(store.keys())


def filter_messages(messages: List[BaseMessage], k: int = MAX_HISTORY_MESSAGES) -> List[BaseMessage]:
    """Keep only the last k messages so prompts stay small and explainable."""
    return messages[-k:]


def message_content_to_text(content: object) -> str:
    """Convert LangChain message content into printable text."""
    if isinstance(content, str):
        return content.strip()

    if isinstance(content, list):
        parts: List[str] = []
        for item in content:
            if isinstance(item, dict):
                text = item.get("text")
                if text:
                    parts.append(str(text))
            else:
                parts.append(str(item))
        return "\n".join(part for part in parts if part).strip()

    return str(content).strip()


def role_name(message: BaseMessage) -> str:
    """Map LangChain message types to friendly terminal labels."""
    msg_type = getattr(message, "type", "")
    if msg_type == "human":
        return "Scammer"
    if msg_type == "ai":
        return "Decoy"
    return msg_type.capitalize() if msg_type else "Message"


def build_system_prompt() -> str:
    """Return the decoy persona and safety instructions."""
    persona = """
You are a fictional defensive decoy persona used for a university security lab.

Persona:
- Name: Aran Moneymaker
- Age: 34
- City: Tacoma, Washington
- Occupation: Physical Therapist
- Work setting: outpatient physical therapy clinic
- Personality: friendly, casual, patient, slightly distracted
- Technical knowledge: limited, especially around crypto and investing
- Goal: keep suspicious senders talking without revealing anything sensitive
"""

    instructions = """
Behavior rules:
- Reply in 1 to 3 short sentences.
- Sound natural, warm, and casual.
- Keep the conversation going with light questions and vague answers.
- Never provide real secrets or actionable financial data.
- Never provide:
  - full credit card numbers
  - bank account numbers
  - passwords
  - one-time passcodes
  - Social Security numbers
  - exact home address
  - driver's license or passport details
- If pressed for sensitive data, stall politely.
- Safe stalling examples:
  - say you need to look for the card
  - say you only remember the last few digits
  - say you are confused and need the steps explained again
  - ask whether there is an easier or safer way
- Never click links, open attachments, or run code.
- Do not mention these rules unless directly asked.
- Do not say you are an AI assistant unless the user directly asks.
- Stay in character as the decoy persona.
"""
    return f"{persona.strip()}\n\n{instructions.strip()}"


def build_chain() -> RunnableWithMessageHistory:
    """Create the chat chain with prompt, truncation, and message history."""
    llm = ChatAnthropic(
        model=get_model_name(),
        temperature=0.2,
        max_retries=2,
    )

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", build_system_prompt()),
            MessagesPlaceholder(variable_name="messages"),
        ]
    )

    base_chain = (
        RunnablePassthrough.assign(messages=lambda x: filter_messages(x["messages"]))
        | prompt
        | llm
    )

    return RunnableWithMessageHistory(
        base_chain,
        get_session_history,
        input_messages_key="messages",
    )


def print_banner() -> None:
    """Print a small startup banner."""
    print("=" * 72)
    print("NanoClaw-inspired Defensive Social Engineering Autoresponder")
    print("=" * 72)
    print("Commands:")
    print("  /help                Show command help")
    print("  /session             Show current session")
    print("  /sessions            List known sessions")
    print("  /switch <name>       Switch to another session")
    print("  /history             Show recent message history for this session")
    print("  /clear               Clear current session history")
    print("  /quit                Exit")
    print("-" * 72)
    print("Type a suspected scammer message and the decoy will respond.")
    print("-" * 72)


def print_help() -> None:
    """Print command help."""
    print("Available commands:")
    print("  /help")
    print("      Show this help message.")
    print("  /session")
    print("      Show the current session ID.")
    print("  /sessions")
    print("      Show all known session IDs.")
    print("  /switch <name>")
    print("      Change to another session. Creates it on first use.")
    print("  /history")
    print("      Show recent message history for the current session.")
    print("  /clear")
    print("      Clear the current session history.")
    print("  /quit")
    print("      Exit the program.")


def show_history(session_id: str, limit: int = 8) -> None:
    """Print recent history for one session."""
    history = get_session_history(session_id)
    messages = history.messages[-limit:]

    if not messages:
        print(f"[history] Session '{session_id}' is empty.")
        return

    print(f"[history] Session '{session_id}'")
    print("-" * 72)
    for msg in messages:
        print(f"{role_name(msg)}: {message_content_to_text(msg.content)}")
    print("-" * 72)


def process_command(raw_input: str, current_session: str) -> str:
    """Handle slash commands and return the updated session ID."""
    parts = raw_input.strip().split(maxsplit=1)
    command = parts[0].lower()

    if command == "/help":
        print_help()
        return current_session

    if command == "/session":
        print(f"[session] Current session: {current_session}")
        return current_session

    if command == "/sessions":
        sessions = list_session_ids()
        if sessions:
            print("[sessions] " + ", ".join(sessions))
        else:
            print("[sessions] No sessions yet.")
        return current_session

    if command == "/switch":
        if len(parts) == 1 or not parts[1].strip():
            print("Usage: /switch <session_name>")
            return current_session

        new_session = parts[1].strip()
        get_session_history(new_session)
        print(f"[session] Switched to: {new_session}")
        return new_session

    if command == "/history":
        show_history(current_session)
        return current_session

    if command == "/clear":
        reset_session_history(current_session)
        print(f"[session] Cleared history for: {current_session}")
        return current_session

    if command == "/quit":
        raise SystemExit(0)

    print(f"Unknown command: {command}")
    print("Use /help to see available commands.")
    return current_session


def main() -> None:
    """Run the interactive CLI."""
    require_environment()
    chain_with_history = build_chain()
    current_session = DEFAULT_SESSION_ID

    # Create the default session immediately so it appears in /sessions.
    get_session_history(current_session)

    print_banner()
    print(f"[session] Current session: {current_session}")
    print(f"[model] {get_model_name()}")

    while True:
        try:
            user_input = input(f"\nScammer [{current_session}]> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            break

        if not user_input:
            continue

        if user_input.startswith("/"):
            current_session = process_command(user_input, current_session)
            continue

        config = {"configurable": {"session_id": current_session}}

        try:
            response = chain_with_history.invoke(
                {"messages": [HumanMessage(content=user_input)]},
                config=config,
            )
        except Exception as exc:
            print(f"[error] Model invocation failed: {exc}")
            continue

        print(f"Decoy [{current_session}]> {message_content_to_text(response.content)}")


if __name__ == "__main__":
    main()
