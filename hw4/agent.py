"""
FastAgent-based MCP agent that talks to the custom NHI MCP server.
""" 
from __future__ import annotations

import asyncio
import os

from fast_agent.core.fastagent import FastAgent


fast = FastAgent("HW4 NHI Security Agent")

INSTRUCTION = (
    "You are a security assistant focused on non-human identity (NHI) hygiene.\n"
    "You can call tools from the NHI MCP server to:\n"
    "- scan_repo: scan paths for likely secrets/machine identities\n"
    "- risk_score: assign explainable risk scores\n"
    "- build_identity_graph: create a simple graph of secret->owner/location\n"
    "- explain_limitations: state known limitations for rubric\n\n"
    "Rules:\n"
    "- Never output raw secrets. Use fingerprints and redacted evidence only.\n"
    "- Default to planning mode: propose remediation steps, do not do destructive actions.\n"
)

def _default_model() -> str:
    # FastAgent model strings are provider.model_name. Google is supported via GOOGLE_API_KEY
    google_model = os.environ.get("GOOGLE_MODEL", "gemini-2.5-flash").strip()
    return f"google.{google_model}"

@fast.agent(
    instruction=INSTRUCTION,
    model=os.environ.get("FASTAGENT_MODEL", _default_model()),
    servers=["nhi_stdio"],
    use_history=True,
)
async def main():
    async with fast.run() as agent:
        await agent.interactive()

if __name__ == "__main__":
    asyncio.run(main())
