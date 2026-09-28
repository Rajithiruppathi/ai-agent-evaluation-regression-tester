"""The AI application being evaluated.

This module stands in for "the app under test" - today it's a simple Claude-based
customer support assistant, but it could be swapped for any other AI system as long
as it exposes generate_response(question: str) -> str. Nothing else in this project
(evaluator.py, regression.py) knows or cares how this function is implemented.
"""

import json
import os

import anthropic
from dotenv import load_dotenv

load_dotenv()

_client = None
_knowledge_base = None


def _get_client():
    global _client
    if _client is None:
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise RuntimeError("ANTHROPIC_API_KEY is not set. Add it to your .env file.")
        _client = anthropic.Anthropic(api_key=api_key)
    return _client


def _load_knowledge_base(path: str = "knowledge_base.json") -> dict:
    global _knowledge_base
    if _knowledge_base is None:
        with open(path, "r", encoding="utf-8") as f:
            _knowledge_base = json.load(f)
    return _knowledge_base


def _format_knowledge(knowledge: dict) -> str:
    return "\n".join(f"- {entry['topic']}: {entry['policy']}" for entry in knowledge.values())


def generate_response(question: str) -> str:
    """Return the AI application's answer to a customer support question."""
    client = _get_client()
    knowledge_text = _format_knowledge(_load_knowledge_base())

    system_prompt = (
        "You are a helpful customer support assistant. Use the company policies below to "
        "answer the customer's question clearly and briefly, in your own words. If the "
        "question isn't covered by these policies, say you don't have that information.\n\n"
        f"Company policies:\n{knowledge_text}"
    )

    response = client.messages.create(
        model="claude-opus-5",
        max_tokens=512,
        system=system_prompt,
        messages=[{"role": "user", "content": question}],
    )
    return next(block.text for block in response.content if block.type == "text")
