"""Optional RAG summarization via the Databricks Foundation Model API.

Calls a workspace model-serving endpoint through the Databricks SDK, so no
separate API key is needed -- auth comes from the app's service principal when
deployed (or `databricks auth login` locally). The endpoint name is
configurable via the LLM_ENDPOINT_NAME env var.

summarize() never raises: on any failure it returns (None, "<reason>") so the
search endpoint can still return its vector results with summary=null.
"""

import os

# Set this to a chat/completions serving endpoint that exists in YOUR workspace
# (e.g. a Databricks-hosted Claude or Llama endpoint).
DEFAULT_ENDPOINT = "databricks-llama-4-maverick"

_client = None


def _get_client():
    """Lazily construct and cache a WorkspaceClient (SDK imported lazily)."""
    global _client
    if _client is None:
        from databricks.sdk import WorkspaceClient

        _client = WorkspaceClient()
    return _client


def _build_prompt(query, results):
    """Assemble the grounding context from the retrieved chunks."""
    lines = []
    for i, r in enumerate(results, start=1):
        lines.append(
            f"[{i}] location={r['location']} | type={r['source_type']} | "
            f"headline={r['headline']}\n{r['chunk_text']}"
        )
    context = "\n\n".join(lines)
    return (
        f"User question: {query}\n\n"
        f"Weather documents retrieved by semantic search:\n{context}\n\n"
        "Write a concise (2-4 sentence) natural-language answer to the user's "
        "question grounded ONLY in the documents above. Mention locations and "
        "any alerts where relevant. If the documents don't address the "
        "question, say so plainly."
    )


def summarize(query, results, endpoint_name=None):
    """Return (summary_text, error). Exactly one is non-None.

    `results` is the list of dicts from pipeline.vector_search().
    """
    if not results:
        return None, "no results to summarize"

    endpoint = endpoint_name or os.environ.get("LLM_ENDPOINT_NAME", DEFAULT_ENDPOINT)

    try:
        from databricks.sdk.service.serving import ChatMessage, ChatMessageRole

        client = _get_client()
        response = client.serving_endpoints.query(
            name=endpoint,
            messages=[
                ChatMessage(
                    role=ChatMessageRole.SYSTEM,
                    content="You are a careful weather assistant. Answer only "
                    "from the provided documents.",
                ),
                ChatMessage(
                    role=ChatMessageRole.USER,
                    content=_build_prompt(query, results),
                ),
            ],
            max_tokens=300,
            temperature=0.2,
        )
        return response.choices[0].message.content, None
    except Exception as exc:  # noqa: BLE001 - degrade gracefully, never 500 here
        return None, f"{type(exc).__name__}: {exc}"
