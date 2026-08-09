"""Shared data-pipeline helpers used by the Flask app, the scheduled sync job,
and the benchmark script -- so the upsert and vector-search SQL live in exactly
one place.

Embedding is intentionally NOT done here: callers pass an already-embedded
query vector so the sentence-transformers model is loaded once at the app/script
level, never per call.
"""

from psycopg2.extras import Json, execute_values

from lakebase import get_connection
from weather_client import fetch_location_documents

VALID_SOURCE_TYPES = ("alert", "forecast")

_UPSERT_SQL = """
    INSERT INTO weather_documents
        (id, location, source_type, headline, narrative_text,
         issued_at, payload, synced_at)
    VALUES %s
    ON CONFLICT (id) DO UPDATE
        SET narrative_text = EXCLUDED.narrative_text,
            payload        = EXCLUDED.payload,
            synced_at      = now();
"""

# A NULL source_type means "no filter". Passing the same param twice keeps the
# query fully parameterized (no string building) while making the filter
# optional.
_SEARCH_SQL = """
    SELECT d.id, d.location, d.headline, d.narrative_text, e.chunk_text,
           d.source_type,
           1 - (e.embedding <=> %s::vector) AS similarity
    FROM weather_embeddings e
    JOIN weather_documents d ON d.id = e.document_id
    WHERE (%s IS NULL OR d.source_type = %s)
    ORDER BY e.embedding <=> %s::vector
    LIMIT %s;
"""


def sync_documents(locations, limit=50):
    """Fetch + normalize NWS docs for the locations and upsert them.

    Returns the number of rows upserted. Dedup/upsert is on `id`, so re-running
    refreshes changed documents instead of creating duplicates.
    """
    records = []
    for location in locations:
        if not isinstance(location, str):
            continue
        records.extend(fetch_location_documents(location))
        if len(records) >= limit:
            break
    records = records[:limit]

    if not records:
        return 0

    rows = [
        (
            r["id"],
            r["location"],
            r["source_type"],
            r["headline"],
            r["narrative_text"],
            r["issued_at"],
            Json(r["payload"]),
            r["synced_at"],
        )
        for r in records
    ]

    with get_connection() as conn:
        with conn.cursor() as cur:
            execute_values(cur, _UPSERT_SQL, rows)
        conn.commit()

    return len(rows)


def embedding_count():
    """Number of rows in weather_embeddings (used to detect an empty index)."""
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT count(*) AS n FROM weather_embeddings;")
            return cur.fetchone()["n"]


def vector_search(query_embedding, top_k, source_type=None):
    """Cosine-similarity search over weather_embeddings.

    `query_embedding` is a Python list of floats. `source_type`, if given,
    restricts to 'alert' or 'forecast'. Returns a list of result dicts.
    """
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                _SEARCH_SQL,
                (query_embedding, source_type, source_type, query_embedding, top_k),
            )
            matches = cur.fetchall()

    return [
        {
            "location": m["location"],
            "headline": m["headline"],
            "source_type": m["source_type"],
            "chunk_text": m["chunk_text"],
            "similarity": float(m["similarity"]),
        }
        for m in matches
    ]
