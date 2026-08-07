"""Flask app: retrieval-augmented search over NWS weather data on Lakebase.

Endpoints:
  POST /weather/sync   - fetch + upsert weather documents
  POST /weather/search - semantic search over embedded chunks
  GET  /health         - row counts for both tables
"""

import json
import os

from flask import Flask, jsonify, request
from psycopg2.extras import Json, execute_values
from sentence_transformers import SentenceTransformer

from lakebase import get_connection
from weather_client import fetch_location_documents

app = Flask(__name__)

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# Load the embedding model ONCE at import time, never per request.
_model = SentenceTransformer(MODEL_NAME)


def _embed(text):
    """Return a Python list of floats for a single string."""
    vector = _model.encode(text, normalize_embeddings=False)
    return vector.tolist()


@app.route("/weather/sync", methods=["POST"])
def weather_sync():
    body = request.get_json(silent=True) or {}
    locations = body.get("locations") or []
    limit = body.get("limit", 50)

    if not isinstance(locations, list) or not locations:
        return jsonify({"error": "Provide a non-empty 'locations' list."}), 400

    try:
        limit = int(limit)
    except (TypeError, ValueError):
        return jsonify({"error": "'limit' must be an integer."}), 400
    if limit < 1:
        return jsonify({"error": "'limit' must be >= 1."}), 400

    # Gather normalized records across all requested locations, capped at limit.
    records = []
    for location in locations:
        if not isinstance(location, str):
            continue
        records.extend(fetch_location_documents(location))
        if len(records) >= limit:
            break
    records = records[:limit]

    if not records:
        return jsonify({"synced": 0}), 200

    upsert_sql = """
        INSERT INTO weather_documents
            (id, location, source_type, headline, narrative_text,
             issued_at, payload, synced_at)
        VALUES %s
        ON CONFLICT (id) DO UPDATE
            SET narrative_text = EXCLUDED.narrative_text,
                payload        = EXCLUDED.payload,
                synced_at      = now();
    """

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
            execute_values(cur, upsert_sql, rows)
        conn.commit()

    return jsonify({"synced": len(rows)}), 200


@app.route("/weather/search", methods=["POST"])
def weather_search():
    body = request.get_json(silent=True) or {}
    query = body.get("query")
    top_k = body.get("top_k", 5)

    if not isinstance(query, str) or not query.strip():
        return jsonify({"error": "Provide a non-blank 'query'."}), 400

    try:
        top_k = int(top_k)
    except (TypeError, ValueError):
        return jsonify({"error": "'top_k' must be an integer."}), 400
    if top_k < 1 or top_k > 20:
        return jsonify({"error": "'top_k' must be between 1 and 20."}), 400

    query_embedding = _embed(query.strip())

    search_sql = """
        SELECT d.id, d.location, d.headline, d.narrative_text, e.chunk_text,
               1 - (e.embedding <=> %s::vector) AS similarity
        FROM weather_embeddings e
        JOIN weather_documents d ON d.id = e.document_id
        ORDER BY e.embedding <=> %s::vector
        LIMIT %s;
    """

    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT count(*) AS n FROM weather_embeddings;")
            if cur.fetchone()["n"] == 0:
                return (
                    jsonify(
                        {
                            "error": "weather_embeddings is empty. Run the sync "
                            "endpoint and the ingest script first."
                        }
                    ),
                    409,
                )
            cur.execute(search_sql, (query_embedding, query_embedding, top_k))
            matches = cur.fetchall()

    results = [
        {
            "location": m["location"],
            "headline": m["headline"],
            "chunk_text": m["chunk_text"],
            "similarity": float(m["similarity"]),
        }
        for m in matches
    ]
    return jsonify({"results": results}), 200


@app.route("/health", methods=["GET"])
def health():
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT count(*) AS n FROM weather_documents;")
            documents = cur.fetchone()["n"]
            cur.execute("SELECT count(*) AS n FROM weather_embeddings;")
            embeddings = cur.fetchone()["n"]

    return jsonify({"weather_documents": documents, "weather_embeddings": embeddings}), 200


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "8080"))
    app.run(host="0.0.0.0", port=port)
