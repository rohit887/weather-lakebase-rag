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
from scripts.ingest_weather_embeddings import run_ingest
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


@app.route("/weather/embed", methods=["POST"])
def weather_embed():
    """Embed any documents that don't have embeddings yet.

    Same logic as `python scripts/ingest_weather_embeddings.py`, exposed as an
    endpoint because Databricks Apps runs only the Flask server (no shell to
    invoke the script). Reuses the already-loaded module-level model.
    """
    stats = run_ingest(model=_model)
    return jsonify(stats), 200


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


@app.route("/", methods=["GET"])
def index():
    """Interactive web interface for the Weather RAG API."""
    html = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Weather RAG Search</title>
        <style>
            body { font-family: Arial, sans-serif; max-width: 900px; margin: 40px auto; padding: 20px; background: #f5f5f5; }
            h1 { color: #FF3621; }
            .card { background: white; padding: 20px; margin: 20px 0; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
            .form-group { margin: 15px 0; }
            label { display: block; margin-bottom: 5px; font-weight: bold; }
            input, textarea { width: 100%; padding: 8px; border: 1px solid #ddd; border-radius: 4px; box-sizing: border-box; }
            button { background: #FF3621; color: white; padding: 10px 20px; border: none; border-radius: 4px; cursor: pointer; font-size: 14px; }
            button:hover { background: #E02E1A; }
            .result { margin-top: 15px; padding: 15px; background: #f9f9f9; border-left: 4px solid #FF3621; border-radius: 4px; }
            .success { border-left-color: #4CAF50; }
            .error { border-left-color: #f44336; }
            .loading { color: #666; font-style: italic; }
            .match { margin: 10px 0; padding: 10px; background: white; border-radius: 4px; }
            .similarity { color: #FF3621; font-weight: bold; }
        </style>
    </head>
    <body>
        <h1>🌤️ Weather RAG Search API</h1>
        <p>Retrieval-augmented search over NWS weather data on Lakebase</p>
        
        <div class="card">
            <h2>📊 System Health</h2>
            <button onclick="checkHealth()">Check Database Status</button>
            <div id="health-result"></div>
        </div>
        
        <div class="card">
            <h2>🔄 Sync Weather Data</h2>
            <div class="form-group">
                <label>Locations (comma-separated, e.g., "Seattle,WA", "Portland,OR"):</label>
                <input type="text" id="sync-locations" placeholder="Seattle,WA, Portland,OR" value="Seattle,WA">
            </div>
            <div class="form-group">
                <label>Limit (max documents):</label>
                <input type="number" id="sync-limit" value="50" min="1">
            </div>
            <button onclick="syncWeather()">Sync Weather Data</button>
            <div id="sync-result"></div>
        </div>

        <div class="card">
            <h2>🧠 Embed Documents</h2>
            <p>Turn synced documents into vectors. Run this after syncing and before searching.</p>
            <button onclick="embedWeather()">Embed Documents</button>
            <div id="embed-result"></div>
        </div>

        <div class="card">
            <h2>🔍 Search Weather Data</h2>
            <div class="form-group">
                <label>Search Query:</label>
                <input type="text" id="search-query" placeholder="What's the weather forecast for Seattle?">
            </div>
            <div class="form-group">
                <label>Top Results:</label>
                <input type="number" id="search-top-k" value="5" min="1" max="20">
            </div>
            <button onclick="searchWeather()">Search</button>
            <div id="search-result"></div>
        </div>
        
        <script>
            async function checkHealth() {
                const resultDiv = document.getElementById('health-result');
                resultDiv.innerHTML = '<div class="result loading">Checking database status...</div>';
                try {
                    const response = await fetch('/health');
                    const data = await response.json();
                    resultDiv.innerHTML = `
                        <div class="result success">
                            <strong>Database Status:</strong><br>
                            Weather Documents: ${data.weather_documents}<br>
                            Weather Embeddings: ${data.weather_embeddings}
                        </div>
                    `;
                } catch (error) {
                    resultDiv.innerHTML = `<div class="result error">Error: ${error.message}</div>`;
                }
            }
            
            async function syncWeather() {
                const resultDiv = document.getElementById('sync-result');
                const locations = document.getElementById('sync-locations').value.split(',').map(s => s.trim()).filter(s => s);
                const limit = parseInt(document.getElementById('sync-limit').value);
                
                if (locations.length === 0) {
                    resultDiv.innerHTML = '<div class="result error">Please enter at least one location</div>';
                    return;
                }
                
                resultDiv.innerHTML = '<div class="result loading">Syncing weather data...</div>';
                try {
                    const response = await fetch('/weather/sync', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({ locations, limit })
                    });
                    const data = await response.json();
                    if (response.ok) {
                        resultDiv.innerHTML = `<div class="result success">Successfully synced ${data.synced} documents!</div>`;
                    } else {
                        resultDiv.innerHTML = `<div class="result error">Error: ${data.error || 'Unknown error'}</div>`;
                    }
                } catch (error) {
                    resultDiv.innerHTML = `<div class="result error">Error: ${error.message}</div>`;
                }
            }
            
            async function embedWeather() {
                const resultDiv = document.getElementById('embed-result');
                resultDiv.innerHTML = '<div class="result loading">Embedding documents (first run downloads the model, this can take a minute)...</div>';
                try {
                    const response = await fetch('/weather/embed', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' }
                    });
                    const data = await response.json();
                    if (response.ok) {
                        resultDiv.innerHTML = `<div class="result success">Embedded ${data.documents_processed} documents into ${data.chunks_inserted} chunks!</div>`;
                    } else {
                        resultDiv.innerHTML = `<div class="result error">Error: ${data.error || 'Unknown error'}</div>`;
                    }
                } catch (error) {
                    resultDiv.innerHTML = `<div class="result error">Error: ${error.message}</div>`;
                }
            }

            async function searchWeather() {
                const resultDiv = document.getElementById('search-result');
                const query = document.getElementById('search-query').value.trim();
                const top_k = parseInt(document.getElementById('search-top-k').value);
                
                if (!query) {
                    resultDiv.innerHTML = '<div class="result error">Please enter a search query</div>';
                    return;
                }
                
                resultDiv.innerHTML = '<div class="result loading">Searching weather data...</div>';
                try {
                    const response = await fetch('/weather/search', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({ query, top_k })
                    });
                    const data = await response.json();
                    if (response.ok) {
                        if (data.results.length === 0) {
                            resultDiv.innerHTML = '<div class="result">No results found</div>';
                        } else {
                            let html = '<div class="result success"><strong>Search Results:</strong></div>';
                            data.results.forEach((match, i) => {
                                html += `
                                    <div class="match">
                                        <strong>${i + 1}. ${match.location}</strong> 
                                        <span class="similarity">(${(match.similarity * 100).toFixed(1)}% match)</span><br>
                                        <em>${match.headline}</em><br>
                                        ${match.chunk_text}
                                    </div>
                                `;
                            });
                            resultDiv.innerHTML = html;
                        }
                    } else {
                        resultDiv.innerHTML = `<div class="result error">Error: ${data.error || 'Unknown error'}</div>`;
                    }
                } catch (error) {
                    resultDiv.innerHTML = `<div class="result error">Error: ${error.message}</div>`;
                }
            }
        </script>
    </body>
    </html>
    """
    return html, 200


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
