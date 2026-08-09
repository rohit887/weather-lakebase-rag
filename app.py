"""Flask app: retrieval-augmented search over NWS weather data on Lakebase.

Endpoints:
  POST /weather/sync   - fetch + upsert weather documents
  POST /weather/embed  - embed unembedded documents
  POST /weather/search - semantic search over embedded chunks
  GET  /weather/search - same search + an LLM natural-language summary (RAG)
  GET  /health         - row counts for both tables
"""

from flask import Flask, jsonify, request
from sentence_transformers import SentenceTransformer

import llm
import pipeline
from lakebase import get_connection
from pipeline import VALID_SOURCE_TYPES
from scripts.ingest_weather_embeddings import run_ingest

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

    synced = pipeline.sync_documents(locations, limit)
    return jsonify({"synced": synced}), 200


@app.route("/weather/embed", methods=["POST"])
def weather_embed():
    """Embed any documents that don't have embeddings yet.

    Same logic as `python scripts/ingest_weather_embeddings.py`, exposed as an
    endpoint because Databricks Apps runs only the Flask server (no shell to
    invoke the script). Reuses the already-loaded module-level model.
    """
    stats = run_ingest(model=_model)
    return jsonify(stats), 200


def _validate_search_params(query, top_k, source_type):
    """Validate shared search params. Returns (clean_dict, None) or
    (None, (error_json, status))."""
    if not isinstance(query, str) or not query.strip():
        return None, ({"error": "Provide a non-blank 'query'."}, 400)

    try:
        top_k = int(top_k)
    except (TypeError, ValueError):
        return None, ({"error": "'top_k' must be an integer."}, 400)
    if top_k < 1 or top_k > 20:
        return None, ({"error": "'top_k' must be between 1 and 20."}, 400)

    if source_type is not None and source_type not in VALID_SOURCE_TYPES:
        return None, (
            {"error": "'source_type' must be 'alert' or 'forecast'."},
            400,
        )

    return {"query": query.strip(), "top_k": top_k, "source_type": source_type}, None


def _run_search(query, top_k, source_type):
    """Embed the query and return (results, None) or (None, (error_json, status))."""
    if pipeline.embedding_count() == 0:
        return None, (
            {
                "error": "weather_embeddings is empty. Run /weather/sync then "
                "/weather/embed first."
            },
            409,
        )
    query_embedding = _embed(query)
    return pipeline.vector_search(query_embedding, top_k, source_type), None


@app.route("/weather/search", methods=["POST"])
def weather_search():
    body = request.get_json(silent=True) or {}
    clean, err = _validate_search_params(
        body.get("query"), body.get("top_k", 5), body.get("source_type")
    )
    if err:
        return jsonify(err[0]), err[1]

    results, err = _run_search(**clean)
    if err:
        return jsonify(err[0]), err[1]
    return jsonify({"results": results}), 200


@app.route("/weather/search", methods=["GET"])
def weather_search_rag():
    """Same vector search as POST, plus an LLM natural-language summary (RAG).

    Query params: query (required), top_k (default 5), source_type (optional).
    The summary degrades gracefully: if the model endpoint is unreachable, the
    vector results still return with summary=null and a summary_error note.
    """
    clean, err = _validate_search_params(
        request.args.get("query"),
        request.args.get("top_k", 5),
        request.args.get("source_type"),
    )
    if err:
        return jsonify(err[0]), err[1]

    results, err = _run_search(**clean)
    if err:
        return jsonify(err[0]), err[1]

    summary, summary_error = llm.summarize(clean["query"], results)
    return (
        jsonify(
            {"results": results, "summary": summary, "summary_error": summary_error}
        ),
        200,
    )


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
            <div class="form-group">
                <label>Source type filter:</label>
                <select id="search-source-type">
                    <option value="">All</option>
                    <option value="alert">Alerts only</option>
                    <option value="forecast">Forecasts only</option>
                </select>
            </div>
            <div class="form-group">
                <label><input type="checkbox" id="search-ai" style="width:auto"> AI summary (RAG)</label>
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
                const sourceType = document.getElementById('search-source-type').value;
                const useAi = document.getElementById('search-ai').checked;

                if (!query) {
                    resultDiv.innerHTML = '<div class="result error">Please enter a search query</div>';
                    return;
                }

                resultDiv.innerHTML = '<div class="result loading">Searching weather data...</div>';
                try {
                    let response;
                    if (useAi) {
                        // RAG variant: GET /weather/search returns an LLM summary too.
                        const params = new URLSearchParams({ query, top_k });
                        if (sourceType) params.set('source_type', sourceType);
                        response = await fetch('/weather/search?' + params.toString());
                    } else {
                        const payload = { query, top_k };
                        if (sourceType) payload.source_type = sourceType;
                        response = await fetch('/weather/search', {
                            method: 'POST',
                            headers: { 'Content-Type': 'application/json' },
                            body: JSON.stringify(payload)
                        });
                    }
                    const data = await response.json();
                    if (response.ok) {
                        let html = '';
                        if (data.summary) {
                            html += `<div class="result success"><strong>🤖 AI Summary:</strong><br>${data.summary}</div>`;
                        } else if (useAi && data.summary_error) {
                            html += `<div class="result error"><strong>AI summary unavailable:</strong> ${data.summary_error}</div>`;
                        }
                        if (data.results.length === 0) {
                            html += '<div class="result">No results found</div>';
                        } else {
                            html += '<div class="result success"><strong>Search Results:</strong></div>';
                            data.results.forEach((match, i) => {
                                html += `
                                    <div class="match">
                                        <strong>${i + 1}. ${match.location}</strong>
                                        <span class="similarity">(${(match.similarity * 100).toFixed(1)}% match)</span>
                                        <em>[${match.source_type}]</em><br>
                                        <em>${match.headline}</em><br>
                                        ${match.chunk_text}
                                    </div>
                                `;
                            });
                        }
                        resultDiv.innerHTML = html;
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
