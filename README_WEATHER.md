# Weather RAG on Databricks Apps + Lakebase

A small Flask app that does retrieval-augmented search over US weather data.
Weather documents come from the National Weather Service, are embedded with a
sentence-transformers model, and are stored in Lakebase (managed Postgres 17
with `pgvector`). Search embeds the query and ranks chunks by cosine
similarity using an HNSW index.

## Why the National Weather Service (NWS)?

- **No API key.** `api.weather.gov` is free and requires only a `User-Agent`
  header with a contact email. That keeps this repo secret-free, which matters
  when the whole point is a clean Databricks Apps + Lakebase example.
- **Structured, normalizable data.** NWS exposes both active *alerts* and
  multi-period *forecasts* as GeoJSON, so both collapse cleanly into one
  document shape.
- **Public-domain text.** US government output has no licensing friction for a
  learning/demo project.

Alternatives considered: OpenWeatherMap and Tomorrow.io are richer but require
API keys and have restrictive free tiers; scraping HTML forecast pages is
brittle. NWS wins on "zero credentials, stable JSON, good enough coverage."

The trade-off: NWS only covers the United States, and the grid lookup is a
two-hop dance (`/points` then `/gridpoints`), which is why grid IDs are cached
in-process.

## Schema decisions

Two tables (see `schema.sql`):

- **`weather_documents`** — one row per alert or forecast period. `source_type`
  is `CHECK`-constrained to `('alert','forecast')`. The full upstream JSON is
  kept in a `payload JSONB` column so nothing is lost in normalization.
- **`weather_embeddings`** — one row per chunk, with a `vector(384)` column, a
  foreign key back to the document (`ON DELETE CASCADE`), and a
  `UNIQUE (document_id, chunk_index)` constraint so re-embedding is idempotent.

Indexing: an **HNSW** index with `vector_cosine_ops` matches the `<=>` cosine
operator used at query time. HNSW gives fast approximate nearest-neighbor
search without needing to pick a list count up front (as IVFFlat would).

### Chunking parameters

`CHUNK_SIZE = 800`, `CHUNK_OVERLAP = 100` characters, sliding window.

**Honest note:** most NWS narrative text is short — an alert description or a
single `detailedForecast` period is typically well under 800 characters — so
chunking usually produces exactly **one** chunk per document and the sliding
window rarely triggers. The windowing is there for the occasional long,
multi-paragraph alert (e.g. a detailed severe-weather statement). Keeping the
logic in place means the pipeline behaves correctly if a longer source is added
later, at essentially no cost for the common short case.

## Embedding model

`sentence-transformers/all-MiniLM-L6-v2` — a small, fast, widely-used model that
outputs **384-dimensional** vectors (hence `vector(384)` in the schema). It's a
good default for semantic search on short English text and runs comfortably on
CPU. The model is loaded **once** at module import in both `app.py` and the
ingest script, never per request.

The *same* model is used for both ingestion and query embedding — this is
mandatory. Mixing models (or dimensions) would make similarities meaningless.

## End-to-end run order

1. **Schema** — run `schema.sql` manually in the Databricks SQL editor. This
   enables `pgvector`, creates both tables, and builds the HNSW index.
2. **Sync** — `POST /weather/sync` with a body like
   `{"locations": ["Chicago, IL", "Austin, TX"], "limit": 50}`. This fetches
   alerts + forecasts, normalizes them, and upserts into `weather_documents`.
3. **Embed** — run `python scripts/ingest_weather_embeddings.py`. It selects
   documents that have no embeddings yet, chunks + embeds them, and batch-inserts
   into `weather_embeddings`.
4. **Search** — `POST /weather/search` with
   `{"query": "flash flood risk this weekend", "top_k": 5}`.

Check `GET /health` at any point to see row counts for both tables and confirm a
deploy landed.

### Configuration before deploy

- `app.yaml` — set `ENDPOINT_NAME` to your Lakebase endpoint resource name.
- `weather_client.py` — set `CONTACT_EMAIL` (NWS returns **403** without a real
  contact email in the `User-Agent`).
- Bind a Lakebase resource to the app. Databricks Apps injects `PGHOST`,
  `PGDATABASE`, `PGUSER`, and `PGPORT` after the first deploy with the resource
  bound. The database password is **not** injected — `lakebase.py` mints a
  short-lived OAuth token per connection.

## How the connection works (and why)

`lakebase.py` builds a fresh psycopg2 connection per use. Databricks Apps
injects the `PG*` variables, but **not** a password: instead the app calls
`WorkspaceClient().postgres.generate_database_credential(...)` to mint a
short-lived OAuth token. Those tokens expire after ~1 hour, so a token is
generated **per connection** rather than cached at module scope (caching would
pass tests and then fail silently an hour later). TLS (`sslmode=require`) is
mandatory on Lakebase.

## Known limitations

- **US-only.** NWS covers the United States; the city gazetteer in
  `weather_client.py` is a small hardcoded set of "City, ST" → lat/lon entries.
  Add cities there to extend coverage.
- **Alerts are point-in-time.** `GET /alerts/active?area={ST}` returns whatever
  is active *now*. Sync captures a snapshot; expired alerts stay in the table
  until you clean them up.
- **Chunking rarely triggers** (see above), so the vector index is effectively
  document-level for most rows.
- **Approximate search.** HNSW is approximate nearest-neighbor; results are
  ranked well but not guaranteed to be the exact top-k.
- **No auth on the endpoints.** This is a learning example; the Flask routes
  have no authentication or rate limiting of their own.
- **No connection pooling.** A new connection (and OAuth token) is created per
  request for clarity over throughput.
- **Sync `limit` is a coarse cap** across all requested locations combined, not
  a per-location limit.
