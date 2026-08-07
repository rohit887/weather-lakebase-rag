# Deploying to Databricks Apps + Lakebase

This walks the app from zero to a working `sync → embed → search` pipeline on
Databricks Apps, backed by your Lakebase endpoint
`projects/weather-lakebase-rag/branches/production/endpoints/primary`.

Steps that must happen in the Databricks UI or SQL editor are marked **(UI)** /
**(SQL)**; everything else is CLI you run from this repo.

---

## 0. Prerequisites (local, one time)

```bash
# Databricks CLI (macOS)
brew tap databricks/tap && brew install databricks
databricks -v

# Authenticate to your workspace
databricks auth login --host https://<your-workspace>.cloud.databricks.com
```

---

## 1. Create the app (UI)

1. **Compute → Apps → Create app**, pick the **Flask** template, name it
   `weather-lakebase-rag`.
2. Open the app's **Environment** tab and copy its **`DATABRICKS_CLIENT_ID`**
   (a UUID). This is the app's service principal — it becomes the Postgres
   username (`PGUSER`).

## 2. Get the Lakebase connection values (UI)

Your Lakebase project/branch/endpoint already exist. From the Lakebase project:

- **Connect** modal → **Parameters only** → copy the **host**. Database is
  `databricks_postgres`, port `5432`.
- **Computes** tab → **Get ID** → confirm the endpoint resource name matches
  `projects/weather-lakebase-rag/branches/production/endpoints/primary`.

## 3. Grant the service principal Postgres access (SQL)

In the Lakebase **SQL Editor**, run this once. Replace `<CLIENT_ID>` with the
UUID from step 1. This enables OAuth login for the app and lets it read/write
our tables.

```sql
-- OAuth: let the app's service principal authenticate with a minted token
CREATE EXTENSION IF NOT EXISTS databricks_auth;
SELECT databricks_create_role('<CLIENT_ID>', 'service_principal');

GRANT CONNECT ON DATABASE databricks_postgres TO "<CLIENT_ID>";
GRANT USAGE, CREATE ON SCHEMA public TO "<CLIENT_ID>";
```

## 4. Create the schema (SQL)

Still in the SQL Editor, run the contents of [`schema.sql`](./schema.sql) as the
**project owner** (needs rights to `CREATE EXTENSION vector`). It creates both
tables and the HNSW index.

Then grant the app access to the objects it created — the service principal
needs table DML **and** `USAGE` on the `BIGSERIAL` sequence behind
`weather_embeddings.id`:

```sql
GRANT SELECT, INSERT, UPDATE, DELETE ON weather_documents  TO "<CLIENT_ID>";
GRANT SELECT, INSERT, UPDATE, DELETE ON weather_embeddings TO "<CLIENT_ID>";
GRANT USAGE, SELECT ON SEQUENCE weather_embeddings_id_seq  TO "<CLIENT_ID>";
```

## 5. Fill in `app.yaml`

Edit [`app.yaml`](./app.yaml) placeholders with the values from steps 1–2:

- `PGHOST` → the Lakebase host
- `PGUSER` → the app's `DATABRICKS_CLIENT_ID`
- `PGDATABASE` (`databricks_postgres`), `PGPORT` (`5432`), `PGSSLMODE`
  (`require`), and `ENDPOINT_NAME` are already set.

The DB password is intentionally absent — `lakebase.py` mints a short-lived
OAuth token per connection.

## 6. Deploy

```bash
# Sync this repo into your workspace
databricks sync . /Workspace/Users/rohit885@gmail.com/weather-lakebase-rag

# Deploy that source into the app
databricks apps deploy weather-lakebase-rag \
  --source-code-path /Workspace/Users/rohit885@gmail.com/weather-lakebase-rag
```

Wait ~2–3 minutes, then grab the app URL from the Apps page. Confirm the deploy
landed:

```bash
curl https://<app-url>/health
# {"weather_documents": 0, "weather_embeddings": 0}
```

## 7. Run the pipeline (against the deployed app)

```bash
APP=https://<app-url>

# a) Sync NWS alerts + forecasts into weather_documents
curl -sX POST $APP/weather/sync \
  -H 'Content-Type: application/json' \
  -d '{"locations": ["Chicago, IL", "Austin, TX"], "limit": 50}'
# {"synced": 42}

# b) Embed everything not yet embedded (runs the ingest logic in-app)
curl -sX POST $APP/weather/embed
# {"documents_processed": 42, "chunks_inserted": 42}

# c) Semantic search
curl -sX POST $APP/weather/search \
  -H 'Content-Type: application/json' \
  -d '{"query": "flash flood risk this weekend", "top_k": 5}'
```

`GET /health` again should now show non-zero counts for both tables.

---

## Notes

- **Why an `/weather/embed` endpoint?** Databricks Apps runs only the Flask
  server (`python app.py`) — there's no shell to invoke
  `scripts/ingest_weather_embeddings.py` inside the container. The endpoint runs
  the exact same `run_ingest()` and reuses the already-loaded model. The
  standalone script still works locally or as a Databricks Job.
- **First `/weather/embed` is slow.** The app downloads the
  `all-MiniLM-L6-v2` weights on first model use.
- **Re-running is safe.** Sync upserts on `id`; embed skips documents that
  already have rows (`WHERE e.id IS NULL` + `ON CONFLICT DO NOTHING`).
- **Local run alternative.** To run from your laptop instead: `databricks auth
  login`, `export` the same `PG*` vars (using your email as `PGUSER` for local
  testing) and `ENDPOINT_NAME`, `pip install -r requirements.txt`, then
  `python scripts/ingest_weather_embeddings.py`.
