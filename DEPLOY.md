# Deploying to Databricks Apps + Lakebase

This walks the app from zero to a working `sync → embed → search` pipeline on
Databricks Apps, backed by your Lakebase endpoint
`projects/weather-lakebase-rag/branches/production/endpoints/primary`.

Steps that must happen in the Databricks UI or SQL editor are marked **(UI)** /
**(SQL)**; everything else is CLI you run from this repo.

---

## Authentication: single connection-URL secret

This app authenticates to Postgres with a **native role + password**
(`student-weather`), passed as a full connection URL that lives in **one
Databricks secret**. `lakebase.py` fetches it at runtime via the SDK
(`WorkspaceClient().secrets.get_secret`) and base64-decodes it once, then hands
the whole URL to psycopg2. `app.yaml` only carries the secret's location:

| Var | Value |
|-----|-------|
| `LAKEBASE_SECRET_SCOPE` | `database` |
| `LAKEBASE_SECRET_KEY` | `lakebase-url` |

The secret holds a URL like:
`postgresql://student-weather:<password>@ep-...cloud.databricks.com:5432/databricks_postgres?sslmode=require`

Why fetch in code instead of an `app.yaml` `secrets:`/`valueFrom` resource? The
`valueFrom` path needs fiddly per-service-principal ACLs; reading via the SDK +
granting the secret scope to the **`users`** group is simpler and more reliable.

---

## 0. Prerequisites (local, one time)

```bash
brew tap databricks/tap && brew install databricks   # macOS
databricks auth login --host https://<your-workspace>.cloud.databricks.com
```

## 1. Create the schema (SQL)

Run [`schema.sql`](./schema.sql) in the Lakebase **SQL Editor** as the project
owner (needs rights to `CREATE EXTENSION vector`). Creates both tables + the
HNSW index. *(Already done for this project — the tables exist.)*

## 2. Grant the `student-weather` role access (SQL)

The native role needs DML on both tables plus `USAGE` on the `BIGSERIAL`
sequence. Run once as the table owner:

```sql
GRANT CONNECT ON DATABASE databricks_postgres TO "student-weather";
GRANT USAGE ON SCHEMA public TO "student-weather";
GRANT SELECT, INSERT, UPDATE, DELETE ON weather_documents  TO "student-weather";
GRANT SELECT, INSERT, UPDATE, DELETE ON weather_embeddings TO "student-weather";
GRANT USAGE, SELECT ON SEQUENCE weather_embeddings_id_seq  TO "student-weather";
```

*(Verified working — a run inserted 93 documents / 99 embeddings.)*

## 3. Store the connection URL as a Databricks secret

Store the **plain** URL — Databricks base64-encodes it at rest, and
`lakebase.py` decodes it once (do **not** pre-encode it yourself, or you get
double-encoding). Then grant READ to the `users` group so the app's service
principal can read it:

```bash
databricks secrets create-scope database
databricks secrets put-secret database lakebase-url \
  --string-value 'postgresql://student-weather:<password>@ep-round-butterfly-d8iwgyc5.database.us-east-2.cloud.databricks.com:5432/databricks_postgres?sslmode=require'
databricks secrets put-acl database users READ
```

> Never commit the URL/password. It lives only in the secret scope.

## 4. Deploy

Create the app (Apps UI → Create app → Flask template, name
`weather-lakebase-rag`), then:

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

## 5. Run the pipeline (against the deployed app)

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

## Optional: RAG summary + scheduled sync

- **LLM summary (`GET /weather/search`).** `llm.py`'s `DEFAULT_ENDPOINT` is
  `databricks-llama-4-maverick`; override with `LLM_ENDPOINT_NAME` if needed. No
  API key — the deployed app's service principal authenticates to the endpoint.
  Locally it needs `databricks auth login`; without it, search still works and
  `summary` is `null`.
- **Scheduled re-sync.** Create a **Databricks Job** with a Python-script task
  pointing at `scripts/scheduled_sync.py`, a cron schedule (quartz, e.g.
  `0 0/15 * * * ?` for every 15 min). The job's identity needs READ on the
  `database` secret scope (granted to the `users` group in step 3). Optionally
  set `SYNC_LOCATIONS` / `SYNC_LIMIT`.

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
- **Local / notebook runs.** `lakebase.py` reads the connection URL from the
  secret via the SDK, so any local run needs workspace auth first:
  ```bash
  pip install -r requirements.txt
  databricks auth login --host https://<your-workspace>.cloud.databricks.com
  python scripts/ingest_weather_embeddings.py   # embed step
  ```
  In a Databricks notebook auth is automatic — see
  `notebooks/rag_summary_demo.py`. (The `database`/`lakebase-url` secret must be
  readable by your identity.)
