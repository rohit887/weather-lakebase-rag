# Deploying to Databricks Apps + Lakebase

This walks the app from zero to a working `sync → embed → search` pipeline on
Databricks Apps, backed by your Lakebase endpoint
`projects/weather-lakebase-rag/branches/production/endpoints/primary`.

Steps that must happen in the Databricks UI or SQL editor are marked **(UI)** /
**(SQL)**; everything else is CLI you run from this repo.

---

## Authentication: native password role

This app authenticates to Postgres with a **native role + password**
(`student-weather`), not OAuth token minting. `lakebase.py` uses `PGPASSWORD`
if it's set and only falls back to minting an OAuth token when it isn't — so
the Databricks SDK/auth is not needed on the password path.

Connection values (already filled into [`app.yaml`](./app.yaml)):

| Var | Value |
|-----|-------|
| `PGHOST` | `ep-round-butterfly-d8iwgyc5.database.us-east-2.cloud.databricks.com` |
| `PGDATABASE` | `databricks_postgres` |
| `PGUSER` | `student-weather` |
| `PGPORT` | `5432` / `PGSSLMODE` `require` |
| `PGPASSWORD` | **from a secret** — never inline |

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

*(Verified working — a local run inserted 60 documents / 62 embeddings.)*

## 3. Store the password as a Databricks secret

The `PGPASSWORD` value in `app.yaml` is `valueFrom: "pg-password"`, which
resolves to a **secret resource** on the app. Create the secret, then bind it
to the app as a resource named `pg-password` (Apps UI → Edit → Resources →
Secret → scope `weather`, key `pg-password`):

```bash
databricks secrets create-scope weather
databricks secrets put-secret weather pg-password   # paste the role password
```

> Never commit the password. It lives only in the secret scope (deployed) or
> your shell env (local).

## 4. Deploy

Create the app (Apps UI → Create app → Flask template, name
`weather-lakebase-rag`), attach the `pg-password` secret resource from step 3,
then:

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
- **Local run alternative (no Databricks auth needed).** With the password
  method you can populate the tables straight from your laptop:
  ```bash
  pip install -r requirements.txt
  export PGHOST="ep-round-butterfly-d8iwgyc5.database.us-east-2.cloud.databricks.com"
  export PGDATABASE="databricks_postgres" PGUSER="student-weather" PGPORT="5432"
  export PGPASSWORD="<the role password>"    # do NOT hard-code in a file
  python scripts/ingest_weather_embeddings.py   # embed step
  ```
  (The sync + search steps run the same way via the endpoints, or Flask's
  `app.test_client()`.)
