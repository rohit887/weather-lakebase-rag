# Lakebase + Databricks App — Secrets, Connection & Permissions Runbook

Reusable reference for setting up / debugging the database connection, secrets,
and permissions for this app. Safe to paste into a new Claude Code session as
context. **Contains NO live password** — the real password lives only in the
Databricks secret; use the placeholder `<PASSWORD>` here.

---

## 1. Known-good facts for THIS project

| Thing | Value |
|-------|-------|
| Lakebase host | `ep-round-butterfly-d8iwgyc5.database.us-east-2.cloud.databricks.com` |
| Database | `databricks_postgres` |
| Postgres role (user) | `student-weather` (native role, static password) |
| Port / TLS | `5432` / `sslmode=require` (TLS mandatory) |
| Endpoint resource name | `projects/weather-lakebase-rag/branches/production/endpoints/primary` (short id `ep-round-butterfly-d8iwgyc5`) |
| Secret scope | `database` |
| Secret key | `lakebase-url` |
| App service principal | `weather-rag-app` (app `app-4j4unu`), UUID `b8079826-18da-41b6-af3a-1db0acc722e0` |
| LLM serving endpoint | `databricks-llama-4-maverick` |
| GitHub repo | https://github.com/rohit887/weather-lakebase-rag |

## 2. The connection URL (stored in the secret)

A single standard Postgres URL — the app reads this from the secret, nothing
else:

```
postgresql://student-weather:<PASSWORD>@ep-round-butterfly-d8iwgyc5.database.us-east-2.cloud.databricks.com:5432/databricks_postgres?sslmode=require
```

## 3. How auth works (the pattern)

- `app.yaml` carries only the secret's **location**:
  ```yaml
  env:
    - name: LAKEBASE_SECRET_SCOPE
      value: "database"
    - name: LAKEBASE_SECRET_KEY
      value: "lakebase-url"
  ```
- `lakebase.py` fetches + decodes the URL at runtime and hands it to psycopg2:
  ```python
  secret = WorkspaceClient().secrets.get_secret(scope=SCOPE, key=KEY)
  url = base64.b64decode(secret.value).decode("utf-8")   # decode ONCE
  psycopg2.connect(url, cursor_factory=RealDictCursor)
  ```
- Auth to the secret/model comes from the **app's service principal** when
  deployed, or `databricks auth login` when running locally / in a notebook.

**Why this pattern (not `PG*` env vars or `app.yaml` valueFrom):** the
`valueFrom`/`secrets:` resource route needs brittle per-service-principal ACLs
that were hard to get working. Fetching one URL secret via the SDK + granting
the scope to the `users` group is simpler and reliable.

## 4. Secret setup (create + store PLAIN URL)

Store the URL **plain** — Databricks base64-encodes secrets at rest, and the
code decodes once. Do NOT pre-encode it yourself (that causes double-base64).

```bash
databricks secrets create-scope database      # if it doesn't exist
databricks secrets put-secret database lakebase-url \
  --string-value 'postgresql://student-weather:<PASSWORD>@ep-round-butterfly-d8iwgyc5.database.us-east-2.cloud.databricks.com:5432/databricks_postgres?sslmode=require'
```

## 5. Permissions checklist (the part that bites)

### a) Secret-scope READ → grant to the `users` group (NOT the SP UUID)

The app runs as a service principal with no secret read access by default.
Granting to `users` covers all workspace users + their SPs and is maintainable:

```bash
databricks secrets put-acl database users READ
databricks secrets list-acls database        # verify: users = READ
```

### b) Postgres grants for the `student-weather` role

Run once in the Lakebase SQL editor as the table owner. The role needs table
DML **and** `USAGE` on the `BIGSERIAL` sequence, else inserts fail:

```sql
GRANT CONNECT ON DATABASE databricks_postgres TO "student-weather";
GRANT USAGE ON SCHEMA public TO "student-weather";
GRANT SELECT, INSERT, UPDATE, DELETE ON weather_documents  TO "student-weather";
GRANT SELECT, INSERT, UPDATE, DELETE ON weather_embeddings TO "student-weather";
GRANT USAGE, SELECT ON SEQUENCE weather_embeddings_id_seq  TO "student-weather";
```

### c) Model serving (only for the LLM RAG summary)

The app SP (or your user, locally) needs **CAN QUERY** on the
`databricks-llama-4-maverick` serving endpoint. If the summary comes back
`null` with a `summary_error`, check this permission and the endpoint name.

## 6. Verify BEFORE deploying

**Test secret decoding + connectivity** (locally, after `databricks auth login`,
or in a Databricks notebook where auth is automatic):

```python
import base64
from databricks.sdk import WorkspaceClient
w = WorkspaceClient()
url = base64.b64decode(w.secrets.get_secret(scope="database", key="lakebase-url").value).decode()
assert url.startswith("postgresql://"), f"double-encoded or wrong: {url[:40]!r}"
print("decoded OK:", url.split('@')[1])   # prints host, hides password

import psycopg2
with psycopg2.connect(url) as c, c.cursor() as cur:
    cur.execute("select 1"); print("db OK", cur.fetchone())
```

**On the deployed app**, hit `/health` — JSON row counts = DB path works; an
HTML error page (the `Unexpected token '<'` symptom in the UI) = a DB/secret
problem, so check the app **Logs** tab for the traceback.

## 7. Issues we hit → fixes (so you don't repeat them)

1. **`NameError: name 'os' is not defined`** → `app.py` was missing `import os`.
2. **`secret-scopes.secrets/get permission denied`** for the app SP → grant the
   secret scope READ to the **`users`** group; fetch the secret in code via the
   SDK rather than an `app.yaml` `valueFrom` resource.
3. **psycopg2 chokes on base64 gibberish** → double base64. Store the **plain**
   URL; Databricks encodes once; decode once in code.
4. **UI buttons show `Unexpected token '<'`** → a DB endpoint 500'd and returned
   an HTML error page; the JSON parse failed. Root cause is always one of 1–3
   above. Check `/health` directly and the app Logs.

## 8. Don't commit secrets

The password/URL lives only in the Databricks secret. `app.yaml` and all source
carry only the scope/key. Rotate the `student-weather` password if it ever lands
in a chat/log/file.
