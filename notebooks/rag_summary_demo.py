# Databricks notebook source
# MAGIC %md
# MAGIC # Weather RAG summary demo
# MAGIC
# MAGIC Runs the RAG summary (`GET /weather/search` logic) from a notebook. In a
# MAGIC Databricks notebook `WorkspaceClient()` authenticates as you automatically,
# MAGIC so the model-serving call needs **no API key and no login**.
# MAGIC
# MAGIC Easiest setup: clone this GitHub repo into **Databricks Repos** and open this
# MAGIC notebook from there, so `import pipeline` / `import llm` resolve.

# COMMAND ----------

# MAGIC %pip install sentence-transformers psycopg2-binary
# MAGIC # databricks-sdk is already on the Databricks runtime

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %md ## Option A — just test the serving endpoint (no Postgres needed)

# COMMAND ----------

from databricks.sdk import WorkspaceClient
from databricks.sdk.service.serving import ChatMessage, ChatMessageRole

ENDPOINT = "databricks-llama-4-maverick"  # matches llm.DEFAULT_ENDPOINT

w = WorkspaceClient()
resp = w.serving_endpoints.query(
    name=ENDPOINT,
    messages=[ChatMessage(role=ChatMessageRole.USER, content="Say hello in one short sentence.")],
    max_tokens=50,
)
print(resp.choices[0].message.content)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Option B — full RAG over your Lakebase data
# MAGIC
# MAGIC Set the connection env vars. Put the password in a secret (or type it into
# MAGIC the widget below) — never hard-code it in the notebook.

# COMMAND ----------

import os

os.environ["PGHOST"] = "ep-round-butterfly-d8iwgyc5.database.us-east-2.cloud.databricks.com"
os.environ["PGDATABASE"] = "databricks_postgres"
os.environ["PGUSER"] = "student-weather"
os.environ["PGPORT"] = "5432"

# Password: prefer a secret scope; fall back to a widget you type into.
try:
    os.environ["PGPASSWORD"] = dbutils.secrets.get(scope="weather", key="pg-password")
except Exception:
    dbutils.widgets.text("pg_password", "")
    os.environ["PGPASSWORD"] = dbutils.widgets.get("pg_password")

# COMMAND ----------

import sys

# If running from Databricks Repos, add the repo root so sibling modules import.
repo_root = os.path.dirname(os.getcwd()) if os.path.basename(os.getcwd()) == "notebooks" else os.getcwd()
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

import llm
import pipeline
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# COMMAND ----------

query = "flash flood risk this weekend"
top_k = 5
source_type = None  # or "alert" / "forecast"

query_vec = model.encode(query).tolist()
results = pipeline.vector_search(query_vec, top_k, source_type)

print(f"Top {len(results)} matches:")
for r in results:
    print(f"  {r['similarity']:.3f}  [{r['source_type']}]  {r['location']}  |  {r['headline']}")

# COMMAND ----------

summary, error = llm.summarize(query, results)
print("SUMMARY:\n", summary)
print("\nERROR:", error)
