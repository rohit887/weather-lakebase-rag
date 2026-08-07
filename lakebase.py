import os
from contextlib import contextmanager

import psycopg2
from psycopg2.extras import RealDictCursor
from databricks.sdk import WorkspaceClient


def _require_env(name: str) -> str:
    """Return an env var or raise a clear, named error (never a KeyError)."""
    value = os.environ.get(name)
    if not value:
        raise RuntimeError(
            f"Required environment variable {name} is not set. "
            "Databricks Apps injects the PG* variables only after the first "
            "deploy with a Lakebase resource bound, and ENDPOINT_NAME must be "
            "set in app.yaml."
        )
    return value


def _generate_password() -> str:
    """Mint a fresh, short-lived OAuth database credential.

    WorkspaceClient() takes no arguments: it reads auth from the environment
    (the app's service principal when deployed, or your user account locally).
    """
    endpoint_name = _require_env("ENDPOINT_NAME")
    w = WorkspaceClient()
    if not hasattr(w, "postgres"):
        raise RuntimeError(
            "The installed databricks-sdk has no `postgres` API. Pin "
            "databricks-sdk>=0.125.0 in requirements.txt and redeploy."
        )
    credential = w.postgres.generate_database_credential(endpoint=endpoint_name)
    return credential.token


@contextmanager
def get_connection():
    """Yield a psycopg2 connection built from injected PG* env vars plus a
    freshly generated OAuth token. TLS is mandatory on Lakebase.
    """
    host = _require_env("PGHOST")
    dbname = _require_env("PGDATABASE")
    user = _require_env("PGUSER")
    port = os.environ.get("PGPORT", "5432")
    password = _generate_password()

    conn = psycopg2.connect(
        host=host,
        dbname=dbname,
        user=user,
        port=port,
        password=password,
        sslmode="require",       # Lakebase mandates TLS
        cursor_factory=RealDictCursor,
    )
    try:
        yield conn
    finally:
        conn.close()             # psycopg2's `with conn` commits but does NOT close
