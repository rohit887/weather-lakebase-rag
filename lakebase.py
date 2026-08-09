import os
from contextlib import contextmanager

import psycopg2
from psycopg2.extras import RealDictCursor


def _require_env(name: str) -> str:
    """Return an env var or raise a clear, named error (never a KeyError)."""
    value = os.environ.get(name)
    if not value:
        raise RuntimeError(
            f"Required environment variable {name} is not set. "
            "Set the PG* variables in app.yaml (or export them locally). "
            "PGPASSWORD should come from a Databricks secret, never inline."
        )
    return value


def _generate_password() -> str:
    """Mint a fresh, short-lived OAuth database credential.

    Only used as a FALLBACK when PGPASSWORD is not set. Imported lazily so
    environments using a native password role don't need databricks-sdk.
    """
    endpoint_name = _require_env("ENDPOINT_NAME")
    from databricks.sdk import WorkspaceClient

    w = WorkspaceClient()  # reads auth from the environment; takes no arguments
    if not hasattr(w, "postgres"):
        raise RuntimeError(
            "The installed databricks-sdk has no `postgres` API. Pin "
            "databricks-sdk>=0.125.0 in requirements.txt and redeploy."
        )
    credential = w.postgres.generate_database_credential(endpoint=endpoint_name)
    return credential.token


def _resolve_password() -> str:
    """Password for the connection.

    Prefer a native Postgres role password (PGPASSWORD) -- the method used
    here. If PGPASSWORD is unset, fall back to minting a short-lived OAuth
    token via the Databricks SDK.
    """
    password = os.environ.get("PGPASSWORD")
    if password:
        return password
    return _generate_password()


@contextmanager
def get_connection():
    """Yield a psycopg2 connection built from the PG* env vars. TLS is
    mandatory on Lakebase.
    """
    host = _require_env("PGHOST")
    dbname = _require_env("PGDATABASE")
    user = _require_env("PGUSER")
    port = os.environ.get("PGPORT", "5432")
    password = _resolve_password()

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
