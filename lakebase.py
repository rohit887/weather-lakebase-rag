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

    When running as a Databricks App with a postgres database resource,
    credentials are automatically injected. Check DATABRICKS_POSTGRES_PASSWORD
    first (from the database resource), then fall back to PGPASSWORD (native
    role), then finally OAuth token generation.
    """
    # Database resource provides DATABRICKS_POSTGRES_* variables
    password = os.environ.get("DATABRICKS_POSTGRES_PASSWORD")
    if password:
        return password
    
    # Fallback to manual PGPASSWORD if set
    password = os.environ.get("PGPASSWORD")
    if password:
        return password
    
    # Last resort: generate OAuth token
    return _generate_password()


@contextmanager
def get_connection():
    """Yield a psycopg2 connection built from env vars.
    
    Prioritizes DATABRICKS_POSTGRES_* variables (from database resource)
    over PG* variables (manual configuration). TLS is mandatory on Lakebase.
    """
    # Check database resource variables first, then fallback to manual PG* vars
    host = os.environ.get("DATABRICKS_POSTGRES_HOST") or os.environ.get("PGHOST")
    dbname = os.environ.get("DATABRICKS_POSTGRES_DATABASE") or os.environ.get("PGDATABASE")
    user = os.environ.get("DATABRICKS_POSTGRES_USER") or os.environ.get("PGUSER")
    port = os.environ.get("DATABRICKS_POSTGRES_PORT") or os.environ.get("PGPORT", "5432")
    
    if not host or not dbname or not user:
        raise RuntimeError(
            "Missing required database connection parameters. "
            "Ensure DATABRICKS_POSTGRES_* variables are set (via database resource) "
            "or set PG* variables manually in app.yaml."
        )
    
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
