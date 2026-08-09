"""Scheduled re-sync: refresh weather docs and embed the new ones.

Designed to run on a cron cadence -- either a Databricks Job with a schedule
(recommended) or a plain system crontab. It reuses the exact same
pipeline.sync_documents() and run_ingest() as the Flask app, so scheduled runs
and manual runs behave identically. Upserts are idempotent (ON CONFLICT), so
re-running every N minutes just refreshes changed alerts.

The database connection comes from lakebase.get_connection(), which reads the
LAKEBASE_URL secret via the Databricks SDK -- so this must run in a context with
workspace auth (a Databricks Job, or locally after `databricks auth login`).

Config via environment:
  SYNC_LOCATIONS        ';'-separated "City, ST" list
                        (default: "Chicago, IL;Austin, TX;Miami, FL;Denver, CO")
  SYNC_LIMIT            max documents per run (default: 50)
  LAKEBASE_SECRET_SCOPE / LAKEBASE_SECRET_KEY  override the secret location
                        (defaults: scope "database", key "lakebase-url")

Databricks Job cron example (every 15 min):
  Task type: Python script -> scripts/scheduled_sync.py
  Schedule (quartz): 0 0/15 * * * ?
  The job's identity needs READ on the secret scope (grant to the "users" group).
"""

import os
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pipeline  # noqa: E402
from scripts.ingest_weather_embeddings import run_ingest  # noqa: E402

DEFAULT_LOCATIONS = "Chicago, IL;Austin, TX;Miami, FL;Denver, CO"


def main():
    raw = os.environ.get("SYNC_LOCATIONS", DEFAULT_LOCATIONS)
    locations = [loc.strip() for loc in raw.split(";") if loc.strip()]
    limit = int(os.environ.get("SYNC_LIMIT", "50"))

    stamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
    print(f"[{stamp}] sync start: {locations} (limit={limit})")

    synced = pipeline.sync_documents(locations, limit)
    print(f"[{stamp}] upserted {synced} documents")

    stats = run_ingest()  # embeds only the newly-unembedded docs
    print(
        f"[{stamp}] embedded {stats['documents_processed']} docs "
        f"into {stats['chunks_inserted']} chunks"
    )


if __name__ == "__main__":
    main()
