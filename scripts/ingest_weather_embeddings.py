"""Embed unembedded weather documents into weather_embeddings.

Standalone batch job. Uses the same get_connection() as the Flask app and
psycopg2 only -- Spark JDBC writes are NOT supported against this Lakebase
instance.

Run order: schema.sql -> POST /weather/sync -> this script -> POST /weather/search
"""

import sys
from pathlib import Path

from psycopg2.extras import execute_values
from sentence_transformers import SentenceTransformer

# Allow running as `python scripts/ingest_weather_embeddings.py` from repo root.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from lakebase import get_connection  # noqa: E402

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
CHUNK_SIZE = 800
CHUNK_OVERLAP = 100


def chunk_text(text):
    """Sliding-window chunks of CHUNK_SIZE chars with CHUNK_OVERLAP overlap.

    Most NWS narratives are short enough to yield a single chunk; the window
    only kicks in on unusually long alert descriptions.
    """
    text = text or ""
    if not text:
        return []
    if len(text) <= CHUNK_SIZE:
        return [text]

    step = CHUNK_SIZE - CHUNK_OVERLAP
    chunks = []
    start = 0
    while start < len(text):
        chunks.append(text[start : start + CHUNK_SIZE])
        start += step
    return chunks


def main():
    model = SentenceTransformer(MODEL_NAME)

    select_sql = """
        SELECT d.id, d.narrative_text
        FROM weather_documents d
        LEFT JOIN weather_embeddings e ON e.document_id = d.id
        WHERE e.id IS NULL;
    """

    insert_sql = """
        INSERT INTO weather_embeddings
            (document_id, chunk_index, chunk_text, embedding, model_name)
        VALUES %s
        ON CONFLICT (document_id, chunk_index) DO NOTHING;
    """

    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(select_sql)
            documents = cur.fetchall()

        docs_processed = 0
        chunks_inserted = 0

        for doc in documents:
            document_id = doc["id"]
            chunks = chunk_text(doc["narrative_text"])
            if not chunks:
                continue

            embeddings = model.encode(chunks, normalize_embeddings=False)

            rows = [
                (
                    document_id,
                    index,
                    chunk,
                    embeddings[index].tolist(),
                    MODEL_NAME,
                )
                for index, chunk in enumerate(chunks)
            ]

            # Cast the embedding literal to vector; pass Python lists as-is.
            with conn.cursor() as cur:
                execute_values(
                    cur,
                    insert_sql,
                    rows,
                    template="(%s, %s, %s, %s::vector, %s)",
                )
                chunks_inserted += cur.rowcount
            docs_processed += 1

        conn.commit()

    print(f"Documents processed: {docs_processed}")
    print(f"Chunks inserted:     {chunks_inserted}")


if __name__ == "__main__":
    main()
