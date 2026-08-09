"""Benchmark: query latency WITH vs WITHOUT the HNSW index.

Two modes:

  python scripts/benchmark_hnsw.py
      Benchmark the real weather_embeddings table. Honest note: with only a few
      dozen rows the planner often seq-scans anyway, so the gap is tiny -- HNSW
      pays off at scale.

  python scripts/benchmark_hnsw.py --synthetic 20000
      Build a TEMP table of N random 384-d vectors + an HNSW index and benchmark
      there, so the with/without difference is actually visible. The temp table
      vanishes when the connection closes -- real data is untouched.

"WITHOUT index" is simulated by disabling index scans for the transaction
(SET LOCAL enable_indexscan = off) so the planner must seq-scan + sort.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from lakebase import get_connection  # noqa: E402

RUNS = 5
DIM = 384


def _exec_time_ms(cur, sql, params):
    """Run EXPLAIN ANALYZE and return (execution_ms, top_node_type)."""
    cur.execute("EXPLAIN (ANALYZE, FORMAT JSON) " + sql, params)
    plan = cur.fetchone()["QUERY PLAN"][0]
    return plan["Execution Time"], plan["Plan"]["Node Type"]


def _avg(cur, sql, params, index_on):
    """Average execution time over RUNS, with index scans on or off."""
    cur.execute("SET LOCAL enable_indexscan = %s;", ("on" if index_on else "off",))
    cur.execute("SET LOCAL enable_bitmapscan = %s;", ("on" if index_on else "off",))
    times, node = [], None
    for _ in range(RUNS):
        ms, node = _exec_time_ms(cur, sql, params)
        times.append(ms)
    return sum(times) / len(times), node


def _report(label, with_ms, with_node, without_ms, without_node):
    print(f"\n=== {label} ===")
    print(f"  WITH index    : {with_ms:8.3f} ms   (top node: {with_node})")
    print(f"  WITHOUT index : {without_ms:8.3f} ms   (top node: {without_node})")
    if with_ms > 0:
        print(f"  speedup       : {without_ms / with_ms:6.2f}x")


def benchmark_real():
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    query_vec = model.encode("flash flood risk this weekend").tolist()

    sql = """
        SELECT e.id
        FROM weather_embeddings e
        ORDER BY e.embedding <=> %s::vector
        LIMIT 5;
    """
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT count(*) AS n FROM weather_embeddings;")
            n = cur.fetchone()["n"]
            with_ms, with_node = _avg(cur, sql, (query_vec,), True)
            without_ms, without_node = _avg(cur, sql, (query_vec,), False)
        conn.rollback()
    _report(f"real weather_embeddings ({n} rows)", with_ms, with_node, without_ms, without_node)
    if n < 1000:
        print("  note: tiny table -- planner may ignore the index; run "
              "--synthetic N to see the real speedup.")


def benchmark_synthetic(n):
    import random

    def rand_vec():
        return "[" + ",".join(f"{random.random():.6f}" for _ in range(DIM)) + "]"

    query_vec = rand_vec()
    with get_connection() as conn:
        with conn.cursor() as cur:
            print(f"building TEMP table of {n} random {DIM}-d vectors...")
            cur.execute("CREATE TEMP TABLE bench_vec (id BIGSERIAL, embedding vector(%s));" % DIM)
            batch = 2000
            for start in range(0, n, batch):
                vals = [(rand_vec(),) for _ in range(min(batch, n - start))]
                from psycopg2.extras import execute_values
                execute_values(cur, "INSERT INTO bench_vec (embedding) VALUES %s",
                               vals, template="(%s::vector)")
            print("building HNSW index...")
            cur.execute("CREATE INDEX ON bench_vec USING hnsw (embedding vector_cosine_ops);")
            cur.execute("ANALYZE bench_vec;")
            conn.commit()

            sql = "SELECT id FROM bench_vec ORDER BY embedding <=> %s::vector LIMIT 5;"
            with_ms, with_node = _avg(cur, sql, (query_vec,), True)
            without_ms, without_node = _avg(cur, sql, (query_vec,), False)
        conn.rollback()
    _report(f"synthetic ({n} rows)", with_ms, with_node, without_ms, without_node)


def main():
    if len(sys.argv) >= 3 and sys.argv[1] == "--synthetic":
        benchmark_synthetic(int(sys.argv[2]))
    else:
        benchmark_real()


if __name__ == "__main__":
    main()
