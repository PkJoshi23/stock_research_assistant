"""Ingestion: idempotency, de-duplication and the edge cases that bit us.

Two behaviours are load-bearing for the demo:
  * re-running ingestion must not duplicate nodes or edges (it is run repeatedly
    while tuning the curated relationship list);
  * an article shared by two stocks must still produce two MENTIONED_IN edges,
    even though its node is written once.
"""

from __future__ import annotations

import pytest

from src import config, relate_db
from src.ingestion import ingest_connected_news as news_ing
from src.ingestion import ingest_stock_data as stock_ing

SNAPSHOT = {
    "ticker": "NVDA",
    "name": "NVIDIA Corporation",
    "sector": "Technology",
    "industry": "Semiconductors",
    "market_cap": 3.4e12,
    "pe_ratio": 55.2,
    "current_price": 150.0,
    "aliases": ["Nvidia"],
}

ARTICLE = {
    "url": "https://ex.com/story",
    "title": "Nvidia beats earnings",
    "source": "Reuters",
    "summary": "Chip demand stays strong.",
    "published": None,
}


@pytest.fixture(autouse=True)
def _no_model_download(monkeypatch):
    """Embeddings are stubbed: tests must not download or run the model."""
    monkeypatch.setattr(
        news_ing.embeddings,
        "embed_texts",
        lambda texts, **kwargs: [[0.25] * config.EMBEDDING_DIM for _ in texts],
    )


# -- stock ingestion ---------------------------------------------------------


def test_upsert_stock_creates_node_sector_and_edge(conn):
    stock_ing.upsert_stock(conn, SNAPSHOT)
    assert relate_db.counts(conn) == {"Stock": 1, "Sector": 1}
    edges = relate_db.run(conn, "MATCH ()-[r:IN_SECTOR]->() RETURN count(r) AS n")
    assert edges[0]["n"] == 1


def test_upsert_stock_is_idempotent_and_updates_values(conn):
    stock_ing.upsert_stock(conn, SNAPSHOT)
    stock_ing.upsert_stock(conn, {**SNAPSHOT, "current_price": 199.0})
    assert relate_db.counts(conn) == {"Stock": 1, "Sector": 1}
    assert relate_db.run(conn, "MATCH ()-[r:IN_SECTOR]->() RETURN count(r) AS n")[0]["n"] == 1
    price = relate_db.run(conn, "MATCH (s:Stock {ticker: 'NVDA'}) RETURN s.current_price AS p")
    assert price[0]["p"] == 199.0


def test_relationship_write_is_idempotent(conn):
    stock_ing.upsert_stock(conn, SNAPSHOT)
    stock_ing.upsert_stock(conn, {**SNAPSHOT, "ticker": "AMD", "name": "AMD", "aliases": []})
    for _ in range(3):
        stock_ing.write_relationship(conn, "NVDA", "AMD", "COMPETES_WITH")
    assert relate_db.run(conn, "MATCH ()-[r:COMPETES_WITH]->() RETURN count(r) AS n")[0]["n"] == 1


def test_person_write_is_idempotent(conn):
    stock_ing.upsert_stock(conn, SNAPSHOT)
    for _ in range(2):
        stock_ing.write_person(conn, "Jensen Huang", "CEO", "NVDA")
    assert relate_db.counts(conn).get("Person") == 1
    assert relate_db.run(conn, "MATCH ()-[r:WORKS_FOR]->() RETURN count(r) AS n")[0]["n"] == 1


def test_aliases_are_written(conn):
    stock_ing.upsert_stock(conn, SNAPSHOT)
    rows = relate_db.run(conn, "MATCH (s:Stock {ticker: 'NVDA'}) RETURN s.aliases AS a")
    assert rows[0]["a"] == ["Nvidia"]


# -- news ingestion ----------------------------------------------------------


def test_write_articles_is_idempotent(conn):
    stock_ing.upsert_stock(conn, SNAPSHOT)
    assert news_ing.write_articles(conn, "NVDA", [ARTICLE]) == (1, 0)
    assert news_ing.write_articles(conn, "NVDA", [ARTICLE]) == (0, 1)
    assert relate_db.counts(conn)["NewsArticle"] == 1
    assert relate_db.run(conn, "MATCH ()-[r:MENTIONED_IN]->() RETURN count(r) AS n")[0]["n"] == 1


def test_shared_article_links_to_every_mentioning_stock(conn):
    """The bug this test pins: skipping a known URL also skipped the edge."""
    stock_ing.upsert_stock(conn, SNAPSHOT)
    stock_ing.upsert_stock(conn, {**SNAPSHOT, "ticker": "AMD", "name": "AMD", "aliases": []})

    news_ing.write_articles(conn, "NVDA", [ARTICLE])
    news_ing.write_articles(conn, "AMD", [ARTICLE])

    assert relate_db.counts(conn)["NewsArticle"] == 1
    edges = relate_db.run(
        conn,
        "MATCH (s:Stock)-[r:MENTIONED_IN]->(n:NewsArticle) "
        "RETURN s.ticker AS ticker ORDER BY ticker",
    )
    assert [row["ticker"] for row in edges] == ["AMD", "NVDA"]


def test_duplicate_urls_within_one_batch_are_written_once(conn):
    stock_ing.upsert_stock(conn, SNAPSHOT)
    banner = {**ARTICLE, "url": "https://ex.com/same"}
    written, _ = news_ing.write_articles(conn, "NVDA", [banner, banner])
    # Both entries point at the same url, so only one node is created; the edge
    # is merged too, so the second pass adds nothing.
    assert written == 1
    assert relate_db.counts(conn)["NewsArticle"] == 1


def test_canonical_url_strips_tracking_and_fragment():
    assert news_ing.canonical_url("https://Ex.com/a/b?oc=5#frag") == "https://ex.com/a/b"
    assert news_ing.canonical_url("https://ex.com/a/b") == "https://ex.com/a/b"


def test_clean_html_strips_markup_and_entities():
    assert news_ing.clean_html("&lt;p&gt;Hi &amp; bye&lt;/p&gt;") == "Hi & bye"
    assert news_ing.clean_html(None) == ""


def test_merge_articles_prefers_the_first_source_and_dedupes():
    rss = [{"url": "u1", "title": "from rss"}, {"url": "u2", "title": "rss 2"}]
    yahoo = [{"url": "u1", "title": "from yahoo"}, {"url": "u3", "title": "yahoo 3"}]
    merged = news_ing.merge_articles(rss, yahoo, limit=10)
    assert [item["url"] for item in merged] == ["u1", "u2", "u3"]
    assert merged[0]["title"] == "from rss"


def test_merge_articles_respects_the_limit():
    batch = [{"url": f"u{i}", "title": str(i)} for i in range(10)]
    assert len(news_ing.merge_articles(batch, limit=3)) == 3


def test_to_datetime_handles_both_source_shapes():
    import datetime as dt

    assert news_ing.to_datetime(None) is None
    assert news_ing.to_datetime((2026, 11, 17, 10, 30, 0, 0, 0, 0)) == dt.datetime(2026, 11, 17, 10, 30)
    assert news_ing.to_datetime(0) == dt.datetime(1970, 1, 1)
    stamp = dt.datetime(2026, 1, 2, 3, 4)
    assert news_ing.to_datetime(stamp) == stamp
