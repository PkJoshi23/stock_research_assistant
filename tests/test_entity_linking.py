"""Entity linking + syndication de-duplication for news ingestion.

What is pinned here:

  * one article naming several watchlist companies produces one
    ``MENTIONED_IN`` edge per company, and the ``via`` property records whether
    the stock's own search returned the article (``search``) or the article
    merely named it (``mention``);
  * a mention pass can never demote a ``search`` edge;
  * syndicated copies of one story (different URLs, near-identical titles)
    collapse to a single node - deterministically, and across tickers within one
    run;
  * every write stays idempotent, including on a second run.

No network, no LLM and no embedding model: the fetch functions are
monkeypatched per test and ``embed_texts`` is stubbed here, which is the rule
set out in ``conftest.py``. Every test uses the tmp store from the ``db_path``
fixture - ``data/stock_graph.rl`` is never opened, because Relate grants one
writer and the lead owns that store.
"""

from __future__ import annotations

import pytest

from src import config, entities, relate_db
from src.ingestion import ingest_connected_news as news_ing
from src.ingestion import ingest_sec_relationships as sec
from src.ingestion import ingest_stock_data as stock_ing
from src.schema import NEWS, REL_MENTIONED_IN, STOCK

#: The subset of the real watchlist these tests care about. Chosen to cover the
#: awkward cases: a one-character ticker (Ford is ``F``), an acronym alias that
#: is also a ticker (``AMD``), a legal name that ends in a full stop, and a
#: lowercase English word that is also a company name (``Meta``).
COMPANIES = [
    ("NVDA", "NVIDIA Corporation", ["Nvidia", "NVIDIA Corp"]),
    ("AMD", "Advanced Micro Devices, Inc.", ["AMD", "Advanced Micro Devices"]),
    (
        "TSM",
        "Taiwan Semiconductor Manufacturing Company Limited",
        ["TSMC", "Taiwan Semiconductor"],
    ),
    ("META", "Meta Platforms, Inc.", ["Meta", "Facebook"]),
    ("F", "Ford Motor Company", ["Ford"]),
    ("INTC", "Intel Corporation", ["Intel"]),
]


@pytest.fixture(autouse=True)
def _no_model_download(monkeypatch):
    """Embeddings are stubbed: tests must not download or run the model."""
    monkeypatch.setattr(
        news_ing.embeddings,
        "embed_texts",
        lambda texts, **kwargs: [[0.5] * config.EMBEDDING_DIM for _ in texts],
    )


def _seed(conn, *tickers: str) -> None:
    """Write the requested stocks through the public ingestion API."""
    for ticker, name, aliases in COMPANIES:
        if tickers and ticker not in tickers:
            continue
        stock_ing.upsert_stock(
            conn,
            {
                "ticker": ticker,
                "name": name,
                "sector": "Technology",
                "industry": "Semiconductors",
                "market_cap": 1.0e9,
                "pe_ratio": 20.0,
                "current_price": 100.0,
                "aliases": aliases,
            },
        )


@pytest.fixture
def graph(conn):
    """A connected store seeded with the watchlist companies."""
    _seed(conn, *(ticker for ticker, _, _ in COMPANIES))
    return conn


def _article(url: str, title: str, summary: str = "") -> dict:
    return {
        "url": url,
        "title": title,
        "source": "Reuters",
        "summary": summary,
        "published": None,
    }


def _edges(conn) -> list[tuple[str, str, str]]:
    """Every MENTIONED_IN edge as (ticker, url, via), sorted for stable asserts."""
    rows = relate_db.run(
        conn,
        f"""
        MATCH (s:{STOCK})-[r:{REL_MENTIONED_IN}]->(n:{NEWS})
        RETURN s.ticker AS ticker, n.url AS url, r.via AS via
        """,
    )
    return sorted((row["ticker"], row["url"], row["via"]) for row in rows)


# -- the lifted resolver -----------------------------------------------------


def test_resolver_is_the_one_shared_implementation():
    """The SEC module must re-export, not keep a second copy that can drift."""
    assert sec.resolve_tracked_stock is entities.resolve_tracked_stock


def test_resolver_matches_ticker_alias_and_legal_name(graph):
    assert entities.resolve_tracked_stock(graph, "TSMC") == "TSM"
    assert entities.resolve_tracked_stock(graph, "Nvidia") == "NVDA"
    assert entities.resolve_tracked_stock(graph, "nvidia corporation") == "NVDA"


def test_resolver_still_rejects_generic_phrases(graph):
    assert entities.resolve_tracked_stock(graph, "Cloud Infrastructure Providers") is None
    assert entities.resolve_tracked_stock(graph, "AI") is None


def test_load_companies_is_sorted_and_skips_tickerless_nodes(graph):
    relate_db.run(graph, f"MERGE (s:{STOCK} {{name: 'No Ticker Corp'}})")
    loaded = entities.load_companies(graph)

    assert [company["ticker"] for company in loaded] == [
        "AMD",
        "F",
        "INTC",
        "META",
        "NVDA",
        "TSM",
    ]
    assert loaded[-1]["aliases"] == ["TSMC", "Taiwan Semiconductor"]


# -- mention scanning --------------------------------------------------------


def test_finds_every_company_named_in_one_article(graph):
    text = "Nvidia and TSMC rise, while Advanced Micro Devices slips"
    mentioned = entities.find_mentioned_tickers(text, entities.load_companies(graph))
    # Ordered by first mention: Nvidia, TSMC, then the AMD legal name.
    assert mentioned == ["NVDA", "TSM", "AMD"]


def test_scanned_ticker_can_be_excluded(graph):
    text = "Nvidia and TSMC rise"
    mentioned = entities.find_mentioned_tickers(
        text, entities.load_companies(graph), exclude=("NVDA",)
    )
    assert mentioned == ["TSM"]


def test_matching_respects_word_boundaries(graph):
    companies = entities.load_companies(graph)
    # "intel" inside "intelligence" and "NVIDIA" inside a longer token are not
    # mentions; a substring hit here would fabricate an edge.
    assert entities.find_mentioned_tickers("market intelligence", companies) == []
    assert entities.find_mentioned_tickers("NVIDIAsomething", companies) == []


def test_legal_name_with_a_full_stop_is_found(graph):
    companies = entities.load_companies(graph)
    text = "Advanced Micro Devices, Inc. reported results"
    assert entities.find_mentioned_tickers(text, companies) == ["AMD"]


def test_one_character_ticker_is_not_matched_in_prose_but_its_name_is(graph):
    companies = entities.load_companies(graph)
    # A stray capital "F" is not evidence; the company's written name is.
    assert entities.find_mentioned_tickers("grade F for the quarter", companies) == []
    assert entities.find_mentioned_tickers("Ford recalls pickups", companies) == ["F"]


def test_company_names_match_regardless_of_capitals(graph):
    companies = entities.load_companies(graph)
    assert entities.find_mentioned_tickers("NVIDIA beats estimates", companies) == ["NVDA"]
    assert entities.find_mentioned_tickers("nvidia beats estimates", companies) == ["NVDA"]
    assert entities.find_mentioned_tickers("TSMC expands capacity", companies) == ["TSM"]


def test_mention_order_is_deterministic(graph):
    companies = entities.load_companies(graph)
    text = "Ford, Meta and AMD all moved"
    assert entities.find_mentioned_tickers(text, companies) == ["F", "META", "AMD"]
    assert entities.find_mentioned_tickers(text, companies) == ["F", "META", "AMD"]


# -- title normalisation and syndication ------------------------------------


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("NVIDIA Beats Earnings!", "nvidia beats earnings"),
        ("  Nvidia   beats\tearnings  ", "nvidia beats earnings"),
        ("Nvidia & AMD: Q3 Results", "nvidia amd q3 results"),
        ("", ""),
        (None, ""),
    ],
)
def test_normalise_title_strips_case_punctuation_and_whitespace(raw, expected):
    assert news_ing.normalise_title(raw) == expected


def test_syndicated_copy_is_dropped_and_a_different_story_is_kept():
    articles = [
        _article("https://ex.com/a", "Nvidia beats earnings as AI demand surges"),
        _article(
            "https://ex.com/b",
            "NVIDIA beats earnings as AI demand surges - Reuters",
        ),
        _article("https://ex.com/c", "Nvidia cuts jobs in Santa Clara"),
    ]

    kept, dropped = news_ing.dedupe_syndicated(articles)

    assert dropped == 1
    assert [article["url"] for article in kept] == ["https://ex.com/a", "https://ex.com/c"]


def test_exact_title_variants_collapse_even_when_short():
    kept, dropped = news_ing.dedupe_syndicated(
        [_article("u1", "AMD!"), _article("u2", "amd")]
    )
    assert dropped == 1
    assert [article["url"] for article in kept] == ["u1"]


def test_dedupe_is_deterministic():
    articles = [
        _article("u1", "Nvidia beats earnings as AI demand surges"),
        _article("u2", "NVIDIA beats earnings as AI demand surges - Reuters"),
        _article("u3", "Meta reports quarterly revenue"),
    ]

    first_kept, first_dropped = news_ing.dedupe_syndicated(articles)
    second_kept, second_dropped = news_ing.dedupe_syndicated(articles)

    assert [article["url"] for article in first_kept] == [
        article["url"] for article in second_kept
    ]
    assert (first_kept[0]["title"], first_dropped) == (second_kept[0]["title"], second_dropped)
    assert first_dropped == second_dropped == 1


def test_a_shared_seen_titles_set_collapses_a_story_across_tickers():
    seen_titles: set[str] = set()

    first_batch, first_dropped = news_ing.dedupe_syndicated(
        [_article("u1", "Nvidia beats earnings")], seen_titles
    )
    second_batch, second_dropped = news_ing.dedupe_syndicated(
        [_article("u2", "NVIDIA BEATS EARNINGS!")], seen_titles
    )

    assert [article["url"] for article in first_batch] == ["u1"]
    assert first_dropped == 0
    assert second_batch == []
    assert second_dropped == 1


def test_batch_local_dedupe_does_not_leak_between_calls():
    """Without a shared set the scope is the batch, so a later call is fresh."""
    article = _article("u1", "Nvidia beats earnings")
    assert news_ing.dedupe_syndicated([article])[1] == 0
    assert news_ing.dedupe_syndicated([article])[1] == 0


# -- graph writes: multi-company linking ------------------------------------


def test_article_links_every_named_watchlist_company(graph):
    article = _article(
        "https://ex.com/one",
        "Nvidia and TSMC lift chip stocks",
        "Advanced Micro Devices also gained.",
    )

    result = news_ing.write_articles_detailed(graph, "NVDA", [article])

    assert result.written == 1
    assert result.mention_links == 2
    assert _edges(graph) == [
        ("AMD", "https://ex.com/one", "mention"),
        ("NVDA", "https://ex.com/one", "search"),
        ("TSM", "https://ex.com/one", "mention"),
    ]


def test_mention_never_demotes_the_search_edge(graph):
    """The stock whose search returned the article must keep ``via='search'``."""
    article = _article("https://ex.com/two", "Nvidia beats estimates", "")
    news_ing.write_articles_detailed(graph, "NVDA", [article])

    result = news_ing.write_articles_detailed(graph, "AMD", [article])

    assert result.written == 0
    assert result.skipped == 1
    assert _edges(graph) == [
        ("AMD", "https://ex.com/two", "search"),
        ("NVDA", "https://ex.com/two", "search"),
    ]


def test_a_first_seen_article_still_gets_the_new_stocks_edge(graph):
    """A known URL must not lose the edge for a stock that mentions it."""
    article = _article("https://ex.com/three", "Nvidia and TSMC lift chip stocks")
    news_ing.write_articles_detailed(graph, "NVDA", [article])

    result = news_ing.write_articles_detailed(graph, "TSM", [article])

    assert result.written == 0
    assert result.skipped == 1
    assert ("TSM", "https://ex.com/three", "search") in _edges(graph)


def test_repeat_write_is_idempotent(graph):
    articles = [
        _article("https://ex.com/x", "Nvidia and TSMC lift chip stocks"),
        _article("https://ex.com/y", "Meta reports ad revenue"),
    ]
    news_ing.write_articles_detailed(graph, "NVDA", articles)
    edges_after_first = _edges(graph)

    result = news_ing.write_articles_detailed(graph, "NVDA", articles)

    assert result.written == 0
    assert result.skipped == 2
    assert result.mention_links == 2
    assert _edges(graph) == edges_after_first


def test_compat_wrapper_keeps_the_two_value_contract(graph):
    article = _article("https://ex.com/compat", "Nvidia beats estimates")
    assert news_ing.write_articles(graph, "NVDA", [article]) == (1, 0)
    assert news_ing.write_articles(graph, "NVDA", [article]) == (0, 1)


def test_empty_batch_writes_nothing(graph):
    result = news_ing.write_articles_detailed(graph, "NVDA", [])
    assert (result.written, result.skipped, result.duplicates, result.mention_links) == (
        0,
        0,
        0,
        0,
    )


# -- graph writes: syndication ----------------------------------------------


def test_syndicated_copy_is_not_stored(graph):
    articles = [
        _article("https://ex.com/a", "Nvidia beats earnings as AI demand surges"),
        _article(
            "https://ex.com/b",
            "NVIDIA beats earnings as AI demand surges - Reuters",
        ),
    ]

    result = news_ing.write_articles_detailed(graph, "NVDA", articles)

    assert result.written == 1
    assert result.duplicates == 1
    assert result.mention_links == 0
    assert relate_db.counts(graph)[NEWS] == 1
    assert _edges(graph) == [("NVDA", "https://ex.com/a", "search")]


# -- run-level orchestration ------------------------------------------------


def test_ingest_ticker_news_collapses_a_story_across_tickers(graph, monkeypatch):
    """Drive the per-ticker helper the way ``main`` does, with one shared set."""
    feeds = {
        "NVDA": [
            _article(
                "https://ex.com/n1",
                "NVIDIA and TSMC lift chip stocks as AI demand surges",
            )
        ],
        "TSM": [
            _article(
                "https://ex.com/t1",
                "NVIDIA and TSMC lift chip stocks as AI demand surges - Reuters",
            )
        ],
    }
    monkeypatch.setattr(
        news_ing, "fetch_rss_articles", lambda ticker, name, limit: feeds[ticker]
    )
    monkeypatch.setattr(news_ing, "fetch_yfinance_articles", lambda ticker, limit: [])

    seen_titles: set[str] = set()
    first = news_ing.ingest_ticker_news(
        graph, "NVDA", "NVIDIA Corporation", seen_titles=seen_titles
    )
    second = news_ing.ingest_ticker_news(
        graph,
        "TSM",
        "Taiwan Semiconductor Manufacturing Company Limited",
        seen_titles=seen_titles,
    )

    assert (first.written, first.duplicates) == (1, 0)
    assert (second.written, second.duplicates) == (0, 1)
    assert relate_db.counts(graph)[NEWS] == 1
    # The surviving article still links both companies: TSMC through a mention.
    assert _edges(graph) == [
        ("NVDA", "https://ex.com/n1", "search"),
        ("TSM", "https://ex.com/n1", "mention"),
    ]


def test_main_threads_one_title_set_across_the_watchlist(db_path, monkeypatch, capsys):
    """``main`` is the only place the run-wide de-duplication scope is set."""
    seed_conn = relate_db.connect()
    try:
        _seed(seed_conn, "NVDA", "TSM")
    finally:
        seed_conn.close()

    feeds = {
        "NVDA": [
            _article(
                "https://ex.com/n1",
                "NVIDIA and TSMC lift chip stocks as AI demand surges",
            )
        ],
        "TSM": [
            _article(
                "https://ex.com/t1",
                "NVIDIA and TSMC lift chip stocks as AI demand surges - Reuters",
            )
        ],
    }
    monkeypatch.setattr(news_ing, "fetch_rss_articles", lambda ticker, name, limit: feeds[ticker])
    monkeypatch.setattr(news_ing, "fetch_yfinance_articles", lambda ticker, limit: [])
    # No yfinance cache setup in a test: it writes outside the tmp store.
    monkeypatch.setattr(news_ing.config, "configure_yfinance_cache", lambda: None)
    monkeypatch.setattr(news_ing.config, "WATCHLIST", ["NVDA", "TSM"])

    assert news_ing.main() == 0

    check_conn = relate_db.connect()
    try:
        assert relate_db.counts(check_conn)[NEWS] == 1
        assert _edges(check_conn) == [
            ("NVDA", "https://ex.com/n1", "search"),
            ("TSM", "https://ex.com/n1", "mention"),
        ]
    finally:
        check_conn.close()

    output = capsys.readouterr().out
    assert "Syndicated duplicates dropped: 1" in output
    assert "Cross-company mention links ensured: 1" in output
