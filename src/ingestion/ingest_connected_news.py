"""News ingestion: Google News RSS + yfinance, embedded into the graph.

Why these two sources (NewsAPI was dropped):
  * **Google News RSS** - free, no key, returns headline + publisher + summary.
    The summary is what makes it the primary source: it gives the embedding
    model real sentences instead of a bare title.
  * **yfinance** - ticker-native and gives a clean ``relatedTickers`` list, but
    note that ``Ticker.news`` is broken in yfinance 1.4.1 (it raises
    ``YFDataException: *** YAHOO! FINANCE IS CURRENTLY DOWN! ***``). The working
    call is ``yf.Search(ticker, news_count=n).news``. It is used here as a
    secondary source, and a failure never aborts the run.

What this writes
----------------
  (:NewsArticle {url, title, source, published, summary, embedding})
  (:Stock)-[:MENTIONED_IN {via: 'search' | 'mention'}]->(:NewsArticle)

``url`` is the identity: re-running ingestion is idempotent because articles
are matched on it. The embedding is a 384-float list (MiniLM), which Relate
accepts directly - ``relate.Vector`` has no Python constructor.

Who is linked, and why ``via`` exists
-------------------------------------
An article is fetched *because of* one ticker, but it is rarely about only that
company: "Nvidia and Taiwan Semiconductor rise as AMD slips" is one story about
three watchlist names. Writing only the searched-for edge threw that context
away and left the graph answering "news about TSMC" with a fraction of the
stories it actually holds.

So every kept article is scanned (title + summary) for all other tracked
companies, and each hit gets its own edge. ``via`` records provenance:

  * ``via='search'``  - this stock's own news search returned the article;
  * ``via='mention'`` - the article merely names the company.

The distinction matters to any query that wants to weight coverage, and it is
why the mention pass never overwrites a search edge (see
:func:`write_articles_detailed`).

Syndication de-duplication
--------------------------
One real story is republished by many outlets with near-identical headlines
("Nvidia beats earnings as AI demand surges" / "NVIDIA beats earnings as AI
demand surges"). They arrive under different URLs, so URL identity cannot
collapse them. Titles are therefore normalised (lower case, punctuation and
whitespace removed) and compared with ``difflib.SequenceMatcher``; at or above
:data:`SYNDICATION_SIMILARITY` the article is treated as the same story and
skipped. The check is scoped to one run and is deterministic (input order in,
same decisions out) - it never reads previously stored titles, so an older
graph cannot change what today's run decides.

Run:
    python -m src.ingestion.ingest_connected_news
    python -m src.ingestion.ingest_connected_news --reset   # re-embed everything
"""

from __future__ import annotations

import argparse
import datetime as dt
import html
import re
import sys
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
from urllib.parse import quote_plus, urlsplit, urlunsplit

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import feedparser
import requests
import yfinance as yf

from src import config, embeddings, entities, relate_db
from src.schema import NEWS, REL_MENTIONED_IN, STOCK

_TAG_RE = re.compile(r"<[^>]+>")
_WS_RE = re.compile(r"\s+")

#: Everything that is not a letter or a digit becomes a separator when titles
#: are normalised. An explicit ASCII class (rather than ``\W`` under some
#: locale) is what makes the comparison reproducible on any machine.
_TITLE_JUNK_RE = re.compile(r"[^a-z0-9]+")

#: Similarity at which two normalised titles are the same story.
#:
#: 0.9 is deliberately high. Lower thresholds start collapsing *related*
#: stories ("Nvidia beats earnings" vs "Nvidia raises guidance") which are
#: different facts, and losing a fact is worse than keeping a duplicate here.
#: At 0.9 the differences that survive are exactly the cosmetic ones syndication
#: introduces (an outlet prefix, "shares" vs "stock", a trailing source name).
SYNDICATION_SIMILARITY = 0.9

#: Normalised titles shorter than this are only collapsed on an exact match.
#: Short strings have few positions to differ in, so the ratio is dominated by
#: the shared prefix and stops meaning "same story".
MIN_SIMILARITY_TITLE_LENGTH = 4


# -----------------------------------------------------------------------------
# Text / URL helpers
# -----------------------------------------------------------------------------


def clean_html(text: str | None) -> str:
    """Strip tags and entities from an RSS summary.

    Google ships the snippet as escaped HTML (``&lt;a href=...&gt;``), so the
    entity decode has to happen *before* tag stripping for the tags to be
    visible to the regex.
    """
    if not text:
        return ""
    plain = html.unescape(html.unescape(text))
    plain = _TAG_RE.sub(" ", plain)
    return _WS_RE.sub(" ", plain).strip()


def canonical_url(url: str) -> str:
    """Drop query and fragment so the same article is one node.

    Tracking parameters (``?oc=5``, ``utm_*``) differ per fetch and per source;
    keeping them would create duplicate nodes on every ingestion run.
    """
    parts = urlsplit(url.strip())
    return urlunsplit((parts.scheme, parts.netloc.lower(), parts.path, "", ""))


def to_datetime(value) -> dt.datetime | None:
    """Normalise the two timestamp shapes the sources return."""
    if value is None:
        return None
    if isinstance(value, dt.datetime):
        return value
    if isinstance(value, (int, float)):  # yfinance: epoch seconds
        return dt.datetime.fromtimestamp(value, tz=dt.timezone.utc).replace(tzinfo=None)
    if isinstance(value, tuple):  # feedparser: struct_time
        return dt.datetime(*value[:6])
    return None


# -----------------------------------------------------------------------------
# Syndication de-duplication
# -----------------------------------------------------------------------------


def normalise_title(title: str | None) -> str:
    """Reduce a headline to the part that identifies the story.

    Case, punctuation and whitespace are noise: syndication adds an outlet
    prefix, changes "Nvidia" to "NVIDIA", swaps "&" for "and" casing and appends
    ": Reuters". What is left - lower-case alphanumeric words - is what two
    copies of the same story share.
    """
    if not title:
        return ""
    text = html.unescape(title).lower()
    return _WS_RE.sub(" ", _TITLE_JUNK_RE.sub(" ", text)).strip()


def is_syndicated(title: str, seen_titles: set[str]) -> bool:
    """True when ``title`` (already normalised) repeats a story already kept.

    ``autojunk=False`` on purpose: ``SequenceMatcher``'s heuristic discards
    characters that appear often, which is tuned for long text and makes short
    headlines score unpredictably. Turning it off keeps the comparison honest
    and deterministic for the string lengths this function sees.
    """
    if title in seen_titles:
        return True
    if len(title) < MIN_SIMILARITY_TITLE_LENGTH:
        return False
    return any(
        SequenceMatcher(None, title, other, autojunk=False).ratio()
        >= SYNDICATION_SIMILARITY
        for other in seen_titles
    )


def dedupe_syndicated(
    articles: list[dict], seen_titles: set[str] | None = None
) -> tuple[list[dict], int]:
    """Drop near-duplicate stories, preserving the first copy seen.

    ``seen_titles`` is the run's normalised-title set. The orchestrator passes
    one set across every ticker, so a story syndicated into two tickers' feeds
    is stored once; callers that pass nothing get batch-local de-duplication.
    Either way the scope is this run only - a previous run's titles are never
    consulted, which keeps the result a function of this run's input.

    Returns ``(kept, dropped)``. Order is preserved and the first copy wins, so
    the caller's source preference (RSS over yfinance, see
    :func:`merge_articles`) survives.
    """
    seen = seen_titles if seen_titles is not None else set()
    kept: list[dict] = []
    dropped = 0
    for article in articles:
        title = normalise_title(article.get("title"))
        if title and is_syndicated(title, seen):
            dropped += 1
            continue
        if title:
            seen.add(title)
        kept.append(article)
    return kept, dropped


# -----------------------------------------------------------------------------
# Sources
# -----------------------------------------------------------------------------


def fetch_rss_articles(ticker: str, company_name: str, limit: int) -> list[dict]:
    """Primary source: Google News RSS search for the company + ticker."""
    query = quote_plus(f"{company_name} OR {ticker} stock")
    url = config.NEWS_RSS_TEMPLATE.format(query=query)
    try:
        response = requests.get(
            url,
            headers={"User-Agent": config.NEWS_USER_AGENT},
            timeout=config.HTTP_TIMEOUT_S,
        )
        response.raise_for_status()
    except requests.RequestException as exc:
        print(f"  ! RSS request failed for {ticker}: {exc}")
        return []

    feed = feedparser.parse(response.content)
    articles: list[dict] = []
    for entry in feed.entries[:limit]:
        link = entry.get("link")
        title = clean_html(entry.get("title"))
        if not link or not title:
            continue
        # feedparser exposes the outlet as entry.source (a FeedParserDict).
        source = ""
        source_obj = entry.get("source")
        if source_obj is not None:
            source = clean_html(source_obj.get("title") or "")
        articles.append(
            {
                "url": canonical_url(link),
                "title": title,
                "source": source or "Google News",
                "summary": clean_html(entry.get("summary"))[:600],
                "published": to_datetime(entry.get("published_parsed")),
            }
        )
    return articles


def fetch_yfinance_articles(ticker: str, limit: int) -> list[dict]:
    """Secondary source: yfinance's search feed (not ``Ticker.news``).

    Wrapped in a broad ``except`` on purpose: this is the optional half of the
    ingestion, and Yahoo rate-limits aggressively. Losing it degrades coverage,
    it must not fail the run.
    """
    try:
        items = yf.Search(ticker, news_count=limit).news or []
    except Exception as exc:  # noqa: BLE001 - third-party endpoint
        print(f"  ~ yfinance news unavailable for {ticker} ({type(exc).__name__})")
        return []

    articles: list[dict] = []
    for item in items[:limit]:
        title = (item.get("title") or "").strip()
        link = item.get("link") or ""
        if not title or not link:
            continue
        articles.append(
            {
                "url": canonical_url(link),
                "title": title,
                # yfinance 1.4.1 renamed this field to "publisher"; older
                # payloads used "provider".
                "source": item.get("publisher") or item.get("provider") or "Yahoo Finance",
                "summary": (item.get("summary") or "").strip()[:600],
                "published": to_datetime(item.get("providerPublishTime")),
            }
        )
    return articles


def merge_articles(*batches: list[dict], limit: int) -> list[dict]:
    """De-duplicate across sources, preferring the first (higher-quality) copy.

    RSS wins over yfinance because it carries a real summary, which is what the
    embedding is built from.
    """
    seen: set[str] = set()
    merged: list[dict] = []
    for batch in batches:
        for article in batch:
            if article["url"] in seen:
                continue
            seen.add(article["url"])
            merged.append(article)
            if len(merged) >= limit:
                return merged
    return merged


# -----------------------------------------------------------------------------
# Graph writes
# -----------------------------------------------------------------------------


@dataclass(frozen=True)
class WriteResult:
    """What one ticker's news write did.

    A dataclass rather than a wider tuple: :func:`write_articles` has to keep
    returning ``(written, already_present)`` for existing callers, while the
    run summary wants the syndication and mention numbers too.
    """

    written: int = 0
    skipped: int = 0
    duplicates: int = 0
    mention_links: int = 0


def _unique_by_url(articles: list[dict]) -> list[dict]:
    """Keep the first article per URL, preserving input order."""
    unique: dict[str, dict] = {}
    for article in articles:
        unique.setdefault(article["url"], article)
    return list(unique.values())


def _write_articles(conn, ticker: str, articles: list[dict]) -> tuple[int, int, int]:
    """Embed and write the node + search edges. Returns (fresh, existing, mentions).

    Split out of :func:`write_articles_detailed` so the de-duplication, the
    embedding pass and the edge pass stay readable as three separate steps.
    """
    existing_urls = {
        row["url"]
        for row in relate_db.run(
            conn,
            f"MATCH (n:{NEWS}) WHERE n.url IN $urls RETURN n.url AS url",
            {"urls": [article["url"] for article in articles]},
        )
    }
    fresh = [article for article in articles if article["url"] not in existing_urls]

    # 1. Embed and write only genuinely new articles (embedding is the cost).
    if fresh:
        # Title + summary gives the model a sentence to reason about; a bare
        # headline embeds poorly and retrieves on keyword overlap instead of
        # meaning.
        vectors = embeddings.embed_texts(
            [f"{article['title']}. {article['summary']}".strip() for article in fresh]
        )
        for article, vector in zip(fresh, vectors):
            relate_db.run(
                conn,
                f"""
                MERGE (n:{NEWS} {{url: $url}})
                SET n.title     = $title,
                    n.source    = $source,
                    n.summary   = $summary,
                    n.published = $published,
                    n.embedding = $embedding
                """,
                {**article, "embedding": vector},
            )

    # 2. Link every article to the ticker that was searched for, new or
    #    pre-existing. SET rather than ON CREATE: an edge first written as a
    #    mention is upgraded once this stock's own search returns the story,
    #    because "its own result" is stronger provenance than "it was named".
    for article in articles:
        relate_db.run(
            conn,
            f"""
            MATCH (s:{STOCK} {{ticker: $ticker}}), (n:{NEWS} {{url: $url}})
            MERGE (s)-[r:{REL_MENTIONED_IN}]->(n)
            SET r.via = $via
            """,
            {"ticker": ticker, "url": article["url"], "via": "search"},
        )

    # 3. Link every OTHER tracked company the article names. ``coalesce`` is
    #    load-bearing: it fills ``via`` when the edge is new and *preserves* an
    #    existing 'search' value, so a mention scan can never demote the stock
    #    whose search produced the article.
    companies = entities.load_companies(conn)
    mention_links = 0
    for article in articles:
        text = f"{article['title']}. {article.get('summary') or ''}".strip()
        for mentioned in entities.find_mentioned_tickers(text, companies, exclude=(ticker,)):
            relate_db.run(
                conn,
                f"""
                MATCH (s:{STOCK} {{ticker: $ticker}}), (n:{NEWS} {{url: $url}})
                MERGE (s)-[r:{REL_MENTIONED_IN}]->(n)
                SET r.via = coalesce(r.via, $via)
                """,
                {"ticker": mentioned, "url": article["url"], "via": "mention"},
            )
            mention_links += 1

    urls = {article["url"] for article in articles}
    return len(fresh), len(existing_urls & urls), mention_links


def write_articles_detailed(
    conn, ticker: str, articles: list[dict], *, seen_titles: set[str] | None = None
) -> WriteResult:
    """Write one ticker's articles and their edges; report everything.

    Steps, in order:

    1. drop syndicated near-duplicates (see :func:`dedupe_syndicated`);
    2. collapse repeated URLs - a feed can repeat a story, and a duplicate
       would otherwise be embedded twice and reported as two writes even though
       MERGE collapses it to one node;
    3. embed + MERGE only the URLs not already stored (embedding is the cost);
    4. MERGE the searched ticker's edge with ``via='search'``;
    5. MERGE ``via='mention'`` edges for every other tracked company named in
       title + summary.

    Every write is a MERGE, so running this twice adds nothing. ``skipped``
    counts articles whose URL was already in the graph - a pre-existing article
    still gets its edge, because "already stored" must not mean "not linked".
    """
    if not articles:
        return WriteResult()

    articles, duplicates = dedupe_syndicated(articles, seen_titles)
    articles = _unique_by_url(articles)
    if not articles:
        return WriteResult(duplicates=duplicates)

    written, skipped, mention_links = _write_articles(conn, ticker, articles)
    return WriteResult(
        written=written,
        skipped=skipped,
        duplicates=duplicates,
        mention_links=mention_links,
    )


def write_articles(conn, ticker: str, articles: list[dict]) -> tuple[int, int]:
    """Embed and write articles for one ticker. Returns (written, skipped).

    Kept as the original two-value API because callers (and tests) unpack it.
    Syndication de-duplication is still applied, batch-local; use
    :func:`write_articles_detailed` when the counts or a run-wide
    ``seen_titles`` set are needed.
    """
    result = write_articles_detailed(conn, ticker, articles)
    return result.written, result.skipped


def ingest_ticker_news(
    conn, ticker: str, company_name: str, *, seen_titles: set[str] | None = None
) -> WriteResult:
    """Fetch, de-duplicate and write the news for one ticker.

    The per-source budget is split deliberately: RSS carries summaries (best
    for embedding) and yfinance adds ticker-native coverage. Taking the whole
    limit from RSS would leave the second source with no slot at all.

    ``seen_titles`` is passed through from :func:`main` so the same story is
    collapsed across tickers, not just inside one ticker's batch.
    """
    rss_limit = max(1, config.NEWS_ITEMS_PER_TICKER - 2)
    yahoo_limit = max(1, min(2, config.NEWS_ITEMS_PER_TICKER // 2))
    rss = fetch_rss_articles(ticker, company_name, rss_limit)
    yahoo = fetch_yfinance_articles(ticker, yahoo_limit)
    articles = merge_articles(rss, yahoo, limit=config.NEWS_ITEMS_PER_TICKER)
    return write_articles_detailed(conn, ticker, articles, seen_titles=seen_titles)


def main(reset: bool = False) -> int:
    # yfinance needs its SQLite caches redirected before first use here too.
    config.configure_yfinance_cache()

    conn = relate_db.connect()
    # Index first: Relate populates a vector index as nodes are written.
    relate_db.ensure_schema(conn)

    if reset:
        conn.execute(f"MATCH (n:{NEWS}) DETACH DELETE n")
        print("Existing news nodes deleted (stocks and relationships kept).")

    total_written = total_skipped = total_duplicates = total_mentions = 0
    # One set for the whole run: syndication crosses tickers, so a story pulled
    # into two feeds must collapse into one node.
    seen_titles: set[str] = set()

    for ticker in config.WATCHLIST:
        rows = relate_db.run(
            conn, f"MATCH (s:{STOCK} {{ticker: $t}}) RETURN s.name AS name", {"t": ticker}
        )
        if not rows:
            print(f"  ! {ticker}: no Stock node, run ingest_stock_data first - skipped")
            continue
        company_name = rows[0]["name"] or ticker

        print(f"\n{ticker} ({company_name})")
        result = ingest_ticker_news(conn, ticker, company_name, seen_titles=seen_titles)
        total_written += result.written
        total_skipped += result.skipped
        total_duplicates += result.duplicates
        total_mentions += result.mention_links

        detail = []
        if result.duplicates:
            detail.append(f"{result.duplicates} syndicated duplicate(s) skipped")
        if result.mention_links:
            detail.append(f"{result.mention_links} cross-company mention link(s)")
        suffix = f" ({', '.join(detail)})" if detail else ""
        print(f"  + {result.written} new article(s), {result.skipped} already present{suffix}")

    print("\nGraph now contains:")
    for label, count in sorted(relate_db.counts(conn).items()):
        print(f"  {label:<12} {count}")
    print(f"\nNews written: {total_written} (already present: {total_skipped})")
    print(f"Syndicated duplicates dropped: {total_duplicates}")
    print(f"Cross-company mention links ensured: {total_mentions}")
    conn.close()
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Delete existing NewsArticle nodes and re-embed from scratch.",
    )
    raise SystemExit(main(reset=parser.parse_args().reset))
