"""Snapshot ingestion: entities, relationships and CURRENT values only.

What this script writes
-----------------------
  (:Stock {ticker, name, sector, industry, market_cap, pe_ratio,
           current_price, updated_at})
  (:Sector {name})
  (:Person {name, role})
  (:Stock)-[:IN_SECTOR]->(:Sector)
  (:Stock)-[:COMPETES_WITH]->(:Stock)
  (:Stock)-[:SUPPLIES_TO]->(:Stock)
  (:Person)-[:WORKS_FOR]->(:Stock)

What it deliberately does NOT write: a price history. No OHLCV rows, no daily
series. ``updated_at`` is the single timestamp on the node and it exists so the
UI can say "as of ...", not so the graph can be queried by date.

Run:
    python -m src.ingestion.ingest_stock_data
    python -m src.ingestion.ingest_stock_data --reset
"""

from __future__ import annotations

import argparse
import datetime as dt
import sys
from pathlib import Path

# Allow both `python -m src.ingestion.ingest_stock_data` and
# `python src/ingestion/ingest_stock_data.py` to resolve `src.*` imports.
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import yfinance as yf

from src import config, quotes, relate_db
from src.schema import (
    PERSON,
    REL_COMPETES_WITH,
    REL_IN_SECTOR,
    REL_SUPPLIES_TO,
    REL_WORKS_FOR,
    SECTOR,
    STOCK,
)

# -----------------------------------------------------------------------------
# Curated relationships
# -----------------------------------------------------------------------------
#
# Public data does not expose a clean "who supplies whom" or "who competes with
# whom" feed, and inferring it with an LLM would make the demo unverifiable.
# These are hand-curated from public reporting and are the "secret sauce" the
# graph is built to show off. Everything here is a statement about a
# relationship, not about a price - which is exactly the hackathon theme.
#
# Keep this list small and correct: a wrong edge is worse than a missing one.

CURATED_COMPETITORS: list[tuple[str, str]] = [
    ("AAPL", "GOOGL"),
    ("MSFT", "GOOGL"),
    ("NVDA", "AMD"),
    ("TSLA", "F"),
    ("INTC", "AMD"),
    ("GOOGL", "META"),
]

CURATED_SUPPLY: list[tuple[str, str]] = [
    ("NVDA", "MSFT"),   # NVIDIA accelerators power Azure
    ("NVDA", "GOOGL"),  # ... and Google Cloud
    ("NVDA", "AMZN"),   # ... and AWS
    ("INTC", "TSLA"),   # Intel silicon in Tesla vehicles
    ("AMD", "MSFT"),    # EPYC in Azure instances
    # Foundry: TSMC manufactures the chips NVDA and AMD design. Without this
    # edge "who supplies Nvidia?" has no answer, which is the single most
    # natural question to ask of this graph.
    ("TSM", "NVDA"),
    ("TSM", "AMD"),
]

CURATED_EXECUTIVES: list[tuple[str, str, str]] = [
    ("Jensen Huang", "CEO", "NVDA"),
    ("Satya Nadella", "CEO", "MSFT"),
    ("Tim Cook", "CEO", "AAPL"),
    ("Elon Musk", "CEO", "TSLA"),
    ("Sundar Pichai", "CEO", "GOOGL"),
    ("Lisa Su", "CEO", "AMD"),
    ("Pat Gelsinger", "CEO", "INTC"),
    ("Andy Jassy", "CEO", "AMZN"),
    ("Mark Zuckerberg", "CEO", "META"),
    ("Jim Farley", "CEO", "F"),
]

#: Names people actually type, mapped to tickers.
#:
#: Without these, "who does TSMC supply?" matches nothing: the node is called
#: "Taiwan Semiconductor Manufacturing Company Limited" and its ticker is TSM,
#: so neither CONTAINS 'TSMC'. Name resolution is a graph property here, not a
#: prompt hack - the agent is told to match ticker, name *or* alias.
ALIASES: dict[str, list[str]] = {
    "AAPL": ["Apple"],
    "MSFT": ["Microsoft"],
    "NVDA": ["Nvidia", "NVIDIA Corp"],
    "TSM": ["TSMC", "Taiwan Semiconductor"],
    "TSLA": ["Tesla"],
    "GOOGL": ["Google", "Alphabet"],
    "AMD": ["AMD", "Advanced Micro Devices"],
    "INTC": ["Intel"],
    "AMZN": ["Amazon", "AWS"],
    "META": ["Meta", "Facebook"],
    "F": ["Ford"],
}


def fetch_snapshot(ticker: str) -> dict | None:
    """Return the current snapshot for one ticker, or ``None`` if unavailable.

    ``Ticker.info`` is the only call that carries sector/industry/market cap,
    but it is also the flakiest endpoint Yahoo has. A failure here is treated as
    "skip this ticker and keep going" rather than "abort the ingestion": a demo
    graph with nine companies beats a crashed script.
    """
    stock = yf.Ticker(ticker)
    try:
        info = stock.info or {}
    except Exception as exc:  # noqa: BLE001 - third-party, message only
        print(f"  ! {ticker}: info unavailable ({type(exc).__name__}: {exc})")
        return None

    if not info or info.get("quoteType") is None:
        print(f"  ! {ticker}: empty quote payload, skipped")
        return None

    price = info.get("currentPrice") or info.get("regularMarketPrice")
    if price is None:
        # One quote path for the whole project: same fallback order as the
        # live-price answer, so the stored snapshot and the live answer cannot
        # disagree about which endpoint "wins".
        quote = quotes.fetch(ticker)
        price = quote.price if quote else None

    return {
        "ticker": ticker,
        "name": info.get("longName") or info.get("shortName") or ticker,
        "sector": info.get("sector") or "Unknown",
        "industry": info.get("industry") or "Unknown",
        "market_cap": float(info.get("marketCap") or 0),
        "pe_ratio": float(info.get("trailingPE") or 0),
        "current_price": float(price or 0),
        "aliases": ALIASES.get(ticker, []),
    }


def upsert_stock(conn, snapshot: dict) -> None:
    """Write one stock and its sector edge. Two statements, no WITH juggling.

    Parameters instead of f-strings: company names contain apostrophes
    ("Lowe's"), and the previous implementation's manual escaping was one
    unescaped character away from a Cypher injection into the demo graph.
    """
    relate_db.query(
        conn,
        f"""
        MERGE (s:{STOCK} {{ticker: $ticker}})
        SET s.name          = $name,
            s.sector        = $sector,
            s.industry      = $industry,
            s.market_cap    = $market_cap,
            s.pe_ratio      = $pe_ratio,
            s.current_price = $current_price,
            s.aliases       = $aliases,
            s.updated_at    = $updated_at
        """,
        {**snapshot, "updated_at": dt.datetime.now()},
    )
    relate_db.query(
        conn, f"MERGE (sec:{SECTOR} {{name: $sector}})", {"sector": snapshot["sector"]}
    )
    relate_db.query(
        conn,
        f"""
        MATCH (s:{STOCK} {{ticker: $ticker}}), (sec:{SECTOR} {{name: $sector}})
        MERGE (s)-[:{REL_IN_SECTOR}]->(sec)
        """,
        {"ticker": snapshot["ticker"], "sector": snapshot["sector"]},
    )


def write_relationship(conn, source: str, target: str, rel_type: str) -> None:
    """Merge one edge between two already-written stocks.

    ``rel_type`` is interpolated because Cypher does not allow a parameter in a
    relationship type. Every caller passes a module-level constant, never user
    input.
    """
    relate_db.query(
        conn,
        f"""
        MATCH (a:{STOCK} {{ticker: $source}}), (b:{STOCK} {{ticker: $target}})
        MERGE (a)-[:{rel_type}]->(b)
        """,
        {"source": source, "target": target},
    )


def write_person(conn, name: str, role: str, ticker: str) -> None:
    """Merge a person and their WORKS_FOR edge."""
    relate_db.query(
        conn, f"MERGE (p:{PERSON} {{name: $name}}) SET p.role = $role",
        {"name": name, "role": role},
    )
    relate_db.query(
        conn,
        f"""
        MATCH (p:{PERSON} {{name: $name}}), (s:{STOCK} {{ticker: $ticker}})
        MERGE (p)-[:{REL_WORKS_FOR}]->(s)
        """,
        {"name": name, "ticker": ticker},
    )


def main(reset: bool = False) -> int:
    # Must run before the first yf.Ticker(): yfinance keeps SQLite caches in a
    # user-level directory that is not writable here and fails with the opaque
    # "OperationalError: unable to open database file".
    config.configure_yfinance_cache()

    conn = relate_db.connect()
    relate_db.ensure_schema(conn)

    if reset:
        relate_db.wipe(conn)
        print("Store wiped (nodes and relationships only, indexes kept).")

    print(f"Fetching snapshots for {len(config.WATCHLIST)} tickers...")
    written: list[str] = []
    for ticker in config.WATCHLIST:
        snapshot = fetch_snapshot(ticker)
        if snapshot is None:
            continue
        upsert_stock(conn, snapshot)
        written.append(ticker)
        print(
            f"  + {ticker:<6} {snapshot['name'][:28]:<28} "
            f"{snapshot['sector'][:20]:<20} ${snapshot['current_price']:,.2f}"
        )

    if not written:
        print("No snapshots were written; aborting before relationships.")
        conn.close()
        return 1

    print("\nCurated relationships...")
    for source, target in CURATED_COMPETITORS:
        if source in written and target in written:
            write_relationship(conn, source, target, REL_COMPETES_WITH)
    for supplier, customer in CURATED_SUPPLY:
        if supplier in written and customer in written:
            write_relationship(conn, supplier, customer, REL_SUPPLIES_TO)
    for name, role, ticker in CURATED_EXECUTIVES:
        if ticker in written:
            write_person(conn, name, role, ticker)

    print("\nGraph now contains:")
    for label, count in sorted(relate_db.counts(conn).items()):
        print(f"  {label:<12} {count}")
    conn.close()
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Delete all nodes and relationships before loading.",
    )
    raise SystemExit(main(reset=parser.parse_args().reset))
