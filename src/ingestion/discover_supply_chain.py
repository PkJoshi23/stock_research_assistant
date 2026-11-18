"""Crawl the AI supply chain from the seed tickers.

    python -m src.ingestion.discover_supply_chain --depth 1 --seeds NVDA NBIS
    python -m src.ingestion.discover_supply_chain --depth 2 --max-nodes 40
    python -m src.ingestion.discover_supply_chain --dry-run     # print, don't write

Each hop reads one annual report per company and asks which companies it says it
depends on. Names are resolved in two stages - first against companies already in
the graph (so "TSMC" lands on the existing TSM node rather than a duplicate),
then against SEC's registrant map (so "Micron Technology, Inc." becomes MU and
can be expanded next hop). Anything that resolves to neither is kept as an
unlisted `Supplier` node, because a private vendor is still a real dependency.

Writes:
    (:Stock)-[r:SUPPLIES_TO {source:'10-K', discovered:true, hop:n}]->(:Stock)
    (:Supplier {name})-[r:SUPPLIES_TO {source:'10-K', discovered:true}]->(:Stock)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src import config, discovery, entities, relate_db, ticker_directory, universe
from src.ingestion import ingest_sec_relationships as sec
from src.ingestion import ingest_stock_data as stocks
from src.schema import STOCK, SUPPLIER


def sec_suppliers(ticker: str) -> list[str]:
    """Names of companies the annual report says it depends on.

    Tries the US annual report first, then the foreign-private-issuer form:
    TSMC and ASML file 20-F, and a 10-K-only crawl would silently lose them.

    The window is anchored on dependency *sentences* first ("we utilize
    foundries, such as …"), falling back to the broader subsidiary/supplier
    sections: the generic word "suppliers" appears in boilerplate long before it
    appears in the paragraph that names anyone.
    """
    saw_text = False
    for form in config.SEC_ANNUAL_FORMS:
        text = sec.fetch_filing_text(ticker, form)
        if not text:
            continue
        saw_text = True

        # 3 windows tested better than 5: extra windows cost budget on earlier
        # boilerplate and made the model return nothing on filings that worked.
        window = sec.section_windows(text, sec.ANNUAL_DEPENDENCY_PATTERNS, max_windows=3)
        if not window:
            window = "\n\n".join(
                part
                for part in (
                    sec.section_windows(text, sec.ANNUAL_SUBSIDIARY_PATTERNS, max_windows=1),
                    sec.section_windows(text, sec.ANNUAL_SUPPLIER_PATTERNS, max_windows=1),
                )
                if part
            )
        if not window:
            continue

        suppliers = sec.extract_dependencies(window, ticker)
        if suppliers:
            return suppliers

        # Fallback: the Business section. Dependency sentences are the best
        # anchor when they exist, but many 10-Ks name suppliers only inside the
        # product description ("we purchase wafers from ..."), which the
        # sentence patterns never reach. This is a larger slice and a second
        # model call, so it only runs when the cheap pass found nothing.
        business = sec.section_windows(
            text, (r"item\s+1\.?\s*business", r"item\s+1\b"), size=24000, max_windows=1
        )
        if business:
            suppliers = sec.extract_dependencies(business, ticker)
            if suppliers:
                return suppliers

    if not saw_text:
        # Distinguish "we could not read a filing" from "the filing names no
        # suppliers": the first is an outage the crawl must report, the second is
        # a legitimate empty result. Reporting both as empty hid a DNS outage.
        raise RuntimeError("no annual report text available (tried 10-K, 20-F)")
    return []


def ensure_company(conn, ticker: str, *, write: bool = True) -> bool:
    """Make sure ``ticker`` exists as a Stock node so edges have real endpoints.

    With ``write=False`` (dry run) the ticker is still validated against
    yfinance - so bogus resolutions fail here rather than producing dangling
    edges - but nothing is written to the graph.
    """
    existing = relate_db.run(
        conn, f"MATCH (s:{STOCK} {{ticker: $t}}) RETURN s.ticker AS ticker", {"t": ticker}
    )
    if existing:
        return True
    snapshot = stocks.fetch_snapshot(ticker)
    if not snapshot:
        return False
    if write:
        stocks.upsert_stock(conn, snapshot)
    return True


def write_edges(conn, result: discovery.DiscoveryResult) -> int:
    """Persist resolved supplier edges. Idempotent; never relabels curated edges."""
    hops = {node.ticker: node.hop for node in result.nodes}
    written = 0
    for supplier, customer in result.edges:
        # Depth of the edge = the customer's hop + 1. Using the supplier's own
        # hop would be wrong when the supplier is also a seed (NVIDIA is hop 0,
        # yet it is a first-hop supplier of Nebius).
        relate_db.run(
            conn,
            f"""
            MATCH (sup:{STOCK} {{ticker: $sup}}), (cust:{STOCK} {{ticker: $cust}})
            MERGE (sup)-[r:SUPPLIES_TO]->(cust)
            SET r.discovered = true,
                r.source = coalesce(r.source, '10-K'),
                r.confirmedBy = '10-K',
                r.hop = $hop
            """,
            {"sup": supplier, "cust": customer, "hop": hops.get(customer, 0) + 1},
        )
        written += 1
    return written


def write_unresolved(conn, result: discovery.DiscoveryResult) -> int:
    """Keep unlisted vendors as Supplier nodes - they are real dependencies."""
    hops = {node.ticker: node.hop for node in result.nodes}
    written = 0
    for customer, name in result.unresolved:
        relate_db.run(conn, f"MERGE (v:{SUPPLIER} {{name: $name}})", {"name": name})
        relate_db.run(
            conn,
            f"""
            MATCH (v:{SUPPLIER} {{name: $name}}), (cust:{STOCK} {{ticker: $cust}})
            MERGE (v)-[r:SUPPLIES_TO]->(cust)
            SET r.discovered = true,
                r.source = coalesce(r.source, '10-K'),
                r.confirmedBy = '10-K',
                r.hop = $hop
            """,
            {"name": name, "cust": customer, "hop": hops.get(customer, 0) + 1},
        )
        written += 1
    return written


def prune_duplicate_vendors(conn, result: discovery.DiscoveryResult) -> int:
    """Collapse Supplier nodes that are really a tracked stock, or each other.

    Two duplicates appear naturally once discovery runs:

    * a name first stored as an unlisted `Supplier` (by the SEC loader) later
      resolves to a ticker, so the real company now has both a Stock edge and a
      vendor edge - "Micron Technology, Inc" next to "MU";
    * the same vendor comes back spelled differently across filings.

    Removing the vendor copy is safe: the Stock edge carries the same
    relationship with better identity.
    """
    companies = entities.load_companies(conn)
    resolved_suppliers = {supplier for supplier, _ in result.edges}
    removed = 0

    vendors = relate_db.run(conn, f"MATCH (v:{SUPPLIER}) RETURN v.name AS name")
    groups: dict[str, list[str]] = {}
    for row in vendors:
        name = (row["name"] or "").strip()
        if not name:
            continue

        matches = entities.resolve(name, companies, policy="strict")
        ticker = matches[0].ticker if matches else None
        if ticker and ticker in resolved_suppliers:
            relate_db.run(conn, f"MATCH (v:{SUPPLIER} {{name: $n}}) DETACH DELETE v", {"n": name})
            removed += 1
            continue

        key = "".join(ch for ch in name.lower() if ch.isalnum())
        groups.setdefault(key, []).append(name)

    # Same vendor under several spellings: keep the longest label, re-point edges.
    # Similarity rather than equality, because filings truncate or hyphenate names
    # ("Hon Hai Precision Industr" vs "Hon Hai Precision Industry Co., Ltd").
    import difflib

    def same_vendor(left: str, right: str) -> bool:
        """Truncation or near-identical spelling - not merely similar names.

        Filings (and models reading them) truncate long legal names, so
        "Hon Hai Precision Industr" is the same vendor as "Hon Hai Precision
        Industry Co., Ltd". A prefix match is the precise signal for that; the
        ratio covers punctuation/spacing drift.
        """
        a, b = left.lower(), right.lower()
        if a.startswith(b) or b.startswith(a):
            return min(len(a), len(b)) >= 12
        return difflib.SequenceMatcher(None, a, b).ratio() >= 0.9

    remaining = sorted({name for names in groups.values() if len(names) == 1 for name in names})
    merged: list[list[str]] = [names for names in groups.values() if len(names) > 1]
    for name in remaining:
        for group in merged:
            if any(same_vendor(name, member) for member in group):
                group.append(name)
                break
        else:
            merged.append([name])

    for names in merged:
        if len(names) < 2:
            continue
        keeper = max(names, key=len)
        for other in names:
            if other == keeper:
                continue
            relate_db.run(
                conn,
                f"""
                MATCH (dead:{SUPPLIER} {{name: $other}})-[r:SUPPLIES_TO]->(cust:{STOCK})
                MATCH (keep:{SUPPLIER} {{name: $keeper}})
                MERGE (keep)-[new:SUPPLIES_TO]->(cust)
                SET new.discovered = true,
                    new.source = coalesce(new.source, r.source),
                    new.confirmedBy = coalesce(new.confirmedBy, r.confirmedBy),
                    new.hop = coalesce(new.hop, r.hop)
                """,
                {"other": other, "keeper": keeper},
            )
            relate_db.run(
                conn, f"MATCH (dead:{SUPPLIER} {{name: $n}}) DETACH DELETE dead", {"n": other}
            )
            removed += 1

    return removed


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", nargs="*", help="override the seed universe")
    parser.add_argument("--depth", type=int, default=config.DISCOVERY_MAX_DEPTH,
                        help="hops to expand (1 = read the seeds' filings only)")
    parser.add_argument("--max-nodes", type=int, default=config.DISCOVERY_MAX_NODES,
                        help="hard cap on companies in the crawl")
    parser.add_argument("--refresh-directory", action="store_true",
                        help="re-download SEC's registrant map")
    parser.add_argument("--dry-run", action="store_true", help="crawl but do not write")
    args = parser.parse_args(argv)

    seeds = args.seeds or universe.SEED_TICKERS
    # yfinance keeps SQLite caches in a user-level directory that is not
    # writable here; the snapshot fetch inside ensure_company needs this first.
    config.configure_yfinance_cache()
    conn = relate_db.connect()
    relate_db.ensure_schema(conn)

    try:
        directory = ticker_directory.TickerDirectory.load(refresh=args.refresh_directory)

        def resolve(name: str) -> str | None:
            # Graph aliases first ("TSMC" -> TSM), then EDGAR's registrant map.
            return entities.resolve_tracked_stock(conn, name) or directory.lookup(name)

        print(f"Seeds ({len(seeds)}): {', '.join(seeds)}")
        print(f"Depth {args.depth}, node budget {args.max_nodes}\n")

        result = discovery.discover(
            seeds=seeds,
            extract_suppliers=sec_suppliers,
            resolve_ticker=resolve,
            ensure_company=lambda ticker: ensure_company(conn, ticker, write=not args.dry_run),
            max_depth=args.depth,
            max_nodes=args.max_nodes,
            on_event=lambda message: print("  " + message),
        )

        print("\nDiscovered tree")
        for line in discovery.format_tree(result):
            print("  " + line)

        resolved = len(result.edges)
        print(
            f"\nCompanies: {len(result.nodes)} (expanded {result.expanded})"
            f" | resolved edges: {resolved}"
            f" | unlisted vendors: {len(result.unresolved)}"
        )
        if result.edges:
            print("\nResolved supplier -> customer edges (review before writing):")
            for supplier, customer in sorted(result.edges):
                print(f"  {supplier:<6} -> {customer}")
        if result.unresolved:
            print("\nUnresolved vendors (kept as Supplier nodes):")
            for customer, name in sorted(result.unresolved):
                print(f"  {customer:<6} <- {name}")
        if result.deferred:
            print(f"Deferred by budget: {len(result.deferred)}")
        if result.exhausted:
            print("Node budget reached - raise --max-nodes or lower --depth.")
        if result.cycles:
            print("\nCircular dependencies found:")
            for cycle in result.cycles:
                print("  " + " -> ".join(cycle + [cycle[0]]))
        else:
            print("\nNo circular dependencies within this crawl depth.")
        if result.errors:
            print("\nSkipped:")
            for error in result.errors:
                print("  " + error)

        if args.dry_run:
            print("\nDry run: nothing written.")
            return 0

        # Both endpoints must exist as Stock nodes or the MERGE silently matches
        # nothing - that is how AVGO -> ANET was lost once: ANET was a seed in a
        # previous dry run, so it was never ingested.
        for ticker in sorted({t for edge in result.edges for t in edge}):
            ensure_company(conn, ticker)

        edges = write_edges(conn, result)
        vendors = write_unresolved(conn, result)
        pruned = prune_duplicate_vendors(conn, result)
        print(f"\nWrote {edges} supplier edges and {vendors} unlisted-vendor edges.")
        if pruned:
            print(
                f"Pruned {pruned} duplicate vendor node(s) "
                "(covered by a tracked stock, or merged spellings)."
            )
        print("Graph now contains:")
        for label, count in sorted(relate_db.counts(conn).items()):
            print(f"  {label:<12} {count}")
        return 0 if result.nodes else 1
    finally:
        conn.close()


if __name__ == "__main__":
    raise SystemExit(main())
