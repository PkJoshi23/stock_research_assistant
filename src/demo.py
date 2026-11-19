"""One-command demo readiness check.

The question this answers: *"can I demo this right now, and if not, what do I
run?"* Everything else in the project assumes the graph is already populated -
a fresh clone is an empty store, and the failure mode is an app that starts
cleanly and answers "no data matching that" to every question.

    python -m src.demo            # check
    python -m src.demo --seed     # build the graph if it is empty
    python -m src.demo --json     # machine-readable

Exit code 0 = ready, 1 = not ready (empty graph or missing credentials), so a
deploy health check can gate on it.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import config, relate_db
from src.schema import PERSON, QUERY, SECTOR, STOCK, SUBSIDIARY, SUPPLIER

#: Questions that exercise each pillar of the graph - deliberately the same
#: shapes the README advertises.
SAMPLE_QUESTIONS = (
    "Which companies compete with Microsoft and in which sector?",
    "Who supplies Nvidia?",
    "What subsidiaries does Nvidia own?",
    "Who is on Nvidia's board of directors?",
    "What is NVDA trading at right now?",
    "How has Tesla's stock performed since 2020?",
)


def _line(ok: bool, label: str, detail: str) -> str:
    return f"{'[ ok ]' if ok else '[ !! ]'}  {label:<26} {detail}"


def build_report(session) -> dict:
    """Collect everything the readiness check needs, without printing."""
    counts = session.counts()
    indexes = session.indexes()

    return {
        "db_path": str(config.DB_PATH),
        "store_exists": config.DB_PATH.exists(),
        "counts": counts,
        "indexes": indexes,
        "nodes": sum(counts.values()),
        "has_llm_key": bool(config.DEEPSEEK_API_KEY),
        "has_sec_contact": bool(config.SEC_CONTACT_EMAIL),
    }


def render(report: dict) -> list[str]:
    """Format the report as the checklist a human reads before demoing."""
    counts = report["counts"]
    lines = [
        "Project readiness",
        "-----------------",
        _line(report["store_exists"], "graph store", report["db_path"]),
        _line(bool(counts), "populated", f"{report['nodes']} nodes" if counts else "EMPTY - run the seed step"),
    ]

    for label in (STOCK, SECTOR, PERSON, SUBSIDIARY, SUPPLIER, QUERY):
        if counts.get(label):
            lines.append(f"         {label:<22} {counts[label]}")

    lines.append(
        _line(bool(report["indexes"]), "indexes", ", ".join(report["indexes"]) or "none")
    )
    lines.append(
        _line(
            report["has_llm_key"],
            "DEEPSEEK_API_KEY",
            "set" if report["has_llm_key"] else "MISSING - graph questions still work via the offline fallback",
        )
    )
    lines.append(
        _line(
            report["has_sec_contact"],
            "SEC_CONTACT_EMAIL",
            "set" if report["has_sec_contact"] else "not set - SEC ingestion is optional",
        )
    )

    return lines


def seed() -> int:
    """Build the graph in dependency order: stocks, then SEC filings."""
    from src.ingestion import (
            ingest_sec_relationships,
        ingest_stock_data,
    )

    print("Seeding: stocks -> SEC filings\n")
    ingest_stock_data.main()
    try:
        ingest_sec_relationships.main()
    except RuntimeError as exc:
        # Missing SEC contact is not fatal: everything else still works.
        print(f"\nSkipping SEC filings: {exc}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", action="store_true", help="build the graph if needed")
    parser.add_argument(
        "--discover",
        action="store_true",
        help="also crawl the seed universe for suppliers (slow; needs SEC_CONTACT_EMAIL)",
    )
    parser.add_argument("--depth", type=int, default=config.DISCOVERY_MAX_DEPTH)
    parser.add_argument("--max-nodes", type=int, default=config.DISCOVERY_MAX_NODES)
    parser.add_argument("--json", action="store_true", help="machine-readable output")
    args = parser.parse_args(argv)

    session = relate_db.GraphSession()
    try:
        report = build_report(session)
        empty = report["nodes"] == 0

        if args.seed and empty:
            session.close()
            seed()
            session = relate_db.GraphSession()
            report = build_report(session)
            empty = report["nodes"] == 0

        if args.discover:
            # The crawler opens its own connection, and Relate grants one
            # writable lease per store - so this session must be closed first.
            session.close()
            from src.ingestion import discover_supply_chain

            discover_supply_chain.main(
                ["--depth", str(args.depth), "--max-nodes", str(args.max_nodes)]
            )
            session = relate_db.GraphSession()
            report = build_report(session)
            empty = report["nodes"] == 0

        if args.json:
            print(json.dumps(report, indent=2, default=str))
        else:
            print("\n".join(render(report)))
            print()
            if empty:
                print("The graph is empty. Build it with:\n")
                print("    python -m src.demo --seed\n")
                print("or step by step:\n")
                print("    python -m src.ingestion.ingest_stock_data")
                print("    python -m src.ingestion.ingest_sec_relationships")
            else:
                print("Ready. Ask from the CLI:")
                print('    python -m src.agent.master_agent "Who supplies Nvidia?"')
                print("\nOr start the UI:")
                print("    streamlit run src/ui/app.py")
                print("\nGrow the graph from the seed universe (needs SEC_CONTACT_EMAIL):")
                print("    python -m src.demo --discover --depth 1")
                print("    python -m src.ingestion.discover_supply_chain --dry-run --depth 2")
                print("\nTry:")
                for question in SAMPLE_QUESTIONS:
                    print(f"  · {question}")

        return 0 if not empty else 1
    finally:
        session.close()


if __name__ == "__main__":
    raise SystemExit(main())
