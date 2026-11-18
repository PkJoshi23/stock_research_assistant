"""Breadth-first supply-chain discovery.

The hand-curated relationship lists only ever covered the seed tickers. This
module replaces that with a crawl: start from the seeds, read each company's
annual filing, ask who it says it depends on, resolve those names to tickers,
and repeat one hop at a time.

Design constraints, all learned from the shape of the data:

* **Expansion is expensive** (a 10-K is 10+ MB and a model call), so the engine
  takes its extractor and resolver as callables. Tests drive it with fakes; the
  CLI wires in SEC + the EDGAR directory.
* **The graph must stay deterministic.** A crawl that produces a different graph
  each run cannot be diffed or tested, so the queue is FIFO, suppliers are
  sorted, and cycles are canonicalised.
* **Budget, not optimism.** Supply chains fan out fast (NVIDIA -> TSMC -> ...).
  Every run has a node budget and a depth limit, and stopping early is a normal
  outcome (`exhausted`), not an error.
* **Cycles are the point.** In this industry the loop closes - a customer is
  also an investor or a supplier. The engine reports cycles explicitly because
  "NVDA -> Nebius -> NVDA" is the story, not a curiosity.
"""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Callable

#: Safety cap on cycle enumeration. Enough to show the loop, small enough that a
#: pathological graph cannot hang the CLI.
MAX_CYCLE_LENGTH = 6
MAX_CYCLES = 20


@dataclass(frozen=True)
class DiscoveredNode:
    """One company in the crawl."""

    ticker: str
    hop: int
    parent: str | None


@dataclass
class DiscoveryResult:
    """Everything the crawl learned, before anything is written to the graph."""

    seeds: list[str]
    nodes: list[DiscoveredNode] = field(default_factory=list)
    #: (supplier, customer) - the direction stored in the graph.
    edges: list[tuple[str, str]] = field(default_factory=list)
    #: (customer, supplier name) pairs that could not be resolved to a ticker.
    unresolved: list[tuple[str, str]] = field(default_factory=list)
    #: Supplier pairs found but not added because the node budget ran out.
    deferred: list[tuple[str, str]] = field(default_factory=list)
    cycles: list[list[str]] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    exhausted: bool = False
    expanded: int = 0


def _canonical_cycle(cycle: list[str]) -> tuple[str, ...]:
    """Rotate a cycle so it starts at its smallest ticker - makes it hashable."""
    body = cycle[:-1] if cycle and cycle[0] == cycle[-1] else cycle
    if not body:
        return tuple(cycle)
    start = body.index(min(body))
    rotated = body[start:] + body[:start]
    return tuple(rotated)


def find_cycles(edges: list[tuple[str, str]]) -> list[list[str]]:
    """Return simple cycles in the supplier -> customer graph."""
    adjacency: dict[str, set[str]] = defaultdict(set)
    for supplier, customer in edges:
        adjacency[supplier].add(customer)
        adjacency.setdefault(customer, set())

    found: set[tuple[str, ...]] = set()

    def walk(node: str, path: list[str]) -> None:
        if len(path) > MAX_CYCLE_LENGTH:
            return
        for nxt in sorted(adjacency.get(node, ())):
            if nxt in path:
                cycle = _canonical_cycle(path[path.index(nxt):] + [nxt])
                if len(cycle) >= 2:
                    found.add(cycle)
                continue
            if len(found) < MAX_CYCLES * 5:  # bound the search, not the result
                walk(nxt, path + [nxt])

    for start in sorted(adjacency):
        walk(start, [start])
        if len(found) >= MAX_CYCLES * 5:
            break

    ordered = sorted(found, key=lambda cycle: (len(cycle), cycle))
    return [list(cycle) for cycle in ordered[:MAX_CYCLES]]


def discover(
    *,
    seeds: list[str],
    extract_suppliers: Callable[[str], list[str]],
    resolve_ticker: Callable[[str], str | None],
    ensure_company: Callable[[str], bool] | None = None,
    max_depth: int = 1,
    max_nodes: int = 25,
    on_event: Callable[[str], None] | None = None,
) -> DiscoveryResult:
    """Crawl supplier edges from ``seeds``.

    ``extract_suppliers(ticker)`` returns company names the filing names.
    ``resolve_ticker(name)`` maps a name to a ticker, or ``None`` for an unlisted
    vendor. ``ensure_company(ticker)`` is called for each newly discovered ticker
    (the CLI uses it to ingest the snapshot) and may veto expansion by returning
    ``False``.
    """
    log = on_event or (lambda message: None)
    clean_seeds = list(dict.fromkeys(str(s).strip().upper() for s in seeds if str(s).strip()))

    result = DiscoveryResult(seeds=clean_seeds)
    queue: deque[tuple[str, int, str | None]] = deque((ticker, 0, None) for ticker in clean_seeds)
    seen: set[str] = set()

    while queue:
        ticker, hop, parent = queue.popleft()
        if ticker in seen:
            continue
        if len(seen) >= max_nodes:
            result.exhausted = True
            break

        seen.add(ticker)
        result.nodes.append(DiscoveredNode(ticker, hop, parent))
        log(f"hop {hop}: {ticker}" + (f" (via {parent})" if parent else " (seed)"))

        if ensure_company is not None:
            try:
                if not ensure_company(ticker):
                    result.errors.append(f"{ticker}: could not be ingested, not expanded")
                    continue
            except Exception as exc:  # noqa: BLE001 - one bad ticker must not stop the crawl
                result.errors.append(f"{ticker}: ingest failed ({type(exc).__name__}: {exc})")
                continue

        if hop >= max_depth:
            continue

        try:
            names = extract_suppliers(ticker)
        except Exception as exc:  # noqa: BLE001
            result.errors.append(f"{ticker}: extraction failed ({type(exc).__name__}: {exc})")
            continue

        result.expanded += 1
        for name in sorted({str(n).strip() for n in names if str(n).strip()}):
            target = resolve_ticker(name)
            if not target:
                result.unresolved.append((ticker, name))
                continue
            target = target.strip().upper()
            if target == ticker:
                continue  # a filing naming itself is not a dependency

            edge = (target, ticker)
            if edge not in result.edges:
                result.edges.append(edge)

            if target in seen:
                continue
            if len(seen) + len(queue) >= max_nodes:
                result.deferred.append(edge)
                result.exhausted = True
                continue
            queue.append((target, hop + 1, ticker))

    result.cycles = find_cycles(result.edges)
    return result


def format_tree(result: DiscoveryResult) -> list[str]:
    """Render the crawl as an indented tree, for the CLI and logs."""
    children: dict[str | None, list[DiscoveredNode]] = defaultdict(list)
    for node in result.nodes:
        children[node.parent].append(node)

    lines: list[str] = []

    def walk(parent: str | None, depth: int) -> None:
        for node in sorted(children.get(parent, []), key=lambda n: (n.hop, n.ticker)):
            prefix = "  " * depth + ("└─ " if depth else "")
            lines.append(f"{prefix}{node.ticker} (hop {node.hop})")
            walk(node.ticker, depth + 1)

    walk(None, 0)
    return lines
