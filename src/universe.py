"""The hardcoded seed universe: the AI circular economy.

This is the only list in the project that stays hand-authored on purpose. The
stocks are the *entry points*; everything downstream - who supplies whom, how
deep the chain goes - is discovered by ``src/discovery.py`` from SEC filings.

Why seeds and not a full universe? Because the interesting structure is the
loop around AI capex:

    compute buyers (neoclouds, hyperscalers)
        -> chip designers (NVDA, AMD, MRVL, AVGO)
        -> foundry/packaging (TSM, ASML, AMAT)
        -> memory and optics (MU, CRDO, ALAB)
        -> data-centre physical layer (VRT, ANET, SMCI)
        -> financed by the same buyers  <-- closes the loop

Seeding both ends of that loop (a buyer like Nebius *and* its suppliers) is what
makes the crawler find both directions of an edge: Nebius's 10-K names NVIDIA as
a supplier, NVIDIA's names TSMC and Micron.

Add or remove tickers here; the crawler takes it from there.
"""

from __future__ import annotations

#: Hardcoded entry points, grouped for readability. The crawler flattens these.
SEED_GROUPS: dict[str, list[str]] = {
    # GPU/accelerator design and networking silicon
    "compute": ["NVDA", "AMD", "MRVL", "AVGO", "CRDO", "ALAB"],
    # Foundry, equipment, memory
    "fabs": ["TSM", "ASML", "AMAT", "MU"],
    # Data-centre physical layer
    "infrastructure": ["VRT", "ANET", "SMCI"],
    # Neoclouds and hyperscalers: the demand side of the loop
    "buyers": ["NBIS", "CRWV", "ORCL", "MSFT", "AMZN"],
}

#: Flattened, de-duplicated seed list used by the crawler.
SEED_TICKERS: list[str] = list(
    dict.fromkeys(ticker for group in SEED_GROUPS.values() for ticker in group)
)
