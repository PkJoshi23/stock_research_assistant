"""Central configuration for the AI Stock Research Assistant.

Every path, model name and credential key lives here so that no other module
hardcodes a location. Keeping this single-sourced matters for a hackathon
entry point: the Streamlit app, the ingestion scripts and the agent all import
the same settings object graph.

Design rules enforced by this module
------------------------------------
1. The database path is resolved from the project root, never from the current
   working directory. ``streamlit run`` has its own idea about CWD, and that
   must not change where the graph lives.
2. There is exactly one database artifact path (``DB_PATH``). Relate's
   persistence model differs from SQLite: it opens either a *directory*
   (WAL + Parquet shards) or a single ``.rl`` file. It does not produce a
   ``.db`` file, so the path below uses the ``.rl`` suffix.
3. Secrets are read from the environment only. Nothing here embeds a key.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

from dotenv import load_dotenv

# -----------------------------------------------------------------------------
# Paths
# -----------------------------------------------------------------------------

#: Repository root: ``.../stock-research-agent``
PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent

#: Everything persistent (the graph, plus any local caches) lives here.
DATA_DIR: Path = PROJECT_ROOT / "data"

#: The Relate store, as a single self-contained file.
#:
#: ``relate.connect(<path>.rl)`` keeps the WAL, the snapshot and the catalog
#: in one file. The alternative is a directory store
#: (``relate.connect("./data/stock_graph")``), which spreads Parquet shards
#: inside a folder. The single-file form is chosen because it makes the
#: hackathon demo trivially portable: copy one file, ship the graph.
DB_PATH: Path = DATA_DIR / "stock_graph.rl"

#: yfinance keeps its cookie and timezone databases in a user-level cache dir.
#: That user-level directory is not writable here, and the
#: failure surfaces as the confusing ``OperationalError: unable to open
#: database file`` from the SQLite layer. yfinance exposes no environment
#: variable for this, so it must be redirected in code - see
#: :func:`configure_yfinance_cache`.
YFINANCE_CACHE_DIR: Path = PROJECT_ROOT / ".cache" / "yfinance"

#: Load ``.env`` from the project root if present. ``override=False`` means a
#: real environment variable always wins over the file - important on judges'
#: machines and in CI, where the file must not silently mask real config.
load_dotenv(PROJECT_ROOT / ".env", override=False)


def _setting(name: str, default: str | None = None) -> str | None:
    """Read a setting from the environment, then from Streamlit secrets.

    On Streamlit Community Cloud there is no ``.env``: secrets are injected into
    ``st.secrets``. Checking the environment first keeps CLI and local runs
    unchanged, and consulting ``st.secrets`` only when Streamlit is already
    imported avoids paying its import cost in every ingestion script.
    """
    value = os.getenv(name)
    if value:
        return value

    streamlit = sys.modules.get("streamlit")
    if streamlit is None:
        return default
    try:
        if name in streamlit.secrets:
            return str(streamlit.secrets[name])
    except Exception:  # noqa: BLE001 - no secrets file configured, or no runtime
        pass
    return default


# -----------------------------------------------------------------------------
# Credentials
# -----------------------------------------------------------------------------

#: DeepSeek key. The DeepSeek API is OpenAI-compatible, so the openai SDK
#: drives it unchanged; only the base_url and model names differ.
DEEPSEEK_API_KEY: str | None = _setting("DEEPSEEK_API_KEY")

#: DeepSeek's OpenAI-compatible endpoint.
DEEPSEEK_BASE_URL: str = _setting("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1") or "https://api.deepseek.com/v1"

#: Reasoning / generation model used for the 4 pipeline steps.
DEEPSEEK_CHAT_MODEL: str = _setting("DEEPSEEK_CHAT_MODEL", "deepseek-chat") or "deepseek-chat"

#: Ingestion watchlist - the tickers this demo tracks.
#:
#: Deliberately small: the graph is the product, and ten well-connected
#: companies demo far better than five hundred disconnected ones. Override with
#: a comma-separated WATCHLIST environment variable.
WATCHLIST: list[str] = [
    ticker.strip().upper()
    for ticker in os.getenv(
        "WATCHLIST",
        "AAPL,MSFT,NVDA,TSM,TSLA,GOOGL,AMD,INTC,AMZN,META,F",
    ).split(",")
    if ticker.strip()
]


# -----------------------------------------------------------------------------
# Agent behaviour
# -----------------------------------------------------------------------------

#: Hard cap on Step 4 auto-correction attempts. A retry loop without a cap is
#: an infinite bill: each failed attempt is another LLM round trip.
MAX_CYPHER_RETRIES: int = int(os.getenv("MAX_CYPHER_RETRIES", "3"))

#: Rows returned per query by default. Guards the UI against a runaway
#: ``MATCH (n) RETURN n`` that would otherwise hang the Streamlit rerun.
DEFAULT_RESULT_LIMIT: int = int(os.getenv("DEFAULT_RESULT_LIMIT", "50"))

#: Set this to a truthy string to print every prompt and generated Cypher.
DEBUG_PROMPTS: bool = os.getenv("DEBUG_PROMPTS", "").lower() in {"1", "true", "yes"}


# -----------------------------------------------------------------------------
# HTTP
# -----------------------------------------------------------------------------

#: Timeout for every outbound request (SEC filings, registrant map, quotes).
#: EDGAR serves 1-20 MB documents, so this is generous but bounded.
HTTP_TIMEOUT_S: float = float(os.getenv("HTTP_TIMEOUT_S", "20"))


# -----------------------------------------------------------------------------
# SEC EDGAR
# -----------------------------------------------------------------------------
#
# EDGAR needs no API key, but it *does* require a User-Agent that identifies you
# with a real contact address, and it rate-limits to ~10 requests/second.
# Requests without a valid contact are rejected.

#: Product/company name sent to EDGAR.
SEC_COMPANY_NAME: str = _setting("SEC_COMPANY_NAME", "graphbase-hackathon") or "graphbase-hackathon"

#: Contact address sent to EDGAR. No default on purpose: a placeholder address
#: is both useless to the SEC and an unnecessary piece of personal data in git.
SEC_CONTACT_EMAIL: str = _setting("SEC_CONTACT_EMAIL", "") or ""

#: Where sec-edgar-downloader caches filings.
#:
#: Careful: the downloader appends its own ``sec-edgar-filings/`` directory to
#: whatever folder it is given, so this is the *parent*. The resulting layout is
#: ``.cache/edgar/sec-edgar-filings/<ticker>/<form>/<accession>/``.
SEC_CACHE_DIR: Path = PROJECT_ROOT / ".cache" / "edgar"

#: Characters of filing text sent to the LLM, measured from the section heading.
#: A full 10-K is far too large, and the first N characters of a filing are the
#: cover page, which contains no directors or subsidiaries.
SEC_WINDOW_CHARS: int = int(os.getenv("SEC_WINDOW_CHARS", "14000"))

#: Filings to read. DEF 14A is the proxy statement (officers/directors);
#: 10-K is the annual report (subsidiaries, disclosed suppliers).
SEC_PROXY_FORM: str = "DEF 14A"
SEC_ANNUAL_FORM: str = "10-K"

#: Annual reports to try, in order, when mining suppliers.
#:
#: Foreign private issuers file 20-F instead of 10-K - TSMC and ASML both do - so
#: a 10-K-only crawl silently loses exactly the companies that sit deepest in the
#: supply chain.
SEC_ANNUAL_FORMS: tuple[str, ...] = ("10-K", "20-F")


# -----------------------------------------------------------------------------
# Supply-chain discovery
# -----------------------------------------------------------------------------

#: Hops to expand in one crawl. 1 = read the seeds' filings only; 2 = also read
#: the filings of the suppliers discovered at hop 1 (expensive, and where the
#: interesting multi-hop chains start).
DISCOVERY_MAX_DEPTH: int = int(os.getenv("DISCOVERY_MAX_DEPTH", "1"))

#: Hard cap on companies per crawl. Supply chains fan out fast; the crawl stops
#: cleanly at the cap rather than running away.
DISCOVERY_MAX_NODES: int = int(os.getenv("DISCOVERY_MAX_NODES", "25"))


# -----------------------------------------------------------------------------
# Startup guards
# -----------------------------------------------------------------------------


def require_secret(name: str) -> str:
    """Return a required credential or fail with an actionable message.

    Called at the point of use rather than at import time, so that tooling
    (tests, schema introspection, linting) can import this module on a machine
    with no ``.env`` at all.
    """
    value = os.getenv(name)
    if not value:
        raise RuntimeError(
            f"{name} is not set. Copy .env.example to .env and fill it in, "
            f"or export {name} in your shell."
        )
    return value


def configure_yfinance_cache() -> Path:
    """Redirect yfinance's SQLite caches into a writable, project-local dir.

    yfinance stores a cookie database and a timezone database on disk and
    refuses to start when the user-level cache directory is not writable:

        OperationalError: unable to open database file

    That message says nothing about caches, so it is worth calling this once at
    startup - the live-price tool does - instead of debugging it twice.

    yfinance is imported *inside* the function on purpose: it is only needed by
    the live-snapshot tool, and keeping it out of module import cost matters for
    the Streamlit app and for any tooling that only touches the graph.

    Returns the cache directory in use, so callers can log it.
    """
    import yfinance as yf  # local import: heavy, and only needed here

    YFINANCE_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    # Both live on different import paths: the timezone helper is re-exported at
    # package level, the cookie-cache helper is not.
    yf.set_tz_cache_location(str(YFINANCE_CACHE_DIR))
    yf.cache.set_cache_location(str(YFINANCE_CACHE_DIR))
    return YFINANCE_CACHE_DIR
