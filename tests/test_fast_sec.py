"""The fast SEC path: anchor slicing, deterministic extraction, and writing.

Everything here runs without network: HTTP is monkeypatched, so the tests pin the
expensive decisions (which document, which slice, which names) rather than the
bytes SEC happens to serve today.
"""

from __future__ import annotations

import pytest

from src import relate_db
from src.ingestion import fast_sec
from src.schema import STOCK, SUBSIDIARY, SUPPLIER

FILLER = "We design and sell optical components for data-centre networks. " * 12

FILING_HTML = f"""
<html><head><style>.x {{ color: red; }}</style><script>var hidden = 1;</script></head>
<body>
<p>Table of Contents</p>
<p>Item 1. Business .... 4</p>
<p>Item 1A. Risk Factors .... 12</p>
<table><tr><td>Item 1. Business</td><td>Item 1A. Risk Factors</td></tr></table>
<p><b>Item 1. Business</b></p>
<p>{FILLER}</p>
<p>We utilize foundries, such as Taiwan Semiconductor Manufacturing Company Limited and
Samsung Electronics Co., Ltd., to produce our wafers. We purchase memory from SK Hynix
Inc., Micron Technology, Inc., and Samsung.</p>
<p><b>Item 1A. Risk Factors</b></p>
<p>We depend on a limited number of suppliers.</p>
</body></html>
"""

EXHIBIT_HTML = """
<html><body>
<p>Exhibit 21 - Subsidiaries of the Registrant</p>
<ul>
  <li>Applied Optoelectronics (Suzhou) Co., Ltd. - China</li>
  <li>Prime World International Holdings Ltd. - British Virgin Islands</li>
</ul>
</body></html>
"""


# -- text handling -----------------------------------------------------------

@pytest.fixture(autouse=True)
def _sec_contact(monkeypatch):
    """EDGAR needs a contact; the fast path now refuses to start without one.

    Every test here except the preflight test wants a configured address.
    """
    from src import config

    monkeypatch.setattr(config, "SEC_CONTACT_EMAIL", "test@example.com")


def test_de_tag_strips_markup_scripts_and_entities():
    text = fast_sec.de_tag("<p>Alpha &amp; Beta</p><script>bad()</script><b>Gamma</b>")

    assert text == "Alpha & Beta Gamma"


def test_business_section_skips_the_table_of_contents():
    text = fast_sec.de_tag(FILING_HTML)

    section = fast_sec.business_section(text)

    assert section.startswith("Item 1. Business")
    assert "Table of Contents" not in section
    assert "Risk Factors" not in section
    assert "Taiwan Semiconductor" in section


def test_business_section_falls_back_when_no_risk_section_follows():
    section = fast_sec.business_section("Item 1. Business. " + ("x " * 400), size=200)

    assert section.startswith("Item 1. Business")
    assert len(section) <= 200


def test_business_section_returns_empty_when_no_anchor():
    assert fast_sec.business_section("no headings here") == ""


# -- deterministic extraction ------------------------------------------------


def test_deterministic_suppliers_reads_list_sentences():
    section = fast_sec.de_tag(FILING_HTML)

    names = fast_sec.deterministic_suppliers(section)

    assert "Taiwan Semiconductor Manufacturing Company Limited" in names
    assert any(name.startswith("Samsung Electronics Co") for name in names)
    assert "SK Hynix Inc" in names
    assert any(name.startswith("Micron") for name in names)


def test_deterministic_suppliers_ignores_generic_categories():
    names = fast_sec.deterministic_suppliers(
        "We rely on suppliers such as cloud providers and various vendors."
    )

    assert names == []


def test_category_phrases_are_not_company_names():
    """Real NVDA 10-K output: acronym categories and 'as' phrases got through."""
    section = (
        "We utilize foundries, such as Taiwan Semiconductor Manufacturing Company "
        "Limited, and ODMs. We sell to CSPs, OEMs, AI model makers and OEMs "
        "who resell to NVIDIA as direct customers."
    )

    names = fast_sec.deterministic_suppliers(section)

    assert "Taiwan Semiconductor Manufacturing Company Limited" in names
    for junk in ("ODMs", "CSPs", "OEMs", "AI model makers", "NVIDIA as direct customers"):
        assert junk not in names


def test_deterministic_suppliers_returns_empty_without_lists():
    assert fast_sec.deterministic_suppliers("We sell optical components.") == []


def test_exhibit21_parsing_handles_lists_and_headers():
    names = fast_sec.parse_exhibit21(EXHIBIT_HTML)

    assert "Applied Optoelectronics (Suzhou) Co., Ltd" in names
    assert "Prime World International Holdings Ltd" in names
    assert not any("Exhibit 21" in name for name in names)


def test_exhibit21_parsing_drops_filenames_and_column_headers():
    """CRWV's real exhibit included its own filename and the table headers."""
    raw = """
    <html><body>
    <p>ex211-coreweaveincx10xk.htm</p>
    <p>Document</p>
    <table>
      <tr><td>Name of Subsidiary</td><td>Jurisdiction</td></tr>
      <tr><td>CoreWeave Compute Acquisition Co. II, LLC</td><td>Delaware</td></tr>
      <tr><td>CoreWeave Compute Acquisition Co. III, LLC</td><td>Delaware</td></tr>
    </table>
    </body></html>
    """

    names = fast_sec.parse_exhibit21(raw)

    assert names == [
        "CoreWeave Compute Acquisition Co. II, LLC",
        "CoreWeave Compute Acquisition Co. III, LLC",
    ]


# -- filing discovery --------------------------------------------------------


def test_filing_urls_are_built_from_cik_and_accession():
    filing = fast_sec.Filing(
        form="10-K",
        accession="0001234567-26-000001",
        primary_document="aaoi-10k.htm",
        filed="2026-02-20",
        cik="0001234567",
    )

    assert filing.document_url == (
        "https://www.sec.gov/Archives/edgar/data/1234567/000123456726000001/aaoi-10k.htm"
    )
    assert filing.index_url == (
        "https://www.sec.gov/Archives/edgar/data/1234567/000123456726000001/index.json"
    )


def test_latest_filing_picks_the_newest_annual_report(monkeypatch):
    payload = {
        "filings": {
            "recent": {
                "form": ["8-K", "10-Q", "10-K", "10-K"],
                "accessionNumber": ["0001-01", "0001-02", "0001-03", "0001-04"],
                "primaryDocument": ["a.htm", "b.htm", "old10k.htm", "new10k.htm"],
                "filingDate": ["2026-01-01", "2026-02-01", "2025-02-20", "2026-02-20"],
            }
        }
    }
    monkeypatch.setattr(fast_sec, "get_json", lambda url: payload)

    filing = fast_sec.latest_filing("0001234567")

    assert filing is not None
    assert filing.form == "10-K"
    assert filing.primary_document == "new10k.htm"


def test_latest_filing_falls_back_to_20f(monkeypatch):
    payload = {
        "filings": {
            "recent": {
                "form": ["20-F"],
                "accessionNumber": ["0001-05"],
                "primaryDocument": ["tsm-20f.htm"],
                "filingDate": ["2026-03-01"],
            }
        }
    }
    monkeypatch.setattr(fast_sec, "get_json", lambda url: payload)

    filing = fast_sec.latest_filing("0001046179")

    assert filing is not None and filing.form == "20-F"


def test_resolve_cik_uses_the_registrant_map(monkeypatch):
    monkeypatch.setattr(
        fast_sec.ticker_directory,
        "load_directory",
        lambda **kwargs: [
            {"ticker": "AAOI", "title": "APPLIED OPTOELECTRONICS INC", "cik": "0001158114"}
        ],
    )

    assert fast_sec.resolve_cik("aaoi") == ("0001158114", "APPLIED OPTOELECTRONICS INC")
    assert fast_sec.resolve_cik("UNKNOWN") is None


# -- end-to-end (dry run) ----------------------------------------------------


def _fake_filing() -> fast_sec.Filing:
    return fast_sec.Filing(
        form="10-K",
        accession="0001158114-26-000010",
        primary_document="aaoi-20251231.htm",
        filed="2026-02-20",
        cik="0001158114",
    )


def _patch_fetch(monkeypatch, *, html_text: str = FILING_HTML, with_exhibit: bool = True):
    monkeypatch.setattr(
        fast_sec, "resolve_cik", lambda ticker: ("0001158114", "APPLIED OPTOELECTRONICS INC")
    )
    monkeypatch.setattr(fast_sec, "latest_filing", lambda cik, forms=("10-K", "20-F"): _fake_filing())
    monkeypatch.setattr(
        fast_sec, "get_text", lambda url, max_bytes=0: (html_text, len(html_text), False)
    )
    monkeypatch.setattr(
        fast_sec, "exhibit21_names", lambda filing: fast_sec.parse_exhibit21(EXHIBIT_HTML)
        if with_exhibit else []
    )
    # Never touch the real document cache from tests.
    monkeypatch.setattr(fast_sec, "_load_cached_text", lambda filing: "")
    monkeypatch.setattr(fast_sec, "_store_cached_text", lambda filing, text: None)


def test_ingest_dry_run_reports_without_writing(seeded, monkeypatch):
    _patch_fetch(monkeypatch)

    report = fast_sec.ingest(seeded, "AAOI", write=False, allow_model=False)

    assert report.cik == "0001158114"
    assert report.form == "10-K"
    assert report.section_chars > 0
    assert report.subsidiaries  # Exhibit 21 parsed
    assert any(name.startswith("Taiwan Semiconductor") for name in report.suppliers)
    assert report.wrote is False
    assert seeded.run(f"MATCH (s:{STOCK} {{ticker: 'AAOI'}}) RETURN s.ticker") == []


def test_ingest_writes_company_suppliers_and_subsidiaries(seeded, monkeypatch):
    _patch_fetch(monkeypatch)
    monkeypatch.setattr(fast_sec.stocks, "fetch_snapshot", lambda ticker: None)
    # No aliases and no registrant match: every name stays an unlisted vendor.
    monkeypatch.setattr(fast_sec, "_resolve_name", lambda conn, name, directory: None)
    monkeypatch.setattr(
        fast_sec.ticker_directory,
        "TickerDirectory",
        type("_Dir", (), {"load": classmethod(lambda cls: None)}),
    )

    report = fast_sec.ingest(seeded, "AAOI", write=True, allow_model=False)

    assert report.wrote is True
    assert seeded.run(f"MATCH (s:{STOCK} {{ticker: 'AAOI'}}) RETURN s.ticker")
    vendor_edges = seeded.run(
        f"MATCH (v:{SUPPLIER})-[:SUPPLIES_TO]->(s:{STOCK} {{ticker: 'AAOI'}}) RETURN count(*) AS n"
    )[0]["n"]
    assert vendor_edges >= 2
    owns = seeded.run(
        f"MATCH (s:{STOCK} {{ticker: 'AAOI'}})-[:OWNS]->(sub:{SUBSIDIARY}) RETURN count(*) AS n"
    )[0]["n"]
    assert owns >= 1


def test_supplier_evidence_quotes_the_disclosure():
    text = fast_sec.de_tag(
        "<p>Some components are available only from a sole source or have been "
        "qualified only from a single supplier.</p>"
    )

    evidence = fast_sec.supplier_evidence(text)

    assert evidence is not None
    assert "sole source" in evidence


def test_ingest_reports_evidence_when_the_filing_names_no_suppliers(seeded, monkeypatch):
    """AAOI's real case: the 10-K discloses a dependency, never a name."""
    html_text = """
    <html><body>
    <p>Item 1. Business</p>
    <p>We manufacture the majority of our laser chips in-house. We depend on a limited
    number of suppliers for certain raw materials and components. We purchase some
    components from external suppliers.</p>
    <p>Item 1A. Risk Factors</p>
    <p>Some components are available only from a sole source.</p>
    </body></html>
    """
    _patch_fetch(monkeypatch, html_text=html_text, with_exhibit=False)

    report = fast_sec.ingest(seeded, "AAOI", write=False, allow_model=False)

    assert report.suppliers == []
    assert report.evidence is not None
    assert "limited number of suppliers" in report.evidence
    assert any("names no suppliers" in note for note in report.notes)


def test_ingest_uses_the_cached_document_on_a_second_run(seeded, monkeypatch):
    _patch_fetch(monkeypatch)
    monkeypatch.setattr(
        fast_sec, "_load_cached_text", lambda filing: fast_sec.de_tag(FILING_HTML)
    )

    def _no_download(url, max_bytes=0):
        raise AssertionError("a cached document must not be downloaded again")

    monkeypatch.setattr(fast_sec, "get_text", _no_download)

    report = fast_sec.ingest(seeded, "AAOI", write=False, allow_model=False)

    assert report.cached is True
    assert report.suppliers  # still extracted from the cached text


def test_ingest_refuses_up_front_without_a_contact_address(seeded, monkeypatch):
    """EDGAR 403s without a contact; do not spend a request finding that out."""
    from src import config

    monkeypatch.setattr(config, "SEC_CONTACT_EMAIL", "")
    monkeypatch.setattr(
        fast_sec,
        "resolve_cik",
        lambda ticker: (_ for _ in ()).throw(AssertionError("must not touch SEC")),
    )

    report = fast_sec.ingest(seeded, "AAOI", write=False)

    assert report.error is not None
    assert "SEC_CONTACT_EMAIL" in report.error
    assert report.suppliers == []


def test_ingest_reports_a_403_instead_of_raising(seeded, monkeypatch):
    """The deployed app crashed on this: a 403 from EDGAR must be a result."""
    import requests

    monkeypatch.setattr(
        fast_sec, "resolve_cik", lambda ticker: ("0001158114", "APPLIED OPTOELECTRONICS INC")
    )
    monkeypatch.setattr(
        fast_sec, "latest_filing", lambda cik, forms=("10-K", "20-F"): _fake_filing()
    )
    monkeypatch.setattr(fast_sec, "_load_cached_text", lambda filing: "")

    def forbidden(url, max_bytes=0):
        raise requests.HTTPError(
            "403 Client Error", response=type("R", (), {"status_code": 403})()
        )

    monkeypatch.setattr(fast_sec, "get_text", forbidden)

    report = fast_sec.ingest(seeded, "AAOI", write=False)

    assert report.error is not None
    assert "SEC_CONTACT_EMAIL" in report.error
    assert report.form == "10-K"  # the filing was found; only the download failed
    assert report.suppliers == []
    assert report.wrote is False


def test_headers_carry_the_configured_contact(monkeypatch):
    from src import config

    monkeypatch.setattr(config, "SEC_CONTACT_EMAIL", "someone@example.com")
    headers = fast_sec._headers()

    assert "someone@example.com" in headers["User-Agent"]


def test_headers_flag_a_missing_contact(monkeypatch):
    from src import config

    monkeypatch.setattr(config, "SEC_CONTACT_EMAIL", "")
    headers = fast_sec._headers()

    assert "contact-not-configured" in headers["User-Agent"]


def test_ingest_unknown_ticker_is_reported_not_invented(seeded, monkeypatch):
    monkeypatch.setattr(fast_sec, "resolve_cik", lambda ticker: None)

    report = fast_sec.ingest(seeded, "NOPE", write=False)

    assert report.suppliers == []
    assert any("registrant map" in note for note in report.notes)


def test_ingest_without_annual_report_is_reported(seeded, monkeypatch):
    monkeypatch.setattr(fast_sec, "resolve_cik", lambda ticker: ("0001158114", "X"))
    monkeypatch.setattr(fast_sec, "latest_filing", lambda cik, forms=("10-K", "20-F"): None)

    report = fast_sec.ingest(seeded, "AAOI", write=False)

    assert any("10-K or 20-F" in note for note in report.notes)


def test_20f_skips_exhibit21(seeded, monkeypatch):
    _patch_fetch(monkeypatch, with_exhibit=False)
    monkeypatch.setattr(
        fast_sec,
        "latest_filing",
        lambda cik, forms=("10-K", "20-F"): fast_sec.Filing(
            form="20-F",
            accession="0001-05",
            primary_document="x.htm",
            filed="2026-03-01",
            cik="0001158114",
        ),
    )
    called: list[str] = []
    monkeypatch.setattr(fast_sec, "exhibit21_names", lambda filing: called.append("x") or [])

    report = fast_sec.ingest(seeded, "AAOI", write=False, allow_model=False)

    assert called == []  # 20-F has no Exhibit 21
    assert any("no Exhibit 21" in note for note in report.notes)
