# Offline evaluation harness

Measures the agent pipeline so the project can quote a number instead of "it
works on my laptop" - and is explicit about which number means what.

```
.\.venv\Scripts\python.exe -m evals.run          # offline: no API key, no network
.\.venv\Scripts\python.exe -m evals.run --live   # opt-in: real DeepSeek calls
```

Run from the repository root, on Windows PowerShell, with the project venv.
A cold offline run takes roughly 5-10 seconds on this machine - most of it
importing the pipeline's dependencies; the checks themselves take well under a
second. It stays comfortably inside the 20-second offline budget.

## Exit codes

| Code | Meaning |
| ---- | ------- |
| 0 | every gated check passed |
| 1 | at least one check failed (a real failure, not a known gap) |
| 2 | the harness could not start (missing key for `--live`, missing pytest, unsafe store path) |

## The dataset

`evals/questions.py` holds **27 questions** in 9 categories:

| category | n | examples |
| -------- | - | -------- |
| competitors | 2 | "Which companies compete with Nvidia?" |
| suppliers | 3 | "Who supplies Nvidia?", "Which companies does Nvidia supply to?" |
| subsidiaries | 2 | "What subsidiaries does Nvidia own?" |
| board | 3 | "Who is on Nvidia's board of directors?", "Who is the CEO of Nvidia?" |
| sector | 3 | "What sector is Nvidia in?", "Which stocks are in the Technology sector?" |
| news | 3 | "What recent news mentions Nvidia?" plus one semantic-search question |
| live_price | 3 | "What is NVDA trading at right now?" |
| historical | 5 | "How has Tesla's stock performed since 2020?" (must be DEFERRED) |
| general | 3 | "What is a P/E ratio?" (unrelated to the graph) |

Each entry records the product expectation (`expected_intent`,
`expected_schema_keys` - a *subset* requirement), a reviewed read-only
`reference_cypher` plus its expected row shape where a graph answer is expected,
and an `offline_gap` marker when the deterministic fallback cannot possibly
reach the product intent.

## Default (offline) mode

No API key, no network, no tokens. Before any pipeline code runs, the harness
stubs `src.llm.chat`, `src.llm.chat_json`, `src.llm.get_client` and
`src.embeddings.embed_text(s)` in-process. The stubs raise, so an accidental
model or embedding call is reported instead of reaching the internet. The
harness counts the calls and fails if a network client or embedding model was
requested.

What it checks:

1. **Router fallback heuristics** (`src/agent/agent_router.py`): with the LLM
   unavailable, historical questions must be `historical_trend`, live-quote
   questions `live_price_fetch`, relationship questions `graph_query`, and every
   returned schema key must be a real `SCHEMA_DICTIONARY` key. Historical
   questions must return no schema keys and must be skipped by Step 3
   (`generate_cypher_query(..., INTENT_HISTORICAL)["action"] == "skip"`), so the
   graph is never asked for price history.
2. **Read-only guard** (`src/agent/cypher_executor.py`): a corpus of writes is
   blocked; reads that merely contain write keywords as data are allowed; and a
   write is pushed through `execute_cypher_with_retry` to prove the guard runs
   *before* the engine, not just as a standalone helper.
3. **Schema validation**: all expected/candidate keys exist, every reference
   query uses only declared labels/relationship types, and those entities are
   described by the question's expected schema keys. This is what catches
   dataset drift when `src/schema.py` changes.
4. **Data contract**: each reference query is executed through Step 4 against a
   real, temporary embedded Relate store, then checked for columns, row count,
   expected values and forbidden values. Direction is covered (supplier ->
   customer, owns -> subsidiary), as are the supplier `coalesce` (a `Stock`
   supplier and a `Supplier` vendor in the same result) and the board filter
   (`role = 'director'`, with the CEO explicitly forbidden from the list).
5. **Safety**: the store node/relationship counts must be identical before and
   after the whole run, proving no query mutated it.

### The eval store

The harness **never opens `data/stock_graph.rl`**. It redirects `config.DB_PATH`
to a throwaway file under `evals/.eval-tmp/`, imports
`tests/conftest.seed_demo_graph` **read-only** (loaded by path, because the
repo's `tests` directory collides with an unrelated site-packages `tests`
package), adds a small fixture of graph shapes the demo seed omits (subsidiaries,
directors, unlisted `Supplier` vendors), runs the checks, then closes and deletes
the store. The redirect is restored afterwards.

### What the offline number means

```
offline gated pass rate   : 24/24 = 100.0%
product-intent pass rate  : 24/27 = 88.9%
```

- `24/24` is the honest offline gate: 24 questions whose intent and data
  contract the deterministic path can actually verify.
- `24/27` is the number against the full product expectation. The three
  unrelated questions are routed to `graph_query` by the deterministic fallback,
  which has no `general_knowledge` branch. They are printed as **known gaps**
  and are **not counted as passes**. Only `--live` can classify an off-graph
  question, because that requires a model. If the fallback later learns to
  classify them, they turn into normal passes automatically.

## `--live` mode

Opt in with `--live`. It makes **9 real DeepSeek routing calls** (one per
selected question) and checks `expected_intent` and `expected_schema_keys` only.
It does not open the graph, does not call yfinance and does not generate Cypher.
A missing API key exits 2 with a clear message rather than silently falling back
to heuristics; if the router *does* fall back during a live run, that question
is marked FAIL, because a fallback coincidence is not a model result.

## What this harness does NOT prove

- **It is not a text-to-Cypher benchmark.** In offline mode the generator and
  summariser never run. The reference queries are human-reviewed, so a passing
  graph question proves the schema, the guard and the data contract work - not
  that the agent can write that query. `--live` measures routing only; it still
  does not score generated Cypher.
- **It does not prove production data quality.** The eval graph is a synthetic
  fixture (demo seed plus a handful of nodes), not the ingested graph. It does
  not measure coverage, freshness or the correctness of real SEC/news data.
- **It does not test live quotes.** The live-price branch calls yfinance; offline
  only the routing decision and the "skip the graph" contract are checked. No
  quote is fetched.
- **It does not measure answer quality, the UI, latency or cost.**
- **Passing does not mean the product is complete.** It means the gated offline
  contract holds today. Known gaps are reported separately and are not hidden in
  the pass count.

## Adding questions

Edit `evals/questions.py` and add a `Question`. For a graph question, include a
reviewed read-only `reference_cypher` and its `expected_columns` /
`expected_row_count` / `expected_cells`; the schema-grounding check will reject a
query that uses an undeclared label or a relationship its expected schema keys do
not describe. Use `live=True` sparingly - each live question costs one API call.
