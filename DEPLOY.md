# Deploying (Streamlit Community Cloud)

The app is ready to deploy; the only steps that need your account are creating
the GitHub repository and connecting it in Streamlit Cloud.

## What is already prepared

- **`streamlit_app.py`** — root entry point Cloud expects (it runs `src/ui/app.py`).
- **`data/stock_graph.rl` is committed** (1.8 MB) — a fresh deploy starts with a
  populated graph instead of an empty one. It is a snapshot of the
  relationships and market values; prices are fetched live at query time.
- **`src/config.py` reads `st.secrets`** when an environment variable is absent,
  so Cloud secrets work without a `.env` file.
- **`requirements.txt`** is small now: no torch, no embedding model.

## 1. Create the GitHub repository

This folder currently lives inside your home git repo, so it needs its own repo:

```bash
# on GitHub: create an empty repo (no README, no .gitignore)
cd "C:\Users\parikshit joshi\Desktop\graphbase\stock-research-agent"
git init
git add .
git commit -m "AI Stock Research Assistant — Relate hackathon"
git branch -M main
git remote add origin https://github.com/<you>/<repo>.git
git push -u origin main
```

If `git commit` complains about identity, set it once:

```bash
git config user.name "Your Name"
git config user.email "you@example.com"
```

## 2. Create the app

1. <https://share.streamlit.io> → **New app** → *Deploy from a public repo*.
2. Repository: the one you just pushed. Branch: `main`.
3. **Main file path: `streamlit_app.py`** (not `src/ui/app.py` — both work, but
   the root file is what Cloud defaults to).
4. Python version: **3.12** (3.11 is also fine).
5. **Deploy.** The build is quick: the graph is committed and there is no model
   download.

## 3. Add secrets

In the app's **Settings → Secrets**, paste:

```toml
DEEPSEEK_API_KEY = "sk-..."
SEC_CONTACT_EMAIL = "you@example.com"   # optional at runtime
```

Saving restarts the app. `DEEPSEEK_API_KEY` is the only required secret.

## 4. Verify

- The sidebar shows `Nodes 151` (or similar) — the committed graph loaded.
- Ask *"Who supplies Nebius?"* → should answer with NVIDIA.
- Ask *"How has Tesla done since 2020?"* → should decline (no price history).
- The **Sources & freshness** panel should list the graph rows and headline URLs.

## Known deployment risks

| Risk | What to do |
| --- | --- |
| **Ephemeral disk** — memory nodes (`Query`), index writes and `ensure_schema` live in the container and reset on restart/redeploy. | Expected; it is a demo. Re-ingesting on Cloud is not supported (single-writer lease + long crawls). |
| **The graph is frozen at commit time** | Re-run ingestion locally and commit the updated `.rl` to refresh relationships or snapshots. Prices are unaffected (fetched live). |
| **Secrets never go in the repo** | `.streamlit/secrets.toml` and `.env` are gitignored. Use the Cloud Secrets editor only. |
| **Cost** | ~1,600 tokens and ~$0.0005 per question at current DeepSeek pricing. |
