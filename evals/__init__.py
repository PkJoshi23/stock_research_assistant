"""Offline evaluation harness for the AI Stock Research Assistant.

This package is measurement only. It never imports the Streamlit UI, never opens
the production store (``data/stock_graph.rl``) and never changes the pipeline it
measures. Kept as a package so ``python -m evals.run`` works from the repo root
without any shell-level ``PYTHONPATH`` fiddling.
"""
