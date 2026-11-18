"""AI Stock Research Assistant - pure-graph implementation.

Layout:
    src/ingestion/  data acquisition (companies, relationships, news)
    src/agent/      the 4-step schema-grounded text-to-Cypher pipeline
    src/ui/         Streamlit frontend
"""

#: Bumped on every deploy that must be observable in the running app. The sidebar
#: shows it, so "is the deployed build older than main?" is answered by looking at
#: the page instead of comparing commit hashes by hand.
__version__ = "0.3.3"
