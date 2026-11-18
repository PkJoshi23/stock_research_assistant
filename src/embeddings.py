"""Local sentence embeddings, loaded once per process.

Two constraints drive this module:

  * ``src.config`` must be imported before ``sentence_transformers`` so that
    ``HF_HOME`` and ``HF_HUB_DISABLE_XET`` are already in the environment -
    importing the config at the top of this file guarantees that ordering for
    every caller.
  * the model is expensive to load (~90 MB, a few seconds on CPU) and is used
    by both ingestion and the query path, so it is cached at module level
    rather than rebuilt per call.
"""

from __future__ import annotations

from . import config  # noqa: F401  (import first: sets HF_HOME/HF_HUB_DISABLE_XET)

_model = None


def get_model():
    """Load (once) and return the sentence-transformers model."""
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer

        _model = SentenceTransformer(config.EMBEDDING_MODEL_NAME)
    return _model


def embed_texts(texts: list[str], *, batch_size: int = 32) -> list[list[float]]:
    """Embed a batch and return plain Python lists.

    Plain lists matter: ``relate.Vector`` has no Python constructor, the engine
    accepts a ``list[float]`` directly and hands a list back on read. Numpy
    arrays are rejected by the parameter encoder.
    """
    if not texts:
        return []
    vectors = get_model().encode(
        texts, batch_size=batch_size, normalize_embeddings=True, show_progress_bar=False
    )
    return [[float(value) for value in vector] for vector in vectors]


def embed_text(text: str) -> list[float]:
    """Embed a single string (the user's question, in the query path)."""
    return embed_texts([text])[0]
