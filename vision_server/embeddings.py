"""Generate text embeddings for semantic search."""

_model = None


def _load_model():
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer
        _model = SentenceTransformer("all-MiniLM-L6-v2")
    return _model


def get_embedding(text):
    """Convert text to a 384-dim vector for semantic search."""
    model = _load_model()
    return model.encode(text).tolist()
