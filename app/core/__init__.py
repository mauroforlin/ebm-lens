"""Cross-cutting infrastructure: LLM access, embeddings, caching, accounting.

Nothing here knows what the application searches for. These are the shared
services every pipeline stage builds on.
"""
