"""External evidence sources.

One module per provider, all implementing ``base.SourceProvider``, plus the
two enrichment paths that also reach the network: the citation-graph expander
and the full-text content extractor.
"""
