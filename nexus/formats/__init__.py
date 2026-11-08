"""Read-only format identification and structural parsers.

These parsers provide deterministic facts for the investigation broker.  They
do not execute the analyzed artifact, resolve external symbols, or claim
semantic understanding.
"""

from nexus.formats.identify import identify_format

__all__ = ["identify_format"]
