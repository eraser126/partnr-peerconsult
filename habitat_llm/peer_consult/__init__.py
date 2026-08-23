"""PeerConsult coordination primitives for PARTNR.

The package is deliberately independent of Habitat simulator internals.  It
only consumes the per-agent world graphs that the selected PARTNR baseline
already exposes to its two planners.
"""

from habitat_llm.peer_consult.board import PeerConsultBoard

__all__ = ["PeerConsultBoard"]
