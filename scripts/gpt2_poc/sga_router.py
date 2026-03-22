"""
SGA Router — COMPATIBILITY SHIM.

Canonical source is now transcender/router.py.
This file re-exports for backward compatibility with scripts that
import directly from sga_router.
"""

from transcender.router import SonRouter, SonRoutingLoss

__all__ = ["SonRouter", "SonRoutingLoss"]
