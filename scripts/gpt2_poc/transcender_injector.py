"""
Transcender Injector — COMPATIBILITY SHIM.

Canonical source is now transcender/model.py.
This file re-exports for backward compatibility.
"""

from transcender.model import TranscenderModel, ArchitectureAdapter
from transcender.router import SonRouter, SonRoutingLoss

__all__ = ["TranscenderModel", "ArchitectureAdapter", "SonRouter", "SonRoutingLoss"]
