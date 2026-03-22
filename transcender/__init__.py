"""
Transcender — Per-token depth-axis routing for transformer language models.

Two routing primitives:
  1. Learned Son Router — trained per model, per exit layer (GPT-2 PoC path)
  2. Entropy Gate — zero-shot, sequence-level (GPT-OSS 20B runtime path)

Core exports:
  - SonRouter, SonRoutingLoss       (from transcender.router)
  - TranscenderModel                (from transcender.model)
  - ArchitectureAdapter             (from transcender.model)
  - GptOssConfig                    (from transcender.engine.config)
  - MLXDynamicExpertEngine          (from transcender.engine.runtime)
"""

from transcender.router import SonRouter, SonRoutingLoss
from transcender.model import TranscenderModel, ArchitectureAdapter

__all__ = [
    "SonRouter",
    "SonRoutingLoss",
    "TranscenderModel",
    "ArchitectureAdapter",
]
