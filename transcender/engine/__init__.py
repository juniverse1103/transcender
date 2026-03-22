"""
Transcender Engine — GPT-OSS 20B runtime with dynamic routing.

Split from the original monolithic transcender_engine.py into:
  - config.py:   GptOssConfig dataclass
  - loading.py:  Model path resolution and loading
  - prompts.py:  Harmony template rendering
  - runtime.py:  MLXDynamicExpertEngine (the core inference engine)
"""

from transcender.engine.config import GptOssConfig
from transcender.engine.loading import (
    resolve_gpt_oss_model_path,
    load_resolved_mlx_model,
    load_resolved_transformers_model,
)
from transcender.engine.prompts import apply_harmony_template, build_harmony_messages

__all__ = [
    "GptOssConfig",
    "resolve_gpt_oss_model_path",
    "load_resolved_mlx_model",
    "load_resolved_transformers_model",
    "apply_harmony_template",
    "build_harmony_messages",
]
