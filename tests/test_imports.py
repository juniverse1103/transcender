"""Smoke tests: verify package imports and basic class instantiation."""

import pytest


def test_import_transcender_package():
    import transcender
    assert hasattr(transcender, "SonRouter")
    assert hasattr(transcender, "SonRoutingLoss")
    assert hasattr(transcender, "TranscenderModel")
    assert hasattr(transcender, "ArchitectureAdapter")


def test_import_router_module():
    from transcender.router import SonRouter, SonRoutingLoss
    assert SonRouter is not None
    assert SonRoutingLoss is not None


def test_import_model_module():
    from transcender.model import TranscenderModel, ArchitectureAdapter
    assert TranscenderModel is not None
    assert ArchitectureAdapter is not None


def test_import_engine_config():
    from transcender.engine.config import GptOssConfig
    config = GptOssConfig()
    assert config.num_hidden_layers == 24
    assert config.hidden_size == 2880
    assert config.vocab_size == 201088
    assert config.gqa_ratio == 8
    assert config.total_params_b == 21.0
    assert config.active_params_b == 3.6


def test_import_engine_loading():
    from transcender.engine.loading import resolve_gpt_oss_model_path
    # Non-existent path should return as-is
    result = resolve_gpt_oss_model_path("/nonexistent/path")
    assert result == "/nonexistent/path"


def test_import_engine_prompts():
    from transcender.engine.prompts import build_harmony_messages
    messages = build_harmony_messages("Hello")
    assert len(messages) == 2
    assert messages[0]["role"] == "system"
    assert messages[1]["role"] == "user"
    assert messages[1]["content"] == "Hello"


def test_backward_compat_sga_router():
    """Verify the sga_router.py shim re-exports correctly."""
    import sga_router
    assert hasattr(sga_router, "SonRouter")
    assert hasattr(sga_router, "SonRoutingLoss")


def test_backward_compat_transcender_injector():
    """Verify the transcender_injector.py shim re-exports correctly."""
    import transcender_injector
    assert hasattr(transcender_injector, "TranscenderModel")
    assert hasattr(transcender_injector, "ArchitectureAdapter")
