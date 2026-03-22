"""Smoke tests for SonRouter and SonRoutingLoss."""

import pytest
import torch


def test_son_router_shapes():
    from transcender.router import SonRouter

    router = SonRouter(hidden_size=768, exit_threshold=0.5)
    batch, seq, hidden = 2, 16, 768
    num_heads = 12

    hidden_states = torch.randn(batch, seq, hidden)
    attn_weights = torch.randn(batch, num_heads, seq, seq).softmax(dim=-1)

    out = router(hidden_states, attn_weights)

    assert out["son_scores"].shape == (batch, seq)
    assert out["exit_probs"].shape == (batch, seq)
    assert out["exit_mask"].shape == (batch, seq)
    assert out["exit_mask"].dtype == torch.bool


def test_son_router_exit_threshold():
    from transcender.router import SonRouter

    router = SonRouter(hidden_size=64, exit_threshold=0.7)
    hidden_states = torch.randn(1, 4, 64)
    attn_weights = torch.randn(1, 4, 4, 4).softmax(dim=-1)

    out = router(hidden_states, attn_weights)
    # exit_mask should reflect threshold=0.7
    expected = out["exit_probs"] > 0.7
    assert (out["exit_mask"] == expected).all()


def test_routing_loss_kl_calibrated():
    from transcender.router import SonRoutingLoss

    loss_fn = SonRoutingLoss()
    batch, seq, vocab = 2, 8, 100

    exit_probs = torch.rand(batch, seq)
    early_logits = torch.randn(batch, seq, vocab)
    deep_logits = torch.randn(batch, seq, vocab)

    loss = loss_fn(exit_probs, early_logits=early_logits, deep_logits=deep_logits)

    assert loss.shape == ()
    assert loss.item() > 0
    assert torch.isfinite(loss)


def test_routing_loss_fallback():
    from transcender.router import SonRoutingLoss

    loss_fn = SonRoutingLoss()
    exit_probs = torch.rand(2, 8)

    # Without logits, falls back to uniform efficiency pressure
    loss = loss_fn(exit_probs)
    assert loss.shape == ()
    assert torch.isfinite(loss)
