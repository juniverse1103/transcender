"""Smoke tests for TranscenderModel forward pass shapes and loss computation."""

import pytest
import torch


@pytest.fixture(scope="module")
def model():
    """Load TranscenderModel once for all tests in this module."""
    from transcender import TranscenderModel
    m = TranscenderModel(
        model_name="gpt2",
        exit_after_layer=2,
        exit_threshold=0.5,
        inference_mode="hard",
    )
    m.eval()
    return m


def test_forward_output_keys(model):
    ids = torch.randint(0, 50257, (1, 16))
    with torch.no_grad():
        out = model(input_ids=ids)

    assert "logits" in out
    assert "routing_info" in out
    assert "layer_counts" in out


def test_logits_shape(model):
    batch, seq = 2, 8
    ids = torch.randint(0, 50257, (batch, seq))
    with torch.no_grad():
        out = model(input_ids=ids)

    assert out["logits"].shape == (batch, seq, 50257)


def test_routing_info_shapes(model):
    batch, seq = 1, 12
    ids = torch.randint(0, 50257, (batch, seq))
    with torch.no_grad():
        out = model(input_ids=ids)

    ri = out["routing_info"]
    assert ri["son_scores"].shape == (batch, seq)
    assert ri["exit_probs"].shape == (batch, seq)
    assert ri["exit_mask"].shape == (batch, seq)
    assert ri["exit_mask"].dtype == torch.bool


def test_layer_counts_range(model):
    ids = torch.randint(0, 50257, (1, 16))
    with torch.no_grad():
        out = model(input_ids=ids)

    lc = out["layer_counts"]
    assert lc.min().item() >= model.exit_after_layer
    assert lc.max().item() <= 12


def test_loss_computation(model):
    ids = torch.randint(0, 50257, (1, 16))
    model.train()
    out = model(input_ids=ids, labels=ids)
    model.eval()

    assert "loss" in out
    assert "lm_loss" in out
    assert "routing_loss" in out
    assert out["loss"].shape == ()
    assert out["loss"].item() > 0
    assert torch.isfinite(out["loss"])


def test_soft_inference_mode(model):
    model.set_inference_mode("soft")
    ids = torch.randint(0, 50257, (1, 8))
    with torch.no_grad():
        out = model(input_ids=ids)
    assert out["logits"].shape == (1, 8, 50257)
    model.set_inference_mode("hard")


def test_adaptive_inference_mode(model):
    model.set_inference_mode("adaptive")
    ids = torch.randint(0, 50257, (1, 8))
    with torch.no_grad():
        out = model(input_ids=ids)
    assert out["logits"].shape == (1, 8, 50257)
    model.set_inference_mode("hard")


def test_exit_probs_in_unit_interval(model):
    ids = torch.randint(0, 50257, (1, 32))
    with torch.no_grad():
        out = model(input_ids=ids)
    probs = out["routing_info"]["exit_probs"]
    assert (probs >= 0).all()
    assert (probs <= 1).all()
