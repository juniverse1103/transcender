"""
SGA Router — Son-Gated Architecture Router Module

Computes the 'Son' metric (R = I × P) per token and decides whether
a token should exit early (bypass remaining layers) or continue
through deeper processing.

    I (Information Amount) = L2 norm of the hidden state (embedding richness)
    P (Probability)        = mean attention weight received by the token
    Son = I × P            = composite "realness" / compute-priority score
"""

import torch
import torch.nn as nn


class SonRouter(nn.Module):
    """
    A lightweight gating module injected between Transformer layers.

    Given hidden states and attention weights, it computes a per-token
    Son score and produces an early-exit probability.

    Args:
        hidden_size: Dimensionality of the model's hidden states (e.g. 768 for GPT-2).
        exit_threshold: Son score below which a token is routed to early exit.
            During training this is soft (sigmoid); during inference it's a hard gate.
    """

    def __init__(self, hidden_size: int, exit_threshold: float = 0.5):
        super().__init__()
        self.exit_threshold = exit_threshold
        self.hidden_size = hidden_size

        # The gate uses the full hidden state (768-dim for GPT-2) to decide
        # exit/continue. This is much richer than [I, P] alone — the hidden
        # state encodes semantic complexity, syntactic role, and positional context.
        # We keep I×P as an interpretable "Son score" for logging/visualization.
        self.gate_proj = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.GELU(),
            nn.Linear(64, 1),   # output: scalar exit logit
        )

    def compute_information(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        I = L2 norm of each token's hidden state, normalized by sqrt(d).
        Shape: (batch, seq_len)
        """
        # Normalize so the magnitude is comparable across model sizes
        return torch.norm(hidden_states, dim=-1) / (hidden_states.size(-1) ** 0.5)

    def compute_probability(self, attention_weights: torch.Tensor) -> torch.Tensor:
        """
        P = mean attention each token *receives* across all heads.
        attention_weights shape: (batch, num_heads, seq_len, seq_len)
        Returns shape: (batch, seq_len)
        """
        # Sum over the "query" dimension (dim=-2) → how much attention flows TO each key token
        # Then average across heads
        return attention_weights.mean(dim=1).sum(dim=-2) / attention_weights.size(-1)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_weights: torch.Tensor,
    ) -> dict:
        """
        Compute the Son metric and produce exit decisions.

        Returns:
            dict with keys:
                - son_scores:      (batch, seq_len) — raw I × P values
                - exit_probs:      (batch, seq_len) — learned exit probability [0, 1]
                - exit_mask:       (batch, seq_len) — boolean mask (True = exit early)
        """
        I = self.compute_information(hidden_states)       # (B, S)
        P = self.compute_probability(attention_weights)   # (B, S)

        son_scores = I * P  # The core Son metric (kept for interpretability)

        # Use the full hidden state for the gate decision
        exit_logits = self.gate_proj(hidden_states).squeeze(-1)  # (B, S)
        exit_probs = torch.sigmoid(exit_logits)           # (B, S) in [0, 1]

        # Hard exit decision (used at inference; gradient flows through exit_probs during training)
        exit_mask = exit_probs > self.exit_threshold

        return {
            "son_scores": son_scores,
            "exit_probs": exit_probs,
            "exit_mask": exit_mask,
        }


class SonRoutingLoss(nn.Module):
    """
    KL-calibrated routing loss that teaches the router WHICH tokens can
    safely exit early, rather than pushing all tokens uniformly.

    Strategy:
      1. Compute KL divergence between early and deep logits (no gradient).
      2. Tokens where KL is low → the early logits are already good → target exit.
      3. Tokens where KL is high → early exit would hurt → target continue.
      4. Train the router via binary cross-entropy against these soft targets.

    This replaces the uniform efficiency loss that couldn't compete with
    the content-aware LM loss gradient.
    """

    def __init__(self, lambda_efficiency: float = 1.0, lambda_quality: float = 1.0):
        super().__init__()
        self.lambda_efficiency = lambda_efficiency
        self.lambda_quality = lambda_quality

    def forward(
        self,
        exit_probs: torch.Tensor,
        early_logits: torch.Tensor = None,
        deep_logits: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        KL-calibrated routing loss.

        Args:
            exit_probs:   (B, S) — router's exit probability
            early_logits: (B, S, V) — logits from early pathway (optional)
            deep_logits:  (B, S, V) — logits from deep pathway (optional)

        If early/deep logits are not provided, falls back to uniform
        efficiency pressure (backward compatible).
        """
        if early_logits is not None and deep_logits is not None:
            return self._kl_calibrated_loss(exit_probs, early_logits, deep_logits)

        # Fallback: uniform efficiency pressure
        efficiency_loss = (1 - exit_probs).mean()
        return self.lambda_efficiency * efficiency_loss

    def _kl_calibrated_loss(self, exit_probs, early_logits, deep_logits):
        """KL-calibrated loss: teach router which tokens are safe to exit."""
        with torch.no_grad():
            # KL(deep || early) per token — measures quality loss from early exit
            deep_log_probs = torch.nn.functional.log_softmax(deep_logits, dim=-1)
            early_probs = torch.nn.functional.softmax(early_logits, dim=-1)
            # KL = sum(p * log(p/q)) = sum(p * (log_p - log_q))
            kl_per_token = torch.nn.functional.kl_div(
                torch.nn.functional.log_softmax(early_logits, dim=-1),
                torch.nn.functional.softmax(deep_logits, dim=-1),
                reduction="none",
            ).sum(dim=-1)  # (B, S)

            # Convert KL to exit targets: low KL → safe to exit (target=1)
            # Use a sigmoid to map KL to [0, 1] with a learnable-like threshold
            # KL=0 → target=1.0 (perfect match, definitely exit)
            # KL=median → target=0.5 (uncertain)
            # KL=high → target=0.0 (big quality gap, must continue)
            kl_median = kl_per_token.median()
            exit_targets = torch.sigmoid(-2.0 * (kl_per_token - kl_median))  # (B, S)

        # Binary cross-entropy: train router to predict safe exit tokens
        calibration_loss = torch.nn.functional.binary_cross_entropy(
            exit_probs, exit_targets,
        )

        # Add mild uniform efficiency pressure to break ties toward exit
        efficiency_loss = (1 - exit_probs).mean()

        return (self.lambda_quality * calibration_loss
                + 0.1 * self.lambda_efficiency * efficiency_loss)
