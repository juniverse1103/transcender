"""
Acceptance policies for dense selective-depth inference.

Each policy takes shallow logits (and optionally deep logits) and returns
an accept/reject decision plus diagnostic metrics.
"""

import mlx.core as mx
from dataclasses import dataclass
from typing import Optional, Dict, Any


@dataclass
class PolicyDecision:
    accept: bool
    shallow_top1: int
    shallow_top1_prob: float
    entropy: float
    margin: float
    # Optional fields populated by policies that need deep logits
    deep_top1: Optional[int] = None
    agree: Optional[bool] = None
    kl_divergence: Optional[float] = None
    topk_overlap: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        d = {
            "accept": self.accept,
            "shallow_top1": self.shallow_top1,
            "shallow_top1_prob": self.shallow_top1_prob,
            "entropy": self.entropy,
            "margin": self.margin,
        }
        if self.deep_top1 is not None:
            d["deep_top1"] = self.deep_top1
        if self.agree is not None:
            d["agree"] = self.agree
        if self.kl_divergence is not None:
            d["kl_divergence"] = self.kl_divergence
        if self.topk_overlap is not None:
            d["topk_overlap"] = self.topk_overlap
        return d


def _compute_base_metrics(logits):
    """Compute entropy, margin, top1 from logits. Returns dict."""
    probs = mx.softmax(logits, axis=-1)
    log_probs = mx.log(probs + 1e-10)
    entropy = -mx.sum(probs * log_probs, axis=-1).item()
    vocab_size = logits.shape[-1]
    normalized_entropy = entropy / (mx.log(mx.array(vocab_size)).item())

    top2_indices = mx.argpartition(-probs, kth=2, axis=-1)[..., :2]
    top2_probs = mx.take_along_axis(probs, top2_indices, axis=-1)
    sorted_idx = mx.argsort(-top2_probs, axis=-1)
    top2_probs = mx.take_along_axis(top2_probs, sorted_idx, axis=-1)
    top2_indices = mx.take_along_axis(top2_indices, sorted_idx, axis=-1)

    top1_id = top2_indices[..., 0].item()
    top1_prob = top2_probs[..., 0].item()
    top2_prob = top2_probs[..., 1].item()
    margin = top1_prob - top2_prob

    return {
        "top1_id": top1_id,
        "top1_prob": top1_prob,
        "entropy": normalized_entropy,
        "margin": margin,
        "probs": probs,
    }


class EntropyPolicy:
    """Accept if normalized entropy <= threshold."""

    def __init__(self, threshold: float = 0.15):
        self.threshold = threshold

    def __call__(self, shallow_logits, deep_logits=None) -> PolicyDecision:
        m = _compute_base_metrics(shallow_logits)
        return PolicyDecision(
            accept=m["entropy"] <= self.threshold,
            shallow_top1=m["top1_id"],
            shallow_top1_prob=m["top1_prob"],
            entropy=m["entropy"],
            margin=m["margin"],
        )


class MarginPolicy:
    """Accept if top1_prob - top2_prob >= threshold."""

    def __init__(self, threshold: float = 0.5):
        self.threshold = threshold

    def __call__(self, shallow_logits, deep_logits=None) -> PolicyDecision:
        m = _compute_base_metrics(shallow_logits)
        return PolicyDecision(
            accept=m["margin"] >= self.threshold,
            shallow_top1=m["top1_id"],
            shallow_top1_prob=m["top1_prob"],
            entropy=m["entropy"],
            margin=m["margin"],
        )


class Top1AgreePolicy:
    """Accept if shallow top1 == deep top1. Requires deep logits."""

    def __call__(self, shallow_logits, deep_logits=None) -> PolicyDecision:
        sm = _compute_base_metrics(shallow_logits)
        if deep_logits is None:
            raise ValueError("Top1AgreePolicy requires deep_logits")
        deep_top1 = mx.argmax(deep_logits, axis=-1).item()
        agree = sm["top1_id"] == deep_top1
        return PolicyDecision(
            accept=agree,
            shallow_top1=sm["top1_id"],
            shallow_top1_prob=sm["top1_prob"],
            entropy=sm["entropy"],
            margin=sm["margin"],
            deep_top1=deep_top1,
            agree=agree,
        )


class HybridTop1EntropyPolicy:
    """Accept if shallow top1 == deep top1 AND entropy <= threshold."""

    def __init__(self, entropy_threshold: float = 0.15):
        self.entropy_threshold = entropy_threshold

    def __call__(self, shallow_logits, deep_logits=None) -> PolicyDecision:
        sm = _compute_base_metrics(shallow_logits)
        if deep_logits is None:
            raise ValueError("HybridTop1EntropyPolicy requires deep_logits")
        deep_top1 = mx.argmax(deep_logits, axis=-1).item()
        agree = sm["top1_id"] == deep_top1
        accept = agree and (sm["entropy"] <= self.entropy_threshold)
        return PolicyDecision(
            accept=accept,
            shallow_top1=sm["top1_id"],
            shallow_top1_prob=sm["top1_prob"],
            entropy=sm["entropy"],
            margin=sm["margin"],
            deep_top1=deep_top1,
            agree=agree,
        )


class TopKOverlapPolicy:
    """Accept if shallow top-k overlaps with deep top-k above threshold."""

    def __init__(self, k: int = 5, overlap_threshold: float = 0.6):
        self.k = k
        self.overlap_threshold = overlap_threshold

    def __call__(self, shallow_logits, deep_logits=None) -> PolicyDecision:
        sm = _compute_base_metrics(shallow_logits)
        if deep_logits is None:
            raise ValueError("TopKOverlapPolicy requires deep_logits")
        dm = _compute_base_metrics(deep_logits)

        shallow_topk = set(mx.argpartition(-shallow_logits, kth=self.k, axis=-1)[..., :self.k].tolist())
        deep_topk = set(mx.argpartition(-deep_logits, kth=self.k, axis=-1)[..., :self.k].tolist())
        overlap = len(shallow_topk & deep_topk) / self.k

        return PolicyDecision(
            accept=overlap >= self.overlap_threshold,
            shallow_top1=sm["top1_id"],
            shallow_top1_prob=sm["top1_prob"],
            entropy=sm["entropy"],
            margin=sm["margin"],
            deep_top1=dm["top1_id"],
            agree=sm["top1_id"] == dm["top1_id"],
            topk_overlap=overlap,
        )


class KLSimilarityPolicy:
    """Accept if KL(shallow || deep) <= threshold. Requires deep logits."""

    def __init__(self, threshold: float = 0.5):
        self.threshold = threshold

    def __call__(self, shallow_logits, deep_logits=None) -> PolicyDecision:
        sm = _compute_base_metrics(shallow_logits)
        if deep_logits is None:
            raise ValueError("KLSimilarityPolicy requires deep_logits")
        dm = _compute_base_metrics(deep_logits)

        p = mx.softmax(deep_logits, axis=-1)
        q = mx.softmax(shallow_logits, axis=-1)
        kl = mx.sum(p * (mx.log(p + 1e-10) - mx.log(q + 1e-10)), axis=-1).item()

        return PolicyDecision(
            accept=kl <= self.threshold,
            shallow_top1=sm["top1_id"],
            shallow_top1_prob=sm["top1_prob"],
            entropy=sm["entropy"],
            margin=sm["margin"],
            deep_top1=dm["top1_id"],
            agree=sm["top1_id"] == dm["top1_id"],
            kl_divergence=kl,
        )


POLICIES = {
    "entropy": EntropyPolicy,
    "margin": MarginPolicy,
    "top1_agree": Top1AgreePolicy,
    "hybrid_top1_entropy": HybridTop1EntropyPolicy,
    "topk_overlap": TopKOverlapPolicy,
    "kl_similarity": KLSimilarityPolicy,
}


def make_policy(name: str, **kwargs):
    """Factory function. Usage: make_policy('entropy', threshold=0.2)"""
    if name not in POLICIES:
        raise ValueError(f"Unknown policy: {name}. Available: {list(POLICIES.keys())}")
    return POLICIES[name](**kwargs)
