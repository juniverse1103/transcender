"""GptOssConfig — Architecture constants for openai/gpt-oss-20b."""

from dataclasses import dataclass, field


@dataclass
class GptOssConfig:
    """Exact architecture of openai/gpt-oss-20b."""
    model_type: str = "gpt_oss"
    num_hidden_layers: int = 24
    hidden_size: int = 2880
    num_attention_heads: int = 64
    num_key_value_heads: int = 8
    head_dim: int = 64
    intermediate_size: int = 2880
    vocab_size: int = 201088
    num_local_experts: int = 32
    num_experts_per_tok: int = 4
    max_position_embeddings: int = 131072
    rope_theta: float = 150000.0
    hidden_act: str = "silu"
    rms_norm_eps: float = 1e-5
    sliding_window: int = 128
    attention_bias: bool = True
    tie_word_embeddings: bool = False
    router_aux_loss_coef: float = 0.9
    swiglu_limit: float = 7.0
    soft_skip_start_layer: int = 19
    hard_exit_layer: int = 22
    entropy_threshold: float = 0.20
    min_entropy_streak: int = 2
    enable_logit_blending: bool = False
    blending_confidence_threshold: float = 0.05
    blend_alpha: float = 0.10
    confidence_signal: str = "entropy"
    margin_threshold: float = 0.08
    blend_alpha_mode: str = "fixed"
    blend_alpha_sigmoid_scale: float = 20.0
    blend_entropy_sigmoid_scale: float = 20.0
    fallback_to_full_depth_on_ambiguity: bool = False
    blend_strategy: str = "full_vocab"
    blend_top_k: int = 5
    anchor_alpha_scale: float = 0.25
    prefill_step_size: int = 2048
    memory_limit_gb: float = 30.0
    cache_cleanup_interval: int = 32
    target_peak_memory_gb: float = 14.0

    layer_types: list = field(default_factory=lambda: [
        "sliding_attention" if i % 2 == 0 else "full_attention"
        for i in range(24)
    ])

    mxfp4_excluded: list = field(default_factory=lambda: [
        "model.layers.*.self_attn",
        "model.layers.*.mlp.router",
        "model.embed_tokens",
        "lm_head",
    ])

    @property
    def gqa_ratio(self) -> int:
        return self.num_attention_heads // self.num_key_value_heads

    @property
    def total_params_b(self) -> float:
        return 21.0

    @property
    def active_params_b(self) -> float:
        return 3.6
