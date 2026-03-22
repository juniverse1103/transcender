"""Model path resolution and loading for GPT-OSS 20B."""

from __future__ import annotations
from pathlib import Path
from typing import Optional


_GPT_OSS_METADATA_FILES = (
    "config.json",
    "tokenizer_config.json",
    "generation_config.json",
    "chat_template.jinja",
)


def _has_nonempty_metadata_file(path: Path) -> bool:
    try:
        return len(path.read_bytes()) > 0
    except OSError:
        return False


def _has_complete_gpt_oss_metadata(model_dir: Path) -> bool:
    return all(
        _has_nonempty_metadata_file(model_dir / filename)
        for filename in _GPT_OSS_METADATA_FILES
    )


def _find_cached_gpt_oss_snapshot() -> Optional[Path]:
    snapshots_dir = (
        Path.home()
        / ".cache"
        / "huggingface"
        / "hub"
        / "models--openai--gpt-oss-20b"
        / "snapshots"
    )
    if not snapshots_dir.exists():
        return None

    candidates = sorted(
        (path for path in snapshots_dir.iterdir() if path.is_dir()),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    for candidate in candidates:
        if _has_complete_gpt_oss_metadata(candidate):
            return candidate
    return None


def resolve_gpt_oss_model_path(model_path: str) -> str:
    """
    Resolve a usable local GPT-OSS model path.

    Some local copies contain sparse placeholder metadata files at the
    top level while still having valid weights. In that case, prefer an
    intact Hugging Face cache snapshot if one is available.
    """
    requested_path = Path(model_path).expanduser()
    if not requested_path.exists() or not requested_path.is_dir():
        return model_path

    if _has_complete_gpt_oss_metadata(requested_path):
        return str(requested_path)

    looks_like_gpt_oss = (
        (requested_path / "model.safetensors.index.json").exists()
        or (requested_path / "original" / "config.json").exists()
    )
    if not looks_like_gpt_oss:
        return str(requested_path)

    cached_snapshot = _find_cached_gpt_oss_snapshot()
    if cached_snapshot is not None:
        return str(cached_snapshot)

    missing = [
        filename
        for filename in _GPT_OSS_METADATA_FILES
        if not _has_nonempty_metadata_file(requested_path / filename)
    ]
    raise RuntimeError(
        "Local GPT-OSS model metadata is empty or missing under "
        f"{requested_path}: {', '.join(missing)}. Re-download the model or "
        "point --model at a valid Hugging Face cache snapshot."
    )


def load_resolved_mlx_model(model_path: str, lazy: bool = True):
    """Load GPT-OSS through mlx_lm after resolving metadata placeholders."""
    from mlx_lm import load as mlx_load

    resolved_model_path = resolve_gpt_oss_model_path(model_path)
    model, tokenizer = mlx_load(resolved_model_path, lazy=lazy)
    return model, tokenizer, resolved_model_path


def load_resolved_transformers_model(model_path: str):
    """Load GPT-OSS through transformers after resolving metadata placeholders."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    resolved_model_path = resolve_gpt_oss_model_path(model_path)
    tokenizer = AutoTokenizer.from_pretrained(resolved_model_path)
    model = AutoModelForCausalLM.from_pretrained(
        resolved_model_path,
        torch_dtype="auto",
        device_map="auto",
        attn_implementation="eager",
    )
    return model, tokenizer, resolved_model_path
