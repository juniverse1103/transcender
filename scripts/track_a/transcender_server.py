"""
FastAPI wrapper for the Transcender MLX engine.

This exposes:
  - /v1/chat/completions
  - /v1/engine/stats

The server applies the GPT-OSS Harmony template internally with
reasoning_effort="medium" for every request.
"""

from __future__ import annotations

import argparse
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    from fastapi import FastAPI, HTTPException
    FASTAPI_AVAILABLE = True
    FASTAPI_IMPORT_ERROR = None
except Exception as exc:  # pragma: no cover - exercised at runtime
    FastAPI = None
    HTTPException = RuntimeError
    FASTAPI_AVAILABLE = False
    FASTAPI_IMPORT_ERROR = exc

try:
    import uvicorn
except Exception:  # pragma: no cover - exercised at runtime
    uvicorn = None

from transcender_engine import GptOssConfig, MLXDynamicExpertEngine, load_resolved_mlx_model


DEFAULT_MODEL_PATH = str(
    (Path(__file__).resolve().parent.parent / "gpt-oss-20b-raw").resolve()
)


def _coerce_message_content(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        text_parts: List[str] = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                text_parts.append(str(item.get("text", "")))
        return "\n".join(part for part in text_parts if part)
    return str(content)


def _normalize_messages(raw_messages: Any) -> List[Dict[str, str]]:
    if not isinstance(raw_messages, list) or not raw_messages:
        raise ValueError("messages must be a non-empty list.")

    messages: List[Dict[str, str]] = []
    for item in raw_messages:
        if not isinstance(item, dict):
            raise ValueError("each message must be an object.")
        role = str(item.get("role", "")).strip()
        if not role:
            raise ValueError("each message must include a role.")
        messages.append(
            {
                "role": role,
                "content": _coerce_message_content(item.get("content", "")),
            }
        )
    return messages


class TranscenderService:
    def __init__(
        self,
        model_path: str,
        config: GptOssConfig,
        consistency_check_interval: int = 10,
        consistency_check_tokens: int = 16,
    ):
        self.model_path = model_path
        self.config = config
        self.consistency_check_interval = max(consistency_check_interval, 1)
        self.consistency_check_tokens = max(consistency_check_tokens, 1)

        self._engine: Optional[MLXDynamicExpertEngine] = None
        self._load_error: Optional[str] = None
        self._resolved_model_path: Optional[str] = None
        self._status = "initializing"
        self._request_count = 0
        self._last_request_at: Optional[float] = None

        self._load_lock = threading.Lock()
        self._inference_lock = threading.Lock()
        self._thread: Optional[threading.Thread] = None

    def start_background_load(self):
        with self._load_lock:
            if self._thread is not None:
                return
            self._status = "loading"
            self._thread = threading.Thread(
                target=self._load_engine,
                name="transcender-loader",
                daemon=True,
            )
            self._thread.start()

    def _load_engine(self):
        try:
            model, tokenizer, resolved_model_path = load_resolved_mlx_model(
                self.model_path,
                lazy=True,
            )
            self._resolved_model_path = resolved_model_path
            self._engine = MLXDynamicExpertEngine(
                model=model,
                tokenizer=tokenizer,
                model_path=resolved_model_path,
                config=self.config,
                soft_skip_start_layer=self.config.soft_skip_start_layer,
                hard_exit_layer=self.config.hard_exit_layer,
                entropy_threshold=self.config.entropy_threshold,
                min_entropy_streak=self.config.min_entropy_streak,
                prefill_step_size=self.config.prefill_step_size,
                memory_limit_gb=self.config.memory_limit_gb,
            )
            self._status = "ready"
        except Exception as exc:  # pragma: no cover - exercised at runtime
            self._load_error = str(exc)
            self._status = "error"

    @property
    def is_ready(self) -> bool:
        return self._engine is not None and self._status == "ready"

    def require_engine(self) -> MLXDynamicExpertEngine:
        if self._status == "error":
            raise RuntimeError(self._load_error or "Engine failed to load.")
        if not self.is_ready:
            raise RuntimeError("Engine is still loading.")
        return self._engine

    def get_stats(self) -> Dict[str, Any]:
        stats: Dict[str, Any] = {
            "status": self._status,
            "model_path": self.model_path,
            "resolved_model_path": self._resolved_model_path,
            "load_error": self._load_error,
            "requests_seen": self._request_count,
            "last_request_at": self._last_request_at,
            "consistency_check_interval": self.consistency_check_interval,
            "consistency_check_tokens": self.consistency_check_tokens,
        }
        if self._engine is not None:
            stats["engine"] = self._engine.get_runtime_stats()
        return stats

    def _should_run_consistency_check(self) -> bool:
        return self._request_count % self.consistency_check_interval == 0

    def chat_completions(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        engine = self.require_engine()
        messages = _normalize_messages(payload.get("messages"))
        max_tokens = int(
            payload.get(
                "max_completion_tokens",
                payload.get("max_tokens", 128),
            )
        )
        if bool(payload.get("stream", False)):
            raise ValueError("stream=true is not implemented.")

        with self._inference_lock:
            self._request_count += 1
            stats = engine.generate_from_messages(
                messages=messages,
                max_new_tokens=max_tokens,
                dynamic=True,
                reasoning_effort="medium",
            )

            consistency = None
            if self._should_run_consistency_check():
                consistency = engine.consistency_check_from_messages(
                    messages=messages,
                    max_new_tokens=min(max_tokens, self.consistency_check_tokens),
                    reasoning_effort="medium",
                )

        self._last_request_at = time.time()

        usage = {
            "prompt_tokens": stats["prompt_tokens"],
            "completion_tokens": stats["tokens_generated"],
            "total_tokens": stats["prompt_tokens"] + stats["tokens_generated"],
        }
        transcender = {
            "reasoning_effort": "medium",
            "ttft_s": stats["ttft_s"],
            "peak_memory_gb": stats["peak_memory_gb"],
            "cache_memory_gb": stats["cache_memory_gb"],
            "avg_layers": stats["avg_layers"],
            "avg_layers_saved": max(engine.num_layers - stats["avg_layers"], 0.0),
            "cache_policy": stats["cache_policy"],
            "consistency_check": consistency,
        }

        return {
            "id": f"chatcmpl-{uuid.uuid4().hex}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": str(payload.get("model", "gpt-oss-20b-transcender-v1")),
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": stats["output_text"],
                    },
                    "finish_reason": "stop",
                }
            ],
            "usage": usage,
            "transcender": transcender,
        }


def create_app(
    model_path: str = DEFAULT_MODEL_PATH,
    config: Optional[GptOssConfig] = None,
    consistency_check_interval: int = 10,
    consistency_check_tokens: int = 16,
):
    if not FASTAPI_AVAILABLE:
        raise RuntimeError(
            "FastAPI is not installed. Install fastapi and uvicorn before "
            f"starting transcender_server.py. Import error: {FASTAPI_IMPORT_ERROR}"
        )

    from contextlib import asynccontextmanager

    config = config or GptOssConfig()
    service = TranscenderService(
        model_path=model_path,
        config=config,
        consistency_check_interval=consistency_check_interval,
        consistency_check_tokens=consistency_check_tokens,
    )

    @asynccontextmanager
    async def lifespan(app):
        service.start_background_load()
        yield

    app = FastAPI(title="Transcender v1.0", version="1.0.0", lifespan=lifespan)

    @app.get("/v1/engine/stats")
    def engine_stats():
        return service.get_stats()

    @app.post("/v1/chat/completions")
    def chat_completions(payload: Dict[str, Any]):
        try:
            return service.chat_completions(payload)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))
        except RuntimeError as exc:
            raise HTTPException(status_code=503, detail=str(exc))

    return app


def main():
    parser = argparse.ArgumentParser(description="Transcender FastAPI server")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--soft-skip-layer", type=int, default=19)
    parser.add_argument("--exit-layer", type=int, default=22)
    parser.add_argument("--entropy-threshold", type=float, default=0.20)
    parser.add_argument("--memory-limit-gb", type=float, default=30.0)
    parser.add_argument("--consistency-check-interval", type=int, default=10)
    parser.add_argument("--consistency-check-tokens", type=int, default=16)
    args = parser.parse_args()

    config = GptOssConfig(
        soft_skip_start_layer=args.soft_skip_layer,
        hard_exit_layer=args.exit_layer,
        entropy_threshold=args.entropy_threshold,
        memory_limit_gb=args.memory_limit_gb,
    )
    app = create_app(
        model_path=args.model,
        config=config,
        consistency_check_interval=args.consistency_check_interval,
        consistency_check_tokens=args.consistency_check_tokens,
    )

    if uvicorn is None:
        raise RuntimeError("uvicorn is not installed. Install it before starting the server.")
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
