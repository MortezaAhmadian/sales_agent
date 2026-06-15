"""Thin embedding wrapper supporting OpenAI and Ollama backends."""

import logging
import time

import requests as _requests

logger = logging.getLogger(__name__)


class Embedder:
    def __init__(self, config):
        self.config = config
        if config.embed_provider == "openai":
            import openai
            self._client = openai.OpenAI(api_key=config.openai_api_key)

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        if self.config.embed_provider == "openai":
            return self._embed_openai(texts)
        if self.config.embed_provider == "ollama":
            return self._embed_ollama(texts)
        raise ValueError(f"Unknown embed_provider: {self.config.embed_provider!r}")

    # ------------------------------------------------------------------

    def _embed_openai(self, texts: list[str]) -> list[list[float]]:
        import openai

        results: list[list[float]] = []
        bs = self.config.embed_batch_size

        for start in range(0, len(texts), bs):
            batch = texts[start : start + bs]
            for attempt in range(3):
                try:
                    resp = self._client.embeddings.create(
                        model=self.config.embed_model,
                        input=batch,
                    )
                    results.extend(e.embedding for e in resp.data)
                    break
                except openai.RateLimitError:
                    wait = 2 ** attempt
                    logger.warning(f"OpenAI rate-limited; retrying in {wait}s")
                    time.sleep(wait)
                except Exception as exc:
                    if attempt == 2:
                        raise
                    logger.warning(f"Embedding error ({exc}); retrying")
                    time.sleep(2 ** attempt)

        return results

    def _embed_ollama(self, texts: list[str]) -> list[list[float]]:
        results: list[list[float]] = []
        for text in texts:
            resp = _requests.post(
                f"{self.config.ollama_base_url}/api/embeddings",
                json={"model": self.config.embed_model, "prompt": text},
                timeout=60,
            )
            resp.raise_for_status()
            results.append(resp.json()["embedding"])
        return results
