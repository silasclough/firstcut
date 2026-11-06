"""LLM providers with a fallback chain: Claude API -> local Ollama -> offline template.

Every call asks for JSON matching a schema. Claude uses structured outputs
(output_config.format), Ollama uses its `format` JSON-schema option, and the template
"provider" is a per-task Python function supplied by the caller (offline testing only).
"""

from __future__ import annotations

import json
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable

import requests

from .log import get_logger

log = get_logger("llm")


class LLMUnavailable(RuntimeError):
    pass


class LLMRefused(RuntimeError):
    pass


@dataclass
class LLMResult:
    data: dict
    provider: str
    model: str
    usage: dict = field(default_factory=dict)


class LLMProvider(ABC):
    name = "base"

    def __init__(self, cfg: dict):
        self.cfg = cfg or {}

    @abstractmethod
    def available(self) -> tuple[bool, str]: ...

    @abstractmethod
    def complete_json(self, system: str, prompt: str, schema: dict, images: list[bytes] | None = None) -> LLMResult: ...


class AnthropicLLM(LLMProvider):
    name = "anthropic"

    def available(self) -> tuple[bool, str]:
        try:
            import anthropic  # noqa: F401
        except ImportError:
            return False, "anthropic package not installed"
        if not (os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN")):
            return False, "ANTHROPIC_API_KEY not set"
        return True, "ok"

    def complete_json(self, system: str, prompt: str, schema: dict, images: list[bytes] | None = None) -> LLMResult:
        import anthropic

        client = anthropic.Anthropic(max_retries=3, timeout=300.0)
        model = self.cfg.get("model", "claude-opus-5-5")
        kwargs: dict[str, Any] = dict(
            model=model,
            max_tokens=int(self.cfg.get("max_tokens", 16000)),
            system=system,
            messages=[{"role": "user", "content": prompt}],
            output_config={"effort": self.cfg.get("effort", "medium"),
                           "format": {"type": "json_schema", "schema": schema}},
        )
        use_fallbacks = bool(self.cfg.get("server_fallbacks", True))
        try:
            if use_fallbacks:
                resp = client.beta.messages.create(**kwargs, betas=["server-side-fallback-2026-07-01"],
                                                   fallbacks="default")
            else:
                resp = client.messages.create(**kwargs)
        except anthropic.BadRequestError as e:
            if use_fallbacks and "fallback" in str(e).lower():
                log.warning("server-side fallbacks rejected; retrying without them")
                resp = client.messages.create(**kwargs)
            else:
                raise
        except anthropic.AuthenticationError as e:
            raise LLMUnavailable(f"Anthropic auth failed: {e.message}") from e

        if resp.stop_reason == "refusal":
            details = getattr(resp, "stop_details", None)
            raise LLMRefused(f"Claude declined ({getattr(details, 'category', None)})")
        if resp.stop_reason == "max_tokens":
            raise RuntimeError("Claude hit max_tokens before finishing the JSON")
        text = next((b.text for b in resp.content if b.type == "text"), "")
        usage = {"input_tokens": resp.usage.input_tokens, "output_tokens": resp.usage.output_tokens,
                 "request_id": getattr(resp, "_request_id", None), "served_by": resp.model}
        return LLMResult(json.loads(text), self.name, model, usage)


class OllamaLLM(LLMProvider):
    name = "ollama"

    @property
    def base(self) -> str:
        return self.cfg.get("base_url", "http://localhost:11434").rstrip("/")

    def available(self) -> tuple[bool, str]:
        try:
            r = requests.get(f"{self.base}/api/tags", timeout=3)
            r.raise_for_status()
        except Exception as e:  # noqa: BLE001
            return False, f"Ollama not reachable at {self.base} ({type(e).__name__})"
        names = {m.get("name") for m in r.json().get("models", [])}
        self._model = next((m for m in [self.cfg.get("model", "qwen3:14b"), *self.cfg.get("fallback_models", [])]
                            if m in names or f"{m}:latest" in names), None)
        if not self._model:
            return False, (f"Ollama model '{self.cfg.get('model')}' not pulled (ollama pull {self.cfg.get('model')}; "
                           f"have: {', '.join(sorted(names)) or 'none'})")
        if self._model != self.cfg.get("model"):
            log.warning(f"Ollama model {self.cfg.get('model')} not pulled; using fallback {self._model}")
        return True, "ok"

    def complete_json(self, system: str, prompt: str, schema: dict, images: list[bytes] | None = None) -> LLMResult:
        import base64
        model = getattr(self, "_model", None) or self.cfg.get("model", "qwen3:14b")
        if images:
            model = self.cfg.get("vision_model") or model
            if not self.cfg.get("vision_model"):
                raise LLMUnavailable("no Ollama vision_model configured for image input")
        user = {"role": "user", "content": prompt + "\n\nRespond with JSON only."}
        if images:
            user["images"] = [base64.b64encode(b).decode() for b in images]
        body = {
            "model": model,
            "messages": [{"role": "system", "content": system}, user],
            "stream": False,
            "format": schema,
            "think": False,
            "keep_alive": self.cfg.get("keep_alive", "5m"),   # "0" frees RAM right away (for image models)
            # A cap on the answer: a JSON answer that starts repeating itself never ends on its own, and Ollama keeps generating after the
            # caller gives up, so every request behind it times out too (the work Mac's catalogue, 2026-11-06: 6,300 tokens and counting).
            # Pictures are described more calmly (a fixed form to fill, not writing).
            "options": {"temperature": float(self.cfg.get("temperature", 0.5 if images else 0.9)),
                        "num_ctx": int(self.cfg.get("num_ctx", 16384 if images else 8192)),
                        "num_predict": int(self.cfg.get("num_predict", 2048 if images else 8192))},
        }
        r = requests.post(f"{self.base}/api/chat", json=body, timeout=float(self.cfg.get("timeout_s", 600)))
        r.raise_for_status()
        d = r.json()
        content = d.get("message", {}).get("content", "")
        usage = {"input_tokens": d.get("prompt_eval_count"), "output_tokens": d.get("eval_count"),
                 "seconds": round((d.get("total_duration") or 0) / 1e9, 1)}
        note_speed(model, d.get("eval_count"), d.get("eval_duration"))
        return LLMResult(json.loads(content), self.name, model, usage)


# How fast this Mac's model writes (tokens a second, from Ollama's own counts), for the app's sidebar when the AI is "only this Mac".
LOCAL_SPEED: dict = {}
LOCAL_SPEEDS: list = []                     # the last 24 answers' speeds, oldest first


def note_speed(model: str, tokens, ns) -> None:
    try:
        if not tokens or not ns or int(tokens) < 8:
            return
        tps = round(int(tokens) / (int(ns) / 1e9), 1)
        LOCAL_SPEED.update(tps=tps, model=model, at=__import__("time").time(), tokens=int(tokens))
        LOCAL_SPEEDS.append(tps)
        del LOCAL_SPEEDS[:-24]
    except (TypeError, ValueError, ZeroDivisionError):
        pass


class OpenAILLM(LLMProvider):
    """OpenAI chat completions with a JSON-schema response format (needs OPENAI_API_KEY)."""
    name = "openai"

    def _key(self) -> str | None:
        return self.cfg.get("api_key") or os.environ.get("OPENAI_API_KEY")

    def available(self) -> tuple[bool, str]:
        if not self._key():
            return False, "OPENAI_API_KEY not set"
        return True, "ok"

    def complete_json(self, system: str, prompt: str, schema: dict, images: list[bytes] | None = None) -> LLMResult:
        import base64
        model = self.cfg.get("model", "gpt-4.1-mini")
        content: Any = prompt
        if images:
            content = [{"type": "text", "text": prompt}] + [
                {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + base64.b64encode(b).decode(),
                                                    "detail": self.cfg.get("image_detail", "high")}} for b in images]
        body = {
            "model": model,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": content}],
            "response_format": {"type": "json_schema", "json_schema": {"name": "result", "schema": schema}},
        }
        if self.cfg.get("reasoning_effort"):               # gpt-5 / o-series: low = faster, cheaper
            body["reasoning_effort"] = self.cfg["reasoning_effort"]
        r = requests.post(f"{self.cfg.get('base_url', 'https://api.openai.com/v1').rstrip('/')}/chat/completions",
                          headers={"Authorization": f"Bearer {self._key()}"},
                          json=body, timeout=float(self.cfg.get("timeout_s", 300)))
        if r.status_code in (401, 403):
            raise LLMUnavailable(f"OpenAI auth failed ({r.status_code})")
        r.raise_for_status()
        d = r.json()
        choice = d["choices"][0]
        if choice.get("message", {}).get("refusal"):
            raise LLMRefused("OpenAI declined")
        usage = d.get("usage", {}) or {}
        return LLMResult(json.loads(choice["message"]["content"]), self.name, d.get("model", model),
                         {"input_tokens": usage.get("prompt_tokens"), "output_tokens": usage.get("completion_tokens")})


PROVIDERS: dict[str, type[LLMProvider]] = {"anthropic": AnthropicLLM, "ollama": OllamaLLM, "openai": OpenAILLM,
                                           "ollama_remote": OllamaLLM}   # another machine's Ollama (cfg llm.ollama_remote)


class LLMRouter:
    def __init__(self, llm_cfg: dict, dry_run: bool = False, chain: list[str] | None = None):
        self.cfg = llm_cfg or {}
        self.dry_run = dry_run
        self.chain = chain or self.cfg.get("provider_chain", ["anthropic", "ollama", "template"])

    def status(self) -> list[tuple[str, bool, str]]:
        out = []
        for name in self.chain:
            if name == "template":
                out.append((name, True, "offline template (testing only)"))
                continue
            cls = PROVIDERS.get(name)
            if not cls:
                out.append((name, False, "unknown provider"))
                continue
            ok, why = cls(self.cfg.get(name, {})).available()
            out.append((name, ok, why))
        return out

    def complete_json(self, task: str, system: str, prompt: str, schema: dict,
                      fallback: Callable[[], dict] | None = None,
                      validate: Callable[[dict], None] | None = None,
                      images: list[bytes] | None = None) -> LLMResult:
        if self.dry_run:
            if fallback is None:
                raise LLMUnavailable(f"[dry-run] no offline fallback for task {task}")
            log.info(f"[dry-run] skipping LLM call for {task}; using offline template")
            return LLMResult(fallback(), "dry-run", "template")
        errors = []
        for name in self.chain:
            if name == "template":
                if fallback is not None:
                    log.warning(f"{task}: using offline template writer (no LLM available)")
                    return LLMResult(fallback(), "template", "template")
                continue
            cls = PROVIDERS.get(name)
            if not cls:
                continue
            provider = cls(self.cfg.get(name, {}))
            ok, why = provider.available()
            if not ok:
                errors.append(f"{name}: {why}")
                log.debug(f"{task}: skip {name}: {why}")
                continue
            try:
                res = provider.complete_json(system, prompt, schema, images=images) if images else \
                    provider.complete_json(system, prompt, schema)
                if validate:
                    validate(res.data)
                log.info(f"{task}: ok via {name}/{res.model}", extra={"provider": name, "usage": res.usage})
                return res
            except Exception as e:  # noqa: BLE001 - fall through to next provider
                errors.append(f"{name}: {type(e).__name__}: {e}")
                log.warning(f"{task}: {name} failed: {type(e).__name__}: {e}")
        raise LLMUnavailable(f"{task}: no LLM provider succeeded: " + "; ".join(errors))


class RemoteJson:
    """Asks another machine's AI (the posting server, which holds the OpenAI access) to answer one JSON question over SSH.
    Only the prompt text travels - the key never leaves the server. Used for the writing that needs the best model."""

    def __init__(self, host: str, project: str = "clipfarm", timeout_s: float = 300.0):
        self.host, self.project, self.timeout_s = host, project, timeout_s

    def complete_json(self, task: str, system: str, prompt: str, schema: dict, **kw) -> "LLMResult":
        import json
        import subprocess
        import base64
        req = {"task": task, "system": system, "prompt": prompt, "schema": schema, "with_usage": True}
        if kw.get("images"):                                     # frames travel as small JPEGs; the key still never leaves the server
            req["images"] = [base64.b64encode(b).decode() for b in kw["images"]]
        if kw.get("model"):
            req["model"] = kw["model"]
        payload = json.dumps(req)
        cmd = ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=20", self.host,
               f"cd ~/{self.project} && .venv/bin/python -m clipfarm llm-json"]
        proc = subprocess.run(cmd, input=payload, capture_output=True, text=True, timeout=self.timeout_s)
        if proc.returncode != 0:
            raise LLMUnavailable(f"remote AI on {self.host} failed: {(proc.stderr or proc.stdout)[-300:]}")
        try:
            data = json.loads(proc.stdout.strip().splitlines()[-1])
        except (json.JSONDecodeError, IndexError) as e:
            raise LLMUnavailable(f"remote AI answered with something unreadable: {e}") from e
        usage: dict = {}
        model = f"server:{self.host}"
        if isinstance(data, dict) and set(data) <= {"data", "usage", "model"} and "data" in data and "usage" in data:
            usage, model = data["usage"] or {}, data.get("model") or model          # newer server: the answer plus what it cost
            data = data["data"]
        return LLMResult(data, "openai", model, usage)


class Layered:
    """Try the first AI; if it can't answer, use the second."""

    def __init__(self, first, second, tasks: set[str] | None = None):
        self.first, self.second, self.tasks = first, second, tasks

    def complete_json(self, task: str, *a, **kw):
        if self.tasks is None or task in self.tasks:
            try:
                return self.first.complete_json(task, *a, **kw)
            except Exception as e:  # noqa: BLE001
                log.warning(f"{task}: the stronger AI wasn't available ({e}); using the local one")
        return self.second.complete_json(task, *a, **kw)

    def __getattr__(self, name):
        return getattr(self.second, name)
