"""On-disk cache around an OpenAI-style chat client, so config sweeps only
pay for calls they haven't already made.

Why it matters here: a sweep over top-k, similarity metric, retriever,
reasoning length L, or background-KB size does NOT change what the attacker's
Table A.1/A.2/A.3 prompts say -- so with a cache, every sweep point reuses the
SAME constructed decisions (zero new gpt-4o calls), and only agent calls whose
retrieved context actually changed are new. (L in particular is applied as a
post-hoc word truncation of CSRM's output, so an L sweep never re-calls CSRM.)
It also means every sweep point compares against identical decisions instead
of fresh, slightly different temperature-0 samples -- removing one source of
noise from the comparison.

Key = sha256 of the full request (model, temperature, messages, any other
kwargs). Only the response text and any tool calls (name + arguments) are stored --
all this codebase reads.
Trade-off, stated plainly: a cache hit returns the earlier answer verbatim, so
re-running the same config does NOT re-sample the model -- delete the file to
force fresh calls.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace


def _response(content: str, tool_calls: list[dict] | None = None):
    calls = [
        SimpleNamespace(function=SimpleNamespace(name=c["name"], arguments=c.get("arguments", "")))
        for c in (tool_calls or [])
    ]
    return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content, tool_calls=calls or None))])


class CacheMiss(RuntimeError):
    """Raised in offline mode (client=None) when a request isn't cached -- the
    guarantee that an offline run can never reach an API."""


class CachedChatClient:
    """Drop-in for `client.chat.completions.create(**kwargs)` -> response with
    `.choices[0].message.content`. `hits` / `misses` count cache use; `misses`
    is exactly the number of real API calls made.

    `client=None` is OFFLINE mode: cached requests are answered, anything else
    raises CacheMiss. Use it for studies that should cost nothing (retrieval
    rates, defense scoring) on top of an already-populated cache."""

    def __init__(self, client, path: Path | str):
        self._client = client
        self._path = Path(path)
        self._cache: dict[str, dict] = {}
        self.hits = 0
        self.misses = 0
        if self._path.exists():
            for line in self._path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    record = json.loads(line)
                    self._cache[record["key"]] = {"content": record["content"], "tool_calls": record.get("tool_calls", [])}
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    @staticmethod
    def _key(kwargs: dict) -> str:
        return hashlib.sha256(json.dumps(kwargs, sort_keys=True, default=str).encode("utf-8")).hexdigest()

    def _create(self, **kwargs):
        key = self._key(kwargs)
        if key in self._cache:
            self.hits += 1
            return _response(self._cache[key]["content"], self._cache[key]["tool_calls"])
        if self._client is None:
            raise CacheMiss(f"offline mode: request for model={kwargs.get('model')!r} is not in the cache ({self._path})")
        response = self._client.chat.completions.create(**kwargs)
        message = response.choices[0].message
        content = message.content or ""
        tool_calls = [
            {"name": c.function.name, "arguments": getattr(c.function, "arguments", "") or ""}
            for c in (getattr(message, "tool_calls", None) or [])
        ]
        self._cache[key] = {"content": content, "tool_calls": tool_calls}
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with self._path.open("a", encoding="utf-8") as f:
            f.write(json.dumps({"key": key, "model": kwargs.get("model"), "content": content, "tool_calls": tool_calls}) + "\n")
        self.misses += 1
        return _response(content, tool_calls)


class SeededClient:
    """Forces an explicit sampling temperature and OpenAI's best-effort `seed`
    onto every chat call, for the seed-stability experiment (Table 8). With our
    default temperature 0 a seed would change nothing, so a nonzero temperature
    is what makes seeds meaningful here. Wrap AROUND a CachedChatClient so seed
    and temperature are part of the cache key: SeededClient(CachedChatClient(raw)).
    `.inner` exposes the wrapped client (for its hit/miss counters)."""

    def __init__(self, client, seed: int, temperature: float):
        self.inner = client
        self._seed = seed
        self._temperature = temperature
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kwargs):
        return self.inner.chat.completions.create(**{**kwargs, "seed": self._seed, "temperature": self._temperature})
