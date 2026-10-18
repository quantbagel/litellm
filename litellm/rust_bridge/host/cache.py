from __future__ import annotations

import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Final, Literal, TypeAlias

from pydantic import TypeAdapter
from typing_extensions import ReadOnly, TypedDict

if TYPE_CHECKING:
    from litellm.caching.caching import Cache
    from litellm.caching.dual_cache import DualCache


class ResponseOutput(TypedDict):
    kind: ReadOnly[Literal["Response"]]
    value: ReadOnly[object]


class StreamOutput(TypedDict):
    kind: ReadOnly[Literal["Stream"]]
    value: ReadOnly[str]


CachedOutput: TypeAlias = ResponseOutput | StreamOutput
_OUTPUT: Final = TypeAdapter[CachedOutput](CachedOutput)
_ENTRY: Final = TypeAdapter(dict[str, object])
_EVENTS: Final = TypeAdapter(list[str])
_SSE_EVENT_BOUNDARY: Final = re.compile(r"(?<=\n\n)")


@dataclass(frozen=True, slots=True)
class EntryFormat:
    entry: Callable[[CachedOutput], object | None]
    output: Callable[[object, bool], CachedOutput | None]
    tiered_writes: bool


def _response_entry(output: CachedOutput) -> object | None:
    return output["value"] if output["kind"] == "Response" else None


def _response_output(entry: object, stream: bool) -> CachedOutput | None:
    return None if stream else ResponseOutput(kind="Response", value=entry)


def _messages_entry(output: CachedOutput) -> object:
    from litellm.types.caching import CACHED_STREAM_EVENTS_KEY

    if output["kind"] == "Response":
        return output["value"]
    return {CACHED_STREAM_EVENTS_KEY: [event for event in _SSE_EVENT_BOUNDARY.split(output["value"]) if event]}


def _stream_events(entry: object) -> list[str] | None:
    from litellm.types.caching import CACHED_STREAM_EVENTS_KEY

    if not isinstance(entry, dict):
        return None
    events: Final = _ENTRY.validate_python(entry).get(CACHED_STREAM_EVENTS_KEY)
    return None if events is None else _EVENTS.validate_python(events)


def _messages_output(entry: object, stream: bool) -> CachedOutput:
    events: Final = _stream_events(entry)
    if events is None:
        return ResponseOutput(kind="Response", value=entry)
    return StreamOutput(kind="Stream", value="".join(events))


RESPONSE_ENTRY: Final = EntryFormat(entry=_response_entry, output=_response_output, tiered_writes=True)
MESSAGES_ENTRY: Final = EntryFormat(entry=_messages_entry, output=_messages_output, tiered_writes=False)


def entry_format(call_type: str) -> EntryFormat:
    match call_type:
        case "anthropic_messages":
            return MESSAGES_ENTRY
        case _:
            return RESPONSE_ENTRY


@dataclass(frozen=True, slots=True)
class CacheAccess:
    cache: Cache
    read: bool
    write: bool
    memory_tier: DualCache | None
    format: EntryFormat


def access(kwargs: Mapping[str, object], call_type: str) -> CacheAccess | None:
    import litellm
    from litellm.caching.caching_handler import cache_access, response_memory_tier

    cache: Final = litellm.cache
    if cache is None:
        return None
    model: Final = kwargs.get("model")
    read, write = cache_access(model if isinstance(model, str) else None, kwargs, call_type)
    if not (read or write):
        return None
    return CacheAccess(
        cache=cache,
        read=read,
        write=write,
        memory_tier=response_memory_tier(cache),
        format=entry_format(call_type),
    )


@dataclass(frozen=True, slots=True)
class CacheHit:
    response: CachedOutput
    key: str


def _request(cache: Cache, kwargs: Mapping[str, object]) -> tuple[str, Mapping[str, object]]:
    request: Final = {name: value for name, value in kwargs.items() if name != "cache_key"}
    preset: Final = kwargs.get("cache_key")
    return (preset if isinstance(preset, str) and preset else cache.get_cache_key(**request), request)


def _hit(access: CacheAccess, entry: object, key: str, kwargs: Mapping[str, object]) -> CacheHit | None:
    output: Final = access.format.output(entry, kwargs.get("stream") is True)
    return None if output is None else CacheHit(response=output, key=key)


def lookup(access: CacheAccess, kwargs: Mapping[str, object]) -> CacheHit | None:
    from litellm.caching.caching_handler import is_response_without_output

    key, request = _request(access.cache, kwargs)
    result: Final = access.cache.get_cache(dynamic_cache_object=None, cache_key=key, **request)  # pyright: ignore[reportUnknownMemberType]  # Cache.get_cache kwargs are untyped
    if result is None or is_response_without_output(result):
        return None
    return _hit(access, result, key, kwargs)


async def alookup(access: CacheAccess, kwargs: Mapping[str, object]) -> CacheHit | None:
    from litellm.caching.caching_handler import is_response_without_output

    key, request = _request(access.cache, kwargs)
    result: Final = await access.cache.async_get_cache(  # pyright: ignore[reportUnknownMemberType]  # Cache.async_get_cache kwargs are untyped
        dynamic_cache_object=access.memory_tier, cache_key=key, **request
    )
    if result is None:
        return None
    if is_response_without_output(result):
        if access.memory_tier is not None:
            access.memory_tier.in_memory_cache.delete_cache(key)
        return None
    return _hit(access, result, key, kwargs)


def _entry(access: CacheAccess, value: object) -> object | None:
    from litellm.caching.caching_handler import is_response_without_output

    entry: Final = access.format.entry(_OUTPUT.validate_python(value))
    return None if entry is None or is_response_without_output(entry) else entry


def store(access: CacheAccess, value: object, kwargs: Mapping[str, object]) -> None:
    entry: Final = _entry(access, value)
    if entry is not None:
        access.cache.add_cache(entry, **kwargs)


async def astore(access: CacheAccess, value: object, kwargs: Mapping[str, object]) -> None:
    from litellm.caching.caching_handler import create_cache_write_task

    entry: Final = _entry(access, value)
    if entry is None:
        return
    tier: Final = access.memory_tier if access.format.tiered_writes else None
    create_cache_write_task(lambda: access.cache.async_add_cache(entry, dynamic_cache_object=tier, **kwargs))
