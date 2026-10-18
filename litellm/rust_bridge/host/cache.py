from __future__ import annotations

import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Annotated, Final, Literal, TypeAlias

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter

from litellm.types.utils import CallTypes

if TYPE_CHECKING:
    from litellm.caching.caching import Cache
    from litellm.caching.dual_cache import DualCache


class ResponseOutput(BaseModel):
    model_config = ConfigDict(frozen=True)
    kind: Literal["Response"] = "Response"
    value: object


class StreamOutput(BaseModel):
    model_config = ConfigDict(frozen=True)
    kind: Literal["Stream"] = "Stream"
    value: str


CachedOutput: TypeAlias = ResponseOutput | StreamOutput
_OUTPUT: Final = TypeAdapter[CachedOutput](Annotated[CachedOutput, Field(discriminator="kind")])
_ENTRY: Final = TypeAdapter(dict[str, object])
_EVENTS: Final = TypeAdapter(list[str])
_SSE_EVENT_BOUNDARY: Final = re.compile(r"(?<=\n\n)")


@dataclass(frozen=True, slots=True)
class EntryFormat:
    entry: Callable[[CachedOutput], object | None]
    output: Callable[[object, bool], CachedOutput | None]
    tiered_writes: bool


def _response_entry(output: CachedOutput) -> object | None:
    match output:
        case ResponseOutput(value=response):
            return response
        case StreamOutput():
            return None


def _response_output(entry: object, stream: bool) -> CachedOutput | None:
    return None if stream else ResponseOutput(value=entry)


def _messages_entry(output: CachedOutput) -> object:
    from litellm.types.caching import CACHED_STREAM_EVENTS_KEY

    match output:
        case ResponseOutput(value=response):
            return response
        case StreamOutput(value=text):
            return {CACHED_STREAM_EVENTS_KEY: [event for event in _SSE_EVENT_BOUNDARY.split(text) if event]}


def _stream_events(entry: object) -> list[str] | None:
    from litellm.types.caching import CACHED_STREAM_EVENTS_KEY

    if not isinstance(entry, dict):
        return None
    events: Final = _ENTRY.validate_python(entry).get(CACHED_STREAM_EVENTS_KEY)
    return None if events is None else _EVENTS.validate_python(events)


def _messages_output(entry: object, stream: bool) -> CachedOutput:
    events: Final = _stream_events(entry)
    if events is None:
        return ResponseOutput(value=entry)
    return StreamOutput(value="".join(events))


RESPONSE_ENTRY: Final = EntryFormat(entry=_response_entry, output=_response_output, tiered_writes=True)
MESSAGES_ENTRY: Final = EntryFormat(entry=_messages_entry, output=_messages_output, tiered_writes=False)


def entry_format(call_type: CallTypes) -> EntryFormat:
    match call_type:
        case CallTypes.anthropic_messages:
            return MESSAGES_ENTRY
        case _:
            return RESPONSE_ENTRY


@dataclass(frozen=True, slots=True)
class CacheAccess:
    cache: Cache
    request: Mapping[str, object]
    key: str
    read: bool
    write: bool
    memory_tier: DualCache | None
    format: EntryFormat


def _key(cache: Cache, kwargs: Mapping[str, object], request: Mapping[str, object]) -> str | None:
    from litellm._logging import verbose_logger
    from litellm.caching.caching_handler import response_cache_key

    try:
        return response_cache_key(cache, kwargs) if cache.should_use_cache(**request) else None
    except Exception:  # noqa: BLE001  # a cache failure is a miss, never a request failure, as in Cache.get_cache
        verbose_logger.exception("LiteLLM Cache: preparing the native cache request failed")
        return None


def grant(cache: Cache, kwargs: Mapping[str, object], call_type: CallTypes) -> CacheAccess | None:
    from litellm.caching.caching_handler import cache_access, cache_key_kwargs, response_memory_tier

    model: Final = kwargs.get("model")
    read, write = cache_access(model if isinstance(model, str) else None, kwargs, call_type)
    if not (read or write):
        return None
    request: Final = cache_key_kwargs(kwargs)
    key: Final = _key(cache, kwargs, request)
    if key is None:
        return None
    return CacheAccess(
        cache=cache,
        request=request,
        key=key,
        read=read,
        write=write,
        memory_tier=response_memory_tier(cache),
        format=entry_format(call_type),
    )


def access(kwargs: Mapping[str, object], call_type: str) -> CacheAccess | None:
    import litellm

    cache: Final = litellm.cache
    return None if cache is None else grant(cache, kwargs, CallTypes(call_type))


@dataclass(frozen=True, slots=True)
class CacheHit:
    response: Mapping[str, object]
    key: str


def _hit(access: CacheAccess, entry: object) -> CacheHit | None:
    output: Final = access.format.output(entry, access.request.get("stream") is True)
    return None if output is None else CacheHit(response=output.model_dump(), key=access.key)


def lookup(access: CacheAccess) -> CacheHit | None:
    from litellm.caching.caching_handler import is_response_without_output

    result: Final = access.cache.get_cache(dynamic_cache_object=None, cache_key=access.key, **access.request)  # pyright: ignore[reportUnknownMemberType]  # Cache.get_cache kwargs are untyped
    if result is None or is_response_without_output(result):
        return None
    return _hit(access, result)


async def alookup(access: CacheAccess) -> CacheHit | None:
    from litellm.caching.caching_handler import is_response_without_output

    result: Final = await access.cache.async_get_cache(  # pyright: ignore[reportUnknownMemberType]  # Cache.async_get_cache kwargs are untyped
        dynamic_cache_object=access.memory_tier, cache_key=access.key, **access.request
    )
    if result is None:
        return None
    if is_response_without_output(result):
        if access.memory_tier is not None:
            access.memory_tier.in_memory_cache.delete_cache(access.key)
        return None
    return _hit(access, result)


def _entry(access: CacheAccess, value: object) -> object | None:
    from litellm.caching.caching_handler import is_response_without_output

    entry: Final = access.format.entry(_OUTPUT.validate_python(value))
    return None if entry is None or is_response_without_output(entry) else entry


def store(access: CacheAccess, value: object) -> None:
    entry: Final = _entry(access, value)
    if entry is not None:
        access.cache.add_cache(entry, cache_key=access.key, **access.request)


async def astore(access: CacheAccess, value: object) -> None:
    from litellm.caching.caching_handler import create_cache_write_task

    entry: Final = _entry(access, value)
    if entry is None:
        return
    tier: Final = access.memory_tier if access.format.tiered_writes else None
    create_cache_write_task(
        lambda: access.cache.async_add_cache(entry, dynamic_cache_object=tier, cache_key=access.key, **access.request)
    )
