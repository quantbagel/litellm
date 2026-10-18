# rules

- Mirrors `litellm/integrations/` and owns deterministic integration behavior, independent of provider I/O and host callbacks
- Shared request contracts belong in `litellm-llms-types`; runtime configuration and provider capability checks are inputs from callers
- Function names follow the Python hook's static methods so a drift report diff maps onto them
- Count breakpoints only where the provider reads them: system blocks, message-level marks, content blocks, tools (top level or under `function`) and the top-level `cache_control`. Explicit nulls are not marks
- Each distinct message target is resolved once against a role index, so cost stays linear in points plus messages
- Use inline named `rstest` cases for behavior

# Known partial ports

- `tool_config` points reserve one slot of the four-breakpoint cap and are never placed. No native Messages provider builds Bedrock Converse `toolConfig`
- OpenAI prompt-cache dialects (`prompt_cache_breakpoint`, `_litellm_openai_dialect`) and Chat-only behavior (`tool_calls` marks, server tool replay) are not ported
- `extra_body` is not part of the typed Messages request projection and is not included in the census

# Deliberate divergences

- A message point with both `index` and `role: system` targets the index. Python sends it to the system prompt
- The system breakpoint lands on the last block that is not an `x-anthropic-billing-header:` block. Python marks the final block, which provider normalization later strips

# references

- `litellm/integrations/anthropic_cache_control_hook.py`
- `litellm-rust/crates/llms-types/src/integrations/anthropic_cache_control_hook.rs`
- `tests/rust-python-harness/strategies/drift/drift.json` mapping `anthropic-cache-control-hook`
