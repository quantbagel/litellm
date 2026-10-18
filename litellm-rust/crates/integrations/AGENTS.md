# rules

- Mirrors `litellm/integrations/` and owns deterministic integration behavior, independent of provider I/O and host callbacks
- Shared request contracts belong in `litellm-llms-types`; runtime configuration and provider capability checks are inputs from callers
- The Messages port preserves unknown injection-point locations and returns non-message points without placing them
- `extra_body` is not part of the typed Messages request projection and is not included in the cache-mark census
- Message strings normalize to text block lists whenever injection points apply, even when no point targets that message
- OpenAI prompt-cache dialects, Bedrock Converse `tool_config` placement, and Chat-only behavior are known partial ports
- Use inline named `rstest` cases for behavior

# references

- `litellm/integrations/anthropic_cache_control_hook.py`
- `litellm-rust/crates/llms-types/src/integrations/anthropic_cache_control_hook.rs`
