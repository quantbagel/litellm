use litellm_llms_types::{
    formats::messages::MessagesRequest,
    integrations::{
        CacheControlInjectionPoint, CacheControlMessageInjectionPoint, PromptCachingTtl, Role,
        Target,
    },
};
use serde_json::{Map, Value, json};

const MAX_CACHE_CONTROL_BLOCKS: usize = 4;

pub struct InjectionInputs<'a> {
    pub points: &'a [CacheControlInjectionPoint],
    pub enable_prompt_caching: bool,
    pub supports_anthropic_cache_control: bool,
    pub one_shot_subagent: bool,
    pub ttl: Option<PromptCachingTtl>,
}

pub struct Injected {
    pub request: MessagesRequest,
    pub added: usize,
    pub remaining_points: Vec<CacheControlInjectionPoint>,
}

pub fn maybe_inject_cache_control(
    request: MessagesRequest,
    inputs: &InjectionInputs<'_>,
) -> Injected {
    let configured = !inputs.points.is_empty();
    let defaults_allowed = inputs.enable_prompt_caching
        && inputs.supports_anthropic_cache_control
        && !inputs.one_shot_subagent;
    let points: Vec<CacheControlInjectionPoint> = if configured {
        inputs.points.to_vec()
    } else if defaults_allowed && !has_client_marks(&request) {
        default_points(inputs.ttl)
    } else {
        Vec::new()
    };
    if points.is_empty() {
        return Injected {
            request,
            added: 0,
            remaining_points: Vec::new(),
        };
    }

    let mut value = serde_json::to_value(request).unwrap_or_else(|_| json!({}));
    normalize_message_strings(&mut value);
    let existing = count_cache_marks(&value);
    let has_tool_config = points
        .iter()
        .any(|point| matches!(point, CacheControlInjectionPoint::ToolConfig { .. }));
    let message_cap = MAX_CACHE_CONTROL_BLOCKS
        .saturating_sub(existing)
        .saturating_sub(usize::from(has_tool_config));
    let mut added = 0;
    let message_points = points.iter().filter_map(message_point).collect::<Vec<_>>();
    for point in message_points
        .iter()
        .copied()
        .filter(|point| point.target == Target::Role(Role("system".into())))
        .chain(
            message_points
                .iter()
                .copied()
                .filter(|point| point.target != Target::Role(Role("system".into()))),
        )
    {
        if added >= message_cap {
            break;
        }
        added += apply_message_point(&mut value, point, message_cap - added);
    }
    let total = existing + added;
    let remaining_points = points
        .iter()
        .filter(|point| !matches!(point, CacheControlInjectionPoint::Message(_)))
        .filter(|point| {
            total < MAX_CACHE_CONTROL_BLOCKS
                || !matches!(point, CacheControlInjectionPoint::ToolConfig { .. })
        })
        .cloned()
        .collect::<Vec<_>>();
    Injected {
        request: serde_json::from_value(value)
            .expect("MessagesRequest serialization must round-trip"),
        added,
        remaining_points,
    }
}

fn default_points(ttl: Option<PromptCachingTtl>) -> Vec<CacheControlInjectionPoint> {
    let control = ttl.map(|ttl| litellm_llms_types::formats::messages::CacheControl {
        cache_type: Some("ephemeral".into()),
        ttl: Some(<&'static str>::from(ttl).into()),
        scope: None,
        extra: Map::new(),
    });
    vec![
        CacheControlInjectionPoint::Message(CacheControlMessageInjectionPoint {
            target: Target::Role(Role("system".into())),
            control: control.clone(),
        }),
        CacheControlInjectionPoint::Message(CacheControlMessageInjectionPoint {
            target: Target::Index(-1),
            control,
        }),
    ]
}

fn message_point(point: &CacheControlInjectionPoint) -> Option<&CacheControlMessageInjectionPoint> {
    match point {
        CacheControlInjectionPoint::Message(point) => Some(point),
        _ => None,
    }
}

fn apply_message_point(
    request: &mut Value,
    point: &CacheControlMessageInjectionPoint,
    slots: usize,
) -> usize {
    if slots == 0 {
        return 0;
    }
    if let Target::Role(role) = &point.target {
        if role.0 == "system" {
            return apply_mark_to_system(request, point.control.as_ref());
        }
    }
    let Some(messages) = request.get_mut("messages").and_then(Value::as_array_mut) else {
        return 0;
    };
    let length = messages.len() as i64;
    let matches: Vec<usize> = messages
        .iter()
        .enumerate()
        .filter_map(|(index, message)| {
            target_matches(&point.target, index as i64, length, message).then_some(index)
        })
        .collect();
    let mut added = 0;
    for index in matches {
        if added >= slots {
            break;
        }
        if mark_message(&mut messages[index], point.control.as_ref()) {
            added += 1;
        }
    }
    added
}

fn target_matches(target: &Target, index: i64, length: i64, message: &Value) -> bool {
    match target {
        Target::None => false,
        Target::Index(target) => {
            let resolved = if *target < 0 {
                length + target
            } else {
                *target
            };
            resolved == index && resolved >= 0 && resolved < length
        }
        Target::Role(role) => message
            .get("role")
            .and_then(Value::as_str)
            .is_some_and(|candidate| candidate == role.0),
    }
}

fn mark_message(
    message: &mut Value,
    control: Option<&litellm_llms_types::formats::messages::CacheControl>,
) -> bool {
    if has_cache_marks(message) {
        return false;
    }
    let Some(content) = message.get_mut("content") else {
        return false;
    };
    mark_content(content, control)
}

fn apply_mark_to_system(
    request: &mut Value,
    control: Option<&litellm_llms_types::formats::messages::CacheControl>,
) -> usize {
    let Some(system) = request.get_mut("system") else {
        return 0;
    };
    if has_cache_marks(system) {
        return 0;
    }
    usize::from(mark_content(system, control))
}

fn mark_content(
    content: &mut Value,
    control: Option<&litellm_llms_types::formats::messages::CacheControl>,
) -> bool {
    let block = cache_block(control);
    match content {
        Value::String(text) => {
            *content = Value::Array(vec![
                json!({"type": "text", "text": text, "cache_control": block}),
            ]);
            true
        }
        Value::Array(blocks) => blocks.last_mut().is_some_and(|last| {
            let Some(object) = last.as_object_mut() else {
                return false;
            };
            if object.contains_key("cache_control") {
                return false;
            }
            object.insert("cache_control".into(), block);
            true
        }),
        _ => false,
    }
}

fn cache_block(control: Option<&litellm_llms_types::formats::messages::CacheControl>) -> Value {
    let mut block = Map::new();
    block.insert(
        "type".into(),
        Value::String(
            control
                .and_then(|control| control.cache_type.as_deref())
                .unwrap_or("ephemeral")
                .to_string(),
        ),
    );
    if let Some(ttl) = control
        .and_then(|control| control.ttl.as_deref())
        .and_then(|ttl| ttl.parse::<PromptCachingTtl>().ok())
    {
        block.insert(
            "ttl".into(),
            Value::String(<&'static str>::from(ttl).into()),
        );
    }
    if let Some(scope) = control.and_then(|control| control.scope.as_ref()) {
        block.insert("scope".into(), Value::String(scope.clone()));
    }
    if let Some(control) = control {
        block.extend(control.extra.clone());
    }
    Value::Object(block)
}

fn normalize_message_strings(request: &mut Value) {
    let Some(messages) = request.get_mut("messages").and_then(Value::as_array_mut) else {
        return;
    };
    for message in messages {
        if let Some(object) = message.as_object_mut() {
            if let Some(Value::String(text)) = object.get("content").cloned() {
                object.insert(
                    "content".into(),
                    Value::Array(vec![json!({"type": "text", "text": text})]),
                );
            }
        }
    }
}

fn has_client_marks(request: &MessagesRequest) -> bool {
    serde_json::to_value(request)
        .map(|value| count_cache_marks(&value) > 0)
        .unwrap_or(false)
}

fn has_cache_marks(value: &Value) -> bool {
    value
        .as_object()
        .is_some_and(|object| object.contains_key("cache_control"))
        || value
            .as_array()
            .is_some_and(|items| items.iter().any(has_cache_marks))
}

fn count_cache_marks(value: &Value) -> usize {
    match value {
        Value::Object(object) => {
            usize::from(object.contains_key("cache_control"))
                + object.values().map(count_cache_marks).sum::<usize>()
        }
        Value::Array(items) => items.iter().map(count_cache_marks).sum(),
        _ => 0,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rstest::rstest;

    fn request(value: Value) -> MessagesRequest {
        serde_json::from_value(value).expect("valid Messages request")
    }

    fn configured_points(value: Value) -> Vec<CacheControlInjectionPoint> {
        serde_json::from_value(value).expect("valid cache-control injection points")
    }

    fn simple_request() -> MessagesRequest {
        request(json!({
            "model": "claude-test",
            "max_tokens": 32,
            "system": "system prompt",
            "messages": [
                {"role": "user", "content": "first"},
                {"role": "assistant", "content": "middle"},
                {"role": "user", "content": "last"}
            ]
        }))
    }

    #[rstest]
    #[case::disabled(false, true, false, 0)]
    #[case::unsupported(false, false, false, 0)]
    #[case::one_shot(false, true, true, 0)]
    #[case::enabled(true, true, false, 2)]
    fn defaults_require_all_gates(
        #[case] enabled: bool,
        #[case] supported: bool,
        #[case] one_shot: bool,
        #[case] expected_added: usize,
    ) {
        let points = [];
        let inputs = InjectionInputs {
            points: &points,
            enable_prompt_caching: enabled,
            supports_anthropic_cache_control: supported,
            one_shot_subagent: one_shot,
            ttl: None,
        };

        let injected = maybe_inject_cache_control(simple_request(), &inputs);

        assert_eq!(injected.added, expected_added);
        if expected_added > 0 {
            let value = serde_json::to_value(injected.request).unwrap();
            assert_eq!(
                value["system"][0]["cache_control"],
                json!({"type": "ephemeral"})
            );
            assert_eq!(
                value["messages"][2]["content"][0]["cache_control"],
                json!({"type": "ephemeral"})
            );
        }
    }

    #[rstest]
    #[case::message_block(json!({
        "model": "claude-test", "max_tokens": 32, "system": "system prompt",
        "messages": [{"role": "user", "content": [{"type": "text", "text": "marked", "cache_control": {"type": "ephemeral"}}]}]
    }))]
    #[case::top_level(json!({
        "model": "claude-test", "max_tokens": 32, "system": "system prompt",
        "messages": [{"role": "user", "content": "prompt"}],
        "cache_control": {"type": "ephemeral"}
    }))]
    #[case::function_tool(json!({
        "model": "claude-test", "max_tokens": 32, "system": "system prompt",
        "messages": [{"role": "user", "content": "prompt"}],
        "tools": [{"type": "function", "function": {"name": "lookup", "cache_control": {"type": "ephemeral"}}}]
    }))]
    fn defaults_stand_down_on_client_marks(#[case] request_value: Value) {
        let points = [];
        let inputs = InjectionInputs {
            points: &points,
            enable_prompt_caching: true,
            supports_anthropic_cache_control: true,
            one_shot_subagent: false,
            ttl: None,
        };

        let injected = maybe_inject_cache_control(request(request_value.clone()), &inputs);

        assert_eq!(injected.added, 0);
        assert!(injected.remaining_points.is_empty());
        assert_eq!(
            serde_json::to_value(injected.request).unwrap(),
            request_value
        );
    }

    #[rstest]
    #[case::five_minutes("5m", "5m")]
    #[case::one_hour("1h", "1h")]
    fn defaults_apply_valid_ttl(#[case] ttl: &str, #[case] expected: &str) {
        let points = [];
        let inputs = InjectionInputs {
            points: &points,
            enable_prompt_caching: true,
            supports_anthropic_cache_control: true,
            one_shot_subagent: false,
            ttl: ttl.parse::<PromptCachingTtl>().ok(),
        };

        let injected = maybe_inject_cache_control(simple_request(), &inputs);
        let value = serde_json::to_value(injected.request).unwrap();

        assert_eq!(
            value["system"][0]["cache_control"],
            json!({"type": "ephemeral", "ttl": expected})
        );
    }

    #[rstest]
    fn configured_points_omit_invalid_ttl_and_default_empty_controls() {
        let points = configured_points(json!([
            {"location": "message", "index": 0, "control": {"ttl": "invalid"}},
            {"location": "message", "index": 1, "control": {}}
        ]));
        let inputs = InjectionInputs {
            points: &points,
            enable_prompt_caching: false,
            supports_anthropic_cache_control: false,
            one_shot_subagent: false,
            ttl: None,
        };
        let request = request(json!({
            "model": "claude-test",
            "max_tokens": 32,
            "messages": [
                {"role": "user", "content": "first"},
                {"role": "user", "content": "second"}
            ]
        }));

        let injected = maybe_inject_cache_control(request, &inputs);
        let value = serde_json::to_value(injected.request).unwrap();

        assert_eq!(injected.added, 2);
        assert_eq!(
            value["messages"][0]["content"][0]["cache_control"],
            json!({"type": "ephemeral"})
        );
        assert_eq!(
            value["messages"][1]["content"][0]["cache_control"],
            json!({"type": "ephemeral"})
        );
    }

    #[rstest]
    fn configured_points_apply_with_client_marks_and_index_wins_over_system_role() {
        let points = configured_points(json!([
            {"location": "message", "index": 0, "role": "system"}
        ]));
        let inputs = InjectionInputs {
            points: &points,
            enable_prompt_caching: false,
            supports_anthropic_cache_control: false,
            one_shot_subagent: true,
            ttl: None,
        };
        let request = request(json!({
            "model": "claude-test",
            "max_tokens": 32,
            "system": "system prompt",
            "messages": [{"role": "user", "content": "prompt"}],
            "cache_control": {"type": "ephemeral"}
        }));

        let injected = maybe_inject_cache_control(request, &inputs);
        let value = serde_json::to_value(injected.request).unwrap();

        assert_eq!(injected.added, 1);
        assert_eq!(value["system"], "system prompt");
        assert_eq!(value["cache_control"], json!({"type": "ephemeral"}));
        assert_eq!(
            value["messages"][0]["content"][0]["cache_control"],
            json!({"type": "ephemeral"})
        );
    }

    #[rstest]
    fn system_point_is_applied_before_message_points_and_tool_config_reserves_a_slot() {
        let points = configured_points(json!([
            {"location": "message", "role": "user"},
            {"location": "message", "role": "system"},
            {"location": "tool_config"}
        ]));
        let inputs = InjectionInputs {
            points: &points,
            enable_prompt_caching: false,
            supports_anthropic_cache_control: false,
            one_shot_subagent: false,
            ttl: None,
        };
        let request = request(json!({
            "model": "claude-test",
            "max_tokens": 32,
            "system": "system prompt",
            "messages": [
                {"role": "user", "content": "first"},
                {"role": "user", "content": "second"},
                {"role": "user", "content": "third"},
                {"role": "user", "content": "fourth"}
            ]
        }));

        let injected = maybe_inject_cache_control(request, &inputs);
        let value = serde_json::to_value(injected.request).unwrap();

        assert_eq!(injected.added, 3);
        assert!(value["system"][0].get("cache_control").is_some());
        assert!(
            value["messages"][0]["content"][0]
                .get("cache_control")
                .is_some()
        );
        assert!(
            value["messages"][1]["content"][0]
                .get("cache_control")
                .is_some()
        );
        assert!(
            value["messages"][2]["content"][0]
                .get("cache_control")
                .is_none()
        );
        assert_eq!(injected.remaining_points.len(), 1);
        assert!(matches!(
            &injected.remaining_points[0],
            CacheControlInjectionPoint::ToolConfig { .. }
        ));
    }

    #[rstest]
    fn configured_message_point_marks_only_the_final_content_block() {
        let points = configured_points(json!([{"location": "message", "index": 0}]));
        let inputs = InjectionInputs {
            points: &points,
            enable_prompt_caching: false,
            supports_anthropic_cache_control: false,
            one_shot_subagent: false,
            ttl: None,
        };
        let request = request(json!({
            "model": "claude-test",
            "max_tokens": 32,
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "text", "text": "first"},
                    {"type": "text", "text": "last"}
                ]
            }]
        }));

        let injected = maybe_inject_cache_control(request, &inputs);
        let value = serde_json::to_value(injected.request).unwrap();

        assert_eq!(injected.added, 1);
        assert!(
            value["messages"][0]["content"][0]
                .get("cache_control")
                .is_none()
        );
        assert_eq!(
            value["messages"][0]["content"][1]["cache_control"],
            json!({"type": "ephemeral"})
        );
    }

    #[rstest]
    fn gateway_injection_observes_the_four_block_cap() {
        let points = configured_points(json!([{"location": "message", "role": "user"}]));
        let inputs = InjectionInputs {
            points: &points,
            enable_prompt_caching: false,
            supports_anthropic_cache_control: false,
            one_shot_subagent: false,
            ttl: None,
        };
        let request = request(json!({
            "model": "claude-test",
            "max_tokens": 32,
            "messages": [
                {"role": "user", "content": "first"},
                {"role": "user", "content": "second"},
                {"role": "user", "content": "third"},
                {"role": "user", "content": "fourth"},
                {"role": "user", "content": "fifth"}
            ]
        }));

        let injected = maybe_inject_cache_control(request, &inputs);
        let value = serde_json::to_value(injected.request).unwrap();

        assert_eq!(injected.added, 4);
        assert_eq!(count_cache_marks(&value), 4);
        assert!(
            value["messages"][3]["content"][0]
                .get("cache_control")
                .is_some()
        );
        assert!(
            value["messages"][4]["content"][0]
                .get("cache_control")
                .is_none()
        );
    }

    #[rstest]
    fn all_message_strings_are_normalized_when_points_apply() {
        let points = configured_points(json!([
            {"location": "message", "index": 99}
        ]));
        let inputs = InjectionInputs {
            points: &points,
            enable_prompt_caching: false,
            supports_anthropic_cache_control: false,
            one_shot_subagent: false,
            ttl: None,
        };

        let injected = maybe_inject_cache_control(simple_request(), &inputs);
        let value = serde_json::to_value(injected.request).unwrap();

        assert_eq!(injected.added, 0);
        assert_eq!(
            value["messages"][0]["content"],
            json!([{"type": "text", "text": "first"}])
        );
        assert_eq!(
            value["messages"][1]["content"],
            json!([{"type": "text", "text": "middle"}])
        );
    }

    #[rstest]
    fn unknown_locations_are_preserved_without_becoming_provider_parameters() {
        let points = configured_points(json!([
            {"location": "future_location", "target": "future", "control": {"x": 1}},
            {"location": "tool_config"}
        ]));
        let inputs = InjectionInputs {
            points: &points,
            enable_prompt_caching: false,
            supports_anthropic_cache_control: false,
            one_shot_subagent: false,
            ttl: None,
        };

        let injected = maybe_inject_cache_control(simple_request(), &inputs);
        let value = serde_json::to_value(injected.request).unwrap();

        assert_eq!(injected.added, 0);
        assert_eq!(injected.remaining_points.len(), 2);
        assert_eq!(
            serde_json::to_value(&injected.remaining_points[0]).unwrap(),
            json!({"location": "future_location", "target": "future", "control": {"x": 1}})
        );
        assert!(value.get("cache_control_injection_points").is_none());
    }

    #[rstest]
    fn tool_config_points_are_dropped_when_existing_marks_use_the_cap() {
        let points = configured_points(json!([
            {"location": "message", "index": 99},
            {"location": "tool_config"},
            {"location": "future_location", "value": 1}
        ]));
        let inputs = InjectionInputs {
            points: &points,
            enable_prompt_caching: false,
            supports_anthropic_cache_control: false,
            one_shot_subagent: false,
            ttl: None,
        };
        let request = request(json!({
            "model": "claude-test", "max_tokens": 32,
            "messages": [{
                "role": "user",
                "content": [{"type": "text", "text": "a", "cache_control": {"type": "ephemeral"}}]
            }],
            "system": [
                {"type": "text", "text": "a", "cache_control": {"type": "ephemeral"}},
                {"type": "text", "text": "b", "cache_control": {"type": "ephemeral"}},
                {"type": "text", "text": "c", "cache_control": {"type": "ephemeral"}}
            ]
        }));

        let injected = maybe_inject_cache_control(request, &inputs);

        assert_eq!(injected.added, 0);
        assert_eq!(injected.remaining_points.len(), 1);
        assert!(matches!(
            &injected.remaining_points[0],
            CacheControlInjectionPoint::Unknown(_)
        ));
    }
}
