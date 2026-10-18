use std::fmt;

use serde::{Deserialize, Deserializer, Serialize, Serializer};
use serde_json::{Map, Value};

use crate::formats::messages::CacheControl;

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Target {
    Index(i64),
    Role(Role),
    None,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Role(pub String);

impl Serialize for Role {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        serializer.serialize_str(&self.0)
    }
}

impl<'de> Deserialize<'de> for Role {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        String::deserialize(deserializer).map(Self)
    }
}

impl fmt::Display for Role {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.0)
    }
}

#[derive(
    Clone,
    Copy,
    Debug,
    PartialEq,
    Eq,
    strum::Display,
    strum::EnumString,
    strum::IntoStaticStr,
    serde_with::DeserializeFromStr,
    serde_with::SerializeDisplay,
)]
pub enum PromptCachingTtl {
    #[strum(serialize = "5m")]
    FiveMinutes,
    #[strum(serialize = "1h")]
    OneHour,
}

#[derive(Clone, Debug, PartialEq)]
pub struct CacheControlMessageInjectionPoint {
    pub target: Target,
    pub control: Option<CacheControl>,
}

#[derive(Clone, Debug, PartialEq)]
pub enum CacheControlInjectionPoint {
    Message(CacheControlMessageInjectionPoint),
    ToolConfig { control: Option<CacheControl> },
    Unknown(Value),
}

impl<'de> Deserialize<'de> for CacheControlInjectionPoint {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let value = Value::deserialize(deserializer)?;
        let Some(object) = value.as_object() else {
            return Ok(Self::Unknown(value));
        };
        let location = object.get("location").and_then(Value::as_str);
        match location {
            Some("message") => {
                let index = object.get("index");
                let target = match index {
                    Some(Value::Number(number)) => number
                        .as_i64()
                        .map(Target::Index)
                        .unwrap_or_else(|| role_target(object)),
                    Some(Value::String(index)) => index
                        .parse::<i64>()
                        .map(Target::Index)
                        .unwrap_or_else(|_| role_target(object)),
                    _ => role_target(object),
                };
                let control = object
                    .get("control")
                    .cloned()
                    .map(serde_json::from_value)
                    .transpose()
                    .map_err(serde::de::Error::custom)?;
                Ok(Self::Message(CacheControlMessageInjectionPoint {
                    target,
                    control,
                }))
            }
            Some("tool_config") => {
                let control = object
                    .get("control")
                    .cloned()
                    .map(serde_json::from_value)
                    .transpose()
                    .map_err(serde::de::Error::custom)?;
                Ok(Self::ToolConfig { control })
            }
            _ => Ok(Self::Unknown(value)),
        }
    }
}

impl Serialize for CacheControlInjectionPoint {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let value = match self {
            Self::Message(point) => {
                let mut object = Map::new();
                object.insert("location".into(), Value::String("message".into()));
                match &point.target {
                    Target::Index(index) => {
                        object.insert("index".into(), Value::Number((*index).into()));
                    }
                    Target::Role(role) => {
                        object.insert("role".into(), Value::String(role.0.clone()));
                    }
                    Target::None => {}
                }
                if let Some(control) = &point.control {
                    object.insert(
                        "control".into(),
                        serde_json::to_value(control).unwrap_or(Value::Null),
                    );
                }
                Value::Object(object)
            }
            Self::ToolConfig { control } => {
                let mut object = Map::new();
                object.insert("location".into(), Value::String("tool_config".into()));
                if let Some(control) = control {
                    object.insert(
                        "control".into(),
                        serde_json::to_value(control).unwrap_or(Value::Null),
                    );
                }
                Value::Object(object)
            }
            Self::Unknown(value) => value.clone(),
        };
        value.serialize(serializer)
    }
}

fn role_target(object: &Map<String, Value>) -> Target {
    object
        .get("role")
        .and_then(Value::as_str)
        .map(|role| Target::Role(Role(role.to_string())))
        .unwrap_or(Target::None)
}

#[cfg(test)]
mod tests {
    use rstest::rstest;
    use serde_json::{Value, json};

    use super::{
        CacheControlInjectionPoint, CacheControlMessageInjectionPoint, PromptCachingTtl, Role,
        Target,
    };

    #[rstest]
    #[case::negative_index_string(json!({"location": "message", "index": "-1"}), Target::Index(-1))]
    #[case::out_of_range_index(json!({"location": "message", "index": 99}), Target::Index(99))]
    #[case::role_only(json!({"location": "message", "role": "user"}), Target::Role(Role("user".into())))]
    #[case::invalid_index_falls_back_to_role(
        json!({"location": "message", "index": "latest", "role": "assistant"}),
        Target::Role(Role("assistant".into()))
    )]
    #[case::index_precedes_system_role(
        json!({"location": "message", "index": 0, "role": "system"}),
        Target::Index(0)
    )]
    #[case::missing_target(json!({"location": "message"}), Target::None)]
    fn message_target_deserialization(#[case] input: Value, #[case] expected: Target) {
        let point: CacheControlInjectionPoint = serde_json::from_value(input).unwrap();

        assert!(matches!(
            point,
            CacheControlInjectionPoint::Message(message) if message.target == expected
        ));
    }

    #[rstest]
    #[case::missing_control(
        json!({"location": "message", "index": 0}),
        CacheControlInjectionPoint::Message(CacheControlMessageInjectionPoint {
            target: Target::Index(0),
            control: None,
        })
    )]
    #[case::tool_config_without_control(
        json!({"location": "tool_config"}),
        CacheControlInjectionPoint::ToolConfig { control: None }
    )]
    fn known_points_preserve_optional_control(
        #[case] input: Value,
        #[case] expected: CacheControlInjectionPoint,
    ) {
        assert_eq!(
            serde_json::from_value::<CacheControlInjectionPoint>(input).unwrap(),
            expected
        );
    }

    #[rstest]
    fn unknown_location_round_trips_unchanged() {
        let input = json!({"location": "future", "index": 7, "custom": {"value": true}});
        let point: CacheControlInjectionPoint = serde_json::from_value(input.clone()).unwrap();

        assert!(matches!(&point, CacheControlInjectionPoint::Unknown(_)));
        assert_eq!(serde_json::to_value(point).unwrap(), input);
    }

    #[rstest]
    #[case::five_minutes("5m", Some(PromptCachingTtl::FiveMinutes))]
    #[case::one_hour("1h", Some(PromptCachingTtl::OneHour))]
    #[case::invalid("five_minutes", None)]
    #[case::variant_name("FiveMinutes", None)]
    fn ttl_accepts_only_public_values(
        #[case] input: &str,
        #[case] expected: Option<PromptCachingTtl>,
    ) {
        assert_eq!(input.parse::<PromptCachingTtl>().ok(), expected);
    }

    #[rstest]
    #[case::five_minutes(PromptCachingTtl::FiveMinutes, "5m")]
    #[case::one_hour(PromptCachingTtl::OneHour, "1h")]
    fn ttl_serializes_to_its_public_value(#[case] input: PromptCachingTtl, #[case] expected: &str) {
        assert_eq!(serde_json::to_value(input).unwrap(), json!(expected));
        assert_eq!(<&'static str>::from(input), expected);
    }

    #[rstest]
    #[case::number(json!(5))]
    #[case::null(Value::Null)]
    fn ttl_rejects_non_string_json(#[case] value: Value) {
        assert!(serde_json::from_value::<PromptCachingTtl>(value).is_err());
    }
}
