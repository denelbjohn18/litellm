use std::{
    collections::HashMap,
    sync::Arc,
    time::{Duration, SystemTime},
};

use axum::{
    Json, Router,
    body::{Body, to_bytes},
    http::Request,
};
use litellm_auth_types::SecretValue;
use litellm_config::Config;
use litellm_gateway_auth::keys::{
    AllowedRoutes, KeyHash, KeyLookup, KeyModels, KeyStatus, LookupFuture, ModelAccess, Owner,
    Refusal, Restriction,
};
use rstest::{fixture, rstest};
use serde_json::{Value, json};
use tokio::net::TcpListener;
use tower::ServiceExt;

struct Keys(HashMap<KeyHash, KeyStatus>);

impl KeyLookup for Keys {
    fn lookup<'a>(&'a self, hash: &'a KeyHash) -> LookupFuture<'a> {
        Box::pin(async move { Ok(self.0.get(hash).cloned()) })
    }
}

fn active(models: &[&str], expires_at: Option<SystemTime>) -> KeyStatus {
    KeyStatus::active(
        expires_at,
        ModelAccess::new(
            models
                .iter()
                .map(|model| (*model).to_owned())
                .collect::<Vec<_>>(),
        ),
    )
}

/// A key of `team_id` whose key and team grant every model.
fn on_team(team_id: &str) -> KeyStatus {
    KeyStatus::Active {
        expires_at: None,
        models: KeyModels {
            key: ModelAccess::default(),
            team: Some(ModelAccess::default()),
        },
        allowed_routes: AllowedRoutes::default(),
        team_id: Some(team_id.to_owned()),
    }
}

fn keys() -> Arc<dyn KeyLookup> {
    let hash = |token: &str| KeyHash::from_token(&SecretValue::new(token));
    Arc::new(Keys(HashMap::from([
        (hash("sk-all"), active(&[], None)),
        (hash("sk-scoped"), active(&["claude-a"], None)),
        (hash("sk-other"), active(&["claude-b"], None)),
        (
            hash("sk-expired"),
            active(&[], Some(SystemTime::now() - Duration::from_secs(60))),
        ),
        (hash("sk-blocked"), KeyStatus::Blocked),
        (
            hash("sk-team-narrowed"),
            KeyStatus::Active {
                expires_at: None,
                models: KeyModels {
                    key: ModelAccess::default(),
                    team: Some(ModelAccess::new(vec!["claude-b".to_owned()])),
                },
                allowed_routes: AllowedRoutes::default(),
                team_id: Some("team-1".into()),
            },
        ),
        (
            hash("sk-chat-only"),
            KeyStatus::Active {
                expires_at: None,
                models: KeyModels::default(),
                allowed_routes: AllowedRoutes::new(vec!["openai_routes".to_owned()]),
                team_id: None,
            },
        ),
        (
            hash("sk-messages-only"),
            KeyStatus::Active {
                expires_at: None,
                models: KeyModels::default(),
                allowed_routes: AllowedRoutes::new(vec!["llm_api_routes".to_owned()]),
                team_id: None,
            },
        ),
        // The key store passes a key whose stored spend is under its budget.
        (hash("sk-under-budget"), active(&[], None)),
        (hash("sk-team-1"), on_team("team-1")),
        (hash("sk-team-2"), on_team("team-2")),
        (
            hash("sk-over-budget"),
            KeyStatus::Refused {
                expires_at: None,
                refusal: Refusal::BudgetExceeded {
                    owner: "Key=key".into(),
                    spend: "2.0".into(),
                    max_budget: "1.0".into(),
                },
            },
        ),
        (
            hash("sk-rate-limited"),
            KeyStatus::Refused {
                expires_at: None,
                refusal: Refusal::Unsupported {
                    restriction: Restriction::RateLimits,
                    owner: Owner::Key,
                },
            },
        ),
        (
            hash("sk-team-blocked"),
            KeyStatus::Refused {
                expires_at: None,
                refusal: Refusal::TeamBlocked {
                    team_id: "team-1".into(),
                },
            },
        ),
    ])))
}

/// An Anthropic-shaped upstream that answers every request with one short message.
async fn upstream() -> String {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let address = listener.local_addr().unwrap();
    let app = Router::new().fallback(|| async {
        Json(json!({
            "id": "msg_1",
            "type": "message",
            "role": "assistant",
            "model": "claude-x",
            "content": [{"type": "text", "text": "hi"}],
            "stop_reason": "end_turn",
            "stop_sequence": null,
            "usage": {"input_tokens": 1, "output_tokens": 1},
        }))
    });
    tokio::spawn(async move { axum::serve(listener, app).await });
    format!("http://{address}")
}

#[fixture]
async fn app() -> Router {
    let upstream = upstream().await;
    let config = Config::from_yaml(&format!(
        "model_list:
  - model_name: claude-a
    litellm_params:
      model: anthropic/claude-x
      api_base: {upstream}
      api_key: sk-upstream
  - model_name: claude-owned
    litellm_params:
      model: anthropic/claude-x
      api_base: {upstream}
      api_key: sk-upstream
    model_info:
      team_id: team-2
  - model_name: claude-mixed
    litellm_params:
      model: anthropic/claude-x
      api_base: {upstream}
      api_key: sk-upstream
  - model_name: claude-mixed
    litellm_params:
      model: anthropic/claude-x
      api_base: http://127.0.0.1:9
      api_key: sk-upstream
    model_info:
      team_id: team-2
router_settings:
  num_retries: 0
general_settings:
  master_key: sk-master
",
    ))
    .unwrap();
    let inference = litellm_gateway::build_inference(&config).unwrap();
    litellm_gateway::router_with_keys(inference, &config, None, None, Some(keys()))
}

async fn send(app: Router, header: &str, value: &str, model: &str) -> (u16, Value) {
    let response = app
        .oneshot(
            Request::post("/v1/messages")
                .header("content-type", "application/json")
                .header(header, value)
                .body(Body::from(
                    json!({
                        "model": model,
                        "max_tokens": 8,
                        "messages": [{"role": "user", "content": "hi"}],
                    })
                    .to_string(),
                ))
                .unwrap(),
        )
        .await
        .unwrap();
    let status = response.status().as_u16();
    let body = to_bytes(response.into_body(), 1 << 20).await.unwrap();
    (status, serde_json::from_slice(&body).unwrap())
}

/// `claude-owned` has one deployment owned by team-2. `claude-mixed` has a shared deployment
/// and one owned by team-2 whose upstream refuses connections, so any request routed to it
/// fails (retries are off).
#[rstest]
#[case::own_team("sk-team-2", "claude-owned", 200)]
#[case::teamless_key("sk-all", "claude-owned", 400)]
#[case::other_team("sk-team-1", "claude-owned", 400)]
#[case::master_key("sk-master", "claude-owned", 200)]
#[case::teamless_key_mixed("sk-all", "claude-mixed", 200)]
#[case::other_team_mixed("sk-team-1", "claude-mixed", 200)]
#[tokio::test]
async fn team_scoped_deployments_serve_only_their_team(
    #[future] app: Router,
    #[case] key: &str,
    #[case] model: &str,
    #[case] status: u16,
) {
    let app = app.await;
    for _ in 0..20 {
        let (actual, body) = send(app.clone(), "x-api-key", key, model).await;
        assert_eq!(actual, status, "{body}");
    }
}

#[rstest]
#[case::x_api_key("x-api-key", "sk-all", 200, None)]
#[case::azure_api_key("api-key", "sk-all", 200, None)]
#[case::bearer("authorization", "Bearer sk-all", 200, None)]
#[case::master_key_in_x_api_key("x-api-key", "sk-master", 200, None)]
#[case::listed_model("x-api-key", "sk-scoped", 200, None)]
#[case::unlisted_model("x-api-key", "sk-other", 403, Some("key_model_access_denied"))]
#[case::expired("x-api-key", "sk-expired", 401, Some("expired_key"))]
#[case::blocked("x-api-key", "sk-blocked", 401, Some("auth_error"))]
#[case::unknown("x-api-key", "sk-unknown", 401, Some("token_not_found_in_db"))]
#[case::missing("x-ignored", "sk-all", 401, Some("auth_error"))]
#[case::team_narrowed("x-api-key", "sk-team-narrowed", 403, Some("team_model_access_denied"))]
#[case::route_not_allowed("x-api-key", "sk-chat-only", 403, Some("auth_error"))]
#[case::route_preset("x-api-key", "sk-messages-only", 200, None)]
#[case::under_budget("x-api-key", "sk-under-budget", 200, None)]
#[case::over_budget("x-api-key", "sk-over-budget", 422, Some("budget_exceeded"))]
#[case::unsupported_restriction("x-api-key", "sk-rate-limited", 401, Some("auth_error"))]
#[case::team_blocked("x-api-key", "sk-team-blocked", 401, Some("auth_error"))]
#[tokio::test]
async fn messages_accept_virtual_keys_from_every_key_header(
    #[future] app: Router,
    #[case] header: &str,
    #[case] value: &str,
    #[case] status: u16,
    #[case] error_type: Option<&str>,
) {
    let (actual, body) = send(app.await, header, value, "claude-a").await;

    assert_eq!(actual, status);
    match error_type {
        None => assert_eq!(body["content"][0]["text"], "hi"),
        Some(error_type) => {
            assert_eq!(body["error"]["type"], error_type);
            assert_eq!(body["error"]["code"], status.to_string());
            assert!(!body.to_string().contains(value));
        }
    }
}
