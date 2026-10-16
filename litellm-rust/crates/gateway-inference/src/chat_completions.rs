use litellm_gateway_auth::AuthenticatedRequest;
use std::sync::Arc;

use axum::{
    Json,
    body::Bytes,
    extract::{Path, State},
    http::header::CONTENT_TYPE,
    response::{IntoResponse, Response},
};
use litellm_host_http::Sse;
use litellm_inference_chat::{
    ChatCompletionsRoute,
    openai::{OpenAiChatCall, OpenAiChatCompletions},
    types::ChatCompletionsCall,
};
use serde_json::{Map, Value};

use crate::{Deployment, Error, Gateway, JsonObject, request};

pub(crate) async fn create(
    State(gateway): State<Arc<Gateway>>,
    identity: AuthenticatedRequest,
    JsonObject(body): JsonObject,
) -> Result<impl IntoResponse, Error> {
    handle(&gateway, &identity, body).await
}

pub(crate) async fn create_from_model_path(
    State(gateway): State<Arc<Gateway>>,
    identity: AuthenticatedRequest,
    Path(model): Path<String>,
    JsonObject(body): JsonObject,
) -> Result<impl IntoResponse, Error> {
    let body = match body.get("model") {
        None | Some(Value::Null) => body
            .into_iter()
            .chain([("model".into(), Value::String(model))])
            .collect(),
        Some(_) => body,
    };
    handle(&gateway, &identity, body).await
}

async fn handle(
    gateway: &Gateway,
    identity: &AuthenticatedRequest,
    body: Map<String, Value>,
) -> Result<Response, Error> {
    let group = request::resolve_group(gateway, &body)?;
    let name = request::model_name(&body)?.to_owned();
    let (body, cache_options) = crate::caching::prepare(identity, body)?;
    let route = gateway.chat_completions.clone();
    let route = match &gateway.cache {
        Some(cache) => route.with_cache(litellm_cache_response::ScopedCache::new(
            cache.clone(),
            cache_options.scope.clone(),
        )),
        None => route,
    };

    let cache_headers = crate::caching::CacheHeaders::default();
    let mut body = Some(body);
    let routed = gateway
        .models
        .route_where(
            group,
            request::usable(identity),
            |attempt| {
                // Only the last attempt may move the body; earlier ones keep it for a retry.
                let body = if attempt.last {
                    body.take()
                } else {
                    body.clone()
                };
                let route = route.clone();
                let cache_headers = cache_headers.clone();
                let name = name.as_str();
                async move {
                    let deployment = attempt.deployment;
                    crate::observe::record(name, deployment);
                    request::authorize(identity, name, deployment).await?;
                    let body = body.unwrap_or_default();
                    if ChatCompletionsRoute::serves_openai(
                        &deployment.model,
                        deployment.custom_llm_provider.as_deref(),
                    ) {
                        return openai(gateway, deployment, body).await;
                    }
                    let (messages, optional_params) = split(body);
                    let call = ChatCompletionsCall {
                        model: deployment.model.clone(),
                        messages,
                        optional_params,
                        api_key: deployment.api_key.clone(),
                        api_base: deployment.api_base.clone(),
                        custom_llm_provider: deployment.custom_llm_provider.clone(),
                        extra_headers: None,
                        timeout: deployment.timeout,
                    };
                    Ok(litellm_host_http::serve_unary(
                        route.machine(call, cache_options.policy),
                        (),
                        cache_headers,
                        litellm_host_http::Unary::new(Json),
                        None,
                    )
                    .await?)
                }
            },
            Error::failure,
        )
        .await
        .ok_or_else(|| Error::UnknownModel(name.clone()))??;
    Ok(cache_headers.apply(request::hold(routed)))
}

/// The `messages` array and the remaining optional params of a chat request.
fn split(mut body: Map<String, Value>) -> (Value, Map<String, Value>) {
    let messages = body.remove("messages").unwrap_or_default();
    body.remove("model");
    (messages, body)
}

async fn openai(
    gateway: &Gateway,
    deployment: &Deployment,
    body: Map<String, Value>,
) -> Result<Response, Error> {
    let machine = gateway.chat_completions.clone().openai_machine(
        OpenAiChatCall {
            model: deployment.model.clone(),
            body,
            api_key: deployment.api_key.clone(),
            api_base: deployment.api_base.clone(),
            custom_llm_provider: deployment.custom_llm_provider.clone(),
            extra_headers: None,
            timeout: deployment.timeout,
        },
        None,
    );
    let encoder = Sse::<OpenAiChatCompletions, _, _>::new(
        |body: Bytes| ([(CONTENT_TYPE, "application/json")], body),
        |error| Bytes::from(Error::from(error).openai_sse_frame()),
    );
    Ok(litellm_host_http::serve(machine, (), (), encoder, None).await?)
}
