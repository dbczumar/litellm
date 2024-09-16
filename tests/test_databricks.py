import asyncio
import httpx
import json
import pytest
import sys
from typing import Any, Dict
from unittest.mock import Mock, patch

import litellm
from litellm.exceptions import BadRequestError
from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler, HTTPHandler

try:
    from databricks.sdk import WorkspaceClient
    databricks_sdk_installed = True
except ImportError:
    databricks_sdk_installed = False


def mock_httpx_response(
    status_code=200, 
    json_data=None, 
    text_data=None, 
    headers=None
) -> httpx.Response:
    # Create a mock response
    mock_response = Mock(spec=httpx.Response)

    # Set the status code
    mock_response.status_code = status_code

    # Mock the .json() method
    if json_data is not None:
        mock_response.json.return_value = json_data


def mock_chat_response() -> Dict[str, Any]:
    return {
        "id": "chatcmpl_3f78f09a-489c-4b8d-a587-f162c7497891",
        "object": "chat.completion",
        "created": 1726285449,
        "model": "dbrx-instruct-071224",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "Hello! I'm an AI assistant. I'm doing well. How can I help?",
                    "function_call": None,
                    "tool_calls": None,
                },
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": 230,
            "completion_tokens": 38,
            "total_tokens": 268
        },
        "system_fingerprint": None
    }


def mock_embedding_response() -> Dict[str, Any]:
    return {
      "object": "list",
      "model": "bge-large-en-v1.5",
      "data": [
        {
          "index": 0,
          "object": "embedding",
          "embedding": [
            0.06768798828125,
            -0.01291656494140625,
            -0.0501708984375,
            0.0245361328125,
            -0.030364990234375
          ]
        }
      ],
      "usage": {
        "prompt_tokens": 8,
        "total_tokens": 8,
        "completion_tokens": 0,
      }
    }


@pytest.mark.parametrize("set_base", [True, False])
def test_throws_if_only_one_of_api_base_or_api_key_set(monkeypatch, set_base):
    err_msg = "Databricks API base URL and API key must both be set, or both must be unset"

    if set_base:
        monkeypatch.setenv(
            "DATABRICKS_API_BASE",
            "https://my.workspace.cloud.databricks.com/serving-endpoints"
        )
    else:
        monkeypatch.setenv("DATABRICKS_API_KEY", "dapimykey")

    with pytest.raises(BadRequestError) as exc:
        litellm.completion(
            model="databricks/dbrx-instruct-071224",
            messages={"role": "user", "content": "How are you?"},
        )
        assert err_msg in str(exc)

    with pytest.raises(BadRequestError) as exc:
        litellm.embedding(
            model="databricks/bge-12312",
            input=["Hello", "World"],
        )
        assert err_msg in str(exc)


def test_throws_if_api_base_and_api_key_absent_and_databricks_sdk_not_installed(monkeypatch):
    # Simulate that the databricks SDK is not installed
    monkeypatch.setitem(sys.modules, 'databricks.sdk', None)
    with pytest.raises(BadRequestError) as exc:
        litellm.completion(
            model="databricks/dbrx-instruct-071224",
            messages={"role": "user", "content": "How are you?"},
        )
    assert "the databricks-sdk Python library must be installed." in str(exc)


def test_throws_for_async_request_when_api_base_and_api_key_absent():
    err_msg = "In order to make asynchronous calls"

    with pytest.raises(BadRequestError) as exc:
        asyncio.run(
            litellm.acompletion(
                model="databricks/dbrx-instruct-071224",
                messages={"role": "user", "content": "How are you?"},
            )
        )
    assert err_msg in str(exc)

    with pytest.raises(BadRequestError) as exc:
        asyncio.run(
            litellm.aembedding(
                model="databricks/bge-12312",
                input=["Hello", "World"],
            )
        )
    assert err_msg in str(exc)


def test_completions_sends_expected_request_with_sync_http_handler(monkeypatch):
    base_url = "https://my.workspace.cloud.databricks.com/serving-endpoints"
    api_key = "dapimykey"
    monkeypatch.setenv("DATABRICKS_API_BASE", base_url)
    monkeypatch.setenv("DATABRICKS_API_KEY", api_key)

    sync_handler = HTTPHandler()
    mock_response = Mock(spec=httpx.Response)
    mock_response.status_code = 200
    mock_response.json.return_value = mock_chat_response()

    expected_response_json = {
        **mock_chat_response(),
        **{
            "model": "databricks/dbrx-instruct-071224",
        }
    }

    messages = [{"role": "user", "content": "How are you?"}]

    with patch.object(HTTPHandler, "post", return_value=mock_response) as mock_post:
        response = litellm.completion(
            model="databricks/dbrx-instruct-071224",
            messages=messages,
            client=sync_handler,
            temperature=0.5,
        )
        assert response.to_dict() == expected_response_json

        mock_post.assert_called_once_with(
            f"{base_url}/chat/completions",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            data=json.dumps({
                "model": "dbrx-instruct-071224",
                "messages": messages,
                "temperature": 0.5,
                "stream": False,
            }),
        )


def test_completions_sends_expected_request_with_async_http_handler(monkeypatch):
    base_url = "https://my.workspace.cloud.databricks.com/serving-endpoints"
    api_key = "dapimykey"
    monkeypatch.setenv("DATABRICKS_API_BASE", base_url)
    monkeypatch.setenv("DATABRICKS_API_KEY", api_key)

    async_handler = AsyncHTTPHandler()
    mock_response = Mock(spec=httpx.Response)
    mock_response.status_code = 200
    mock_response.json.return_value = mock_chat_response()

    expected_response_json = {
        **mock_chat_response(),
        **{
            "model": "databricks/dbrx-instruct-071224",
        }
    }

    messages = [{"role": "user", "content": "How are you?"}]

    with patch.object(AsyncHTTPHandler, "post", return_value=mock_response) as mock_post:
        response = asyncio.run(
            litellm.acompletion(
                model="databricks/dbrx-instruct-071224",
                messages=messages,
                client=async_handler,
                temperature=0.5,
            )
        )
        assert response.to_dict() == expected_response_json

        mock_post.assert_called_once_with(
            f"{base_url}/chat/completions",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            data=json.dumps({
                "model": "dbrx-instruct-071224",
                "messages": messages,
                "temperature": 0.5,
                "stream": False,
            }),
        )


@pytest.mark.skipif(not databricks_sdk_installed, reason="Databricks SDK not installed")
@pytest.mark.parametrize("set_base_key", [True, False])
def test_completions_sends_expected_request_with_sync_databricks_client(monkeypatch, set_base_key):
    from databricks.sdk import WorkspaceClient
    from databricks.sdk.core import ApiClient

    base_url = "https://my.workspace.cloud.databricks.com/serving-endpoints"
    api_key = "dapimykey"
    if set_base_key:
        monkeypatch.setenv("DATABRICKS_API_BASE", base_url)
        monkeypatch.setenv("DATABRICKS_API_KEY", api_key)

    expected_response_json = {
        **mock_chat_response(),
        **{
            "model": "databricks/dbrx-instruct-071224",
        }
    }

    messages = [{"role": "user", "content": "How are you?"}]

    with patch("databricks.sdk.WorkspaceClient", wraps=WorkspaceClient) as mock_workspace_client, \
         patch.object(ApiClient, "do", return_value=mock_chat_response()) as mock_api_request:
        response = litellm.completion(
            model="databricks/dbrx-instruct-071224",
            messages=messages,
            temperature=0.5,
        )
        assert response.to_dict() == expected_response_json

        mock_workspace_client.assert_called_once_with(
            host=base_url if set_base_key else None,
            token=api_key if set_base_key else None,
        )
        mock_api_request.assert_called_once_with(
            method="POST",
            path=f"/serving-endpoints/dbrx-instruct-071224/invocations",
            body={
                "messages": messages,
                "temperature": 0.5,
                "stream": False,
            },
            headers=None,
        )


@pytest.mark.skipif(not databricks_sdk_installed, reason="Databricks SDK not installed")
@pytest.mark.parametrize("set_base_key", [True, False])
def test_embeddings_sends_expected_request_with_sync_databricks_client(monkeypatch, set_base_key):
    from databricks.sdk import WorkspaceClient
    from databricks.sdk.core import ApiClient

    base_url = "https://my.workspace.cloud.databricks.com/serving-endpoints"
    api_key = "dapimykey"
    if set_base_key:
        monkeypatch.setenv("DATABRICKS_API_BASE", base_url)
        monkeypatch.setenv("DATABRICKS_API_KEY", api_key)

    inputs = ["Hello", "World"]

    with patch("databricks.sdk.WorkspaceClient", wraps=WorkspaceClient) as mock_workspace_client, \
         patch.object(ApiClient, "do", return_value=mock_embedding_response()) as mock_api_request:
        response = litellm.embedding(
            model="databricks/bge-large-en-v1.5",
            input=inputs,
        )
        assert response.to_dict() == mock_embedding_response()

        mock_workspace_client.assert_called_once_with(
            host=base_url if set_base_key else None,
            token=api_key if set_base_key else None,
        )
        mock_api_request.assert_called_once_with(
            method="POST",
            path=f"/serving-endpoints/bge-large-en-v1.5/invocations",
            body={"input": inputs},
            headers=None,
        )




