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
# from litellm.llms.databricks.exceptions import DatabricksError

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
        assert expected_response_json == response.to_dict()

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
        assert expected_response_json == response.to_dict()

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
