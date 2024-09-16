import httpx
import json
import pytest
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


def test_foo(monkeypatch):
    base_url = "https://my.workspace.cloud.databricks.com/serving-endpoints"
    monkeypatch.setenv("DATABRICKS_API_BASE", base_url)
    monkeypatch.setenv("DATABRICKS_API_KEY", "dapimykey")

    sync_handler = HTTPHandler()
    mock_response_json = {
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
       
    mock_response = Mock(spec=httpx.Response)
    mock_response.status_code = 200
    mock_response.json.return_value = mock_response_json

    expected_response_json = {
        **mock_response_json,
        **{
            "model": "databricks/dbrx-instruct-071224",
        }
    }

    messages = [
        {
            "role": "user",
            "content": "How are you?",
        }
    ]

    with patch.object(HTTPHandler, "post", return_value=mock_response) as mock_post:
        response = litellm.completion(
            model="databricks/dbrx-instruct-071224",
            messages=messages,
            client=sync_handler,
        )
        assert expected_response_json == response.to_dict()

        mock_post.assert_called_once_with(
            f"{base_url}/chat/completions",
            headers={
                "Authorization": "Bearer dapimykey",
                "Content-Type": "application/json",
            },
            data=json.dumps({
                "model": "dbrx-instruct-071224",
                "messages": messages,
                "stream": False,
            }),
        )
