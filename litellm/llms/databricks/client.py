import json
from abc import abstractmethod
from typing import Optional, List, Dict, Any, Literal, Union

import httpx  # type: ignore

from litellm.llms.databricks.exceptions import DatabricksError
from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler, HTTPHandler
from litellm.utils import ModelResponse, Choices


class DatabricksModelServingClient:
    def __init__(self, api_base: Optional[str], api_key: Optional[str], client: HTTPHandler):
        self.api_base = api_base
        self.api_key = api_key
        self.client = client or DatabricksModelServingWorkspaceClient


class DatabricksModelServingClientWrapper:

    @abstractmethod
    def completions(
        self,
        endpoint_name: str,
        messages: Optional[List[Dict[str, str]]] = None,
        # litellm params?
        optional_params: Optional[Dict[str, Any]] = None,
    ) -> ModelResponse:
        pass


def get_databricks_model_serving_client_wrapper(
    api_base: Optional[str],
    api_key: Optional[str],
    http_handler: Optional[HTTPHandler],
    custom_endpoint: Optional[bool] = False,
    headers: Optional[Dict[str, str]] = None,
    timeout: Optional[Union[float, httpx.Timeout]] = None,
) -> DatabricksModelServingClientWrapper:
    if (api_base, api_key).count(None) == 1:
        raise DatabricksError(status_code=400, message="api_base and api_key must be provided together, or both must be empty.")

    if (custom_endpoint is not None or http_handler is not None) and (api_base, api_key).count(None) > 0:
        raise DatabricksError(status_code=400, message="If http_handler or custom_endpoint is provided, api_base and api_key must be provided.")

    if (api_base, api_key).count(None) == 2:
        return DatabricksModelServingWorkspaceClientWrapper()
    else:
        return DatabricksModelServingHTTPHandlerWrapper(
            api_base=api_base,
            api_key=api_key,
            http_handler=http_handler or HTTPHandler(timeout=timeout),
            custom_endpoint=custom_endpoint,
            headers=headers,
        )


class DatabricksModelServingWorkspaceClientWrapper(DatabricksModelServingClientWrapper):

    def __init__(self):
        from databricks.sdk import WorkspaceClient

        self.client = WorkspaceClient()

    def completions(
        self,
        endpoint_name: str,
        messages: Optional[List[Dict[str, str]]] = None,
        optional_params: Optional[Dict[str, Any]] = None,
        # litellm params?
    ) -> ModelResponse:
        from databricks.sdk.service.serving import ChatMessage, ChatMessageRole, QueryEndpointResponse

        endpoint_response: QueryEndpointResponse = self.client.serving_endpoints.query(
          name="databricks-meta-llama-3-1-70b-instruct",
          messages=[
            self. _translate_chat_message_for_query(message) for message in messages
          ],
          **optional_params
        )
        return self._translate_endpoint_query_response_to_model_response(endpoint_response)

    def _translate_chat_message_for_query(self, message: Dict[str, str]) -> 'ChatMessage':
        from databricks.sdk.service.serving import ChatMessage, ChatMessageRole

        return ChatMessage(
            content=message["content"],
            role=ChatMessageRole(message["role"])
        )

    def _translate_endpoint_query_response_to_model_response(self, query_response: 'QueryEndpointResponse') -> ModelResponse:
        from databricks.sdk.service.serving import V1ResponseChoiceElement
        # Convert V1ResponseChoiceElement to Choices
        def convert_choice(v1_choice: V1ResponseChoiceElement) -> Choices:
            return Choices(
                finish_reason=v1_choice.finish_reason,
                index=v1_choice.index,  # Since index is always present
                message=v1_choice.message.as_dict(),  # message is always present, so we directly convert it to a dict
                logprobs=v1_choice.logprobs,  # logprobs is always present
                # Assuming 'enhancements' is not part of V1ResponseChoiceElement
            )

        # Direct mapping of fields assuming all fields in QueryEndpointResponse are present
        model_response = ModelResponse(
            id=query_response.id,
            choices=[
                convert_choice(choice) for choice in query_response.choices
            ],
            created=query_response.created,
            model=query_response.model,
            object=query_response.object.object_type,  # Assuming QueryEndpointResponseObject has object_type
            system_fingerprint=query_response.served_model_name,  # Mapping served_model_name to system_fingerprint
            _hidden_params={},  # No direct mapping, so set to empty dictionary
            _response_headers=None  # No direct mapping, set to None
        )

        return model_response


class DatabricksModelServingHTTPHandlerWrapper(DatabricksModelServingClientWrapper):

    def __init__(self, api_base: str, api_key: str, http_handler: HTTPHandler, custom_endpoint: Optional[bool], headers: Optional[Dict[str, str]] = None):
        self.api_base = api_base
        self.api_key = api_key
        self.http_handler = http_handler
        self.headers = headers
        self.custom_endpoint = custom_endpoint

    def completions(
        self,
        endpoint_name: str,
        messages: Optional[List[Dict[str, str]]] = None,
        optional_params: Optional[Dict[str, Any]] = None,
        # litellm params?
    ) -> ModelResponse:
        try:
            data = {
                "model": endpoint_name,
                "messages": messages,
                **optional_params,
            }
            response = self.http_handler.post(
                self._get_api_base(endpoint_type="chat_completions"),
                headers={
                    **self.headers,
                    **{
                        "Authorization": "Bearer {}".format(self.api_key)
                    }
                },
                data=json.dumps(data)
            )
            response.raise_for_status()
            return ModelResponse(**response.json())
        except httpx.HTTPStatusError as e:
            raise DatabricksError(
                status_code=e.response.status_code, message=response.text
            )
        except httpx.TimeoutException as e:
            raise DatabricksError(
                status_code=408, message="Timeout error occurred."
            )
        except Exception as e:
            raise DatabricksError(status_code=500, message=str(e))

    def _get_api_base(self, endpoint_type: Literal["chat_completions", "embeddings"]) -> str:
        if self.custom_endpoint is True:
            return self.api_base
        elif endpoint_type == "chat_completions":
            return "{}/chat/completions".format(self.api_base)
        elif endpoint_type == "embeddings":
            return "{}/embeddings".format(self.api_base)
        else:
            raise DatabricksError(status_code=500, message="Invalid endpoint type: {}".format(endpoint_type))


