import json
from abc import abstractmethod
from typing import Optional, List, Dict, Any, Literal, Union

import httpx  # type: ignore

from litellm.llms.databricks.exceptions import DatabricksError
from litellm.llms.databricks.streaming_utils import ModelResponseIterator
from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler, HTTPHandler
from litellm.utils import ModelResponse, Choices, CustomStreamWrapper
from litellm.types.utils import CustomStreamingDecoder


class DatabricksModelServingClientWrapper:

    def completion(
        self,
        endpoint_name: str,
        messages: List[Dict[str, str]],
        # litellm params?
        optional_params: Dict[str, Any],
        stream: bool,
    ) -> ModelResponse:
        if stream:
            return self._streaming_completion(endpoint_name, messages, optional_params)
        else:
            return self._completion(endpoint_name, messages, optional_params)

    @abstractmethod
    def _completion(
        self,
        endpoint_name: str,
        messages: List[Dict[str, str]],
        optional_params: Dict[str, Any],
    ) -> ModelResponse:
        pass

    @abstractmethod
    def _streaming_completion(
        self,
        endpoint_name: str,
        messages: List[Dict[str, str]],
        optional_params: Dict[str, Any],
    ) -> CustomStreamWrapper:
        pass


def get_databricks_model_serving_client_wrapper(
    synchronous: bool,
    custom_llm_provider: str,
    logging_obj,
    api_base: Optional[str],
    api_key: Optional[str],
    http_handler: Optional[Union[HTTPHandler, AsyncHTTPHandler]],
    timeout: Optional[Union[float, httpx.Timeout]],
    custom_endpoint: Optional[bool],
    headers: Optional[Dict[str, str]],
    streaming_decoder: Optional[CustomStreamingDecoder],
) -> DatabricksModelServingClientWrapper:
    if (api_base, api_key).count(None) == 1:
        raise DatabricksError(status_code=400, message="Databricks API base and API key must both be set, or both must be unset.")

    if custom_endpoint is not None and (api_base, api_key).count(None) > 0:
        raise DatabricksError(status_code=400, message="If a custom endpoint is specified, Databricks API base and API key must both be set.")

    if (http_handler is not None) and (api_base, api_key).count(None) > 0:
        raise DatabricksError(status_code=500, message="If http_handler is provided, api_base and api_key must be provided.")

    if (api_base, api_key).count(None) == 2:
        if not synchronous:
            raise DatabricksError(status_code=500, message="In order to make asynchronous calls, Databricks API base and API key must both be set.")

        try:
            import databricks.sdk

            return DatabricksModelServingWorkspaceClientWrapper()
        except ImportError:
            raise DatabricksError(status_code=400, message="If Databricks API base and API key are not provided, the databricks-sdk Python library must be installed.")
    elif synchronous:
        http_handler = http_handler or HTTPHandler(timeout=timeout)
        return DatabricksModelServingHTTPHandlerWrapper(
            api_base=api_base,
            api_key=api_key,
            http_handler=http_handler,
            custom_llm_provider=custom_llm_provider,
            logging_obj=logging_obj,
            custom_endpoint=custom_endpoint,
            headers=headers,
            streaming_decoder=streaming_decoder,
        )
    else:
        async_http_handler = http_handler or AsyncHTTPHandler(timeout=timeout)
        return DatabricksModelServingAsyncHTTPHandlerWrapper(
            api_base=api_base,
            api_key=api_key,
            http_handler=async_http_handler,
            custom_llm_provider=custom_llm_provider,
            logging_obj=logging_obj,
            custom_endpoint=custom_endpoint,
            headers=headers,
        )


class DatabricksModelServingWorkspaceClientWrapper(DatabricksModelServingClientWrapper):

    def __init__(self):
        from databricks.sdk import WorkspaceClient

        self.client = WorkspaceClient()

    def _completion(
        self,
        endpoint_name: str,
        messages: List[Dict[str, str]],
        # litellm params?
        optional_params: Dict[str, Any],
    ) -> ModelResponse:
        from databricks.sdk.service.serving import ChatMessage, ChatMessageRole, QueryEndpointResponse

        endpoint_response: QueryEndpointResponse = self.client.serving_endpoints.query(
          name=endpoint_name,
          messages=[
            self._translate_chat_message_for_query(message) for message in messages
          ],
          **optional_params
        )
        return self._translate_endpoint_query_response_to_model_response(endpoint_response)

    def _streaming_completion(
        self,
        endpoint_name: str,
        messages: List[Dict[str, str]],
        # litellm params?
        optional_params: Dict[str, Any],
    ) -> CustomStreamWrapper:
        raise DatabricksError(status_code=500, message="In order to make asynchronous or streaming calls, Databricks API base and API key must both be set.")

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
            object=query_response.object.value,
            system_fingerprint=query_response.served_model_name,
            _hidden_params={},
            _response_headers=None, 
        )

        return model_response


class DatabricksModelServingHandlerWrapper(DatabricksModelServingClientWrapper):
    def __init__(
        self,
        api_base: str,
        api_key: str,
        http_handler: Union[HTTPHandler, AsyncHTTPHandler],
        custom_llm_provider: str,
        logging_obj,
        custom_endpoint: Optional[bool],
        headers: Optional[Dict[str, str]] = None,
        streaming_decoder: Optional[CustomStreamingDecoder] = None,
    ):
        self.api_base = api_base
        self.api_key = api_key
        self.http_handler = http_handler
        self.headers = headers or {}
        self.custom_endpoint = custom_endpoint
        self.streaming_decoder = streaming_decoder
        self.custom_llm_provider = custom_llm_provider
        self.logging_obj = logging_obj

    def _get_api_base(self, endpoint_type: Literal["chat_completions", "embeddings"]) -> str:
        if self.custom_endpoint:
            return self.api_base
        elif endpoint_type == "chat_completions":
            return f"{self.api_base}/chat/completions"
        elif endpoint_type == "embeddings":
            return f"{self.api_base}/embeddings"
        else:
            raise DatabricksError(status_code=500, message=f"Invalid endpoint type: {endpoint_type}")

    def _build_headers(self) -> Dict[str, str]:
        return {
            **self.headers,
            "Authorization": f"Bearer {self.api_key}"
        }

    def _prepare_data(
        self,
        endpoint_name: str, 
        messages: List[Dict[str, str]],
        optional_params: Dict[str, Any],
    ) -> Dict[str, Any]:
        return {
            "model": endpoint_name,
            "messages": messages,
            **optional_params,
        }

    def _handle_errors(self, exception: Exception, response: Optional[httpx.Response]):
        if isinstance(exception, httpx.HTTPStatusError) and response is not None:
            raise DatabricksError(status_code=exception.response.status_code, message=response.text)
        elif isinstance(exception, httpx.TimeoutException):
            raise DatabricksError(status_code=408, message="Timeout error occurred.")
        else:
            raise DatabricksError(status_code=500, message=str(exception))


class DatabricksModelServingHTTPHandlerWrapper(DatabricksModelServingHandlerWrapper):

    def _completion(
        self,
        endpoint_name: str,
        messages: List[Dict[str, str]],
        optional_params: Dict[str, Any],
        streaming: bool,
    ) -> ModelResponse:
        data = self._prepare_data(endpoint_name, messages, optional_params)
        response = None
        try:
            print(self._build_headers())
            response = self.http_handler.post(
                self._get_api_base(endpoint_type="chat_completions"),
                headers=self._build_headers(),
                data=json.dumps(data)
            )
            response.raise_for_status()
            return ModelResponse(**response.json())
        except (httpx.HTTPStatusError, httpx.TimeoutException, Exception) as e:
            self._handle_errors(e, response)

    def _streaming_completion(
        self,
        endpoint_name: str,
        messages: List[Dict[str, str]],
        optional_params: Dict[str, Any],
    ):
        data = self._prepare_data(endpoint_name, messages, optional_params)

        def make_call(client: HTTPHandler):
            response = None
            try:
                response = client.post(
                    self._get_api_base(endpoint_type="chat_completions"),
                    headers=self._build_headers(),
                    data=json.dumps(data),
                    stream=True
                )
                response.raise_for_status()
            except (httpx.HTTPStatusError, httpx.TimeoutException, Exception) as e:
                self._handle_errors(e, response)

            if self.streaming_decoder is not None:
                return self.streaming_decoder.iter_bytes(
                    response.iter_bytes(chunk_size=1024)
                )
            else:
                return ModelResponseIterator(
                    streaming_response=response.iter_lines(), sync_stream=True
                )

        return CustomStreamWrapper(
            completion_stream=None,
            make_call=make_call,
            model=endpoint_name,
            custom_llm_provider=self.custom_llm_provider,
            logging_obj=self.logging_obj,
        )


class DatabricksModelServingAsyncHTTPHandlerWrapper(DatabricksModelServingHandlerWrapper):

    async def _completion(
        self,
        endpoint_name: str, 
        messages: List[Dict[str, str]],
        optional_params: Dict[str, Any] = None
    ) -> ModelResponse:
        data = self._prepare_data(endpoint_name, messages, optional_params)
        response = None
        try:
            response = await self.http_handler.post(
                self._get_api_base(endpoint_type="chat_completions"),
                headers=self._build_headers(),
                data=json.dumps(data)
            )
            response.raise_for_status()
            return ModelResponse(**response.json())
        except (httpx.HTTPStatusError, httpx.TimeoutException, Exception) as e:
            self._handle_errors(e, response)


    async def _streaming_completion(
        self,
        endpoint_name: str,
        messages: List[Dict[str, str]],
        optional_params: Dict[str, Any],
    ):
        data = self._prepare_data(endpoint_name, messages, optional_params)

        async def make_call(client: AsyncHTTPHandler):
            response = None
            try:
                response = await client.post(
                    self._get_api_base(endpoint_type="chat_completions"),
                    headers=self._build_headers(),
                    data=json.dumps(data),
                    stream=True
                )
                response.raise_for_status()
            except (httpx.HTTPStatusError, httpx.TimeoutException, Exception) as e:
                self._handle_errors(e, response)

            if self.streaming_decoder is not None:
                return self.streaming_decoder.iter_bytes(
                    response.aiter_bytes(chunk_size=1024)
                )
            else:
                return ModelResponseIterator(
                    streaming_response=response.aiter_lines(), sync_stream=True
                )

        return CustomStreamWrapper(
            completion_stream=None,
            make_call=make_call,
            model=endpoint_name,
            custom_llm_provider=self.custom_llm_provider,
            logging_obj=self.logging_obj,
        )
