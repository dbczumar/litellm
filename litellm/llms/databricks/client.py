import json
from abc import abstractmethod
from typing import Optional, List, Dict, Any, Literal, Union

import httpx  # type: ignore

from litellm.llms.databricks.exceptions import DatabricksError
from litellm.llms.databricks.streaming_utils import ModelResponseIterator
from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj
from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler, HTTPHandler
from litellm.utils import ModelResponse, Choices, CustomStreamWrapper
from litellm.types.utils import CustomStreamingDecoder


class DatabricksModelServingClientWrapper:
    """
    A wrapper around a Databricks Model Serving API client that exposes inference APIs
    (e.g. chat completion).
    """

    def completion(
        self,
        endpoint_name: str,
        messages: List[Dict[str, str]],
        optional_params: Dict[str, Any],
        stream: bool,
    ) -> Union[ModelResponse, CustomStreamWrapper]:
        """
        Send a synchronous chat completion request to a Databricks Model Serving endpoint and
        retrieve a response.

        Args:
            endpoint_name (str): The name of the endpoint to query.
            messages (List[Dict[str, str]]): The list of messages (chat history) to send to the
                                             endpoint.
            optional_params (Dict[str, Any]): Optional parameters to include in the request (e.g.
                                              temperature, max_tokens, etc.).
            stream (bool): Whether or not to return a streaming response.

        Returns:
            A streaming response (CustomStreamWrapper) or a buffered / standard response
            (ModelResponse) from the endpoint.
        """
        if stream:
            return self._streaming_completion(endpoint_name, messages, optional_params)
        else:
            return self._completion(endpoint_name, messages, optional_params)

    async def acompletion(
        self,
        endpoint_name: str,
        messages: List[Dict[str, str]],
        optional_params: Dict[str, Any],
        stream: bool,
    ) -> Union[ModelResponse, CustomStreamWrapper]:
        """
        Send an asynchronous chat completion request to a Databricks Model Serving endpoint and
        retrieve a response.

        Args:
            endpoint_name (str): The name of the endpoint to query.
            messages (List[Dict[str, str]]): The list of messages (chat history) to send to the
                                             endpoint.
            optional_params (Dict[str, Any]): Optional parameters to include in the request (e.g.
                                              temperature, max_tokens, etc.).
            stream (bool): Whether or not to return a streaming response.

        Returns:
            A streaming response (CustomStreamWrapper) or a buffered / standard response
            (ModelResponse) from the endpoint.
        """

        if stream:
            return await self._streaming_completion(endpoint_name, messages, optional_params)
        else:
            return await self._completion(endpoint_name, messages, optional_params)

    @abstractmethod
    def _completion(
        self,
        endpoint_name: str,
        messages: List[Dict[str, str]],
        optional_params: Dict[str, Any],
    ) -> ModelResponse:
        """
        Base method for sending a synchronous chat completion request to a Databricks Model Serving
        endpoint and retrieving a response. This method should be implemented by subclasses.
        """
        pass

    @abstractmethod
    def _streaming_completion(
        self,
        endpoint_name: str,
        messages: List[Dict[str, str]],
        optional_params: Dict[str, Any],
    ) -> CustomStreamWrapper:
        """
        Base method for sending an asynchronous chat completion request to a Databricks Model
        Serving endpoint and retrieving a response. This method should be implemented by subclasses.
        """
        pass


def get_databricks_model_serving_client_wrapper(
    support_async: bool,
    custom_llm_provider: str,
    logging_obj: LiteLLMLoggingObj,
    api_base: Optional[str],
    api_key: Optional[str],
    http_handler: Optional[Union[HTTPHandler, AsyncHTTPHandler]],
    timeout: Optional[Union[float, httpx.Timeout]],
    custom_endpoint: Optional[bool],
    headers: Optional[Dict[str, str]],
    streaming_decoder: Optional[CustomStreamingDecoder],
) -> DatabricksModelServingClientWrapper:
    """
    Obtain a wrapper around a Databricks Model Serving API client that exposes inference APIs
    (e.g. chat completion). The wrapper is constructed with the provided configuration.

    Args:
        support_async (bool): Indicates whether the wrapper needs to support asynchronous execution.
        custom_llm_provider (str): The name of the custom LLM provider (typically "Databricks").
        logging_obj (object): The LiteLLM logging object to use for event logging.
        api_base (Optional[str]): The base URL of the Databricks Model Serving API. If not provided,
            the Databricks SDK will be used to automatically retrieve the API key and base URL
            from the current environment. If `api_base` is provided, `api_key` must also be provided.
        api_key (Optional[str]): The API key to use for authentication. If not provided, the
            Databricks SDK  will automatically retrieve the API key and base URL from the current
            environment. If `api_base` is provided, `api_key` must also be provided.
        http_handler (Optional[Union[HTTPHandler, AsyncHTTPHandler]]): The HTTP handler to use for
            requests. If an HTTP handler is provided, `api_base` and `api_key` must also be
            provided. If not provided, a default HTTP handler will be used.
        timeout (Optional[Union[float, httpx.Timeout]]): The timeout to use for requests.
        custom_endpoint (Optional[bool]): Indicates whether a custom endpoint is being used. If
            `custom_endpoint` is `True`, `api_base` and `api_key` must be provided.
        headers (Optional[Dict[str, str]]): Additional headers to include in requests.
        streaming_decoder (Optional[CustomStreamingDecoder]): The decoder to use for
            processing streaming responses.

    """
    if (api_base, api_key).count(None) == 1:
        raise DatabricksError(status_code=400, message="Databricks API base URL and API key must both be set, or both must be unset.")

    if custom_endpoint is not None and (api_base, api_key).count(None) > 0:
        raise DatabricksError(status_code=400, message="If a custom endpoint is specified, Databricks API base URL and API key must both be set.")

    if (http_handler is not None) and (api_base, api_key).count(None) > 0:
        raise DatabricksError(status_code=500, message="If http_handler is provided, api_base and api_key must be provided.")

    if (api_base, api_key).count(None) == 2:
        # If no API base URL or API key is provided we will use the Databricks SDK, which can
        # automatically retrieve an API key and API base URL configuration from the current
        # environment.
        if support_async:
            # The Databricks SDK does not support streaming or asynchronous calls.
            raise DatabricksError(status_code=400, message="In order to make asynchronous calls, Databricks API base URL and API key must both be set.")

        try:
            import databricks.sdk

            return DatabricksModelServingWorkspaceClientWrapper()
        except ImportError:
            raise DatabricksError(status_code=400, message="If Databricks API base URL and API key are not provided, the databricks-sdk Python library must be installed.")
    elif support_async:
        async_http_handler = http_handler if isinstance(http_handler, AsyncHTTPHandler) else AsyncHTTPHandler(timeout=timeout)
        return DatabricksModelServingAsyncHTTPHandlerWrapper(
            api_base=api_base,
            api_key=api_key,
            http_handler=async_http_handler,
            custom_llm_provider=custom_llm_provider,
            logging_obj=logging_obj,
            custom_endpoint=custom_endpoint,
            headers=headers,
        )
    else:
        http_handler = http_handler if isinstance(http_handler, HTTPHandler) else HTTPHandler(timeout=timeout)
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


class DatabricksModelServingWorkspaceClientWrapper(DatabricksModelServingClientWrapper):
    """
    A wrapper around the Databricks Model Serving API client that uses the Databricks SDK to
    make inference calls. This wrapper is used when the Databricks API base URL and API key are not
    provided, and the Databricks SDK is used to automatically retrieve the API key and base URL
    from the current environment.

    The Databricks SDK does not support asynchronous calls or streaming, so this wrapper is only
    used for synchronous, non-streaming calls.

    TODO: Support streaming inference and asynchronous execution with the Databricks SDK
    """

    def __init__(self):
        from databricks.sdk import WorkspaceClient

        self.client = WorkspaceClient()

    def _completion(
        self,
        endpoint_name: str,
        messages: List[Dict[str, str]],
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
        optional_params: Dict[str, Any],
    ) -> CustomStreamWrapper:
        # NB: The user-facing error message here indicates that the
        # DatabricksModelServingWorkspaceClientWrapper should not have been used to handle a
        # streaming request, implying that the user should have provided an API base URL and API
        # key. The location of the error message is not ideal, but it's temporary.
        #
        # TODO: Support streaming inference with the Databricks SDK
        raise DatabricksError(status_code=500, message="In order to make asynchronous or streaming calls, Databricks API base URL and API key must both be set.")

    def _translate_chat_message_for_query(self, message: Dict[str, str]) -> 'ChatMessage':
        from databricks.sdk.service.serving import ChatMessage, ChatMessageRole

        return ChatMessage(
            content=message["content"],
            role=ChatMessageRole(message["role"])
        )

    def _translate_endpoint_query_response_to_model_response(self, query_response: 'QueryEndpointResponse') -> ModelResponse:
        from databricks.sdk.service.serving import V1ResponseChoiceElement

        def convert_choice(v1_choice: V1ResponseChoiceElement) -> Choices:
            return Choices(
                finish_reason=v1_choice.finish_reason,
                index=v1_choice.index,
                message=v1_choice.message.as_dict(),
                logprobs=v1_choice.logprobs,
            )

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
    """
    A wrapper around the Databricks Model Serving API client that uses a LiteLLM synchronous
    or asynchronous HTTP handler to make inference calls. This wrapper requires an API base
    URL and API key to be provided. It supports both synchronous and asynchronous calls, as well
    as streaming.
    """
    def __init__(
        self,
        api_base: str,
        api_key: str,
        http_handler: Union[HTTPHandler, AsyncHTTPHandler],
        custom_llm_provider: str,
        logging_obj: LiteLLMLoggingObj,
        custom_endpoint: Optional[bool],
        headers: Optional[Dict[str, str]] = None,
        streaming_decoder: Optional[CustomStreamingDecoder] = None,
    ):
        """
        Args:
            api_base (str): The base URL of the Databricks Model Serving API.
            api_key (str): The API key to use for authentication.
            http_handler (Union[HTTPHandler, AsyncHTTPHandler]): The HTTP handler to use for
                requests.
            custom_llm_provider (str): The name of the custom LLM provider (typically "Databricks").
            logging_obj (object): The LiteLLM logging object to use for event logging.
            custom_endpoint (Optional[bool]): Indicates whether a custom endpoint is being used.
            headers (Optional[Dict[str, str]]): Additional headers to include in requests.
            streaming_decoder (Optional[CustomStreamingDecoder]): The decoder to use for
                processing streaming responses.
        """
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
    """
    A wrapper around the Databricks Model Serving API client that uses a LiteLLM synchronous
    HTTP handler to make inference calls. This wrapper requires an API base URL and API key to
    be provided. It supports streaming.
    """

    def _completion(
        self,
        endpoint_name: str,
        messages: List[Dict[str, str]],
        optional_params: Dict[str, Any],
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
    """
    A wrapper around the Databricks Model Serving API client that uses a LiteLLM asynchronous
    HTTP handler to make inference calls. This wrapper requires an API base URL and API key to
    be provided. It supports streaming.
    """

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
