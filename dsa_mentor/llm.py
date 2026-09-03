"""OpenAI-compatible chat client + agentic tool-call loop for DSA Mentor.

Talks to the local OpenAI-compatible endpoint (see EXECUTION_NOTES.md) using
plain ``requests`` — no SDK dependencies. All tunables come from
:class:`dsa_mentor.config.Config`; nothing here is hardcoded.

Public API:

- :class:`LLMClient` — POSTs to ``{base_url}/chat/completions``. Model names
  resolve from a tier (``"large"``/``"medium"``/``"small"``) or an explicit
  model id; sampling defaults come from config with per-call overrides;
  timeouts / connection errors / HTTP 5xx (and 429) are retried with
  exponential backoff.
- :func:`run_tool_loop` — the agentic retrieval loop of spec §23/§25: repeated
  chat calls while the model requests tools, executing them through an injected
  ``tool_executor(name, args_dict) -> str``, bounded by a hard tool-call budget.

Failures raise :class:`LLMError` subclasses with actionable messages (HTTP
failures include the status code and a snippet of the response body).
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

import requests

from dsa_mentor.config import Config

logger = logging.getLogger(__name__)

# HTTP statuses worth retrying (transient server overload / rate limiting).
_RETRYABLE_STATUSES = frozenset({429, 500, 502, 503, 504})
# How much of a failure body to include in error messages.
_BODY_SNIPPET_LIMIT = 1000


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------

class LLMError(Exception):
    """Base class for all LLM client errors."""


class LLMConnectionError(LLMError):
    """The endpoint could not be reached (timeout or connection failure)."""


class LLMHTTPError(LLMError):
    """The endpoint answered with an HTTP error status.

    Carries ``status_code``, ``url`` and a truncated ``body_snippet`` for
    debugging; the exception message itself includes all three.
    """

    def __init__(
        self,
        message: str,
        *,
        status_code: Optional[int] = None,
        url: Optional[str] = None,
        body_snippet: Optional[str] = None,
    ) -> None:
        if body_snippet is not None:
            message = f"{message} — response body: {body_snippet}"
        super().__init__(message)
        self.status_code = status_code
        self.url = url
        self.body_snippet = body_snippet


class LLMResponseError(LLMError):
    """The endpoint returned a success response with an unusable structure."""


def _snippet(text: Optional[str], limit: int = _BODY_SNIPPET_LIMIT) -> str:
    if not text:
        return "<empty body>"
    text = " ".join(text.split())  # collapse whitespace for compact messages
    if len(text) > limit:
        return text[:limit] + f"… [+{len(text) - limit} chars truncated]"
    return text


# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------

class LLMClient:
    """Minimal OpenAI-compatible chat completions client.

    Args:
        config: a validated :class:`dsa_mentor.config.Config` (from
            ``load_config()``). The client reads the ``llm`` section once at
            construction; edit config.json and rebuild to change behavior.
    """

    def __init__(self, config: Config) -> None:
        if not isinstance(config, Config):
            raise TypeError(
                f"LLMClient expects a dsa_mentor.config.Config instance "
                f"(got {type(config).__name__}); call load_config() first"
            )
        llm = config.llm
        self._config = config
        self.base_url: str = llm.base_url.rstrip("/")
        if not self.base_url:
            raise ValueError("llm.base_url must be a non-empty URL")
        self.api_key: str = llm.api_key or ""
        self.models: dict[str, str] = dict(llm.models)
        self.default_model: str = llm.default_model
        self.timeout_seconds: float = llm.timeout_seconds
        self.max_retries: int = llm.max_retries
        self._backoff_base: float = llm.retry_backoff_base_seconds
        self._backoff_cap: float = llm.retry_max_delay_seconds
        self.sampling_defaults: dict[str, Any] = {
            "temperature": llm.sampling.temperature,
            "top_p": llm.sampling.top_p,
            "top_k": llm.sampling.top_k,
            "max_tokens": llm.sampling.max_tokens,
        }
        self._session = requests.Session()

    # -- lifecycle ---------------------------------------------------------

    def close(self) -> None:
        """Close the underlying HTTP session (releases pooled connections)."""
        self._session.close()

    def __enter__(self) -> "LLMClient":
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    # -- model resolution ----------------------------------------------------

    def resolve_model(self, model: Optional[str] = None) -> str:
        """Resolve ``model`` to a concrete model id.

        ``None`` → the configured default; a known tier name (e.g. ``"large"``)
        → that tier's model id; anything else is passed through as an explicit
        model id.
        """
        if model is None:
            model = self.default_model
        if not isinstance(model, str) or not model.strip():
            raise ValueError(
                f"model must be a non-empty tier name ({sorted(self.models)}) "
                f"or an explicit model id (got {model!r})"
            )
        model = model.strip()
        return self.models.get(model, model)

    # -- chat -----------------------------------------------------------------

    def chat(
        self,
        messages: list[dict],
        model: Optional[str] = None,
        tools: Optional[list[dict]] = None,
        tool_choice: Any = None,
        **sampling_overrides: Any,
    ) -> dict:
        """Send one chat completion request and return the parsed JSON response.

        Args:
            messages: OpenAI-style message list (``role`` + ``content``, plus
                ``tool_calls``/``tool_call_id`` where applicable). Must be a
                non-empty list of dicts, each with a string ``role``.
            model: tier name or explicit model id; defaults to config's
                ``llm.default_model``.
            tools: optional OpenAI-style tool schema list (function calling).
            tool_choice: optional pass-through (e.g. ``"auto"``, ``"none"``);
                omitted from the payload when ``None``.
            **sampling_overrides: per-call overrides for any of
                ``temperature``, ``top_p``, ``top_k``, ``max_tokens``.

        Returns:
            The full parsed response dict (``choices[0].message`` holds the
            assistant message, possibly with ``tool_calls``).

        Raises:
            ValueError: invalid messages/tools/overrides.
            LLMConnectionError / LLMHTTPError / LLMResponseError: transport or
                protocol failures (see module docstring).
        """
        payload = self._build_payload(
            messages, model=model, tools=tools, tool_choice=tool_choice,
            sampling_overrides=sampling_overrides,
        )
        return self._post_with_retries(payload)

    # -- retrieval convenience -------------------------------------------------

    def chat_with_retrieval(
        self,
        retrieval_result: Any,
        user_query: str,
        conversation_context: str = "",
        rag_enabled: bool = True,
        **sampling_overrides: Any,
    ) -> dict:
        """Build context from a retrieval result and send to the LLM.

        Convenience wrapper that constructs the message list via
        :class:`ContextBuilder` and then calls :meth:`chat`.

        Parameters
        ----------
        retrieval_result : RetrievalResult
            Output of a retriever (``KneeHierarchicalRetriever.retrieve()``,
            ``FlatRetriever.search()``, etc.).
        user_query : str
            The user's question.
        conversation_context : str
            Optional summary of prior conversation turns.
        rag_enabled : bool
            When ``False`` the context contains only the system prompt and
            user query (no retrieved knowledge).
        **sampling_overrides :
            Per-call overrides for ``temperature``, ``top_p``, ``top_k``,
            ``max_tokens``.

        Returns
        -------
        dict
            The full parsed response dict (same as :meth:`chat`).
        """
        from .context import ContextBuilder

        messages = ContextBuilder().build(
            retrieval_result=retrieval_result,
            user_query=user_query,
            conversation_context=conversation_context,
            rag_enabled=rag_enabled,
        )
        return self.chat(messages, **sampling_overrides)

    # -- agentic retrieval convenience -------------------------------------------

    def chat_with_tools(
        self,
        retriever: Any,
        user_query: str,
        conversation_context: str = "",
        rag_enabled: bool = True,
        max_tool_calls: Optional[int] = None,
        **sampling_overrides: Any,
    ) -> ToolLoopResult:
        """Build initial context via ContextBuilder, run the tool-call loop, and return.

        Convenience wrapper that:

        1. Builds initial context (system + user with retrieved knowledge) via
           :class:`ContextBuilder` — no tool results yet, just the first retrieval
           turn and the user's question.
        2. Adds the ``search_knowledge`` tool schema and executor (when
           ``rag_enabled=True``) per spec §21–§22.
        3. Calls :func:`run_tool_loop` to let the model request additional
           searches (spec §23) up to the configured budget (spec §25).

        Parameters
        ----------
        retriever : KneeHierarchicalRetriever
            The knee-aware hierarchical retriever (provides both the initial
            retrieval and the tool executor).
        user_query : str
            The user's question.
        conversation_context : str
            Optional summary of prior conversation turns.
        rag_enabled : bool
            When ``False`` the tool is NOT offered to the model (spec §72).
            Only the initial retrieval context + question are sent.
        max_tool_calls : int or None
            Hard cap on tool invocations. Defaults to
            ``config.agentic_retrieval.max_tool_calls``.
        **sampling_overrides :
            Per-call overrides for ``temperature``, ``top_p``, ``top_k``,
            ``max_tokens``.

        Returns
        -------
        ToolLoopResult
            Final content, full transcript, tool_calls_made count, and
            final_response dict.
        """
        from .context import ContextBuilder
        from .retrieval.tools import create_tool_executor, search_knowledge_tool

        # Initial retrieval to build context (spec §21: "initial retrieval → LLM")
        initial_result = retriever.retrieve(user_query)

        # Build initial messages with retrieved context
        messages = ContextBuilder().build(
            retrieval_result=initial_result,
            user_query=user_query,
            conversation_context=conversation_context,
            rag_enabled=rag_enabled,
        )

        # Determine tool-call budget
        if max_tool_calls is None:
            max_tool_calls = self._config.agentic_retrieval.max_tool_calls

        # Only offer the tool when RAG is enabled (spec §72)
        if rag_enabled:
            tool_def = search_knowledge_tool(retriever)
            tool_executor = create_tool_executor(
                retriever,
                max_results_from_config=max_tool_calls,
            )
            return run_tool_loop(
                self,
                messages,
                tools=[tool_def],
                tool_executor=tool_executor,
                max_calls=max_tool_calls,
                **sampling_overrides,
            )
        else:
            # RAG OFF: no tool, just send the initial context and get an answer
            response = self.chat(messages, **sampling_overrides)
            message = _extract_message(response)
            content = _content_of(message)
            return ToolLoopResult(
                content=content,
                transcript=list(messages) + [{"role": "assistant", "content": content}],
                tool_calls_made=0,
                final_response=response,
            )

    # -- internals -------------------------------------------------------------

    def _build_payload(
        self,
        messages: list[dict],
        model: Optional[str],
        tools: Optional[list[dict]],
        tool_choice: Any,
        sampling_overrides: dict[str, Any],
    ) -> dict:
        if not isinstance(messages, (list, tuple)) or len(messages) == 0:
            raise ValueError("messages must be a non-empty list of message dicts")
        for i, message in enumerate(messages):
            if not isinstance(message, dict) or not isinstance(
                message.get("role"), str
            ) or not message["role"].strip():
                raise ValueError(
                    f"messages[{i}] must be a dict with a non-empty string 'role' "
                    f"(got {message!r})"
                )

        sampling = dict(self.sampling_defaults)
        unknown = set(sampling_overrides) - set(sampling)
        if unknown:
            raise ValueError(
                f"unknown sampling override(s): {sorted(unknown)}; "
                f"allowed per-call overrides are {sorted(sampling)}"
            )
        sampling.update(sampling_overrides)

        payload: dict[str, Any] = {
            "model": self.resolve_model(model),
            "messages": [dict(m) for m in messages],  # copy: never mutate caller's list
            **sampling,
        }

        if tools is not None:
            if not isinstance(tools, (list, tuple)) or len(tools) == 0:
                raise ValueError("tools must be a non-empty list of tool schemas")
            for i, tool in enumerate(tools):
                function = tool.get("function") if isinstance(tool, dict) else None
                name = function.get("name") if isinstance(function, dict) else None
                if not isinstance(name, str) or not name.strip():
                    raise ValueError(
                        f"tools[{i}] must be an object with 'function.name' "
                        f"(got {tool!r})"
                    )
            payload["tools"] = list(tools)

        if tool_choice is not None:
            payload["tool_choice"] = tool_choice
        return payload

    def chat_stream(
        self,
        messages: list[dict],
        model: Optional[str] = None,
        tools: Optional[list[dict]] = None,
        tool_choice: Any = None,
        **sampling_overrides: Any,
    ):
        """Send a streaming chat completion request and yield tokens.

        Yields (content_delta: str, full_content: str) tuples as tokens arrive.
        Uses Server-Sent Events (SSE) streaming from the OpenAI-compatible API.

        Args:
            messages: OpenAI-style message list.
            model: tier name or explicit model id.
            tools: optional tool schemas.
            tool_choice: optional tool_choice.
            **sampling_overrides: per-call sampling overrides.

        Yields:
            tuple[str, str]: (token_delta, accumulated_full_content)
        """
        payload = self._build_payload(
            messages, model=model, tools=tools, tool_choice=tool_choice,
            sampling_overrides=sampling_overrides,
        )
        payload["stream"] = True
        url = f"{self.base_url}/chat/completions"

        try:
            response = self._session.post(
                url, json=payload, headers=self._headers(),
                timeout=self.timeout_seconds, stream=True,
            )
        except (requests.exceptions.Timeout, requests.exceptions.ConnectionError) as exc:
            raise LLMConnectionError(f"streaming request to {url} failed: {exc}") from exc

        if response.status_code != 200:
            raise LLMHTTPError(
                f"LLM endpoint {url} returned HTTP {response.status_code}",
                status_code=response.status_code,
                url=url,
                body_snippet=_snippet(response.text),
            )

        full_content = ""
        for line in response.iter_lines():
            if not line:
                continue
            text = line.decode("utf-8")
            if text.startswith("data: "):
                data_str = text[6:]  # strip "data: " prefix
                if data_str.strip() == "[DONE]":
                    break
                try:
                    data = json.loads(data_str)
                except json.JSONDecodeError:
                    continue
                choices = data.get("choices", [])
                if not choices:
                    continue
                delta = choices[0].get("delta", {})
                content_delta = delta.get("content", "")
                if content_delta:
                    full_content += content_delta
                    yield content_delta, full_content

    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.api_key:  # empty key (placeholder) → no Authorization header
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    def _post_with_retries(self, payload: dict) -> dict:
        url = f"{self.base_url}/chat/completions"
        attempts = self.max_retries + 1
        last_error: Optional[LLMError] = None
        last_cause: Optional[BaseException] = None

        for attempt in range(1, attempts + 1):
            try:
                response = self._session.post(
                    url, json=payload, headers=self._headers(),
                    timeout=self.timeout_seconds,
                )
            except (requests.exceptions.Timeout, requests.exceptions.ConnectionError) as exc:
                last_error = LLMConnectionError(
                    f"request to {url} failed on attempt {attempt}/{attempts}: {exc}"
                )
                last_cause = exc
                logger.warning("LLM request %d/%d failed: %s", attempt, attempts, exc)
            else:
                if response.status_code in _RETRYABLE_STATUSES:
                    last_error = LLMHTTPError(
                        f"LLM endpoint {url} returned HTTP {response.status_code} "
                        f"on attempt {attempt}/{attempts}",
                        status_code=response.status_code,
                        url=url,
                        body_snippet=_snippet(response.text),
                    )
                    last_cause = None  # an HTTP error is not caused by a transport exception
                    logger.warning(
                        "LLM request %d/%d got HTTP %d: %s",
                        attempt, attempts, response.status_code, _snippet(response.text),
                    )
                elif response.status_code >= 400:
                    # Non-retryable client error — fail fast with the body.
                    raise LLMHTTPError(
                        f"LLM endpoint {url} returned HTTP {response.status_code}",
                        status_code=response.status_code,
                        url=url,
                        body_snippet=_snippet(response.text),
                    )
                else:
                    try:
                        data = response.json()
                    except ValueError as exc:
                        raise LLMResponseError(
                            f"LLM endpoint {url} returned a non-JSON success "
                            f"response (HTTP {response.status_code}): "
                            f"{_snippet(response.text)}"
                        ) from exc
                    _validate_response_shape(data, url)
                    return data

            if attempt < attempts:
                delay = min(self._backoff_cap, self._backoff_base * 2 ** (attempt - 1))
                time.sleep(delay)

        assert last_error is not None  # loop always sets it before sleeping/returning
        raise last_error from last_cause


def _validate_response_shape(data: Any, url: str) -> None:
    """Ensure the parsed response has the OpenAI chat-completion skeleton."""
    if not isinstance(data, dict):
        raise LLMResponseError(
            f"malformed chat completion from {url}: expected a JSON object, "
            f"got {_type_name(data)}"
        )
    choices = data.get("choices")
    if not isinstance(choices, list) or len(choices) == 0:
        raise LLMResponseError(
            f"malformed chat completion from {url}: missing 'choices' array; "
            f"body: {_snippet(_safe_dumps(data))}"
        )
    first = choices[0]
    if not isinstance(first, dict) or not isinstance(first.get("message"), dict):
        raise LLMResponseError(
            f"malformed chat completion from {url}: 'choices[0].message' is "
            f"missing; body: {_snippet(_safe_dumps(data))}"
        )


def _type_name(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, (int, float)):
        return "number"
    if isinstance(value, str):
        return "string"
    if isinstance(value, list):
        return "array"
    if isinstance(value, dict):
        return "object"
    return type(value).__name__


def _safe_dumps(data: Any) -> str:
    try:
        return json.dumps(data, ensure_ascii=False)
    except (TypeError, ValueError):
        return repr(data)[:_BODY_SNIPPET_LIMIT]


# ---------------------------------------------------------------------------
# Agentic tool-call loop (spec §23–§25)
# ---------------------------------------------------------------------------

@dataclass
class ToolLoopResult:
    """Outcome of :func:`run_tool_loop`.

    Attributes:
        content: the model's final answer text ("" if it produced none).
        transcript: the full message history — the input messages, every
            assistant/tool message appended during the loop, and (when the
            model finished cleanly) its final answer. A fresh list you may
            safely extend for follow-up turns.
        tool_calls_made: total number of tool invocations executed.
        final_response: raw parsed response dict of the last chat call.
    """

    content: str
    transcript: list = field(default_factory=list)
    tool_calls_made: int = 0
    final_response: Optional[dict] = None


def run_tool_loop(
    client: LLMClient,
    messages: list[dict],
    tools: list[dict],
    tool_executor: Callable[[str, dict], str],
    max_calls: int,
    *,
    model: Optional[str] = None,
    tool_choice: Any = None,
    **sampling_overrides: Any,
) -> ToolLoopResult:
    """Run the agentic retrieval loop of spec §23/§25.

    Repeatedly calls ``client.chat`` with the given tools; whenever the model's
    message contains ``tool_calls``, each one is executed via
    ``tool_executor(name, args_dict) -> str`` and the assistant + tool messages
    are appended to the transcript before looping again. The loop stops when
    the model answers without requesting tools, or — once ``max_calls`` tool
    invocations have been made (spec §25 guardrail) — on a final chat call made
    *without* tools so the model is forced to answer from the evidence gathered.

    Tool failures never crash the loop: executor exceptions and malformed
    arguments are reported back to the model as error strings in the tool
    message, giving it a chance to recover (e.g. rephrase the query).

    Args:
        client: an :class:`LLMClient`.
        messages: initial conversation (system + user, possibly with retrieved
            context already injected by the caller).
        tools: OpenAI-style tool schemas offered to the model.
        tool_executor: callable mapping ``(tool_name, args_dict)`` to a string
            result; retrieval wiring is injected here in a later phase.
        max_calls: hard cap on executed tool invocations (>= 0).
        model / tool_choice / **sampling_overrides: passed through to every
            :meth:`LLMClient.chat` call.

    Returns:
        A :class:`ToolLoopResult` with the final content and full transcript.
    """
    if not callable(tool_executor):
        raise TypeError(
            f"tool_executor must be a callable (name, args_dict) -> str "
            f"(got {type(tool_executor).__name__})"
        )
    if isinstance(max_calls, bool) or not isinstance(max_calls, int) or max_calls < 0:
        raise ValueError(f"max_calls must be an integer >= 0 (got {max_calls!r})")

    transcript = [dict(m) for m in messages]  # copy: never mutate caller's list
    calls_made = 0

    while True:
        budget_left = calls_made < max_calls
        if budget_left:
            response = client.chat(
                transcript, model=model, tools=tools, tool_choice=tool_choice,
                **sampling_overrides,
            )
        else:
            logger.info(
                "tool-call budget (%d) exhausted; requesting final answer without tools",
                max_calls,
            )
            response = client.chat(transcript, model=model, **sampling_overrides)

        message = _extract_message(response)
        tool_calls = _message_tool_calls(message)
        if not tool_calls or not budget_left:
            content = _content_of(message)
            if not tool_calls and not content.strip():
                logger.warning("tool loop ended with an empty final answer")
            elif not budget_left:
                logger.warning(
                    "model still requested %d tool call(s) after the budget of %d "
                    "was exhausted; returning its text as-is without executing them",
                    len(tool_calls), max_calls,
                )
            if not tool_calls:  # record the final answer so the transcript stays continuable
                transcript.append({**message, "role": "assistant"})
            return ToolLoopResult(
                content=content,
                transcript=transcript,
                tool_calls_made=calls_made,
                final_response=response,
            )

        # Append the assistant turn (with normalized, id-bearing tool calls)
        # and execute each requested call.
        normalized_calls = []
        for idx, call in enumerate(tool_calls):
            call_id = call.get("id") or f"call_{len(transcript)}_{idx}"
            function = call["function"]  # validated by _message_tool_calls
            normalized_calls.append(
                {
                    "id": call_id,
                    "type": "function",
                    "function": {
                        "name": function["name"],
                        "arguments": function.get("arguments") or "",
                    },
                }
            )
        assistant_message = {**message, "role": "assistant", "tool_calls": normalized_calls}
        transcript.append(assistant_message)

        for idx, call in enumerate(tool_calls):
            name = call["function"]["name"]
            args_raw = call["function"].get("arguments")
            logger.info(
                "tool call %d: %s(%s)", calls_made + idx + 1, name, _safe_dumps(args_raw)
            )

            try:
                args = _parse_tool_args(args_raw)
            except ValueError as exc:
                result = (
                    f"ERROR: invalid arguments for tool '{name}': {exc}. "
                    f"Re-issue the call with a valid JSON object of arguments."
                )
            else:
                try:
                    result = tool_executor(name, args)
                except Exception as exc:  # contain executor failures in the loop
                    logger.warning("tool '%s' raised %s: %s", name, type(exc).__name__, exc)
                    result = f"ERROR: tool '{name}' failed with {type(exc).__name__}: {exc}"
                else:
                    if not isinstance(result, str):  # be lenient: serialize non-str results
                        try:
                            result = json.dumps(result, ensure_ascii=False)
                        except (TypeError, ValueError):
                            result = str(result)

            transcript.append(
                {
                    "role": "tool",
                    "tool_call_id": normalized_calls[idx]["id"],
                    "name": name,
                    "content": result,
                }
            )
            calls_made += 1


def _extract_message(response: dict) -> dict:
    """Return ``choices[0].message`` from a validated chat response."""
    choices = response.get("choices")
    if not isinstance(choices, list) or len(choices) == 0:
        raise LLMResponseError(
            f"malformed chat completion: missing 'choices' array; body: {_snippet(_safe_dumps(response))}"
        )
    first = choices[0]
    message = first.get("message") if isinstance(first, dict) else None
    if not isinstance(message, dict):
        raise LLMResponseError(
            f"malformed chat completion: 'choices[0].message' is missing; "
            f"body: {_snippet(_safe_dumps(response))}"
        )
    return message


def _content_of(message: dict) -> str:
    content = message.get("content")
    return content if isinstance(content, str) else ""


def _message_tool_calls(message: dict) -> list[dict]:
    """Return the message's tool calls (validated), or [] for a plain answer."""
    raw = message.get("tool_calls")
    if raw is None:
        return []
    if not isinstance(raw, list):
        raise LLMResponseError(
            f"malformed chat completion: 'message.tool_calls' must be an array, "
            f"got {_type_name(raw)}"
        )
    calls = []
    for i, call in enumerate(raw):
        function = call.get("function") if isinstance(call, dict) else None
        name = function.get("name") if isinstance(function, dict) else None
        if not isinstance(name, str) or not name.strip():
            raise LLMResponseError(
                f"malformed chat completion: tool_calls[{i}] is missing "
                f"'function.name'; body: {_snippet(_safe_dumps(call))}"
            )
        calls.append(call)
    return calls


def _parse_tool_args(args_raw: Any) -> dict:
    """Parse a tool call's arguments into a dict (OpenAI sends a JSON string)."""
    if args_raw is None or args_raw == "":
        return {}
    if isinstance(args_raw, dict):  # be lenient with servers that send objects
        return args_raw
    if not isinstance(args_raw, str):
        raise ValueError(f"arguments must be a JSON string (got {_type_name(args_raw)})")
    try:
        parsed = json.loads(args_raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"{args_raw[:200]!r} is not valid JSON ({exc.msg})") from None
    if not isinstance(parsed, dict):
        raise ValueError(
            f"arguments must be a JSON object (got {_type_name(parsed)})"
        )
    return parsed
