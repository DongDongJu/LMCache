# SPDX-License-Identifier: Apache-2.0
"""One reusable async HTTP client per remote service.

This package also holds the typed error model and the two request
primitives every client shares: :func:`get_json` (bounded retries, GET
only) and :func:`post_json_once` (exactly one attempt). The distinction
between :class:`ClientConnectionError` (the request provably never reached
the server) and :class:`AmbiguousMutationError` (it may have) is what lets
the controller reconcile a failed POST against status instead of retrying.
"""

# Standard
import asyncio
import json

# Third Party
import httpx

# First Party
from lmcache.logging import init_logger

logger = init_logger(__name__)


class ClientError(Exception):
    """Base of every error raised by the HTTP clients."""


class ClientConnectionError(ClientError):
    """The request could not be sent at all (connect failure/timeout)."""


class ClientTimeoutError(ClientError):
    """A GET timed out after every bounded attempt."""


class ClientHTTPError(ClientError):
    """The server answered with a non-2xx status.

    Attributes:
        status_code: The HTTP status code.
        body: The raw response text (truncated for logs by the caller).
    """

    def __init__(self, status_code: int, body: str, url: str) -> None:
        """Args:
        status_code: The HTTP status code.
        body: The raw response text.
        url: The request URL, for the message.
        """
        super().__init__(f"HTTP {status_code} from {url}: {body[:200]}")
        self.status_code = status_code
        self.body = body


class ClientResponseError(ClientError):
    """The response was malformed or violated the documented contract."""


class AmbiguousMutationError(ClientError):
    """A POST was sent (or may have been) but no usable response arrived.

    The effect may or may not have been applied. The caller must reconcile
    with the remote's status and must never re-issue the POST blindly.
    """


class JSONBody:
    """A decoded JSON body with typed accessors.

    Wraps the ``object`` returned by :func:`json.loads` so callers do not
    pass untyped values around.

    Attributes:
        value: The decoded value.
    """

    def __init__(self, value: object) -> None:
        """Args:
        value: The decoded JSON value.
        """
        self.value = value

    def as_dict(self) -> dict[str, object]:
        """Return the value as a JSON object.

        Raises:
            ClientResponseError: If the value is not an object.
        """
        if not isinstance(self.value, dict):
            raise ClientResponseError("expected a JSON object")
        return {str(k): v for k, v in self.value.items()}


async def get_json(
    client: httpx.AsyncClient,
    url: str,
    *,
    attempts: int,
    backoff_seconds: float = 0.2,
) -> JSONBody:
    """GET ``url`` with bounded retries and decode the JSON body.

    Retries on connect/read failures, timeouts, and ``5xx`` responses up to
    ``attempts`` times with linear backoff. A ``4xx`` is returned to the
    caller immediately as :class:`ClientHTTPError`.

    Args:
        client: The shared client of the remote.
        url: Absolute URL.
        attempts: Maximum attempts (``>= 1``).
        backoff_seconds: Sleep between attempts, multiplied by the attempt
            number.

    Returns:
        The decoded body.

    Raises:
        ClientHTTPError: On a ``4xx`` response, or a ``5xx`` on the last
            attempt.
        ClientTimeoutError: If every attempt timed out.
        ClientConnectionError: If every attempt failed to connect.
        ClientResponseError: If the final body is not valid JSON.
    """
    last: ClientError = ClientConnectionError(f"no attempt made for {url}")
    for attempt in range(1, attempts + 1):
        try:
            response = await client.get(url)
        except httpx.TimeoutException as exc:
            last = ClientTimeoutError(f"GET {url} timed out: {exc}")
        except httpx.HTTPError as exc:
            last = ClientConnectionError(f"GET {url} failed: {exc}")
        else:
            if 200 <= response.status_code < 300:
                return _decode(response, url)
            last = ClientHTTPError(response.status_code, response.text, url)
            if response.status_code < 500:
                raise last
        if attempt < attempts:
            await asyncio.sleep(backoff_seconds * attempt)
    raise last


async def post_json_once(
    client: httpx.AsyncClient,
    url: str,
    body: dict[str, object],
) -> tuple[int, JSONBody]:
    """POST ``body`` exactly once and decode the response.

    Args:
        client: The shared client of the remote.
        url: Absolute URL.
        body: The JSON body; sent verbatim.

    Returns:
        ``(status_code, decoded_body)`` for any HTTP status. The caller
        decides what a non-2xx status means for its contract.

    Raises:
        ClientConnectionError: If the connection could not be established,
            i.e. the request was never sent.
        AmbiguousMutationError: On any other transport failure -- the
            request may have been applied.
        ClientResponseError: If the body is not valid JSON.
    """
    try:
        response = await client.post(url, json=body)
    except (httpx.ConnectError, httpx.ConnectTimeout) as exc:
        raise ClientConnectionError(f"POST {url} not sent: {exc}") from exc
    except httpx.HTTPError as exc:
        raise AmbiguousMutationError(f"POST {url} outcome unknown: {exc}") from exc
    return response.status_code, _decode(response, url)


def _decode(response: httpx.Response, url: str) -> JSONBody:
    """Decode a JSON response body.

    Raises:
        ClientResponseError: If the body is not valid JSON.
    """
    try:
        return JSONBody(json.loads(response.text) if response.text else None)
    except ValueError as exc:
        raise ClientResponseError(f"non-JSON body from {url}: {exc}") from exc
