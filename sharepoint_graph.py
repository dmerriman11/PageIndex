"""Microsoft Graph client for SharePoint sync: app-only tokens, retries, paging and safe downloads.

Everything that talks to graph.microsoft.com or login.microsoftonline.com goes through GraphClient,
which takes its HTTP session as a parameter so tests can fake it. Error messages never contain
tokens or the client secret.
"""
import os
import random as _random
import threading
import time
from pathlib import Path
from typing import Callable, Iterator, Optional
from urllib.parse import quote

import requests

GRAPH_BASE_URL = "https://graph.microsoft.com/v1.0"
TOKEN_URL = "https://login.microsoftonline.com/{tenant}/oauth2/v2.0/token"
TOKEN_SCOPE = "https://graph.microsoft.com/.default"
REQUEST_TIMEOUT = (10, 60)  # seconds: connect, read
RETRY_STATUSES = {429, 503, 504}
MAX_RETRIES = 5
MAX_RETRY_AFTER_SECONDS = 120.0
TOKEN_REFRESH_MARGIN_SECONDS = 300
DOWNLOAD_CHUNK_BYTES = 1024 * 1024
MAX_ERROR_MESSAGE_CHARS = 300

Credentials = tuple[str, str, str]  # tenant id, client id, client secret


class GraphError(Exception):
    """A Graph or sign-in failure; the message is safe to show to admins."""

    def __init__(self, message: str, status: Optional[int] = None):
        super().__init__(message)
        self.status = status


class GraphCredentialsMissing(GraphError):
    pass


def graph_url(path_or_url: str) -> str:
    if path_or_url.startswith("https://"):
        return path_or_url
    return f"{GRAPH_BASE_URL}/{path_or_url.lstrip('/')}"


def download_deadline_seconds(size_bytes: int) -> float:
    """Overall time allowed for one download: 60 s plus 1 s per MB, capped at 10 minutes."""
    return min(600.0, 60.0 + size_bytes / (1024 * 1024))


def _graph_error(response) -> GraphError:
    status = response.status_code
    try:
        error = (response.json() or {}).get("error") or {}
    except ValueError:
        error = {}
    detail = ""
    if isinstance(error, dict):
        detail = ": ".join(str(part) for part in (error.get("code"), error.get("message")) if part)[:200]
    if status in (401, 403):
        message = (
            f"Microsoft Graph denied access ({status}): the app has no access to this site. Check its Graph "
            "application permissions, admin consent and, for Sites.Selected, a grant on this site."
        )
    elif status == 404:
        message = "Not found in SharePoint (404)."
    elif status == 410:
        message = "The SharePoint change list expired (410)."
    else:
        message = f"Microsoft Graph request failed ({status})."
    full_message = f"{message} {detail}".strip()[:MAX_ERROR_MESSAGE_CHARS]
    return GraphError(full_message, status=status)


def _token_error(response, secret: str) -> GraphError:
    try:
        payload = response.json() or {}
    except ValueError:
        payload = {}
    lines = str(payload.get("error_description") or payload.get("error") or "").splitlines()
    detail = (lines[0] if lines else "")[:200]
    if secret:
        detail = detail.replace(secret, "…")
    full_message = f"Microsoft sign-in failed ({response.status_code}). {detail}".strip()[:MAX_ERROR_MESSAGE_CHARS]
    return GraphError(full_message, status=response.status_code)


class GraphTokenCache:
    """Access tokens per (tenant id, client id); a token is refreshed when under 5 minutes remain."""

    def __init__(self, clock: Callable[[], float] = time.time):
        self._clock = clock
        self._lock = threading.Lock()
        self._tokens: dict[tuple[str, str], tuple[str, float]] = {}

    def get(self, key: tuple[str, str]) -> Optional[str]:
        with self._lock:
            token, expires_at = self._tokens.get(key, ("", 0.0))
        return token if token and expires_at - self._clock() > TOKEN_REFRESH_MARGIN_SECONDS else None

    def put(self, key: tuple[str, str], token: str, expires_in: int) -> None:
        with self._lock:
            self._tokens[key] = (token, self._clock() + expires_in)

    def clear(self) -> None:
        with self._lock:
            self._tokens.clear()


class GraphClient:
    def __init__(
        self,
        credentials: Callable[[], Credentials],
        session,
        *,
        sleep: Callable[[float], None] = time.sleep,
        clock: Callable[[], float] = time.monotonic,
        random: Callable[[], float] = _random.random,
        token_cache: Optional[GraphTokenCache] = None,
    ):
        self._credentials = credentials
        self._session = session
        self._sleep = sleep
        self._clock = clock
        self._random = random
        self._tokens = token_cache or GraphTokenCache()

    def configured(self) -> bool:
        return all(self._credentials())

    def clear_tokens(self) -> None:
        self._tokens.clear()

    def _token(self, force: bool = False) -> str:
        tenant, client_id, secret = self._credentials()
        if not (tenant and client_id and secret):
            raise GraphCredentialsMissing(
                "SharePoint credentials are not configured. Enter the tenant ID, client ID and client secret "
                "in Settings → Connectors."
            )
        key = (tenant, client_id)
        cached = None if force else self._tokens.get(key)
        if cached:
            return cached
        try:
            response = self._session.request(
                "POST",
                TOKEN_URL.format(tenant=quote(tenant, safe="")),
                data={
                    "client_id": client_id,
                    "client_secret": secret,
                    "scope": TOKEN_SCOPE,
                    "grant_type": "client_credentials",
                },
                timeout=REQUEST_TIMEOUT,
            )
        except requests.RequestException as exc:
            raise GraphError(f"Microsoft sign-in could not be reached ({type(exc).__name__}).") from exc
        if response.status_code >= 400:
            raise _token_error(response, secret)
        try:
            payload = response.json() or {}
        except ValueError:
            payload = {}
        token = payload.get("access_token")
        if not token:
            raise GraphError("Microsoft sign-in returned no access token.")
        self._tokens.put(key, token, int(payload.get("expires_in") or 3600))
        return token

    def _backoff(self, attempt: int, retry_after: Optional[str]) -> float:
        if retry_after:
            try:
                return min(float(retry_after), MAX_RETRY_AFTER_SECONDS)
            except ValueError:
                pass
        return 2 ** attempt + self._random()

    def _get(self, url: str, *, stream: bool = False):
        attempt = 0
        refreshed = False
        force_token = False
        while True:
            token = self._token(force=force_token)
            force_token = False
            try:
                response = self._session.request(
                    "GET", url, headers={"Authorization": f"Bearer {token}"}, timeout=REQUEST_TIMEOUT, stream=stream
                )
            except (requests.ConnectionError, requests.Timeout, requests.exceptions.ChunkedEncodingError) as exc:
                if attempt >= MAX_RETRIES:
                    raise GraphError(f"Microsoft Graph could not be reached ({type(exc).__name__}).") from exc
                self._sleep(self._backoff(attempt, None))
                attempt += 1
                continue
            except requests.RequestException as exc:
                raise GraphError(f"Microsoft Graph request failed ({type(exc).__name__}).") from exc
            if response.status_code == 401 and not refreshed:
                refreshed = True
                force_token = True
                response.close()
                continue
            if response.status_code in RETRY_STATUSES and attempt < MAX_RETRIES:
                delay = self._backoff(attempt, response.headers.get("Retry-After"))
                response.close()
                self._sleep(delay)
                attempt += 1
                continue
            if response.status_code >= 400:
                error = _graph_error(response)
                response.close()
                raise error
            return response

    def get_json(self, path_or_url: str) -> dict:
        response = self._get(graph_url(path_or_url))
        try:
            return response.json()
        except ValueError:
            raise GraphError("Microsoft Graph returned a response that isn't JSON.") from None

    def iter_pages(self, path_or_url: str) -> Iterator[dict]:
        url = graph_url(path_or_url)
        while url:
            page = self.get_json(url)
            yield page
            url = page.get("@odata.nextLink") or ""

    def get_all(self, path_or_url: str) -> list[dict]:
        return [item for page in self.iter_pages(path_or_url) for item in page.get("value", [])]

    def download(self, path_or_url: str, dest: Path, *, expected_size: int, max_bytes: int, deadline_seconds: float) -> Path:
        """Stream a file to `dest` through `dest.part`, enforcing a size limit, completeness and a deadline.

        On any failure the .part file is removed and an existing `dest` is left as it was.
        """
        limit_mb = max_bytes // (1024 * 1024)
        if expected_size > max_bytes:
            raise GraphError(f"File is larger than the {limit_mb} MB limit.")
        dest = Path(dest)
        part = dest.with_name(dest.name + ".part")
        deadline = self._clock() + deadline_seconds
        try:
            dest.parent.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            raise GraphError(f"Could not save the downloaded file on the server ({type(exc).__name__}).") from exc
        response = self._get(graph_url(path_or_url), stream=True)
        written = 0
        try:
            try:
                with part.open("wb") as output:
                    for chunk in response.iter_content(chunk_size=DOWNLOAD_CHUNK_BYTES):
                        if not chunk:
                            continue
                        written += len(chunk)
                        if written > max_bytes:
                            raise GraphError(f"File is larger than the {limit_mb} MB limit.")
                        if self._clock() > deadline:
                            raise GraphError(f"Download took longer than {int(deadline_seconds)} seconds and was stopped.")
                        output.write(chunk)
            except requests.RequestException as exc:
                raise GraphError(f"Download was interrupted ({type(exc).__name__}).") from exc
            except OSError as exc:
                raise GraphError(f"Could not save the downloaded file on the server ({type(exc).__name__}).") from exc
            if written != expected_size:
                raise GraphError(f"Download was incomplete: received {written} of {expected_size} bytes.")
            try:
                os.replace(part, dest)
            except OSError as exc:
                raise GraphError(f"Could not save the downloaded file on the server ({type(exc).__name__}).") from exc
        except BaseException:
            part.unlink(missing_ok=True)
            raise
        finally:
            response.close()
        return dest
