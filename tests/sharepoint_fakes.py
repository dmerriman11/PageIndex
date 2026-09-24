"""Microsoft Graph faked at the HTTP boundary: a requests.Session stand-in with scripted responses.

GraphClient takes its session as a parameter, so tests exercise the real client code
(tokens, retries, paging, downloads) without the network.
"""
from sharepoint_graph import GRAPH_BASE_URL, GraphClient, graph_url

TENANT, CLIENT_ID, SECRET = "tenant-1", "client-1", "s3cret-value-abcd"
TOKEN_URL = f"https://login.microsoftonline.com/{TENANT}/oauth2/v2.0/token"


class FakeResponse:
    def __init__(self, status=200, body=None, headers=None, content=b""):
        self.status_code = status
        self._body = body
        self.headers = headers or {}
        self._content = content
        self.closed = False

    def json(self):
        if self._body is None:
            raise ValueError("no JSON body")
        return self._body

    def iter_content(self, chunk_size=1):
        for start in range(0, len(self._content), chunk_size):
            yield self._content[start:start + chunk_size]

    def close(self):
        self.closed = True


class FakeSession:
    """Serves scripted responses per (method, url): each queued response is used once, the last one repeats.

    A queued Exception instance is raised instead of returned.
    """

    def __init__(self):
        self._routes = {}
        self.calls = []

    def set(self, method, url, *responses):
        self._routes[(method, url)] = list(responses)

    def request(self, method, url, **kwargs):
        self.calls.append((method, url, kwargs))
        queue = self._routes.get((method, url))
        if not queue:
            raise AssertionError(f"Unexpected request: {method} {url}")
        response = queue.pop(0) if len(queue) > 1 else queue[0]
        if isinstance(response, Exception):
            raise response
        return response

    def count(self, method, url):
        return sum(1 for call in self.calls if call[0] == method and call[1] == url)


def token_response(token="token-1", expires_in=3600):
    return FakeResponse(200, {"access_token": token, "expires_in": expires_in, "token_type": "Bearer"})


def make_client(session, credentials=(TENANT, CLIENT_ID, SECRET), clock=None):
    """A real GraphClient on a fake session; returns (client, list of sleep durations)."""
    sleeps = []
    extra = {"clock": clock} if clock else {}
    client = GraphClient(lambda: credentials, session, sleep=sleeps.append, random=lambda: 0.0, **extra)
    return client, sleeps
