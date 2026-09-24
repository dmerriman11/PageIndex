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


# ── A fake SharePoint site ────────────────────────────────────────────────────

SITE_URL = "https://contoso.sharepoint.com/sites/team"
SITE = {"id": "site-1", "webUrl": SITE_URL}
DRIVE = {"id": "drive-1", "name": "Documents", "webUrl": f"{SITE_URL}/Shared%20Documents"}
OTHER_DRIVE = {"id": "drive-2", "name": "Archive", "webUrl": f"{SITE_URL}/Archive"}
SCOPE_ROOT = {"id": "folder-1", "name": "Amerihome", "folder": {"childCount": 1}, "parentReference": {"id": "root-1", "driveId": "drive-1"}}
FOLDER_DELTA = "/drives/drive-1/items/folder-1/delta"
DRIVE_DELTA = "/drives/drive-1/root/delta"


def folder_item(item_id, name, parent_id):
    return {"id": item_id, "name": name, "folder": {"childCount": 0}, "parentReference": {"id": parent_id, "driveId": "drive-1"}}


def deleted_item(item_id, parent_id="folder-1"):
    return {"id": item_id, "deleted": {"state": "deleted"}, "parentReference": {"id": parent_id, "driveId": "drive-1"}}


class FakeSharePoint:
    """Site "team" with document libraries "Documents" (drive-1) and "Archive"; the synced folder is Amerihome (folder-1)."""

    def __init__(self):
        self.session = FakeSession()
        self.session.set("POST", TOKEN_URL, token_response())
        self._links = 0
        self.get("/sites/contoso.sharepoint.com:/sites/team", SITE)
        self.get("/sites/site-1", SITE)
        self.get("/sites/site-1/drives", {"value": [DRIVE, OTHER_DRIVE]})
        self.get("/drives/drive-1", DRIVE)
        self.get("/drives/drive-1/root:/Amerihome", SCOPE_ROOT)
        self.get("/drives/drive-1/items/folder-1/children?$top=5", {"value": []})

    def get(self, path, body):
        self.session.set("GET", graph_url(path), FakeResponse(200, body))

    def fail(self, path, status, headers=None):
        self.session.set("GET", graph_url(path), FakeResponse(status, {"error": {"code": "failed", "message": "failed"}}, headers))

    def add_file(self, item_id, name="Guide.pdf", parent_id="folder-1", content=b"%PDF-1.4 guide", ctag="c1", etag="e1"):
        """Serve the file's content and item metadata; returns the driveItem as delta would list it."""
        item = {
            "id": item_id,
            "name": name,
            "size": len(content),
            "cTag": ctag,
            "eTag": etag,
            "lastModifiedDateTime": "2026-09-20T12:00:00Z",
            "webUrl": f"{SITE_URL}/Shared%20Documents/{name}",
            "file": {"mimeType": "application/octet-stream"},
            "parentReference": {"id": parent_id, "driveId": "drive-1"},
        }
        self.session.set("GET", graph_url(f"/drives/drive-1/items/{item_id}/content"), FakeResponse(200, content=content))
        self.get(f"/drives/drive-1/items/{item_id}", item)
        return item

    def changes(self, items, at=FOLDER_DELTA):
        """Serve `items` as one delta page at `at` (a start path or a previous delta link); returns the next delta link."""
        self._links += 1
        link = f"{GRAPH_BASE_URL}/delta-links/{self._links}"
        self.session.set("GET", graph_url(at), FakeResponse(200, {"value": list(items), "@odata.deltaLink": link}))
        return link
