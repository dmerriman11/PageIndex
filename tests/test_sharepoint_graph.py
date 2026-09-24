import pytest
import requests

from sharepoint_graph import GraphCredentialsMissing, GraphError, graph_url
from sharepoint_fakes import SECRET, TOKEN_URL, FakeResponse, FakeSession, make_client, token_response

SITES = graph_url("/sites/site-1")


def session_with_token(*token_responses):
    session = FakeSession()
    session.set("POST", TOKEN_URL, *(token_responses or (token_response(),)))
    return session


def test_missing_credentials_raise_a_clear_error_without_calling_out():
    session = FakeSession()
    client, _ = make_client(session, credentials=("", "", ""))

    with pytest.raises(GraphCredentialsMissing, match="Settings → Connectors"):
        client.get_json("/sites/site-1")
    assert session.calls == []


def test_the_token_is_fetched_once_and_reused():
    session = session_with_token()
    session.set("GET", SITES, FakeResponse(200, {"id": "site-1"}))
    client, _ = make_client(session)

    client.get_json("/sites/site-1")
    client.get_json("/sites/site-1")

    assert session.count("POST", TOKEN_URL) == 1
    assert session.calls[-1][2]["headers"]["Authorization"] == "Bearer token-1"


def test_a_token_near_expiry_is_refreshed():
    session = session_with_token(token_response(expires_in=200))
    session.set("GET", SITES, FakeResponse(200, {"id": "site-1"}))
    client, _ = make_client(session)

    client.get_json("/sites/site-1")
    client.get_json("/sites/site-1")

    assert session.count("POST", TOKEN_URL) == 2


def test_a_sign_in_error_never_includes_the_secret():
    session = session_with_token(FakeResponse(401, {
        "error": "invalid_client",
        "error_description": f"AADSTS7000215: Invalid client secret provided: {SECRET}\r\nTrace ID: 1",
    }))
    client, _ = make_client(session)

    with pytest.raises(GraphError) as caught:
        client.get_json("/sites/site-1")

    assert "sign-in failed (401)" in str(caught.value)
    assert SECRET not in str(caught.value)


def test_a_401_refreshes_the_token_once():
    session = session_with_token(token_response("token-1"), token_response("token-2"))
    session.set("GET", SITES, FakeResponse(401, {"error": {"code": "InvalidAuthenticationToken"}}), FakeResponse(200, {"id": "site-1"}))
    client, _ = make_client(session)

    assert client.get_json("/sites/site-1") == {"id": "site-1"}
    assert session.count("POST", TOKEN_URL) == 2
    assert session.calls[-1][2]["headers"]["Authorization"] == "Bearer token-2"


def test_a_persistent_401_fails_after_one_refresh():
    session = session_with_token()
    session.set("GET", SITES, FakeResponse(401, {"error": {"code": "InvalidAuthenticationToken"}}))
    client, _ = make_client(session)

    with pytest.raises(GraphError) as caught:
        client.get_json("/sites/site-1")

    assert caught.value.status == 401
    assert session.count("GET", SITES) == 2


def test_a_429_waits_for_retry_after():
    session = session_with_token()
    session.set("GET", SITES, FakeResponse(429, {}, {"Retry-After": "7"}), FakeResponse(200, {"id": "site-1"}))
    client, sleeps = make_client(session)

    client.get_json("/sites/site-1")

    assert sleeps == [7.0]


def test_retry_after_is_capped_at_two_minutes():
    session = session_with_token()
    session.set("GET", SITES, FakeResponse(429, {}, {"Retry-After": "600"}), FakeResponse(200, {"id": "site-1"}))
    client, sleeps = make_client(session)

    client.get_json("/sites/site-1")

    assert sleeps == [120.0]


def test_server_errors_back_off_exponentially_then_give_up():
    session = session_with_token()
    session.set("GET", SITES, FakeResponse(503, {}))
    client, sleeps = make_client(session)

    with pytest.raises(GraphError) as caught:
        client.get_json("/sites/site-1")

    assert caught.value.status == 503
    assert sleeps == [1.0, 2.0, 4.0, 8.0, 16.0]
    assert session.count("GET", SITES) == 6


def test_connection_errors_are_retried():
    session = session_with_token()
    session.set("GET", SITES, requests.ConnectionError("reset"), FakeResponse(200, {"id": "site-1"}))
    client, sleeps = make_client(session)

    assert client.get_json("/sites/site-1") == {"id": "site-1"}
    assert sleeps == [1.0]


def test_a_404_is_not_retried():
    session = session_with_token()
    session.set("GET", SITES, FakeResponse(404, {"error": {"code": "itemNotFound", "message": "gone"}}))
    client, _ = make_client(session)

    with pytest.raises(GraphError) as caught:
        client.get_json("/sites/site-1")

    assert caught.value.status == 404
    assert session.count("GET", SITES) == 1


def test_a_403_explains_what_access_is_missing():
    session = session_with_token()
    session.set("GET", SITES, FakeResponse(403, {"error": {"code": "accessDenied", "message": "Access denied"}}))
    client, _ = make_client(session)

    with pytest.raises(GraphError, match="no access to this site"):
        client.get_json("/sites/site-1")


def test_pages_are_followed_through_next_links():
    session = session_with_token()
    first = graph_url("/sites/site-1/drives")
    second = graph_url("/sites/site-1/drives?$skiptoken=2")
    session.set("GET", first, FakeResponse(200, {"value": [{"id": "a"}], "@odata.nextLink": second}))
    session.set("GET", second, FakeResponse(200, {"value": [{"id": "b"}]}))
    client, _ = make_client(session)

    assert [item["id"] for item in client.get_all("/sites/site-1/drives")] == ["a", "b"]


def test_clear_tokens_forces_a_new_sign_in():
    session = session_with_token()
    session.set("GET", SITES, FakeResponse(200, {"id": "site-1"}))
    client, _ = make_client(session)
    client.get_json("/sites/site-1")

    client.clear_tokens()
    client.get_json("/sites/site-1")

    assert session.count("POST", TOKEN_URL) == 2


def test_error_messages_are_capped_at_300_characters():
    session = session_with_token()
    long_detail = "x" * 500
    session.set("GET", SITES, FakeResponse(403, {"error": {"code": "accessDenied", "message": long_detail}}))
    client, _ = make_client(session)

    with pytest.raises(GraphError) as caught:
        client.get_json("/sites/site-1")

    error_msg = str(caught.value)
    assert len(error_msg) <= 300
    assert "no access to this site" in error_msg


# ── downloads ─────────────────────────────────────────────────────────────────

from sharepoint_graph import download_deadline_seconds

CONTENT_URL = graph_url("/drives/d/items/i/content")
MB = 1024 * 1024


def download_session(content, status=200):
    session = session_with_token()
    session.set("GET", CONTENT_URL, FakeResponse(status, {} if status >= 400 else None, content=content))
    return session


def test_download_writes_the_file_and_leaves_no_part_file(tmp_path):
    client, _ = make_client(download_session(b"hello"))
    dest = tmp_path / "doc.pdf"

    client.download("/drives/d/items/i/content", dest, expected_size=5, max_bytes=MB, deadline_seconds=60)

    assert dest.read_bytes() == b"hello"
    assert list(tmp_path.iterdir()) == [dest]


def test_an_oversized_file_is_refused_before_downloading(tmp_path):
    session = download_session(b"x" * 10)
    client, _ = make_client(session)

    with pytest.raises(GraphError, match="limit"):
        client.download("/drives/d/items/i/content", tmp_path / "doc.pdf", expected_size=2 * MB, max_bytes=MB, deadline_seconds=60)
    assert session.count("GET", CONTENT_URL) == 0


def test_a_stream_that_passes_the_limit_is_aborted(tmp_path):
    client, _ = make_client(download_session(b"x" * (2 * MB)))

    with pytest.raises(GraphError, match="limit"):
        client.download("/drives/d/items/i/content", tmp_path / "doc.pdf", expected_size=10, max_bytes=MB, deadline_seconds=60)
    assert list(tmp_path.iterdir()) == []


def test_an_incomplete_download_keeps_the_previous_file(tmp_path):
    dest = tmp_path / "doc.pdf"
    dest.write_bytes(b"previous")
    client, _ = make_client(download_session(b"hel"))

    with pytest.raises(GraphError, match="incomplete"):
        client.download("/drives/d/items/i/content", dest, expected_size=5, max_bytes=MB, deadline_seconds=60)
    assert dest.read_bytes() == b"previous"
    assert list(tmp_path.iterdir()) == [dest]


def test_a_download_past_its_deadline_is_stopped(tmp_path):
    ticks = iter([0.0, 1000.0, 1000.0, 1000.0])
    client, _ = make_client(download_session(b"x" * (2 * MB)), clock=lambda: next(ticks))

    with pytest.raises(GraphError, match="longer than 60 seconds"):
        client.download("/drives/d/items/i/content", tmp_path / "doc.pdf", expected_size=2 * MB, max_bytes=4 * MB, deadline_seconds=60)
    assert list(tmp_path.iterdir()) == []


def test_download_deadline_is_60s_plus_1s_per_mb_capped_at_10_minutes():
    assert download_deadline_seconds(0) == 60
    assert download_deadline_seconds(200 * MB) == 260
    assert download_deadline_seconds(2000 * MB) == 600
