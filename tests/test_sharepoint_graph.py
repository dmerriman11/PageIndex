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
