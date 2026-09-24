import api_server as api
from sharepoint_graph import GraphClient


def routes():
    return {(route.path, method) for route in api.app.routes for method in getattr(route, "methods", set())}


def test_sharepoint_settings_routes_are_mounted():
    assert ("/api/settings/sharepoint", "GET") in routes()
    assert ("/api/settings/sharepoint", "PATCH") in routes()
    assert ("/api/admin/sharepoint-config", "PATCH") in routes()


def test_the_graph_client_reads_credentials_from_app_settings():
    assert isinstance(api.SHAREPOINT_GRAPH, GraphClient)
    assert api.SHAREPOINT_GRAPH._credentials == api.APP_SETTINGS.sharepoint_credentials


def test_the_engine_no_longer_writes_env_files_or_keeps_credential_globals():
    for name in ("_write_env_values", "SHAREPOINT_CLIENT_SECRET", "SHAREPOINT_TOKEN_CACHE", "_graph_get_json"):
        assert not hasattr(api, name), name


def test_short_errors_are_trimmed():
    assert api._short_error(ValueError("x" * 500)) == "x" * 300
    assert api._short_error(ValueError()) == "ValueError"
