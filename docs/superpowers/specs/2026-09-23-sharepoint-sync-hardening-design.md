# SharePoint sync hardening: design

**Date:** 2026-09-23
**Status:** Approved. Implementation plans: `docs/superpowers/plans/2026-09-23-sharepoint-sync-hardening-{engine,frontend}.md`
**Repos:** engine `dmerriman11/PageIndex` (this repo), frontend `dmerriman11/lemur-pageindex`
**Builds on:** engine PR #2 and frontend PR #3 (retrieval settings), which touch the same settings files and merge first.

## Goal

SharePoint sync (commit `4a3905e`, merged to engine `main`) works on the happy path. This
work makes it safe and dependable enough to point at a real tenant:

1. The client secret is never stored in plaintext and never readable through the API.
2. Only admins can change connector credentials or point a library at a SharePoint site.
3. A sync never skips a file permanently, never syncs the wrong target, and never
   re-downloads a whole library because of a rename or a transient error.
4. The frontend exposes all of it in the NOVA design on `master`.

## Out of scope

- The Dokploy/Nixpacks deploy commits on the frontend `SharePoint-Sync` branch. They stay on
  that branch for a later change.
- Webhooks (Graph change notifications). Polling stays the only trigger.
- Delegated (per-user) auth, and SharePoint permissions mirrored into query results. The
  connector uses one app-only identity, and every synced document is visible to anyone who
  can query the library.
- OCR for image-only PDFs.

## Current state (what the survey found)

| Area | Problem |
|---|---|
| Secret storage | `SHAREPOINT_CLIENT_SECRET` lives in `.env`, which `PATCH /api/admin/sharepoint-config` rewrites without an atomic write. |
| Credential updates | PATCH replaces every field, so a field the client omits is wiped. |
| Access | `POST /api/libraries/sharepoint/test`, and SharePoint fields on `POST /api/libraries` and `PATCH /api/libraries/{id}`, accept any API key, not only admin keys. The frontend proxy routes on the `SharePoint-Sync` branch don't call `requireDashboardSession`. |
| Target changes | Changing `siteUrl`, `driveId`/`driveName` or `folderPath` keeps the cached `siteId`, `driveId`, `rootItemId` and `deltaLink`, so the next sync still reads the old target. |
| Failed items | `_sync_library_sharepoint` stores the new `deltaLink` even when some items failed, so those items are never offered again. |
| Change detection | The fingerprint uses the eTag, which changes on rename and metadata edits, so those re-download and re-index the file. |
| Error handling | Any delta error, including 429 throttling and 5xx, falls back to a full scan. There are no retries, no backoff and no `Retry-After` handling. |
| Downloads | Files download straight to their final path with no size limit and no overall timeout. |
| Paging | `/sites/{id}/drives` is read as a single page. |
| Status | `lastConnectionError` is never set. |
| Concurrency | If settings change while a sync runs, the sync writes results for the old target. |
| Scoping | Folder-scoped delta (`/drives/{id}/items/{folder}/delta`) is only supported on OneDrive for Business and SharePoint for some tenants, and `_sharepoint_item_relative_path` depends on `parentReference.path`, which delta responses omit. Neither is verified. |
| Tests | `tests/test_sharepoint_sync.py` mocks `_resolve_sharepoint_source` and `_iter_sharepoint_delta_items`, so the Graph client code has no coverage. |

## Design

### 1. Credentials and access

**Storage.** Add a `connectors.sharepoint` section to `AppSettings` (`app_settings.py`), next to
`ai` and `retrieval`:

| Stored key | API name | Env fallback | Notes |
|---|---|---|---|
| `tenant_id` | `tenantId` | `SHAREPOINT_TENANT_ID` | GUID or `*.onmicrosoft.com` domain |
| `client_id` | `clientId` | `SHAREPOINT_CLIENT_ID` | GUID |
| `client_secret` | `clientSecret` | `SHAREPOINT_CLIENT_SECRET` | Fernet-encrypted with the same key as the AI provider keys; never returned |

Resolution order is the same as retrieval settings: saved value, then env var, then empty.
The view reports each field's `source` (`saved`, `env` or `default`), plus
`clientSecretSet` and `clientSecretMasked` (last 4 characters), and `configured` when all
three resolve.

The engine stops writing to `.env`. `_refresh_sharepoint_env_values` and the module-level
`SHAREPOINT_*` globals are replaced by a single `sharepoint_credentials()` lookup on
`APP_SETTINGS`, read when a token is requested.

**API.** A new router in `settings_api.py`,
`create_sharepoint_settings_router(settings, admin_dependency, on_change)`:

- `GET /api/settings/sharepoint` returns the view above.
- `PATCH /api/settings/sharepoint` takes `tenantId`, `clientId`, `clientSecret` (all optional)
  and `clearClientSecret: bool`. Only fields that are sent are changed. Sending `clientSecret`
  together with `clearClientSecret: true` is a 400. Values are trimmed; an empty string clears a field.
- `on_change` clears the token cache when any credential changes.

`GET/PATCH /api/admin/sharepoint-config` stay for one release, as thin aliases of the new
routes, then are removed. Nothing on the frontend's `master` calls them.

**Access control.**

- `POST /api/libraries/sharepoint/test` moves to `require_admin_api_key`.
- `POST /api/libraries` and `PATCH /api/libraries/{id}` still accept any API key, but
  return 403 when the body sets `syncSourceType: "sharepoint"` or any `sharePoint` field and
  the key is not an admin key. A shared check, `_require_admin_for_sharepoint(req, key)`,
  runs in both handlers.
- Frontend: every SharePoint proxy route calls `requireDashboardSession` and forwards the
  admin key, the same as `app/api/settings/retrieval/route.ts`.

**Token cache.** A small `GraphTokenCache` class with a `threading.Lock`, keyed by
`(tenant_id, client_id)` so a credential change can't reuse an old token. A token is refreshed
when it expires in under 5 minutes.

### 2. Sync correctness

**Target identity.** A library's SharePoint target is
`(siteUrl, driveId or driveName, folderPath)`. When `PATCH /api/libraries/{id}` changes any part of it:

- `siteId`, `driveId` (when resolved from a name), `rootItemId` and `deltaLink` are cleared.
- A `targetVersion` counter on the library's `sharePoint` settings is incremented.
- The next sync resolves the target from scratch.

Documents from the old target are handled like documents deleted at the source: on the next
full scan they no longer appear and are removed. The edit dialog warns that this will happen
before it saves.

**Stale syncs.** `_sync_library_sharepoint` reads `targetVersion` when it starts. Before each
write to the library (document upserts, deletions, delta link, status), it re-checks the
version under the library lock. If the version changed, it stops, writes nothing further, and
records `lastSyncResult: "superseded"`. The next poll then syncs the new target.

**Failed items.** Add `sharePoint.pendingItems`, a map of item ID to
`{name, attempts, lastError, lastAttemptAt}`:

- A download or index failure adds the item, or increments its `attempts`.
- Each sync first retries every pending item by fetching `/drives/{driveId}/items/{itemId}`,
  then applies the delta page.
- On success the item leaves the map. Items that are gone at the source (404) leave the map,
  and their document is removed.
- After 5 attempts an item stays in the map, but is only retried on a full scan or a manual
  sync, and its document shows `status: "error"` with the last error.
- Documents already in `status: "error"` when this ships go into `pendingItems` on the first sync.

The delta link advances once every item in the page has either succeeded or been recorded in
`pendingItems`. Because failures are persisted, advancing the link no longer loses them. If
the process dies mid-page, the link hasn't advanced and the page is replayed. Replays are
idempotent, because an unchanged fingerprint is a no-op.

**Change detection.** The fingerprint becomes `(cTag, size)`: the content tag changes only when
the file's content changes. When the fingerprint is unchanged but `name` or the parent folder
changed, only the document's metadata is updated (`name`, `sharePointWebUrl`, relative path).
The file is not downloaded or re-indexed. eTag is still stored for display, but ignored for
change detection.

**Full rescans.** A full scan runs only when:

- the library has no delta link yet,
- Graph returns 410 Gone, which includes `resyncRequired`, or
- the user presses **Full resync**, a new `?full=true` on `POST /api/libraries/{id}/sync`.

A full scan lists every item under the target. Documents whose item ID isn't in the listing are
removed. 429 and 5xx responses never trigger a full scan: they retry (section 3), and if the
retries run out, the sync fails with the delta link unchanged.

**Folder scoping.** The folder-scoped delta endpoint is not supported on every tenant. The
first live-tenant step (section 5) checks whether
`/drives/{driveId}/items/{folderId}/delta` works on your tenant:

- **If it works:** keep it. Store `scopeMode: "folder"`.
- **If it doesn't** (400/404/501): fall back to drive-level delta, and keep only items inside
  the folder. Store `scopeMode: "drive"`.

The fallback checks whether an item is inside the folder using a persisted
`sharePoint.folderIndex` map, holding each folder's item ID, `parentId` and `name`. Every folder
item in the delta updates it. An item is in scope when following `parentId` reaches
`rootItemId`. The relative path is built from the same chain, so it no longer depends on
`parentReference.path`, which delta responses omit. The mode is chosen once per target, and is
reset when the target changes.

### 3. Robustness

**Graph client.** Move the Graph calls out of `api_server.py` into a new `sharepoint_graph.py`,
so they can be tested on their own:

- `GraphClient(credentials, session, sleep=time.sleep, clock=time.monotonic)` with
  `get(url)`, `iter_pages(url)`, `download(url, dest, expected_size, max_bytes, deadline)`.
- **Timeouts:** 10 s to connect and 60 s to read, per request.
- **Retries:** 429, 503, 504, and connection or read errors retry up to 5 times with
  exponential backoff (1, 2, 4, 8, 16 s, with jitter). A `Retry-After` header, up to 120 s,
  replaces the backoff delay. A 401 refreshes the token once, then fails.
- **Paging:** `iter_pages` follows `@odata.nextLink`. `/sites/{id}/drives` uses it.

**Downloads.**

- Each file streams to `<dest>.part` in 1 MB chunks, then is renamed into place with
  `os.replace`.
- The download is refused before it starts when Graph's `size` is over
  `PAGEINDEX_SHAREPOINT_MAX_FILE_MB` (default 200). It is aborted if the stream passes the
  limit, or if the written size doesn't match `size`.
- An overall deadline applies to each file: 10 minutes, or 60 s plus 1 s per MB.
- Oversized and failed files go into `pendingItems` (section 2) with a readable error.

**Status and logging.**

- `lastConnectionError` is set to a short, secret-free message whenever the token or site
  lookup fails, and cleared on the next success.
- Every sync logs one line when it starts and one when it ends, with the library ID, mode
  (delta or full), counts of added, updated, renamed, removed, failed and skipped items, and
  duration.
- Logs contain item IDs and file names, but never tokens, secrets or file contents.

**Source switching.** A library switched between `folder` and `sharepoint` keeps its documents
from the old source only if the request sets `keepExistingDocuments: true`. Otherwise those
documents are removed when the switch is saved. The frontend asks the user which they want.

**Root-site URLs.** `_sharepoint_url_parts` accepts a root site
(`https://tenant.sharepoint.com` and `https://tenant.sharepoint.com/Shared Documents/...`),
using the Graph path `/sites/{host}` instead of `/sites/{host}:/{path}`.

### 4. Frontend (`lemur-pageindex`, branched from `master`)

Port only the SharePoint UI from `30793cc` on `origin/SharePoint-Sync`, rebuilt with the NOVA
components on `master`, not merged:

- **Settings → Connectors tab**, replacing a mock tab: tenant ID, client ID, and a
  write-only secret field showing "Secret set (…abcd)" with **Replace** and **Clear** actions.
  It has "from .env" badges, a **Test connection** button, and its own **Save** button that
  sends only changed fields, following `retrieval-settings-tab.tsx`.
- **Library create/edit:** a *Source* choice (Upload only, Local folder or SharePoint). The
  SharePoint fields are site URL, a document-library picker filled by the test call, a folder
  path and a polling interval. It warns before a target or source change and asks the
  keep-or-remove question.
- **Library detail:** sync status (last sync time, mode, counts), `lastConnectionError`, and
  the pending-items list with its errors. Adds **Sync now** and **Full resync** buttons.
- **Proxy routes:** `app/api/settings/sharepoint/route.ts` and
  `app/api/libraries/sharepoint/test/route.ts`, both gated by `requireDashboardSession`.
- The Connectors tab and SharePoint source options are hidden from non-admin sessions.

### 5. Tests and verification

**Engine unit tests.** HTTP-level mocks using the `responses` library, added as a dev
dependency. No mocking of the engine's own functions.

- `tests/test_sharepoint_graph.py`:
  - token fetch, caching and refresh
  - 401 refresh
  - 429 with `Retry-After`
  - 5xx backoff and giving up
  - paging
  - download (size limit, size mismatch, `.part` cleanup, deadline)
- `tests/test_sharepoint_settings.py`:
  - encrypted storage, and the `.env` fallback
  - partial PATCH, clear, and the 400 when `clientSecret` and `clearClientSecret` are both sent
  - the admin-only 403s
  - the token cache is cleared when credentials change
- `tests/test_sharepoint_sync.py` (rewritten against mocked Graph):
  - initial full scan, then delta add, update, rename and delete
  - 410 leads to a full scan that removes missing documents
  - 429 retries without a full scan
  - failed items are retried on the next sync, and dropped on 404
  - a target change resets state, and a stale sync is superseded
  - `scopeMode: "drive"` filtering and relative paths
  - source switching with keep and remove

**Frontend.** `npm run build`, then a browser check of the Connectors tab and the library
source flows against the local engine.

**Live tenant** (after you enter the credentials in the app; I never see the secret):

1. Test connection from the Connectors tab.
2. Check folder-scoped delta, and record `scopeMode`.
3. Initial sync of a small test library.
4. In SharePoint: add a file, edit it, rename it, move it within the folder, delete it. Sync
   after each step, and check that only the edit re-indexes.
5. Change the library's folder, and check that the old documents go and the new ones arrive.
6. Query a synced document from Chat.

### 6. Docs

- `.env.example`: `SHAREPOINT_TENANT_ID`, `SHAREPOINT_CLIENT_ID`,
  `SHAREPOINT_CLIENT_SECRET` (marked as a fallback; prefer the Connectors tab),
  `PAGEINDEX_SHAREPOINT_MAX_FILE_MB`.
- README section **SharePoint sync**:
  - The Entra ID app registration: a client secret, and **application** permissions.
    `Sites.Selected` is recommended, plus a per-site grant; `Sites.Read.All` or
    `Files.Read.All` also work. Admin consent is required.
  - Where the secret is stored and how to rotate it.
  - How polling, delta sync, full resync and pending items work.
  - Which file types sync, the size limit, and what the user sees when a file fails.

## Data model changes

The library `folderMonitor.sharePoint` object gains `targetVersion` (int, default 0),
`scopeMode` (`"folder" | "drive" | ""`), `pendingItems` (map) and `folderIndex` (map), plus
`lastSyncResult` and `lastSyncCounts` on `folderMonitor`. `_normalize_sharepoint_settings`
fills defaults, so existing `_libraries.json` files load unchanged. `folderIndex` is only
populated in drive scope mode.

## Delivery

1. Merge engine PR #2 and frontend PR #3; rebase `feature/sharepoint-hardening` onto `main`.
2. Engine PR: sections 1–3, the engine tests, and section 6.
3. Frontend PR: section 4.
4. Live-tenant verification (section 5), with fixes as follow-up commits on the same PRs
   before merge.

## Risks

- **Folder-scoped delta** behaviour varies by tenant. The drive-scope fallback covers this,
  but on a very large drive it reads every change in the drive. That volume is acceptable
  for polling at 5-minute intervals or longer.
- **`Sites.Selected`** needs a per-site grant (`POST /sites/{id}/permissions`) made by a
  tenant admin. The README gives the exact call. The connection test reports a 403 as
  "app has no access to this site".
- **Stored PII.** Synced documents may contain borrower PII. They are stored the same way as
  uploads today, and no new vendor receives them.

## Revisions during planning

Planning against the code turned up these changes to the approved design. The plans implement the revised version.

1. **No new test dependency.** `GraphClient` takes its HTTP session as a parameter, so tests pass a scripted fake session (`tests/sharepoint_fakes.py`) instead of adding `responses`. Tests still exercise the real client code, and nothing is downloaded.
2. **Folder index in both scope modes, stored beside the library.** Delta responses omit `parentReference.path` in folder scope too, so paths always come from the folder index. It lives in `workspace/_sharepoint/<library_id>.json`, tagged with `targetVersion`, rather than in the library record. That keeps library API responses small; on a large drive the index can hold thousands of folders.
3. **Folder renames update the paths of files inside them.** Those files don't appear in the delta, so after each delta sync the paths of existing documents are re-derived from the index. A file whose folder moved out of scope is removed.
4. **No mass re-index after upgrading.** Change detection compares the stored content tag and size with the item's, instead of comparing the old fingerprint hash, so existing documents aren't re-downloaded. The first sync after upgrading is a full scan, because no folder index exists yet.
5. **Sync status reuses `folderMonitor.lastResult`.** It gains `mode`, `renamed`, `skipped`, `pendingCount`, `outcome` and `durationSeconds`, instead of new `lastSyncResult`/`lastSyncCounts` fields; the frontend already reads `lastResult`.
6. **Failed files are shown from the retry list.** An existing document keeps its last good content when a re-download fails. The library page lists pending files with their errors, rather than flipping each document to `error`. Indexing failures still mark the document `error`, as today.
7. **After a superseded sync, the new target syncs straight away** rather than waiting for the next poll.
8. **No role gating in the dashboard.** The dashboard has no admin and non-admin sessions: every signed-in session reaches the engine as an admin through the proxies. The engine enforces admin-only access for API keys; the UI doesn't hide anything.
9. **The connection test also returns the site's document libraries**, which the library picker uses.
10. **The download deadline is `min(600, 60 + size in MB)` seconds.**
