# SharePoint Sync Hardening: Frontend Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Expose the hardened SharePoint connector in the Lemur dashboard: a Connectors settings tab, SharePoint as a library source (create and edit), and sync status with pending files and a full resync. Then verify end to end on the user's tenant.

**Architecture:** Three new self-contained components (`SharePointSettingsTab`, `SharePointSourceFields`, `SyncStatusPanel`) do the work; the two existing library pages only gain state, payload fields and placements. Browser code calls Next.js route handlers, which proxy to the engine and require a dashboard session. The old `SharePoint-Sync` branch UI (commit `30793cc`) is used only as a reference; nothing is merged from it.

**Tech Stack:** Next.js 16 App Router, React 19, Tailwind v4 with NOVA tokens, lucide-react, existing `components/ui/*`.

**Spec:** `pageindex-engine/docs/superpowers/specs/2026-09-23-sharepoint-sync-hardening-design.md`, section 4 and **Revisions during planning**.

**Depends on:** the engine plan `2026-09-23-sharepoint-sync-hardening-engine.md` (API shapes below). Run the engine from `feature/sharepoint-hardening` while working on this.

## Global Constraints

- Repo `dmerriman11/lemur-pageindex`, working directory `lemur-pageindex/`. Branch `feat/sharepoint-connectors` from `master` once frontend PR #3 (`feat/retrieval-settings`) is merged: `git fetch origin && git switch -c feat/sharepoint-connectors origin/master`. If PR #3 isn't merged yet, branch from `feat/retrieval-settings` and rebase later.
- `AGENTS.md`: this Next.js version differs from training data. Before writing a route handler, read `node_modules/next/dist/docs/` for route handlers and follow the existing `app/api/settings/retrieval/route.ts` pattern.
- There is no test runner or linter. Validation is `npm run build` after every task, plus the browser checks in Task 6.
- NOVA visual language (`AGENTS.md`): cards 8px radius, controls 2px; terracotta `primary` is the only action colour; `font-display`/`font-ui` utilities; notices via `noticeVariants({ tone })`; badges via `Badge`/`badgeVariants`. Match `components/settings/retrieval-settings-tab.tsx`.
- Every new route handler calls `requireDashboardSession(req)` first and forwards `X-API-Key` when present.
- The dashboard has no admin/non-admin roles; every signed-in session acts as admin through the proxies. Do not add role gating.
- Claude never types credentials. The client secret field is filled by the user only. Secret inputs are `type="password"` with `autoComplete="new-password"`, and the secret is never logged or kept after a save.
- Stage files by path. Commit after each task.

## Engine API used (from the engine plan)

| Call | Body / query | Response |
|---|---|---|
| `GET /api/settings/sharepoint` | — | `SharePointSettings` |
| `PATCH /api/settings/sharepoint` | any of `tenantId`, `clientId`, `clientSecret`, `clearClientSecret: true` | `SharePointSettings`; 400 `{detail}` on invalid input |
| `POST /api/libraries/sharepoint/test` | `{siteUrl, driveId?, driveName?, folderPath?}` | `SharePointTestResult`; 400 `{detail}` |
| `POST /api/libraries` / `PATCH /api/libraries/{id}` | adds `syncSourceType`, `sharePointSiteUrl`, `sharePointDriveId`, `sharePointDriveName`, `sharePointFolderPath`, `keepExistingDocuments` (PATCH) | library |
| `POST /api/libraries/{id}/sync?full=true` | — | `{status, message, folderMonitor}` |
| library `folderMonitor` | `sourceType`, `sharePoint.{siteUrl, driveId, driveName, folderPath, scopeMode, lastConnectedAt, lastConnectionError, pendingItems}`, `lastResult.{mode, renamed, skipped, pendingCount, outcome, durationSeconds}` | |

## File Structure

| File | Status | Responsibility |
|---|---|---|
| `lib/api-client.ts` | modify | SharePoint types |
| `lib/server/ai-settings-proxy.ts` | modify | allow the `sharepoint` settings section |
| `app/api/settings/sharepoint/route.ts` | create | GET/PATCH proxy |
| `app/api/libraries/sharepoint/test/route.ts` | create | connection test proxy |
| `app/api/libraries/[id]/sync/route.ts` | modify | pass `?full=true` |
| `components/settings/sharepoint-settings-tab.tsx` | create | Connectors tab |
| `app/(dashboard)/settings/page.tsx` | modify | add the Connectors tab |
| `components/libraries/sharepoint-source-fields.tsx` | create | site URL, library picker, folder |
| `components/libraries/sync-status-panel.tsx` | create | sync state, connection error, pending files, full resync |
| `app/(dashboard)/libraries/[id]/page.tsx` | modify | source choice, save flow, dialogs, status, document source |
| `app/(dashboard)/libraries/page.tsx` | modify | SharePoint in the create dialog |

---

### Task 1: Types and proxy routes

**Files:**
- Modify: `lib/api-client.ts`, `lib/server/ai-settings-proxy.ts`, `app/api/libraries/[id]/sync/route.ts`
- Create: `app/api/settings/sharepoint/route.ts`, `app/api/libraries/sharepoint/test/route.ts`

**Interfaces:**
- Produces (exported from `lib/api-client.ts`): `SharePointSettings`, `SharePointDrive`, `SharePointTestResult`, `SharePointPendingItem`, `SharePointSourceSettings`.

- [ ] **Step 1: Types**

In `lib/api-client.ts`, after the `RetrievalSettings` interface, add:

```ts
// ─── SharePoint connector ─────────────────────────────────────────────────────

export interface SharePointSettings {
  tenantId: string;
  clientId: string;
  clientSecretSet: boolean;
  clientSecretMasked: string | null;
  configured: boolean;
  needsReentry: boolean;
  sources: Record<"tenantId" | "clientId" | "clientSecret", SettingSource>;
}

export interface SharePointDrive {
  id: string;
  name: string;
}

export interface SharePointTestResult {
  status: "ok";
  siteId: string;
  driveId: string;
  driveName: string;
  folderPath: string;
  rootItemId: string;
  sampleSupportedFiles: number;
  drives: SharePointDrive[];
}

export interface SharePointPendingItem {
  name: string;
  attempts: number;
  lastError: string | null;
  lastAttemptAt: string | null;
}

export interface SharePointSourceSettings {
  siteUrl: string;
  driveId: string;
  driveName: string;
  folderPath: string;
  scopeMode: "folder" | "drive" | "";
  lastConnectedAt: string | null;
  lastConnectionError: string | null;
  pendingItems: Record<string, SharePointPendingItem>;
}
```

- [ ] **Step 2: Settings proxy section**

In `lib/server/ai-settings-proxy.ts`, change the `section` parameter type of `proxyEngineSettings` from `"ai" | "retrieval"` to `"ai" | "retrieval" | "sharepoint"`. Nothing else changes: the function already forwards `X-API-Key`, disables retries for mutations and never logs bodies.

- [ ] **Step 3: Settings route**

Create `app/api/settings/sharepoint/route.ts`:

```ts
import { NextRequest } from "next/server";
import { proxyEngineSettings } from "@/lib/server/ai-settings-proxy";
import { requireDashboardSession } from "@/lib/server/dashboard-session";

// The PATCH body can carry the SharePoint client secret; it is passed through and never logged.
export async function GET(req: NextRequest) {
  const denied = await requireDashboardSession(req);
  if (denied) return denied;

  return proxyEngineSettings(req, "sharepoint", "");
}

export async function PATCH(req: NextRequest) {
  const denied = await requireDashboardSession(req);
  if (denied) return denied;

  return proxyEngineSettings(req, "sharepoint", "", { method: "PATCH", body: await req.text() });
}
```

- [ ] **Step 4: Connection test route**

Create `app/api/libraries/sharepoint/test/route.ts`:

```ts
import { NextRequest, NextResponse } from "next/server";
import { BACKEND_API_URL, fetchBackendWithRetry } from "@/lib/server/backend-proxy";
import { requireDashboardSession } from "@/lib/server/dashboard-session";

export async function POST(req: NextRequest) {
  const denied = await requireDashboardSession(req);
  if (denied) return denied;

  const headers: Record<string, string> = { "Content-Type": "application/json" };
  const inboundApiKey = req.headers.get("x-api-key");
  if (inboundApiKey) {
    headers["X-API-Key"] = inboundApiKey;
  }

  try {
    const res = await fetchBackendWithRetry(
      `${BACKEND_API_URL}/api/libraries/sharepoint/test`,
      { method: "POST", headers, body: await req.text() },
      // Each attempt calls Microsoft Graph, which already retries; don't multiply that here.
      { attempts: 1 },
    );
    const data = await res.json();
    return NextResponse.json(data, { status: res.status });
  } catch {
    return NextResponse.json(
      { detail: "Backend unavailable. Start the PageIndex API server on port 7777." },
      { status: 503 },
    );
  }
}
```

(The static `sharepoint` segment takes precedence over `app/api/libraries/[id]`.)

- [ ] **Step 5: Full resync pass-through**

In `app/api/libraries/[id]/sync/route.ts`, replace the `fetchBackendWithRetry(...)` call with:

```ts
    const full = req.nextUrl.searchParams.get("full") === "true";
    const res = await fetchBackendWithRetry(`${BACKEND_API_URL}/api/libraries/${id}/sync${full ? "?full=true" : ""}`, {
      method: "POST",
      headers,
    });
```

- [ ] **Step 6: Build**

Run: `npm run build`
Expected: build succeeds; the route list includes `/api/settings/sharepoint` and `/api/libraries/sharepoint/test`.

- [ ] **Step 7: Commit**

```bash
git add lib/api-client.ts lib/server/ai-settings-proxy.ts app/api/settings/sharepoint/route.ts app/api/libraries/sharepoint/test/route.ts "app/api/libraries/[id]/sync/route.ts"
git commit -m "feat(sharepoint): types and session-gated proxy routes for the connector"
```

---

### Task 2: Connectors settings tab

**Files:**
- Create: `components/settings/sharepoint-settings-tab.tsx`
- Modify: `app/(dashboard)/settings/page.tsx`

**Interfaces:**
- Consumes: `SharePointSettings`, `SharePointTestResult`, `SettingSource` (Task 1); `/api/settings/sharepoint`, `/api/libraries/sharepoint/test`.
- Produces: `export function SharePointSettingsTab()`.

- [ ] **Step 1: The component**

Create `components/settings/sharepoint-settings-tab.tsx`:

```tsx
"use client";

import { useCallback, useEffect, useState } from "react";
import { CheckCircle2, Loader2, Plug, Save } from "lucide-react";
import { noticeVariants } from "@/components/ui/alert";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { cn } from "@/lib/utils";
import type { SettingSource, SharePointSettings, SharePointTestResult } from "@/lib/api-client";

const LABEL_CLASS = "text-[10px] font-bold uppercase tracking-widest";
const TITLE_CLASS = "text-sm font-bold uppercase tracking-widest flex items-center gap-2";

type SecretAction = "keep" | "replace" | "clear";

async function readDetail(res: Response, fallback: string) {
  try {
    const data = await res.json();
    return typeof data?.detail === "string" ? data.detail : fallback;
  } catch {
    return fallback;
  }
}

function SourceBadge({ source }: { source: SettingSource }) {
  if (source !== "env") return null;
  return (
    <Badge variant="muted" title="Set by an environment variable on the engine; saving here overrides it">
      from .env
    </Badge>
  );
}

export function SharePointSettingsTab() {
  const [settings, setSettings] = useState<SharePointSettings | null>(null);
  const [loadError, setLoadError] = useState<string | null>(null);
  const [tenantId, setTenantId] = useState("");
  const [clientId, setClientId] = useState("");
  const [secretAction, setSecretAction] = useState<SecretAction>("keep");
  const [newSecret, setNewSecret] = useState("");
  const [saveState, setSaveState] = useState<"idle" | "saving" | "saved">("idle");
  const [saveError, setSaveError] = useState<string | null>(null);
  const [testSiteUrl, setTestSiteUrl] = useState("");
  const [isTesting, setIsTesting] = useState(false);
  const [testResult, setTestResult] = useState<SharePointTestResult | null>(null);
  const [testError, setTestError] = useState<string | null>(null);

  const apply = useCallback((next: SharePointSettings) => {
    setSettings(next);
    setTenantId(next.tenantId);
    setClientId(next.clientId);
    setSecretAction("keep");
    setNewSecret("");
  }, []);

  const loadSettings = useCallback(async () => {
    setLoadError(null);
    try {
      const res = await fetch("/api/settings/sharepoint", { cache: "no-store" });
      if (!res.ok) throw new Error(await readDetail(res, "Could not load SharePoint settings."));
      apply((await res.json()) as SharePointSettings);
    } catch (err) {
      setLoadError(err instanceof Error ? err.message : "Could not load SharePoint settings.");
    }
  }, [apply]);

  useEffect(() => {
    void loadSettings();
  }, [loadSettings]);

  const secretEditable = !settings?.clientSecretSet || secretAction === "replace";
  const changes = settings
    ? {
        ...(tenantId.trim() !== settings.tenantId && { tenantId: tenantId.trim() }),
        ...(clientId.trim() !== settings.clientId && { clientId: clientId.trim() }),
        ...(secretEditable && newSecret.trim() && { clientSecret: newSecret.trim() }),
        ...(secretAction === "clear" && { clearClientSecret: true }),
      }
    : {};
  const dirty = Object.keys(changes).length > 0;

  async function handleSave() {
    setSaveState("saving");
    setSaveError(null);
    try {
      const res = await fetch("/api/settings/sharepoint", {
        method: "PATCH",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(changes),
      });
      if (!res.ok) throw new Error(await readDetail(res, "Could not save SharePoint settings."));
      apply((await res.json()) as SharePointSettings);
      setTestResult(null);
      setSaveState("saved");
      setTimeout(() => setSaveState("idle"), 2000);
    } catch (err) {
      setSaveError(err instanceof Error ? err.message : "Could not save SharePoint settings.");
      setSaveState("idle");
    }
  }

  async function handleTest() {
    setIsTesting(true);
    setTestError(null);
    setTestResult(null);
    try {
      const res = await fetch("/api/libraries/sharepoint/test", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ siteUrl: testSiteUrl.trim() }),
      });
      if (!res.ok) throw new Error(await readDetail(res, "Connection test failed."));
      setTestResult((await res.json()) as SharePointTestResult);
    } catch (err) {
      setTestError(err instanceof Error ? err.message : "Connection test failed.");
    } finally {
      setIsTesting(false);
    }
  }

  if (loadError) {
    return (
      <Card className="border-border/50">
        <CardContent className="flex items-center justify-between gap-3 py-6">
          <p className="text-sm text-error">{loadError}</p>
          <Button size="sm" variant="outline" onClick={() => void loadSettings()}>
            Retry
          </Button>
        </CardContent>
      </Card>
    );
  }

  if (!settings) {
    return (
      <div className="flex items-center gap-2 py-6 text-sm text-muted-foreground">
        <Loader2 className="h-4 w-4 animate-spin" />
        Loading SharePoint settings…
      </div>
    );
  }

  const { sources } = settings;

  return (
    <Card className="border-border/50">
      <CardHeader>
        <CardTitle className={TITLE_CLASS}>
          <Plug className="h-4 w-4 text-primary" />
          SharePoint
          <Badge variant={settings.configured ? "success" : "muted"}>{settings.configured ? "Configured" : "Not configured"}</Badge>
        </CardTitle>
        <CardDescription>
          Lets libraries sync from SharePoint document libraries through Microsoft Graph, using one app registration.
        </CardDescription>
      </CardHeader>
      <CardContent className="space-y-5">
        {settings.needsReentry ? (
          <div className={cn(noticeVariants({ tone: "warning" }), "px-3 py-2 text-xs")}>
            The saved client secret can&apos;t be decrypted (the engine&apos;s settings encryption key changed). Enter it again.
          </div>
        ) : null}

        <div className="space-y-2">
          <div className="flex items-center gap-2">
            <Label htmlFor="sharepoint-tenant" className={LABEL_CLASS}>
              Tenant ID
            </Label>
            <SourceBadge source={sources.tenantId} />
          </div>
          <Input
            id="sharepoint-tenant"
            value={tenantId}
            onChange={(e) => setTenantId(e.target.value)}
            placeholder="00000000-0000-0000-0000-000000000000 or contoso.onmicrosoft.com"
            autoComplete="off"
          />
        </div>

        <div className="space-y-2">
          <div className="flex items-center gap-2">
            <Label htmlFor="sharepoint-client" className={LABEL_CLASS}>
              Client (application) ID
            </Label>
            <SourceBadge source={sources.clientId} />
          </div>
          <Input
            id="sharepoint-client"
            value={clientId}
            onChange={(e) => setClientId(e.target.value)}
            placeholder="00000000-0000-0000-0000-000000000000"
            autoComplete="off"
          />
        </div>

        <div className="space-y-2">
          <div className="flex items-center gap-2">
            <Label htmlFor="sharepoint-secret" className={LABEL_CLASS}>
              Client secret
            </Label>
            <SourceBadge source={sources.clientSecret} />
          </div>
          {settings.clientSecretSet && secretAction === "keep" ? (
            <div className="flex flex-wrap items-center gap-2">
              <p className="text-sm text-foreground">
                Secret set <span className="font-mono text-muted-foreground">({settings.clientSecretMasked})</span>
              </p>
              <Button size="sm" variant="outline" onClick={() => setSecretAction("replace")}>
                Replace
              </Button>
              <Button
                size="sm"
                variant="outline"
                onClick={() => setSecretAction("clear")}
                disabled={sources.clientSecret !== "saved"}
                title={sources.clientSecret === "saved" ? undefined : "This secret comes from the engine's .env and can't be cleared here"}
              >
                Clear
              </Button>
            </div>
          ) : secretAction === "clear" ? (
            <div className="flex flex-wrap items-center gap-2">
              <p className="text-sm text-muted-foreground">The saved secret will be removed when you save.</p>
              <Button size="sm" variant="outline" onClick={() => setSecretAction("keep")}>
                Undo
              </Button>
            </div>
          ) : (
            <div className="flex flex-wrap items-center gap-2">
              <Input
                id="sharepoint-secret"
                type="password"
                autoComplete="new-password"
                value={newSecret}
                onChange={(e) => setNewSecret(e.target.value)}
                placeholder="Paste the client secret value"
                className="max-w-md"
              />
              {secretAction === "replace" ? (
                <Button
                  size="sm"
                  variant="outline"
                  onClick={() => {
                    setSecretAction("keep");
                    setNewSecret("");
                  }}
                >
                  Cancel
                </Button>
              ) : null}
            </div>
          )}
          <p className="text-[10px] text-muted-foreground">
            Stored encrypted on the engine. Only the last four characters are ever shown.
          </p>
        </div>

        <div className="flex flex-wrap items-center justify-between gap-3 border-t border-border/50 pt-4">
          <p className="text-[10px] text-muted-foreground">
            Needs Microsoft Graph application permission Sites.Selected (granted per site) or Sites.Read.All, with admin
            consent.
          </p>
          <Button size="sm" onClick={() => void handleSave()} disabled={!dirty || saveState === "saving"}>
            {saveState === "saving" ? (
              <Loader2 className="h-3 w-3 animate-spin" />
            ) : saveState === "saved" ? (
              <CheckCircle2 className="h-3 w-3" />
            ) : (
              <Save className="h-3 w-3" />
            )}
            {saveState === "saved" ? "Saved" : "Save"}
          </Button>
        </div>
        {saveError ? <p className="text-xs text-error">{saveError}</p> : null}

        <div className="space-y-2 border-t border-border/50 pt-4">
          <Label htmlFor="sharepoint-test-site" className={LABEL_CLASS}>
            Test connection
          </Label>
          <div className="flex flex-wrap gap-2">
            <Input
              id="sharepoint-test-site"
              value={testSiteUrl}
              onChange={(e) => setTestSiteUrl(e.target.value)}
              placeholder="https://contoso.sharepoint.com/sites/team"
              className="max-w-md"
            />
            <Button
              size="sm"
              variant="outline"
              onClick={() => void handleTest()}
              disabled={!settings.configured || dirty || !testSiteUrl.trim() || isTesting}
            >
              {isTesting ? <Loader2 className="h-3 w-3 animate-spin" /> : <Plug className="h-3 w-3" />}
              Test
            </Button>
          </div>
          <p className="text-[10px] text-muted-foreground">
            {dirty ? "Save your changes before testing." : "Signs in with these credentials and reads the site's document libraries."}
          </p>
          {testResult ? (
            <div className={cn(noticeVariants({ tone: "success" }), "px-3 py-2 text-xs")}>
              Connected. Document libraries: {testResult.drives.map((drive) => drive.name).join(", ") || "none"}.{" "}
              {testResult.sampleSupportedFiles} supported file{testResult.sampleSupportedFiles === 1 ? "" : "s"} among the first
              items of {testResult.driveName}.
            </div>
          ) : null}
          {testError ? <div className={cn(noticeVariants({ tone: "error" }), "px-3 py-2 text-xs")}>{testError}</div> : null}
        </div>
      </CardContent>
    </Card>
  );
}
```

- [ ] **Step 2: Add the tab**

In `app/(dashboard)/settings/page.tsx`:

- Add `import { SharePointSettingsTab } from "@/components/settings/sharepoint-settings-tab";` after the `RetrievalSettingsTab` import.
- Add `Plug` to the existing `lucide-react` import list.
- In the `TabsList` array, add `{ value: "connectors", label: "Connectors", icon: Plug },` right after the `search` entry.
- After the `{/* Search Settings */}` `TabsContent`, add:

```tsx
          {/* Connectors */}
          <TabsContent value="connectors" className="space-y-4">
            <SharePointSettingsTab />
          </TabsContent>
```

- [ ] **Step 3: Build**

Run: `npm run build`
Expected: succeeds.

- [ ] **Step 4: Commit**

```bash
git add components/settings/sharepoint-settings-tab.tsx "app/(dashboard)/settings/page.tsx"
git commit -m "feat(sharepoint): Connectors settings tab with masked secret and connection test"
```

---

### Task 3: SharePoint source fields and sync status panel

**Files:**
- Create: `components/libraries/sharepoint-source-fields.tsx`, `components/libraries/sync-status-panel.tsx`

**Interfaces:**
- Consumes: `SharePointDrive`, `SharePointTestResult`, `SharePointSourceSettings` (Task 1).
- Produces:
  - `SharePointSourceValue { siteUrl; driveId; driveName; folderPath }`, `EMPTY_SHAREPOINT_SOURCE`, `sharePointSourceComplete(value) -> boolean`, `SharePointSourceFields({ idPrefix, value, onChange, disabled? })`
  - `SyncStatusMonitor`, `SyncStatusResult`, `SyncStatusPanel({ monitor, onFullResync, disabled? })`

- [ ] **Step 1: Source fields**

Create `components/libraries/sharepoint-source-fields.tsx`:

```tsx
"use client";

import { useState } from "react";
import { Loader2, Search } from "lucide-react";
import { noticeVariants } from "@/components/ui/alert";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { cn } from "@/lib/utils";
import type { SharePointDrive, SharePointTestResult } from "@/lib/api-client";

export interface SharePointSourceValue {
  siteUrl: string;
  driveId: string;
  driveName: string;
  folderPath: string;
}

export const EMPTY_SHAREPOINT_SOURCE: SharePointSourceValue = { siteUrl: "", driveId: "", driveName: "", folderPath: "" };

export function sharePointSourceComplete(value: SharePointSourceValue) {
  return Boolean(value.siteUrl.trim() && (value.driveId || value.driveName));
}

const SELECT_CLASS =
  "flex h-11 w-full rounded-[2px] border border-input bg-card px-3 py-2 text-sm ring-offset-background transition-colors duration-200 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 focus-visible:border-primary/45 disabled:cursor-not-allowed disabled:opacity-50";

export function SharePointSourceFields({
  idPrefix,
  value,
  onChange,
  disabled,
}: {
  idPrefix: string;
  value: SharePointSourceValue;
  onChange: (next: SharePointSourceValue) => void;
  disabled?: boolean;
}) {
  const [drives, setDrives] = useState<SharePointDrive[] | null>(null);
  const [isLooking, setIsLooking] = useState(false);
  const [lookupError, setLookupError] = useState<string | null>(null);
  const [lookupResult, setLookupResult] = useState<SharePointTestResult | null>(null);

  async function findLibraries() {
    setIsLooking(true);
    setLookupError(null);
    setLookupResult(null);
    try {
      const res = await fetch("/api/libraries/sharepoint/test", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          siteUrl: value.siteUrl.trim(),
          driveId: value.driveId,
          driveName: value.driveName,
          folderPath: value.folderPath.trim(),
        }),
      });
      const data = await res.json();
      if (!res.ok) throw new Error(typeof data?.detail === "string" ? data.detail : "Couldn't reach that site.");
      const result = data as SharePointTestResult;
      setLookupResult(result);
      setDrives(result.drives);
      onChange({ ...value, driveId: result.driveId, driveName: result.driveName, folderPath: result.folderPath });
    } catch (err) {
      setLookupError(err instanceof Error ? err.message : "Couldn't reach that site.");
    } finally {
      setIsLooking(false);
    }
  }

  return (
    <div className="grid gap-4">
      <div className="space-y-1.5">
        <Label htmlFor={`${idPrefix}-site`}>Site URL</Label>
        <div className="flex flex-wrap gap-2">
          <Input
            id={`${idPrefix}-site`}
            value={value.siteUrl}
            onChange={(e) => {
              // Libraries belong to a site: pick again after changing it.
              setDrives(null);
              setLookupResult(null);
              onChange({ ...value, siteUrl: e.target.value, driveId: "", driveName: "" });
            }}
            placeholder="https://contoso.sharepoint.com/sites/team"
            disabled={disabled}
            className="min-w-0 flex-1"
          />
          <Button
            type="button"
            variant="outline"
            onClick={() => void findLibraries()}
            disabled={disabled || isLooking || !value.siteUrl.trim()}
          >
            {isLooking ? <Loader2 className="h-4 w-4 animate-spin" /> : <Search className="h-4 w-4" />}
            Find libraries
          </Button>
        </div>
        <p className="text-[11px] text-muted-foreground">
          Paste the site address, or a browser link to the document library or folder.
        </p>
      </div>

      <div className="grid gap-4 md:grid-cols-2">
        <div className="space-y-1.5">
          <Label htmlFor={`${idPrefix}-drive`}>Document library</Label>
          {drives ? (
            <select
              id={`${idPrefix}-drive`}
              value={value.driveId}
              onChange={(e) => {
                const drive = drives.find((item) => item.id === e.target.value);
                onChange({ ...value, driveId: drive?.id ?? "", driveName: drive?.name ?? "" });
              }}
              disabled={disabled}
              className={SELECT_CLASS}
            >
              {drives.map((drive) => (
                <option key={drive.id} value={drive.id}>
                  {drive.name}
                </option>
              ))}
            </select>
          ) : (
            <p id={`${idPrefix}-drive`} className="flex h-11 items-center text-sm text-muted-foreground">
              {value.driveName || "Use Find libraries to choose"}
            </p>
          )}
        </div>
        <div className="space-y-1.5">
          <Label htmlFor={`${idPrefix}-folder`}>Folder (optional)</Label>
          <Input
            id={`${idPrefix}-folder`}
            value={value.folderPath}
            onChange={(e) => onChange({ ...value, folderPath: e.target.value })}
            placeholder="Amerihome/Rate sheets"
            disabled={disabled}
          />
        </div>
      </div>

      {lookupResult ? (
        <div className={cn(noticeVariants({ tone: "success" }), "px-3 py-2 text-xs")}>
          Connected to {lookupResult.driveName}
          {lookupResult.folderPath ? ` › ${lookupResult.folderPath}` : ""}. {lookupResult.sampleSupportedFiles} supported
          file{lookupResult.sampleSupportedFiles === 1 ? "" : "s"} among the first items. Supported: PDF, Markdown, EML, MSG.
        </div>
      ) : null}
      {lookupError ? <div className={cn(noticeVariants({ tone: "error" }), "px-3 py-2 text-xs")}>{lookupError}</div> : null}
    </div>
  );
}
```

- [ ] **Step 2: Sync status panel**

Create `components/libraries/sync-status-panel.tsx`:

```tsx
"use client";

import { AlertTriangle, Loader2, RotateCcw } from "lucide-react";
import { noticeVariants } from "@/components/ui/alert";
import { Button } from "@/components/ui/button";
import { cn } from "@/lib/utils";
import type { SharePointSourceSettings } from "@/lib/api-client";

const MAX_AUTOMATIC_ATTEMPTS = 5;

export interface SyncStatusResult {
  added: number;
  updated: number;
  removed: number;
  unchanged: number;
  errorCount: number;
  errors: Array<{ path: string; error: string }>;
  mode?: "delta" | "full" | null;
  renamed?: number;
  skipped?: number;
  pendingCount?: number;
  outcome?: "completed" | "failed" | "superseded" | null;
}

export interface SyncStatusMonitor {
  sourceType: "folder" | "sharepoint";
  enabled: boolean;
  syncInProgress: boolean;
  lastCompletedAt?: string | null;
  lastRequestedAt?: string | null;
  lastError?: string | null;
  lastResult?: SyncStatusResult | null;
  sharePoint: SharePointSourceSettings;
}

function formatDateTime(value?: string | null) {
  if (!value) return "—";
  const date = new Date(value);
  return Number.isNaN(date.getTime()) ? "—" : date.toLocaleString();
}

function summary(result: SyncStatusResult) {
  if (result.outcome === "superseded") return "Last sync stopped because the source changed; a new sync started.";
  const kind = result.mode === "full" ? "full scan" : result.mode === "delta" ? "incremental sync" : "sync";
  const parts = [
    `+${result.added} added`,
    `${result.updated} updated`,
    ...(result.renamed ? [`${result.renamed} renamed`] : []),
    `${result.removed} removed`,
    `${result.unchanged} unchanged`,
    ...(result.skipped ? [`${result.skipped} skipped (unsupported type)`] : []),
  ];
  return `Last ${kind}: ${parts.join(", ")}`;
}

export function SyncStatusPanel({
  monitor,
  onFullResync,
  disabled,
}: {
  monitor: SyncStatusMonitor;
  onFullResync: () => void;
  disabled?: boolean;
}) {
  const result = monitor.lastResult;
  const isSharePoint = monitor.sourceType === "sharepoint";
  const pending = isSharePoint ? Object.entries(monitor.sharePoint.pendingItems) : [];

  return (
    <div className="rounded-[5px] border border-border/60 bg-background/70 p-3">
      <p className="text-[10px] font-bold uppercase tracking-widest text-muted-foreground">Current sync state</p>
      <div className="mt-2 grid gap-2 text-sm text-foreground md:grid-cols-2">
        <p>Status: {monitor.syncInProgress ? "Syncing" : monitor.enabled ? "Ready" : "Disabled"}</p>
        <p>Last sync: {formatDateTime(monitor.lastCompletedAt)}</p>
        <p>Last requested: {formatDateTime(monitor.lastRequestedAt)}</p>
        <p>Last error: {monitor.lastError || "None"}</p>
        {isSharePoint ? (
          <>
            <p>Last connected: {formatDateTime(monitor.sharePoint.lastConnectedAt)}</p>
            <p>
              Change tracking:{" "}
              {monitor.sharePoint.scopeMode === "drive"
                ? "whole drive, filtered to the folder"
                : monitor.sharePoint.scopeMode === "folder"
                  ? "folder"
                  : "not started"}
            </p>
          </>
        ) : null}
      </div>

      {isSharePoint && monitor.sharePoint.lastConnectionError ? (
        <div className={cn(noticeVariants({ tone: "error" }), "mt-3 px-3 py-2 text-xs")}>
          Connection problem: {monitor.sharePoint.lastConnectionError}
        </div>
      ) : null}

      {result ? (
        <div className="mt-3 rounded-[5px] border border-border/60 bg-background/80 p-3 text-xs text-muted-foreground">
          <p className="font-semibold text-foreground">{summary(result)}</p>
          {result.errorCount > 0 && result.errors.length > 0 ? (
            <p className="mt-2 text-error">
              {result.errors[0].path ? `${result.errors[0].path}: ${result.errors[0].error}` : result.errors[0].error}
            </p>
          ) : null}
        </div>
      ) : null}

      {pending.length > 0 ? (
        <div className="mt-3 space-y-1.5">
          <p className="flex items-center gap-1.5 text-[10px] font-bold uppercase tracking-widest text-muted-foreground">
            <AlertTriangle className="h-3 w-3 text-warning" />
            Waiting to retry ({pending.length})
          </p>
          <ul className="space-y-1 text-xs">
            {pending.map(([itemId, item]) => (
              <li key={itemId} className="rounded-[5px] border border-border/60 bg-background/80 px-2 py-1.5">
                <span className="font-semibold text-foreground">{item.name}</span>
                <span className="text-muted-foreground">
                  {" "}· {item.attempts} attempt{item.attempts === 1 ? "" : "s"}
                  {item.attempts >= MAX_AUTOMATIC_ATTEMPTS ? " · retried on Sync now or Full resync" : ""}
                </span>
                {item.lastError ? <p className="mt-0.5 break-words text-error">{item.lastError}</p> : null}
              </li>
            ))}
          </ul>
        </div>
      ) : null}

      {isSharePoint ? (
        <Button
          type="button"
          variant="outline"
          size="sm"
          className="mt-3"
          onClick={onFullResync}
          disabled={disabled || monitor.syncInProgress}
          title="List the whole folder again and remove documents that are no longer there"
        >
          {monitor.syncInProgress ? <Loader2 className="h-3 w-3 animate-spin" /> : <RotateCcw className="h-3 w-3" />}
          Full resync
        </Button>
      ) : null}
    </div>
  );
}
```

- [ ] **Step 3: Build**

Run: `npm run build`
Expected: succeeds (the components aren't used yet; the build still type-checks them).

- [ ] **Step 4: Commit**

```bash
git add components/libraries/sharepoint-source-fields.tsx components/libraries/sync-status-panel.tsx
git commit -m "feat(sharepoint): source fields with library picker, and a sync status panel"
```

---

### Task 4: Library detail page

**Files:**
- Modify: `app/(dashboard)/libraries/[id]/page.tsx`

**Interfaces:**
- Consumes: `SharePointSourceFields`, `EMPTY_SHAREPOINT_SOURCE`, `SharePointSourceValue` (Task 3), `SyncStatusPanel` (Task 3), `SharePointSourceSettings` (Task 1).

- [ ] **Step 1: Imports**

Add `ExternalLink` to the `lucide-react` import list. Add after the `useSavedTopPages` import:

```tsx
import { Dialog, DialogContent, DialogDescription, DialogFooter, DialogHeader, DialogTitle } from "@/components/ui/dialog";
import {
  EMPTY_SHAREPOINT_SOURCE,
  SharePointSourceFields,
  type SharePointSourceValue,
} from "@/components/libraries/sharepoint-source-fields";
import { SyncStatusPanel } from "@/components/libraries/sync-status-panel";
import type { SharePointSourceSettings } from "@/lib/api-client";
```

- [ ] **Step 2: Types**

- In `LibraryDocument`, change `sourceType?: "upload" | "folder";` to `sourceType?: "upload" | "folder" | "sharepoint";` and add `sharePointWebUrl?: string | null;` below it.
- In `FolderMonitorResult`, add after `errors`:

```tsx
  mode?: "delta" | "full" | null;
  renamed?: number;
  skipped?: number;
  pendingCount?: number;
  outcome?: "completed" | "failed" | "superseded" | null;
```

- In `FolderMonitor`, add after `enabled: boolean;`:

```tsx
  sourceType: "folder" | "sharepoint";
  sharePoint: SharePointSourceSettings;
```

- [ ] **Step 3: Mapping**

Add above `function mapLibrary`:

```tsx
function mapSharePoint(value?: Partial<SharePointSourceSettings>): SharePointSourceSettings {
  return {
    siteUrl: value?.siteUrl ?? "",
    driveId: value?.driveId ?? "",
    driveName: value?.driveName ?? "",
    folderPath: value?.folderPath ?? "",
    scopeMode: value?.scopeMode === "folder" || value?.scopeMode === "drive" ? value.scopeMode : "",
    lastConnectedAt: value?.lastConnectedAt ?? null,
    lastConnectionError: value?.lastConnectionError ?? null,
    pendingItems: value?.pendingItems && typeof value.pendingItems === "object" ? value.pendingItems : {},
  };
}
```

In `mapLibrary`:
- in the document mapping, add `sharePointWebUrl: doc.sharePointWebUrl,` after `sourceModifiedAt`.
- in `folderMonitor`, add after `enabled: Boolean(data.folderMonitor?.enabled),`:

```tsx
      sourceType: data.folderMonitor?.sourceType === "sharepoint" ? "sharepoint" : "folder",
      sharePoint: mapSharePoint(data.folderMonitor?.sharePoint),
```

- in the `lastResult` object, add after the `errors: ...` property:

```tsx
            mode: data.folderMonitor.lastResult.mode ?? null,
            renamed: data.folderMonitor.lastResult.renamed ?? 0,
            skipped: data.folderMonitor.lastResult.skipped ?? 0,
            pendingCount: data.folderMonitor.lastResult.pendingCount ?? 0,
            outcome: data.folderMonitor.lastResult.outcome ?? null,
```

- [ ] **Step 4: Edit state and initialisation**

After `const [editPollingIntervalMinutes, ...] = useState<1 | 5 | 10 | 60>(5);` add:

```tsx
  const [editSourceType, setEditSourceType] = useState<"folder" | "sharepoint">("folder");
  const [editSharePoint, setEditSharePoint] = useState<SharePointSourceValue>(EMPTY_SHAREPOINT_SOURCE);
  const [sourceSwitchDialogOpen, setSourceSwitchDialogOpen] = useState(false);
  const [retargetDialogOpen, setRetargetDialogOpen] = useState(false);
```

In the first-load effect, after `setEditPollingIntervalMinutes(...)`, add:

```tsx
      setEditSourceType(library.folderMonitor.sourceType);
      setEditSharePoint({
        siteUrl: library.folderMonitor.sharePoint.siteUrl,
        driveId: library.folderMonitor.sharePoint.driveId,
        driveName: library.folderMonitor.sharePoint.driveName,
        folderPath: library.folderMonitor.sharePoint.folderPath,
      });
```

- [ ] **Step 5: Save flow**

Rename `const handleSaveSettings = async () => {` to `const saveSettings = async (keepExistingDocuments?: boolean) => {`, and in its `JSON.stringify({...})` body add after `pollingIntervalMinutes: editPollingIntervalMinutes,`:

```tsx
          syncSourceType: editSourceType,
          ...(editSourceType === "sharepoint" && {
            sharePointSiteUrl: editSharePoint.siteUrl.trim(),
            sharePointDriveId: editSharePoint.driveId,
            sharePointDriveName: editSharePoint.driveName,
            sharePointFolderPath: editSharePoint.folderPath.trim(),
          }),
          ...(keepExistingDocuments !== undefined && { keepExistingDocuments }),
```

Directly after the `saveSettings` function, add:

```tsx
  const currentSourceType = library?.folderMonitor.sourceType ?? "folder";
  const currentSharePoint = library?.folderMonitor.sharePoint;
  const currentSourceDocumentCount = Object.values(library?.documents ?? {}).filter(
    (doc) => doc.sourceType === currentSourceType,
  ).length;
  const sharePointTargetChanged =
    editSourceType === "sharepoint" &&
    currentSourceType === "sharepoint" &&
    Boolean(currentSharePoint) &&
    (editSharePoint.siteUrl.trim() !== currentSharePoint?.siteUrl ||
      editSharePoint.driveId !== currentSharePoint?.driveId ||
      editSharePoint.folderPath.trim().replace(/^\/+|\/+$/g, "") !== currentSharePoint?.folderPath);

  // Ask before a save that removes synced documents: a source switch, or a new SharePoint location.
  const handleSaveSettings = () => {
    if (editSourceType !== currentSourceType && currentSourceDocumentCount > 0) {
      setSourceSwitchDialogOpen(true);
      return;
    }
    if (sharePointTargetChanged && currentSourceDocumentCount > 0) {
      setRetargetDialogOpen(true);
      return;
    }
    void saveSettings();
  };
```

Change the Save settings button's `onClick={() => void handleSaveSettings()}` to `onClick={handleSaveSettings}`.

- [ ] **Step 6: Sync actions**

Replace `const handleSyncNow = async () => {` with `const handleSyncNow = async (full = false) => {`, the fetch URL with `` `/api/libraries/${libraryId}/sync${full ? "?full=true" : ""}` ``, and the success fallback `"Folder sync started."` with `full ? "Full resync started." : "Sync started."`; the error fallback `"Folder sync failed"` becomes `"Sync failed"`.

Replace

```tsx
  const folderMonitorSummary = library?.folderMonitor.lastResult;
  const canSyncNow = Boolean(library?.folderMonitor.folderPath.trim()) && !library?.folderMonitor.syncInProgress;
```

with

```tsx
  const hasSyncTarget =
    library?.folderMonitor.sourceType === "sharepoint"
      ? Boolean(library.folderMonitor.sharePoint.siteUrl && (library.folderMonitor.sharePoint.driveId || library.folderMonitor.sharePoint.driveName))
      : Boolean(library?.folderMonitor.folderPath.trim());
  const canSyncNow = hasSyncTarget && !library?.folderMonitor.syncInProgress;
```

(Every existing `onClick={() => void handleSyncNow()}` keeps working: `full` defaults to false.)

- [ ] **Step 7: Metadata card**

In the "Library metadata" card, change the label `Folder monitor` to `Sync` and replace the "Source folder" block's contents with:

```tsx
                <p className="text-[10px] font-bold uppercase tracking-widest text-muted-foreground">Source</p>
                <p className="mt-1 break-all text-sm text-foreground/90">
                  {library?.folderMonitor.sourceType === "sharepoint"
                    ? [library.folderMonitor.sharePoint.siteUrl, library.folderMonitor.sharePoint.driveName, library.folderMonitor.sharePoint.folderPath]
                        .filter(Boolean)
                        .join(" › ") || "No SharePoint location configured"
                    : library?.folderMonitor.folderPath || "No monitored folder configured"}
                </p>
```

- [ ] **Step 8: Document rows**

After the `{doc.sourceType === "folder" && doc.sourceRelativePath ? (...) : null}` block, add:

```tsx
                          {doc.sourceType === "sharepoint" && doc.sourceRelativePath ? (
                            <p className="truncate text-[11px] text-muted-foreground">
                              SharePoint:{" "}
                              {doc.sharePointWebUrl ? (
                                <a
                                  href={doc.sharePointWebUrl}
                                  target="_blank"
                                  rel="noreferrer"
                                  className="inline-flex items-center gap-1 underline-offset-2 hover:underline"
                                >
                                  {doc.sourceRelativePath}
                                  <ExternalLink className="h-3 w-3" />
                                </a>
                              ) : (
                                doc.sourceRelativePath
                              )}
                            </p>
                          ) : null}
```

- [ ] **Step 9: Settings tab: source choice and status**

Replace the whole "Folder monitor" block (the `<div className="rounded-[5px] border border-border/60 bg-background/60 p-4">` that starts with the "Folder monitor" heading and ends before `{saveError ? (`) with:

```tsx
                <div className="rounded-[5px] border border-border/60 bg-background/60 p-4">
                  <div className="flex flex-wrap items-start justify-between gap-3">
                    <div>
                      <p className="text-sm font-semibold text-foreground">Automatic sync</p>
                      <p className="mt-1 text-xs text-muted-foreground">
                        Keep this library in sync with a local or network folder, or a SharePoint document library.
                      </p>
                    </div>
                    <label className="inline-flex items-center gap-2 rounded-md border border-border/60 px-3 py-2 text-sm text-foreground">
                      <input
                        type="checkbox"
                        className="h-4 w-4 rounded border-input bg-background accent-primary"
                        checked={editFolderMonitorEnabled}
                        onChange={(event) => {
                          setEditFolderMonitorEnabled(event.target.checked);
                          setSaveSuccess(false);
                        }}
                        disabled={isSaving}
                      />
                      Enable sync
                    </label>
                  </div>

                  <div className="mt-4 grid gap-4">
                    <div className="space-y-1.5">
                      <Label>Source</Label>
                      <div className="inline-flex rounded-[5px] border border-border/60 p-0.5" role="radiogroup" aria-label="Sync source">
                        {(["folder", "sharepoint"] as const).map((type) => (
                          <button
                            key={type}
                            type="button"
                            role="radio"
                            aria-checked={editSourceType === type}
                            disabled={isSaving}
                            onClick={() => {
                              setEditSourceType(type);
                              setSaveSuccess(false);
                            }}
                            className={cn(
                              "rounded-[4px] px-3 py-1.5 text-xs font-semibold transition-colors disabled:cursor-not-allowed disabled:opacity-40",
                              editSourceType === type ? "bg-primary text-primary-foreground" : "text-muted-foreground hover:text-foreground",
                            )}
                          >
                            {type === "folder" ? "Local folder" : "SharePoint"}
                          </button>
                        ))}
                      </div>
                    </div>

                    {editSourceType === "folder" ? (
                      <div className="space-y-1.5">
                        <Label htmlFor="folder-path">Folder path</Label>
                        <Input
                          id="folder-path"
                          value={editFolderPath}
                          onChange={(event) => {
                            setEditFolderPath(event.target.value);
                            setSaveSuccess(false);
                          }}
                          placeholder="\\\\server\\share\\documents or D:\\Libraries\\Source"
                          disabled={isSaving}
                        />
                        <p className="text-[11px] text-muted-foreground">
                          Supported files: PDF, Markdown, EML, and MSG. Subfolders are scanned recursively.
                        </p>
                      </div>
                    ) : (
                      <SharePointSourceFields
                        idPrefix="edit-sharepoint"
                        value={editSharePoint}
                        onChange={(next) => {
                          setEditSharePoint(next);
                          setSaveSuccess(false);
                        }}
                        disabled={isSaving}
                      />
                    )}

                    <div className="grid gap-4 md:grid-cols-[minmax(0,_220px)_1fr]">
                      <div className="space-y-1.5">
                        <Label htmlFor="poll-interval">Polling interval</Label>
                        <select
                          id="poll-interval"
                          value={editPollingIntervalMinutes}
                          onChange={(event) => {
                            setEditPollingIntervalMinutes(Number(event.target.value) as 1 | 5 | 10 | 60);
                            setSaveSuccess(false);
                          }}
                          disabled={isSaving}
                          className="flex h-11 w-full rounded-[2px] border border-input bg-card px-3 py-2 text-sm ring-offset-background transition-colors duration-200 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 focus-visible:border-primary/45 disabled:cursor-not-allowed disabled:opacity-50"
                        >
                          <option value={1}>1 minute</option>
                          <option value={5}>5 minutes</option>
                          <option value={10}>10 minutes</option>
                          <option value={60}>60 minutes</option>
                        </select>
                      </div>

                      {library ? (
                        <SyncStatusPanel
                          monitor={library.folderMonitor}
                          onFullResync={() => void handleSyncNow(true)}
                          disabled={isSyncingNow || isSaving}
                        />
                      ) : null}
                    </div>
                  </div>
                </div>
```

- [ ] **Step 10: Dialogs**

After the second `ConfirmDialog` (delete library), add:

```tsx
      <Dialog open={sourceSwitchDialogOpen} onOpenChange={setSourceSwitchDialogOpen}>
        <DialogContent>
          <DialogHeader>
            <DialogTitle>Switch to {editSourceType === "sharepoint" ? "SharePoint" : "a local folder"}?</DialogTitle>
            <DialogDescription>
              This library has {currentSourceDocumentCount} document{currentSourceDocumentCount === 1 ? "" : "s"} from its
              current {currentSourceType === "sharepoint" ? "SharePoint location" : "folder"}. Keep them, or remove them and
              start fresh from the new source? Uploaded documents are kept either way.
            </DialogDescription>
          </DialogHeader>
          <DialogFooter>
            <Button variant="outline" onClick={() => setSourceSwitchDialogOpen(false)}>
              Cancel
            </Button>
            <Button
              variant="outline"
              onClick={() => {
                setSourceSwitchDialogOpen(false);
                void saveSettings(true);
              }}
            >
              Keep documents
            </Button>
            <Button
              variant="destructive"
              onClick={() => {
                setSourceSwitchDialogOpen(false);
                void saveSettings(false);
              }}
            >
              Remove documents
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>

      <ConfirmDialog
        open={retargetDialogOpen}
        onOpenChange={setRetargetDialogOpen}
        title="Change the SharePoint location?"
        description={`The ${currentSourceDocumentCount} document${currentSourceDocumentCount === 1 ? "" : "s"} synced from the current location will be removed on the next sync, and files from the new location added.`}
        confirmLabel="Change location"
        onConfirm={() => {
          setRetargetDialogOpen(false);
          void saveSettings();
        }}
        isLoading={isSaving}
      />
```

- [ ] **Step 11: Build**

Run: `npm run build`
Expected: succeeds with no type errors. If `folderMonitorSummary` is reported unused/undefined anywhere, it was only used by the removed status block; remove any leftover reference.

- [ ] **Step 12: Commit**

```bash
git add "app/(dashboard)/libraries/[id]/page.tsx"
git commit -m "feat(sharepoint): SharePoint source, keep-or-remove switch and sync status on the library page"
```

---

### Task 5: SharePoint in the create-library dialog

**Files:**
- Modify: `app/(dashboard)/libraries/page.tsx`

**Interfaces:**
- Consumes: `SharePointSourceFields`, `EMPTY_SHAREPOINT_SOURCE`, `sharePointSourceComplete`, `SharePointSourceValue` (Task 3).

- [ ] **Step 1: Imports and state**

Add:

```tsx
import {
  EMPTY_SHAREPOINT_SOURCE,
  SharePointSourceFields,
  sharePointSourceComplete,
  type SharePointSourceValue,
} from "@/components/libraries/sharepoint-source-fields";
```

After `const [formTags, setFormTags] = useState("");` add:

```tsx
  const [formSource, setFormSource] = useState<"upload" | "sharepoint">("upload");
  const [formSharePoint, setFormSharePoint] = useState<SharePointSourceValue>(EMPTY_SHAREPOINT_SOURCE);
  const [formPollingIntervalMinutes, setFormPollingIntervalMinutes] = useState<1 | 5 | 10 | 60>(5);
```

In `resetCreateDialog`, after `setFormTags("");` add:

```tsx
    setFormSource("upload");
    setFormSharePoint(EMPTY_SHAREPOINT_SOURCE);
    setFormPollingIntervalMinutes(5);
```

- [ ] **Step 2: Validation and payload**

In the manual branch of `handleCreateLibrary`, after the `if (!name) { ... return; }` check, add:

```tsx
    if (formSource === "sharepoint" && !sharePointSourceComplete(formSharePoint)) {
      setCreateError("Enter the SharePoint site URL and choose a document library with Find libraries.");
      return;
    }
```

In the manual `payload` object, add after `tags: ...`:

```tsx
        ...(formSource === "sharepoint" && {
          syncSourceType: "sharepoint",
          folderMonitorEnabled: true,
          pollingIntervalMinutes: formPollingIntervalMinutes,
          sharePointSiteUrl: formSharePoint.siteUrl.trim(),
          sharePointDriveId: formSharePoint.driveId,
          sharePointDriveName: formSharePoint.driveName,
          sharePointFolderPath: formSharePoint.folderPath.trim(),
        }),
```

- [ ] **Step 3: Form fields**

In the manual form (`createMode === "manual"`), after the Tags `div`, add:

```tsx
                <div className="space-y-2">
                  <Label>Documents come from</Label>
                  <div className="inline-flex rounded-[5px] border border-border/60 p-0.5" role="radiogroup" aria-label="Document source">
                    {(["upload", "sharepoint"] as const).map((source) => (
                      <button
                        key={source}
                        type="button"
                        role="radio"
                        aria-checked={formSource === source}
                        onClick={() => setFormSource(source)}
                        className={cn(
                          "rounded-[4px] px-3 py-1.5 text-xs font-semibold transition-colors",
                          formSource === source ? "bg-primary text-primary-foreground" : "text-muted-foreground hover:text-foreground",
                        )}
                      >
                        {source === "upload" ? "Uploads" : "SharePoint"}
                      </button>
                    ))}
                  </div>
                </div>
                {formSource === "sharepoint" ? (
                  <div className="space-y-4 rounded-[5px] border border-border/60 bg-background/60 p-4">
                    <SharePointSourceFields idPrefix="create-sharepoint" value={formSharePoint} onChange={setFormSharePoint} />
                    <div className="space-y-1.5">
                      <Label htmlFor="create-poll-interval">Check for changes every</Label>
                      <select
                        id="create-poll-interval"
                        value={formPollingIntervalMinutes}
                        onChange={(e) => setFormPollingIntervalMinutes(Number(e.target.value) as 1 | 5 | 10 | 60)}
                        className="flex h-11 w-full max-w-[220px] rounded-[2px] border border-input bg-card px-3 py-2 text-sm ring-offset-background transition-colors duration-200 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 focus-visible:border-primary/45"
                      >
                        <option value={1}>1 minute</option>
                        <option value={5}>5 minutes</option>
                        <option value={10}>10 minutes</option>
                        <option value={60}>60 minutes</option>
                      </select>
                    </div>
                  </div>
                ) : null}
```

If `cn` is not yet imported in this file, add `import { cn } from "@/lib/utils";`.

- [ ] **Step 4: Build**

Run: `npm run build`
Expected: succeeds.

- [ ] **Step 5: Commit**

```bash
git add "app/(dashboard)/libraries/page.tsx"
git commit -m "feat(sharepoint): create a library that syncs from SharePoint"
```

---

### Task 6: Browser verification (no tenant)

Start PocketBase, the engine (from `feature/sharepoint-hardening`) and the frontend with the existing `.claude/launch.json` preview configs, sign in, then check with the in-app browser (use `read_page`/`get_page_text` first, screenshots for layout):

- [ ] Settings → **Connectors** loads: "Not configured", three empty fields, Test disabled. No console errors.
- [ ] Typing in Tenant ID enables Save; an invalid tenant ID shows the engine's 400 message and nothing is saved (reload shows the old value). Clear the field again without saving.
- [ ] A library's Settings tab shows **Automatic sync** with the Local folder / SharePoint choice. Choosing SharePoint shows the site URL, **Find libraries**, library and folder fields; **Find libraries** with any URL shows "SharePoint credentials are not configured…" as an error notice.
- [ ] Switching a folder library that has folder documents to SharePoint and pressing Save opens the keep-or-remove dialog; **Cancel** saves nothing.
- [ ] The create dialog shows **Documents come from: Uploads | SharePoint**; creating with SharePoint and no library chosen shows the validation message.
- [ ] Layout holds at 375 px width (no horizontal scroll) for the Connectors tab and the library Settings tab.
- [ ] Commit any fixes found (by path), then `npm run build` once more.

---

### Task 7: Live-tenant verification (with the user)

The user enters the credentials; Claude never types or sees the secret.

- [ ] **Step 1:** Ask the user to open Settings → Connectors, enter the tenant ID, client ID and client secret, and press Save. Confirm with `GET /api/settings/sharepoint` (through the browser page) that `configured` is true and only the masked tail is shown.
- [ ] **Step 2:** Ask the user for a test site URL (not a secret) and run **Test** in Connectors. Record the listed document libraries. On a 403, point the user at the README's `Sites.Selected` grant.
- [ ] **Step 3:** Create a library "SharePoint test" from a small test folder: SharePoint → site URL → Find libraries → choose the library → folder → Create. Wait for the first sync; record `scopeMode` from the status panel, the added count and the document paths.
- [ ] **Step 4:** Ask the user to do each of these in SharePoint, one at a time; after each, press **Sync now** and check the status panel:
  - add a PDF → `+1 added`
  - edit its content → `1 updated` (re-indexed)
  - rename it → `1 renamed`, no re-index (the document's indexed time doesn't change)
  - move it into a subfolder of the synced folder → `1 renamed`, path shows the subfolder
  - delete it → `1 removed`
- [ ] **Step 5:** Change the library's folder to another folder; confirm the dialog, save; the old documents go and the new folder's documents arrive (`outcome` is completed; if a sync was running, the superseded message appears first).
- [ ] **Step 6:** Ask a question in Chat that only a synced document answers; the answer cites it.
- [ ] **Step 7:** Press **Full resync**; `Last full scan` with no unexpected removals.
- [ ] **Step 8:** Write the results (scope mode, counts, any errors) into the PR descriptions. Fix problems as follow-up commits on the same branches before merging.

### Task 8: Push and PR (only when the user asks)

```bash
git push -u origin feat/sharepoint-connectors
gh pr create --repo dmerriman11/lemur-pageindex --base master --head feat/sharepoint-connectors --title "SharePoint connector UI"
```
