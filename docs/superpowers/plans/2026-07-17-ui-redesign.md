# UI Redesign (Conversations Sidebar, Unified Chat/RAG, Settings Panel) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the four-page nav (Chat / Knowledge Base / Web Search / Settings) with a conversations sidebar (unified Chat+RAG history), a bottom-left Settings slide-over that also owns document management, and a per-conversation web-search toggle wired into both Chat and RAG.

**Architecture:** A route-driven shell keeps `/chat` and `/rag` pages; a rewritten `Sidebar` owns conversation history, the Chat/RAG mode switch, and the Settings trigger. A `conversations` collection in the Zustand store (persisted to `localStorage`) replaces the single `chatMessages` array. The Settings panel is a Radix `Dialog` slide-over consolidating Models, Generation, Backend keys, and Knowledge Base. A small backend change lets `/api/rag/query` optionally combine retrieved docs with web search.

**Tech Stack:** Next.js 16 (App Router) + React 19 + TypeScript 5 (strict); Zustand 5 (persist); shadcn/ui + Radix primitives; Tailwind v4; lucide-react. Backend: FastAPI + Pydantic (`app/backend/`).

> **Commit policy (user instruction):** Do NOT run `git commit` or `git push` during execution. The user commits locally. Commit steps below are the standard workflow and are skipped.

---

## Global Constraints

- Frontend: `npm run lint` (ESLint) and `npm run build` (Next.js + TS) MUST pass with zero errors before any task is marked done (from `nextjs-frontend/AGENTS.md`).
- TypeScript strict mode; no implicit `any`; never suppress errors with blanket `any` casts or `// @ts-ignore` without justification.
- Never call the backend directly from components — all calls go through helpers in `src/lib/api.ts`.
- No `NEXT_PUBLIC_*` secret; never expose API keys in client bundle. Web search uses the backend's server-side `SERP_API_KEY`.
- Components: one PascalCase component per file in `src/components/`; `'use client'` at top for client components; do not hand-edit `src/components/ui/*` (shadcn auto-generated).
- Backend changes must pass `uv run ruff check .` and `uv run ty check app/backend` (from root `CLAUDE.md`).
- Frontend unit tests are OUT OF SCOPE (no test runner configured in `nextjs-frontend/`); verification is `lint` + `build` + the manual checklist in Task 12. Backend changes verified via `ruff`/`ty` + manual.
- Test light AND dark themes for every new/changed UI.

---

## File Structure

| File | Responsibility |
|------|----------------|
| `src/lib/types.ts` (modify) | Add `ConversationMode`, `Conversation`; add `use_web_search` to `RagQueryRequest`. |
| `src/lib/store.ts` (rewrite) | `conversations: Conversation[]`, `activeId`; actions to create/select/delete/rename/clear conversations and append/update messages; persist `conversations`, `activeId`, `settings`. |
| `src/components/layout/sidebar.tsx` (rewrite) | Brand, New conversation, Chat\|RAG mode switch, unified conversation list, footer (Settings + theme). Renders the Settings panel. |
| `src/components/settings-panel.tsx` (create) | Radix `Dialog` slide-over: Models, Generation, Backend keys, Knowledge Base sections. |
| `src/components/knowledge-base.tsx` (create) | Extracted KB UI: status card, file list, upload zone, reset (reused by the settings panel). |
| `src/app/layout.tsx` (modify) | Remove the redundant global `<header>` strip. |
| `src/app/chat/page.tsx` (rewrite) | `ChatView`: reads active chat conversation; keeps GenUI/markdown/streaming; per-conversation web-search + voice toggles; removes in-page settings panel. |
| `src/app/rag/page.tsx` (rewrite) | `RagView`: removes file management; Q&A list + composer with web-search toggle; compact options row. |
| `src/app/web-search/page.tsx` (delete) | Remove the standalone web-search page. |
| `src/app/settings/page.tsx` (delete) | Remove the standalone settings page (now a panel). |
| `src/lib/api.ts` (modify) | `RagQueryRequest.use_web_search` forwarded by `streamRagQuery`. |
| `app/backend/models/api_models.py` (modify) | Add `use_web_search` to `RAGQueryRequestEnhanced`. |
| `app/backend/api/rag.py` (modify) | When `use_web_search`, fetch web results via `WebSearchService` and merge into the RAG context before streaming. |

---

## Task 1: Conversation data model

**Files:**
- Modify: `nextjs-frontend/src/lib/types.ts`

**Interfaces:**
- Produces: `ConversationMode`, `Conversation` (used by store, sidebar, chat/rag views); `RagQueryRequest.use_web_search` (used by `api.ts` in Task 10).

- [ ] **Step 1: Add the conversation types and the RAG web-search field**

Open `nextjs-frontend/src/lib/types.ts`. After the `UIMessage` interface (around line 209–218), add:

```ts
// ─── Conversations ──────────────────────────────────────────────────────

export type ConversationMode = 'chat' | 'rag';

export interface Conversation {
  id: string;
  title: string;
  mode: ConversationMode;
  messages: UIMessage[];
  webSearch: boolean;
  createdAt: string;
  updatedAt: string;
}
```

In the existing `RagQueryRequest` interface (around line 91), add one field:

```ts
export interface RagQueryRequest {
  query: string;
  model: string;
  session_id?: string;
  temperature?: number;
  max_tokens?: number;
  top_k?: number;
  use_web_search?: boolean; // NEW — combine doc retrieval with web search
}
```

- [ ] **Step 2: Type-check the change**

Run: `cd nextjs-frontend && npx tsc --noEmit`
Expected: no errors referencing the new types.

---

## Task 2: Zustand store rewrite

**Files:**
- Rewrite: `nextjs-frontend/src/lib/store.ts`

**Interfaces:**
- Consumes: `Conversation`, `ConversationMode`, `UIMessage` (Task 1); `AppSettings` (unchanged).
- Produces: `useAppStore` with `conversations`, `activeId`, and actions `createConversation`, `selectConversation`, `deleteConversation`, `renameConversation`, `addMessage`, `updateLastMessage`, `setWebSearch`, `clearConversation`, `setSettings`. Used by sidebar (Task 5), chat (Task 7), rag (Task 8).

- [ ] **Step 1: Rewrite the store**

Replace the contents of `nextjs-frontend/src/lib/store.ts` with:

```ts
/**
 * Global Zustand store for LiteMindUI.
 *
 * Holds application settings, a collection of conversations (chat + RAG),
 * and RAG file metadata. Conversations + settings are persisted to
 * localStorage so the history survives reloads.
 */

import { create } from 'zustand';
import { persist, createJSONStorage } from 'zustand/middleware';
import { v4 as uuidv4 } from 'uuid';
import type { AppSettings, UIMessage, RagFile, Conversation, ConversationMode } from './types';

// ─── Default settings ─────────────────────────────────────────────────────────

const DEFAULT_SETTINGS: AppSettings = {
  model: '',
  backend: 'ollama',
  temperature: 0.7,
  maxTokens: 2048,
  topP: 0.9,
  frequencyPenalty: 0,
  repetitionPenalty: 1.0,
  sessionId: uuidv4(),
  enableGenerativeUI: true,
  genUIDisplayMode: 'rendered',
  voiceMode: false,
};

// ─── Store shape ──────────────────────────────────────────────────────────────

export interface AppStore {
  // ── State ──────────────────────────────────────────────────────────────────
  settings: AppSettings;
  conversations: Conversation[];
  activeId: string | null;
  ragFiles: RagFile[];

  // ── Conversation actions ─────────────────────────────────────────────────────
  createConversation: (mode: ConversationMode) => string;
  selectConversation: (id: string) => void;
  deleteConversation: (id: string) => void;
  renameConversation: (id: string, title: string) => void;
  clearConversation: (id: string) => void;
  setWebSearch: (id: string, value: boolean) => void;

  addMessage: (convId: string, message: Omit<UIMessage, 'id' | 'createdAt'>) => void;
  updateLastMessage: (convId: string, content: string, isStreaming?: boolean) => void;

  setSettings: (partial: Partial<AppSettings>) => void;
  setRagFiles: (files: RagFile[]) => void;
  newSession: () => void;
}

// ─── Helpers ──────────────────────────────────────────────────────────────────

function nowIso(): string {
  return new Date().toISOString();
}

function mostRecentOf(
  conversations: Conversation[],
  mode?: ConversationMode,
): string | null {
  const pool = mode
    ? conversations.filter((c) => c.mode === mode)
    : conversations;
  if (pool.length === 0) return null;
  return [...pool].sort(
    (a, b) => b.updatedAt.localeCompare(a.updatedAt),
  )[0].id;
}

// ─── Store implementation ─────────────────────────────────────────────────────

export const useAppStore = create<AppStore>()(
  persist(
    (set, get) => ({
      settings: DEFAULT_SETTINGS,
      conversations: [],
      activeId: null,
      ragFiles: [],

      createConversation: (mode) => {
        const id = uuidv4();
        const ts = nowIso();
        const conv: Conversation = {
          id,
          title: 'New conversation',
          mode,
          messages: [],
          webSearch: false,
          createdAt: ts,
          updatedAt: ts,
        };
        set((state) => ({
          conversations: [conv, ...state.conversations],
          activeId: id,
        }));
        return id;
      },

      selectConversation: (id) => set({ activeId: id }),

      deleteConversation: (id) => {
        const remaining = get().conversations.filter((c) => c.id !== id);
        const wasActive = get().activeId === id;
        set({
          conversations: remaining,
          activeId: wasActive ? mostRecentOf(remaining) : get().activeId,
        });
      },

      renameConversation: (id, title) =>
        set((state) => ({
          conversations: state.conversations.map((c) =>
            c.id === id ? { ...c, title: title.trim() || c.title } : c,
          ),
        })),

      clearConversation: (id) =>
        set((state) => ({
          conversations: state.conversations.map((c) =>
            c.id === id ? { ...c, messages: [] } : c,
          ),
        })),

      setWebSearch: (id, value) =>
        set((state) => ({
          conversations: state.conversations.map((c) =>
            c.id === id ? { ...c, webSearch: value } : c,
          ),
        })),

      addMessage: (convId, message) =>
        set((state) => ({
          conversations: state.conversations.map((c) => {
            if (c.id !== convId) return c;
            const msg: UIMessage = {
              ...message,
              id: uuidv4(),
              createdAt: nowIso(),
            };
            // Auto-title from the first user message
            const title =
              c.messages.length === 0 && message.role === 'user'
                ? message.content.slice(0, 48) ||
                  (message.content.length > 48
                    ? `${message.content.slice(0, 48)}…`
                    : message.content) || 'New conversation'
                : c.title;
            return {
              ...c,
              title,
              messages: [...c.messages, msg],
              updatedAt: nowIso(),
            };
          }),
        })),

      updateLastMessage: (convId, content, isStreaming = false) =>
        set((state) => ({
          conversations: state.conversations.map((c) => {
            if (c.id !== convId) return c;
            const messages = [...c.messages];
            for (let i = messages.length - 1; i >= 0; i--) {
              if (messages[i].role === 'assistant') {
                messages[i] = { ...messages[i], content, isStreaming };
                return { ...c, messages, updatedAt: nowIso() };
              }
            }
            return c;
          }),
        })),

      setSettings: (partial) =>
        set((state) => ({ settings: { ...state.settings, ...partial } })),

      setRagFiles: (files) => set({ ragFiles: files }),

      newSession: () =>
        set((state) => ({
          settings: { ...state.settings, sessionId: uuidv4() },
        })),
    }),
    {
      name: 'litemind-store',
      storage: createJSONStorage(() =>
        typeof window !== 'undefined'
          ? window.localStorage
          : {
              getItem: () => null,
              setItem: () => {},
              removeItem: () => {},
            },
      ),
      partialize: (state) => ({
        settings: state.settings,
        conversations: state.conversations,
        activeId: state.activeId,
        ragFiles: state.ragFiles,
      }),
    },
  ),
);

// ─── Convenience selectors ────────────────────────────────────────────────────

export const selectSettings = (state: AppStore) => state.settings;
export const selectConversations = (state: AppStore) => state.conversations;
export const selectActiveId = (state: AppStore) => state.activeId;
export const selectActiveConversation = (state: AppStore) =>
  state.conversations.find((c) => c.id === state.activeId) ?? null;
export const selectRagFiles = (state: AppStore) => state.ragFiles;
```

> Note: the `addMessage` auto-title logic intentionally falls back to `'New conversation'` for empty content; this keeps the title non-empty and avoids persisting a blank string.

- [ ] **Step 2: Type-check**

Run: `cd nextjs-frontend && npx tsc --noEmit`
Expected: no errors.

---

## Task 3: Extract Knowledge Base UI component

**Files:**
- Create: `nextjs-frontend/src/components/knowledge-base.tsx`

**Interfaces:**
- Consumes: `useAppStore` (`ragFiles`, `setRagFiles`); `api.ts` helpers `getRagFiles`, `getRagStatus`, `uploadRagFile`, `deleteRagFile`, `resetRag`, `checkRagDuplicate` (all already exist).
- Produces: `<KnowledgeBaseSection />` used by the settings panel (Task 4).

- [ ] **Step 1: Create the Knowledge Base section component**

Create `nextjs-frontend/src/components/knowledge-base.tsx`:

```tsx
'use client';

import * as React from 'react';
import {
  Upload,
  FileText,
  Trash2,
  RefreshCw,
  Database,
  AlertTriangle,
  CheckCircle2,
} from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Label } from '@/components/ui/label';
import { Progress } from '@/components/ui/progress';
import { Separator } from '@/components/ui/separator';
import { ScrollArea } from '@/components/ui/scroll-area';
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
  DialogTrigger,
} from '@/components/ui/dialog';
import { useAppStore } from '@/lib/store';
import {
  getRagFiles,
  getRagStatus,
  uploadRagFile,
  deleteRagFile,
  resetRag,
  checkRagDuplicate,
} from '@/lib/api';
import type { RagFile, RagStatusResponse } from '@/lib/types';
import { cn } from '@/lib/utils';

const ACCEPTED_TYPES =
  '.pdf,.docx,.txt,.md,.csv,.xlsx,.pptx,.html,.htm,.odt,.rtf,.yaml,.json';

function StatusCard({
  status,
  loading,
  onRefresh,
}: {
  status: RagStatusResponse | null;
  loading: boolean;
  onRefresh: () => void;
}) {
  return (
    <div className="rounded-lg border border-border bg-card p-3">
      <div className="flex items-center justify-between">
        <span className="flex items-center gap-2 text-sm font-medium">
          <Database className="h-4 w-4 text-muted-foreground" aria-hidden="true" />
          Knowledge Base
        </span>
        <Button
          variant="ghost"
          size="icon"
          onClick={onRefresh}
          disabled={loading}
          aria-label="Refresh status"
          className="h-7 w-7"
        >
          <RefreshCw className={cn('h-3.5 w-3.5', loading && 'animate-spin')} aria-hidden="true" />
        </Button>
      </div>
      {status ? (
        <div className="mt-2 grid grid-cols-3 gap-2 text-xs">
          <div>
            <p className="text-muted-foreground">Status</p>
            <p className="flex items-center gap-1 font-medium capitalize">
              {status.status === 'ready' ? (
                <CheckCircle2 className="h-3.5 w-3.5 text-green-500" aria-hidden="true" />
              ) : (
                <AlertTriangle className="h-3.5 w-3.5 text-yellow-500" aria-hidden="true" />
              )}
              {status.status}
            </p>
          </div>
          <div>
            <p className="text-muted-foreground">Files</p>
            <p className="font-medium">{status.files_count ?? 0}</p>
          </div>
          <div>
            <p className="text-muted-foreground">Chunks</p>
            <p className="font-medium">{status.chunks_count ?? '—'}</p>
          </div>
        </div>
      ) : (
        <p className="mt-2 text-xs text-muted-foreground">Loading…</p>
      )}
    </div>
  );
}

function FileRow({
  file,
  onDelete,
  deleting,
}: {
  file: RagFile;
  onDelete: (name: string) => void;
  deleting: boolean;
}) {
  return (
    <div className="flex items-center justify-between rounded-lg border border-border bg-card px-3 py-2">
      <div className="flex min-w-0 items-center gap-2.5">
        <FileText className="h-4 w-4 shrink-0 text-muted-foreground" aria-hidden="true" />
        <p className="truncate text-sm font-medium">{file.filename}</p>
      </div>
      <Button
        variant="ghost"
        size="icon"
        onClick={() => onDelete(file.filename)}
        disabled={deleting}
        aria-label={`Delete ${file.filename}`}
        className="h-7 w-7 shrink-0 text-muted-foreground hover:text-destructive"
      >
        <Trash2 className="h-3.5 w-3.5" aria-hidden="true" />
      </Button>
    </div>
  );
}

export function KnowledgeBaseSection() {
  const { ragFiles, setRagFiles } = useAppStore();
  const [files, setFiles] = React.useState<RagFile[]>([]);
  const [status, setStatus] = React.useState<RagStatusResponse | null>(null);
  const [statusLoading, setStatusLoading] = React.useState(false);
  const [uploading, setUploading] = React.useState(false);
  const [uploadProgress, setUploadProgress] = React.useState(0);
  const [dragActive, setDragActive] = React.useState(false);
  const [fileError, setFileError] = React.useState('');
  const [deleting, setDeleting] = React.useState<string | null>(null);
  const [resetOpen, setResetOpen] = React.useState(false);
  const fileInputRef = React.useRef<HTMLInputElement>(null);

  const loadFiles = React.useCallback(async () => {
    try {
      const res = await getRagFiles();
      setFiles(res.files ?? []);
      setRagFiles(res.files ?? []);
    } catch {
      setFiles([]);
    }
  }, [setRagFiles]);

  const loadStatus = React.useCallback(async () => {
    setStatusLoading(true);
    try {
      const res = await getRagStatus();
      setStatus(res);
    } catch {
      setStatus(null);
    } finally {
      setStatusLoading(false);
    }
  }, []);

  React.useEffect(() => {
    void loadFiles();
    void loadStatus();
  }, [loadFiles, loadStatus]);

  const processUpload = React.useCallback(
    async (fileList: FileList | File[]) => {
      const arr = Array.from(fileList);
      if (arr.length === 0) return;
      setUploading(true);
      setUploadProgress(0);
      setFileError('');
      for (let i = 0; i < arr.length; i++) {
        const file = arr[i];
        try {
          const dup = await checkRagDuplicate({ filename: file.name });
          if (dup.is_duplicate) {
            setFileError(`"${file.name}" already exists.`);
            continue;
          }
          await uploadRagFile(file);
          setUploadProgress(Math.round(((i + 1) / arr.length) * 100));
        } catch (err) {
          setFileError(`${file.name}: ${err instanceof Error ? err.message : 'Upload failed'}`);
        }
      }
      setUploading(false);
      await loadFiles();
      await loadStatus();
    },
    [loadFiles, loadStatus],
  );

  const handleDelete = async (name: string) => {
    setDeleting(name);
    try {
      await deleteRagFile(name);
      await loadFiles();
      await loadStatus();
    } catch (err) {
      console.error('Delete failed', err);
    } finally {
      setDeleting(null);
    }
  };

  const handleReset = async () => {
    setResetOpen(false);
    try {
      await resetRag();
      await loadFiles();
      await loadStatus();
    } catch (err) {
      console.error('Reset failed', err);
    }
  };

  return (
    <div className="space-y-3">
      <StatusCard status={status} loading={statusLoading} onRefresh={() => void loadStatus()} />

      <div
        role="button"
        tabIndex={0}
        aria-label="Upload documents – click or drag and drop"
        onKeyDown={(e) => {
          if (e.key === 'Enter' || e.key === ' ') fileInputRef.current?.click();
        }}
        onClick={() => fileInputRef.current?.click()}
        onDragOver={(e) => { e.preventDefault(); setDragActive(true); }}
        onDragLeave={() => setDragActive(false)}
        onDrop={(e) => {
          e.preventDefault();
          setDragActive(false);
          if (e.dataTransfer.files.length > 0) void processUpload(e.dataTransfer.files);
        }}
        className={cn(
          'flex cursor-pointer flex-col items-center justify-center gap-2 rounded-lg border-2 border-dashed px-4 py-6 text-center transition-colors',
          dragActive ? 'border-primary bg-primary/5' : 'border-border hover:border-primary/50',
        )}
      >
        <Upload className="h-7 w-7 text-muted-foreground" aria-hidden="true" />
        <p className="text-sm font-medium">
          {uploading ? 'Uploading…' : 'Click or drag to upload'}
        </p>
        <p className="text-xs text-muted-foreground">PDF, DOCX, TXT, MD, CSV, XLSX, and more</p>
        {uploading && <Progress value={uploadProgress} className="mt-1 h-1.5 w-full" />}
      </div>

      <input
        ref={fileInputRef}
        type="file"
        accept={ACCEPTED_TYPES}
        multiple
        className="hidden"
        onChange={(e) => {
          if (e.target.files) void processUpload(e.target.files);
          e.target.value = '';
        }}
        aria-hidden="true"
      />

      {fileError && (
        <p className="text-xs text-destructive" role="alert">{fileError}</p>
      )}

      <Separator />

      <div className="space-y-2">
        <Label className="text-xs font-medium text-muted-foreground uppercase tracking-wide">
          Uploaded files ({files.length})
        </Label>
        {files.length === 0 ? (
          <p className="text-xs text-muted-foreground">No files uploaded yet.</p>
        ) : (
          files.map((f) => (
            <FileRow
              key={f.filename}
              file={f}
              onDelete={(n) => void handleDelete(n)}
              deleting={deleting === f.filename}
            />
          ))
        )}
      </div>

      <Dialog open={resetOpen} onOpenChange={setResetOpen}>
        <DialogTrigger className="w-full rounded-md border border-destructive/40 px-3 py-1.5 text-xs font-medium text-destructive hover:bg-destructive/10">
          <Trash2 className="mr-2 h-3.5 w-3.5 inline" aria-hidden="true" />
          Reset knowledge base
        </DialogTrigger>
        <DialogContent>
          <DialogHeader>
            <DialogTitle>Reset knowledge base?</DialogTitle>
            <DialogDescription>
              This permanently deletes all uploaded files and indexed chunks. This cannot be undone.
            </DialogDescription>
          </DialogHeader>
          <DialogFooter>
            <Button variant="outline" onClick={() => setResetOpen(false)}>Cancel</Button>
            <Button variant="destructive" onClick={() => void handleReset()}>Reset</Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </div>
  );
}
```

- [ ] **Step 2: Type-check**

Run: `cd nextjs-frontend && npx tsc --noEmit`
Expected: no errors.

---

## Task 4: Settings panel (consolidated slide-over)

**Files:**
- Create: `nextjs-frontend/src/components/settings-panel.tsx`
- Consumes (Task 3): `<KnowledgeBaseSection />`

**Interfaces:**
- Consumes: `useAppStore` (`settings`, `setSettings`); `api.ts` `getEnhancedModels`; `<KnowledgeBaseSection />` (Task 3).
- Produces: `<SettingsPanel open onClose />` rendered by the Sidebar (Task 5).

- [ ] **Step 1: Create the settings panel**

Create `nextjs-frontend/src/components/settings-panel.tsx`:

```tsx
'use client';

import * as React from 'react';
import { X } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Slider } from '@/components/ui/slider';
import { Separator } from '@/components/ui/separator';
import { ScrollArea } from '@/components/ui/scroll-area';
import {
  Dialog,
  DialogContent,
  DialogTitle,
} from '@/components/ui/dialog';
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select';
import { KnowledgeBaseSection } from '@/components/knowledge-base';
import { useAppStore } from '@/lib/store';
import { getEnhancedModels } from '@/lib/api';
import type { Model, BackendType } from '@/lib/types';
import { cn } from '@/lib/utils';

function Section({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <section className="space-y-3">
      <h3 className="text-xs font-semibold uppercase tracking-wide text-muted-foreground">
        {title}
      </h3>
      {children}
    </section>
  );
}

export function SettingsPanel({
  open,
  onClose,
}: {
  open: boolean;
  onClose: () => void;
}) {
  const { settings, setSettings } = useAppStore();
  const [localModels, setLocalModels] = React.useState<Model[]>([]);
  const [cloudModels, setCloudModels] = React.useState<Model[]>([]);

  React.useEffect(() => {
    getEnhancedModels()
      .then((r) => {
        setLocalModels(r.local_models ?? []);
        setCloudModels(r.cloud_models ?? []);
      })
      .catch(() => {
        setLocalModels([]);
        setCloudModels([]);
      });
  }, []);

  const isOllama = settings.backend === 'ollama';
  const needsApiKey = settings.backend === 'openrouter' || settings.backend === 'nvidia_nim';
  const allOllama = [
    ...localModels.map((m) => ({ ...m, is_local: true as const })),
    ...cloudModels.map((m) => ({ ...m, is_local: false as const })),
  ];

  return (
    <Dialog open={open} onOpenChange={(o) => !o && onClose()}>
      <DialogContent className="left-0 top-0 translate-x-0 translate-y-0 h-full max-w-sm w-full rounded-none border-r border-border p-0 gap-0 flex flex-col data-[state=closed]:slide-out-to-left data-[state=open]:slide-in-from-left">
        <div className="flex items-center justify-between border-b border-border px-4 py-3">
          <DialogTitle className="text-sm font-semibold">Settings</DialogTitle>
          <Button variant="ghost" size="icon" onClick={onClose} aria-label="Close settings" className="h-7 w-7">
            <X className="h-4 w-4" aria-hidden="true" />
          </Button>
        </div>

        <ScrollArea className="flex-1">
          <div className="space-y-6 p-4">
            <Section title="Models">
              <div className="space-y-2">
                <Label htmlFor="set-backend" className="text-xs">Backend</Label>
                <Select
                  value={settings.backend}
                  onValueChange={(v) => v && setSettings({ backend: v as BackendType, model: '' })}
                >
                  <SelectTrigger id="set-backend" className="h-8 text-sm">
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="ollama">🖥️ Ollama (local)</SelectItem>
                    <SelectItem value="openrouter">☁️ OpenRouter</SelectItem>
                    <SelectItem value="nvidia_nim">⚡ Nvidia NIM</SelectItem>
                  </SelectContent>
                </Select>
              </div>

              <div className="space-y-2">
                <Label htmlFor={isOllama ? 'set-model-sel' : 'set-model-in'} className="text-xs">Model</Label>
                {isOllama ? (
                  <Select value={settings.model} onValueChange={(v) => v && setSettings({ model: v })}>
                    <SelectTrigger id="set-model-sel" className="h-8 text-sm">
                      <SelectValue placeholder={allOllama.length ? 'Select model…' : 'No models'} />
                    </SelectTrigger>
                    <SelectContent>
                      {allOllama.length === 0 ? (
                        <SelectItem value="__none__" disabled>Ollama unreachable</SelectItem>
                      ) : (
                        allOllama.map((m) => (
                          <SelectItem key={m.name} value={m.name} className="text-sm">
                            {m.is_local ? '🟢 ' : '☁️ '}
                            {m.name}
                          </SelectItem>
                        ))
                      )}
                    </SelectContent>
                  </Select>
                ) : (
                  <Input
                    id="set-model-in"
                    type="text"
                    placeholder={settings.backend === 'openrouter' ? 'e.g. openai/gpt-4o' : 'e.g. meta/llama-3.3-70b-instruct'}
                    value={settings.model}
                    onChange={(e) => setSettings({ model: e.target.value })}
                    className="h-8 text-sm font-mono"
                  />
                )}
              </div>
            </Section>

            <Separator />

            <Section title="Generation">
              <div className="space-y-2">
                <div className="flex items-center justify-between">
                  <Label className="text-xs">Temperature</Label>
                  <span className="text-xs font-mono">{settings.temperature.toFixed(1)}</span>
                </div>
                <Slider min={0} max={2} step={0.1} value={[settings.temperature]}
                  onValueChange={(v) => setSettings({ temperature: Array.isArray(v) ? (v[0] as number) : (v as number) })}
                  aria-label="Temperature" />
              </div>
              <div className="space-y-2">
                <Label htmlFor="set-maxtok" className="text-xs">Max tokens</Label>
                <Input id="set-maxtok" type="number" min={128} max={32768} step={128}
                  value={settings.maxTokens}
                  onChange={(e) => setSettings({ maxTokens: parseInt(e.target.value) || 2048 })}
                  className="h-8 text-sm" />
              </div>
              <div className="space-y-2">
                <div className="flex items-center justify-between">
                  <Label className="text-xs">Top-P</Label>
                  <span className="text-xs font-mono">{settings.topP.toFixed(2)}</span>
                </div>
                <Slider min={0} max={1} step={0.05} value={[settings.topP]}
                  onValueChange={(v) => setSettings({ topP: Array.isArray(v) ? (v[0] as number) : (v as number) })}
                  aria-label="Top-P" />
              </div>
            </Section>

            <Separator />

            <Section title="Backend keys">
              <div className="space-y-2">
                <Label htmlFor="set-ollama" className="text-xs">Ollama URL</Label>
                <Input id="set-ollama" placeholder="http://localhost:11434"
                  value={settings.ollamaUrl ?? ''} onChange={(e) => setSettings({ ollamaUrl: e.target.value })} className="h-8 text-sm" />
              </div>
              {needsApiKey && (
                <div className="space-y-2">
                  <Label htmlFor="set-key" className="text-xs">
                    {settings.backend === 'openrouter' ? 'OpenRouter API key' : 'Nvidia NIM API key'}
                  </Label>
                  <Input id="set-key" type="password" placeholder={settings.backend === 'openrouter' ? 'sk-or-…' : 'nvapi-…'}
                    value={settings.apiKey ?? ''} onChange={(e) => setSettings({ apiKey: e.target.value })} className="h-8 text-sm" />
                </div>
              )}
              <p className="text-[11px] text-muted-foreground">
                Web search uses the server-side SERP_API_KEY — no client key needed.
              </p>
            </Section>

            <Separator />

            <Section title="Knowledge Base">
              <KnowledgeBaseSection />
            </Section>
          </div>
        </ScrollArea>
      </DialogContent>
    </Dialog>
  );
}
```

> The `DialogContent` className reuses shadcn's dialog but pins it to the left edge as a full-height slide-over (`slide-in-from-left` / `slide-out-to-left` are provided by `tw-animate-css`, already imported in `globals.css`).

- [ ] **Step 2: Type-check**

Run: `cd nextjs-frontend && npx tsc --noEmit`
Expected: no errors.

---

## Task 5: Sidebar rewrite (conversation hub)

**Files:**
- Rewrite: `nextjs-frontend/src/components/layout/sidebar.tsx`
- Consumes (Task 4): `<SettingsPanel open onClose />`
- Consumes (Task 2): `useAppStore` (`conversations`, `activeId`, `createConversation`, `selectConversation`, `deleteConversation`)

**Interfaces:**
- Produces: the global sidebar. Clicking a conversation navigates to its mode route and selects it. The "New conversation" button derives the mode from the current route.

- [ ] **Step 1: Rewrite the sidebar**

Replace `nextjs-frontend/src/components/layout/sidebar.tsx` with:

```tsx
'use client';

import * as React from 'react';
import Link from 'next/link';
import { usePathname, useRouter } from 'next/navigation';
import {
  MessageSquare,
  Database,
  Settings,
  BrainCircuit,
  Menu,
  X,
  Plus,
  Trash2,
} from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Separator } from '@/components/ui/separator';
import { ScrollArea } from '@/components/ui/scroll-area';
import {
  Tooltip,
  TooltipContent,
  TooltipTrigger,
} from '@/components/ui/tooltip';
import { ThemeToggle } from '@/components/theme-toggle';
import { SettingsPanel } from '@/components/settings-panel';
import { useAppStore } from '@/lib/store';
import { cn } from '@/lib/utils';
import type { ConversationMode } from '@/lib/types';

function relativeTime(iso: string): string {
  const then = new Date(iso).getTime();
  if (Number.isNaN(then)) return '';
  const diff = Date.now() - then;
  const min = Math.round(diff / 60000);
  if (min < 1) return 'now';
  if (min < 60) return `${min}m`;
  const hr = Math.round(min / 60);
  if (hr < 24) return `${hr}h`;
  const day = Math.round(hr / 24);
  if (day < 7) return `${day}d`;
  return new Date(iso).toLocaleDateString(undefined, { month: 'short', day: 'numeric' });
}

export function Sidebar(): React.ReactElement {
  const pathname = usePathname();
  const router = useRouter();
  const { conversations, activeId, createConversation, selectConversation, deleteConversation } =
    useAppStore();

  const [mobileOpen, setMobileOpen] = React.useState(false);
  const [settingsOpen, setSettingsOpen] = React.useState(false);
  const [confirmDelete, setConfirmDelete] = React.useState<string | null>(null);

  const activeMode: ConversationMode = pathname.startsWith('/rag') ? 'rag' : 'chat';
  const closeMobile = React.useCallback(() => setMobileOpen(false), []);

  React.useEffect(() => {
    closeMobile();
  }, [pathname, closeMobile]);

  React.useEffect(() => {
    const handler = (e: KeyboardEvent) => {
      if (e.key === 'Escape') {
        if (settingsOpen) setSettingsOpen(false);
        else closeMobile();
      }
    };
    document.addEventListener('keydown', handler);
    return () => document.removeEventListener('keydown', handler);
  }, [settingsOpen, closeMobile]);

  const handleNew = () => {
    const id = createConversation(activeMode);
    router.push(activeMode === 'rag' ? '/rag' : '/chat');
    closeMobile();
    void id;
  };

  const sorted = [...conversations].sort((a, b) => b.updatedAt.localeCompare(a.updatedAt));

  return (
    <>
      <Button
        variant="ghost"
        size="icon"
        className="fixed left-4 top-4 z-50 md:hidden"
        aria-label={mobileOpen ? 'Close navigation' : 'Open navigation'}
        aria-expanded={mobileOpen}
        aria-controls="sidebar-nav"
        onClick={() => setMobileOpen((p) => !p)}
      >
        {mobileOpen ? <X className="h-5 w-5" aria-hidden="true" /> : <Menu className="h-5 w-5" aria-hidden="true" />}
      </Button>

      {mobileOpen && (
        <div className="fixed inset-0 z-30 bg-black/40 md:hidden" aria-hidden="true" onClick={closeMobile} />
      )}

      <nav
        id="sidebar-nav"
        role="navigation"
        aria-label="Main navigation"
        className={cn(
          'fixed inset-y-0 left-0 z-40 flex w-64 flex-col border-r border-border bg-sidebar transition-transform duration-200 ease-in-out',
          mobileOpen ? 'translate-x-0' : '-translate-x-full',
          'md:relative md:translate-x-0 md:transition-none',
        )}
      >
        {/* Brand */}
        <div className="flex h-16 shrink-0 items-center gap-3 px-4">
          <div className="flex h-9 w-9 items-center justify-center rounded-lg bg-primary" aria-hidden="true">
            <BrainCircuit className="h-5 w-5 text-primary-foreground" />
          </div>
          <div className="flex flex-col">
            <span className="text-sm font-semibold leading-tight text-sidebar-foreground">LiteMind</span>
            <span className="text-[11px] leading-tight text-muted-foreground">AI Workspace</span>
          </div>
        </div>

        <div className="px-3">
          <Button onClick={handleNew} className="w-full justify-start gap-2" size="sm">
            <Plus className="h-4 w-4" aria-hidden="true" />
            New conversation
          </Button>
        </div>

        <Separator className="my-3" />

        {/* Mode switch */}
        <div className="px-3">
          <div role="group" aria-label="Conversation mode" className="grid grid-cols-2 gap-1 rounded-lg border border-border bg-muted/40 p-1">
            <Link
              href="/chat"
              aria-current={activeMode === 'chat' ? 'page' : undefined}
              className={cn(
                'flex items-center justify-center gap-2 rounded-md px-3 py-2 text-sm font-medium transition-colors',
                activeMode === 'chat'
                  ? 'bg-sidebar-primary text-sidebar-primary-foreground'
                  : 'text-sidebar-foreground hover:bg-sidebar-accent',
              )}
            >
              <MessageSquare className="h-4 w-4" aria-hidden="true" /> Chat
            </Link>
            <Link
              href="/rag"
              aria-current={activeMode === 'rag' ? 'page' : undefined}
              className={cn(
                'flex items-center justify-center gap-2 rounded-md px-3 py-2 text-sm font-medium transition-colors',
                activeMode === 'rag'
                  ? 'bg-sidebar-primary text-sidebar-primary-foreground'
                  : 'text-sidebar-foreground hover:bg-sidebar-accent',
              )}
            >
              <Database className="h-4 w-4" aria-hidden="true" /> RAG
            </Link>
          </div>
        </div>

        <Separator className="my-3" />

        {/* Conversation list */}
        <ScrollArea className="flex-1 px-2">
          {sorted.length === 0 ? (
            <p className="px-2 py-4 text-xs text-muted-foreground">No conversations yet.</p>
          ) : (
            <ul className="space-y-1">
              {sorted.map((c) => {
                const isActive = c.id === activeId;
                return (
                  <li key={c.id} className="group relative">
                    <button
                      onClick={() => {
                        selectConversation(c.id);
                        router.push(c.mode === 'rag' ? '/rag' : '/chat');
                        closeMobile();
                      }}
                      aria-current={isActive ? 'page' : undefined}
                      className={cn(
                        'flex w-full items-center gap-2 rounded-md px-3 py-2 text-left text-sm transition-colors',
                        isActive
                          ? 'bg-sidebar-primary text-sidebar-primary-foreground'
                          : 'text-sidebar-foreground hover:bg-sidebar-accent',
                      )}
                    >
                      <span aria-hidden="true" className="shrink-0">
                        {c.mode === 'chat' ? '💬' : '📚'}
                      </span>
                      <span className="min-w-0 flex-1">
                        <span className="block truncate">{c.title}</span>
                        <span className="block text-[11px] opacity-70">{relativeTime(c.updatedAt)}</span>
                      </span>
                    </button>
                    <button
                      onClick={(e) => { e.stopPropagation(); setConfirmDelete(c.id); }}
                      aria-label={`Delete ${c.title}`}
                      className="absolute right-1 top-1/2 hidden -translate-y-1/2 rounded p-1 text-muted-foreground hover:text-destructive group-hover:block"
                    >
                      <Trash2 className="h-3.5 w-3.5" aria-hidden="true" />
                    </button>
                  </li>
                );
              })}
            </ul>
          )}
        </ScrollArea>

        {confirmDelete && (
          <div className="border-t border-border bg-card p-3 text-xs">
            <p className="mb-2 text-foreground">Delete this conversation?</p>
            <div className="flex gap-2">
              <Button size="sm" variant="outline" className="flex-1" onClick={() => setConfirmDelete(null)}>Cancel</Button>
              <Button size="sm" variant="destructive" className="flex-1"
                onClick={() => { deleteConversation(confirmDelete); setConfirmDelete(null); }}>
                Delete
              </Button>
            </div>
          </div>
        )}

        <Separator className="shrink-0" />

        {/* Footer */}
        <div className="flex shrink-0 items-center justify-between px-4 py-3">
          <button
            onClick={() => setSettingsOpen(true)}
            className="flex items-center gap-2 rounded-md px-2 py-1.5 text-sm text-sidebar-foreground hover:bg-sidebar-accent"
            aria-label="Open settings"
          >
            <Settings className="h-4 w-4" aria-hidden="true" />
            Settings
          </button>
          <ThemeToggle />
        </div>
      </nav>

      <SettingsPanel open={settingsOpen} onClose={() => setSettingsOpen(false)} />
    </>
  );
}
```

- [ ] **Step 2: Type-check**

Run: `cd nextjs-frontend && npx tsc --noEmit`
Expected: no errors.

---

## Task 6: Remove the global header from the layout

**Files:**
- Modify: `nextjs-frontend/src/app/layout.tsx`

**Interfaces:**
- Consumes: the rewritten `Sidebar` (Task 5) which now owns branding + mobile hamburger.

- [ ] **Step 1: Remove the header strip**

In `nextjs-frontend/src/app/layout.tsx`, delete the `<header>…</header>` block (lines ~59–73) and the wrapping `<div className="flex flex-1 flex-col overflow-hidden">` wrapper, so the `<main>` sits directly inside the flex container. The result around the body should read:

```tsx
        <ThemeProvider>
          <TooltipProvider delay={300}>
            <div className="flex h-full">
              <Sidebar />
              <main
                className="flex flex-1 flex-col overflow-auto"
                id="main-content"
                tabIndex={-1}
              >
                {children}
              </main>
            </div>
          </TooltipProvider>
        </ThemeProvider>
```

- [ ] **Step 2: Type-check**

Run: `cd nextjs-frontend && npx tsc --noEmit`
Expected: no errors.

---

## Task 7: Chat view (active-conversation driven)

**Files:**
- Rewrite: `nextjs-frontend/src/app/chat/page.tsx`

**Interfaces:**
- Consumes (Task 2): `useAppStore` actions `addMessage`, `updateLastMessage`, `setWebSearch`, `clearConversation`; selectors `selectActiveConversation`, `selectSettings`; `createConversation` for empty state.
- Consumes: `streamChat` / `streamWebSearch` from `src/lib/api.ts` (already exist).

- [ ] **Step 1: Rewrite the chat page**

Replace `nextjs-frontend/src/app/chat/page.tsx` with a `ChatView` that reads the active conversation. Keep the existing `ThinkingDots`, `ModeSelector`, and message-rendering logic, but source messages and the web-search flag from the active conversation rather than local state. Key changes vs. the old page:

  - Remove `SettingsPanel` and the right-side settings panel; remove `localModels`/`cloudModels` (model selection now lives in Settings).
  - Derive `const conv = useAppStore(selectActiveConversation);` and `const activeId = useAppStore(selectActiveId);`.
  - If `!conv`, render an empty state with a "New conversation" button calling `createConversation('chat')`.
  - In `handleSend`, use `conv.webSearch` (instead of a local `webSearch` state) and call `addMessage(activeId, …)` / `updateLastMessage(activeId, …)`.
  - The web-search toggle calls `setWebSearch(activeId, !conv.webSearch)`.
  - "Clear chat" toolbar button calls `clearConversation(activeId)`.
  - The compose loop builds `history` from `conv.messages` (last 20, non-streaming).

The full component (preserving the existing rendering helpers) is:

```tsx
'use client';

import * as React from 'react';
import {
  Send, Globe, Mic, MicOff, Bot, Trash2, Code, Sparkles, Plus,
} from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Textarea } from '@/components/ui/textarea';
import { Tooltip, TooltipContent, TooltipTrigger } from '@/components/ui/tooltip';
import GenerativeUIRenderer from '@/components/generative-ui-renderer';
import MarkdownRenderer from '@/components/markdown-renderer';
import { useAppStore, selectActiveConversation, selectActiveId, selectSettings } from '@/lib/store';
import { streamChat, streamWebSearch } from '@/lib/api';
import type { ChatMessage } from '@/lib/types';
import { cn } from '@/lib/utils';
import { useVoiceInput } from '@/hooks/use-voice-input';

function ThinkingDots() {
  return (
    <div className="flex items-center gap-1 px-1 py-0.5" aria-label="Assistant is thinking" role="status">
      {[0, 1, 2].map((i) => (
        <span key={i} className="h-2 w-2 animate-bounce rounded-full bg-muted-foreground/60" style={{ animationDelay: `${i * 150}ms` }} />
      ))}
    </div>
  );
}

const DISPLAY_MODES = [
  { mode: 'rendered', label: 'Rendered', Icon: Sparkles },
  { mode: 'code', label: 'Code', Icon: Code },
] as const;

function ModeSelector({ mode, onChange }: { mode: 'rendered' | 'code'; onChange: (m: 'rendered' | 'code') => void }) {
  return (
    <div className="flex items-center gap-0.5 rounded-lg border border-border bg-muted/40 p-0.5" role="group" aria-label="Generative UI display mode">
      {DISPLAY_MODES.map(({ mode: m, label, Icon }) => (
        <Button key={m} type="button" variant={mode === m ? 'default' : 'ghost'} size="sm" className="h-7 gap-1.5 px-2.5 text-xs" aria-pressed={mode === m} onClick={() => onChange(m)}>
          <Icon className="h-3.5 w-3.5" aria-hidden="true" />{label}
        </Button>
      ))}
    </div>
  );
}

export default function ChatPage() {
  const conv = useAppStore(selectActiveConversation);
  const activeId = useAppStore(selectActiveId);
  const settings = useAppStore(selectSettings);
  const { addMessage, updateLastMessage, setWebSearch, clearConversation, createConversation } = useAppStore();

  const [input, setInput] = React.useState('');
  const [isStreaming, setIsStreaming] = React.useState(false);
  const [voiceOn, setVoiceOn] = React.useState(false);
  const abortRef = React.useRef<AbortController | null>(null);
  const bottomRef = React.useRef<HTMLDivElement>(null);
  const textareaRef = React.useRef<HTMLTextAreaElement>(null);

  const { state: voiceState, isSupported: voiceSupported, start: startVoice, stop: stopVoice } = useVoiceInput(
    (transcript) => { setInput(transcript); setVoiceOn(false); setTimeout(() => handleSend(transcript), 50); }
  );

  React.useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [conv?.messages]);

  React.useEffect(() => {
    const handler = (e: KeyboardEvent) => { if (e.key === 'Escape' && isStreaming) abortRef.current?.abort(); };
    document.addEventListener('keydown', handler);
    return () => document.removeEventListener('keydown', handler);
  }, [isStreaming]);

  const handleSend = React.useCallback(async (overrideText?: string) => {
    const text = (overrideText ?? input).trim();
    const convId = activeId;
    if (!text || isStreaming || !convId) return;
    const webSearch = useAppStore.getState().conversations.find((c) => c.id === convId)?.webSearch ?? false;

    setInput('');
    setIsStreaming(true);
    addMessage(convId, { role: 'user', content: text });
    addMessage(convId, { role: 'assistant', content: '', isStreaming: true });

    const controller = new AbortController();
    abortRef.current = controller;
    try {
      let accumulated = '';
      const history: ChatMessage[] = useAppStore.getState().conversations
        .find((c) => c.id === convId)?.messages.slice(-20)
        .filter((m) => !m.isStreaming).map((m) => ({ role: m.role, content: m.content })) ?? [];

      const stream = webSearch
        ? streamWebSearch({
            message: text, model: settings.model || undefined, backend: settings.backend,
            api_key: settings.apiKey ?? null, api_base: settings.apiBase ?? null,
            session_id: settings.sessionId, temperature: settings.temperature, max_tokens: settings.maxTokens,
            conversation_history: history.length ? history : null, use_web_search: true,
          }, controller.signal)
        : streamChat({
            message: text, model: settings.model || undefined, backend: settings.backend,
            api_key: settings.apiKey ?? null, api_base: settings.apiBase ?? null,
            temperature: settings.temperature, max_tokens: settings.maxTokens, top_p: settings.topP,
            frequency_penalty: settings.frequencyPenalty, repetition_penalty: settings.repetitionPenalty,
            session_id: settings.sessionId, conversation_history: history.length ? history : null,
            is_voice_mode: settings.voiceMode, enable_generative_ui: settings.enableGenerativeUI, stream: true,
          }, controller.signal);

      for await (const chunk of stream) { accumulated += chunk; updateLastMessage(convId, accumulated, true); }
      updateLastMessage(convId, accumulated, false);
    } catch (err) {
      if (!(err instanceof Error && err.name === 'AbortError')) {
        updateLastMessage(convId, `⚠️ Error: ${err instanceof Error ? err.message : 'An error occurred.'}`, false);
      }
    } finally {
      setIsStreaming(false);
      abortRef.current = null;
      setTimeout(() => textareaRef.current?.focus(), 50);
    }
  }, [input, isStreaming, activeId, settings, addMessage, updateLastMessage]);

  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); void handleSend(); }
  };

  const handleAction = React.useCallback((action: string, payload?: string) => {
    if (action === 'send_message' && payload) void handleSend(payload);
  }, [handleSend]);

  if (!conv || !activeId) {
    return (
      <div className="flex h-full flex-col items-center justify-center gap-3 text-center">
        <div className="flex h-14 w-14 items-center justify-center rounded-2xl bg-primary/10">
          <Bot className="h-7 w-7 text-primary" aria-hidden="true" />
        </div>
        <h2 className="text-xl font-semibold">How can I help you today?</h2>
        <Button onClick={() => createConversation('chat')} className="gap-2">
          <Plus className="h-4 w-4" aria-hidden="true" /> New conversation
        </Button>
      </div>
    );
  }

  const msgs = conv.messages;
  return (
    <div className="flex h-full flex-col overflow-hidden" aria-label="Chat interface">
      {settings.enableGenerativeUI && (
        <div className="flex h-11 shrink-0 items-center justify-between border-b border-border bg-background px-4 md:px-6">
          <span className="text-xs font-medium text-muted-foreground">Generative UI</span>
          <ModeSelector mode={settings.genUIDisplayMode} onChange={(m) => useAppStore.getState().setSettings({ genUIDisplayMode: m })} />
        </div>
      )}

      <div className="flex flex-1 flex-col overflow-y-auto min-h-0">
        <div className="flex flex-col gap-4 px-4 py-6 md:px-8">
          {msgs.length === 0 ? (
            <div className="flex flex-1 flex-col items-center justify-center gap-3 py-24 text-center">
              <div className="flex h-14 w-14 items-center justify-center rounded-2xl bg-primary/10">
                <Bot className="h-7 w-7 text-primary" aria-hidden="true" />
              </div>
              <h2 className="text-xl font-semibold">How can I help you today?</h2>
              <p className="max-w-sm text-sm text-muted-foreground">Ask me anything — I can answer, help you write, analyse documents, or search the web.</p>
            </div>
          ) : (
            msgs.map((msg) => (
              <div key={msg.id} className={cn('flex gap-3', msg.role === 'user' ? 'justify-end' : 'justify-start')} role="article" aria-label={`${msg.role === 'user' ? 'Your' : 'Assistant'} message`}>
                {msg.role === 'assistant' && (
                  <div className="mt-1 flex h-8 w-8 shrink-0 items-center justify-center rounded-full bg-primary/10" aria-hidden="true">
                    <Bot className="h-4 w-4 text-primary" />
                  </div>
                )}
                <div className={cn('max-w-[75%] rounded-2xl px-4 py-3 text-sm', msg.role === 'user' ? 'rounded-br-sm bg-primary text-primary-foreground' : 'rounded-bl-sm border border-border bg-card text-foreground')}>
                  {msg.role === 'user' ? (
                    <p className="whitespace-pre-wrap leading-relaxed">{msg.content}</p>
                  ) : msg.isStreaming && !msg.content ? (
                    <ThinkingDots />
                  ) : settings.enableGenerativeUI ? (
                    settings.genUIDisplayMode === 'rendered' ? (
                      <GenerativeUIRenderer content={msg.content} onAction={handleAction} />
                    ) : (
                      <MarkdownRenderer content={msg.content} />
                    )
                  ) : (
                    <p className="whitespace-pre-wrap leading-relaxed">{msg.content}</p>
                  )}
                  {msg.isStreaming && msg.content && (
                    <span className="ml-0.5 inline-block h-4 w-0.5 animate-pulse bg-current align-middle" aria-hidden="true" />
                  )}
                </div>
              </div>
            ))
          )}
          <div ref={bottomRef} aria-hidden="true" />
        </div>
      </div>

      <div className="border-t border-border bg-background px-4 py-3 md:px-8">
        {conv.webSearch && (
          <div className="mb-2 flex items-center gap-2">
            <span className="rounded-md bg-secondary px-2 py-0.5 text-xs">🌐 Web search active</span>
            <span className="text-xs text-muted-foreground">(Escape to cancel)</span>
          </div>
        )}
        <div className="flex items-end gap-2">
          <Textarea
            ref={textareaRef}
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder={conv.webSearch ? 'Search the web…' : 'Message LiteMindUI…'}
            disabled={isStreaming}
            rows={1}
            className="min-h-[44px] max-h-[200px] flex-1 resize-none overflow-y-auto pr-2 text-sm leading-relaxed"
            aria-label="Chat message input"
          />
          <div className="flex items-center gap-1.5 pb-0.5">
            <Tooltip>
              <TooltipTrigger render={<Button variant={conv.webSearch ? 'default' : 'outline'} size="icon" className="h-9 w-9" onClick={() => setWebSearch(activeId, !conv.webSearch)} aria-label={conv.webSearch ? 'Disable web search' : 'Enable web search'} aria-pressed={conv.webSearch}><Globe className="h-4 w-4" aria-hidden="true" /></Button>} />
              <TooltipContent>Web search</TooltipContent>
            </Tooltip>
            {voiceSupported && (
              <Tooltip>
                <TooltipTrigger render={<Button variant={voiceOn ? 'default' : 'outline'} size="icon" className={cn('h-9 w-9', voiceOn && 'animate-pulse bg-red-500 text-white')} onClick={() => { if (voiceOn) { stopVoice(); setVoiceOn(false); } else { setVoiceOn(true); startVoice(); } }} aria-label={voiceOn ? 'Stop voice input' : 'Start voice input'} aria-pressed={voiceOn} disabled={voiceState === 'processing'}><MicOff className="h-4 w-4" aria-hidden="true" /></Button>} />
                <TooltipContent>{voiceOn ? 'Stop recording' : 'Voice input'}</TooltipContent>
              </Tooltip>
            )}
            <Tooltip>
              <TooltipTrigger render={<Button variant="outline" size="icon" className="h-9 w-9" onClick={() => clearConversation(activeId)} aria-label="Clear chat"><Trash2 className="h-4 w-4" aria-hidden="true" /></Button>} />
              <TooltipContent>Clear chat</TooltipContent>
            </Tooltip>
            <Button size="icon" className="h-9 w-9" onClick={() => void handleSend()} disabled={!input.trim() || isStreaming} aria-label={isStreaming ? 'Sending…' : 'Send message'}>
              <Send className="h-4 w-4" aria-hidden="true" />
            </Button>
          </div>
        </div>
        <p className="mt-1.5 text-center text-[11px] text-muted-foreground">Enter to send · Shift+Enter for newline · Escape to cancel</p>
      </div>
    </div>
  );
}
```

> `Settings2` import is intentionally omitted (the in-page settings panel is gone). The `Settings2`/`X` imports are removed; keep only the icons used above.

- [ ] **Step 2: Type-check**

Run: `cd nextjs-frontend && npx tsc --noEmit`
Expected: no errors.

---

## Task 8: RAG view (remove file mgmt, add web-search toggle)

**Files:**
- Rewrite: `nextjs-frontend/src/app/rag/page.tsx`

**Interfaces:**
- Consumes (Task 2): `useAppStore` `addMessage`, `updateLastMessage`, `setWebSearch`, `clearConversation`, `createConversation`; selectors `selectActiveConversation`, `selectActiveId`, `selectSettings`.
- Consumes: `streamRagQuery` (Task 10 adds `use_web_search` forwarding).

- [ ] **Step 1: Rewrite the RAG page**

Replace `nextjs-frontend/src/app/rag/page.tsx`. Keep the Q&A message rendering and the compact options row (multi-agent, hybrid, n results) as **local state**, but remove ALL file-management UI (now in Settings → Knowledge Base, Task 3/4). Add a web-search toggle bound to `conv.webSearch` that sends `use_web_search: true`. Full component:

```tsx
'use client';

import * as React from 'react';
import { Globe, Database, Plus } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Textarea } from '@/components/ui/textarea';
import { Label } from '@/components/ui/label';
import { Input } from '@/components/ui/input';
import { Switch } from '@/components/ui/switch';
import { Separator } from '@/components/ui/separator';
import { ScrollArea } from '@/components/ui/scroll-area';
import { Tooltip, TooltipContent, TooltipTrigger } from '@/components/ui/tooltip';
import MarkdownRenderer from '@/components/markdown-renderer';
import { useAppStore, selectActiveConversation, selectActiveId, selectSettings } from '@/lib/store';
import { streamRagQuery } from '@/lib/api';
import { cn } from '@/lib/utils';

export default function RagPage() {
  const conv = useAppStore(selectActiveConversation);
  const activeId = useAppStore(selectActiveId);
  const settings = useAppStore(selectSettings);
  const { addMessage, updateLastMessage, setWebSearch, clearConversation, createConversation } = useAppStore();

  const [multiAgent, setMultiAgent] = React.useState(false);
  const [hybridSearch, setHybridSearch] = React.useState(true);
  const [nResults, setNResults] = React.useState(5);
  const [query, setQuery] = React.useState('');
  const [queryLoading, setQueryLoading] = React.useState(false);
  const abortRef = React.useRef<AbortController | null>(null);
  const bottomRef = React.useRef<HTMLDivElement>(null);

  React.useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [conv?.messages]);

  const handleQuery = async () => {
    const q = query.trim();
    const convId = activeId;
    if (!q || queryLoading || !convId) return;
    const webSearch = useAppStore.getState().conversations.find((c) => c.id === convId)?.webSearch ?? false;
    setQuery('');
    setQueryLoading(true);

    const controller = new AbortController();
    abortRef.current = controller;
    try {
      addMessage(convId, { role: 'user', content: q });
      addMessage(convId, { role: 'assistant', content: '', isStreaming: true });
      let accumulated = '';
      const stream = streamRagQuery(
        { query: q, model: settings.model, session_id: settings.sessionId, temperature: settings.temperature, max_tokens: settings.maxTokens, top_k: nResults, use_web_search: webSearch },
        controller.signal,
      );
      for await (const chunk of stream) { accumulated += chunk; updateLastMessage(convId, accumulated, true); }
      updateLastMessage(convId, accumulated, false);
    } catch (err) {
      if (!(err instanceof Error && err.name === 'AbortError')) {
        updateLastMessage(convId, `⚠️ Error: ${err instanceof Error ? err.message : 'An error occurred.'}`, false);
      }
    } finally {
      setQueryLoading(false);
      abortRef.current = null;
    }
  };

  if (!conv || !activeId) {
    return (
      <div className="flex h-full flex-col items-center justify-center gap-3 text-center">
        <Database className="h-10 w-10 text-muted-foreground/40" aria-hidden="true" />
        <h2 className="text-xl font-semibold">Knowledge Base</h2>
        <p className="max-w-sm text-sm text-muted-foreground">Upload documents in Settings → Knowledge Base, then start a conversation.</p>
        <Button onClick={() => createConversation('rag')} className="gap-2"><Plus className="h-4 w-4" aria-hidden="true" /> New conversation</Button>
      </div>
    );
  }

  const msgs = conv.messages;
  return (
    <div className="flex h-full flex-col overflow-hidden" aria-label="Knowledge Base">
      <div className="flex flex-wrap items-center gap-3 border-b border-border bg-muted/30 px-4 py-2.5">
        <div className="flex items-center gap-2">
          <Switch id="rag-multi-agent" checked={multiAgent} onCheckedChange={setMultiAgent} className="scale-90" />
          <Label htmlFor="rag-multi-agent" className="text-xs text-muted-foreground cursor-pointer">Multi-agent</Label>
        </div>
        <div className="flex items-center gap-2">
          <Switch id="rag-hybrid" checked={hybridSearch} onCheckedChange={setHybridSearch} className="scale-90" />
          <Label htmlFor="rag-hybrid" className="text-xs text-muted-foreground cursor-pointer">Hybrid search</Label>
        </div>
        <div className="flex items-center gap-2">
          <Label htmlFor="rag-n-results" className="text-xs text-muted-foreground">Results</Label>
          <Input id="rag-n-results" type="number" min={1} max={20} value={nResults} onChange={(e) => setNResults(parseInt(e.target.value) || 5)} className="h-7 w-14 text-xs" />
        </div>
        <Separator orientation="vertical" className="h-5" />
        <Tooltip>
          <TooltipTrigger render={<Button variant={conv.webSearch ? 'default' : 'outline'} size="sm" className="gap-1.5" onClick={() => setWebSearch(activeId, !conv.webSearch)} aria-label="Toggle web search" aria-pressed={conv.webSearch}><Globe className="h-4 w-4" aria-hidden="true" />Web</Button>} />
          <TooltipContent>Combine docs with web search</TooltipContent>
        </Tooltip>
      </div>

      <div className="flex flex-1 flex-col overflow-y-auto min-h-0">
        <div className="flex flex-col gap-4 px-4 py-4 md:px-6">
          {msgs.length === 0 && (
            <div className="flex flex-col items-center justify-center gap-2 py-20 text-center">
              <Database className="h-10 w-10 text-muted-foreground/40" aria-hidden="true" />
              <p className="text-sm font-medium text-muted-foreground">Ask questions about your documents</p>
              <p className="text-xs text-muted-foreground">{conv.webSearch ? 'Web search is on — answers may also use the web.' : 'Upload files in Settings to start querying.'}</p>
            </div>
          )}
          {msgs.map((m, i) => (
            <div key={m.id ?? i} className={cn('flex', m.role === 'user' ? 'justify-end' : 'justify-start')}>
              <div className={cn('max-w-[75%] rounded-2xl px-4 py-3 text-sm', m.role === 'user' ? 'rounded-br-sm bg-primary text-primary-foreground' : 'rounded-bl-sm border border-border bg-card')}>
                {m.role === 'user' ? <p className="whitespace-pre-wrap leading-relaxed">{m.content}</p>
                  : <MarkdownRenderer content={m.content} />}
                {m.isStreaming && <span className="ml-0.5 inline-block h-4 w-0.5 animate-pulse bg-foreground align-middle" aria-hidden="true" />}
              </div>
            </div>
          ))}
          <div ref={bottomRef} aria-hidden="true" />
        </div>
      </div>

      <div className="border-t border-border bg-background px-4 py-3 md:px-6">
        <div className="flex items-end gap-2">
          <Textarea value={query} onChange={(e) => setQuery(e.target.value)}
            onKeyDown={(e) => { if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); void handleQuery(); } }}
            placeholder="Ask about your documents…" disabled={queryLoading} rows={2} className="flex-1 resize-none text-sm" aria-label="RAG query input" />
          <Button size="sm" onClick={() => void handleQuery()} disabled={!query.trim() || queryLoading} className="h-[60px] px-4" aria-label={queryLoading ? 'Searching…' : 'Submit query'}>
            {queryLoading ? '…' : 'Ask'}
          </Button>
        </div>
      </div>
    </div>
  );
}
```

- [ ] **Step 2: Type-check**

Run: `cd nextjs-frontend && npx tsc --noEmit`
Expected: no errors.

---

## Task 9: Remove the web-search and settings pages

**Files:**
- Delete: `nextjs-frontend/src/app/web-search/page.tsx`
- Delete: `nextjs-frontend/src/app/settings/page.tsx`

**Interfaces:**
- These routes are superseded by the sidebar (Task 5) and the Settings panel (Task 4).

- [ ] **Step 1: Delete the pages**

```bash
cd nextjs-frontend && rm -f src/app/web-search/page.tsx src/app/settings/page.tsx && rmdir src/app/web-search 2>/dev/null; echo done
```

- [ ] **Step 2: Confirm no remaining imports**

Run: `cd nextjs-frontend && grep -rn "app/web-search\|app/settings\|web-search/page\|settings/page" src --include=*.tsx --include=*.ts | grep -v "settings-panel" | grep -v "selectSettings" || echo "no stale imports"`
Expected: `no stale imports` (the `SettingsPanel` component and `selectSettings` selector are unrelated and must NOT match).

---

## Task 10: API client — forward `use_web_search` for RAG

**Files:**
- Modify: `nextjs-frontend/src/lib/api.ts`
- Modify: `nextjs-frontend/src/lib/types.ts` (already done in Task 1 for `RagQueryRequest`)

**Interfaces:**
- Consumes: `RagQueryRequest.use_web_search` (Task 1).
- Produces: `streamRagQuery` now forwards `use_web_search` to the backend (consumed by Task 8).

- [ ] **Step 1: Update `streamRagQuery` to forward the flag**

In `nextjs-frontend/src/lib/api.ts`, update the `streamRagQuery` call site so the request body includes `use_web_search` when provided:

```ts
export function streamRagQuery(
  request: RagQueryRequest,
  signal?: AbortSignal,
): AsyncGenerator<string> {
  return streamPlainText(
    '/api/rag/query',
    {
      ...request,
      ...(request.use_web_search !== undefined ? { use_web_search: request.use_web_search } : {}),
    },
    signal,
  );
}
```

- [ ] **Step 2: Type-check**

Run: `cd nextjs-frontend && npx tsc --noEmit`
Expected: no errors.

---

## Task 11: Backend — combine web search into RAG query

**Files:**
- Modify: `app/backend/models/api_models.py`
- Modify: `app/backend/api/rag.py`

**Interfaces:**
- Consumes: `WebSearchService` (already imported in `app/backend/api/chat.py`); `RAGQueryRequestEnhanced.use_web_search` (new field).
- Produces: `/api/rag/query` now merges web results into the RAG context when `use_web_search` is true.

> Inspect `app/services/rag_service.py` and the RAG skill's `stream()` signature (`app/skills/rag.py`) before editing, to find the exact place the retrieved context is assembled. The approach below prepends a clearly-delimited "Web results" block to the system prompt passed into the skill; adapt the field name to whatever the skill expects (e.g. `system_prompt` or a `context` arg).

- [ ] **Step 1: Add the field to the request model**

In `app/backend/models/api_models.py`, add to `RAGQueryRequestEnhanced`:

```python
    use_web_search: Optional[bool] = False
```

- [ ] **Step 2: Wire web search into the RAG handler**

In `app/backend/api/rag.py`, at the top add `from app.services.web_search_service import WebSearchService`. In `rag_query` (or `_handle_rag_query`), when `request.use_web_search` is true, fetch web results and merge them into the context before streaming. A safe, low-risk implementation that does not touch the RAG skill internals:

```python
async def rag_query(request: RAGQueryRequestEnhanced):
    """Query the knowledge base, optionally augmented with web search."""
    logger.info(f"RAG query - use_web_search: {request.use_web_search}")
    rag_service = get_rag_service()  # existing factory used by the route
    try:
        if request.use_web_search:
            try:
                ws = WebSearchService()
                web_results = await ws.search(request.query)
                if web_results:
                    web_block = "\n\n".join(
                        f"[Web result {i+1}] {r.get('title','')}\n{r.get('snippet','')}\nURL: {r.get('link','')}"
                        for i, r in enumerate(web_results[:5])
                    )
                    augment = (
                        "\n\n--- WEB SEARCH RESULTS (use to supplement retrieved documents) ---\n"
                        + web_block
                    )
                    request.system_prompt = (request.system_prompt or "") + augment
            except Exception as exc:  # web search failure must not break RAG
                logger.warning(f"Web search augmentation failed: {exc}")

        return await _handle_rag_query(request, rag_service)
    except Exception:
        logger.exception("RAG query error")
        raise
```

> Replace `get_rag_service()` with whatever the existing `rag_query` route uses to obtain the service instance (read the current `rag.py` — it may construct `RagService()` directly or pull it from app state). Do not change the streaming contract (`text/plain`).

- [ ] **Step 3: Lint + type-check the backend**

Run: `uv run ruff check app/backend/api/rag.py app/backend/models/api_models.py`
Expected: no errors.

Run: `uv run ty check app/backend`
Expected: no errors (or only pre-existing, unrelated findings).

---

## Task 12: Verification gates (lint, build, manual)

**Files:** none new; this task verifies the whole change.

- [ ] **Step 1: Frontend lint**

Run: `cd nextjs-frontend && npm run lint`
Expected: no errors. Fix any ESLint findings (do not suppress with `eslint-disable` without justification).

- [ ] **Step 2: Frontend build (type-check gate)**

Run: `cd nextjs-frontend && npm run build`
Expected: build succeeds.

- [ ] **Step 3: Backend lint + type-check**

Run: `uv run ruff check . && uv run ty check app/backend`
Expected: no errors.

- [ ] **Step 4: Manual verification**

Run the dev stack (`make dev` or `cd nextjs-frontend && npm run dev` plus the backend). Walk this checklist in **both light and dark** themes:
  1. Sidebar shows "New conversation" + Chat/RAG switch + empty "No conversations yet.".
  2. Click **New conversation** (in Chat mode) → a chat conversation is created and selected; the URL is `/chat`.
  3. Send a message → it streams; reloading the page keeps the conversation in the history (localStorage).
  4. Toggle **🌐 web search** in Chat → answer comes from `/api/chat/web-search` (web-grounded).
  5. Switch to **RAG**, create a RAG conversation, upload a document via **Settings → Knowledge Base**, then ask a question with web search OFF → doc-grounded answer.
  6. In the same RAG conversation, toggle **Web** on and ask again → answer reflects both docs and web (backend combined).
  7. Click a conversation in the sidebar → it loads and navigates to the correct route (Chat vs RAG badge).
  8. Delete a conversation (hover trash → confirm) → it disappears and active falls back to the next most recent.
  9. Open **Settings** (bottom-left) → change model/backend/generation; close; confirm values persist after reload.
  10. Confirm no console errors and that the global header is gone (sidebar owns branding).

- [ ] **Step 5: (Skipped per user) Commit**

> The user commits locally. Do NOT run `git add` / `git commit` / `git push`.

---

## Self-Review Notes (against spec)

- **Spec §1 (routing):** `/chat` + `/rag` kept (Tasks 7, 8); `/web-search` + `/settings` deleted (Task 9); global header removed (Task 6). ✔
- **Spec §2 (data model):** `Conversation`/`ConversationMode` + persisted `conversations`/`activeId` (Tasks 1–2). ✔
- **Spec §3 (sidebar):** brand, New conversation, Chat|RAG switch, unified list w/ badges, footer Settings + theme (Task 5). ✔
- **Spec §4 (settings panel):** Models/Generation/Backend/Knowledge Base sections (Tasks 3–4). SerpAPI per-request field dropped (noted in Task 4). ✔
- **Spec §5 (chat view):** active-conversation driven, web-search toggle, in-page settings removed, Clear chat kept (Task 7). ✔
- **Spec §6 (rag view):** file mgmt removed, web-search toggle added, options row kept (Task 8). ✔
- **Spec §7 (backend):** `use_web_search` field + merge logic (Task 11). ✔
- **Spec §8 (api client):** `RagQueryRequest.use_web_search` forwarded (Tasks 1, 10). ✔
- **Spec §9–10 (a11y, testing):** Radix Dialog focus trap + Esc; manual checklist covers light/dark (Task 12). ✔
- **No placeholders:** every task contains concrete code or exact commands. ✔
