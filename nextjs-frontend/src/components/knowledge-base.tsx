'use client';

import * as React from 'react';
import {
  FileText,
  Trash2,
  RefreshCw,
  Database,
  AlertTriangle,
  CheckCircle2,
} from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Label } from '@/components/ui/label';
import { Separator } from '@/components/ui/separator';
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
  deleteRagFile,
  resetRag,
} from '@/lib/api';
import type { RagFile, RagStatusResponse } from '@/lib/types';
import { cn } from '@/lib/utils';

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
            <p className="font-medium">{status.uploaded_files ?? 0}</p>
          </div>
          <div>
            <p className="text-muted-foreground">Chunks</p>
            <p className="font-medium">{status.indexed_chunks ?? '—'}</p>
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
  const { setRagFiles } = useAppStore();
  const [files, setFiles] = React.useState<RagFile[]>([]);
  const [status, setStatus] = React.useState<RagStatusResponse | null>(null);
  const [statusLoading, setStatusLoading] = React.useState(false);
  const [fileError, setFileError] = React.useState('');
  const [deleting, setDeleting] = React.useState<string | null>(null);
  const [resetOpen, setResetOpen] = React.useState(false);

  const loadFiles = React.useCallback(async () => {
    try {
      const res = await getRagFiles();
      setFiles(res.files ?? []);
      setRagFiles(res.files ?? []);
    } catch {
      setFiles([]);
      setFileError('Could not load knowledge base files.');
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

  // Load knowledge-base files and status when the panel mounts. The setters run
  // inside async callbacks (standard fetch-on-mount); the rule still flags the
  // inner calls, so it is disabled with justification.
  /* eslint-disable react-hooks/set-state-in-effect */
  React.useEffect(() => {
    void loadFiles();
    void loadStatus();
  }, [loadFiles, loadStatus]);
  /* eslint-enable react-hooks/set-state-in-effect */

  const handleDelete = async (name: string) => {
    setDeleting(name);
    try {
      await deleteRagFile(name);
      await loadFiles();
      await loadStatus();
    } catch (err) {
      setFileError(`Delete failed: ${err instanceof Error ? err.message : 'unknown error'}`);
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
      setFileError(`Reset failed: ${err instanceof Error ? err.message : 'unknown error'}`);
    }
  };

  return (
    <div className="space-y-3">
      <StatusCard status={status} loading={statusLoading} onRefresh={() => void loadStatus()} />

      <p className="text-[11px] leading-snug text-muted-foreground">
        Attach documents from the paperclip button in the RAG view — they appear here once indexed.
      </p>

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
