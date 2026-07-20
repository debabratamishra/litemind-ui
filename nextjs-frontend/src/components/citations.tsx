"use client";

import * as React from "react";
import { BookOpen } from "lucide-react";
import { Button } from "@/components/ui/button";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { cn } from "@/lib/utils";
import type { Citation } from "@/lib/web-search-citations";

/** Clickable chip that opens the citations Dialog for a message. */
export function SourcesButton({
  count,
  onClick,
}: {
  count: number;
  onClick: () => void;
}) {
  return (
    <Button
      type="button"
      variant="outline"
      size="sm"
      className="mt-2 gap-1.5"
      onClick={onClick}
      aria-haspopup="dialog"
    >
      <BookOpen className="h-3.5 w-3.5" aria-hidden="true" />
      Sources ({count})
    </Button>
  );
}

/** Modal listing every citation with a working source link. */
export function CitationsDialog({
  open,
  onOpenChange,
  sources,
  focusIndex,
}: {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  sources: Citation[];
  /** Citation index to highlight when opened from an inline chip. */
  focusIndex: number | null;
}) {
  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="w-full max-w-[calc(100%-2rem)] p-0 sm:max-w-lg">
        <DialogHeader className="px-4 pt-4">
          <DialogTitle>Sources</DialogTitle>
          <DialogDescription>
            {sources.length} reference{sources.length === 1 ? "" : "s"} cited by
            the assistant.
          </DialogDescription>
        </DialogHeader>
        <div className="max-h-[60vh] space-y-2 overflow-y-auto px-4 pb-4">
          {sources.map((s) => (
            <div
              key={s.index}
              className={cn(
                "rounded-lg border border-border p-3 transition-colors",
                s.index === focusIndex
                  ? "bg-primary/5 ring-1 ring-primary/40"
                  : "bg-card",
              )}
            >
              <div className="flex items-start gap-2">
                <span className="mt-0.5 inline-flex h-5 min-w-5 shrink-0 items-center justify-center rounded bg-primary/10 px-1 text-xs font-semibold text-primary">
                  [{s.index}]
                </span>
                {s.url ? (
                  <a
                    href={s.url}
                    target="_blank"
                    rel="noopener noreferrer"
                    className="text-sm font-medium text-primary underline-offset-2 hover:underline"
                  >
                    {s.title}
                  </a>
                ) : (
                  <span className="text-sm font-medium text-foreground">
                    {s.title}
                  </span>
                )}
              </div>
              {s.domain && (
                <div className="mt-0.5 pl-7 text-xs text-muted-foreground">
                  {s.domain}
                </div>
              )}
              {s.snippet && (
                <p className="mt-1 pl-7 text-xs leading-relaxed text-muted-foreground">
                  {s.snippet}
                </p>
              )}
            </div>
          ))}
        </div>
      </DialogContent>
    </Dialog>
  );
}
