"use client";

import * as React from "react";
import { Bot } from "lucide-react";
import { cn } from "@/lib/utils";
import MarkdownRenderer from "@/components/markdown-renderer";
import GenerativeUIRenderer from "@/components/generative-ui-renderer";
import { SourcesButton, CitationsDialog } from "@/components/citations";
import {
  convertCitationMarkers,
  normalizeAnswerWhitespace,
  parseWebSearchContent,
} from "@/lib/web-search-citations";
import type { AppSettings, UIMessage } from "@/lib/types";

function ThinkingDots() {
  return (
    <div
      className="flex items-center gap-1 px-1 py-0.5"
      aria-label="Assistant is thinking"
      role="status"
    >
      {[0, 1, 2].map((i) => (
        <span
          key={i}
          className="h-2 w-2 animate-bounce rounded-full bg-muted-foreground/60"
          style={{ animationDelay: `${i * 150}ms` }}
        />
      ))}
    </div>
  );
}

/**
 * A single chat message bubble.
 *
 * Assistant messages that contain a web-search `Sources:` block are rendered
 * with the answer prose (whitespace-normalised, inline `[n]` markers turned
 * into clickable chips) plus a "Sources (N)" button that opens a Dialog
 * listing every citation. All other messages render exactly as before.
 */
export default function MessageBubble({
  msg,
  settings,
  onAction,
}: {
  msg: UIMessage;
  settings: AppSettings;
  onAction: (action: string, payload?: string) => void;
}) {
  const [citeOpen, setCiteOpen] = React.useState(false);
  const [citeFocus, setCiteFocus] = React.useState<number | null>(null);

  const openCitations = React.useCallback((focus: number | null) => {
    setCiteFocus(focus);
    setCiteOpen(true);
  }, []);

  const parsed =
    msg.role === "assistant" && !msg.isStreaming && msg.content
      ? parseWebSearchContent(msg.content)
      : null;
  const hasSources = (parsed?.sources.length ?? 0) > 0;

  const body = (() => {
    if (msg.role === "user") {
      return <p className="whitespace-pre-wrap leading-relaxed">{msg.content}</p>;
    }
    if (msg.isStreaming && !msg.content) {
      return <ThinkingDots />;
    }
    if (settings.enableGenerativeUI) {
      return settings.genUIDisplayMode === "rendered" ? (
        <GenerativeUIRenderer content={msg.content} onAction={onAction} />
      ) : (
        <MarkdownRenderer content={msg.content} />
      );
    }
    if (hasSources && parsed) {
      const answer = convertCitationMarkers(
        normalizeAnswerWhitespace(parsed.answer),
      );
      return (
        <>
          <MarkdownRenderer
            content={answer}
            onCitationClick={(index) => openCitations(index)}
          />
          <SourcesButton
            count={parsed.sources.length}
            onClick={() => openCitations(null)}
          />
          <CitationsDialog
            open={citeOpen}
            onOpenChange={setCiteOpen}
            sources={parsed.sources}
            focusIndex={citeFocus}
          />
        </>
      );
    }
    return <MarkdownRenderer content={msg.content} />;
  })();

  return (
    <div
      className={cn(
        "flex gap-3",
        msg.role === "user" ? "justify-end" : "justify-start",
      )}
      role="article"
      aria-label={`${msg.role === "user" ? "Your" : "Assistant"} message`}
    >
      {msg.role === "assistant" && (
        <div
          className="mt-1 flex h-8 w-8 shrink-0 items-center justify-center rounded-full bg-primary/10"
          aria-hidden="true"
        >
          <Bot className="h-4 w-4 text-primary" />
        </div>
      )}
      <div
        className={cn(
          "max-w-[75%] rounded-2xl px-4 py-3 text-sm",
          msg.role === "user"
            ? "rounded-br-sm bg-primary text-primary-foreground"
            : "rounded-bl-sm border border-border bg-card text-foreground",
        )}
      >
        {body}
        {msg.isStreaming && msg.content && (
          <span
            className="ml-0.5 inline-block h-4 w-0.5 animate-pulse bg-current align-middle"
            aria-hidden="true"
          />
        )}
      </div>
    </div>
  );
}
