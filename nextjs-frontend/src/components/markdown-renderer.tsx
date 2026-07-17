'use client';

import * as React from 'react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import rehypeHighlight from 'rehype-highlight';
import { Check, Copy } from 'lucide-react';
import { cn } from '@/lib/utils';

// ─── Code Block with copy button ─────────────────────────────────────────────

interface CodeBlockProps {
  language?: string;
  children: string;
  inline?: boolean;
}

function CodeBlock({ language, children, inline }: CodeBlockProps) {
  const [copied, setCopied] = React.useState(false);

  const handleCopy = React.useCallback(async () => {
    try {
      await navigator.clipboard.writeText(children);
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    } catch {
      // clipboard not available – silently ignore
    }
  }, [children]);

  if (inline) {
    return (
      <code className="rounded bg-muted px-1.5 py-0.5 font-mono text-sm text-foreground">
        {children}
      </code>
    );
  }

  return (
    <div className="group relative my-4 overflow-hidden rounded-lg border border-border bg-muted/50">
      {/* Language label + copy button header */}
      <div className="flex items-center justify-between border-b border-border bg-muted/80 px-4 py-1.5">
        <span className="font-mono text-xs text-muted-foreground">
          {language ?? 'code'}
        </span>
        <button
          onClick={handleCopy}
          aria-label={copied ? 'Copied' : 'Copy code'}
          className={cn(
            'flex items-center gap-1.5 rounded px-2 py-1 text-xs transition-colors',
            'text-muted-foreground hover:text-foreground',
            'focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-ring',
          )}
        >
          {copied ? (
            <Check className="h-3.5 w-3.5 text-green-500" aria-hidden="true" />
          ) : (
            <Copy className="h-3.5 w-3.5" aria-hidden="true" />
          )}
          <span>{copied ? 'Copied!' : 'Copy'}</span>
        </button>
      </div>
      <pre className="overflow-x-auto p-4 text-sm leading-relaxed">
        <code className={language ? `language-${language}` : undefined}>
          {children}
        </code>
      </pre>
    </div>
  );
}

// ─── MarkdownRenderer ─────────────────────────────────────────────────────────

export default function MarkdownRenderer({ content }: { content: string }) {
  return (
    <div className="prose prose-sm dark:prose-invert max-w-none break-words">
      <ReactMarkdown
        remarkPlugins={[remarkGfm]}
        rehypePlugins={[rehypeHighlight]}
        components={{
          // ── Headings ──────────────────────────────────────────────────────
          h1: ({ children }) => (
            <h1 className="mb-4 mt-6 text-2xl font-bold leading-tight text-foreground first:mt-0">
              {children}
            </h1>
          ),
          h2: ({ children }) => (
            <h2 className="mb-3 mt-5 text-xl font-semibold leading-tight text-foreground first:mt-0">
              {children}
            </h2>
          ),
          h3: ({ children }) => (
            <h3 className="mb-2 mt-4 text-lg font-semibold text-foreground first:mt-0">
              {children}
            </h3>
          ),
          h4: ({ children }) => (
            <h4 className="mb-2 mt-3 text-base font-semibold text-foreground first:mt-0">
              {children}
            </h4>
          ),
          h5: ({ children }) => (
            <h5 className="mb-1 mt-2 text-sm font-semibold text-foreground first:mt-0">
              {children}
            </h5>
          ),
          h6: ({ children }) => (
            <h6 className="mb-1 mt-2 text-xs font-semibold text-muted-foreground first:mt-0">
              {children}
            </h6>
          ),

          // ── Paragraphs ────────────────────────────────────────────────────
          p: ({ children }) => (
            <p className="mb-3 leading-relaxed text-foreground last:mb-0">
              {children}
            </p>
          ),

          // ── Lists ─────────────────────────────────────────────────────────
          ul: ({ children }) => (
            <ul className="mb-3 list-disc space-y-1 pl-6 text-foreground">
              {children}
            </ul>
          ),
          ol: ({ children }) => (
            <ol className="mb-3 list-decimal space-y-1 pl-6 text-foreground">
              {children}
            </ol>
          ),
          li: ({ children }) => (
            <li className="leading-relaxed text-foreground">{children}</li>
          ),

          // ── Blockquote ────────────────────────────────────────────────────
          blockquote: ({ children }) => (
            <blockquote className="my-3 border-l-4 border-primary/40 pl-4 italic text-muted-foreground">
              {children}
            </blockquote>
          ),

          // ── Horizontal rule ───────────────────────────────────────────────
          hr: () => <hr className="my-4 border-border" />,

          // ── Strong / Em ───────────────────────────────────────────────────
          strong: ({ children }) => (
            <strong className="font-semibold text-foreground">{children}</strong>
          ),
          em: ({ children }) => (
            <em className="italic text-foreground">{children}</em>
          ),

          // ── Links ─────────────────────────────────────────────────────────
          a: ({ href, children }) => (
            <a
              href={href}
              target="_blank"
              rel="noopener noreferrer"
              className="text-primary underline-offset-2 hover:underline"
            >
              {children}
            </a>
          ),

          // ── Images ────────────────────────────────────────────────────────
          img: ({ src, alt }) => (
            // eslint-disable-next-line @next/next/no-img-element
            <img
              src={src}
              alt={alt ?? ''}
              className="my-3 max-w-full rounded-lg border border-border"
              loading="lazy"
            />
          ),

          // ── Tables ────────────────────────────────────────────────────────
          table: ({ children }) => (
            <div className="my-4 overflow-x-auto rounded-lg border border-border">
              <table className="w-full border-collapse text-sm">
                {children}
              </table>
            </div>
          ),
          thead: ({ children }) => (
            <thead className="bg-muted/50">{children}</thead>
          ),
          tbody: ({ children }) => (
            <tbody className="divide-y divide-border">{children}</tbody>
          ),
          tr: ({ children }) => (
            <tr className="hover:bg-muted/20 transition-colors">{children}</tr>
          ),
          th: ({ children }) => (
            <th className="px-4 py-2.5 text-left text-xs font-semibold uppercase tracking-wide text-muted-foreground">
              {children}
            </th>
          ),
          td: ({ children }) => (
            <td className="px-4 py-2.5 text-foreground">{children}</td>
          ),

          // ── Code ──────────────────────────────────────────────────────────
          code: ({ className, children, ...props }) => {
            // react-markdown passes className like "language-ts"
            const match = /language-(\w+)/.exec(className ?? '');
            const language = match ? match[1] : undefined;
            const isInline = !className && typeof children === 'string' && !children.includes('\n');

            // Check if it's inside a pre (block code) vs inline
            if ('node' in props) {
              // The parent context isn't directly available; use heuristic
            }

            return (
              <CodeBlock
                language={language}
                inline={isInline}
              >
                {String(children).replace(/\n$/, '')}
              </CodeBlock>
            );
          },

          pre: ({ children }) => (
            // react-markdown wraps <code> in <pre> for block code.
            // We render our own styled container inside CodeBlock, so just pass through.
            <>{children}</>
          ),
        }}
      >
        {content}
      </ReactMarkdown>
    </div>
  );
}
