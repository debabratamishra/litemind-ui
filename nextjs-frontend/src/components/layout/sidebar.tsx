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
  LogIn,
  LogOut,
} from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Separator } from '@/components/ui/separator';
import { ScrollArea } from '@/components/ui/scroll-area';
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
  const { conversations, activeId, createConversation, selectConversation, deleteConversation, user, isAuthenticated, logout } =
    useAppStore();

  const [mobileOpen, setMobileOpen] = React.useState(false);
  const [settingsOpen, setSettingsOpen] = React.useState(false);
  const [confirmDelete, setConfirmDelete] = React.useState<string | null>(null);

  const activeMode: ConversationMode = pathname.startsWith('/rag') ? 'rag' : 'chat';
  const closeMobile = React.useCallback(() => setMobileOpen(false), []);

  // Close the mobile overlay whenever the route changes. This is UI state
  // synced to navigation (not derived from props), so an effect is correct here.
  /* eslint-disable react-hooks/set-state-in-effect */
  React.useEffect(() => {
    closeMobile();
  }, [pathname, closeMobile]);
  /* eslint-enable react-hooks/set-state-in-effect */

  // Close the mobile overlay / settings panel on Escape. The setState calls
  // live in the keydown callback (an external-system subscription), which the
  // rule permits, so no disable is needed here.
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

        {/* Auth section */}
        <div className="px-3">
          {isAuthenticated && user ? (
            <div className="flex items-center gap-2 rounded-md border border-border bg-muted/40 p-2">
              <div
                className="flex h-7 w-7 shrink-0 items-center justify-center rounded-full bg-primary/10 text-xs font-semibold text-primary"
                aria-hidden="true"
              >
                {(user.email?.[0] ?? '?').toUpperCase()}
              </div>
              <span className="min-w-0 flex-1 truncate text-xs text-sidebar-foreground">
                {user.email}
              </span>
              <Button
                size="sm"
                variant="ghost"
                className="h-7 gap-1 px-2 text-xs"
                onClick={() => void logout()}
                aria-label="Sign out"
              >
                <LogOut className="h-3.5 w-3.5" aria-hidden="true" />
                Sign out
              </Button>
            </div>
          ) : (
            <Button
              variant="outline"
              className="w-full justify-start gap-2"
              size="sm"
              onClick={() => router.push('/login')}
            >
              <LogIn className="h-4 w-4" aria-hidden="true" />
              Sign in
            </Button>
          )}
        </div>

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
