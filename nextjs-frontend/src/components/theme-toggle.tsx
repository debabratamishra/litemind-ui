'use client';

import * as React from 'react';
import { Moon, Sun } from 'lucide-react';
import { useTheme } from 'next-themes';
import {
  Tooltip,
  TooltipContent,
  TooltipTrigger,
} from '@/components/ui/tooltip';
import { cn } from '@/lib/utils';

/**
 * Icon button that toggles between light and dark themes.
 *
 * All theme-dependent attributes (aria-label, icon, onClick target) are
 * deferred until after mount to prevent SSR ↔ client hydration mismatches.
 * Before mount the button renders a neutral placeholder so the layout is
 * stable and no attributes differ between the server-rendered HTML and the
 * first client paint.
 */
// Returns false during SSR and the first client render, then true once the
// component has hydrated on the client. Using useSyncExternalStore (instead of
// a setState-in-effect) keeps server and client HTML identical while satisfying
// the react-hooks/set-state-in-effect lint rule.
const emptySubscribe = () => () => {};

export function ThemeToggle(): React.ReactElement {
  const { resolvedTheme, setTheme } = useTheme();
  const mounted = React.useSyncExternalStore(
    emptySubscribe,
    () => true,
    () => false,
  );

  // Before mount: stable, theme-agnostic values so server HTML == client HTML.
  const isDark = mounted ? resolvedTheme === 'dark' : false;
  const label = mounted
    ? isDark
      ? 'Switch to light theme'
      : 'Switch to dark theme'
    : 'Toggle theme';

  return (
    <Tooltip>
      <TooltipTrigger
        // Use suppressHydrationWarning so React doesn't warn about the
        // aria-label changing on the very first client render when next-themes
        // reads localStorage.  The value IS correct after mount — the warning
        // would be a false positive.
        suppressHydrationWarning
        aria-label={label}
        onClick={mounted ? () => setTheme(isDark ? 'light' : 'dark') : undefined}
        className={cn(
          'inline-flex h-9 w-9 items-center justify-center rounded-md',
          'text-muted-foreground transition-colors',
          'hover:bg-muted hover:text-foreground',
          'focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring',
        )}
      >
        {mounted ? (
          isDark ? (
            <Sun className="h-4 w-4" aria-hidden="true" />
          ) : (
            <Moon className="h-4 w-4" aria-hidden="true" />
          )
        ) : (
          // Same dimensions as the real icon so there's no layout shift.
          <span className="h-4 w-4 block" aria-hidden="true" />
        )}
      </TooltipTrigger>
      <TooltipContent side="right" suppressHydrationWarning>
        <p suppressHydrationWarning>{label}</p>
      </TooltipContent>
    </Tooltip>
  );
}
