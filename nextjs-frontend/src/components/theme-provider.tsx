'use client';

import * as React from 'react';
import { ThemeProvider as NextThemesProvider } from 'next-themes';
import type { ThemeProviderProps } from 'next-themes';

/**
 * Thin wrapper around `next-themes` ThemeProvider.
 *
 * Defaults:
 *  - `attribute="class"` so Tailwind dark-mode (`dark:` variants) works.
 *  - `defaultTheme="system"` to respect the OS preference.
 *  - `enableSystem` to allow system-preference detection.
 *  - `disableTransitionOnChange` to prevent a flash when the theme switches.
 */
export function ThemeProvider({
  children,
  ...props
}: ThemeProviderProps): React.ReactElement {
  return (
    <NextThemesProvider
      attribute="class"
      defaultTheme="system"
      enableSystem
      disableTransitionOnChange
      {...props}
    >
      {children}
    </NextThemesProvider>
  );
}
