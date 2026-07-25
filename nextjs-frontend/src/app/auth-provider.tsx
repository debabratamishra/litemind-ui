'use client';

import * as React from 'react';
import { useAppStore } from '@/lib/store';

/**
 * Rehydrates the auth session on mount by calling `/api/auth/me` (which reads
 * the httpOnly cookie). Wrapped around the whole app in the root layout so both
 * the (main) and (auth) route groups share a single auth state.
 */
export function AuthProvider({ children }: { children: React.ReactNode }): React.ReactElement {
  const fetchCurrentUser = useAppStore((s) => s.fetchCurrentUser);

  React.useEffect(() => {
    void fetchCurrentUser();
  }, [fetchCurrentUser]);

  return <>{children}</>;
}
