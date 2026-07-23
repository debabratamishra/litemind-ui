import * as React from 'react';

/**
 * Centered layout for the unauthenticated (auth) route group: login/register.
 */
export default function AuthLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>): React.ReactElement {
  return (
    <div className="flex min-h-full items-center justify-center bg-background p-4">
      {children}
    </div>
  );
}
