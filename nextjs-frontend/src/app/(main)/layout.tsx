import * as React from 'react';
import { Sidebar } from '@/components/layout/sidebar';

/**
 * Main application shell: sidebar chrome + content area.
 * Route groups do not affect the URL, so `/`, `/chat`, and `/rag` all render
 * inside this layout.
 */
export default function MainLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>): React.ReactElement {
  return (
    <div className="flex h-full">
      {/* ── Sidebar (desktop: static; mobile: overlay) ── */}
      <Sidebar />

      {/* ── Main content area ── */}
      <main className="flex flex-1 flex-col overflow-auto" id="main-content" tabIndex={-1}>
        {children}
      </main>
    </div>
  );
}
