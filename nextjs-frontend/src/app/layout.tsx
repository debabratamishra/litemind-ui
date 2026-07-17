import type { Metadata, Viewport } from 'next';
import { Inter } from 'next/font/google';
import './globals.css';
import { ThemeProvider } from '@/components/theme-provider';
import { TooltipProvider } from '@/components/ui/tooltip';
import { Sidebar } from '@/components/layout/sidebar';

// ─── Font ─────────────────────────────────────────────────────────────────────

const inter = Inter({
  subsets: ['latin'],
  variable: '--font-inter',
  display: 'swap',
});

// ─── Metadata ─────────────────────────────────────────────────────────────────

export const metadata: Metadata = {
  title: {
    default: 'LiteMindUI',
    template: '%s · LiteMindUI',
  },
  description:
    'Local-first AI workspace with chat, RAG, web search, and voice workflows.',
  icons: { icon: '/favicon.ico' },
};

export const viewport: Viewport = {
  themeColor: [
    { media: '(prefers-color-scheme: light)', color: '#ffffff' },
    { media: '(prefers-color-scheme: dark)', color: '#0a0a0a' },
  ],
};

// ─── Layout ───────────────────────────────────────────────────────────────────

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html
      lang="en"
      className={`${inter.variable} h-full`}
      suppressHydrationWarning
    >
      <body className="h-full bg-background font-sans antialiased">
        <ThemeProvider>
          {/* delay=300 gives tooltips a 300ms hover delay before showing */}
          <TooltipProvider delay={300}>
            {/* Full-height flex container: sidebar + main content */}
            <div className="flex h-full">
              {/* ── Sidebar (desktop: static; mobile: overlay) ── */}
              <Sidebar />

              {/* ── Main content area ── */}
              <main
                className="flex flex-1 flex-col overflow-auto"
                id="main-content"
                tabIndex={-1}
              >
                {children}
              </main>
            </div>
          </TooltipProvider>
        </ThemeProvider>
      </body>
    </html>
  );
}
