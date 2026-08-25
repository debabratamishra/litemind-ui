import { useEffect, useState } from 'react';

const KEY = 'lm-theme';

export default function ThemeToggle() {
  // Server render assumes light; the effect syncs to reality right after
  // hydration, so the pre-hydration glyph matches the FOUC script's flash-free
  // paint and aria/icon never mismatch for more than a frame.
  const [theme, setTheme] = useState<'light' | 'dark'>('light');

  useEffect(() => {
    setTheme(document.documentElement.getAttribute('data-theme') === 'dark' ? 'dark' : 'light');
  }, []);

  function toggle() {
    const next = theme === 'dark' ? 'light' : 'dark';
    document.documentElement.setAttribute('data-theme', next);
    try {
      localStorage.setItem(KEY, next);
    } catch {
      /* storage unavailable (hardened browsers) — theme still applies for session */
    }
    setTheme(next);
  }

  return (
    <button
      id="theme-toggle"
      className="theme-toggle"
      type="button"
      onClick={toggle}
      aria-label="Switch between light and dark theme"
      aria-pressed={theme === 'dark'}
      title="Switch theme"
    >
      <span className="t-icon" aria-hidden="true">
        {theme === 'dark' ? '☀️' : '🌙'}
      </span>
    </button>
  );
}
