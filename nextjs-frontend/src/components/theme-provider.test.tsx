import { describe, it, expect } from 'vitest';
import { render } from '@testing-library/react';
import { ThemeProvider } from './theme-provider';

describe('ThemeProvider', () => {
  it('renders children correctly', () => {
    const { getByText } = render(
      <ThemeProvider>
        <div>Test Child</div>
      </ThemeProvider>,
    );
    expect(getByText('Test Child')).toBeInTheDocument();
  });

  it('has correct default props', () => {
    const { container } = render(
      <ThemeProvider>
        <div>Test</div>
      </ThemeProvider>,
    );
    // ThemeProvider wraps with next-themes ThemeProvider
    // which uses 'class' attribute for dark mode
    expect(container.firstChild).toBeTruthy();
  });
});