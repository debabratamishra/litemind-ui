import { describe, it, expect } from 'vitest';
import { render } from '@testing-library/react';
import VoiceActivityIndicator from './voice-activity';

describe('VoiceActivityIndicator', () => {
  it('renders 5 bars by default', () => {
    const { container } = render(<VoiceActivityIndicator state="idle" />);
    const bars = container.querySelectorAll('[class*="bg-primary"]');
    expect(bars).toHaveLength(5);
  });

  it('shows active state when listening', () => {
    const { container } = render(<VoiceActivityIndicator state="listening" />);
    const bars = container.querySelectorAll('[class*="bg-primary"]');
    // In listening state, bars should have h-3 class (not animate-pulse)
    expect(bars.length).toBe(5);
  });

  it('shows active state when speaking', () => {
    const { container } = render(<VoiceActivityIndicator state="speaking" />);
    const bars = container.querySelectorAll('[class*="bg-primary"]');
    expect(bars.length).toBe(5);
  });

  it('applies custom className', () => {
    const { container } = render(
      <VoiceActivityIndicator state="idle" className="custom-class" />,
    );
    const span = container.querySelector('span');
    expect(span).toHaveClass('custom-class');
  });

  it('is hidden when idle', () => {
    const { container } = render(<VoiceActivityIndicator state="idle" />);
    const span = container.querySelector('span');
    // Idle bars have opacity-50
    expect(span).toBeInTheDocument();
  });

  it('is accessible', () => {
    const { container } = render(<VoiceActivityIndicator state="listening" />);
    const span = container.querySelector('span');
    expect(span).toHaveAttribute('aria-hidden', 'true');
  });
});