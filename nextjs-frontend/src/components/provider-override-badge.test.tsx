import { describe, it, expect, vi } from 'vitest';
import { render, screen, fireEvent } from '@testing-library/react';
import { ProviderOverrideBadge } from './provider-override-badge';
import type { ProviderOverride } from '@/lib/types';

describe('ProviderOverrideBadge', () => {
  const baseOverride: ProviderOverride = {
    alias: 'nim',
    backend: 'nvidia_nim',
    model: 'meta/llama-3.3-70b-instruct',
    text: 'Explain quantum computing',
    hasKey: true,
  };

  it('renders provider name, model, and key status when key is present', () => {
    render(<ProviderOverrideBadge override={baseOverride} onRemove={vi.fn()} onSetKey={vi.fn()} />);
    expect(screen.getByText('NIM')).toBeInTheDocument();
    expect(screen.getByText(/meta\/llama-3.3-70b-instruct/)).toBeInTheDocument();
    expect(screen.getByText('✓')).toBeInTheDocument();
  });

  it('renders warning state when key is missing', () => {
    const override = { ...baseOverride, hasKey: false };
    render(<ProviderOverrideBadge override={override} onRemove={vi.fn()} onSetKey={vi.fn()} />);
    expect(screen.getByText('⚠️')).toBeInTheDocument();
  });

  it('renders OpenRouter provider correctly', () => {
    const override: ProviderOverride = { ...baseOverride, backend: 'openrouter', alias: 'or', model: 'openai/gpt-4o' };
    render(<ProviderOverrideBadge override={override} onRemove={vi.fn()} onSetKey={vi.fn()} />);
    expect(screen.getByText('OpenRouter')).toBeInTheDocument();
    expect(screen.getByText(/openai\/gpt-4o/)).toBeInTheDocument();
  });

  it('renders Ollama provider correctly', () => {
    const override: ProviderOverride = { ...baseOverride, backend: 'ollama', alias: 'ollama', model: 'gemma3:1b' };
    render(<ProviderOverrideBadge override={override} onRemove={vi.fn()} onSetKey={vi.fn()} />);
    expect(screen.getByText('Ollama')).toBeInTheDocument();
    expect(screen.getByText(/gemma3:1b/)).toBeInTheDocument();
  });

  it('shows "default" in aria-label when no model is specified', () => {
    const override = { ...baseOverride, model: '' };
    render(<ProviderOverrideBadge override={override} onRemove={vi.fn()} onSetKey={vi.fn()} />);
    expect(screen.getByRole('button', { name: /model: default/i })).toBeInTheDocument();
  });

  it('calls onSetKey when badge is clicked', () => {
    const onSetKey = vi.fn();
    render(<ProviderOverrideBadge override={baseOverride} onRemove={vi.fn()} onSetKey={onSetKey} />);
    const badge = screen.getByRole('button', { name: /provider: NIM/i });
    fireEvent.click(badge);
    expect(onSetKey).toHaveBeenCalledTimes(1);
  });

  it('calls onRemove when X button is clicked', () => {
    const onRemove = vi.fn();
    render(<ProviderOverrideBadge override={baseOverride} onRemove={onRemove} onSetKey={vi.fn()} />);
    const removeButton = screen.getByRole('button', { name: /remove provider override/i });
    fireEvent.click(removeButton);
    expect(onRemove).toHaveBeenCalledTimes(1);
  });

  it('has correct aria-label with key status', () => {
    render(<ProviderOverrideBadge override={baseOverride} onRemove={vi.fn()} onSetKey={vi.fn()} />);
    const badge = screen.getByRole('button', { name: /provider: NIM.*Key configured/i });
    expect(badge).toBeInTheDocument();
  });

  it('has correct aria-label when key is missing', () => {
    const override = { ...baseOverride, hasKey: false };
    render(<ProviderOverrideBadge override={override} onRemove={vi.fn()} onSetKey={vi.fn()} />);
    const badge = screen.getByRole('button', { name: /provider: NIM.*Key missing/i });
    expect(badge).toBeInTheDocument();
  });

  it('applies amber background when key is missing', () => {
    const override = { ...baseOverride, hasKey: false };
    const { container } = render(<ProviderOverrideBadge override={override} onRemove={vi.fn()} onSetKey={vi.fn()} />);
    const badge = container.querySelector('[class*="bg-amber-500"]');
    expect(badge).toBeInTheDocument();
  });

  it('applies cyan background for NIM provider with key', () => {
    const { container } = render(<ProviderOverrideBadge override={baseOverride} onRemove={vi.fn()} onSetKey={vi.fn()} />);
    const badge = container.querySelector('[class*="bg-cyan-500"]');
    expect(badge).toBeInTheDocument();
  });
});
