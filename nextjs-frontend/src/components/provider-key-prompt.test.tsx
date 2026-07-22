import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen, fireEvent } from '@testing-library/react';
import { ProviderKeyPrompt } from './provider-key-prompt';

describe('ProviderKeyPrompt', () => {
  const defaultProps = {
    open: true,
    provider: 'nvidia_nim' as const,
    model: 'meta/llama-3.3-70b-instruct',
    onSetKey: vi.fn(),
    onCancel: vi.fn(),
    onFallback: vi.fn(),
  };

  beforeEach(() => {
    vi.clearAllMocks();
  });

  it('renders when open', () => {
    render(<ProviderKeyPrompt {...defaultProps} />);
    expect(screen.getByText('Provider: NIM')).toBeInTheDocument();
    expect(screen.getByText(/Model:.*meta\/llama-3.3-70b-instruct/)).toBeInTheDocument();
    expect(screen.getByText(/Status:.*API key required/)).toBeInTheDocument();
  });

  it('does not render when closed', () => {
    render(<ProviderKeyPrompt {...defaultProps} open={false} />);
    expect(screen.queryByText('Provider: NIM')).not.toBeInTheDocument();
  });

  it('renders the API key input', () => {
    render(<ProviderKeyPrompt {...defaultProps} />);
    const input = screen.getByLabelText('API Key');
    expect(input).toBeInTheDocument();
    expect(input).toHaveAttribute('type', 'password');
  });

  it('renders all three buttons', () => {
    render(<ProviderKeyPrompt {...defaultProps} />);
    expect(screen.getByRole('button', { name: 'Cancel' })).toBeInTheDocument();
    expect(screen.getByRole('button', { name: 'Use Default Ollama Instead' })).toBeInTheDocument();
    expect(screen.getByRole('button', { name: 'Set API Key' })).toBeInTheDocument();
  });

  it('calls onSetKey with entered key when Set API Key is clicked', () => {
    render(<ProviderKeyPrompt {...defaultProps} />);
    const input = screen.getByLabelText('API Key');
    fireEvent.change(input, { target: { value: 'my-secret-key' } });
    fireEvent.click(screen.getByRole('button', { name: 'Set API Key' }));
    expect(defaultProps.onSetKey).toHaveBeenCalledWith('my-secret-key');
  });

  it('does not call onSetKey when key is empty', () => {
    render(<ProviderKeyPrompt {...defaultProps} />);
    fireEvent.click(screen.getByRole('button', { name: 'Set API Key' }));
    expect(defaultProps.onSetKey).not.toHaveBeenCalled();
  });

  it('disables Set API Key button when input is empty', () => {
    render(<ProviderKeyPrompt {...defaultProps} />);
    expect(screen.getByRole('button', { name: 'Set API Key' })).toBeDisabled();
  });

  it('enables Set API Key button when input has value', () => {
    render(<ProviderKeyPrompt {...defaultProps} />);
    const input = screen.getByLabelText('API Key');
    fireEvent.change(input, { target: { value: 'key' } });
    expect(screen.getByRole('button', { name: 'Set API Key' })).not.toBeDisabled();
  });

  it('calls onFallback when Use Default Ollama Instead is clicked', () => {
    render(<ProviderKeyPrompt {...defaultProps} />);
    fireEvent.click(screen.getByRole('button', { name: 'Use Default Ollama Instead' }));
    expect(defaultProps.onFallback).toHaveBeenCalledTimes(1);
  });

  it('calls onCancel when Cancel is clicked', () => {
    render(<ProviderKeyPrompt {...defaultProps} />);
    fireEvent.click(screen.getByRole('button', { name: 'Cancel' }));
    expect(defaultProps.onCancel).toHaveBeenCalledTimes(1);
  });

  it('shows correct provider label for OpenRouter', () => {
    render(<ProviderKeyPrompt {...defaultProps} provider="openrouter" />);
    expect(screen.getByText('Provider: OpenRouter')).toBeInTheDocument();
  });

  it('shows correct provider label for Ollama', () => {
    render(<ProviderKeyPrompt {...defaultProps} provider="ollama" />);
    expect(screen.getByText('Provider: Ollama')).toBeInTheDocument();
  });

  it('shows "default" when model is empty', () => {
    render(<ProviderKeyPrompt {...defaultProps} model="" />);
    expect(screen.queryByText('Model:')).not.toBeInTheDocument();
  });

  it('clears input after Set API Key is clicked', () => {
    render(<ProviderKeyPrompt {...defaultProps} />);
    const input = screen.getByLabelText('API Key');
    fireEvent.change(input, { target: { value: 'test-key' } });
    fireEvent.click(screen.getByRole('button', { name: 'Set API Key' }));
    expect(defaultProps.onSetKey).toHaveBeenCalledWith('test-key');
    // Input should be cleared after submission
    expect(screen.getByLabelText('API Key')).toHaveValue('');
  });

  it('supports Enter key to submit', () => {
    render(<ProviderKeyPrompt {...defaultProps} />);
    const input = screen.getByLabelText('API Key');
    fireEvent.change(input, { target: { value: 'enter-key' } });
    fireEvent.keyDown(input, { key: 'Enter' });
    expect(defaultProps.onSetKey).toHaveBeenCalledWith('enter-key');
  });
});
