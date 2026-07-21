import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen } from '@testing-library/react';
import { RagAttachButton } from './rag-attach-button';
// API is mocked for uploadRagFile testing

// Mock the API module
vi.mock('@/lib/api', () => ({
  uploadRagFile: vi.fn(),
}));

describe('RagAttachButton', () => {
  const mockOnFiles = vi.fn();

  beforeEach(() => {
    vi.clearAllMocks();
  });

  it('renders the button with correct aria-label', () => {
    render(<RagAttachButton onFiles={mockOnFiles} uploading={false} />);
    const button = screen.getByRole('button', { name: /attach documents/i });
    expect(button).toBeInTheDocument();
  });

  it('shows loading state when uploading', () => {
    render(<RagAttachButton onFiles={mockOnFiles} uploading={true} />);
    const button = screen.getByRole('button', { name: /uploading documents/i });
    expect(button).toBeInTheDocument();
  });

  it('is disabled when uploading', () => {
    render(<RagAttachButton onFiles={mockOnFiles} uploading={true} />);
    const button = screen.getByRole('button');
    expect(button).toBeDisabled();
  });

  it('is disabled when disabled prop is true', () => {
    render(<RagAttachButton onFiles={mockOnFiles} uploading={false} disabled={true} />);
    const button = screen.getByRole('button');
    expect(button).toBeDisabled();
  });

  it('has correct default styling', () => {
    const { container } = render(<RagAttachButton onFiles={mockOnFiles} uploading={false} />);
    const button = container.querySelector('button');
    expect(button).toHaveClass('h-[60px]', 'w-11');
  });

  it('applies custom className', () => {
    const { container } = render(
      <RagAttachButton onFiles={mockOnFiles} uploading={false} className="custom-class" />,
    );
    const button = container.querySelector('button');
    expect(button).toHaveClass('custom-class');
  });

  it('has hidden file input with correct attributes', () => {
    render(<RagAttachButton onFiles={mockOnFiles} uploading={false} />);
    const input = document.querySelector('input[type="file"]');
    expect(input).toBeInTheDocument();
    expect(input).toHaveAttribute('accept', expect.stringContaining('.pdf'));
    expect(input).toHaveAttribute('multiple');
    expect(input).toHaveClass('hidden');
  });

  it('shows spinner when uploading', () => {
    render(<RagAttachButton onFiles={mockOnFiles} uploading={true} />);
    const spinner = document.querySelector('[class*="animate-spin"]');
    expect(spinner).toBeInTheDocument();
  });

  it('shows paperclip icon when not uploading', () => {
    render(<RagAttachButton onFiles={mockOnFiles} uploading={false} />);
    const icon = document.querySelector('[class*="lucide-paperclip"]');
    expect(icon).toBeInTheDocument();
  });
});