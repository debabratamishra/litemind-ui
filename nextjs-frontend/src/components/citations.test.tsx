import { describe, it, expect, vi } from 'vitest';
import { render, screen, fireEvent } from '@testing-library/react';
import { SourcesButton, CitationsDialog } from './citations';
import type { Citation } from '@/lib/web-search-citations';

describe('SourcesButton', () => {
  it('renders with correct count', () => {
    render(<SourcesButton count={3} onClick={() => {}} />);
    expect(screen.getByText('Sources (3)')).toBeInTheDocument();
  });

  it('renders with singular label for count 1', () => {
    render(<SourcesButton count={1} onClick={() => {}} />);
    expect(screen.getByText('Sources (1)')).toBeInTheDocument();
  });

  it('calls onClick when clicked', () => {
    const onClick = vi.fn();
    render(<SourcesButton count={2} onClick={onClick} />);
    const button = screen.getByRole('button', { name: /sources/i });
    fireEvent.click(button);
    expect(onClick).toHaveBeenCalledTimes(1);
  });

  it('has correct accessibility attributes', () => {
    render(<SourcesButton count={2} onClick={() => {}} />);
    const button = screen.getByRole('button');
    expect(button).toHaveAttribute('aria-haspopup', 'dialog');
  });

  it('has correct styling classes', () => {
    const { container } = render(<SourcesButton count={2} onClick={() => {}} />);
    const button = container.querySelector('button');
    expect(button).toHaveClass('gap-1.5');
  });
});

describe('CitationsDialog', () => {
  const mockSources: Citation[] = [
    { index: 0, title: 'Source 1', url: 'https://example.com/1' },
    { index: 1, title: 'Source 2', url: 'https://example.com/2', domain: 'example.com', snippet: 'Test snippet' },
  ];

  it('renders dialog with correct title', () => {
    render(<CitationsDialog open={true} onOpenChange={() => {}} sources={mockSources} focusIndex={null} />);
    expect(screen.getByText('Sources')).toBeInTheDocument();
  });

  it('displays correct source count in description', () => {
    render(<CitationsDialog open={true} onOpenChange={() => {}} sources={mockSources} focusIndex={null} />);
    expect(screen.getByText('2 references cited by the assistant.')).toBeInTheDocument();
  });

  it('displays correct singular source count', () => {
    render(<CitationsDialog open={true} onOpenChange={() => {}} sources={[mockSources[0]]} focusIndex={null} />);
    expect(screen.getByText('1 reference cited by the assistant.')).toBeInTheDocument();
  });

  it('renders source titles', () => {
    render(<CitationsDialog open={true} onOpenChange={() => {}} sources={mockSources} focusIndex={null} />);
    expect(screen.getByText('Source 1')).toBeInTheDocument();
    expect(screen.getByText('Source 2')).toBeInTheDocument();
  });

  it('renders source URLs as links', () => {
    render(<CitationsDialog open={true} onOpenChange={() => {}} sources={mockSources} focusIndex={null} />);
    const links = screen.getAllByRole('link');
    expect(links).toHaveLength(2);
  });

  it('renders source domains', () => {
    render(<CitationsDialog open={true} onOpenChange={() => {}} sources={mockSources} focusIndex={null} />);
    expect(screen.getByText('example.com')).toBeInTheDocument();
  });

  it('renders source snippets', () => {
    render(<CitationsDialog open={true} onOpenChange={() => {}} sources={mockSources} focusIndex={null} />);
    expect(screen.getByText('Test snippet')).toBeInTheDocument();
  });

  it('renders source index numbers', () => {
    render(<CitationsDialog open={true} onOpenChange={() => {}} sources={mockSources} focusIndex={null} />);
    expect(screen.getByText('[0]')).toBeInTheDocument();
    expect(screen.getByText('[1]')).toBeInTheDocument();
  });

  it('highlights focused citation', () => {
    render(<CitationsDialog open={true} onOpenChange={() => {}} sources={mockSources} focusIndex={0} />);
    const sourceContainer = screen.getByText('Source 1').closest('.rounded-lg');
    expect(sourceContainer).toHaveClass('bg-primary/5');
  });

  it('renders sources without URLs', () => {
    const sourcesWithoutUrl: Citation[] = [
      { index: 0, title: 'Source without URL' },
    ];
    render(<CitationsDialog open={true} onOpenChange={() => {}} sources={sourcesWithoutUrl} focusIndex={null} />);
    expect(screen.getByText('Source without URL')).toBeInTheDocument();
    const links = screen.queryAllByRole('link');
    expect(links).toHaveLength(0);
  });

  it('calls onOpenChange when closed', () => {
    const onOpenChange = vi.fn();
    render(<CitationsDialog open={true} onOpenChange={onOpenChange} sources={mockSources} focusIndex={null} />);
    // Dialog should be open
    expect(screen.getByRole('dialog')).toBeInTheDocument();
  });
});