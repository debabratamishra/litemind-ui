import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen, waitFor, fireEvent } from '@testing-library/react';
import { MemorySettings } from './settings-panel';
import * as api from '@/lib/api';
import type { MemoryRecord } from '@/lib/types';

vi.mock('@/lib/api', () => ({
  getMemories: vi.fn(),
  addMemory: vi.fn(),
  updateMemory: vi.fn(),
  deleteMemory: vi.fn(),
  clearMemories: vi.fn(),
}));

const mem = (id: string, content: string): MemoryRecord => ({
  id,
  content,
  source: 'auto',
  created_at: '2026-08-23T00:00:00',
  updated_at: '2026-08-23T00:00:00',
});

describe('MemorySettings', () => {
  beforeEach(() => {
    vi.clearAllMocks();
    vi.mocked(api.getMemories).mockResolvedValue([mem('m1', 'Prefers dark mode')]);
  });

  it('lists memories on load', async () => {
    render(<MemorySettings />);
    expect(await screen.findByText('Prefers dark mode')).toBeInTheDocument();
  });

  it('adds a memory', async () => {
    vi.mocked(api.addMemory).mockResolvedValue(mem('m2', 'New fact'));
    render(<MemorySettings />);
    await screen.findByText('Prefers dark mode');
    fireEvent.change(screen.getByPlaceholderText(/add a memory/i), {
      target: { value: 'New fact' },
    });
    fireEvent.click(screen.getByRole('button', { name: /add/i }));
    await waitFor(() => expect(api.addMemory).toHaveBeenCalledWith('New fact'));
    expect(await screen.findByText('New fact')).toBeInTheDocument();
  });

  it('deletes a memory', async () => {
    vi.mocked(api.deleteMemory).mockResolvedValue(undefined);
    render(<MemorySettings />);
    await screen.findByText('Prefers dark mode');
    fireEvent.click(screen.getByRole('button', { name: /delete/i }));
    await waitFor(() => expect(api.deleteMemory).toHaveBeenCalledWith('m1'));
    await waitFor(() =>
      expect(screen.queryByText('Prefers dark mode')).not.toBeInTheDocument(),
    );
  });
});
