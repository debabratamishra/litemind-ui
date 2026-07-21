import { describe, it, expect, vi, beforeEach } from 'vitest';
import { renderHook, act } from '@testing-library/react';
import * as api from '@/lib/api';

// Mock the API functions
vi.mock('@/lib/api', () => ({
  uploadRagFile: vi.fn(),
  checkRagDuplicate: vi.fn(),
  getRagFiles: vi.fn(),
}));

// Mock the store
const mockSetRagFiles = vi.fn();

vi.mock('@/lib/store', () => ({
  useAppStore: vi.fn((selector) => selector({ setRagFiles: mockSetRagFiles })),
}));

// Import the hook after mocks are set up
import { useRagUpload } from './use-rag-upload';

describe('useRagUpload', () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it('returns initial state with no upload in progress', () => {
    const { result } = renderHook(() => useRagUpload());
    expect(result.current.uploading).toBe(false);
    expect(result.current.progress).toBe(0);
    expect(result.current.error).toBe('');
  });

  it('does nothing when file list is empty', async () => {
    const { result } = renderHook(() => useRagUpload());
    const fileList = { length: 0 } as FileList;

    await act(async () => {
      await result.current.upload(fileList);
    });

    expect(api.uploadRagFile).not.toHaveBeenCalled();
    expect(result.current.uploading).toBe(false);
  });

  it('uploads a single file and updates progress', async () => {
    const mockFile = new File(['content'], 'test.txt', { type: 'text/plain' });
    (api.checkRagDuplicate as any).mockResolvedValue({ is_duplicate: false });
    (api.uploadRagFile as any).mockResolvedValue(undefined);
    (api.getRagFiles as any).mockResolvedValue({ files: [{ filename: 'test.txt' }] });

    const { result } = renderHook(() => useRagUpload());
    const fileList = { length: 1 } as FileList;
    Object.defineProperty(fileList, '0', { value: mockFile });

    await act(async () => {
      await result.current.upload(fileList);
    });

    expect(api.checkRagDuplicate).toHaveBeenCalledWith({ filename: 'test.txt' });
    expect(api.uploadRagFile).toHaveBeenCalledWith(mockFile);
    expect(result.current.progress).toBe(100);
    expect(result.current.error).toBe('');
  });

  it('skips upload when file is a duplicate', async () => {
    const mockFile = new File(['content'], 'existing.txt', { type: 'text/plain' });
    (api.checkRagDuplicate as any).mockResolvedValue({ is_duplicate: true, reason: 'exists' });

    const { result } = renderHook(() => useRagUpload());
    const fileList = { length: 1 } as FileList;
    Object.defineProperty(fileList, '0', { value: mockFile });

    await act(async () => {
      await result.current.upload(fileList);
    });

    expect(api.uploadRagFile).not.toHaveBeenCalled();
    expect(result.current.error).toContain('already exists');
  });

  it('handles upload errors gracefully', async () => {
    const mockFile = new File(['content'], 'test.txt', { type: 'text/plain' });
    (api.checkRagDuplicate as any).mockResolvedValue({ is_duplicate: false });
    (api.uploadRagFile as any).mockRejectedValue(new Error('Network error'));
    (api.getRagFiles as any).mockResolvedValue({ files: [] });

    const { result } = renderHook(() => useRagUpload());
    const fileList = { length: 1 } as FileList;
    Object.defineProperty(fileList, '0', { value: mockFile });

    await act(async () => {
      await result.current.upload(fileList);
    });

    expect(result.current.error).toContain('Network error');
  });

  it('processes multiple files with correct progress', async () => {
    const mockFile1 = new File(['content1'], 'test1.txt', { type: 'text/plain' });
    const mockFile2 = new File(['content2'], 'test2.txt', { type: 'text/plain' });
    (api.checkRagDuplicate as any).mockResolvedValue({ is_duplicate: false });
    (api.uploadRagFile as any).mockResolvedValue(undefined);
    (api.getRagFiles as any).mockResolvedValue({ files: [{ filename: 'test1.txt' }, { filename: 'test2.txt' }] });

    const { result } = renderHook(() => useRagUpload());
    const fileList = { length: 2 } as FileList;
    Object.defineProperty(fileList, '0', { value: mockFile1 });
    Object.defineProperty(fileList, '1', { value: mockFile2 });

    await act(async () => {
      await result.current.upload(fileList);
    });

    expect(api.uploadRagFile).toHaveBeenCalledTimes(2);
    expect(mockSetRagFiles).toHaveBeenCalledWith([
      { filename: 'test1.txt' },
      { filename: 'test2.txt' },
    ]);
  });

  it('sets error when duplicate check fails', async () => {
    const mockFile = new File(['content'], 'test.txt', { type: 'text/plain' });
    (api.checkRagDuplicate as any).mockRejectedValue(new Error('Check failed'));

    const { result } = renderHook(() => useRagUpload());
    const fileList = { length: 1 } as FileList;
    Object.defineProperty(fileList, '0', { value: mockFile });

    await act(async () => {
      await result.current.upload(fileList);
    });

    expect(result.current.error).toContain('Check failed');
  });

  it('clears error when setError is called', async () => {
    const { result } = renderHook(() => useRagUpload());

    act(() => {
      result.current.setError('Some error');
    });

    expect(result.current.error).toBe('Some error');

    act(() => {
      result.current.setError('');
    });

    expect(result.current.error).toBe('');
  });
});