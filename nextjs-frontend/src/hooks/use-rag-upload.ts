'use client';

import * as React from 'react';
import { uploadRagFile, checkRagDuplicate, getRagFiles } from '@/lib/api';
import { useAppStore } from '@/lib/store';

export const ACCEPTED_TYPES =
  '.pdf,.docx,.txt,.md,.csv,.xlsx,.pptx,.html,.htm,.odt,.rtf,.yaml,.json';

/**
 * Encapsulates document upload for the knowledge base: duplicate checking,
 * chunked progress, and store sync of the resulting file list.
 */
export function useRagUpload() {
  const setRagFiles = useAppStore((s) => s.setRagFiles);
  const [uploading, setUploading] = React.useState(false);
  const [progress, setProgress] = React.useState(0);
  const [error, setError] = React.useState('');

  const upload = React.useCallback(
    async (fileList: FileList | File[]) => {
      const arr = Array.from(fileList);
      if (arr.length === 0) return;
      setUploading(true);
      setProgress(0);
      setError('');
      for (let i = 0; i < arr.length; i++) {
        const file = arr[i];
        try {
          const dup = await checkRagDuplicate({ filename: file.name });
          if (dup.is_duplicate) {
            setError(`"${file.name}" already exists.`);
            continue;
          }
          await uploadRagFile(file);
          setProgress(Math.round(((i + 1) / arr.length) * 100));
        } catch (err) {
          setError(`${file.name}: ${err instanceof Error ? err.message : 'Upload failed'}`);
        }
      }
      setUploading(false);
      try {
        const res = await getRagFiles();
        setRagFiles(res.files ?? []);
      } catch {
        // Keep the previous file list if the refresh fails.
      }
    },
    [setRagFiles],
  );

  return { upload, uploading, progress, error, setError };
}
