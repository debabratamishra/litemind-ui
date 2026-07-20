'use client';

import * as React from 'react';
import { Paperclip, Loader2 } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { ACCEPTED_TYPES } from '@/hooks/use-rag-upload';
import { cn } from '@/lib/utils';

/**
 * Paperclip button that sits beside the RAG query field. Opens a file picker
 * on click and also accepts files dropped directly onto it.
 */
export function RagAttachButton({
  onFiles,
  uploading,
  disabled,
  className,
}: {
  onFiles: (files: FileList) => void;
  uploading: boolean;
  disabled?: boolean;
  className?: string;
}) {
  const inputRef = React.useRef<HTMLInputElement>(null);
  const [dragActive, setDragActive] = React.useState(false);

  return (
    <>
      <Button
        type="button"
        variant="outline"
        size="sm"
        disabled={disabled || uploading}
        onClick={() => inputRef.current?.click()}
        onDragOver={(e) => {
          e.preventDefault();
          setDragActive(true);
        }}
        onDragLeave={() => setDragActive(false)}
        onDrop={(e) => {
          e.preventDefault();
          setDragActive(false);
          if (e.dataTransfer.files?.length) onFiles(e.dataTransfer.files);
        }}
        aria-label={uploading ? 'Uploading documents…' : 'Attach documents'}
        className={cn(
          'h-[60px] w-11 shrink-0',
          dragActive && 'border-primary bg-primary/5',
          className,
        )}
      >
        {uploading ? (
          <Loader2 className="size-4 animate-spin" aria-hidden="true" />
        ) : (
          <Paperclip className="size-4" aria-hidden="true" />
        )}
      </Button>
      <input
        ref={inputRef}
        type="file"
        accept={ACCEPTED_TYPES}
        multiple
        className="hidden"
        onChange={(e) => {
          if (e.target.files?.length) onFiles(e.target.files);
          e.target.value = '';
        }}
        aria-hidden="true"
      />
    </>
  );
}
