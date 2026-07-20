'use client';

import { Mic, MicOff, Loader2 } from 'lucide-react';
import { cn } from '@/lib/utils';
import { useVoiceInput } from '@/hooks/use-voice-input';
import { Tooltip, TooltipTrigger, TooltipContent } from '@/components/ui/tooltip';

interface VoiceInputButtonProps {
  onTranscript: (text: string) => void;
  className?: string;
  disabled?: boolean;
}

export default function VoiceInputButton({
  onTranscript,
  className,
  disabled = false,
}: VoiceInputButtonProps) {
  const { state, isSupported, start, stop } = useVoiceInput(onTranscript);

  if (!isSupported) {
    return null; // silently omit if browser doesn't support it
  }

  const isListening = state === 'listening';
  const isProcessing = state === 'processing';
  const active = isListening || isProcessing;

  const label = isListening
    ? 'Stop recording'
    : isProcessing
      ? 'Processing…'
      : 'Start voice input';

  return (
    <Tooltip>
      <TooltipTrigger
        render={
          <button
            type="button"
            aria-label={label}
            disabled={disabled || isProcessing}
            onClick={active ? stop : start}
            className={cn(
              'flex h-8 w-8 items-center justify-center rounded-full transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring disabled:opacity-50',
              active
                ? 'bg-red-500 text-white hover:bg-red-600 animate-pulse'
                : 'bg-muted text-muted-foreground hover:bg-muted/80 hover:text-foreground',
              className,
            )}
          />
        }
      >
        {isProcessing ? (
          <Loader2 className="h-4 w-4 animate-spin" aria-hidden="true" />
        ) : isListening ? (
          <MicOff className="h-4 w-4" aria-hidden="true" />
        ) : (
          <Mic className="h-4 w-4" aria-hidden="true" />
        )}
      </TooltipTrigger>
      <TooltipContent>{label}</TooltipContent>
    </Tooltip>
  );
}
