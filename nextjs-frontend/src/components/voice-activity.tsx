'use client';

import { cn } from '@/lib/utils';
import type { VoiceState } from '@/hooks/use-realtime-voice';

const BARS = 5;

export default function VoiceActivityIndicator({ state, className }: { state: VoiceState; className?: string }) {
  const active = state === 'listening' || state === 'speaking';
  return (
    <span className={cn('inline-flex h-4 items-end gap-0.5', className)} aria-hidden="true">
      {Array.from({ length: BARS }).map((_, i) => (
        <span
          key={i}
          className={cn(
            'w-0.5 rounded-full bg-primary transition-transform duration-150 motion-reduce:animate-none',
            active ? (state === 'speaking' ? 'h-4 animate-pulse' : 'h-3') : 'h-1 opacity-50',
          )}
          style={active ? { animationDelay: `${i * 80}ms` } : undefined}
        />
      ))}
    </span>
  );
}
