import type { NextRequest } from 'next/server';
import { proxyStreamingPost } from '@/lib/backend-proxy';

// Streaming responses must never be statically optimized / cached.
export const runtime = 'nodejs';
export const dynamic = 'force-dynamic';

export async function POST(req: NextRequest) {
  // Backend returns plain-text chunks (no SSE envelope) for this endpoint.
  return proxyStreamingPost(req, '/api/rag/query', {
    defaultContentType: 'text/plain',
  });
}
