import type { NextRequest } from 'next/server';
import { proxyStreamingPost } from '@/lib/backend-proxy';

// Streaming responses must never be statically optimized / cached.
export const runtime = 'nodejs';
export const dynamic = 'force-dynamic';

export async function POST(req: NextRequest) {
  return proxyStreamingPost(req, '/api/chat/stream');
}
