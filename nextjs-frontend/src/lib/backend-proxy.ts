import type { NextRequest } from 'next/server';

/**
 * Server-side streaming proxy used by the Next.js route handlers. It forwards a
 * POST (with its JSON body) to the backend and pipes the upstream response body
 * straight back to the client, preserving the streaming semantics.
 */

const BACKEND_BASE =
  process.env.NEXT_PUBLIC_API_URL ?? process.env.BACKEND_URL ?? 'http://localhost:8000';

export interface ProxyOptions {
  /** Content-Type to advertise on the proxied response. Defaults to the
   *  upstream's own content type, falling back to application/json. */
  defaultContentType?: string;
}

export async function proxyStreamingPost(
  req: NextRequest,
  backendPath: string,
  opts: ProxyOptions = {},
): Promise<Response> {
  const target = `${BACKEND_BASE}${backendPath}`;
  const body = await req.text();
  const contentType = req.headers.get('content-type') ?? 'application/json';

  const upstream = await fetch(target, {
    method: 'POST',
    headers: { 'Content-Type': contentType },
    body,
    cache: 'no-store',
  });

  const responseContentType =
    opts.defaultContentType ?? upstream.headers.get('content-type') ?? 'application/json';

  return new Response(upstream.body, {
    status: upstream.status,
    headers: {
      'content-type': responseContentType,
      'cache-control': 'no-store',
    },
  });
}
