import type { NextConfig } from 'next';

const nextConfig: NextConfig = {
  // Produce a minimal self-contained server bundle for Docker
  output: 'standalone',

  // Allow images from localhost (backend) and common CDN domains
  images: {
    remotePatterns: [
      { protocol: 'http', hostname: 'localhost' },
      { protocol: 'http', hostname: 'backend' },
    ],
  },

  // The FastAPI backend is on a different origin. Proxy non-streaming API
  // traffic to it so the browser only ever talks to the same origin (no CORS)
  // and the Next server — which can resolve the Docker `backend` hostname —
  // reaches the backend on the client's behalf. `:path*` is required so
  // subpaths such as /models/enhanced and /health/ready are proxied too.
  //
  // NOTE: the streaming endpoints (/api/chat/stream, /api/chat/web-search,
  // /api/rag/query) are served by route handlers under src/app/api/** so the
  // response body is piped through chunk-by-chunk. The rewrite proxy below
  // buffers streams until completion, which would defeat streaming in the UI.
  async rewrites() {
    const apiUrl = process.env.NEXT_PUBLIC_API_URL ?? 'http://localhost:8000';
    return [
      {
        source: '/api/:path*',
        destination: `${apiUrl}/api/:path*`,
      },
      {
        source: '/health/:path*',
        destination: `${apiUrl}/health/:path*`,
      },
      {
        source: '/models/:path*',
        destination: `${apiUrl}/models/:path*`,
      },
    ];
  },
};

export default nextConfig;
