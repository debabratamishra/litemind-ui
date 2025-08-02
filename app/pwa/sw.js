// pwa/sw.js
self.addEventListener('install', evt => self.skipWaiting());
self.addEventListener('fetch', () => {});      // no caching
