const CACHE_NAME = "raysource-shell-context-component-audit-20260812d";
const SHELL = ["/", "/ragv6/", "/ragv6/audit-contract.js", "/ragv6/app.js", "/ragv6/styles.css", "/manifest.json", "/ragv6/ray-source-icon-192.png", "/ragv6/ray-source-icon-512.png"];

self.addEventListener("install", (event) => {
  event.waitUntil(caches.open(CACHE_NAME).then((cache) => cache.addAll(SHELL)).then(() => self.skipWaiting()));
});

self.addEventListener("activate", (event) => {
  event.waitUntil(self.clients.claim());
});

self.addEventListener("fetch", (event) => {
  const request = event.request;
  if (request.method !== "GET" || new URL(request.url).pathname.includes("/api/")) return;
  event.respondWith(fetch(request).catch(() => caches.match(request).then((cached) => cached || caches.match("/ragv6/"))));
});
