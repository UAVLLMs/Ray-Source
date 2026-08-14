# Raysource Web Client: Independent Deployment

This package contains the browser UI, manual viewer, local manual images and
captions, monitoring UI, chunk-management UI, and the Node.js API gateway. It
does not import or read files from the retrieval backend.

## Connect to a backend

1. Copy `.env.example` to `.env`.
2. Set `RAGV6_API_ORIGIN` to the backend HTTP origin.
3. Set `RAGV6_API_TOKEN` to the backend's `KAFU_API_TOKEN` value.
4. Run `npm start` or `./start.ps1`.

When the backend is unavailable, `/health` reports `backend.reachable=false`;
the UI, manual pages, images, and caption endpoint remain available. Chat,
translation, retrieval, and chunk-management calls return an explicit upstream
unavailable error instead of terminating the web process.

The browser never receives the backend token. Only the Node.js gateway adds it
to server-to-server requests.
