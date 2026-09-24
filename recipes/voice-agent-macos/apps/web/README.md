# Muse Glimmer local voice web app

A redistributable React interface for a private Muse Glimmer voice conversation. The browser requests a short-lived token from the fixed local endpoint `http://127.0.0.1:8787/api/token` and accepts media connections only to `ws://127.0.0.1:7880`.

## Requirements

- Node.js 22.12 or newer
- A local token service on `127.0.0.1:8787`
- A local LiveKit server on `127.0.0.1:7880`
- A LiveKit voice agent registered with the public name `assistant`

## Development

```bash
npm ci
npm run dev
```

Open `http://127.0.0.1:5173`. The development server binds only to loopback.

## Production

```bash
npm ci
npm run build
npm run serve
```

The production server binds only to `127.0.0.1` on port `5173` by default. The supervisor uses the constrained command `npm run serve -- --host 127.0.0.1 --port 5173`; all other host values are rejected. The server delivers the built application with a Content Security Policy and defensive browser headers. Production source maps are disabled.

## Quality checks

```bash
npm run typecheck
npm test
npm run lint
npm run format:check
npm run build
```

The package contains no token secrets, generated distribution files, third-party avatar definitions, or image branding assets.
