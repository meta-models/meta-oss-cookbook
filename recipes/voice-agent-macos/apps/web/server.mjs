import { createReadStream, existsSync, statSync } from "node:fs";
import { createServer } from "node:http";
import { extname, join, normalize, resolve, sep } from "node:path";
import { fileURLToPath } from "node:url";

const HOST = "127.0.0.1";
const DEFAULT_PORT = 5173;
const APP_DIRECTORY = resolve(
  fileURLToPath(new URL("./dist", import.meta.url)),
);
const CSP = [
  "default-src 'self'",
  "base-uri 'none'",
  "connect-src 'self' http://127.0.0.1:8787 ws://127.0.0.1:7880",
  "font-src 'self'",
  "form-action 'none'",
  "frame-ancestors 'none'",
  "img-src 'self' data:",
  "media-src 'self' blob:",
  "object-src 'none'",
  "script-src 'self'",
  "style-src 'self'",
  "worker-src 'self' blob:",
].join("; ");

const MIME_TYPES = new Map([
  [".css", "text/css; charset=utf-8"],
  [".html", "text/html; charset=utf-8"],
  [".js", "text/javascript; charset=utf-8"],
  [".json", "application/json; charset=utf-8"],
  [".svg", "image/svg+xml"],
  [".woff2", "font/woff2"],
]);

function applySecurityHeaders(response) {
  response.setHeader("Cache-Control", "no-store");
  response.setHeader("Content-Security-Policy", CSP);
  response.setHeader("Cross-Origin-Opener-Policy", "same-origin");
  response.setHeader(
    "Permissions-Policy",
    "camera=(), geolocation=(), microphone=(self)",
  );
  response.setHeader("Referrer-Policy", "no-referrer");
  response.setHeader("X-Content-Type-Options", "nosniff");
  response.setHeader("X-Frame-Options", "DENY");
}

function resolveRequestPath(requestUrl) {
  const pathname = decodeURIComponent(
    new URL(requestUrl ?? "/", `http://${HOST}`).pathname,
  );
  const normalizedPath = normalize(pathname).replace(/^[/\\]+/, "");
  const requestedPath = resolve(join(APP_DIRECTORY, normalizedPath));
  if (
    requestedPath !== APP_DIRECTORY &&
    !requestedPath.startsWith(`${APP_DIRECTORY}${sep}`)
  ) {
    return undefined;
  }
  if (existsSync(requestedPath) && statSync(requestedPath).isFile()) {
    return requestedPath;
  }
  return join(APP_DIRECTORY, "index.html");
}

export function createAppServer() {
  if (!existsSync(join(APP_DIRECTORY, "index.html"))) {
    throw new Error("Production assets are missing. Run npm run build first.");
  }

  return createServer((request, response) => {
    applySecurityHeaders(response);
    if (request.method !== "GET" && request.method !== "HEAD") {
      response.writeHead(405, { Allow: "GET, HEAD" });
      response.end("Method Not Allowed");
      return;
    }

    let filePath;
    try {
      filePath = resolveRequestPath(request.url);
    } catch {
      response.writeHead(400);
      response.end("Bad Request");
      return;
    }
    if (!filePath) {
      response.writeHead(403);
      response.end("Forbidden");
      return;
    }

    response.setHeader(
      "Content-Type",
      MIME_TYPES.get(extname(filePath)) ?? "application/octet-stream",
    );
    response.writeHead(200);
    if (request.method === "HEAD") {
      response.end();
      return;
    }
    createReadStream(filePath).pipe(response);
  });
}

export function parseServeOptions(args, environment = process.env) {
  let host = HOST;
  let port = environment.PORT ?? String(DEFAULT_PORT);
  const provided = new Set();

  for (let index = 0; index < args.length; index += 1) {
    const argument = args[index];
    const [flag, inlineValue] = argument.split("=", 2);
    if (flag !== "--host" && flag !== "--port") {
      throw new Error(`Unknown serve argument: ${flag}`);
    }
    if (provided.has(flag)) {
      throw new Error(`Duplicate serve argument: ${flag}`);
    }

    const value = inlineValue ?? args[++index];
    if (!value || value.startsWith("--")) {
      throw new Error(`${flag} requires a value.`);
    }
    provided.add(flag);
    if (flag === "--host") {
      host = value;
    } else {
      port = value;
    }
  }

  if (host !== HOST) {
    throw new Error(`Host must be ${HOST}.`);
  }
  if (!/^\d+$/.test(port)) {
    throw new Error("Port must be an integer between 1 and 65535.");
  }
  const portValue = Number.parseInt(port, 10);
  if (portValue !== DEFAULT_PORT) {
    throw new Error(`Port must be ${DEFAULT_PORT}.`);
  }

  return { host, port: portValue };
}

const entryPoint = process.argv[1] ? resolve(process.argv[1]) : undefined;
if (entryPoint === fileURLToPath(import.meta.url)) {
  const { host, port } = parseServeOptions(process.argv.slice(2));
  const server = createAppServer();
  server.listen(port, host, () => {
    console.log(`Muse Glimmer web app listening at http://${host}:${port}`);
  });
}
