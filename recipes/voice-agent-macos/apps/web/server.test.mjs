import assert from "node:assert/strict";
import test from "node:test";

import { parseServeOptions } from "./server.mjs";

test("defaults to the approved loopback endpoint", () => {
  assert.deepEqual(parseServeOptions([], {}), {
    host: "127.0.0.1",
    port: 5173,
  });
});

test("accepts the supervisor serve contract", () => {
  assert.deepEqual(
    parseServeOptions(["--host", "127.0.0.1", "--port", "5173"], {}),
    {
      host: "127.0.0.1",
      port: 5173,
    },
  );
});

test("accepts equals-style options", () => {
  assert.deepEqual(parseServeOptions(["--host=127.0.0.1", "--port=5173"], {}), {
    host: "127.0.0.1",
    port: 5173,
  });
});

test("rejects non-loopback host overrides", () => {
  assert.throws(
    () => parseServeOptions(["--host", "0.0.0.0"], {}),
    /Host must be 127\.0\.0\.1/,
  );
  assert.throws(
    () => parseServeOptions(["--host", "localhost"], {}),
    /Host must be 127\.0\.0\.1/,
  );
});

test("rejects malformed ports, unknown flags, and duplicates", () => {
  assert.throws(
    () => parseServeOptions(["--port", "5173oops"], {}),
    /Port must be an integer/,
  );
  assert.throws(
    () => parseServeOptions(["--port", "0"], {}),
    /Port must be 5173/,
  );
  assert.throws(
    () => parseServeOptions(["--port", "4173"], {}),
    /Port must be 5173/,
  );
  assert.throws(
    () => parseServeOptions(["--public"], {}),
    /Unknown serve argument/,
  );
  assert.throws(
    () => parseServeOptions(["--host", "127.0.0.1", "--host", "127.0.0.1"], {}),
    /Duplicate serve argument/,
  );
});
