import assert from "node:assert/strict";
import test from "node:test";

import { normalizeFps, normalizePort, resolveStreamConfig } from "../src/config.mjs";

test("resolves runtime stream config", () => {
  assert.deepEqual(
    resolveStreamConfig({
      stream_host: "203.0.113.10",
      signaling_port: 49100,
      media_port: 47998,
      width: 1280,
      height: 720,
      fps: 30,
    }),
    {
      signalingServer: "203.0.113.10",
      signalingPort: 49100,
      mediaServer: "203.0.113.10",
      mediaPort: 47998,
      width: 1280,
      height: 720,
      fps: 30,
    },
  );
});

test("URL parameters override stream endpoints", () => {
  const config = resolveStreamConfig(
    { stream_host: "server", signaling_port: 49100, media_port: 47998, width: 1280, height: 720 },
    "?server=localhost&signalingPort=50000&mediaServer=media.example&mediaPort=50001",
  );
  assert.equal(config.signalingServer, "localhost");
  assert.equal(config.signalingPort, 50000);
  assert.equal(config.mediaServer, "media.example");
  assert.equal(config.mediaPort, 50001);
});

test("rejects missing host and invalid ports", () => {
  assert.throws(
    () => resolveStreamConfig({ signaling_port: 49100, media_port: 47998, width: 1280, height: 720 }),
    /stream host is missing/,
  );
  assert.throws(() => normalizePort(70000, 49100), /Invalid port/);
  assert.throws(() => normalizeFps(0), /Invalid FPS/);
});
