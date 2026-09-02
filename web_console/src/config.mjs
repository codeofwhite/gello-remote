export function normalizePort(value, fallback) {
  const port = Number(value ?? fallback);
  if (!Number.isInteger(port) || port < 1 || port > 65535) {
    throw new Error(`Invalid port: ${value}`);
  }
  return port;
}

export function normalizeFps(value, fallback = 30) {
  const fps = Number(value ?? fallback);
  if (!Number.isInteger(fps) || fps < 1 || fps > 240) {
    throw new Error(`Invalid FPS: ${value}`);
  }
  return fps;
}

export function resolveStreamConfig(runtimeConfig, search = "") {
  const params = new URLSearchParams(search);
  const signalingServer = params.get("server") || runtimeConfig.stream_host;
  if (!signalingServer) {
    throw new Error("WebRTC stream host is missing; set PUBLIC_IP or --stream-host on the Isaac server");
  }
  const mediaServer = params.get("mediaServer") || runtimeConfig.media_host || signalingServer;
  return {
    signalingServer,
    signalingPort: normalizePort(params.get("signalingPort"), runtimeConfig.signaling_port),
    mediaServer,
    mediaPort: normalizePort(params.get("mediaPort"), runtimeConfig.media_port),
    width: normalizePort(runtimeConfig.width, 960),
    height: normalizePort(runtimeConfig.height, 720),
    fps: normalizeFps(runtimeConfig.fps, 30),
  };
}
