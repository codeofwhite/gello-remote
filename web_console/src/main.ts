/* SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved. */
/* SPDX-License-Identifier: MIT */

import "./index.css";
import {
  AppStreamer,
  LogLevel,
  StreamType,
  eAction,
  eStatus,
  type DirectConfig,
  type StreamEvent,
  type StreamProps,
} from "@nvidia/omniverse-webrtc-streaming-library";

import {
  GelloSerialReader,
  classifyGripper,
  mapJointPositions,
} from "../../src/gello_remote/web/gello_web_serial.mjs";
import { resolveStreamConfig } from "./config.mjs";

const byId = <T extends HTMLElement>(id: string): T => {
  const element = document.getElementById(id);
  if (!element) throw new Error(`Missing UI element #${id}`);
  return element as T;
};

const streamStatus = byId<HTMLSpanElement>("stream-status");
const streamOverlay = byId<HTMLDivElement>("stream-overlay");
const streamMessage = byId<HTMLElement>("stream-message");
const streamDetail = byId<HTMLSpanElement>("stream-detail");
const streamEndpoint = byId<HTMLSpanElement>("stream-endpoint");
const videoDimensions = byId<HTMLSpanElement>("video-dimensions");
const activeViewLabel = byId<HTMLSpanElement>("active-view-label");
const mainDiv = byId<HTMLDivElement>("main-div");
const video = byId<HTMLVideoElement>("remote-video");

function setStreamState(state: "connecting" | "live" | "failed", message: string, detail: string) {
  streamStatus.textContent = state === "live" ? "Live" : state === "failed" ? "Failed" : "Connecting";
  streamStatus.className = `status status-${state}`;
  streamMessage.textContent = message;
  streamDetail.textContent = detail;
  const live = state === "live";
  streamOverlay.hidden = live;
  mainDiv.hidden = !live;
}

video.addEventListener("loadedmetadata", () => {
  videoDimensions.textContent = `Video: ${video.videoWidth}×${video.videoHeight}`;
});
video.addEventListener("resize", () => {
  videoDimensions.textContent = `Video: ${video.videoWidth}×${video.videoHeight}`;
});

async function connectStream() {
  setStreamState("connecting", "Connecting to Isaac Sim WebRTC…", "Loading stream configuration.");
  try {
    const response = await fetch("/teleop-config.json", { cache: "no-store" });
    if (!response.ok) throw new Error(`Runtime config returned HTTP ${response.status}`);
    const endpoint = resolveStreamConfig(await response.json(), window.location.search);
    streamEndpoint.textContent = `WebRTC: ${endpoint.signalingServer}:${endpoint.signalingPort} / UDP ${endpoint.mediaPort}`;
    setStreamState(
      "connecting",
      "Waiting for Isaac Sim stream…",
      `Signaling ${endpoint.signalingServer}:${endpoint.signalingPort}; media ${endpoint.mediaServer}:${endpoint.mediaPort}`,
    );
    const streamConfig: DirectConfig = {
      videoElementId: "remote-video",
      signalingServer: endpoint.signalingServer,
      signalingPort: endpoint.signalingPort,
      mediaServer: endpoint.mediaServer,
      mediaPort: endpoint.mediaPort,
      forceWSS: false,
      width: endpoint.width,
      height: endpoint.height,
      fps: endpoint.fps,
      onStart: (event: StreamEvent) => {
        if (event.action !== eAction.start) return;
        if (event.status === eStatus.success) {
          setStreamState("live", "Stream live", "Isaac Sim video connected.");
        } else if (event.status === eStatus.error) {
          setStreamState("failed", "WebRTC connection failed", `${event.info || "Unknown stream error"}`);
        }
      },
      onCustomEvent: (event: unknown) => console.debug("WebRTC custom event", event),
      onStop: (event: StreamEvent) => {
        setStreamState("failed", "WebRTC stream stopped", `${event.info || "Connection closed"}`);
      },
    };
    const streamProps: StreamProps = {
      streamSource: StreamType.DIRECT,
      logLevel: LogLevel.INFO,
      streamConfig,
    };
    await AppStreamer.connect(streamProps);
  } catch (error) {
    setStreamState("failed", "WebRTC setup failed", error instanceof Error ? error.message : String(error));
  }
}

const connectButton = byId<HTMLButtonElement>("serial-connect");
const disconnectButton = byId<HTMLButtonElement>("serial-disconnect");
const captureHomeButton = byId<HTMLButtonElement>("capture-home");
const captureOpenButton = byId<HTMLButtonElement>("capture-open");
const captureClosedButton = byId<HTMLButtonElement>("capture-closed");
const enableControl = byId<HTMLInputElement>("enable-control");
const directionElements = [...document.querySelectorAll<HTMLSelectElement>("[data-direction]")];
const rawElement = byId<HTMLDivElement>("raw");
const serialStatusElement = byId<HTMLDivElement>("serial-status");
const simStatusElement = byId<HTMLDivElement>("sim-status");
const telemetryElement = byId<HTMLDivElement>("telemetry");
const taskInput = byId<HTMLInputElement>("episode-task");
const startRecordingButton = byId<HTMLButtonElement>("record-start");
const saveSuccessButton = byId<HTMLButtonElement>("record-success");
const saveFailureButton = byId<HTMLButtonElement>("record-failure");
const discardRecordingButton = byId<HTMLButtonElement>("record-discard");
const resetEnvironmentButton = byId<HTMLButtonElement>("environment-reset");
const recordingStatusElement = byId<HTMLDivElement>("recording-status");

type PreviewName = "base" | "top" | "side" | "wrist";
type StreamView = "free" | PreviewName;
type PreviewState = {
  image: HTMLImageElement;
  overlay: HTMLSpanElement;
  status: HTMLSpanElement;
  frame: HTMLSpanElement;
  age: HTMLSpanElement;
  lastReceivedMs: number | null;
  objectUrl: string | null;
};

const previewNames: PreviewName[] = ["base", "top", "side", "wrist"];
const previewByCameraId: Record<number, PreviewName> = {
  0: "base",
  1: "top",
  2: "side",
  3: "wrist",
};
const previewCards = [...document.querySelectorAll<HTMLElement>("[data-preview-camera]")];
const previewStates = Object.fromEntries(previewNames.map((name) => [name, {
  image: byId<HTMLImageElement>(`${name}-preview-image`),
  overlay: byId<HTMLSpanElement>(`${name}-preview-overlay`),
  status: byId<HTMLSpanElement>(`${name}-preview-status`),
  frame: byId<HTMLSpanElement>(`${name}-preview-frame`),
  age: byId<HTMLSpanElement>(`${name}-preview-age`),
  lastReceivedMs: null,
  objectUrl: null,
}])) as Record<PreviewName, PreviewState>;

const PREVIEW_HEADER_BYTES = 22;
const PREVIEW_STALE_AFTER_MS = 750;

function setActiveView(view: StreamView) {
  activeViewLabel.textContent = `Main: ${view[0].toUpperCase()}${view.slice(1)}`;
  for (const card of previewCards) {
    const active = card.dataset.previewCamera === view;
    card.classList.toggle("sensor-preview-active", active);
    card.setAttribute("aria-pressed", String(active));
  }
}

function setPreviewState(name: PreviewName, state: "live" | "stale" | "waiting", detail?: string) {
  const preview = previewStates[name];
  preview.status.textContent = state.toUpperCase();
  preview.status.className = `preview-status preview-${state}`;
  if (state === "waiting") {
    preview.overlay.textContent = detail ?? "Waiting for camera frame…";
    preview.overlay.hidden = false;
  } else {
    preview.overlay.hidden = true;
  }
}

async function updatePreview(payload: ArrayBuffer | Blob) {
  const buffer = payload instanceof Blob ? await payload.arrayBuffer() : payload;
  if (buffer.byteLength <= PREVIEW_HEADER_BYTES) throw new Error("Preview packet is truncated");
  const bytes = new Uint8Array(buffer);
  if (bytes[0] !== 0x52 || bytes[1] !== 0x44 || bytes[2] !== 0x53 || bytes[3] !== 0x43) {
    throw new Error("Preview packet has invalid magic");
  }
  const view = new DataView(buffer);
  const version = view.getUint8(4);
  const name = previewByCameraId[view.getUint8(5)];
  if (version !== 1 || !name) throw new Error(`Unsupported preview protocol ${version}`);
  const frameIndex = view.getBigUint64(6, false);
  // Bytes 14..21 carry the server monotonic capture timestamp for protocol provenance.
  view.getBigUint64(14, false);
  const preview = previewStates[name];
  const nextUrl = URL.createObjectURL(new Blob([buffer.slice(PREVIEW_HEADER_BYTES)], { type: "image/jpeg" }));
  const previousUrl = preview.objectUrl;
  preview.objectUrl = nextUrl;
  preview.image.onload = () => {
    if (previousUrl) URL.revokeObjectURL(previousUrl);
  };
  preview.image.src = nextUrl;
  preview.image.hidden = false;
  preview.lastReceivedMs = performance.now();
  preview.frame.textContent = `Frame ${frameIndex}`;
  preview.age.textContent = "Age 0.0 s";
  setPreviewState(name, "live");
}

function refreshPreviewFreshness() {
  const now = performance.now();
  for (const name of previewNames) {
    const preview = previewStates[name];
    if (preview.lastReceivedMs === null) continue;
    const ageMs = now - preview.lastReceivedMs;
    preview.age.textContent = `Age ${(ageMs / 1000).toFixed(1)} s`;
    setPreviewState(name, ageMs > PREVIEW_STALE_AFTER_MS ? "stale" : "live");
  }
}

function markPreviewsWaiting(detail: string) {
  for (const name of previewNames) {
    previewStates[name].lastReceivedMs = null;
    setPreviewState(name, "waiting", detail);
  }
}

let socket: WebSocket | undefined;
let rawSample: number[] | null = null;
let armZero: number[] | null = null;
let gripperOpen: number | null = null;
let gripperClosed: number | null = null;
let sequence = 0;
let sessionId = newGelloSessionId();

function newGelloSessionId() {
  return globalThis.crypto?.randomUUID?.() ?? `browser-${Date.now()}-${Math.random().toString(16).slice(2)}`;
}

function calibrationReady() {
  return armZero !== null && gripperOpen !== null && gripperClosed !== null && gripperOpen !== gripperClosed;
}

function stopControl() {
  enableControl.checked = false;
}

function updateButtons() {
  const hasSample = rawSample !== null;
  captureHomeButton.disabled = !hasSample;
  captureOpenButton.disabled = !hasSample;
  captureClosedButton.disabled = !hasSample;
  enableControl.disabled = !calibrationReady();
}

function serialStatus({ level, message, connected }: { level: string; message: string; connected?: boolean }) {
  serialStatusElement.textContent = message;
  serialStatusElement.className = `status-text ${level}`;
  if (level === "error") stopControl();
  if (connected === true) {
    connectButton.disabled = true;
    disconnectButton.disabled = false;
  } else if (connected === false) {
    connectButton.disabled = !("serial" in navigator);
    disconnectButton.disabled = true;
    stopControl();
  }
}

function handleSample(raw: number[]) {
  rawSample = [...raw];
  rawElement.textContent = `Raw ID 1–8: ${raw.join(" | ")}`;
  updateButtons();
  if (!enableControl.checked || !calibrationReady() || socket?.readyState !== WebSocket.OPEN) return;
  try {
    const directions = directionElements.map((element) => Number(element.value));
    socket.send(JSON.stringify({
      type: "gello",
      version: 1,
      session_id: sessionId,
      sequence: sequence++,
      timestamp_ns: Date.now() * 1e6,
      joint_targets: mapJointPositions(raw, armZero, directions),
      gripper_closed: classifyGripper(raw[7], gripperOpen, gripperClosed),
    }));
  } catch (error) {
    serialStatus({ level: "error", message: `Mapping failed: ${error instanceof Error ? error.message : error}` });
  }
}

const reader = new GelloSerialReader({ onSample: handleSample, onStatus: serialStatus });

connectButton.onclick = async () => {
  stopControl();
  rawSample = null;
  armZero = null;
  gripperOpen = null;
  gripperClosed = null;
  sequence = 0;
  sessionId = newGelloSessionId();
  updateButtons();
  try {
    await reader.connect();
  } catch (error) {
    serialStatus({ level: "error", message: `Connect failed: ${error instanceof Error ? error.message : error}`, connected: false });
  }
};

disconnectButton.onclick = async () => {
  stopControl();
  await reader.disconnect();
  rawSample = null;
  updateButtons();
};

captureHomeButton.onclick = () => {
  if (!rawSample) return;
  armZero = rawSample.slice(0, 7);
  serialStatus({ level: "ok", message: "Arm home captured; verify directions before enabling control." });
  updateButtons();
};
captureOpenButton.onclick = () => {
  if (!rawSample) return;
  gripperOpen = rawSample[7];
  serialStatus({ level: "ok", message: `Gripper open captured at ${gripperOpen}.` });
  updateButtons();
};
captureClosedButton.onclick = () => {
  if (!rawSample) return;
  gripperClosed = rawSample[7];
  serialStatus({ level: "ok", message: `Gripper closed captured at ${gripperClosed}.` });
  updateButtons();
};
enableControl.onchange = () => {
  if (enableControl.checked && !calibrationReady()) {
    stopControl();
    serialStatus({ level: "error", message: "Capture arm home and both gripper endpoints first." });
  } else if (enableControl.checked) {
    serialStatus({ level: "ok", message: "GELLO control enabled." });
  } else {
    serialStatus({ level: "warning", message: "Control disabled; watchdog will hold the last target." });
  }
};

function updateTelemetry(message: any) {
  const gello = message.gello;
  const previews = message.previews ?? {};
  if (!taskInput.dataset.sceneInitialized && message.scene_task) {
    taskInput.value = message.scene_task;
    taskInput.dataset.sceneInitialized = "true";
  }
  const previewLagged = previews.encoded > 5 && previews.actual_fps < previews.target_fps * 0.8;
  simStatusElement.textContent = `Simulator: ${message.sim_ready ? "ready" : "starting"} | sequence: ${gello.sequence ?? "–"} | age: ${gello.age_ms ?? "–"} ms | watchdog: ${gello.watchdog_holding ? "HOLD" : "fresh"} | holds: ${gello.watchdog_holds}`;
  simStatusElement.className = `status-text ${gello.watchdog_holding || previewLagged ? "warning" : "ok"}`;
  const fingers = message.finger_positions?.map((value: number) => Number(value).toFixed(4)).join(", ") ?? "–";
  const surface = message.work_surface_top_m == null ? "–" : Number(message.work_surface_top_m).toFixed(3);
  const tcp = message.tcp_z_m == null ? "–" : Number(message.tcp_z_m).toFixed(3);
  const clearance = message.tcp_surface_clearance_m == null ? "–" : Number(message.tcp_surface_clearance_m).toFixed(3);
  telemetryElement.textContent = `Scene: ${message.scene_id ?? "–"} | TCP/surface/clearance: ${tcp}/${surface}/${clearance} m | tracking max error: ${message.tracking_error_max_rad ?? "–"} rad | fingers: ${fingers} m | preview: ${previews.actual_fps ?? "–"}/${previews.target_fps ?? "–"} FPS, ${previews.encode_ms ?? "–"} ms encode, ${previews.dropped ?? 0} dropped | control: ${message.control_actual_hz ?? "–"} Hz, p95 ${message.control_tick_p95_ms ?? "–"} ms${message.log ? ` | ${message.log}` : ""}`;
  const activeView = message.view?.active;
  if (["free", ...previewNames].includes(activeView)) setActiveView(activeView as StreamView);
  const recording = message.recording ?? {};
  const active = Boolean(recording.recording);
  startRecordingButton.disabled = active;
  saveSuccessButton.disabled = !active;
  saveFailureButton.disabled = !active;
  discardRecordingButton.disabled = !active;
  resetEnvironmentButton.disabled = active;
  taskInput.disabled = active;
  recordingStatusElement.textContent = active
    ? `Recording episode ${recording.episode ?? "–"}: ${recording.frames ?? 0} frames | ${recording.task ?? ""}`
    : `Recording: idle${recording.last_saved_path ? ` | saved ${recording.last_saved_path}` : ""}${recording.last_event ? ` | ${recording.last_event}` : ""}`;
  recordingStatusElement.className = `status-text ${recording.error ? "error" : active ? "warning" : "ok"}`;
}

function sendRecordingCommand(command: "start" | "success" | "failure" | "discard" | "reset") {
  if (socket?.readyState !== WebSocket.OPEN) {
    recordingStatusElement.textContent = "Simulator WebSocket is not connected.";
    recordingStatusElement.className = "status-text error";
    return;
  }
  socket.send(JSON.stringify({ type: "command", command, task: taskInput.value.trim() }));
}

function sendViewCommand(view: StreamView) {
  if (socket?.readyState !== WebSocket.OPEN) {
    simStatusElement.textContent = "Simulator WebSocket is not connected.";
    simStatusElement.className = "status-text error";
    return;
  }
  socket.send(JSON.stringify({ type: "view", view }));
}

startRecordingButton.onclick = () => sendRecordingCommand("start");
saveSuccessButton.onclick = () => sendRecordingCommand("success");
saveFailureButton.onclick = () => sendRecordingCommand("failure");
discardRecordingButton.onclick = () => sendRecordingCommand("discard");
resetEnvironmentButton.onclick = () => sendRecordingCommand("reset");

for (const card of previewCards) {
  const view = card.dataset.previewCamera as PreviewName;
  card.onclick = () => sendViewCommand(view);
  card.onkeydown = (event) => {
    if (event.key !== "Enter" && event.key !== " ") return;
    event.preventDefault();
    sendViewCommand(view);
  };
}

const shortcutViews: Record<string, StreamView> = {
  Digit0: "free",
  Digit1: "base",
  Digit2: "top",
  Digit3: "side",
  Digit4: "wrist",
};
document.addEventListener("keydown", (event) => {
  const target = event.target as HTMLElement | null;
  if (event.repeat || target?.matches("input, select, button, textarea")) return;
  const view = shortcutViews[event.code];
  if (!view) return;
  event.preventDefault();
  sendViewCommand(view);
});

function connectControlSocket() {
  const scheme = window.location.protocol === "https:" ? "wss" : "ws";
  socket = new WebSocket(`${scheme}://${window.location.host}/control`);
  socket.binaryType = "arraybuffer";
  socket.onopen = () => {
    simStatusElement.textContent = "Simulator WebSocket connected.";
    simStatusElement.className = "status-text ok";
  };
  socket.onmessage = (event) => {
    if (typeof event.data === "string") {
      updateTelemetry(JSON.parse(event.data));
      return;
    }
    void updatePreview(event.data).catch((error) => {
      console.error("Camera preview rejected", error);
    });
  };
  socket.onclose = () => {
    stopControl();
    markPreviewsWaiting("Control socket disconnected");
    simStatusElement.textContent = "Simulator WebSocket disconnected; reconnecting…";
    simStatusElement.className = "status-text error";
    window.setTimeout(connectControlSocket, 2000);
  };
}

if (!("serial" in navigator)) {
  connectButton.disabled = true;
  serialStatus({ level: "error", message: "Web Serial unavailable — use desktop Chrome/Edge at http://localhost." });
}

connectControlSocket();
setActiveView("free");
void connectStream();
window.setInterval(refreshPreviewFreshness, 500);
window.addEventListener("beforeunload", () => {
  void reader.disconnect();
  for (const preview of Object.values(previewStates)) {
    if (preview.objectUrl) URL.revokeObjectURL(preview.objectUrl);
  }
});
