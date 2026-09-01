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
      fps: 60,
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

let socket: WebSocket | undefined;
let rawSample: number[] | null = null;
let armZero: number[] | null = null;
let gripperOpen: number | null = null;
let gripperClosed: number | null = null;
let sequence = 0;

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
  simStatusElement.textContent = `Simulator: ${message.sim_ready ? "ready" : "starting"} | sequence: ${gello.sequence ?? "–"} | age: ${gello.age_ms ?? "–"} ms | watchdog: ${gello.watchdog_holding ? "HOLD" : "fresh"} | holds: ${gello.watchdog_holds}`;
  simStatusElement.className = `status-text ${gello.watchdog_holding ? "warning" : "ok"}`;
  const fingers = message.finger_positions?.map((value: number) => Number(value).toFixed(4)).join(", ") ?? "–";
  telemetryElement.textContent = `Tracking max error: ${message.tracking_error_max_rad ?? "–"} rad | fingers: ${fingers} m${message.log ? ` | ${message.log}` : ""}`;
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

startRecordingButton.onclick = () => sendRecordingCommand("start");
saveSuccessButton.onclick = () => sendRecordingCommand("success");
saveFailureButton.onclick = () => sendRecordingCommand("failure");
discardRecordingButton.onclick = () => sendRecordingCommand("discard");
resetEnvironmentButton.onclick = () => sendRecordingCommand("reset");

function connectControlSocket() {
  const scheme = window.location.protocol === "https:" ? "wss" : "ws";
  socket = new WebSocket(`${scheme}://${window.location.host}/control`);
  socket.onopen = () => {
    simStatusElement.textContent = "Simulator WebSocket connected.";
    simStatusElement.className = "status-text ok";
  };
  socket.onmessage = (event) => updateTelemetry(JSON.parse(event.data));
  socket.onclose = () => {
    stopControl();
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
void connectStream();
window.addEventListener("beforeunload", () => void reader.disconnect());
