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
  DualGelloBridge,
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
const controlSections = [...document.querySelectorAll<HTMLElement>("[data-control-source]")];
let controlSource: "keyboard" | "gello" = "gello";
let controlHz = 20;
let dualArm = false;
let keyboardSynchronized = false;
let activeArm: "left" | "right" = "left";
const armGrippers = { left: false, right: false };
let keyboardHeartbeat: number | undefined;

function configureControlSource(source: unknown, hz: unknown) {
  if (source !== "keyboard" && source !== "gello") throw new Error(`Unsupported control source: ${source}`);
  controlSource = source;
  const parsedHz = Number(hz);
  if (Number.isFinite(parsedHz) && parsedHz > 0) controlHz = parsedHz;
  for (const section of controlSections) section.hidden = section.dataset.controlSource !== controlSource;
  if (keyboardHeartbeat !== undefined) window.clearInterval(keyboardHeartbeat);
  keyboardHeartbeat = controlSource === "keyboard"
    ? window.setInterval(() => sendKeyboardState(), Math.max(20, Math.round(1000 / controlHz)))
    : undefined;
}

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
    const runtimeConfig = await response.json();
    dualArm = runtimeConfig.dual_arm === true;
    byId("arm-switch-help").hidden = !dualArm;
    if (dualArm) {
      const labels = { side: "Left oblique", wrist: "Right oblique", ...runtimeConfig.camera_labels };
      for (const [name, label] of Object.entries(labels) as [string, string][]) {
        const card = document.querySelector<HTMLElement>(`[data-preview-camera="${name}"]`);
        const heading = card?.querySelector("strong");
        if (heading) heading.textContent = label;
        card?.setAttribute("aria-label", `Show ${label} camera in main view`);
        byId<HTMLImageElement>(`${name}-preview-image`).alt = `${label} camera`;
      }
    }
    configureControlSource(runtimeConfig.control_source, runtimeConfig.control_hz);
    robotHomes = runtimeConfig.robot_homes ?? {};
    byId("right-gello-panel").hidden = !dualArm || controlSource !== "gello";
    if (dualArm) connectButton.textContent = "Connect LEFT GELLO";
    if (runtimeConfig.local_display) {
      streamEndpoint.textContent = "Local simulator window; WebRTC disabled";
      setStreamState("connecting", "Use the local Isaac Sim window", "USB controls and recording remain available here. Preview images are auxiliary.");
      return;
    }
    const endpoint = resolveStreamConfig(runtimeConfig, window.location.search);
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
const keyboardStatusElement = byId<HTMLDivElement>("keyboard-status");

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
let robotHomes: Record<string, number[]> = {};
let mappedRobotHome: number[] | undefined;
let armZero: number[] | null = null;
let gripperOpen: number | null = null;
let gripperClosed: number | null = null;
let sequence = 0;
let sessionId = newGelloSessionId();
let keyboardSequence = 0;
let keyboardSessionId = newGelloSessionId();
let keyboardGripperClosed = false;
const pressedKeyboardKeys = new Set<string>();
const keyboardMotionKeys = new Set([
  "KeyW", "KeyS", "KeyA", "KeyD", "KeyQ", "KeyE",
  "KeyZ", "KeyX", "KeyT", "KeyG", "KeyC", "KeyV",
]);

function newGelloSessionId() {
  return globalThis.crypto?.randomUUID?.() ?? `browser-${Date.now()}-${Math.random().toString(16).slice(2)}`;
}

function calibrationReady() {
  return mappedRobotHome?.length === 7 && armZero !== null && gripperOpen !== null && gripperClosed !== null && gripperOpen !== gripperClosed;
}

function stopControl() {
  enableControl.checked = false;
}

function sendKeyboardState() {
  if (controlSource !== "keyboard" || socket?.readyState !== WebSocket.OPEN) return;
  if (dualArm && !keyboardSynchronized) return;
  socket.send(JSON.stringify({
    type: "keyboard",
    version: 1,
    session_id: keyboardSessionId,
    sequence: keyboardSequence++,
    timestamp_ns: Date.now() * 1e6,
    pressed: [...pressedKeyboardKeys].sort(),
    gripper_closed: keyboardGripperClosed,
    ...(dualArm ? { active_arm: activeArm } : {}),
  }));
}

function clearKeyboardMotion(message: string) {
  pressedKeyboardKeys.clear();
  keyboardStatusElement.textContent = message;
  keyboardStatusElement.className = "status-text warning";
  sendKeyboardState();
}

function editableTarget(target: EventTarget | null) {
  return target instanceof HTMLElement && Boolean(target.closest("input, select, button, textarea, [contenteditable=true]"));
}

function inControlArea(target: EventTarget | null) {
  return target instanceof HTMLElement && Boolean(target.closest("#main-div, .sensor-preview-card"));
}

mainDiv.tabIndex = 0;
mainDiv.addEventListener("pointerdown", () => mainDiv.focus());

window.addEventListener("keydown", (event) => {
  if (controlSource !== "keyboard" || editableTarget(event.target)) return;
  if (event.code === "Tab" && !inControlArea(event.target)) return;
  const reserved = keyboardMotionKeys.has(event.code) || event.code === "KeyK" || event.code === "KeyL" || (dualArm && event.code === "Tab");
  if (!reserved) return;
  // Capture before the NVIDIA WebRTC video parent forwards the same key to Kit.
  event.preventDefault();
  event.stopPropagation();
  if (event.code === "Tab" && !event.repeat) {
    armGrippers[activeArm] = keyboardGripperClosed;
    activeArm = activeArm === "left" ? "right" : "left";
    keyboardGripperClosed = armGrippers[activeArm];
    pressedKeyboardKeys.clear();
  } else if (keyboardMotionKeys.has(event.code)) {
    if (!event.repeat) pressedKeyboardKeys.add(event.code);
  } else if (event.code === "KeyK" && !event.repeat) {
    keyboardGripperClosed = !keyboardGripperClosed;
  } else if (event.code === "KeyL" && !event.repeat) {
    pressedKeyboardKeys.clear();
    keyboardGripperClosed = false;
  } else {
    return;
  }
  keyboardStatusElement.textContent = `${dualArm ? `Active: ${activeArm.toUpperCase()} | ` : ""}Keys: ${[...pressedKeyboardKeys].join(" + ") || "idle"} | gripper: ${keyboardGripperClosed ? "closed" : "open"}`;
  keyboardStatusElement.className = "status-text ok";
  sendKeyboardState();
}, { capture: true });

window.addEventListener("keyup", (event) => {
  if (controlSource !== "keyboard" || editableTarget(event.target)) return;
  if (event.code === "Tab" && !inControlArea(event.target)) return;
  const reserved = keyboardMotionKeys.has(event.code) || event.code === "KeyK" || event.code === "KeyL" || (dualArm && event.code === "Tab");
  if (!reserved) return;
  event.preventDefault();
  event.stopPropagation();
  if (!pressedKeyboardKeys.delete(event.code)) return;
  keyboardStatusElement.textContent = `${dualArm ? `Active: ${activeArm.toUpperCase()} | ` : ""}Keys: ${[...pressedKeyboardKeys].join(" + ") || "idle"} | gripper: ${keyboardGripperClosed ? "closed" : "open"}`;
  sendKeyboardState();
}, { capture: true });

window.addEventListener("blur", () => clearKeyboardMotion("Window lost focus; motion cleared."));
document.addEventListener("focusin", (event) => {
  if (editableTarget(event.target)) clearKeyboardMotion("Form focused; motion cleared.");
});

function updateButtons() {
  const hasSample = rawSample !== null;
  captureHomeButton.disabled = !hasSample;
  captureOpenButton.disabled = !hasSample;
  captureClosedButton.disabled = !hasSample;
  enableControl.disabled = !calibrationReady() || (dualArm && !rightReady());
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
    if (socket.bufferedAmount > 4096) { stopControl(); throw new Error("Control transport backed up; motion disabled"); }
    const directions = directionElements.map((element) => Number(element.value));
    socket.send(JSON.stringify({
      type: "gello",
      version: 1,
      ...(dualArm ? { arm: "left" } : {}),
      session_id: sessionId,
      sequence: sequence++,
      timestamp_ns: Date.now() * 1e6,
      joint_targets: mapJointPositions(raw, armZero, directions, mappedRobotHome),
      gripper_closed: classifyGripper(raw[7], gripperOpen, gripperClosed),
    }));
  } catch (error) {
    serialStatus({ level: "error", message: `Mapping failed: ${error instanceof Error ? error.message : error}` });
  }
}

type RightCalibration = { raw: number[] | null; zero: number[] | null; home?: number[]; open: number | null; closed: number | null };
const rightCalibration: RightCalibration = { raw: null, zero: null, open: null, closed: null };
const rightStatus = byId<HTMLDivElement>("right-status");
function rightReady() {
  return rightCalibration.zero !== null && rightCalibration.home?.length === 7 && rightCalibration.open !== null && rightCalibration.closed !== null && rightCalibration.open !== rightCalibration.closed;
}
const bridge = new DualGelloBridge({
  onSample: (arm: string, raw: number[]) => {
    if (arm === "left") { handleSample(raw); return; }
    rightCalibration.raw = [...raw]; updateButtons();
    if (!dualArm || !enableControl.checked || !calibrationReady() || !rightReady() || socket?.readyState !== WebSocket.OPEN) return;
    try {
      if (socket.bufferedAmount > 4096) throw new Error("Control transport backed up");
      socket.send(JSON.stringify({type: "gello", version: 1, arm: "right", session_id: sessionId,
        sequence: sequence++, timestamp_ns: Date.now()*1e6,
        joint_targets: mapJointPositions(raw, rightCalibration.zero, [...document.querySelectorAll<HTMLSelectElement>("[data-right-direction]")].map(x=>Number(x.value)), rightCalibration.home),
        gripper_closed: classifyGripper(raw[7], rightCalibration.open, rightCalibration.closed)}));
    } catch(error) { stopControl(); rightStatus.textContent=String(error); }
  },
  onStatus: (arm: string, status: {level: string; message: string; connected?: boolean}) => {
    if (arm === "left") serialStatus(status);
    else {rightStatus.textContent=status.message; if (status.level === "error" || status.connected === false) stopControl();}
  },
});
const reader = bridge.reader("left");
byId<HTMLButtonElement>("right-connect").onclick = async () => {
  stopControl(); rightCalibration.raw=null; rightCalibration.zero=null; rightCalibration.open=null; rightCalibration.closed=null;
  try {await bridge.connect("right");} catch(error) {rightStatus.textContent=String(error);}
};
byId<HTMLButtonElement>("right-disconnect").onclick = async () => {stopControl(); await bridge.disconnect("right"); rightCalibration.raw=null; updateButtons();};
byId<HTMLButtonElement>("right-home").onclick = () => {
  if (!rightCalibration.raw || robotHomes.right?.length !== 7) return;
  stopControl(); rightCalibration.zero=rightCalibration.raw.slice(0,7); rightCalibration.home=[...robotHomes.right];
  rightStatus.textContent="Right robot home captured; verify all joint directions."; updateButtons();
};
for (const kind of ["open", "closed"] as const) byId<HTMLButtonElement>(`right-${kind}`).onclick=()=>{
  if (!rightCalibration.raw) return; stopControl(); rightCalibration[kind]=rightCalibration.raw[7];
  rightStatus.textContent=`Right gripper ${kind} captured.`; updateButtons();
};

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
  if (robotHomes.left?.length !== 7) { serialStatus({level:"error", message:"No measured robot home available"}); return; }
  stopControl();
  mappedRobotHome = [...robotHomes.left];
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
  if (enableControl.checked && (!calibrationReady() || (dualArm && !rightReady()))) {
    stopControl();
    serialStatus({ level: "error", message: "Capture arm home and both gripper endpoints first." });
  } else if (enableControl.checked) {
    serialStatus({ level: "ok", message: "GELLO control enabled." });
  } else {
    serialStatus({ level: "warning", message: "Control disabled; watchdog will hold the last target." });
  }
};

function updateTelemetry(message: any) {
  const control = message.control ?? message.gello;
  if (message.actual_joint_positions?.length === 7) robotHomes.left=[...message.actual_joint_positions];
  if (message.right_actual_joint_positions?.length === 7) robotHomes.right=[...message.right_actual_joint_positions];
  const previews = message.previews ?? {};
  if (control?.dual_arm && control.active_arm &&
      (!keyboardSynchronized || (!pressedKeyboardKeys.size && control.sequence === keyboardSequence - 1))) {
    dualArm = true;
    activeArm = control.active_arm;
    if (control.grippers) {
      armGrippers.left = control.grippers.left;
      armGrippers.right = control.grippers.right;
      keyboardGripperClosed = armGrippers[activeArm];
    }
    keyboardSynchronized = true;
  }
  if (dualArm) byId("active-arm").textContent = activeArm.toUpperCase();
  if (!taskInput.dataset.sceneInitialized && message.scene_task) {
    taskInput.value = message.scene_task;
    taskInput.dataset.sceneInitialized = "true";
  }
  const previewLagged = previews.encoded > 5 && previews.actual_fps < previews.target_fps * 0.8;
  simStatusElement.textContent = `Simulator: ${message.sim_ready ? "ready" : "starting"} | source: ${message.control_source ?? "gello"} | sequence: ${control.sequence ?? "–"} | age: ${control.age_ms ?? "–"} ms | watchdog: ${control.watchdog_holding ? "HOLD" : "fresh"} | holds: ${control.watchdog_holds}`;
  if (dualArm && controlSource === "gello") simStatusElement.textContent = ["left", "right"].map(arm => `${arm}: age ${control[arm]?.age_ms ?? "–"}ms, receive→tick p95 ${control[arm]?.receive_to_tick_ms_p95?.toFixed(1) ?? "–"}ms`).join(" | ")+` | coupled hold: ${Boolean(control.coupled_hold)}`;
  simStatusElement.className = `status-text ${control.watchdog_holding || control.coupled_hold || previewLagged ? "warning" : "ok"}`;
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

startRecordingButton.onclick = () => {
  if (dualArm) {
    activeArm = "left";
    keyboardGripperClosed = false;
    armGrippers.left = armGrippers.right = false;
    pressedKeyboardKeys.clear();
  }
  sendRecordingCommand("start");
};
saveSuccessButton.onclick = () => sendRecordingCommand("success");
saveFailureButton.onclick = () => sendRecordingCommand("failure");
discardRecordingButton.onclick = () => sendRecordingCommand("discard");
resetEnvironmentButton.onclick = () => {
  if (controlSource === "keyboard") {
    keyboardGripperClosed = false;
    activeArm = "left";
    armGrippers.left = armGrippers.right = false;
    clearKeyboardMotion("Environment reset requested; keyboard command cleared.");
  }
  sendRecordingCommand("reset");
};

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
    keyboardSynchronized = false;
    simStatusElement.textContent = "Simulator WebSocket connected.";
    simStatusElement.className = "status-text ok";
    sendKeyboardState();
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
    pressedKeyboardKeys.clear();
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
  if (keyboardHeartbeat !== undefined) window.clearInterval(keyboardHeartbeat);
  void reader.disconnect();
  for (const preview of Object.values(previewStates)) {
    if (preview.objectUrl) URL.revokeObjectURL(preview.objectUrl);
  }
});
