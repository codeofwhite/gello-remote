import { bootCollector } from "/collector_common.mjs";
import { GelloSerialReader, classifyGripper, mapJointPositions } from "/gello_web_serial.mjs";

const serialConnect = document.getElementById("serial-connect");
const serialDisconnect = document.getElementById("serial-disconnect");
const captureHome = document.getElementById("capture-home");
const captureOpen = document.getElementById("capture-open");
const captureClosed = document.getElementById("capture-closed");
const enableControl = document.getElementById("enable-control");
const disableControl = document.getElementById("disable-control");
const directionsReviewed = document.getElementById("directions-reviewed");
const directionInputs = [...document.querySelectorAll("[data-joint-direction]")];
const serialStatusElement = document.getElementById("serial-status");
const rawElement = document.getElementById("gello-raw");
const diagnosticsElement = document.getElementById("gello-diagnostics");
const startEpisode = document.getElementById("start");
const taskInput = document.getElementById("task");
const steps = Object.fromEntries(
  [...document.querySelectorAll("[data-step]")].map((element) => [element.dataset.step, element]),
);
const maxWebSocketBacklog = 64 * 1024;

let currentRaw = null;
let armZero = null;
let gripperOpen = null;
let gripperClosed = null;
let serialConnected = false;
let controlEnabled = false;
let sequence = 0;
let droppedCommands = 0;
let sessionId = "pending";

const ui = bootCollector({
  mode: "gello",
  onStatus(message) {
    const gello = message.gello || {};
    diagnosticsElement.textContent = [
      `session: ${gello.session_id ?? "-"}`,
      `sequence: ${gello.sequence ?? "-"}`,
      `receive age: ${gello.age_ms ?? "-"} ms`,
      `source age: ${gello.source_age_ms ?? "-"} ms`,
      `watchdog holds: ${gello.watchdog_holds ?? 0}`,
      `browser drops: ${droppedCommands}`,
    ].join(" · ");
    updateSetupUi();
  },
});

function newSessionId() {
  return globalThis.crypto?.randomUUID?.() ?? `browser-${Date.now()}-${Math.random().toString(16).slice(2)}`;
}

function setSerialNotice(message, level = "neutral") {
  serialStatusElement.textContent = message;
  serialStatusElement.className = `notice notice-${level}`;
}

function calibrationReady() {
  return armZero !== null
    && gripperOpen !== null
    && gripperClosed !== null
    && gripperOpen !== gripperClosed
    && directionsReviewed.checked;
}

function setStep(name, complete, active = false) {
  steps[name].classList.toggle("complete", complete);
  steps[name].classList.toggle("active", active && !complete);
}

function updateSetupUi() {
  const hasSample = currentRaw !== null;
  const gripperReady = gripperOpen !== null && gripperClosed !== null && gripperOpen !== gripperClosed;
  setStep("serial", serialConnected, !serialConnected);
  setStep("ids", hasSample, serialConnected && !hasSample);
  setStep("home", armZero !== null, hasSample && armZero === null);
  setStep("gripper", gripperReady, armZero !== null && !gripperReady);
  setStep("directions", directionsReviewed.checked, gripperReady && !directionsReviewed.checked);
  setStep("control", controlEnabled, calibrationReady() && !controlEnabled);

  serialConnect.disabled = serialConnected || !navigator.serial;
  serialDisconnect.disabled = !serialConnected;
  captureHome.disabled = !hasSample || controlEnabled;
  captureOpen.disabled = !hasSample || controlEnabled;
  captureClosed.disabled = !hasSample || controlEnabled;
  directionsReviewed.disabled = !hasSample || controlEnabled;
  for (const input of directionInputs) input.disabled = controlEnabled;
  enableControl.disabled = !serialConnected || !calibrationReady() || controlEnabled || !ui.isConnected();
  disableControl.disabled = !controlEnabled;
  startEpisode.disabled = !controlEnabled || taskInput.disabled;
}

function stopControl(message = "GELLO 控制已停止；服务器将保持当前位置。", level = "warn") {
  controlEnabled = false;
  setSerialNotice(message, level);
  updateSetupUi();
}

function clearCalibration() {
  currentRaw = null;
  armZero = null;
  gripperOpen = null;
  gripperClosed = null;
  directionsReviewed.checked = false;
  sequence = 0;
  sessionId = newSessionId();
  rawElement.textContent = "raw ID 1–8：等待数据";
}

function handleRawSample(raw) {
  currentRaw = [...raw];
  rawElement.textContent = `raw ID 1–8：${raw.join(" | ")}`;
  updateSetupUi();
  if (!controlEnabled) return;
  if (!ui.isConnected()) {
    stopControl("服务器连接中断，GELLO 控制已停止。", "danger");
    return;
  }
  if (ui.bufferedAmount() > maxWebSocketBacklog) {
    droppedCommands += 1;
    if (droppedCommands === 1 || droppedCommands % 20 === 0) {
      setSerialNotice(`网络发送积压，已丢弃 ${droppedCommands} 条旧控制指令。`, "warn");
    }
    return;
  }
  try {
    const directions = directionInputs.map((element) => Number(element.value));
    ui.send({
      type: "gello",
      version: 1,
      session_id: sessionId,
      sequence: sequence++,
      timestamp_ns: Date.now() * 1e6,
      joint_targets: mapJointPositions(raw, armZero, directions),
      gripper_closed: classifyGripper(raw[7], gripperOpen, gripperClosed),
    }, maxWebSocketBacklog);
  } catch (error) {
    stopControl(`映射失败：${error.message}`, "danger");
  }
}

function handleSerialStatus({ level, message, connected }) {
  const uiLevel = level === "error" ? "danger" : level === "warning" ? "warn" : "ok";
  setSerialNotice(message, uiLevel);
  if (connected === true) serialConnected = true;
  if (connected === false) {
    serialConnected = false;
    controlEnabled = false;
    clearCalibration();
  }
  if (level === "error") controlEnabled = false;
  updateSetupUi();
}

const gelloSerial = new GelloSerialReader({ onSample: handleRawSample, onStatus: handleSerialStatus });

serialConnect.addEventListener("click", async () => {
  controlEnabled = false;
  clearCalibration();
  updateSetupUi();
  setSerialNotice("等待浏览器选择 GELLO 串口…", "neutral");
  try {
    await gelloSerial.connect();
  } catch (error) {
    setSerialNotice(`连接失败：${error.message}`, "danger");
    updateSetupUi();
  }
});

serialDisconnect.addEventListener("click", async () => {
  stopControl();
  await gelloSerial.disconnect();
});

captureHome.addEventListener("click", () => {
  armZero = currentRaw?.slice(0, 7) ?? null;
  setSerialNotice("Home 已记录；请小幅移动各关节并检查方向。", "ok");
  updateSetupUi();
});

captureOpen.addEventListener("click", () => {
  gripperOpen = currentRaw?.[7] ?? null;
  setSerialNotice(`夹爪张开端点已记录：${gripperOpen}`, "ok");
  updateSetupUi();
});

captureClosed.addEventListener("click", () => {
  gripperClosed = currentRaw?.[7] ?? null;
  setSerialNotice(`夹爪闭合端点已记录：${gripperClosed}`, "ok");
  updateSetupUi();
});

directionsReviewed.addEventListener("change", updateSetupUi);

enableControl.addEventListener("click", () => {
  if (!calibrationReady() || !serialConnected || !ui.isConnected()) {
    setSerialNotice("请完成串口、Home、夹爪和方向检查后再启用。", "danger");
    return;
  }
  controlEnabled = true;
  droppedCommands = 0;
  setSerialNotice("GELLO 控制已启用；标定参数已锁定。", "ok");
  updateSetupUi();
});

disableControl.addEventListener("click", () => stopControl());

if (!navigator.serial) {
  setSerialNotice("当前浏览器不支持 Web Serial；请在 localhost 使用桌面版 Chrome 或 Edge。", "danger");
}
clearCalibration();
updateSetupUi();
window.addEventListener("beforeunload", () => void gelloSerial.disconnect());
