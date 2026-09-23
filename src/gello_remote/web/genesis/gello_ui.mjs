import { bootCollector } from "/collector_common.mjs";
import { DualGelloBridge, classifyGripper, mapJointPositions } from "/gello_web_serial.mjs";

const maxWebSocketBacklog = 64 * 1024;
const ARM_SLOT_KEYS = ["left", "right"];

const topologyNotice = document.getElementById("topology-notice");
const startEpisode = document.getElementById("start");
const taskInput = document.getElementById("task");

/**
 * Arm topology is owned by the scene and arrives in the first server status.
 * Until then nothing is sent: a single-arm scene uses one panel and an unsuffixed
 * session_id, a dual-arm scene uses both panels and `:left` / `:right` suffixes.
 */
let topology = null;

const ui = bootCollector({
  mode: "gello",
  onStatus(message) {
    applyTopology(message);
    for (const slot of activeSlots()) {
      const gello = topology.dual ? (message.gello?.arms?.[slot.key] || {}) : (message.gello || {});
      slot.diagnostics.textContent = [
        `session: ${gello.session_id ?? "-"}`,
        `sequence: ${gello.sequence ?? "-"}`,
        `receive age: ${gello.age_ms ?? "-"} ms`,
        `source age: ${gello.source_age_ms ?? "-"} ms`,
        `watchdog holds: ${gello.watchdog_holds ?? 0}`,
        `force holds: ${gello.force_holds ?? 0}`,
        `browser drops: ${slot.droppedCommands}`,
      ].join(" · ");
      updateSetupUi(slot);
    }
    updateStartGate();
  },
});

function newSessionId() {
  return globalThis.crypto?.randomUUID?.() ?? `browser-${Date.now()}-${Math.random().toString(16).slice(2)}`;
}

function createArmSlot(key, root) {
  const q = (role) => root.querySelector(`[data-role="${role}"]`);
  return {
    key,
    root,
    title: q("title"),
    badge: q("badge"),
    steps: {
      serial: q("step-serial"),
      ids: q("step-ids"),
      home: q("step-home"),
      gripper: q("step-gripper"),
      directions: q("step-directions"),
      control: q("step-control"),
    },
    connect: q("connect"),
    disconnect: q("disconnect"),
    captureHome: q("capture-home"),
    captureOpen: q("capture-open"),
    captureClosed: q("capture-closed"),
    reviewed: q("reviewed"),
    directionInputs: [...root.querySelectorAll("[data-role='direction']")],
    enable: q("enable"),
    disable: q("disable"),
    statusElement: q("status"),
    rawElement: q("raw"),
    diagnostics: q("diagnostics"),
    sessionSuffix: "",
    sessionId: "pending",
    currentRaw: null,
    armZero: null,
    gripperOpen: null,
    gripperClosed: null,
    serialConnected: false,
    controlEnabled: false,
    sequence: 0,
    droppedCommands: 0,
  };
}

const slots = Object.fromEntries(
  ARM_SLOT_KEYS.map((key) => [
    key,
    createArmSlot(key, document.querySelector(`[data-arm="${key}"]`)),
  ]),
);

const bridge = new DualGelloBridge({
  onSample: (arm, raw) => handleRawSample(slots[arm], raw),
  onStatus: (arm, status) => handleSerialStatus(slots[arm], status),
});

function activeSlots() {
  return topology ? topology.slots.map((entry) => slots[entry.key]) : [];
}

function applyTopology(message) {
  const dual = Boolean(message.dual_arm);
  const signature = dual ? "dual" : "single";
  if (topology?.signature === signature) return;
  const entries = dual
    ? [
        { key: "left", suffix: ":left", label: "LEFT GELLO" },
        { key: "right", suffix: ":right", label: "RIGHT GELLO" },
      ]
    : [{ key: "left", suffix: "", label: "GELLO" }];
  for (const key of ARM_SLOT_KEYS) {
    const slot = slots[key];
    const entry = entries.find((item) => item.key === key);
    slot.root.hidden = entry == null;
    if (entry == null) continue;
    slot.sessionSuffix = entry.suffix;
    slot.title.textContent = entry.label;
  }
  topology = { signature, dual, slots: entries };
  topologyNotice.textContent = dual
    ? "双臂场景：左右两条 GELLO 流都必须在线才能开始采集（同步遥操，无单臂降级）。"
    : "单臂场景：连接一条 GELLO 即可。";
  topologyNotice.className = dual ? "notice notice-warn" : "notice notice-ok";
  for (const slot of activeSlots()) {
    clearCalibration(slot);
    updateSetupUi(slot);
  }
  updateStartGate();
}

function setSlotNotice(slot, message, level = "neutral") {
  slot.statusElement.textContent = message;
  slot.statusElement.className = `notice notice-${level}`;
}

function calibrationReady(slot) {
  return slot.armZero !== null
    && slot.gripperOpen !== null
    && slot.gripperClosed !== null
    && slot.gripperOpen !== slot.gripperClosed
    && slot.reviewed.checked;
}

function setStep(slot, name, complete, active = false) {
  slot.steps[name].classList.toggle("complete", complete);
  slot.steps[name].classList.toggle("active", active && !complete);
}

function updateSetupUi(slot) {
  const hasSample = slot.currentRaw !== null;
  const gripperReady = slot.gripperOpen !== null
    && slot.gripperClosed !== null
    && slot.gripperOpen !== slot.gripperClosed;
  setStep(slot, "serial", slot.serialConnected, !slot.serialConnected);
  setStep(slot, "ids", hasSample, slot.serialConnected && !hasSample);
  setStep(slot, "home", slot.armZero !== null, hasSample && slot.armZero === null);
  setStep(slot, "gripper", gripperReady, slot.armZero !== null && !gripperReady);
  setStep(slot, "directions", slot.reviewed.checked, gripperReady && !slot.reviewed.checked);
  setStep(slot, "control", slot.controlEnabled, calibrationReady(slot) && !slot.controlEnabled);

  slot.badge.textContent = slot.controlEnabled ? "LIVE" : slot.serialConnected ? "CALIBRATING" : "OFFLINE";
  slot.connect.disabled = slot.serialConnected || !navigator.serial || topology == null;
  slot.disconnect.disabled = !slot.serialConnected;
  slot.captureHome.disabled = !hasSample || slot.controlEnabled;
  slot.captureOpen.disabled = !hasSample || slot.controlEnabled;
  slot.captureClosed.disabled = !hasSample || slot.controlEnabled;
  slot.reviewed.disabled = !hasSample || slot.controlEnabled;
  for (const input of slot.directionInputs) input.disabled = slot.controlEnabled;
  slot.enable.disabled = !slot.serialConnected
    || !calibrationReady(slot)
    || slot.controlEnabled
    || !ui.isConnected();
  slot.disable.disabled = !slot.controlEnabled;
}

function updateStartGate() {
  const required = activeSlots();
  const ready = topology != null
    && required.length > 0
    && required.every((slot) => slot.controlEnabled);
  // Dual-arm episodes need every arm streaming; there is no single-arm fallback.
  startEpisode.disabled = !ready || taskInput.disabled;
}

function stopControl(slot, message = "GELLO 控制已停止；服务器将保持当前位置。", level = "warn") {
  slot.controlEnabled = false;
  setSlotNotice(slot, message, level);
  updateSetupUi(slot);
  updateStartGate();
}

function stopAllControl(message, level = "warn") {
  for (const slot of activeSlots()) stopControl(slot, message, level);
}

function clearCalibration(slot) {
  slot.currentRaw = null;
  slot.armZero = null;
  slot.gripperOpen = null;
  slot.gripperClosed = null;
  slot.reviewed.checked = false;
  slot.sequence = 0;
  // A fresh calibration is a new logical stream: the per-arm session id must
  // change so the server-side sequence gate re-opens instead of rejecting the
  // restarted counter. The `:left` / `:right` suffix is what routes the arm.
  slot.sessionId = newSessionId() + slot.sessionSuffix;
  slot.rawElement.textContent = "raw ID 1–8：等待数据";
}

function handleRawSample(slot, raw) {
  slot.currentRaw = [...raw];
  slot.rawElement.textContent = `raw ID 1–8：${raw.join(" | ")}`;
  updateSetupUi(slot);
  updateStartGate();
  if (!slot.controlEnabled) return;
  if (!ui.isConnected()) {
    stopControl(slot, "服务器连接中断，GELLO 控制已停止。", "danger");
    return;
  }
  if (ui.bufferedAmount() > maxWebSocketBacklog) {
    slot.droppedCommands += 1;
    if (slot.droppedCommands === 1 || slot.droppedCommands % 20 === 0) {
      setSlotNotice(slot, `网络发送积压，已丢弃 ${slot.droppedCommands} 条旧控制指令。`, "warn");
    }
    return;
  }
  try {
    const directions = slot.directionInputs.map((element) => Number(element.value));
    ui.send({
      type: "gello",
      version: 1,
      session_id: slot.sessionId,
      sequence: slot.sequence++,
      timestamp_ns: Date.now() * 1e6,
      joint_targets: mapJointPositions(slot.currentRaw, slot.armZero, directions),
      gripper_closed: classifyGripper(slot.currentRaw[7], slot.gripperOpen, slot.gripperClosed),
    }, maxWebSocketBacklog);
  } catch (error) {
    stopControl(slot, `映射失败：${error.message}`, "danger");
  }
}

function handleSerialStatus(slot, { level, message, connected }) {
  const uiLevel = level === "error" ? "danger" : level === "warning" ? "warn" : "ok";
  setSlotNotice(slot, message, uiLevel);
  if (connected === true) slot.serialConnected = true;
  if (connected === false) {
    slot.serialConnected = false;
    slot.controlEnabled = false;
    clearCalibration(slot);
  }
  if (level === "error") slot.controlEnabled = false;
  updateSetupUi(slot);
  updateStartGate();
}

for (const key of ARM_SLOT_KEYS) {
  const slot = slots[key];

  slot.connect.addEventListener("click", async () => {
    slot.controlEnabled = false;
    clearCalibration(slot);
    updateSetupUi(slot);
    updateStartGate();
    setSlotNotice(slot, "等待浏览器选择 GELLO 串口…", "neutral");
    try {
      // Arm identity is which panel the operator clicked, never USB order.
      await bridge.connect(key);
    } catch (error) {
      setSlotNotice(slot, `连接失败：${error.message}`, "danger");
      updateSetupUi(slot);
    }
  });

  slot.disconnect.addEventListener("click", async () => {
    stopControl(slot);
    await bridge.disconnect(key);
  });

  slot.captureHome.addEventListener("click", () => {
    slot.armZero = slot.currentRaw?.slice(0, 7) ?? null;
    setSlotNotice(slot, "Home 已记录；请小幅移动各关节并检查方向。", "ok");
    updateSetupUi(slot);
    updateStartGate();
  });

  slot.captureOpen.addEventListener("click", () => {
    slot.gripperOpen = slot.currentRaw?.[7] ?? null;
    setSlotNotice(slot, `夹爪张开端点已记录：${slot.gripperOpen}`, "ok");
    updateSetupUi(slot);
    updateStartGate();
  });

  slot.captureClosed.addEventListener("click", () => {
    slot.gripperClosed = slot.currentRaw?.[7] ?? null;
    setSlotNotice(slot, `夹爪闭合端点已记录：${slot.gripperClosed}`, "ok");
    updateSetupUi(slot);
    updateStartGate();
  });

  slot.reviewed.addEventListener("change", () => {
    updateSetupUi(slot);
    updateStartGate();
  });

  slot.enable.addEventListener("click", () => {
    if (!calibrationReady(slot) || !slot.serialConnected || !ui.isConnected()) {
      setSlotNotice(slot, "请完成串口、Home、夹爪和方向检查后再启用。", "danger");
      return;
    }
    slot.controlEnabled = true;
    slot.droppedCommands = 0;
    setSlotNotice(slot, "GELLO 控制已启用；标定参数已锁定。", "ok");
    updateSetupUi(slot);
    updateStartGate();
  });

  slot.disable.addEventListener("click", () => {
    stopControl(slot);
    updateStartGate();
  });

  clearCalibration(slot);
  updateSetupUi(slot);
}

if (!navigator.serial) {
  topologyNotice.textContent = "当前浏览器不支持 Web Serial；请在 localhost 使用桌面版 Chrome 或 Edge。";
  topologyNotice.className = "notice notice-danger";
}
updateStartGate();
window.addEventListener("beforeunload", () => void bridge.disconnectAll());
