import { bootCollector } from "/collector_common.mjs";

const activeArmChip = document.getElementById("active-arm-chip");
const topologyNotice = document.getElementById("topology-notice");
const safetyHint = document.getElementById("safety-hint");

/**
 * Arm topology is owned by the scene and arrives in server status. A dual-arm
 * scene runs a time-shared keyboard: Tab moves the keys to the other arm. This
 * is deliberately not synchronous dual-arm input.
 */
let dual = false;

const ui = bootCollector({
  mode: "keyboard",
  onStatus(message) {
    applyTopology(message);
  },
});

const pressed = new Set();
const motionKeys = new Set([
  "ArrowUp", "ArrowDown", "ArrowLeft", "ArrowRight",
  "KeyW", "KeyS", "KeyA", "KeyD", "KeyQ", "KeyE", "Space",
]);
// Tab's KeyboardEvent.code is "Tab"; the Key* prefix only covers KeyA-KeyZ.
const switchKeys = new Set(["Tab"]);
const episodeKeys = new Set(["KeyR", "KeyC", "KeyX", "Backspace"]);
// Tab would otherwise move focus off the page and Backspace would navigate back.
const preventKeys = new Set([...motionKeys, ...switchKeys, "Backspace"]);
let sequence = 0;

function applyTopology(message) {
  const isDual = Boolean(message.dual_arm);
  if (isDual !== dual) {
    dual = isDual;
    if (dual) {
      topologyNotice.textContent =
        "双臂场景：键盘分时控制（Tab 切换激活臂），不是同步双臂遥操。一次只有一条臂接收按键，另一条保持。";
      topologyNotice.className = "notice notice-warn";
      safetyHint.textContent =
        "先用小步幅确认方向。分时，非同步；这是一条冒烟通道，证据上限＝数据被接受。此页面不会接收 GELLO 指令。";
    } else {
      topologyNotice.textContent = "单臂场景：全部按键作用在同一条臂上。";
      topologyNotice.className = "notice notice-ok";
      safetyHint.textContent = "先用小步幅确认方向。此页面不会接收 GELLO 指令。";
    }
  }
  if (dual) {
    const arm = String(message.active_arm ?? "left").toUpperCase();
    activeArmChip.hidden = false;
    activeArmChip.textContent = `ACTIVE: ${arm}`;
  } else {
    activeArmChip.hidden = true;
  }
}

function keyboardEvent(type, code) {
  return {
    type,
    code,
    sequence: sequence++,
    timestamp_ns: Date.now() * 1_000_000,
  };
}

function editableTarget(target) {
  return ["INPUT", "SELECT", "TEXTAREA", "BUTTON"].includes(target?.tagName);
}

document.addEventListener("keydown", (event) => {
  if (editableTarget(event.target)) return;
  const handled = motionKeys.has(event.code)
    || switchKeys.has(event.code)
    || episodeKeys.has(event.code);
  if (!handled) return;
  if (preventKeys.has(event.code)) event.preventDefault();
  if (event.repeat || pressed.has(event.code)) return;
  pressed.add(event.code);
  ui.send({ ...keyboardEvent("keydown", event.code), task: ui.task.value });
});

document.addEventListener("keyup", (event) => {
  if (preventKeys.has(event.code)) event.preventDefault();
  if (!pressed.delete(event.code)) return;
  ui.send(keyboardEvent("keyup", event.code));
});

window.addEventListener("blur", () => {
  for (const code of pressed) ui.send(keyboardEvent("keyup", code));
  pressed.clear();
});
