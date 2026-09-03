import { bootCollector } from "/collector_common.mjs";

const ui = bootCollector({ mode: "keyboard" });
const pressed = new Set();
const motionKeys = new Set([
  "ArrowUp", "ArrowDown", "ArrowLeft", "ArrowRight",
  "KeyW", "KeyS", "KeyA", "KeyD", "KeyQ", "KeyE", "Space",
]);
const episodeKeys = new Set(["KeyR", "KeyC", "KeyX", "Backspace"]);
let sequence = 0;

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
  const handled = motionKeys.has(event.code) || episodeKeys.has(event.code);
  if (!handled) return;
  if (motionKeys.has(event.code) || event.code === "Backspace") event.preventDefault();
  if (event.repeat || pressed.has(event.code)) return;
  pressed.add(event.code);
  ui.send({ ...keyboardEvent("keydown", event.code), task: ui.task.value });
});

document.addEventListener("keyup", (event) => {
  if (motionKeys.has(event.code) || event.code === "Backspace") event.preventDefault();
  if (!pressed.delete(event.code)) return;
  ui.send(keyboardEvent("keyup", event.code));
});

window.addEventListener("blur", () => {
  for (const code of pressed) ui.send(keyboardEvent("keyup", code));
  pressed.clear();
});
