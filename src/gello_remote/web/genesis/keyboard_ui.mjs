import { bootCollector } from "/collector_common.mjs";

const ui = bootCollector({ mode: "keyboard" });
const pressed = new Set();
const motionKeys = new Set([
  "ArrowUp", "ArrowDown", "ArrowLeft", "ArrowRight",
  "KeyW", "KeyS", "KeyA", "KeyD", "KeyQ", "KeyE", "Space",
]);
const episodeKeys = new Set(["KeyR", "KeyC", "KeyX", "Backspace"]);

function editableTarget(target) {
  return ["INPUT", "SELECT", "TEXTAREA", "BUTTON"].includes(target?.tagName);
}

document.addEventListener("keydown", (event) => {
  if (editableTarget(event.target) || event.repeat || pressed.has(event.code)) return;
  if (!motionKeys.has(event.code) && !episodeKeys.has(event.code)) return;
  pressed.add(event.code);
  ui.send({ type: "keydown", code: event.code, task: ui.task.value });
  if (motionKeys.has(event.code) || event.code === "Backspace") event.preventDefault();
});

document.addEventListener("keyup", (event) => {
  if (!pressed.delete(event.code)) return;
  ui.send({ type: "keyup", code: event.code });
});

window.addEventListener("blur", () => {
  for (const code of pressed) ui.send({ type: "keyup", code });
  pressed.clear();
});
