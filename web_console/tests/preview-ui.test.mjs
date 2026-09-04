import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";

const root = new URL("../", import.meta.url);

test("three-view console exposes both recorded camera panes", async () => {
  const html = await readFile(new URL("index.html", root), "utf8");

  for (const camera of ["base", "top", "side", "wrist"]) {
    assert.match(html, new RegExp(`id="${camera}-preview-image"`));
    assert.match(html, new RegExp(`id="${camera}-preview-status"`));
    assert.match(html, new RegExp(`id="${camera}-preview-frame"`));
    assert.match(html, new RegExp(`id="${camera}-preview-age"`));
  }
});

test("preview transport stays on the binary control WebSocket", async () => {
  const source = await readFile(new URL("src/main.ts", root), "utf8");

  assert.match(source, /socket\.binaryType = "arraybuffer"/);
  assert.match(source, /PREVIEW_HEADER_BYTES = 22/);
  assert.match(source, /updatePreview\(event\.data\)/);
  assert.match(source, /type: "view", view/);
  assert.match(source, /Digit4: "wrist"/);
});

test("keyboard mode follows Isaac Lab Se3Keyboard semantics", async () => {
  const html = await readFile(new URL("index.html", root), "utf8");
  const source = await readFile(new URL("src/main.ts", root), "utf8");

  for (const binding of ["W / S", "A / D", "Q / E", "Z / X", "T / G", "C / V"]) {
    assert.match(html, new RegExp(binding.replace(" / ", " \\/ ")));
  }
  assert.match(html, /data-control-source="keyboard"/);
  assert.match(source, /type: "keyboard"/);
  assert.match(source, /pressed: \[\.\.\.pressedKeyboardKeys\]\.sort\(\)/);
  assert.match(source, /window\.addEventListener\("keydown"/);
  assert.match(source, /event\.stopPropagation\(\)/);
  assert.match(source, /\{ capture: true \}/);
  assert.match(source, /window\.addEventListener\("blur"/);
  assert.match(source, /message\.control \?\? message\.gello/);
});
