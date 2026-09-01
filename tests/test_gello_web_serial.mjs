import assert from "node:assert/strict";
import test from "node:test";

import {
  DynamixelPacketParser,
  RADIANS_PER_TICK,
  buildInstructionPacket,
  buildSyncReadPacket,
  classifyGripper,
  dynamixelCrc,
  mapJointPositions,
  missingDynamixelIds,
  parseStatusPosition,
} from "../src/gello_remote/web/gello_web_serial.mjs";

const hex = (bytes) => Array.from(bytes, (value) => value.toString(16).padStart(2, "0")).join(" ");

test("builds the official Protocol 2.0 Sync Read example", () => {
  assert.equal(
    hex(buildSyncReadPacket([1, 2], 132, 4)),
    "ff ff fd 00 fe 09 00 82 84 00 04 00 01 02 ce fa",
  );
});

test("parses fragmented and concatenated signed status packets", () => {
  const negative = buildInstructionPacket(3, 0x55, [0x00, 0xfe, 0xff, 0xff, 0xff]);
  const positive = buildInstructionPacket(4, 0x55, [0x00, 0x1f, 0x08, 0x00, 0x00]);
  const combined = Uint8Array.from([...negative, ...positive]);
  const parser = new DynamixelPacketParser();
  assert.deepEqual(parser.feed(combined.slice(0, 6)).packets, []);
  const result = parser.feed(combined.slice(6));
  assert.deepEqual(result.errors, []);
  assert.deepEqual(result.packets.map(parseStatusPosition), [
    { id: 3, position: -2 },
    { id: 4, position: 2079 },
  ]);
});

test("round trips byte stuffing and detects CRC corruption", () => {
  const stuffed = buildInstructionPacket(1, 0x55, [0x00, 0xff, 0xff, 0xfd, 0x01]);
  assert.match(hex(stuffed), /ff ff fd fd/);
  const result = new DynamixelPacketParser().feed(stuffed);
  assert.deepEqual(result.errors, []);
  assert.deepEqual(Array.from(result.packets[0].body), [0x00, 0xff, 0xff, 0xfd, 0x01]);
  const corrupted = stuffed.slice();
  corrupted[8] ^= 0x01;
  assert.match(new DynamixelPacketParser().feed(corrupted).errors[0], /CRC mismatch/);
});

test("maps seven calibrated joints around shared Franka home", () => {
  const zero = [100, 200, 300, 400, 500, 600, 700];
  const raw = zero.map((value, index) => value + (index + 1) * 10);
  const mapped = mapJointPositions(raw, zero, [1, -1, 1, -1, 1, -1, 1]);
  assert.ok(Math.abs(mapped[0] - 10 * RADIANS_PER_TICK) < 1e-12);
  assert.ok(Math.abs(mapped[1] - (-0.65 - 20 * RADIANS_PER_TICK)) < 1e-12);
  assert.equal(mapped.length, 7);
});

test("classifies gripper by nearest calibrated endpoint", () => {
  assert.equal(classifyGripper(880, 100, 900), true);
  assert.equal(classifyGripper(130, 100, 900), false);
  assert.throws(() => classifyGripper(100, 100, 100), /must differ/);
});

test("reports missing IDs and validates official packet CRC", () => {
  assert.deepEqual(missingDynamixelIds([1, 2, 3, 4], new Map([[1, 100], [3, 300]])), [2, 4]);
  const withoutCrc = [0xff, 0xff, 0xfd, 0x00, 0xfe, 0x09, 0x00, 0x82, 0x84, 0x00, 0x04, 0x00, 0x01, 0x02];
  assert.equal(dynamixelCrc(withoutCrc), 0xface);
});
