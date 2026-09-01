export const DYNAMIXEL_BAUD_RATE = 57600;
export const DYNAMIXEL_IDS = Object.freeze([1, 2, 3, 4, 5, 6, 7, 8]);
export const PRESENT_POSITION_ADDRESS = 132;
export const PRESENT_POSITION_LENGTH = 4;
export const FRANKA_HOME = Object.freeze([0.0, -0.65, 0.0, -2.25, 0.0, 1.75, 0.78]);
export const RADIANS_PER_TICK = (2.0 * Math.PI) / 4096.0;

const HEADER = [0xff, 0xff, 0xfd, 0x00];
const STATUS_INSTRUCTION = 0x55;
const SYNC_READ_INSTRUCTION = 0x82;
const BROADCAST_ID = 0xfe;

export function dynamixelCrc(bytes) {
  let crc = 0;
  for (const byte of bytes) {
    crc ^= byte << 8;
    for (let bit = 0; bit < 8; bit += 1) {
      crc = crc & 0x8000 ? ((crc << 1) ^ 0x8005) & 0xffff : (crc << 1) & 0xffff;
    }
  }
  return crc;
}

function stuffBody(body) {
  const stuffed = [];
  for (const byte of body) {
    stuffed.push(byte);
    const size = stuffed.length;
    if (
      size >= 3 &&
      stuffed[size - 3] === 0xff &&
      stuffed[size - 2] === 0xff &&
      stuffed[size - 1] === 0xfd
    ) {
      stuffed.push(0xfd);
    }
  }
  return stuffed;
}

function unstuffBody(body) {
  const unstuffed = [];
  for (let index = 0; index < body.length; index += 1) {
    unstuffed.push(body[index]);
    const size = unstuffed.length;
    if (
      size >= 3 &&
      unstuffed[size - 3] === 0xff &&
      unstuffed[size - 2] === 0xff &&
      unstuffed[size - 1] === 0xfd &&
      body[index + 1] === 0xfd
    ) {
      index += 1;
    }
  }
  return unstuffed;
}

export function buildInstructionPacket(id, instruction, parameters = []) {
  if (!Number.isInteger(id) || id < 0 || id > 0xfe) throw new Error(`Invalid DYNAMIXEL ID: ${id}`);
  const body = stuffBody([instruction, ...parameters]);
  const length = body.length + 2;
  const packet = [...HEADER, id, length & 0xff, (length >> 8) & 0xff, ...body];
  const crc = dynamixelCrc(packet);
  return Uint8Array.from([...packet, crc & 0xff, (crc >> 8) & 0xff]);
}

export function buildSyncReadPacket(
  ids = DYNAMIXEL_IDS,
  address = PRESENT_POSITION_ADDRESS,
  dataLength = PRESENT_POSITION_LENGTH,
) {
  if (!ids.length || ids.some((id) => !Number.isInteger(id) || id < 0 || id >= BROADCAST_ID)) {
    throw new Error("Sync Read requires at least one valid device ID");
  }
  const parameters = [
    address & 0xff,
    (address >> 8) & 0xff,
    dataLength & 0xff,
    (dataLength >> 8) & 0xff,
    ...ids,
  ];
  return buildInstructionPacket(BROADCAST_ID, SYNC_READ_INSTRUCTION, parameters);
}

export class DynamixelPacketParser {
  constructor() {
    this.buffer = [];
  }

  feed(chunk) {
    this.buffer.push(...chunk);
    const packets = [];
    const errors = [];
    while (this.buffer.length >= 7) {
      let headerIndex = -1;
      for (let index = 0; index <= this.buffer.length - HEADER.length; index += 1) {
        if (HEADER.every((byte, offset) => this.buffer[index + offset] === byte)) {
          headerIndex = index;
          break;
        }
      }
      if (headerIndex < 0) {
        this.buffer.splice(0, Math.max(0, this.buffer.length - 3));
        break;
      }
      if (headerIndex > 0) this.buffer.splice(0, headerIndex);
      if (this.buffer.length < 7) break;
      const length = this.buffer[5] | (this.buffer[6] << 8);
      const packetSize = length + 7;
      if (length < 3 || packetSize > 65542) {
        errors.push(`invalid packet length ${length}`);
        this.buffer.shift();
        continue;
      }
      if (this.buffer.length < packetSize) break;
      const raw = this.buffer.splice(0, packetSize);
      const expectedCrc = raw[packetSize - 2] | (raw[packetSize - 1] << 8);
      const actualCrc = dynamixelCrc(raw.slice(0, -2));
      if (actualCrc !== expectedCrc) {
        errors.push(`CRC mismatch for ID ${raw[4]}: expected ${expectedCrc}, got ${actualCrc}`);
        continue;
      }
      const body = unstuffBody(raw.slice(7, -2));
      packets.push({ id: raw[4], instruction: body[0], body: body.slice(1), raw: Uint8Array.from(raw) });
    }
    return { packets, errors };
  }
}

export function parseStatusPosition(packet) {
  if (packet.instruction !== STATUS_INSTRUCTION) {
    throw new Error(`Unexpected instruction 0x${packet.instruction.toString(16)}`);
  }
  if (packet.body.length !== PRESENT_POSITION_LENGTH + 1) {
    throw new Error(`ID ${packet.id} returned ${packet.body.length - 1} position bytes`);
  }
  const error = packet.body[0];
  if (error !== 0) {
    throw new Error(`ID ${packet.id} hardware error 0x${error.toString(16).padStart(2, "0")}`);
  }
  const bytes = Uint8Array.from(packet.body.slice(1));
  const position = new DataView(bytes.buffer).getInt32(0, true);
  return { id: packet.id, position };
}

export function mapJointPositions(rawPositions, zeroPositions, directions) {
  if (rawPositions.length < 7 || zeroPositions.length !== 7 || directions.length !== 7) {
    throw new Error("Joint mapping requires seven raw, zero, and direction values");
  }
  return FRANKA_HOME.map((home, index) => {
    const direction = Number(directions[index]);
    if (direction !== -1 && direction !== 1) throw new Error(`Joint ${index + 1} direction must be -1 or 1`);
    const raw = Number(rawPositions[index]);
    const zero = Number(zeroPositions[index]);
    if (!Number.isFinite(raw) || !Number.isFinite(zero)) {
      throw new Error(`Joint ${index + 1} calibration is not finite`);
    }
    return home + direction * (raw - zero) * RADIANS_PER_TICK;
  });
}

export function classifyGripper(rawPosition, openPosition, closedPosition) {
  for (const [name, value] of Object.entries({ rawPosition, openPosition, closedPosition })) {
    if (!Number.isFinite(Number(value))) throw new Error(`${name} is not finite`);
  }
  if (Number(openPosition) === Number(closedPosition)) {
    throw new Error("Gripper open and closed calibration positions must differ");
  }
  return Math.abs(Number(rawPosition) - Number(closedPosition)) <=
    Math.abs(Number(rawPosition) - Number(openPosition));
}

export function missingDynamixelIds(expectedIds, positions) {
  return expectedIds.filter((id) => !positions.has(id));
}

export class GelloSerialReader {
  constructor({ ids = DYNAMIXEL_IDS, pollHz = 20, onSample = () => {}, onStatus = () => {} } = {}) {
    this.ids = [...ids];
    this.pollPeriodMs = 1000 / pollHz;
    this.onSample = onSample;
    this.onStatus = onStatus;
    this.port = null;
    this.reader = null;
    this.writer = null;
    this.readTask = null;
    this.pollTimer = null;
    this.pending = new Map();
    this.awaitingResponse = false;
    this.parser = new DynamixelPacketParser();
    this.connected = false;
    this.writeInProgress = false;
  }

  async connect() {
    if (this.connected) return;
    if (!globalThis.navigator?.serial) {
      throw new Error("Web Serial is unavailable; use desktop Chrome or Edge on localhost");
    }
    this.port = await globalThis.navigator.serial.requestPort();
    await this.port.open({ baudRate: DYNAMIXEL_BAUD_RATE });
    this.reader = this.port.readable.getReader();
    this.writer = this.port.writable.getWriter();
    this.connected = true;
    this.readTask = this.readLoop();
    await this.poll();
    this.pollTimer = globalThis.setInterval(() => void this.poll(), this.pollPeriodMs);
    this.onStatus({
      level: "info",
      message: `Serial connected at ${DYNAMIXEL_BAUD_RATE} bps`,
      connected: true,
    });
  }

  async poll() {
    if (!this.connected || this.writeInProgress) return;
    if (this.awaitingResponse) {
      const missing = missingDynamixelIds(this.ids, this.pending);
      if (missing.length) this.onStatus({ level: "warning", message: `Missing IDs: ${missing.join(",")}` });
      this.pending.clear();
      this.awaitingResponse = false;
    }
    this.writeInProgress = true;
    try {
      await this.writer.write(buildSyncReadPacket(this.ids));
      this.awaitingResponse = true;
    } catch (error) {
      this.onStatus({ level: "error", message: `Serial write failed: ${error.message}` });
      void this.disconnect();
    } finally {
      this.writeInProgress = false;
    }
  }

  async readLoop() {
    try {
      while (this.connected) {
        const { value, done } = await this.reader.read();
        if (done) break;
        const result = this.parser.feed(value);
        for (const message of result.errors) this.onStatus({ level: "warning", message });
        for (const packet of result.packets) {
          if (!this.ids.includes(packet.id)) continue;
          try {
            const { id, position } = parseStatusPosition(packet);
            this.pending.set(id, position);
            if (this.ids.every((expectedId) => this.pending.has(expectedId))) {
              this.onSample(this.ids.map((expectedId) => this.pending.get(expectedId)));
              this.pending.clear();
              this.awaitingResponse = false;
            }
          } catch (error) {
            this.onStatus({ level: "warning", message: error.message });
          }
        }
      }
    } catch (error) {
      if (this.connected) {
        this.onStatus({ level: "error", message: `Serial read failed: ${error.message}` });
        void this.disconnect();
      }
    }
  }

  async disconnect() {
    if (!this.port && !this.connected) return;
    this.connected = false;
    if (this.pollTimer !== null) globalThis.clearInterval(this.pollTimer);
    this.pollTimer = null;
    try {
      await this.reader?.cancel();
      await this.readTask;
    } catch (_) {
      // Cancellation is expected while releasing a Web Serial reader.
    }
    this.reader?.releaseLock();
    this.writer?.releaseLock();
    this.reader = null;
    this.writer = null;
    const port = this.port;
    this.port = null;
    if (port) await port.close();
    this.pending.clear();
    this.awaitingResponse = false;
    this.onStatus({ level: "info", message: "Serial disconnected", connected: false });
  }
}
