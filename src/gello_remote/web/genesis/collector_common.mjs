const byId = (id) => document.getElementById(id);

function editableTarget(target) {
  return ["INPUT", "SELECT", "TEXTAREA", "BUTTON"].includes(target?.tagName);
}

export function bootCollector({ mode, onStatus = () => {} }) {
  const canvas = byId("cam");
  const context = canvas.getContext("2d");
  const task = byId("task");
  const status = byId("status");
  const recording = byId("recording");
  const metrics = byId("metrics");
  const log = byId("log");
  const connectionChip = byId("connection-chip");
  const modeChip = byId("mode-chip");
  const fpsChip = byId("fps-chip");
  const latencyChip = byId("latency-chip");
  const sceneSelect = byId("scene-select");
  const parameterFields = byId("parameter-fields");
  const runtimeFields = byId("runtime-fields");
  const parameterStatus = byId("parameter-status");
  const parameterReset = byId("parameter-reset");
  const parameterCopy = byId("parameter-copy");
  const parameterApply = byId("parameter-apply");
  const buttons = Object.fromEntries(
    ["reset", "start", "success", "failure", "discard"].map((id) => [id, byId(id)]),
  );
  let websocket = null;
  let reconnectTimer = null;
  let focusedView = -1;
  let sceneCatalog = new Map();
  let activeServerScene = null;
  let isRecording = false;

  function setChip(element, text, level = "") {
    element.textContent = text;
    element.className = `chip${level ? ` chip-${level}` : ""}`;
  }

  function setNotice(message, level = "neutral") {
    status.textContent = message;
    status.className = `notice notice-${level}`;
  }

  function send(payload, maxBufferedAmount = Number.POSITIVE_INFINITY) {
    if (!websocket || websocket.readyState !== WebSocket.OPEN) return false;
    if (websocket.bufferedAmount > maxBufferedAmount) return false;
    websocket.send(JSON.stringify(payload));
    return true;
  }

  function command(name) {
    if (name === "discard" && !window.confirm("确定丢弃当前 Episode？此操作不会保存数据。")) return;
    if (!send({ type: "command", command: name, task: task.value })) {
      setNotice("服务器未连接，指令没有发送。", "danger");
    }
  }

  for (const name of Object.keys(buttons)) buttons[name].addEventListener("click", () => command(name));

  function selectedScene() {
    return sceneCatalog.get(sceneSelect.value);
  }

  function collectParameterValues() {
    return Object.fromEntries(
      [...parameterFields.querySelectorAll("input[data-parameter]")].map((input) => [
        input.dataset.parameter,
        Number(input.value),
      ]),
    );
  }

  function setParameterDisabled(disabled) {
    sceneSelect.disabled = disabled;
    for (const input of parameterFields.querySelectorAll("input")) input.disabled = disabled;
    parameterReset.disabled = disabled;
    parameterCopy.disabled = !sceneCatalog.size;
    parameterApply.disabled = disabled;
  }

  function renderParameters(values = null) {
    const scene = selectedScene();
    parameterFields.replaceChildren();
    if (!scene?.parameters?.length) {
      const empty = document.createElement("p");
      empty.className = "parameter-empty";
      empty.textContent = "该刚体场景没有开放构建参数；可直接应用以切换任务。";
      parameterFields.append(empty);
    } else {
      for (const parameter of scene.parameters) {
        const label = document.createElement("label");
        label.className = "parameter-field";
        label.textContent = parameter.label;
        const input = document.createElement("input");
        input.type = "number";
        input.dataset.parameter = parameter.key;
        input.min = String(parameter.minimum);
        input.max = String(parameter.maximum);
        input.step = parameter.step == null ? "any" : String(parameter.step);
        input.value = String(values?.[parameter.key] ?? parameter.default);
        input.setAttribute("aria-label", parameter.label);
        const hint = document.createElement("small");
        hint.textContent = `${parameter.unit || "无量纲"} · ${parameter.minimum}–${parameter.maximum}${parameter.scale === "log" ? " · 对数范围" : ""}`;
        label.append(input, hint);
        parameterFields.append(label);
      }
    }
    const changingScene = sceneSelect.value !== activeServerScene;
    parameterStatus.textContent = changingScene
      ? `将从 ${activeServerScene} 切换到 ${sceneSelect.value}；应用后服务器重启。`
      : "参数只在场景构建时生效；采集中不可修改。";
    setParameterDisabled(isRecording);
  }

  sceneSelect.addEventListener("change", () => renderParameters());
  parameterReset.addEventListener("click", () => renderParameters());
  parameterCopy.addEventListener("click", async () => {
    const configuration = {
      scene_id: sceneSelect.value,
      parameters: collectParameterValues(),
    };
    try {
      await navigator.clipboard.writeText(JSON.stringify(configuration, null, 2));
      parameterStatus.textContent = "配置已复制到剪贴板。";
    } catch (error) {
      parameterStatus.textContent = `复制失败：${error.message}`;
    }
  });
  parameterApply.addEventListener("click", () => {
    const parameters = collectParameterValues();
    if (Object.values(parameters).some((value) => !Number.isFinite(value))) {
      parameterStatus.textContent = "存在无效参数，请检查后重试。";
      return;
    }
    if (!window.confirm("应用配置会安全重启采集服务器，页面随后自动重连。继续吗？")) return;
    if (!send({ type: "reconfigure", scene_id: sceneSelect.value, parameters })) {
      parameterStatus.textContent = "服务器未连接，配置没有发送。";
      return;
    }
    parameterStatus.textContent = "配置已提交，等待服务器重启…";
    setParameterDisabled(true);
  });

  function updateStatus(message) {
    if (message.control_mode !== mode) {
      setNotice(`控制模式不匹配：页面=${mode}，服务器=${message.control_mode}`, "danger");
      return;
    }
    byId("scene-title").textContent = `Franka ${mode === "gello" ? "GELLO" : "键盘"}采集 · ${message.scene_id}`;
    if (!task.dataset.initialized) {
      task.value = message.default_task;
      task.dataset.initialized = "true";
    }
    setChip(connectionChip, "服务器在线", "ok");
    setChip(modeChip, mode === "gello" ? "GELLO 模式" : "键盘模式", "info");
    setChip(fpsChip, `${Number(message.stream_fps || 0).toFixed(1)} FPS`);
    if (mode === "gello") {
      const age = message.gello?.source_age_ms;
      setChip(
        latencyChip,
        age == null ? "等待 GELLO" : `延迟 ${Number(age).toFixed(1)} ms`,
        age == null ? "warn" : Number(age) <= 100 ? "ok" : "warn",
      );
    }
    setNotice(`已连接 · ${message.scene_description}`, "ok");
    isRecording = Boolean(message.recording);
    recording.textContent = message.recording
      ? `● Episode ${message.episode} · ${message.frames} 帧 · ${message.task}`
      : "空闲，可开始新 Episode";
    recording.classList.toggle("recording-active", message.recording);
    task.disabled = message.recording;
    buttons.start.disabled = message.recording;
    buttons.reset.disabled = message.recording;
    buttons.success.disabled = !message.recording;
    buttons.failure.disabled = !message.recording;
    buttons.discard.disabled = !message.recording;
    const sceneChanged = activeServerScene !== message.scene_id;
    if (!sceneCatalog.size || sceneChanged) {
      sceneCatalog = new Map((message.scene_catalog || []).map((scene) => [scene.scene_id, scene]));
      sceneSelect.replaceChildren();
      for (const scene of sceneCatalog.values()) {
        const option = document.createElement("option");
        option.value = scene.scene_id;
        option.textContent = `${scene.scene_id} — ${scene.description}`;
        sceneSelect.append(option);
      }
      activeServerScene = message.scene_id;
      sceneSelect.value = message.scene_id;
      renderParameters(message.resolved_scene_spec || {});
    } else {
      setParameterDisabled(isRecording);
    }
    runtimeFields.replaceChildren();
    const runtime = message.runtime || {};
    for (const [label, value] of [
      ["控制频率", `${runtime.control_fps ?? "—"} Hz`],
      ["仿真步长", `${runtime.sim_dt_s ?? "—"} s`],
      ["Substeps", runtime.substeps ?? "—"],
      ["物理步/控制", runtime.physics_steps_per_control ?? "—"],
      ["分辨率", (runtime.camera_resolution || []).join("×") || "—"],
    ]) {
      const item = document.createElement("span");
      item.textContent = `${label} ${value}`;
      runtimeFields.append(item);
    }
    metrics.textContent = Object.entries(message.metrics || {})
      .map(([key, value]) => `${key}: ${value}`)
      .join(" · ");
    if (message.log) log.textContent = message.log;
    onStatus(message);
  }

  function draw(image) {
    if (focusedView < 0) {
      canvas.width = image.width;
      canvas.height = image.height;
      context.drawImage(image, 0, 0);
      canvas.style.cursor = "zoom-in";
      return;
    }
    const width = image.width / 2;
    const height = image.height / 2;
    const column = focusedView % 2;
    const row = Math.floor(focusedView / 2);
    canvas.width = width;
    canvas.height = height;
    context.drawImage(image, column * width, row * height, width, height, 0, 0, width, height);
    canvas.style.cursor = "zoom-out";
  }

  function connect() {
    const protocol = location.protocol === "https:" ? "wss" : "ws";
    websocket = new WebSocket(`${protocol}://${location.host}`);
    websocket.binaryType = "arraybuffer";
    websocket.addEventListener("open", () => {
      setChip(connectionChip, "服务器在线", "ok");
      setNotice("服务器已连接，等待第一帧画面…", "ok");
    });
    websocket.addEventListener("close", () => {
      setChip(connectionChip, "服务器断开", "danger");
      setNotice("服务器连接已断开，2 秒后自动重连。", "danger");
      clearTimeout(reconnectTimer);
      reconnectTimer = setTimeout(connect, 2000);
    });
    websocket.addEventListener("message", (event) => {
      if (typeof event.data === "string") {
        try {
          updateStatus(JSON.parse(event.data));
        } catch (error) {
          setNotice(`状态消息解析失败：${error.message}`, "danger");
        }
        return;
      }
      const blob = new Blob([event.data], { type: "image/jpeg" });
      const image = new Image();
      image.onload = () => {
        draw(image);
        URL.revokeObjectURL(image.src);
      };
      image.src = URL.createObjectURL(blob);
    });
  }

  canvas.addEventListener("click", (event) => {
    if (focusedView >= 0) {
      focusedView = -1;
      return;
    }
    const rectangle = canvas.getBoundingClientRect();
    const x = (event.clientX - rectangle.left) / rectangle.width;
    const y = (event.clientY - rectangle.top) / rectangle.height;
    focusedView = (y >= 0.5 ? 2 : 0) + (x >= 0.5 ? 1 : 0);
  });

  document.addEventListener("keydown", (event) => {
    if (editableTarget(event.target)) return;
    if (event.code === "Digit0") {
      focusedView = -1;
      event.preventDefault();
    } else if (["Digit1", "Digit2", "Digit3", "Digit4"].includes(event.code)) {
      focusedView = Number(event.code.slice(-1)) - 1;
      event.preventDefault();
    }
  });

  connect();
  return {
    send,
    command,
    setNotice,
    task,
    isConnected: () => websocket?.readyState === WebSocket.OPEN,
    bufferedAmount: () => websocket?.bufferedAmount ?? 0,
  };
}
