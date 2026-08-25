(() => {
  const $ = (id) => document.getElementById(id);

  const MOVE_KEYS = new Set([
    "w", "a", "s", "d", "q", "e",
    "arrowup", "arrowdown", "arrowleft", "arrowright",
  ]);

  const state = {
    vx: 0,
    vy: 0,
    yaw: 0,
    keys: new Set(),
    maxVx: 0.45,
    maxVy: 0.3,
    maxYaw: 0.6,
    dragging: false,
    /** true only while user is actively commanding */
    active: false,
    controlMode: "sport",
    ws: null,
  };

  function applyModeInfo(msg) {
    const mode = msg.control_mode || "sport";
    state.controlMode = mode;
    document.body.classList.toggle("mode-sport", mode === "sport");
    document.body.classList.toggle("mode-fpo", mode === "fpo");

    const badge = $("modeBadge");
    const label = $("modeLabel");
    const blurb = $("modeBlurb");
    const hint = $("modeHint");
    badge.textContent = msg.mode_short || (mode === "fpo" ? "FPO" : "SPORT");
    label.textContent = msg.mode_label || (mode === "fpo" ? "FPO 模型" : "内置步态 Sport");
    blurb.textContent =
      msg.mode_blurb ||
      (mode === "fpo"
        ? "自训 FPO：C++ LowState → ONNX → LowCmd。网页只发速度命令。"
        : "宇树内置步态（SportClient.Move）。不是 FPO 模型。");

    $("metaBackend").textContent = msg.backend || "—";
    $("metaModel").textContent =
      mode === "fpo"
        ? `${msg.active_model || "—"} (${msg.fpo_variant || "?"})`
        : "(sport 不用模型)";
    const pol = msg.policy || {};
    $("metaHz").textContent =
      mode === "fpo"
        ? String(pol.policy_hz != null ? pol.policy_hz : "50")
        : `${msg.control_hz != null ? msg.control_hz : "50"} (teleop)`;
    $("metaUdp").textContent = msg.udp_cmd || (msg.udp_bridge ? "on" : "off");

    if (mode === "fpo") {
      const onnxOk = pol.onnx_exists ? "ONNX ready" : "ONNX missing";
      hint.textContent = `FPO：网页 50Hz 发命令；后端 ${pol.policy_hz ?? 50}Hz ONNX 推理。${onnxOk} · scale=${pol.action_scale ?? 0.25} kp=${pol.kp ?? 25} kd=${pol.kd ?? 0.5}`;
    } else {
      hint.textContent =
        "Sport：速度直接进内置步态。切 FPO：--mode fpo（需先 export ONNX）";
    }

    document.querySelectorAll(".sport-only").forEach((el) => {
      el.disabled = mode !== "sport";
      el.title = mode === "sport" ? "" : "仅 Sport 模式可用";
    });
  }

  const pad = $("pad");
  const knob = $("knob");
  const dot = $("dot");
  const statusText = $("statusText");

  function clamp(v, lo, hi) {
    return Math.max(lo, Math.min(hi, v));
  }

  function setLimitsFromUI() {
    state.maxVx = parseFloat($("maxVx").value);
    state.maxVy = parseFloat($("maxVy").value);
    state.maxYaw = parseFloat($("maxYaw").value);
    $("maxVxLabel").textContent = state.maxVx.toFixed(2);
    $("maxVyLabel").textContent = state.maxVy.toFixed(2);
    $("maxYawLabel").textContent = state.maxYaw.toFixed(2);
  }

  ["maxVx", "maxVy", "maxYaw"].forEach((id) => {
    $(id).addEventListener("input", setLimitsFromUI);
  });
  setLimitsFromUI();

  function updateReadout() {
    $("vxVal").textContent = state.vx.toFixed(2);
    $("vyVal").textContent = state.vy.toFixed(2);
    $("yawVal").textContent = state.yaw.toFixed(2);
    document.body.classList.toggle("idle", !state.active);
  }

  function setKnobFromNorm(nx, ny) {
    const r = pad.clientWidth * 0.36;
    knob.style.transform = `translate(calc(-50% + ${nx * r}px), calc(-50% + ${-ny * r}px))`;
  }

  function hasMoveKeys() {
    for (const k of state.keys) {
      if (MOVE_KEYS.has(k)) return true;
    }
    return false;
  }

  /** Recompute velocity from keys/joystick. No input => full stop (pause). */
  function refreshCommand() {
    const keyed = hasMoveKeys();
    state.active = state.dragging || keyed;

    if (!state.active) {
      state.vx = 0;
      state.vy = 0;
      state.yaw = 0;
      setKnobFromNorm(0, 0);
      updateReadout();
      return;
    }

    let vx = 0;
    let vy = 0;
    let yaw = 0;
    if (state.keys.has("w") || state.keys.has("arrowup")) vx += state.maxVx;
    if (state.keys.has("s") || state.keys.has("arrowdown")) vx -= state.maxVx;
    if (state.keys.has("a") || state.keys.has("arrowleft")) vy += state.maxVy;
    if (state.keys.has("d") || state.keys.has("arrowright")) vy -= state.maxVy;
    if (state.keys.has("q")) yaw += state.maxYaw;
    if (state.keys.has("e")) yaw -= state.maxYaw;

    if (state.dragging) {
      // planar from pad; yaw still from Q/E if held
      state.yaw = yaw;
    } else {
      state.vx = vx;
      state.vy = vy;
      state.yaw = yaw;
      const nx = state.maxVy > 0 ? -state.vy / state.maxVy : 0;
      const ny = state.maxVx > 0 ? state.vx / state.maxVx : 0;
      setKnobFromNorm(clamp(nx, -1, 1), clamp(ny, -1, 1));
    }
    updateReadout();
  }

  function pointerToCmd(clientX, clientY) {
    const rect = pad.getBoundingClientRect();
    const cx = rect.left + rect.width / 2;
    const cy = rect.top + rect.height / 2;
    const dx = clientX - cx;
    const dy = clientY - cy;
    const maxR = rect.width * 0.36;
    const dist = Math.hypot(dx, dy);
    const scale = dist > maxR && dist > 1e-6 ? maxR / dist : 1;
    const nx = (dx * scale) / maxR;
    const ny = -(dy * scale) / maxR;
    state.vx = clamp(ny * state.maxVx, -state.maxVx, state.maxVx);
    state.vy = clamp(-nx * state.maxVy, -state.maxVy, state.maxVy);
    setKnobFromNorm(nx, ny);
    updateReadout();
  }

  pad.addEventListener("pointerdown", (e) => {
    state.dragging = true;
    state.active = true;
    pad.setPointerCapture(e.pointerId);
    pointerToCmd(e.clientX, e.clientY);
  });
  pad.addEventListener("pointermove", (e) => {
    if (!state.dragging) return;
    pointerToCmd(e.clientX, e.clientY);
  });
  function endDrag(e) {
    if (!state.dragging) return;
    state.dragging = false;
    try {
      pad.releasePointerCapture(e.pointerId);
    } catch (_) {}
    refreshCommand(); // no keys => pause/zero
  }
  pad.addEventListener("pointerup", endDrag);
  pad.addEventListener("pointercancel", endDrag);

  window.addEventListener("keydown", (e) => {
    const k = e.key.toLowerCase();
    if (MOVE_KEYS.has(k) || k === " ") e.preventDefault();
    if (k === " ") {
      sendStop();
      return;
    }
    if (!MOVE_KEYS.has(k)) return;
    state.keys.add(k);
    refreshCommand();
  });
  window.addEventListener("keyup", (e) => {
    state.keys.delete(e.key.toLowerCase());
    refreshCommand(); // release all move keys => pause
  });

  // Tab blur / hide => hard pause
  window.addEventListener("blur", sendStop);
  document.addEventListener("visibilitychange", () => {
    if (document.hidden) sendStop();
  });

  function sendStop() {
    state.vx = state.vy = state.yaw = 0;
    state.keys.clear();
    state.dragging = false;
    state.active = false;
    setKnobFromNorm(0, 0);
    updateReadout();
    if (state.ws && state.ws.readyState === WebSocket.OPEN) {
      state.ws.send(JSON.stringify({ type: "stop" }));
    }
  }

  $("btnStop").addEventListener("click", sendStop);
  $("btnStandUp").addEventListener("click", () => {
    if (state.ws && state.ws.readyState === WebSocket.OPEN) {
      state.ws.send(JSON.stringify({ type: "stand_up" }));
    }
  });
  $("btnStandDown").addEventListener("click", () => {
    if (state.ws && state.ws.readyState === WebSocket.OPEN) {
      state.ws.send(JSON.stringify({ type: "stand_down" }));
    }
  });

  function connect() {
    const proto = location.protocol === "https:" ? "wss" : "ws";
    const ws = new WebSocket(`${proto}://${location.host}/ws`);
    state.ws = ws;
    statusText.textContent = "connecting…";
    dot.className = "dot";

    ws.onopen = () => {
      statusText.textContent = "online";
      dot.className = "dot ok";
    };
    ws.onclose = () => {
      statusText.textContent = "offline · retry";
      dot.className = "dot bad";
      setTimeout(connect, 1000);
    };
    ws.onerror = () => {
      statusText.textContent = "error";
      dot.className = "dot bad";
    };
    ws.onmessage = (ev) => {
      try {
        const msg = JSON.parse(ev.data);
        if (msg.type === "hello") {
          applyModeInfo(msg);
          const tag = msg.mode_short || msg.control_mode || msg.backend;
          statusText.textContent = `online · ${tag}`;
        }
      } catch (_) {}
    };
  }

  // Stream latest command at 50Hz (align with policy_hz / training).
  setInterval(() => {
    if (!state.ws || state.ws.readyState !== WebSocket.OPEN) return;
    if (!state.active) {
      state.ws.send(JSON.stringify({ type: "cmd", vx: 0, vy: 0, yaw: 0, active: false }));
      return;
    }
    state.ws.send(
      JSON.stringify({
        type: "cmd",
        vx: state.vx,
        vy: state.vy,
        yaw: state.yaw,
        active: true,
      })
    );
  }, 20);

  connect();
  refreshCommand();
})();
