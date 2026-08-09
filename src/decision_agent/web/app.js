(() => {
  "use strict";

  const SESSION_STORAGE_KEY = "decision-agent-demo-session-v1";
  const HEALTH_ENDPOINT = "/health";
  const READY_ENDPOINT = "/ready";
  const EXECUTE_ENDPOINT = "/api/v1/agent/execute";
  const STATUS_POLL_MS = 12000;

  const elements = {
    form: document.getElementById("query-form"),
    query: document.getElementById("query"),
    send: document.getElementById("send-request"),
    cancel: document.getElementById("cancel-request"),
    clearSession: document.getElementById("clear-session"),
    sessionId: document.getElementById("session-id"),
    message: document.getElementById("request-message"),
    healthBadge: document.getElementById("health-badge"),
    healthText: document.getElementById("health-text"),
    readyBadge: document.getElementById("ready-badge"),
    readyText: document.getElementById("ready-text"),
    answer: document.getElementById("answer"),
    citations: document.getElementById("citations"),
    metadata: document.getElementById("metadata"),
    resultStatus: document.getElementById("result-status"),
    traceCard: document.getElementById("trace-card"),
    traceSummary: document.getElementById("trace-summary"),
    traceStages: document.getElementById("trace-stages"),
  };

  const TRACE_ATTRIBUTE_LABELS = Object.freeze({
    route: "Route",
    skill_name: "Skill",
    tool_name: "Tool",
    provider: "Provider",
    model: "Model",
    input_tokens: "Input Tokens",
    output_tokens: "Output Tokens",
    retry_count: "Retry",
    retrieved_count: "Retrieved",
    reranked_count: "Reranked",
    selected_evidence_count: "Evidence",
    row_count: "Rows",
    denied: "Denied",
    timeout: "Timeout",
    answerable: "Answerable",
    review_outcome: "Review",
  });

  let activeController = null;
  let pollTimer = null;
  let sessionId = loadOrCreateSession();

  function randomIdentifier(prefix) {
    let value;
    if (globalThis.crypto && typeof globalThis.crypto.randomUUID === "function") {
      value = globalThis.crypto.randomUUID();
    } else if (globalThis.crypto && typeof globalThis.crypto.getRandomValues === "function") {
      const bytes = new Uint8Array(16);
      globalThis.crypto.getRandomValues(bytes);
      value = Array.from(bytes, (byte) => byte.toString(16).padStart(2, "0")).join("");
    } else {
      value = `${Date.now().toString(36)}-${Math.random().toString(36).slice(2)}`;
    }
    return `${prefix}-${value}`;
  }

  function loadOrCreateSession() {
    try {
      const stored = globalThis.localStorage.getItem(SESSION_STORAGE_KEY);
      if (stored && stored.length <= 128) {
        return stored;
      }
    } catch {
      // Storage can be unavailable in hardened/private browser modes.
    }
    return createAndStoreSession();
  }

  function createAndStoreSession() {
    const nextSession = randomIdentifier("session");
    try {
      globalThis.localStorage.setItem(SESSION_STORAGE_KEY, nextSession);
    } catch {
      // The in-memory value still provides a valid session for this page.
    }
    return nextSession;
  }

  function showSession() {
    elements.sessionId.textContent = sessionId;
    elements.sessionId.title = sessionId;
  }

  function setBadge(badge, textElement, state, label) {
    badge.classList.remove(
      "status-checking",
      "status-ready",
      "status-unavailable",
      "status-liveness",
    );
    badge.classList.add(`status-${state}`);
    textElement.textContent = label;
  }

  async function fetchJson(url, options = {}) {
    const response = await fetch(url, options);
    let payload = null;
    try {
      payload = await response.json();
    } catch {
      payload = null;
    }
    return { response, payload };
  }

  async function refreshServiceStatus() {
    const results = await Promise.allSettled([
      fetchJson(HEALTH_ENDPOINT, { cache: "no-store" }),
      fetchJson(READY_ENDPOINT, { cache: "no-store" }),
    ]);

    const health = results[0];
    if (
      health.status === "fulfilled" &&
      health.value.response.ok &&
      health.value.payload &&
      health.value.payload.status === "ok"
    ) {
      setBadge(elements.healthBadge, elements.healthText, "ready", "进程在线");
    } else {
      setBadge(elements.healthBadge, elements.healthText, "unavailable", "进程不可达");
    }

    const readiness = results[1];
    if (
      readiness.status === "fulfilled" &&
      readiness.value.response.ok &&
      readiness.value.payload &&
      readiness.value.payload.status === "ready"
    ) {
      setBadge(elements.readyBadge, elements.readyText, "ready", "Runtime 已就绪");
    } else if (
      readiness.status === "fulfilled" &&
      readiness.value.payload &&
      readiness.value.payload.status === "not_ready"
    ) {
      setBadge(elements.readyBadge, elements.readyText, "liveness", "Runtime 未就绪");
    } else {
      setBadge(elements.readyBadge, elements.readyText, "unavailable", "就绪状态未知");
    }
  }

  function startStatusPolling() {
    stopStatusPolling();
    void refreshServiceStatus();
    pollTimer = globalThis.setInterval(() => {
      void refreshServiceStatus();
    }, STATUS_POLL_MS);
  }

  function stopStatusPolling() {
    if (pollTimer !== null) {
      globalThis.clearInterval(pollTimer);
      pollTimer = null;
    }
  }

  function setRequestState(isLoading) {
    elements.send.disabled = isLoading;
    elements.cancel.disabled = !isLoading;
    elements.query.disabled = isLoading;
    for (const button of document.querySelectorAll(".example-button")) {
      button.disabled = isLoading;
    }
  }

  function showMessage(message, isError = false) {
    elements.message.textContent = message;
    elements.message.classList.toggle("message-error", isError);
  }

  function replaceCitations(citations) {
    elements.citations.replaceChildren();
    if (!Array.isArray(citations) || citations.length === 0) {
      const emptyItem = document.createElement("li");
      emptyItem.textContent = "无引用";
      elements.citations.append(emptyItem);
      return;
    }
    for (const citation of citations) {
      const item = document.createElement("li");
      item.textContent = String(citation);
      elements.citations.append(item);
    }
  }

  function addMetadataRow(label, value) {
    const row = document.createElement("div");
    const term = document.createElement("dt");
    const description = document.createElement("dd");
    term.textContent = label;
    description.textContent = value || "—";
    row.append(term, description);
    elements.metadata.append(row);
  }

  function replaceMetadata(data) {
    elements.metadata.replaceChildren();
    addMetadataRow("状态", data.status);
    addMetadataRow("Request ID", data.request_id);
    addMetadataRow("路由", data.route);
    addMetadataRow("Skill", data.skill);
    addMetadataRow("Memory 读取", data.memory_context_status);
    addMetadataRow("Memory 写入", data.memory_persistence_status);
    addMetadataRow("Memory 摘要", data.memory_summarization_status);
    if (data.error_code) {
      addMetadataRow("错误码", data.error_code);
    }
  }

  function traceStatusPresentation(status) {
    const states = {
      completed: { label: "已完成", tone: "completed" },
      failed: { label: "失败", tone: "failed" },
      unsupported: { label: "暂不支持", tone: "neutral" },
      cancelled: { label: "已取消", tone: "neutral" },
      not_requested: { label: "未请求", tone: "neutral" },
      skipped: { label: "已跳过", tone: "neutral" },
    };
    return states[status] || { label: "状态未知", tone: "neutral" };
  }

  function formatDuration(value) {
    return typeof value === "number" && Number.isFinite(value) ? `${value.toFixed(0)} ms` : "未知";
  }

  function renderTraceAttributes(attributes) {
    if (!Array.isArray(attributes)) {
      return "";
    }
    return attributes
      .filter(
        (attribute) =>
          attribute &&
          typeof attribute.key === "string" &&
          Object.hasOwn(TRACE_ATTRIBUTE_LABELS, attribute.key),
      )
      .map((attribute) => `${TRACE_ATTRIBUTE_LABELS[attribute.key]}: ${String(attribute.value)}`)
      .join(" · ");
  }

  function renderTrace(trace) {
    elements.traceStages.replaceChildren();
    elements.traceSummary.textContent = "";
    elements.traceCard.hidden = true;
    if (!trace || typeof trace !== "object" || !Array.isArray(trace.stages)) {
      return;
    }
    const summaryParts = [
      `状态：${traceStatusPresentation(trace.final_status).label}`,
      `耗时：${formatDuration(trace.duration_ms)}`,
    ];
    if (typeof trace.truncated_stage_count === "number" && trace.truncated_stage_count > 0) {
      summaryParts.push(`已省略 ${trace.truncated_stage_count} 个阶段`);
    }
    if (typeof trace.dropped_span_count === "number" && trace.dropped_span_count > 0) {
      summaryParts.push(`已丢弃 ${trace.dropped_span_count} 个阶段`);
    }
    elements.traceSummary.textContent = summaryParts.join(" · ");
    for (const stage of trace.stages) {
      if (!stage || typeof stage !== "object") {
        continue;
      }
      const item = document.createElement("li");
      const heading = document.createElement("div");
      const name = document.createElement("strong");
      const status = document.createElement("span");
      const duration = document.createElement("span");
      const detail = document.createElement("p");
      const presentation = traceStatusPresentation(stage.status);
      name.textContent = typeof stage.stage === "string" ? stage.stage : "阶段";
      status.textContent = presentation.label;
      status.className = `trace-stage-status trace-stage-status-${presentation.tone}`;
      duration.textContent = formatDuration(stage.duration_ms);
      heading.append(name, status, duration);
      detail.textContent = renderTraceAttributes(stage.attributes);
      item.append(heading);
      if (detail.textContent) {
        item.append(detail);
      }
      elements.traceStages.append(item);
    }
    elements.traceCard.hidden = false;
  }

  function responsePresentation(status) {
    if (status === "completed") {
      return { label: "已完成", tone: "success", message: "请求已完成。" };
    }
    if (status === "unsupported") {
      return {
        label: "暂不支持",
        tone: "neutral",
        message: "当前 Agent 暂不支持此类请求。",
      };
    }
    return { label: "执行失败", tone: "failed", message: "请求执行失败。" };
  }

  function renderFormalResponse(data) {
    const presentation = responsePresentation(data.status);
    elements.resultStatus.textContent = presentation.label;
    elements.resultStatus.classList.toggle(
      "result-status-success",
      presentation.tone === "success",
    );
    elements.resultStatus.classList.toggle(
      "result-status-failed",
      presentation.tone === "failed",
    );
    elements.answer.textContent =
      data.answer ||
      (data.status === "unsupported"
        ? "当前 Agent 暂不支持此类请求。"
        : "本次请求没有可展示的回答。");
    replaceCitations(data.citations);
    replaceMetadata(data);
    renderTrace(data.trace);
    return presentation;
  }

  function publicErrorMessage(statusCode, payload) {
    if (statusCode === 422) {
      return "请求未通过验证，请检查问题内容后重试。";
    }
    if (statusCode === 503) {
      return "正式 Agent Runtime 尚未就绪，请稍后重试。";
    }
    if (statusCode >= 500) {
      return "服务暂时无法完成请求，请稍后重试。";
    }
    if (payload && payload.code === "runtime_unavailable") {
      return "正式 Agent Runtime 尚未就绪，请稍后重试。";
    }
    return `请求失败（HTTP ${statusCode}）。`;
  }

  async function submitQuery(event) {
    event.preventDefault();
    if (activeController !== null) {
      return;
    }
    const query = elements.query.value.trim();
    if (!query) {
      showMessage("请输入问题后再发送。", true);
      elements.query.focus();
      return;
    }

    const requestId = randomIdentifier("request");
    const payload = { request_id: requestId, session_id: sessionId, query };
    activeController = new AbortController();
    setRequestState(true);
    showMessage("正在请求正式 Agent……");
    elements.resultStatus.textContent = "处理中";
    elements.resultStatus.classList.remove("result-status-success", "result-status-failed");

    try {
      const result = await fetchJson(EXECUTE_ENDPOINT, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
        signal: activeController.signal,
      });
      if (!result.response.ok) {
        showMessage(publicErrorMessage(result.response.status, result.payload), true);
        elements.resultStatus.textContent = "请求失败";
        elements.resultStatus.classList.add("result-status-failed");
        return;
      }
      if (!result.payload || typeof result.payload !== "object") {
        showMessage("服务返回了无法识别的响应。", true);
        elements.resultStatus.textContent = "响应异常";
        elements.resultStatus.classList.add("result-status-failed");
        return;
      }
      const presentation = renderFormalResponse(result.payload);
      showMessage(presentation.message, presentation.tone === "failed");
    } catch (error) {
      if (error instanceof DOMException && error.name === "AbortError") {
        showMessage("请求已取消。");
        elements.resultStatus.textContent = "已取消";
      } else {
        showMessage("无法连接服务，请确认进程状态后重试。", true);
        elements.resultStatus.textContent = "连接失败";
        elements.resultStatus.classList.add("result-status-failed");
      }
    } finally {
      activeController = null;
      setRequestState(false);
      void refreshServiceStatus();
    }
  }

  elements.form.addEventListener("submit", (event) => {
    void submitQuery(event);
  });

  elements.query.addEventListener("keydown", (event) => {
    if ((event.ctrlKey || event.metaKey) && event.key === "Enter") {
      event.preventDefault();
      elements.form.requestSubmit();
    }
  });

  elements.cancel.addEventListener("click", () => {
    if (activeController !== null) {
      activeController.abort();
    }
  });

  elements.clearSession.addEventListener("click", () => {
    sessionId = createAndStoreSession();
    showSession();
    showMessage("已创建新会话。");
  });

  for (const button of document.querySelectorAll(".example-button")) {
    button.addEventListener("click", () => {
      elements.query.value = button.dataset.query || "";
      elements.query.focus();
    });
  }

  document.addEventListener("visibilitychange", () => {
    if (document.hidden) {
      stopStatusPolling();
    } else {
      startStatusPolling();
    }
  });

  showSession();
  startStatusPolling();
})();
