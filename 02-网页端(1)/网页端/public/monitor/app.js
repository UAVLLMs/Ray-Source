"use strict";

const $ = (selector) => document.querySelector(selector);
const els = {
  liveStatus: $("#liveStatus"),
  serviceState: $("#serviceState"),
  uptime: $("#uptime"),
  totalRequests: $("#totalRequests"),
  requestsMinute: $("#requestsMinute"),
  activeRequests: $("#activeRequests"),
  chatRequests: $("#chatRequests"),
  avgLatency: $("#avgLatency"),
  p95Latency: $("#p95Latency"),
  p95LatencyCard: $("#p95LatencyCard"),
  p99LatencyCard: $("#p99LatencyCard"),
  errorRate: $("#errorRate"),
  errorRequests: $("#errorRequests"),
  solutionRate: $("#solutionRate"),
  solutionFeedback: $("#solutionFeedback"),
  modelFailureRate: $("#modelFailureRate"),
  modelFailures: $("#modelFailures"),
  noEvidenceRate: $("#noEvidenceRate"),
  noEvidenceCount: $("#noEvidenceCount"),
  transferRate: $("#transferRate"),
  transferCount: $("#transferCount"),
  memoryRss: $("#memoryRss"),
  heapUsed: $("#heapUsed"),
  webDot: $("#webDot"),
  ragDot: $("#ragDot"),
  webDetail: $("#webDetail"),
  ragDetail: $("#ragDetail"),
  activeModel: $("#activeModel"),
  listenAddress: $("#listenAddress"),
  lastUpdated: $("#lastUpdated"),
  serviceLogs: $("#serviceLogs"),
  consoleForm: $("#consoleForm"),
  consoleInput: $("#consoleInput"),
  consoleOutput: $("#consoleOutput"),
  commandButtons: $("#commandButtons"),
  refreshLogs: $("#refreshLogs"),
  productStatsBody: $("#productStatsBody"),
  alertSummary: $("#alertSummary"),
  alertChannels: $("#alertChannels"),
  alertThresholds: $("#alertThresholds"),
  recentAlerts: $("#recentAlerts"),
  traceRows: $("#traceRows"),
};

let lastSnapshot = null;

function formatDuration(seconds) {
  const total = Math.max(0, Math.floor(Number(seconds) || 0));
  const days = Math.floor(total / 86400);
  const hours = Math.floor((total % 86400) / 3600);
  const minutes = Math.floor((total % 3600) / 60);
  return [days ? `${days}天` : "", hours ? `${hours}小时` : "", `${minutes}分钟`].filter(Boolean).join(" ");
}

function setText(element, value) {
  if (element) element.textContent = value;
}

function escapeHtml(value) {
  return String(value ?? "")
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#039;");
}

function formatLatency(value) {
  const ms = Math.max(0, Math.round(Number(value) || 0));
  return ms >= 1000 ? `${(ms / 1000).toFixed(ms >= 10000 ? 1 : 2)} s` : `${ms} ms`;
}

function drawChart(canvas, series, options = {}) {
  if (!canvas) return;
  const rect = canvas.getBoundingClientRect();
  const dpr = window.devicePixelRatio || 1;
  canvas.width = Math.max(1, Math.round(rect.width * dpr));
  canvas.height = Math.max(1, Math.round(rect.height * dpr));
  const ctx = canvas.getContext("2d");
  ctx.scale(dpr, dpr);
  const width = rect.width;
  const height = rect.height;
  const pad = { top: 12, right: 8, bottom: 23, left: 32 };
  const plotW = width - pad.left - pad.right;
  const plotH = height - pad.top - pad.bottom;
  const allValues = series.flatMap((line) => line.values);
  const maxValue = Math.max(options.minMax || 1, ...allValues, 1);

  ctx.strokeStyle = "rgba(83, 128, 167, .18)";
  ctx.fillStyle = "#718ca5";
  ctx.font = "10px Segoe UI";
  ctx.lineWidth = 1;
  for (let row = 0; row <= 4; row += 1) {
    const y = pad.top + (plotH * row / 4);
    ctx.beginPath();
    ctx.moveTo(pad.left, y);
    ctx.lineTo(width - pad.right, y);
    ctx.stroke();
    const label = Math.round(maxValue * (1 - row / 4));
    ctx.fillText(String(label), 2, y + 3);
  }

  for (const line of series) {
    const values = line.values.length ? line.values : [0];
    ctx.beginPath();
    ctx.strokeStyle = line.color;
    ctx.lineWidth = 2;
    ctx.lineJoin = "round";
    values.forEach((value, index) => {
      const x = pad.left + (plotW * index / Math.max(1, values.length - 1));
      const y = pad.top + plotH - (Number(value || 0) / maxValue * plotH);
      if (!index) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    });
    ctx.stroke();
  }

  const labels = lastSnapshot?.timeline?.map((row) => row.label) || [];
  if (labels.length) {
    ctx.fillStyle = "#617f99";
    ctx.textAlign = "left";
    ctx.fillText(labels[0], pad.left, height - 5);
    ctx.textAlign = "right";
    ctx.fillText(labels[labels.length - 1], width - pad.right, height - 5);
  }
}

function renderCharts(snapshot) {
  const rows = snapshot.timeline || [];
  drawChart($("#requestChart"), [{ values: rows.map((row) => row.requests), color: "#2e8cff" }]);
  drawChart($("#latencyChart"), [
    { values: rows.map((row) => row.avg_latency_ms), color: "#26d5f4" },
    { values: rows.map((row) => row.p95_latency_ms), color: "#735dff" },
    { values: rows.map((row) => row.p99_latency_ms), color: "#ffb84d" },
  ]);
  drawChart($("#errorChart"), [{ values: rows.map((row) => row.errors), color: "#ff5d70" }]);
}

function renderProductStats(products) {
  const rows = Array.isArray(products) ? products : [];
  els.productStatsBody.innerHTML = rows.length ? rows.slice(0, 30).map((row) => `
    <tr>
      <td class="product-name">${escapeHtml(row.product)}</td>
      <td>${Number(row.consultations || 0).toLocaleString()}</td>
      <td><strong>${Number(row.solution_rate_percent || 0).toFixed(2)}%</strong><small>${Number(row.feedback_count || 0)} 条反馈</small></td>
      <td>${Number(row.transfer_rate_percent || 0).toFixed(2)}%</td>
      <td class="${Number(row.model_failure_rate_percent || 0) > 0 ? "metric-warn" : ""}">${Number(row.model_failure_rate_percent || 0).toFixed(2)}%</td>
      <td class="${Number(row.rag_no_evidence_rate_percent || 0) > 0 ? "metric-warn" : ""}">${Number(row.rag_no_evidence_rate_percent || 0).toFixed(2)}%</td>
      <td>${escapeHtml(formatLatency(row.avg_latency_ms))}</td>
    </tr>
  `).join("") : '<tr><td colspan="7" class="empty-cell">暂无客服咨询数据</td></tr>';
}

function renderAlerts(alerts = {}) {
  const labels = { email: "邮件", wecom: "企业微信", feishu: "飞书" };
  const channelRows = Object.entries(alerts.channels || {});
  const configuredCount = channelRows.filter(([, config]) => config?.configured).length;
  setText(els.alertSummary, configuredCount ? `${configuredCount}/3 通道已启用` : "通道待配置");
  els.alertSummary.classList.toggle("ready", configuredCount > 0);
  els.alertChannels.innerHTML = channelRows.map(([key, config]) => `
    <div class="channel ${config?.configured ? "configured" : ""}">
      <i></i><span>${escapeHtml(labels[key] || key)}</span>
      <strong>${config?.configured ? "已配置" : "待配置"}</strong>
    </div>
  `).join("");
  const thresholds = alerts.thresholds || {};
  els.alertThresholds.innerHTML = `
    <span>HTTP 错误率 ≥ ${Number(thresholds.error_rate_percent || 0)}%</span>
    <span>模型失败率 ≥ ${Number(thresholds.model_failure_rate_percent || 0)}%</span>
    <span>无证据率 ≥ ${Number(thresholds.rag_no_evidence_rate_percent || 0)}%</span>
    <span>P95 ≥ ${escapeHtml(formatLatency(thresholds.p95_latency_ms))}</span>
  `;
  const recent = Array.isArray(alerts.recent) ? alerts.recent : [];
  els.recentAlerts.innerHTML = recent.length ? recent.slice(0, 6).map((row) => {
    const delivered = (row.delivery || []).some((item) => item.ok);
    return `
      <div class="alert-row ${escapeHtml(row.severity || "warning")}">
        <i></i>
        <div><strong>${escapeHtml(row.alert_key)}</strong><p>${escapeHtml(row.message)}</p></div>
        <span>${delivered ? "已送达" : "未送达"} · ${new Date(Number(row.created_at)).toLocaleString()}</span>
      </div>
    `;
  }).join("") : '<div class="empty-cell">暂无告警记录</div>';
}

function renderTraces(traces) {
  const rows = Array.isArray(traces) ? traces : [];
  els.traceRows.innerHTML = rows.length ? rows.map((row) => {
    const flags = [];
    if (row.client_type === "qa") flags.push('<span class="flag qa">QA</span>');
    if (row.model_failed) flags.push('<span class="flag bad">模型失败</span>');
    if (row.rag_no_evidence) flags.push('<span class="flag warn">无证据</span>');
    if (!flags.length) flags.push('<span class="flag good">正常</span>');
    const status = Number(row.status_code || 0);
    return `
      <tr>
        <td>${new Date(Number(row.completed_at)).toLocaleString()}</td>
        <td><button class="request-id" data-trace-id="${escapeHtml(row.request_id)}" title="点击在控制台查询">${escapeHtml(row.request_id)}</button></td>
        <td>${escapeHtml(row.category)}</td>
        <td class="product-name">${escapeHtml(row.product || "—")}</td>
        <td class="${status >= 400 ? "metric-bad" : ""}">HTTP ${status}</td>
        <td>${escapeHtml(formatLatency(row.duration_ms))}</td>
        <td>${Number(row.evidence_count || 0)} / 图 ${Number(row.image_count || 0)}</td>
        <td>${flags.join(" ")}</td>
      </tr>
    `;
  }).join("") : '<tr><td colspan="8" class="empty-cell">暂无请求记录</td></tr>';
}

function renderSnapshot(snapshot) {
  lastSnapshot = snapshot;
  const metrics = snapshot.metrics || {};
  const quality = snapshot.quality || {};
  const memory = snapshot.process?.memory || {};
  const backendOk = Boolean(snapshot.backend?.reachable);
  els.liveStatus.classList.add("online");
  els.liveStatus.innerHTML = "<i></i>实时连接";
  setText(els.serviceState, backendOk ? "全部正常" : "检索异常");
  setText(els.uptime, `运行时间 ${formatDuration(snapshot.process?.uptime_seconds)}`);
  setText(els.totalRequests, metrics.total_requests || 0);
  setText(els.requestsMinute, `最近一分钟 ${metrics.last_minute_requests || 0}`);
  setText(els.activeRequests, metrics.active_requests || 0);
  setText(els.chatRequests, `客服请求 ${metrics.chat_requests || 0}`);
  setText(els.avgLatency, formatLatency(metrics.avg_latency_ms));
  setText(els.p95Latency, `P50 ${formatLatency(metrics.p50_latency_ms)} · 30 天窗口`);
  setText(els.p95LatencyCard, formatLatency(metrics.p95_latency_ms));
  setText(els.p99LatencyCard, formatLatency(metrics.p99_latency_ms));
  setText(els.errorRate, `${Number(metrics.error_rate_percent || 0).toFixed(2)}%`);
  setText(els.errorRequests, `错误 ${metrics.error_requests || 0}`);
  setText(els.solutionRate, `${Number(quality.solution_rate_percent || 0).toFixed(2)}%`);
  setText(els.solutionFeedback, `反馈样本 ${quality.resolution_feedback_count || 0}`);
  setText(els.modelFailureRate, `${Number(quality.model_failure_rate_percent || 0).toFixed(2)}%`);
  setText(els.modelFailures, `失败 ${quality.model_failures || 0}`);
  setText(els.noEvidenceRate, `${Number(quality.rag_no_evidence_rate_percent || 0).toFixed(2)}%`);
  setText(els.noEvidenceCount, `无证据 ${quality.rag_no_evidence || 0}`);
  setText(els.transferRate, `${Number(quality.transfer_rate_percent || 0).toFixed(2)}%`);
  setText(els.transferCount, `转人工 ${quality.transfers || 0}`);
  setText(els.memoryRss, `${Number(memory.rss_mb || 0).toFixed(1)} MB`);
  setText(els.heapUsed, `Heap ${Number(memory.heap_used_mb || 0).toFixed(1)} MB`);
  els.webDot.className = "ok";
  els.ragDot.className = backendOk ? "ok" : "bad";
  setText(els.webDetail, `PID ${snapshot.process?.pid || "—"}`);
  setText(els.ragDetail, backendOk ? `HTTP ${snapshot.backend.status || 200}` : "无法连接");
  setText(els.activeModel, snapshot.model?.label || snapshot.model?.model || "—");
  setText(els.listenAddress, snapshot.service?.listen || "—");
  setText(els.lastUpdated, new Date(snapshot.timestamp).toLocaleTimeString());
  setText(els.serviceLogs, (snapshot.logs || []).join("\n") || "暂无日志");
  renderProductStats(snapshot.products);
  renderAlerts(snapshot.alerts);
  renderTraces(snapshot.traces);
  renderCharts(snapshot);
}

async function fetchSnapshot() {
  try {
    const response = await fetch("/internal-monitor-api/snapshot", { cache: "no-store" });
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    renderSnapshot(await response.json());
  } catch (error) {
    els.liveStatus.classList.remove("online");
    els.liveStatus.innerHTML = `<i></i>连接失败：${error.message}`;
  }
}

async function runCommand(command) {
  const value = String(command || "").trim();
  if (!value) return;
  els.consoleOutput.textContent += `\n\n> ${value}\n执行中…`;
  els.consoleOutput.scrollTop = els.consoleOutput.scrollHeight;
  try {
    const response = await fetch("/internal-monitor-api/command", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ command: value }),
    });
    const payload = await response.json();
    els.consoleOutput.textContent = els.consoleOutput.textContent.replace(/执行中…$/, payload.output || payload.error || "无输出");
  } catch (error) {
    els.consoleOutput.textContent = els.consoleOutput.textContent.replace(/执行中…$/, `失败：${error.message}`);
  }
  els.consoleOutput.scrollTop = els.consoleOutput.scrollHeight;
}

els.consoleForm.addEventListener("submit", (event) => {
  event.preventDefault();
  runCommand(els.consoleInput.value);
  els.consoleInput.value = "";
});
els.commandButtons.addEventListener("click", (event) => {
  const command = event.target.closest("[data-command]")?.dataset.command;
  if (command) runCommand(command);
});
els.refreshLogs.addEventListener("click", () => runCommand("logs"));
els.traceRows.addEventListener("click", (event) => {
  const requestId = event.target.closest("[data-trace-id]")?.dataset.traceId;
  if (requestId) runCommand(`trace ${requestId}`);
});
window.addEventListener("resize", () => lastSnapshot && renderCharts(lastSnapshot));

fetchSnapshot();
window.setInterval(fetchSnapshot, 2000);
