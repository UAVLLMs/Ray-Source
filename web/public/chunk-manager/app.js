"use strict";

const state = {
  repository: null,
  manuals: [],
  currentManual: null,
  rowMode: "sections",
  selectedRow: null,
  preview: null,
  previewMode: "sections",
  upload: { filename: "", contentBase64: null },
  jobs: new Map(),
  polling: null,
  translationCache: new Map(),
  translationRequest: 0,
};

const $ = (selector) => document.querySelector(selector);
const $$ = (selector) => [...document.querySelectorAll(selector)];
const escapeHtml = (value) => String(value ?? "").replace(/[&<>'"]/g, (char) => ({
  "&": "&amp;", "<": "&lt;", ">": "&gt;", "'": "&#39;", '"': "&quot;",
}[char]));
const formatNumber = (value) => Number(value || 0).toLocaleString("zh-CN");
const formatDate = (value) => value ? new Date(value).toLocaleString("zh-CN", { hour12: false }) : "—";
const truncate = (value, length = 180) => {
  const text = String(value || "").replace(/\s+/g, " ").trim();
  return text.length > length ? `${text.slice(0, length)}…` : text;
};

async function api(path, options = {}) {
  const response = await fetch(path, {
    ...options,
    headers: { "Content-Type": "application/json", ...(options.headers || {}) },
  });
  const payload = await response.json().catch(() => ({}));
  if (!response.ok || Number(payload.code || 0) !== 0) {
    const message = payload.detail || payload.msg || `请求失败 (${response.status})`;
    throw new Error(typeof message === "string" ? message : JSON.stringify(message));
  }
  return payload.data;
}

function toast(message, tone = "default") {
  const node = $("#toast");
  node.textContent = message;
  node.dataset.tone = tone;
  node.classList.add("show");
  clearTimeout(toast.timer);
  toast.timer = setTimeout(() => node.classList.remove("show"), 3600);
}

function clearToast() {
  const node = $("#toast");
  clearTimeout(toast.timer);
  node.classList.remove("show");
  node.textContent = "";
  delete node.dataset.tone;
}

function setConnection(ok, text) {
  const node = $("#connection");
  node.classList.toggle("ok", ok);
  node.classList.toggle("bad", !ok);
  node.lastChild.textContent = text;
}

function optionsPayload() {
  return {
    target_chars: Number($("#target-size").value),
    min_chars: Number($("#min-size").value),
    max_chars: Number($("#max-size").value),
    overlap_chars: Number($("#overlap-size").value),
    infer_headings: $("#infer-headings").checked,
    drop_table_of_contents: $("#drop-toc").checked,
  };
}

function sourcePayload() {
  return {
    manual: $("#import-name").value.trim(),
    filename: state.upload.filename || "manual.md",
    text: state.upload.contentBase64 ? null : $("#source-text").value,
    content_base64: state.upload.contentBase64,
    options: optionsPayload(),
  };
}

async function refreshRepository() {
  try {
    const [status, manualData] = await Promise.all([
      api("/ragv6-api/chunks/status"),
      api("/ragv6-api/chunks/manuals"),
    ]);
    state.repository = status;
    state.manuals = manualData.manuals || [];
    setConnection(true, "数据库已连接");
    renderRepository();
    renderManuals();
    mergeJobs(status.active_jobs || []);
    if (!state.currentManual && state.manuals.length) {
      await loadManual(state.manuals[0].manual);
    }
  } catch (error) {
    setConnection(false, "数据库不可用");
    toast(error.message, "error");
  }
}

function renderRepository() {
  const row = state.repository || {};
  const cards = [
    [formatNumber(row.manual_count), "手册", "catalog 产品条目"],
    [formatNumber(row.section_count), "父章节", "完整证据单元"],
    [formatNumber(row.retrieval_chunk_count), "检索 Chunk", "BM25 / FAISS 定位块"],
    [formatNumber(row.pending_manuals?.length || 0), "待发布索引", "已改数据、待向量重建"],
  ];
  $("#global-metrics").innerHTML = cards.map(([value, label, note]) => `
    <article class="metric"><strong>${value}</strong><span>${label}</span><small>${note}</small></article>
  `).join("");
  const stale = row.index_stale || !row.index_ready;
  $("#hero-state").className = `hero-state ${stale ? "warning" : "ready"}`;
  $("#hero-state").innerHTML = stale
    ? `<span>索引状态</span><strong>${row.index_ready ? "待重建" : "未就绪"}</strong><small>${row.pending_manuals?.length ? `涉及：${escapeHtml(row.pending_manuals.join("、"))}` : "数据文件晚于当前向量索引"}</small>`
    : `<span>索引状态</span><strong>已发布</strong><small>最后重建：${escapeHtml(formatDate(row.last_index_build_at))}</small>`;
}

function renderManuals() {
  const query = $("#manual-search").value.trim().toLowerCase();
  const rows = state.manuals.filter((row) => {
    const search = `${row.manual} ${row.lang} ${row.index_status}`.toLowerCase();
    return search.includes(query);
  });
  $("#manual-count").textContent = state.manuals.length;
  $("#manual-list").innerHTML = rows.length ? rows.map((row) => `
    <button class="manual-item ${state.currentManual?.manual === row.manual ? "active" : ""}" data-manual="${escapeHtml(row.manual)}">
      <span class="manual-line"><strong>${escapeHtml(row.manual)}</strong><i class="${row.index_status === "published" ? "ready" : "warning"}"></i></span>
      <small>${row.section_count} 章节 · ${row.retrieval_chunk_count} Chunk · ${escapeHtml(String(row.lang || "").toUpperCase())}</small>
    </button>
  `).join("") : `<div class="empty">没有匹配的手册。</div>`;
  $$(".manual-item").forEach((node) => node.addEventListener("click", () => loadManual(node.dataset.manual)));
}

async function loadManual(manual) {
  const previousManual = state.currentManual;
  try {
    $("#manual-title").textContent = "正在载入…";
    const loadedManual = await api(`/ragv6-api/chunks/manual/${encodeURIComponent(manual)}`);
    state.currentManual = loadedManual;
    state.selectedRow = currentRows().length
      ? { mode: state.rowMode, index: 0 }
      : null;
    renderManuals();
    renderManualDetail();
    clearToast();
  } catch (error) {
    // Never leave the page stuck on "正在载入…" or mix the failed manual
    // selection with rows and metrics from the previously opened manual.
    state.currentManual = previousManual;
    renderManuals();
    if (previousManual) {
      renderManualDetail();
    } else {
      $("#manual-title").textContent = "手册读取失败";
      $("#manual-subtitle").textContent = `${manual} 暂时无法读取，请重试。`;
      const status = $("#manual-status");
      status.className = "status-pill warning";
      status.textContent = "读取失败";
    }
    toast(error.message, "error");
  }
}

function renderManualDetail() {
  const current = state.currentManual;
  if (!current) return;
  const meta = current.catalog || {};
  $("#manual-title").textContent = current.manual;
  $("#manual-subtitle").textContent = `源摘要 ${String(meta.source_sha256 || "旧数据无摘要").slice(0, 12)} · 生成于 ${formatDate(meta.generated_at)}`;
  const pending = state.manuals.find((row) => row.manual === current.manual)?.index_status !== "published";
  const status = $("#manual-status");
  status.className = `status-pill ${pending ? "warning" : "ready"}`;
  status.textContent = pending ? "待重建索引" : "已发布";
  $("#manual-metrics").innerHTML = [
    [meta.section_count, "父章节"],
    [meta.retrieval_chunk_count, "检索 Chunk"],
    [meta.total_chars, "正文字符"],
    [meta.total_pics, "绑定图片"],
  ].map(([value, label]) => `<div><strong>${formatNumber(value)}</strong><span>${label}</span></div>`).join("");
  renderChunkRows();
}

function currentRows() {
  if (!state.currentManual) return [];
  return state.rowMode === "sections"
    ? state.currentManual.sections || []
    : state.currentManual.retrieval_chunks || [];
}

function renderChunkRows() {
  const query = $("#chunk-search").value.trim().toLowerCase();
  const rows = currentRows();
  const filtered = rows.map((row, index) => ({ row, index })).filter(({ row }) => {
    const value = `${row.heading} ${row.text} ${(row.tags || []).join(" ")} ${row.chunk_id ?? ""}`.toLowerCase();
    return value.includes(query);
  });
  const tableHead = `
    <div class="chunk-table-head" aria-hidden="true">
      <span>ID</span><span>内容预览</span><span>字符</span>
    </div>`;
  $("#chunk-list").innerHTML = filtered.length ? tableHead + filtered.map(({ row, index }) => {
    const id = state.rowMode === "sections" ? `S${String(row.section_id ?? index).padStart(3, "0")}` : `C${String(row.chunk_id ?? index).padStart(4, "0")}`;
    const active = state.selectedRow?.mode === state.rowMode && state.selectedRow?.index === index;
    return `<button class="chunk-row ${active ? "active" : ""}" data-index="${index}">
      <span class="chunk-id">${id}</span>
      <span class="chunk-copy"><strong>${escapeHtml(row.heading || "未命名")}</strong><small>${escapeHtml(truncate(row.text, 130))}</small></span>
      <span class="chunk-size">${formatNumber(row.char_len || String(row.text || "").length)}</span>
    </button>`;
  }).join("") : `<div class="empty">没有匹配的数据块。</div>`;
  $$(".chunk-row").forEach((node) => node.addEventListener("click", () => {
    state.selectedRow = { mode: state.rowMode, index: Number(node.dataset.index) };
    renderChunkRows();
    renderChunkDetail();
  }));
  renderChunkDetail();
}

function renderTags(tags) {
  return (tags || []).map((tag) => `<span class="tag">${escapeHtml(tag)}</span>`).join("") || `<span class="muted">无标签</span>`;
}

function chunkTranslationKey() {
  const selected = state.selectedRow;
  if (!state.currentManual || !selected) return "";
  return `${state.currentManual.manual}\u0000${selected.mode}\u0000${selected.index}`;
}

function translationToneForRow(row) {
  const text = `${row.heading || ""}\n${row.text || ""}`;
  const cjkCount = (text.match(/[\u3400-\u9fff]/g) || []).length;
  const latinCount = (text.match(/[A-Za-z]/g) || []).length;
  return cjkCount >= 2 && cjkCount > latinCount * .12 ? "green" : "red";
}

function splitTranslationText(value) {
  const segments = [];
  const append = (candidate) => {
    let text = String(candidate || "").trim();
    while (text.length > 3600) {
      let end = text.lastIndexOf(" ", 3400);
      if (end < 1200) end = 3400;
      segments.push(text.slice(0, end).trim());
      text = text.slice(end).trim();
    }
    if (text) segments.push(text);
  };

  for (const line of String(value || "").replace(/\r\n/g, "\n").split(/\n+/)) {
    const text = line.trim();
    if (!text) continue;
    let start = 0;
    for (let index = 0; index < text.length; index += 1) {
      const char = text[index];
      const cjkBoundary = "。！？；".includes(char);
      const latinBoundary = ".!?;".includes(char)
        && (index === text.length - 1 || /\s/.test(text[index + 1]))
        && !(char === "." && index <= 2 && /^\d+\./.test(text));
      if (!cjkBoundary && !latinBoundary) continue;
      append(text.slice(start, index + 1));
      start = index + 1;
      while (/\s/.test(text[start] || "")) start += 1;
      index = start - 1;
    }
    append(text.slice(start));
  }
  return segments;
}

function appendTranslationPairs(target, originals, translations, tone) {
  target.innerHTML = "";
  originals.forEach((originalText, index) => {
    const pair = document.createElement("div");
    pair.className = "translation-pair";
    const original = document.createElement("p");
    original.className = "translation-original";
    original.textContent = originalText;
    const translated = document.createElement("p");
    translated.className = `translation-text ${tone}`;
    translated.textContent = translations[index] || "";
    pair.append(original, translated);
    target.appendChild(pair);
  });
}

function applyChunkTranslation(translation) {
  const tone = translation.tone;
  const heading = $("#chunk-heading-translation");
  heading.className = `chunk-heading-translation ${tone}`;
  heading.textContent = translation.heading || "";
  heading.hidden = !translation.heading;

  appendTranslationPairs(
    $("#chunk-text"),
    translation.bodySegments,
    translation.bodyTranslations,
    tone,
  );

  const summary = $("#chunk-summary-translation");
  if (summary) {
    summary.className = `summary-translation ${tone}`;
    summary.textContent = translation.summary || "";
    summary.hidden = !translation.summary;
  }

  const button = $("#toggle-chunk-translation");
  button.dataset.translated = "true";
  button.textContent = "显示原文";
  const status = $("#chunk-translation-status");
  status.className = `translation-status ${tone}`;
  status.textContent = tone === "red" ? "已显示红色中文译文" : "已显示绿色英文译文";
}

async function toggleChunkTranslation() {
  const button = $("#toggle-chunk-translation");
  if (!button) return;
  if (button.dataset.translated === "true") {
    renderChunkDetail();
    return;
  }

  const selected = state.selectedRow;
  const row = selected && currentRows()[selected.index];
  if (!row) return;
  const key = chunkTranslationKey();
  const cached = state.translationCache.get(key);
  if (cached) {
    applyChunkTranslation(cached);
    return;
  }

  const requestId = ++state.translationRequest;
  const status = $("#chunk-translation-status");
  button.disabled = true;
  button.textContent = "翻译中…";
  status.className = "translation-status";
  status.textContent = "正在调用翻译模型";

  const headingSource = String(row.heading || "").trim();
  const bodySegments = splitTranslationText(row.text);
  const summarySource = String(row.summary || "").trim();
  const segments = [
    ...(headingSource ? [headingSource] : []),
    ...bodySegments,
    ...(summarySource ? [summarySource] : []),
  ];
  if (!segments.length) {
    button.disabled = false;
    button.textContent = "红绿翻译";
    status.textContent = "当前数据块没有可翻译文本";
    return;
  }

  try {
    const data = await api("/ragv6-api/translate", {
      method: "POST",
      body: JSON.stringify({ segments }),
    });
    if (!Array.isArray(data.translations) || data.translations.length !== segments.length) {
      throw new Error("翻译结果与原文段落数量不一致");
    }

    let offset = 0;
    const translation = {
      tone: translationToneForRow(row),
      heading: headingSource ? String(data.translations[offset++] || "").trim() : "",
      bodySegments,
      bodyTranslations: data.translations
        .slice(offset, offset + bodySegments.length)
        .map((value) => String(value || "").trim()),
      summary: "",
    };
    offset += bodySegments.length;
    if (summarySource) translation.summary = String(data.translations[offset] || "").trim();
    state.translationCache.set(key, translation);

    if (requestId === state.translationRequest && key === chunkTranslationKey()) {
      applyChunkTranslation(translation);
    }
  } catch (error) {
    if (requestId === state.translationRequest && key === chunkTranslationKey()) {
      button.disabled = false;
      button.textContent = "重试红绿翻译";
      status.className = "translation-status error";
      status.textContent = error.message || "翻译失败";
      toast(error.message || "翻译失败", "error");
    }
  } finally {
    if (requestId === state.translationRequest && key === chunkTranslationKey()) {
      button.disabled = false;
    }
  }
}

function renderChunkDetail() {
  const selected = state.selectedRow;
  if (!selected || selected.mode !== state.rowMode) {
    state.translationRequest += 1;
    $("#chunk-detail").innerHTML = `<div class="empty">选择一个数据块查看完整结构。</div>`;
    return;
  }
  const row = currentRows()[selected.index];
  if (!row) return;
  const facts = state.rowMode === "sections"
    ? [
        ["Section ID", row.section_id],
        ["标题层级", row.heading_level],
        ["图片", (row.pics || []).join("、") || "—"],
        ["特殊证据", row.is_special ? "是" : "否"],
      ]
    : [
        ["Chunk ID", row.chunk_id],
        ["Parent", row.parent_section_id],
        ["切分类型", row.split_kind],
        ["字符区间", `${row.char_start ?? "—"}—${row.char_end ?? "—"}`],
      ];
  $("#chunk-detail").innerHTML = `
    <p class="eyebrow">${state.rowMode === "sections" ? "Parent evidence" : "Retrieval locator"}</p>
    <h3>${escapeHtml(row.heading || "未命名")}</h3>
    <p class="chunk-heading-translation" id="chunk-heading-translation" hidden></p>
    <div class="tag-row">${renderTags(row.tags)}</div>
    <dl>${facts.map(([key, value]) => `<div><dt>${escapeHtml(key)}</dt><dd>${escapeHtml(value)}</dd></div>`).join("")}</dl>
    <div class="translation-tools">
      <div class="translation-legend" aria-label="翻译颜色说明">
        <span class="red"><i></i>英文 → 中文</span>
        <span class="green"><i></i>中文 → 英文</span>
      </div>
      <button class="button translation-button" id="toggle-chunk-translation" type="button">红绿翻译</button>
      <span class="translation-status" id="chunk-translation-status"></span>
    </div>
    <div class="text-block" id="chunk-text">${escapeHtml(row.text || "")}</div>
    ${row.summary ? `<div class="summary"><strong>检索摘要</strong><p id="chunk-summary-text">${escapeHtml(row.summary)}</p><p class="summary-translation" id="chunk-summary-translation" hidden></p></div>` : ""}
  `;
  $("#toggle-chunk-translation").addEventListener("click", toggleChunkTranslation);
}

async function handleFile(file) {
  if (!file) return;
  if (file.size > 25 * 1024 * 1024) return toast("文件超过 25MB 限制", "error");
  state.upload.filename = file.name;
  $("#file-name").textContent = `${file.name} · ${(file.size / 1024).toFixed(1)} KB`;
  $("#source-status").className = "status-pill ready";
  $("#source-status").textContent = "文件已载入";
  if (!$("#import-name").value.trim()) $("#import-name").value = file.name.replace(/\.[^.]+$/, "");
  const suffix = file.name.split(".").pop().toLowerCase();
  if (["md", "markdown", "txt"].includes(suffix)) {
    state.upload.contentBase64 = null;
    $("#source-text").value = await file.text();
    return;
  }
  const bytes = new Uint8Array(await file.arrayBuffer());
  let binary = "";
  const stride = 0x8000;
  for (let index = 0; index < bytes.length; index += stride) {
    binary += String.fromCharCode(...bytes.subarray(index, index + stride));
  }
  state.upload.contentBase64 = btoa(binary);
  $("#source-text").value = "";
  $("#source-text").placeholder = `${file.name} 将由后端提取正文；生成预览后可检查结果。`;
}

async function generatePreview() {
  const payload = sourcePayload();
  if (!payload.manual) return toast("请填写手册规范名称", "error");
  if (!payload.content_base64 && !String(payload.text || "").trim()) return toast("请上传文件或粘贴手册正文", "error");
  const button = $("#preview-split");
  button.disabled = true;
  button.textContent = "正在解析标题、步骤与图片…";
  try {
    state.preview = await api("/ragv6-api/chunks/preview", {
      method: "POST",
      body: JSON.stringify(payload),
    });
    state.previewMode = "sections";
    $$("#preview-mode button").forEach((node) => node.classList.toggle("selected", node.dataset.mode === "sections"));
    renderPreview();
    $("#publish-manual").disabled = Boolean(state.preview.quality.errors?.length);
    toast("切分预览与质量检查完成", "success");
  } catch (error) {
    state.preview = null;
    toast(error.message, "error");
  } finally {
    button.disabled = false;
    button.textContent = "生成切分与质量预览";
  }
}

function renderPreview() {
  const preview = state.preview;
  if (!preview) return;
  const quality = preview.quality || {};
  const metrics = quality.metrics || {};
  $("#preview-title").textContent = `${preview.manual} · ${metrics.retrieval_chunk_count || 0} 个检索 Chunk`;
  $("#preview-description").textContent = `源文件摘要 ${String(preview.source_sha256 || "").slice(0, 16)} · 此时尚未写入正式数据库`;
  const banner = $("#quality-banner");
  const status = quality.status || "warning";
  banner.className = `quality-banner ${status === "ready" ? "ready" : status === "error" ? "error" : "warning"}`;
  const messages = [...(quality.errors || []), ...(quality.warnings || [])];
  banner.innerHTML = `<strong>${status === "ready" ? "质量检查通过" : status === "error" ? "存在阻断错误" : "可发布，但建议复核"}</strong><span>${escapeHtml(messages.join("；") || "标题、父子关系、长度和图片绑定均通过校验。")}</span>`;
  $("#preview-metrics").innerHTML = [
    [metrics.section_count, "父章节"],
    [metrics.retrieval_chunk_count, "检索 Chunk"],
    [metrics.average_chunk_chars, "平均字符"],
    [metrics.min_chunk_chars, "最短 Chunk"],
    [metrics.max_chunk_chars, "最长 Chunk"],
    [metrics.picture_count, "图片锚点"],
  ].map(([value, label]) => `<div><strong>${formatNumber(value)}</strong><span>${label}</span></div>`).join("");
  renderPreviewRows();
}

function renderPreviewRows() {
  if (!state.preview) return;
  const rows = state.previewMode === "sections" ? state.preview.sections : state.preview.retrieval_chunks;
  const query = $("#preview-search").value.trim().toLowerCase();
  const filtered = rows.filter((row) => `${row.heading} ${row.text} ${(row.tags || []).join(" ")}`.toLowerCase().includes(query));
  $("#preview-list").innerHTML = filtered.length ? filtered.map((row, index) => {
    const length = Number(row.char_len || String(row.text || "").length);
    const max = Number(state.preview.options.max_chars || 1100);
    const min = Number(state.preview.options.min_chars || 160);
    const tone = length > max ? "error" : length < min ? "warning" : "ready";
    const id = state.previewMode === "sections" ? `S${String(row.section_id ?? index).padStart(3, "0")}` : `D${String(index + 1).padStart(4, "0")}`;
    return `<article class="preview-row">
      <header><span class="chunk-id">${id}</span><strong>${escapeHtml(row.heading || "未命名")}</strong><span class="length ${tone}">${length} 字</span></header>
      <div class="tag-row">${renderTags(row.tags)}</div>
      <p>${escapeHtml(truncate(row.text, 320))}</p>
      <small>Parent ${row.parent_section_id ?? row.section_id ?? "—"} · ${escapeHtml(row.split_kind || "parent")} · 图片 ${(row.pics || []).length}</small>
    </article>`;
  }).join("") : `<div class="empty">没有匹配的预览块。</div>`;
}

function confirmAction(title, message, confirmLabel = "确认") {
  return new Promise((resolve) => {
    const modal = $("#confirm-modal");
    $("#modal-title").textContent = title;
    $("#modal-message").textContent = message;
    $("#modal-confirm").textContent = confirmLabel;
    modal.classList.add("show");
    modal.setAttribute("aria-hidden", "false");
    const cleanup = (result) => {
      modal.classList.remove("show");
      modal.setAttribute("aria-hidden", "true");
      $("#modal-confirm").onclick = null;
      $("#modal-cancel").onclick = null;
      resolve(result);
    };
    $("#modal-confirm").onclick = () => cleanup(true);
    $("#modal-cancel").onclick = () => cleanup(false);
  });
}

async function publishManual() {
  if (!state.preview) return;
  const replace = $("#replace-existing").checked;
  const rebuild = $("#rebuild-after-publish").checked;
  const confirmed = await confirmAction(
    "发布检索资产",
    `将 ${state.preview.manual} 的父章节、检索 Chunk、catalog、摘要和源文档写入正式数据库。${replace ? "同名手册将被完整替换。" : ""}${rebuild ? "随后会在后台重建向量索引。" : "索引将保持待重建状态。"}`,
    "确认发布",
  );
  if (!confirmed) return;
  const button = $("#publish-manual");
  button.disabled = true;
  button.textContent = "正在发布…";
  try {
    const result = await api("/ragv6-api/chunks/publish", {
      method: "POST",
      body: JSON.stringify({
        ...sourcePayload(),
        replace_existing: replace,
        rebuild_index: rebuild,
      }),
    });
    if (result.job_id) {
      state.jobs.set(result.job_id, { job_id: result.job_id, status: "queued", kind: "rebuild_index" });
      startJobPolling();
    }
    toast(`已发布 ${result.manual}，备份 ${result.backup_id}`, "success");
    await refreshRepository();
    switchTab("operations");
  } catch (error) {
    toast(error.message, "error");
  } finally {
    button.disabled = false;
    button.textContent = "发布到 Chunk 数据库";
  }
}

async function rebuildIndex() {
  const confirmed = await confirmAction(
    "重建检索索引",
    "将根据当前全部 retrieval_chunks 重新生成 BM25、向量索引和元数据，并在成功后热切换到新索引。",
    "开始重建",
  );
  if (!confirmed) return;
  try {
    const job = await api("/ragv6-api/chunks/rebuild", {
      method: "POST",
      body: JSON.stringify({ batch_size: 32 }),
    });
    state.jobs.set(job.job_id, job);
    startJobPolling();
    renderJobs();
    switchTab("operations");
    toast("索引任务已进入后台", "success");
  } catch (error) {
    toast(error.message, "error");
  }
}

function mergeJobs(jobs) {
  jobs.forEach((job) => state.jobs.set(job.job_id, job));
  renderJobs();
  if ([...state.jobs.values()].some((job) => ["queued", "running"].includes(job.status))) startJobPolling();
}

async function pollJobs() {
  const active = [...state.jobs.values()].filter((job) => ["queued", "running"].includes(job.status));
  await Promise.all(active.map(async (job) => {
    try {
      const fresh = await api(`/ragv6-api/chunks/jobs/${encodeURIComponent(job.job_id)}`);
      state.jobs.set(fresh.job_id, fresh);
      if (fresh.status === "succeeded") {
        toast("索引重建与热切换完成", "success");
        await refreshRepository();
      } else if (fresh.status === "failed") {
        toast(`索引任务失败：${fresh.error || "未知错误"}`, "error");
      }
    } catch (error) {
      console.warn(error);
    }
  }));
  renderJobs();
  if (![...state.jobs.values()].some((job) => ["queued", "running"].includes(job.status))) {
    clearInterval(state.polling);
    state.polling = null;
  }
}

function startJobPolling() {
  if (state.polling) return;
  state.polling = setInterval(pollJobs, 2200);
  pollJobs();
}

function renderJobs() {
  const jobs = [...state.jobs.values()].sort((a, b) => String(b.created_at || "").localeCompare(String(a.created_at || "")));
  $("#job-list").innerHTML = jobs.length ? jobs.map((job) => `
    <article class="job-row">
      <div class="job-head"><strong>${job.kind === "rebuild_index" ? "检索索引重建" : escapeHtml(job.kind)}</strong><span class="status-pill ${job.status === "succeeded" ? "ready" : job.status === "failed" ? "error" : "warning"}">${escapeHtml(job.status)}</span></div>
      <div class="progress"><i style="width:${Math.max(0, Math.min(100, Number(job.progress || 0)))}%"></i></div>
      <p>${escapeHtml(job.message || "等待执行")}</p>
      <small>${escapeHtml(job.job_id)} · ${formatDate(job.started_at || job.created_at)}</small>
      ${job.error ? `<pre>${escapeHtml(job.error)}</pre>` : ""}
    </article>
  `).join("") : `<div class="empty">当前没有运行中的任务。</div>`;
}

async function refreshBackups() {
  try {
    const data = await api("/ragv6-api/chunks/backups");
    const backups = data.backups || [];
    $("#backup-list").innerHTML = backups.length ? backups.map((row) => `
      <article class="backup-row">
        <div><strong>${escapeHtml(row.manual || "未知手册")}</strong><small>${formatDate(row.created_at)} · ${row.files?.length || 0} 个文件</small></div>
        <button class="button" data-backup="${escapeHtml(row.backup_id)}">回滚</button>
      </article>
    `).join("") : `<div class="empty">尚未生成发布备份。</div>`;
    $$("[data-backup]").forEach((button) => button.addEventListener("click", () => rollback(button.dataset.backup)));
  } catch (error) {
    toast(error.message, "error");
  }
}

async function rollback(backupId) {
  const confirmed = await confirmAction(
    "回滚发布备份",
    `将恢复备份 ${backupId} 中的数据库与源文件，并立即启动索引重建。当前对应文件会被覆盖。`,
    "确认回滚",
  );
  if (!confirmed) return;
  try {
    const result = await api("/ragv6-api/chunks/rollback", {
      method: "POST",
      body: JSON.stringify({ backup_id: backupId, rebuild_index: true }),
    });
    if (result.job_id) state.jobs.set(result.job_id, { job_id: result.job_id, status: "queued", kind: "rebuild_index" });
    startJobPolling();
    renderJobs();
    toast("备份已恢复，正在重建索引", "success");
  } catch (error) {
    toast(error.message, "error");
  }
}

async function runSearchTest() {
  const question = $("#test-question").value.trim();
  if (!question) return toast("请输入测试问题", "error");
  const button = $("#run-search-test");
  button.disabled = true;
  try {
    const data = await api("/ragv6-api/chunks/search-test", {
      method: "POST",
      body: JSON.stringify({
        question,
        manual: state.currentManual?.manual || null,
        top_k: 6,
      }),
    });
    $("#search-results").innerHTML = (data.results || []).map((row, index) => `
      <article><span>${index + 1}</span><div><strong>${escapeHtml(row.product)} · ${escapeHtml(row.heading)}</strong><p>${escapeHtml(truncate(row.text, 220))}</p></div></article>
    `).join("") || `<div class="empty">当前索引没有返回结果。</div>`;
  } catch (error) {
    toast(error.message, "error");
  } finally {
    button.disabled = false;
  }
}

function switchTab(name) {
  document.body.dataset.activeView = name;
  $$(".tab").forEach((node) => node.classList.toggle("active", node.dataset.tab === name));
  $$(".view").forEach((node) => node.classList.toggle("active", node.id === `${name}-view`));
  if (name === "operations") refreshBackups();
}

function bind() {
  $$(".tab").forEach((node) => node.addEventListener("click", () => switchTab(node.dataset.tab)));
  $("#manual-search").addEventListener("input", renderManuals);
  $("#chunk-search").addEventListener("input", renderChunkRows);
  $("#preview-search").addEventListener("input", renderPreviewRows);
  $("#row-mode").addEventListener("click", (event) => {
    const button = event.target.closest("button[data-mode]");
    if (!button) return;
    state.rowMode = button.dataset.mode;
    state.selectedRow = currentRows().length
      ? { mode: state.rowMode, index: 0 }
      : null;
    $$("#row-mode button").forEach((node) => node.classList.toggle("selected", node === button));
    renderChunkRows();
  });
  $("#preview-mode").addEventListener("click", (event) => {
    const button = event.target.closest("button[data-mode]");
    if (!button) return;
    state.previewMode = button.dataset.mode;
    $$("#preview-mode button").forEach((node) => node.classList.toggle("selected", node === button));
    renderPreviewRows();
  });
  $("#file-input").addEventListener("change", (event) => handleFile(event.target.files?.[0]));
  const zone = $("#drop-zone");
  ["dragenter", "dragover"].forEach((eventName) => zone.addEventListener(eventName, (event) => {
    event.preventDefault(); zone.classList.add("dragging");
  }));
  ["dragleave", "drop"].forEach((eventName) => zone.addEventListener(eventName, (event) => {
    event.preventDefault(); zone.classList.remove("dragging");
  }));
  zone.addEventListener("drop", (event) => handleFile(event.dataTransfer.files?.[0]));
  $("#source-text").addEventListener("input", () => {
    if ($("#source-text").value.trim()) {
      state.upload.contentBase64 = null;
      state.upload.filename = state.upload.filename || "manual.md";
      $("#source-status").className = "status-pill ready";
      $("#source-status").textContent = "正文已载入";
    }
  });
  $("#preview-split").addEventListener("click", generatePreview);
  $("#publish-manual").addEventListener("click", publishManual);
  $("#rebuild-index").addEventListener("click", rebuildIndex);
  $("#refresh-all").addEventListener("click", refreshRepository);
  $("#refresh-jobs").addEventListener("click", pollJobs);
  $("#refresh-backups").addEventListener("click", refreshBackups);
  $("#run-search-test").addEventListener("click", runSearchTest);
  $("#test-question").addEventListener("keydown", (event) => {
    if (event.key === "Enter") runSearchTest();
  });
}

bind();
switchTab("library");
refreshRepository();
refreshBackups();
