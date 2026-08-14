"use strict";

const fs = require("fs");
const fsp = fs.promises;
const http = require("http");
const net = require("net");
const path = require("path");
const crypto = require("crypto");
const tls = require("tls");
const zlib = require("zlib");
const { DatabaseSync } = require("node:sqlite");
const { buildContextPacket } = require("./context-packet");
const { ChannelConcurrencyManager } = require("./channel-concurrency");
const { SharedState } = require("./shared-state");
const {
  findRecommendedAnswer,
  loadRecommendedAnswerCache,
  normalizeRecommendedQuestion,
} = require("./recommended-answer-cache");
const {
  hasMeaningfulAuditTrace,
  mergeAuditTrace,
} = require("./public/ragv6-ui/audit-contract");

function loadEnv(filePath) {
  if (!fs.existsSync(filePath)) return;
  const raw = fs.readFileSync(filePath, "utf8");
  for (const line of raw.split(/\r?\n/)) {
    const trimmed = line.trim();
    if (!trimmed || trimmed.startsWith("#")) continue;
    const match = trimmed.match(/^([A-Za-z_][A-Za-z0-9_]*)=(.*)$/);
    if (!match || process.env[match[1]] != null) continue;
    let value = match[2].trim();
    if ((value.startsWith('"') && value.endsWith('"')) || (value.startsWith("'") && value.endsWith("'"))) {
      value = value.slice(1, -1);
    }
    process.env[match[1]] = value;
  }
}

const ROOT = __dirname;
loadEnv(path.join(ROOT, ".env"));
const recommendedAnswerIndex = loadRecommendedAnswerCache();
const RECOMMENDED_ANSWER_MAX_WAIT_MS = Math.max(
  1_000,
  Number(process.env.RAGV6_RECOMMENDED_ANSWER_MAX_WAIT_MS || 16_000),
);
const sharedState = new SharedState();

const PRODUCT_MEMORY_MAX_TURNS = 6;
const PRODUCT_MEMORY_MAX_CHARS = 1200;
const PRODUCT_MEMORY_TTL_SECONDS = Math.max(
  3600,
  Number(process.env.RAGV6_PRODUCT_MEMORY_TTL_SECONDS || 7 * 24 * 60 * 60),
);
const productCatalog = (() => {
  try {
    const data = JSON.parse(fs.readFileSync(path.join(ROOT, "public", "ragv6-ui", "answers.json"), "utf8"));
    return (data.products || []).map((row) => String(row.name || "").trim()).filter(Boolean)
      .sort((left, right) => right.length - left.length);
  } catch {
    return [];
  }
})();
const productMemorySessions = new Map();

// Retrieval uses the canonical manual titles, while the browser catalog uses
// concise Chinese product labels. Keep source/manual names untouched, but
// normalize the product identity used by the UI and product-scoped memory.
const CANONICAL_MANUAL_TO_UI_PRODUCT = new Map([
  ["air fryer", "空气炸锅"],
  ["boat", "摩托艇"],
  ["camera", "相机"],
  ["相机手册", "混合即时相机"],
  ["earphones", "耳机"],
  ["espresso machine", "咖啡机"],
  ["gas grill", "烤架"],
  ["lawn mower", "割草机"],
  ["media player", "电子阅读器"],
  ["microwave", "微波炉"],
  ["motherboard", "主板"],
  ["phone", "固定电话"],
  ["printer", "传真机"],
  ["snowmobile", "雪地摩托"],
  ["tv", "电视/天线"],
  ["toothbrush", "电动牙刷"],
  ["vr headset", "VR头显"],
  ["vacuum", "扫地机器人"],
  ["waverunner", "水上摩托"],
  ["ergonomic chair", "人体工学椅"],
  ["exercise bike", "健身单车"],
  ["fitness tracker", "健身追踪器"],
  ["kids electric scooter", "儿童电动摩托车"],
  ["refrigerator", "冰箱"],
  ["keyboard", "功能键盘"],
  ["generator", "发电机"],
  ["programmable temperature controller", "可编程温控器"],
  ["leaf blower", "吹风机"],
  ["personal watercraft", "摩托艇"],
  ["water pump", "水泵"],
  ["dishwasher", "洗碗机"],
  ["oven", "烤箱"],
  ["power drill", "电钻"],
  ["camera manual", "混合即时相机"],
  ["air purifier", "空气净化器"],
  ["air conditioner", "空调"],
  ["steam cleaner", "蒸汽清洁机"],
  ["bluetooth laser mouse", "蓝牙鼠标"],
]);

function uiProductForManual(value) {
  const manual = String(value || "").trim();
  const normalized = manual.toLowerCase();
  const mapped = CANONICAL_MANUAL_TO_UI_PRODUCT.get(normalized);
  if (mapped) return mapped;
  const shortName = manual.replace(/(?:用户)?手册$/u, "").trim();
  const catalogProduct = productCatalog.find((name) => name.toLowerCase() === shortName.toLowerCase());
  return catalogProduct || manual;
}

function compactMemoryText(value, limit) {
  const text = String(value || "").replace(/\s+/g, " ").trim();
  return text.length <= limit ? text : `${text.slice(0, Math.max(0, limit - 1)).trim()}…`;
}

function explicitQuestionProduct(question) {
  const text = String(question || "").toLowerCase();
  const matches = productCatalog.filter((name) => text.includes(name.toLowerCase()));
  const distinct = [...new Set(matches.filter((name) => !matches.some((other) => other !== name && other.includes(name))))];
  if (!distinct.length) return { product: "", ambiguous: false };
  if (distinct.length > 1) return { product: "", ambiguous: true };
  return { product: distinct[0], ambiguous: false };
}

const GENERIC_COMPONENT_SECTION_RE = /(?:注意事项|安全说明|操作方法|使用方法|功能说明|推荐使用|清洁设备|日常维护|烹饪参考|烹饪指南)$/u;
const PRONOUN_FOLLOWUP_RE = /^(?:它|这个|那个|该(?:部件|配件|功能|装置)?)(?:的|在|要|该|怎么|如何|是否|能否|可以|需要|用于|有什么|有哪些|是)/u;

function componentFromAnswerSources(question, sources) {
  const rows = Array.isArray(sources) ? sources : [];
  const primary = rows.find((item) => item?.primary_evidence) || rows[0];
  const section = String(primary?.section || primary?.heading || "").trim();
  if (!section) return "";
  const leaf = section
    .split("/")
    .map((part) => part.trim())
    .filter(Boolean)
    .at(-1)
    ?.replace(/[（(](?:图|表|步骤|第).{0,20}[）)]\s*$/u, "")
    .trim();
  if (!leaf || leaf.length < 2 || leaf.length > 40 || GENERIC_COMPONENT_SECTION_RE.test(leaf)) return "";

  const current = String(question || "").replace(/\s+/g, "");
  const namedPart = leaf
    .split(/[、与和及]/u)
    .map((part) => part.trim())
    .filter((part) => part.length >= 2 && current.includes(part))
    .sort((left, right) => right.length - left.length)[0];
  return namedPart || leaf;
}

function memorySessionStorageKey(sessionId) {
  return `product-memory:${crypto.createHash("sha256").update(String(sessionId || "")).digest("hex")}`;
}

function hydrateMemorySession(payload) {
  if (!payload || typeof payload !== "object") return null;
  const buckets = new Map();
  for (const [key, turns] of Object.entries(payload.buckets || {})) {
    if (!key || !Array.isArray(turns)) continue;
    buckets.set(key, turns.slice(-PRODUCT_MEMORY_MAX_TURNS));
  }
  return {
    lastProduct: String(payload.lastProduct || "").slice(0, 200),
    lastMode: String(payload.lastMode || "").slice(0, 40),
    buckets,
    updatedAt: Number(payload.updatedAt || 0),
  };
}

function serializeMemorySession(session) {
  return {
    lastProduct: String(session?.lastProduct || "").slice(0, 200),
    lastMode: String(session?.lastMode || "").slice(0, 40),
    buckets: Object.fromEntries(
      [...(session?.buckets || new Map()).entries()]
        .map(([key, turns]) => [String(key), Array.isArray(turns) ? turns.slice(-PRODUCT_MEMORY_MAX_TURNS) : []]),
    ),
    updatedAt: Number(session?.updatedAt || Date.now()),
  };
}

async function getMemorySession(sessionId) {
  const key = String(sessionId || "").slice(0, 200);
  let existing = productMemorySessions.get(key);
  const shared = hydrateMemorySession(await sharedState.getJson(memorySessionStorageKey(key)));
  if (shared && (!existing || shared.updatedAt > Number(existing.updatedAt || 0))) {
    existing = shared;
    productMemorySessions.set(key, existing);
  }
  if (!existing) {
    existing = { lastProduct: "", lastMode: "", buckets: new Map(), updatedAt: Date.now() };
    productMemorySessions.set(key, existing);
  }
  if (productMemorySessions.size > 500) {
    [...productMemorySessions.entries()]
      .sort((left, right) => left[1].updatedAt - right[1].updatedAt)
      .slice(0, productMemorySessions.size - 500)
      .forEach(([oldKey]) => productMemorySessions.delete(oldKey));
  }
  return existing;
}

async function persistMemorySession(sessionId, session) {
  const key = String(sessionId || "").slice(0, 200);
  if (!key || !session) return;
  productMemorySessions.set(key, session);
  await sharedState.setJson(
    memorySessionStorageKey(key),
    serializeMemorySession(session),
    PRODUCT_MEMORY_TTL_SECONDS,
  );
}

function summarizeProductTurns(turns) {
  const lines = turns.slice(-PRODUCT_MEMORY_MAX_TURNS).map((turn, index) => (
    `${index + 1}. 用户：${compactMemoryText(turn.question, 160)}\n   助手：${compactMemoryText(turn.answer, 260)}`
  ));
  return compactMemoryText(lines.join("\n"), PRODUCT_MEMORY_MAX_CHARS);
}

function scopedProductSessionId(sessionId, product, memoryEpoch, requestId) {
  if (!product) return `raysource_unscoped_${String(requestId || Date.now()).slice(0, 80)}`;
  const digest = crypto.createHash("sha256")
    .update(`${String(sessionId || "")}\u0000${product}\u0000${memoryEpoch}`)
    .digest("hex")
    .slice(0, 24);
  return `raysource_product_${digest}`;
}

const ACCOUNT_DB = path.join(ROOT, "data", "customer-accounts.sqlite");
fs.mkdirSync(path.dirname(ACCOUNT_DB), { recursive: true });
const accountDb = new DatabaseSync(ACCOUNT_DB);
accountDb.exec(`
  PRAGMA journal_mode = WAL;
  PRAGMA foreign_keys = ON;
  CREATE TABLE IF NOT EXISTS users (
    id TEXT PRIMARY KEY,
    username TEXT NOT NULL UNIQUE COLLATE NOCASE,
    display_name TEXT NOT NULL,
    password_salt TEXT NOT NULL,
    password_hash TEXT NOT NULL,
    role TEXT NOT NULL DEFAULT 'customer',
    created_at INTEGER NOT NULL
  );
  CREATE TABLE IF NOT EXISTS sessions (
    token_hash TEXT PRIMARY KEY,
    user_id TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    created_at INTEGER NOT NULL,
    expires_at INTEGER NOT NULL
  );
  CREATE INDEX IF NOT EXISTS idx_sessions_user ON sessions(user_id);
  CREATE INDEX IF NOT EXISTS idx_sessions_expiry ON sessions(expires_at);
  CREATE TABLE IF NOT EXISTS conversations (
    id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    title TEXT NOT NULL,
    created_at INTEGER NOT NULL,
    updated_at INTEGER NOT NULL
  );
  CREATE INDEX IF NOT EXISTS idx_conversations_user_updated ON conversations(user_id, updated_at DESC);
  CREATE TABLE IF NOT EXISTS conversation_messages (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    conversation_id TEXT NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
    role TEXT NOT NULL CHECK(role IN ('user', 'assistant')),
    content TEXT NOT NULL,
    payload_json TEXT,
    created_at INTEGER NOT NULL
  );
  CREATE INDEX IF NOT EXISTS idx_messages_conversation ON conversation_messages(conversation_id, id);
`);

const MONITOR_DB = path.join(ROOT, "data", "service-monitor.sqlite");
const monitorDb = new DatabaseSync(MONITOR_DB);
monitorDb.exec(`
  PRAGMA journal_mode = WAL;
  PRAGMA synchronous = NORMAL;
  CREATE TABLE IF NOT EXISTS monitor_requests (
    request_id TEXT PRIMARY KEY,
    started_at INTEGER NOT NULL,
    completed_at INTEGER NOT NULL,
    method TEXT NOT NULL,
    path TEXT NOT NULL,
    category TEXT NOT NULL,
    status_code INTEGER NOT NULL,
    duration_ms INTEGER NOT NULL,
    product TEXT NOT NULL DEFAULT '',
    model TEXT NOT NULL DEFAULT '',
    answer_mode TEXT NOT NULL DEFAULT '',
    evidence_count INTEGER NOT NULL DEFAULT 0,
    image_count INTEGER NOT NULL DEFAULT 0,
    model_failed INTEGER NOT NULL DEFAULT 0,
    rag_no_evidence INTEGER NOT NULL DEFAULT 0,
    client_type TEXT NOT NULL DEFAULT ''
  );
  CREATE INDEX IF NOT EXISTS idx_monitor_requests_completed ON monitor_requests(completed_at DESC);
  CREATE INDEX IF NOT EXISTS idx_monitor_requests_category_completed ON monitor_requests(category, completed_at DESC);
  CREATE INDEX IF NOT EXISTS idx_monitor_requests_product_completed ON monitor_requests(product, completed_at DESC);
  CREATE TABLE IF NOT EXISTS monitor_feedback (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    request_id TEXT NOT NULL,
    product TEXT NOT NULL DEFAULT '',
    action TEXT NOT NULL,
    created_at INTEGER NOT NULL,
    UNIQUE(request_id, action)
  );
  CREATE INDEX IF NOT EXISTS idx_monitor_feedback_created ON monitor_feedback(created_at DESC);
  CREATE INDEX IF NOT EXISTS idx_monitor_feedback_request ON monitor_feedback(request_id);
  CREATE TABLE IF NOT EXISTS monitor_alerts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    alert_key TEXT NOT NULL,
    severity TEXT NOT NULL,
    message TEXT NOT NULL,
    channels_json TEXT NOT NULL DEFAULT '[]',
    delivery_json TEXT NOT NULL DEFAULT '[]',
    created_at INTEGER NOT NULL
  );
  CREATE INDEX IF NOT EXISTS idx_monitor_alerts_key_created ON monitor_alerts(alert_key, created_at DESC);
  CREATE INDEX IF NOT EXISTS idx_monitor_alerts_created ON monitor_alerts(created_at DESC);
`);

const RUNTIME_LOG_DIR = path.resolve(ROOT, "..", "runtime", "logs", "web-client");
const SERVICE_LOG = path.join(RUNTIME_LOG_DIR, "service.log");
const MAX_LOG_BYTES = Math.max(1024, Number(process.env.RAGV6_MAX_LOG_BYTES || 2 * 1024 * 1024));
const KEEP_LOG_ROTATIONS = Math.max(1, Number(process.env.RAGV6_KEEP_LOG_ROTATIONS || 3));
fs.mkdirSync(RUNTIME_LOG_DIR, { recursive: true });

function writeServiceLog(level, message) {
  try {
    if (fs.existsSync(SERVICE_LOG) && fs.statSync(SERVICE_LOG).size >= MAX_LOG_BYTES) {
      fs.rmSync(`${SERVICE_LOG}.${KEEP_LOG_ROTATIONS}`, { force: true });
      for (let index = KEEP_LOG_ROTATIONS - 1; index >= 1; index -= 1) {
        const source = `${SERVICE_LOG}.${index}`;
        if (fs.existsSync(source)) fs.renameSync(source, `${SERVICE_LOG}.${index + 1}`);
      }
      fs.renameSync(SERVICE_LOG, `${SERVICE_LOG}.1`);
    }
    fs.appendFileSync(SERVICE_LOG, `${new Date().toISOString()} [${level}] ${message}\n`, "utf8");
  } catch (error) {
    console.error(`service log write failed: ${error.message}`);
  }
}

const CANONICAL_RETRIEVAL_ORIGIN = "http://127.0.0.1:8014";

const CONFIG = {
  host: String(process.env.WEB_HOST || "127.0.0.1"),
  port: Number(process.env.WEB_PORT || 3000),
  apiOrigin: CANONICAL_RETRIEVAL_ORIGIN,
  chatOrigin: CANONICAL_RETRIEVAL_ORIGIN,
  chatBackend: String(process.env.RAGV6_CHAT_BACKEND || "vnext-fast"),
  apiToken: String(process.env.RAGV6_API_TOKEN || "change-me"),
  requestTimeoutMs: Number(process.env.RAGV6_REQUEST_TIMEOUT_MS || 360000),
};

const WECOM = {
  enabled: /^(1|true|yes|on)$/i.test(String(process.env.WECOM_ENABLED || "0")),
  token: String(process.env.WECOM_TOKEN || ""),
  encodingAesKey: String(process.env.WECOM_ENCODING_AES_KEY || ""),
  corpId: String(process.env.WECOM_CORP_ID || ""),
  agentId: String(process.env.WECOM_AGENT_ID || ""),
  secret: String(process.env.WECOM_SECRET || ""),
  sendApiKey: String(process.env.WECOM_SEND_API_KEY || ""),
};

// Web, WeCom, QQ and generic API traffic share one work-conserving scheduler.
// Weighted round-robin prevents one busy channel from starving the others,
// while session serialization protects conversation history ordering.
function chatEnvInteger(name, fallback, minimum = 0) {
  const parsed = Number(process.env[name]);
  return Number.isFinite(parsed) ? Math.max(minimum, Math.floor(parsed)) : fallback;
}

const CHAT_MAX_CONCURRENCY = chatEnvInteger("RAGV6_CHAT_MAX_CONCURRENCY", 4, 1);
const CHAT_DEFAULT_MAX_QUEUE = chatEnvInteger("RAGV6_CHAT_MAX_QUEUE", 8, 0);
const CHAT_QUEUE_TIMEOUT_MS = chatEnvInteger("RAGV6_CHAT_QUEUE_TIMEOUT_MS", 120_000, 1_000);
const CHAT_DISTRIBUTED_LEASE_TTL_MS = chatEnvInteger(
  "RAGV6_CHAT_DISTRIBUTED_LEASE_TTL_MS",
  180_000,
  5_000,
);
const chatCapacity = new ChannelConcurrencyManager({
  maxConcurrency: CHAT_MAX_CONCURRENCY,
  queueTimeoutMs: CHAT_QUEUE_TIMEOUT_MS,
  sessionQueueLimit: chatEnvInteger("RAGV6_CHAT_SESSION_QUEUE_LIMIT", 2, 0),
  channels: {
    web: {
      queueLimit: chatEnvInteger("RAGV6_CHAT_MAX_QUEUE_WEB", CHAT_DEFAULT_MAX_QUEUE, 0),
      weight: chatEnvInteger("RAGV6_CHAT_WEIGHT_WEB", 2, 1),
    },
    wecom: {
      queueLimit: chatEnvInteger("RAGV6_CHAT_MAX_QUEUE_WECOM", CHAT_DEFAULT_MAX_QUEUE, 0),
      weight: chatEnvInteger("RAGV6_CHAT_WEIGHT_WECOM", 1, 1),
    },
    qq: {
      queueLimit: chatEnvInteger("RAGV6_CHAT_MAX_QUEUE_QQ", CHAT_DEFAULT_MAX_QUEUE, 0),
      weight: chatEnvInteger("RAGV6_CHAT_WEIGHT_QQ", 1, 1),
    },
    api: {
      queueLimit: chatEnvInteger("RAGV6_CHAT_MAX_QUEUE_API", CHAT_DEFAULT_MAX_QUEUE, 0),
      weight: chatEnvInteger("RAGV6_CHAT_WEIGHT_API", 1, 1),
    },
  },
});

function chatRequestChannel(req) {
  const declared = String(req.headers["x-rag-channel"] || "").trim().toLowerCase();
  if (["web", "wecom", "qq", "api"].includes(declared)) return declared;
  return String(req.headers["x-client-type"] || "").trim().toLowerCase() === "web" ? "web" : "api";
}

function chatRequestSession(req) {
  return String(req.headers["x-rag-session-id"] || "").trim().slice(0, 240);
}

async function acquireChatSlot(req, requestId = "") {
  const startedAt = Date.now();
  const channel = chatRequestChannel(req);
  const sessionId = chatRequestSession(req);
  const localLease = await chatCapacity.acquire({
    channel,
    requestId,
    sessionId,
  });
  if (!localLease) return null;
  const distributedLease = await sharedState.acquireAdmission({
    channel,
    requestId,
    sessionId,
    limit: CHAT_MAX_CONCURRENCY,
    ttlMs: CHAT_DISTRIBUTED_LEASE_TTL_MS,
    waitTimeoutMs: Math.max(0, CHAT_QUEUE_TIMEOUT_MS - (Date.now() - startedAt)),
  });
  if (!distributedLease) {
    localLease.release();
    return null;
  }
  let released = false;
  return Object.freeze({
    channel: localLease.channel,
    requestId: localLease.requestId,
    sessionId: localLease.sessionId,
    queued: localLease.queued || distributedLease.waitMs > 0,
    localWaitMs: localLease.waitMs,
    distributedWaitMs: distributedLease.waitMs,
    distributedBackend: distributedLease.backend,
    waitMs: localLease.waitMs + distributedLease.waitMs,
    release: () => {
      if (released) return;
      released = true;
      localLease.release();
      void distributedLease.release();
    },
  });
}

function releaseChatSlot(lease) {
  lease?.release?.();
}

function chatCapacitySnapshot() {
  const local = chatCapacity.snapshot();
  const distributed = sharedState.health();
  return {
    ...local,
    distributed: {
      backend: distributed.mode,
      ready: distributed.ready,
      degraded: distributed.degraded,
      active_leases: distributed.active_distributed_leases,
      lease_acquired: distributed.lease_acquired,
      lease_rejected_global: distributed.lease_rejected_global,
      lease_rejected_session: distributed.lease_rejected_session,
      lease_rejected_duplicate: distributed.lease_rejected_duplicate,
    },
  };
}

const CHAT_BACKEND_CONFIG_FILE = path.join(ROOT, "backend-switch", "backend-active.json");

function currentChatBackend() {
  const fallback = {
    mode: "vnext-fast",
    origin: CANONICAL_RETRIEVAL_ORIGIN,
  };
  try {
    const parsed = JSON.parse(fs.readFileSync(CHAT_BACKEND_CONFIG_FILE, "utf8"));
    const mode = String(parsed.mode || "").trim();
    const origin = String(parsed.origin || "").trim().replace(/\/$/, "");
    // Production is deliberately single-source. Candidate and historical
    // ports are never selectable through a mutable file, so a future edit or
    // stale switch artifact cannot silently roll the website back.
    if (origin !== CANONICAL_RETRIEVAL_ORIGIN || mode !== "vnext-fast") return fallback;
    return { mode, origin };
  } catch {
    return fallback;
  }
}

const SESSION_COOKIE = "ragv6_session";
const SESSION_TTL_MS = 7 * 24 * 60 * 60 * 1000;
const GUEST_HISTORY_RETENTION_DAYS = Math.max(1, Number(process.env.RAGV6_GUEST_HISTORY_RETENTION_DAYS || 30));
const GUEST_HISTORY_RETENTION_MS = GUEST_HISTORY_RETENTION_DAYS * 24 * 60 * 60 * 1000;
const GUEST_IP_SECRET_FILE = path.join(ROOT, "data", "guest-ip.secret");
const authAttempts = new Map();
let lastGuestHistoryCleanupAt = 0;

function loadGuestIpSecret() {
  const configured = String(process.env.RAGV6_GUEST_IP_SECRET || "").trim();
  if (configured) return configured;
  try {
    if (fs.existsSync(GUEST_IP_SECRET_FILE)) {
      const stored = fs.readFileSync(GUEST_IP_SECRET_FILE, "utf8").trim();
      if (stored.length >= 32) return stored;
    }
    const generated = crypto.randomBytes(32).toString("hex");
    fs.writeFileSync(GUEST_IP_SECRET_FILE, `${generated}\n`, { encoding: "utf8", mode: 0o600 });
    return generated;
  } catch (error) {
    writeServiceLog("ERROR", `guest IP secret unavailable: ${error.message}`);
    return crypto.createHash("sha256").update(`${CONFIG.apiToken}|${ROOT}|ragv6-guest-ip`).digest("hex");
  }
}

const GUEST_IP_SECRET = loadGuestIpSecret();

function parseCookies(req) {
  const cookies = {};
  for (const part of String(req.headers.cookie || "").split(";")) {
    const index = part.indexOf("=");
    if (index < 1) continue;
    const key = part.slice(0, index).trim();
    const value = part.slice(index + 1).trim();
    try {
      cookies[key] = decodeURIComponent(value);
    } catch {
      cookies[key] = value;
    }
  }
  return cookies;
}

function hashSessionToken(token) {
  return crypto.createHash("sha256").update(String(token || "")).digest("hex");
}

function passwordDigest(password, salt) {
  return crypto.scryptSync(String(password), salt, 64, {
    N: 16384,
    r: 8,
    p: 1,
    maxmem: 64 * 1024 * 1024,
  }).toString("hex");
}

function publicUser(row) {
  if (!row) return null;
  return {
    id: row.id,
    username: row.username,
    display_name: row.display_name,
    role: row.role,
    created_at: row.created_at,
  };
}

function authenticatedUser(req) {
  const token = parseCookies(req)[SESSION_COOKIE];
  if (!token) return null;
  const tokenHash = hashSessionToken(token);
  const now = Date.now();
  const row = accountDb.prepare(`
    SELECT u.id, u.username, u.display_name, u.role, u.created_at
    FROM sessions s
    JOIN users u ON u.id = s.user_id
    WHERE s.token_hash = ? AND s.expires_at > ?
  `).get(tokenHash, now);
  return publicUser(row);
}

function requestIp(req) {
  const cloudflareIp = String(req.headers["cf-connecting-ip"] || "").trim();
  const forwardedIp = String(req.headers["x-forwarded-for"] || "").split(",")[0].trim();
  const socketIp = String(req.socket.remoteAddress || "").trim();
  return (cloudflareIp || forwardedIp || socketIp || "unknown")
    .replace(/^::ffff:/i, "")
    .replace(/%.+$/, "")
    .toLowerCase();
}

function guestIpHash(req) {
  return crypto.createHmac("sha256", GUEST_IP_SECRET).update(requestIp(req)).digest("hex");
}

function cleanupGuestHistory(now = Date.now(), force = false) {
  if (!force && now - lastGuestHistoryCleanupAt < 60 * 60 * 1000) return;
  lastGuestHistoryCleanupAt = now;
  const cutoff = now - GUEST_HISTORY_RETENTION_MS;
  accountDb.prepare(`
    DELETE FROM users
    WHERE role = 'guest_ip'
      AND (
        NOT EXISTS (SELECT 1 FROM conversations c WHERE c.user_id = users.id)
        OR COALESCE(
          (SELECT MAX(c.updated_at) FROM conversations c WHERE c.user_id = users.id),
          users.created_at
        ) < ?
      )
  `).run(cutoff);
}

function historyPrincipal(req) {
  const user = authenticatedUser(req);
  if (user) return { ...user, mode: "account" };
  const now = Date.now();
  cleanupGuestHistory(now);
  const hash = guestIpHash(req);
  const id = `guest-ip-${hash.slice(0, 40)}`;
  const username = `guest_${hash.slice(0, 24)}`;
  accountDb.prepare(`
    INSERT OR IGNORE INTO users(
      id, username, display_name, password_salt, password_hash, role, created_at
    ) VALUES (?, ?, '游客', '', '', 'guest_ip', ?)
  `).run(id, username, now);
  return {
    id,
    username,
    display_name: "游客",
    role: "guest_ip",
    created_at: now,
    mode: "guest",
  };
}

function consumeAuthAttempt(req, username = "") {
  const now = Date.now();
  const key = `${requestIp(req)}|${String(username).trim().toLowerCase()}`;
  const row = authAttempts.get(key) || { startedAt: now, count: 0 };
  if (now - row.startedAt > 15 * 60 * 1000) {
    row.startedAt = now;
    row.count = 0;
  }
  row.count += 1;
  authAttempts.set(key, row);
  if (authAttempts.size > 2000) {
    for (const [ip, value] of authAttempts) {
      if (now - value.startedAt > 15 * 60 * 1000) authAttempts.delete(ip);
    }
  }
  return row.count <= 12;
}

function verifySameSiteRequest(req) {
  const fetchSite = String(req.headers["sec-fetch-site"] || "").toLowerCase();
  return !fetchSite || fetchSite === "same-origin" || fetchSite === "same-site";
}

function sessionCookie(req, token, maxAgeSeconds = Math.floor(SESSION_TTL_MS / 1000)) {
  const forwardedProto = String(req.headers["x-forwarded-proto"] || "").toLowerCase();
  const secure = forwardedProto.split(",").some((value) => value.trim() === "https");
  return [
    `${SESSION_COOKIE}=${encodeURIComponent(token)}`,
    "Path=/",
    "HttpOnly",
    "SameSite=Lax",
    secure ? "Secure" : "",
    `Max-Age=${Math.max(0, maxAgeSeconds)}`,
  ].filter(Boolean).join("; ");
}

function createLoginSession(req, userId) {
  const token = crypto.randomBytes(32).toString("base64url");
  const now = Date.now();
  accountDb.prepare("DELETE FROM sessions WHERE expires_at <= ?").run(now);
  accountDb.prepare(`
    INSERT INTO sessions(token_hash, user_id, created_at, expires_at)
    VALUES (?, ?, ?, ?)
  `).run(hashSessionToken(token), userId, now, now + SESSION_TTL_MS);
  return { token, cookie: sessionCookie(req, token) };
}

function validateCredentials(username, password) {
  const normalizedUsername = String(username || "").trim();
  const normalizedPassword = String(password || "");
  if (!/^[\p{L}\p{N}_.-]{3,32}$/u.test(normalizedUsername)) {
    throw new Error("账号需为 3–32 位文字、数字、下划线、点或短横线");
  }
  if (normalizedPassword.length < 8 || normalizedPassword.length > 128) {
    throw new Error("密码长度需为 8–128 位");
  }
  return { username: normalizedUsername, password: normalizedPassword };
}

function requireAccount(req, res) {
  const user = authenticatedUser(req);
  if (!user) {
    sendJson(res, 401, { code: 401, msg: "请先登录账号", data: null });
    return null;
  }
  return user;
}

function safeConversationTitle(value) {
  return String(value || "新的咨询").replace(/\s+/g, " ").trim().slice(0, 60) || "新的咨询";
}

const UI_DIR = path.join(ROOT, "public", "ragv6-ui");
const MANUAL_ROOT = path.join(ROOT, "public", "ragv6-manual-index");
// The public manual page must retain the full handbook corpus. The lightweight
// directory replaced it during the July 30 change and broke the expected view.
const MANUAL_INDEX = path.join(MANUAL_ROOT, "manual-index-full-20260730.html");
// Source citations use the same complete document shell so URL fragments can
// expand the cited section and apply text/image highlights.
const MANUAL_LOCATOR = path.join(MANUAL_ROOT, "manual-index-full-20260730.html");
const MANUALS_DIR = path.join(MANUAL_ROOT, "manuals");
const MANUAL_MANIFEST = path.join(MANUALS_DIR, "manifest.json");
const IMAGE_DIR = path.join(ROOT, "public", "manual-images");
const CAPTIONS_FILE = path.join(ROOT, "data", "image_captions_v4_final.json");
const CHUNK_MANAGER_DIR = path.join(ROOT, "public", "chunk-manager");
const ANDROID_APP_FILE = path.join(ROOT, "public", "downloads", "raysource-android.apk");
const MONITOR_DIR = path.join(ROOT, "public", "monitor");
// Keep static/manual fallback data inside the web module. Backend management
// always goes through RAGV6_API_ORIGIN, so the two packages can run on separate
// machines without sharing a filesystem.
const LOCAL_DATA_DIR = path.join(ROOT, "data");
const MANUAL_SECTIONS_DIR = path.join(LOCAL_DATA_DIR, "manual_sections");
const RETRIEVAL_CHUNKS_FILE = path.join(LOCAL_DATA_DIR, "retrieval_chunks.json");
const CHUNK_BACKUP_DIR = path.join(LOCAL_DATA_DIR, "chunk-manager-backups");

const MIME = {
  ".html": "text/html; charset=utf-8",
  ".js": "text/javascript; charset=utf-8",
  ".css": "text/css; charset=utf-8",
  ".json": "application/json; charset=utf-8",
  ".jpg": "image/jpeg",
  ".jpeg": "image/jpeg",
  ".png": "image/png",
  ".svg": "image/svg+xml",
  ".gif": "image/gif",
  ".webp": "image/webp",
  ".bmp": "image/bmp",
  ".apk": "application/vnd.android.package-archive",
};

const MODEL_PROFILES = [
  { id: "v6-luna", label: "GPT Luna", description: "复杂推理与图像理解", provider: "GPT", model: "gpt-5.6-luna", wire_api: "responses", icon: "gpt-luna", multimodal: true, reasoning_options: ["low", "medium", "high"], default_reasoning: "medium" },
  { id: "v6-sol", label: "GPT Sol", description: "速度与回答质量均衡", provider: "GPT", model: "gpt-5.6-sol", wire_api: "responses", icon: "gpt-sol", multimodal: true, reasoning_options: ["low", "medium"], default_reasoning: "medium" },
  { id: "v6-terra", label: "GPT Terra", description: "支持文本与图像理解", provider: "GPT", model: "gpt-5.6-terra", wire_api: "responses", icon: "gpt-terra", multimodal: true, reasoning_options: ["low", "medium", "high"], default_reasoning: "medium" },
  { id: "deepseek-v4-flash", label: "DeepSeek V4 Flash", description: "快速文本问答", provider: "DeepSeek", model: "deepseek-v4-flash", wire_api: "openai", icon: "deepseek-v4-flash", text_only: true, reasoning_options: ["low", "high"], default_reasoning: "high" },
  { id: "deepseek-v4-pro", label: "DeepSeek V4 Pro", description: "高质量文本问答", provider: "DeepSeek", model: "deepseek-v4-pro", wire_api: "openai", icon: "deepseek-v4-pro", text_only: true, reasoning_options: ["medium", "high"], default_reasoning: "high" },
  { id: "qwen-open-source", label: "Qwen 开源模型", description: "文本问答；图片由 GPT 多模态识别", provider: "Qwen", model: "qwen3-plus", wire_api: "openai", icon: "qwen3-plus", multimodal: true, force_gpt_for_media: true, reasoning_options: ["low", "medium", "high"], default_reasoning: "medium" },
  { id: "glm-5.2", label: "GLM 5.2", description: "文本模型接入准备中", provider: "GLM", model: "glm-5-2", wire_api: "openai", icon: "glm-5-2", text_only: true, available: false, reasoning_options: ["medium", "high"], default_reasoning: "medium" },
  { id: "kimi-2.6", label: "Kimi 2.6", description: "文本模型接入准备中", provider: "Kimi", model: "kimi-2.6", wire_api: "openai", icon: "kimi-2-6", text_only: true, available: false, reasoning_options: [], default_reasoning: "" },
  { id: "minimax-2.5", label: "MiniMax 2.5", description: "文本模型接入准备中", provider: "MiniMax", model: "minimax-2.5", wire_api: "openai", icon: "minimax-2-5", text_only: true, available: false, reasoning_options: ["low", "medium"], default_reasoning: "medium" },
  { id: "grok-4.5", label: "Grok 4.5", description: "文本模型接入准备中", provider: "xAI", model: "grok-4.5", wire_api: "openai", icon: "grok-4-5", text_only: true, available: false, reasoning_options: ["high"], default_reasoning: "high" },
];
// Public profiles remain distinct in the UI.  They are presentation labels
// only: all answer generation is deliberately served by the single stable
// Terra Medium route below.
const UNIFIED_RUNTIME_MODEL = "gpt-5.6-terra";
const UNIFIED_RUNTIME_REASONING = "medium";
let activeProfileId = String(process.env.RAGV6_DEFAULT_PROFILE || "qwen-open-source");
if (!MODEL_PROFILES.some((profile) => profile.id === activeProfileId)) activeProfileId = "qwen-open-source";

const progressStore = new Map();
let manualManifest = null;
let captions = null;
let captionImageIndex = null;
let sectionCaptionIndex = null;

const MANUAL_TITLE_ALIASES = {
  "VR Headset": ["VR头显手册"],
  "Ergonomic Chair": ["人体工学椅手册"],
  "Exercise Bike": ["健身单车手册"],
  "Fitness Tracker": ["健身追踪器手册"],
  "Kids Electric Scooter": ["儿童电动摩托车手册"],
  Refrigerator: ["冰箱手册"],
  Keyboard: ["功能键盘手册"],
  Generator: ["发电机手册"],
  "Programmable Temperature Controller": ["可编程温控器手册"],
  "Leaf Blower": ["吹风机手册"],
  "Personal Watercraft": ["摩托艇手册"],
  "Water Pump": ["水泵手册"],
  Dishwasher: ["洗碗机手册"],
  Oven: ["烤箱手册"],
  "Power Drill": ["电钻手册"],
  "Camera Manual": ["相机手册"],
  "Air Purifier": ["空气净化器手册"],
  "Air Conditioner": ["空调手册"],
  "Steam Cleaner": ["蒸汽清洁机手册"],
  "Bluetooth Laser Mouse": ["蓝牙激光鼠标手册"],
};

const monitorState = {
  startedAt: Date.now(),
  activeRequests: 0,
  backendFailureStreak: 0,
};
const MONITOR_RETENTION_DAYS = Math.max(7, Number(process.env.RAGV6_MONITOR_RETENTION_DAYS || 90));
const METRIC_WINDOW_MS = 30 * 24 * 60 * 60 * 1000;
const REQUEST_ID_PATTERN = /^[A-Za-z0-9._:-]{8,128}$/;

function monitorPathname(rawUrl) {
  try {
    return new URL(rawUrl || "/", "http://localhost").pathname.toLowerCase();
  } catch {
    return "/";
  }
}

function normalizeRequestId(value) {
  const candidate = String(value || "").trim();
  return REQUEST_ID_PATTERN.test(candidate) ? candidate : crypto.randomUUID();
}

function requestCategory(pathname) {
  if (pathname === "/ragv6-api/chat" || pathname === "/chat") return "chat";
  if (pathname === "/ragv6-api/feedback") return "feedback";
  if (pathname.startsWith("/ragv6-api/account")) return "account";
  if (pathname.startsWith("/ragv6-api/chunks")) return "knowledge_admin";
  if (pathname.startsWith("/ragv6-api")) return "api";
  if (["/health", "/livez", "/readyz", "/metrics"].includes(pathname)) return "health";
  if (pathname === "/rag" || pathname.startsWith("/rag/")) return "page";
  if (pathname.startsWith("/ragv6")) return "page";
  return "other";
}

function recordMonitorRequest(req, statusCode, durationMs, completedAt = Date.now()) {
  const pathname = monitorPathname(req.url);
  if (pathname.startsWith("/internal-monitor-api")) return;
  const context = req.monitorContext || {};
  try {
    monitorDb.prepare(`
      INSERT OR REPLACE INTO monitor_requests(
        request_id, started_at, completed_at, method, path, category,
        status_code, duration_ms, product, model, answer_mode,
        evidence_count, image_count, model_failed, rag_no_evidence, client_type
      ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    `).run(
      normalizeRequestId(context.requestId),
      Number(context.startedAt || completedAt - durationMs),
      completedAt,
      String(req.method || "GET"),
      pathname,
      String(context.category || requestCategory(pathname)),
      Number(statusCode || 0),
      Math.max(0, Math.round(durationMs)),
      String(context.product || "").slice(0, 160),
      String(context.model || "").slice(0, 120),
      String(context.answerMode || "").slice(0, 40),
      Math.max(0, Number(context.evidenceCount || 0)),
      Math.max(0, Number(context.imageCount || 0)),
      context.modelFailed ? 1 : 0,
      context.ragNoEvidence ? 1 : 0,
      String(context.channel || req.headers["x-rag-channel"] || req.headers["x-client-type"] || "").slice(0, 40),
    );
  } catch (error) {
    writeServiceLog("ERROR", `monitor persistence failed request_id=${context.requestId || "unknown"} error=${error.message}`);
  }
}

function percentile(values, pct) {
  if (!values.length) return 0;
  const sorted = [...values].sort((a, b) => a - b);
  return sorted[Math.max(0, Math.min(sorted.length - 1, Math.ceil(sorted.length * pct) - 1))] || 0;
}

function monitorTimeline() {
  const currentMinute = Math.floor(Date.now() / 60000) * 60000;
  const buckets = new Map();
  const recent = monitorDb.prepare(`
    SELECT completed_at, status_code, duration_ms, category
    FROM monitor_requests WHERE completed_at >= ?
  `).all(currentMinute - (59 * 60000));
  for (const row of recent) {
    const timestamp = Math.floor(Number(row.completed_at) / 60000) * 60000;
    const bucket = buckets.get(timestamp) || {
      timestamp,
      requests: 0,
      errors: 0,
      chats: 0,
      latency_total_ms: 0,
      max_latency_ms: 0,
      latencies: [],
    };
    bucket.requests += 1;
    if (Number(row.status_code) >= 400) bucket.errors += 1;
    if (row.category === "chat") bucket.chats += 1;
    bucket.latency_total_ms += Number(row.duration_ms || 0);
    bucket.max_latency_ms = Math.max(bucket.max_latency_ms, Number(row.duration_ms || 0));
    bucket.latencies.push(Number(row.duration_ms || 0));
    buckets.set(timestamp, bucket);
  }
  const rows = [];
  for (let offset = 59; offset >= 0; offset -= 1) {
    const timestamp = currentMinute - (offset * 60000);
    const bucket = buckets.get(timestamp) || {
      timestamp,
      requests: 0,
      errors: 0,
      chats: 0,
      latency_total_ms: 0,
      max_latency_ms: 0,
      latencies: [],
    };
    const { latencies, ...publicBucket } = bucket;
    rows.push({
      ...publicBucket,
      label: new Date(timestamp).toLocaleTimeString("zh-CN", { hour: "2-digit", minute: "2-digit", hour12: false }),
      avg_latency_ms: bucket.requests ? Math.round(bucket.latency_total_ms / bucket.requests) : 0,
      max_latency_ms: Math.round(bucket.max_latency_ms || 0),
      p95_latency_ms: percentile(latencies, 0.95),
      p99_latency_ms: percentile(latencies, 0.99),
    });
  }
  return rows;
}

function latestResolutionByRequest(since) {
  const result = new Map();
  const rows = monitorDb.prepare(`
    SELECT request_id, action, created_at
    FROM monitor_feedback
    WHERE created_at >= ? AND action IN ('solved', 'unsolved')
    ORDER BY created_at ASC, id ASC
  `).all(since);
  for (const row of rows) result.set(row.request_id, row.action);
  return result;
}

function productAndQualityMetrics(since) {
  const chats = monitorDb.prepare(`
    SELECT request_id, product, model_failed, rag_no_evidence, duration_ms
    FROM monitor_requests
    WHERE category = 'chat' AND completed_at >= ? AND client_type <> 'qa'
    ORDER BY completed_at DESC
  `).all(since);
  const feedback = monitorDb.prepare(`
    SELECT request_id, action, created_at
    FROM monitor_feedback WHERE created_at >= ?
  `).all(since);
  const resolutions = latestResolutionByRequest(since);
  const transfers = new Set(feedback.filter((row) => row.action === "transfer").map((row) => row.request_id));
  const tickets = new Set(feedback.filter((row) => row.action === "ticket_submit").map((row) => row.request_id));
  const productMap = new Map();

  for (const row of chats) {
    const product = String(row.product || "").trim() || "未识别产品";
    const stats = productMap.get(product) || {
      product,
      consultations: 0,
      solved: 0,
      unsolved: 0,
      transfers: 0,
      model_failures: 0,
      no_evidence: 0,
      latency_total_ms: 0,
    };
    stats.consultations += 1;
    stats.model_failures += Number(row.model_failed || 0);
    stats.no_evidence += Number(row.rag_no_evidence || 0);
    stats.latency_total_ms += Number(row.duration_ms || 0);
    if (resolutions.get(row.request_id) === "solved") stats.solved += 1;
    if (resolutions.get(row.request_id) === "unsolved") stats.unsolved += 1;
    if (transfers.has(row.request_id)) stats.transfers += 1;
    productMap.set(product, stats);
  }

  const percent = (value, total) => total ? Math.round(value / total * 10000) / 100 : 0;
  const productStats = [...productMap.values()]
    .map((row) => ({
      ...row,
      feedback_count: row.solved + row.unsolved,
      solution_rate_percent: percent(row.solved, row.solved + row.unsolved),
      transfer_rate_percent: percent(row.transfers, row.consultations),
      model_failure_rate_percent: percent(row.model_failures, row.consultations),
      rag_no_evidence_rate_percent: percent(row.no_evidence, row.consultations),
      avg_latency_ms: row.consultations ? Math.round(row.latency_total_ms / row.consultations) : 0,
    }))
    .sort((left, right) => right.consultations - left.consultations || left.product.localeCompare(right.product, "zh-CN"));

  const consultations = chats.length;
  const solved = chats.filter((row) => resolutions.get(row.request_id) === "solved").length;
  const unsolved = chats.filter((row) => resolutions.get(row.request_id) === "unsolved").length;
  const modelFailures = chats.reduce((sum, row) => sum + Number(row.model_failed || 0), 0);
  const noEvidence = chats.reduce((sum, row) => sum + Number(row.rag_no_evidence || 0), 0);
  const transferCount = chats.filter((row) => transfers.has(row.request_id)).length;
  const ticketCount = chats.filter((row) => tickets.has(row.request_id)).length;
  return {
    products: productStats,
    quality: {
      window_days: 30,
      consultations,
      solved,
      unsolved,
      resolution_feedback_count: solved + unsolved,
      solution_rate_percent: percent(solved, solved + unsolved),
      model_failures: modelFailures,
      model_failure_rate_percent: percent(modelFailures, consultations),
      rag_no_evidence: noEvidence,
      rag_no_evidence_rate_percent: percent(noEvidence, consultations),
      transfers: transferCount,
      transfer_rate_percent: percent(transferCount, consultations),
      tickets: ticketCount,
      ticket_rate_percent: percent(ticketCount, consultations),
    },
  };
}

function readRecentServiceLogs(limit = 80) {
  try {
    if (!fs.existsSync(SERVICE_LOG)) return [];
    const stat = fs.statSync(SERVICE_LOG);
    const start = Math.max(0, stat.size - 128 * 1024);
    const buffer = Buffer.alloc(stat.size - start);
    const descriptor = fs.openSync(SERVICE_LOG, "r");
    fs.readSync(descriptor, buffer, 0, buffer.length, start);
    fs.closeSync(descriptor);
    return buffer.toString("utf8").split(/\r?\n/).filter(Boolean).slice(-limit);
  } catch (error) {
    return [`日志读取失败：${error.message}`];
  }
}

async function backendHealth() {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), 1800);
  try {
    const response = await fetch(new URL("/health", CONFIG.apiOrigin), { signal: controller.signal });
    return { reachable: response.ok, status: response.status };
  } catch (error) {
    return { reachable: false, error: error.name === "AbortError" ? "timeout" : error.message };
  } finally {
    clearTimeout(timer);
  }
}

function alertChannelConfiguration() {
  const recipients = String(process.env.RAGV6_ALERT_EMAIL_TO || "").split(",").map((value) => value.trim()).filter(Boolean);
  return {
    email: {
      configured: Boolean(process.env.RAGV6_ALERT_SMTP_HOST && recipients.length),
      recipients: recipients.length,
    },
    wecom: { configured: Boolean(process.env.RAGV6_ALERT_WECOM_WEBHOOK) },
    feishu: { configured: Boolean(process.env.RAGV6_ALERT_FEISHU_WEBHOOK) },
  };
}

function recentAlertRows(limit = 20) {
  return monitorDb.prepare(`
    SELECT id, alert_key, severity, message, channels_json, delivery_json, created_at
    FROM monitor_alerts ORDER BY created_at DESC LIMIT ?
  `).all(Math.max(1, Math.min(100, Number(limit) || 20))).map((row) => {
    let channels = [];
    let delivery = [];
    try { channels = JSON.parse(row.channels_json || "[]"); } catch {}
    try { delivery = JSON.parse(row.delivery_json || "[]"); } catch {}
    return { ...row, channels, delivery };
  });
}

async function monitorSnapshot() {
  const since = Date.now() - METRIC_WINDOW_MS;
  const aggregate = monitorDb.prepare(`
    SELECT
      COUNT(*) AS total_requests,
      SUM(CASE WHEN status_code >= 400 THEN 1 ELSE 0 END) AS error_requests,
      SUM(CASE WHEN category = 'chat' THEN 1 ELSE 0 END) AS chat_requests,
      AVG(duration_ms) AS avg_latency_ms
    FROM monitor_requests
  `).get();
  const durations = monitorDb.prepare(`
    SELECT duration_ms FROM monitor_requests
    WHERE completed_at >= ? ORDER BY completed_at DESC LIMIT 50000
  `).all(since).map((row) => Number(row.duration_ms || 0));
  const total = Number(aggregate.total_requests || 0);
  const errorRequests = Number(aggregate.error_requests || 0);
  let memory = { rss: 0, heapUsed: 0 };
  try { memory = process.memoryUsage(); } catch {}
  const timeline = monitorTimeline();
  const productMetrics = productAndQualityMetrics(since);
  const traces = monitorDb.prepare(`
    SELECT request_id, completed_at, category, status_code, duration_ms, product,
           model, answer_mode, evidence_count, image_count, model_failed, rag_no_evidence, client_type
    FROM monitor_requests
    ORDER BY completed_at DESC LIMIT 80
  `).all();
  return {
    timestamp: new Date().toISOString(),
    service: { name: "ragv6-web-client", listen: `${CONFIG.host}:${CONFIG.port}` },
    process: {
      pid: process.pid,
      uptime_seconds: Math.round(process.uptime()),
      memory: {
        rss_mb: Math.round(memory.rss / 1024 / 1024 * 10) / 10,
        heap_used_mb: Math.round(memory.heapUsed / 1024 / 1024 * 10) / 10,
      },
    },
    backend: await backendHealth(),
    shared_state: sharedState.health(),
    chat_capacity: chatCapacitySnapshot(),
    model: currentModelProfile().data?.active || {},
    metrics: {
      total_requests: total,
      active_requests: monitorState.activeRequests,
      error_requests: errorRequests,
      chat_requests: Number(aggregate.chat_requests || 0),
      last_minute_requests: timeline[timeline.length - 1]?.requests || 0,
      avg_latency_ms: Math.round(Number(aggregate.avg_latency_ms || 0)),
      p50_latency_ms: percentile(durations, 0.50),
      p95_latency_ms: percentile(durations, 0.95),
      p99_latency_ms: percentile(durations, 0.99),
      error_rate_percent: total ? Math.round(errorRequests / total * 10000) / 100 : 0,
      percentile_window_days: 30,
    },
    quality: productMetrics.quality,
    products: productMetrics.products,
    timeline,
    traces,
    alerts: {
      channels: alertChannelConfiguration(),
      thresholds: {
        error_rate_percent: Number(process.env.RAGV6_ALERT_ERROR_RATE || 20),
        model_failure_rate_percent: Number(process.env.RAGV6_ALERT_MODEL_FAILURE_RATE || 10),
        rag_no_evidence_rate_percent: Number(process.env.RAGV6_ALERT_NO_EVIDENCE_RATE || 40),
        p95_latency_ms: Number(process.env.RAGV6_ALERT_P95_MS || 60000),
      },
      recent: recentAlertRows(20),
    },
    logs: readRecentServiceLogs(60),
  };
}

function prometheusMetrics() {
  const capacity = chatCapacitySnapshot();
  const shared = sharedState.health();
  const aggregate = monitorDb.prepare(`
    SELECT COUNT(*) AS total_requests,
           SUM(CASE WHEN status_code >= 400 THEN 1 ELSE 0 END) AS error_requests,
           AVG(duration_ms) AS avg_latency_ms
    FROM monitor_requests
  `).get();
  const lines = [
    "# HELP raysource_gateway_up Whether the Node gateway process is running.",
    "# TYPE raysource_gateway_up gauge",
    "raysource_gateway_up 1",
    "# HELP raysource_chat_active Active chat requests in this gateway instance.",
    "# TYPE raysource_chat_active gauge",
    `raysource_chat_active ${capacity.active}`,
    "# HELP raysource_chat_queued Queued chat requests in this gateway instance.",
    "# TYPE raysource_chat_queued gauge",
    `raysource_chat_queued ${capacity.queued}`,
    "# HELP raysource_chat_concurrency_limit Configured cross-channel concurrency limit.",
    "# TYPE raysource_chat_concurrency_limit gauge",
    `raysource_chat_concurrency_limit ${capacity.limit}`,
    "# HELP raysource_shared_state_ready Whether Redis/Garnet shared state is ready.",
    "# TYPE raysource_shared_state_ready gauge",
    `raysource_shared_state_ready ${shared.ready ? 1 : 0}`,
    "# HELP raysource_shared_state_degraded Whether shared state is using local-memory degradation.",
    "# TYPE raysource_shared_state_degraded gauge",
    `raysource_shared_state_degraded ${shared.degraded ? 1 : 0}`,
    "# HELP raysource_distributed_leases Active Redis-backed admission leases owned by this instance.",
    "# TYPE raysource_distributed_leases gauge",
    `raysource_distributed_leases ${shared.active_distributed_leases}`,
    "# HELP raysource_shared_state_errors_total Shared-state operation errors.",
    "# TYPE raysource_shared_state_errors_total counter",
    `raysource_shared_state_errors_total ${shared.errors}`,
    "# HELP raysource_shared_state_memory_fallbacks_total Shared-state operations that used local-memory fallback.",
    "# TYPE raysource_shared_state_memory_fallbacks_total counter",
    `raysource_shared_state_memory_fallbacks_total ${shared.memory_fallbacks}`,
    "# HELP raysource_requests_total Persisted gateway request count.",
    "# TYPE raysource_requests_total counter",
    `raysource_requests_total ${Number(aggregate.total_requests || 0)}`,
    "# HELP raysource_request_errors_total Persisted gateway error response count.",
    "# TYPE raysource_request_errors_total counter",
    `raysource_request_errors_total ${Number(aggregate.error_requests || 0)}`,
    "# HELP raysource_request_latency_average_ms Average persisted request latency in milliseconds.",
    "# TYPE raysource_request_latency_average_ms gauge",
    `raysource_request_latency_average_ms ${Math.round(Number(aggregate.avg_latency_ms || 0))}`,
    "# HELP raysource_product_memory_sessions Local hot product-memory session count.",
    "# TYPE raysource_product_memory_sessions gauge",
    `raysource_product_memory_sessions ${productMemorySessions.size}`,
  ];
  for (const [channel, stats] of Object.entries(capacity.channels || {})) {
    const label = String(channel).replace(/[^A-Za-z0-9_-]/g, "_");
    lines.push(`raysource_chat_channel_active{channel="${label}"} ${stats.active}`);
    lines.push(`raysource_chat_channel_queued{channel="${label}"} ${stats.queued}`);
    lines.push(`raysource_chat_channel_completed_total{channel="${label}"} ${stats.completed}`);
    lines.push(`raysource_chat_channel_rejected_total{channel="${label}",reason="full"} ${stats.rejected_full}`);
    lines.push(`raysource_chat_channel_rejected_total{channel="${label}",reason="session"} ${stats.rejected_session}`);
    lines.push(`raysource_chat_channel_rejected_total{channel="${label}",reason="duplicate"} ${stats.rejected_duplicate}`);
  }
  return `${lines.join("\n")}\n`;
}

async function runMonitorCommand(command) {
  const rawCommand = String(command || "").trim();
  const [rawName = "", ...args] = rawCommand.split(/\s+/);
  const commandName = rawName.toLowerCase();
  if (commandName === "help") {
    return [
      "可用白名单命令：",
      "  status   Web 服务进程与运行时间",
      "  health   Web 与 RAG 检索服务健康检查",
      "  metrics  当前请求、并发、延迟与错误指标",
      "  quality  模型失败、无证据、解决与转人工指标",
      "  products 各产品咨询量与解决率",
      "  alerts   告警通道、阈值和近期告警",
      "  trace <Request ID>  查看单个请求的全链路元数据",
      "  model    当前启用的模型配置",
      "  logs     最近 40 行服务日志",
      "  help     显示本帮助",
      "",
      "安全说明：此控制台不执行 PowerShell、CMD 或任意系统命令。",
    ].join("\n");
  }
  const snapshot = await monitorSnapshot();
  if (commandName === "status") {
    return [
      `service: ${snapshot.service.name}`,
      `listen: ${snapshot.service.listen}`,
      `pid: ${snapshot.process.pid}`,
      `uptime: ${snapshot.process.uptime_seconds}s`,
      `memory_rss: ${snapshot.process.memory.rss_mb} MB`,
    ].join("\n");
  }
  if (commandName === "health") {
    return [
      "web: healthy",
      `rag_backend: ${snapshot.backend.reachable ? "healthy" : "unreachable"}`,
      `backend_status: ${snapshot.backend.status || snapshot.backend.error || "unknown"}`,
    ].join("\n");
  }
  if (commandName === "metrics") return JSON.stringify(snapshot.metrics, null, 2);
  if (commandName === "quality") return JSON.stringify(snapshot.quality, null, 2);
  if (commandName === "products") return JSON.stringify(snapshot.products, null, 2);
  if (commandName === "alerts") return JSON.stringify(snapshot.alerts, null, 2);
  if (commandName === "trace") {
    const requestId = String(args[0] || "");
    if (!REQUEST_ID_PATTERN.test(requestId)) throw new Error("请输入有效的 Request ID");
    const trace = monitorDb.prepare("SELECT * FROM monitor_requests WHERE request_id = ?").get(requestId);
    if (!trace) throw new Error("未找到该 Request ID");
    const feedback = monitorDb.prepare(`
      SELECT action, product, created_at FROM monitor_feedback WHERE request_id = ? ORDER BY created_at
    `).all(requestId);
    const progress = progressStore.get(requestId)?.events || [];
    return JSON.stringify({ request: trace, feedback, live_progress: progress }, null, 2);
  }
  if (commandName === "model") return JSON.stringify(snapshot.model, null, 2);
  if (commandName === "logs") return readRecentServiceLogs(40).join("\n") || "暂无日志";
  throw new Error("命令不在安全白名单中；输入 help 查看可用命令");
}

function smtpResponseReader(socket) {
  let buffer = "";
  let current = [];
  const responses = [];
  const waiters = [];
  const settle = (error) => {
    while (waiters.length && (error || responses.length)) {
      const waiter = waiters.shift();
      if (error) waiter.reject(error);
      else waiter.resolve(responses.shift());
    }
  };
  const onData = (chunk) => {
    buffer += chunk.toString("utf8");
    let newline;
    while ((newline = buffer.indexOf("\n")) >= 0) {
      const line = buffer.slice(0, newline + 1).trimEnd();
      buffer = buffer.slice(newline + 1);
      current.push(line);
      if (/^\d{3} /.test(line)) {
        responses.push(current.join("\n"));
        current = [];
        settle();
      }
    }
  };
  const onError = (error) => settle(error);
  const onClose = () => settle(new Error("SMTP connection closed"));
  socket.on("data", onData);
  socket.on("error", onError);
  socket.on("close", onClose);
  return {
    next() {
      if (responses.length) return Promise.resolve(responses.shift());
      return new Promise((resolve, reject) => waiters.push({ resolve, reject }));
    },
    stop() {
      socket.off("data", onData);
      socket.off("error", onError);
      socket.off("close", onClose);
    },
  };
}

function assertSmtpResponse(response, codes) {
  const code = Number(String(response || "").slice(0, 3));
  if (!codes.includes(code)) throw new Error(`SMTP ${code || "unknown"}: ${String(response || "").slice(0, 240)}`);
}

async function connectSocket(socket, eventName) {
  await new Promise((resolve, reject) => {
    const timer = setTimeout(() => reject(new Error("SMTP connection timeout")), 10000);
    const cleanup = () => {
      clearTimeout(timer);
      socket.off(eventName, onConnected);
      socket.off("error", onError);
    };
    const onConnected = () => { cleanup(); resolve(); };
    const onError = (error) => { cleanup(); reject(error); };
    socket.once(eventName, onConnected);
    socket.once("error", onError);
  });
  socket.setTimeout(15000, () => socket.destroy(new Error("SMTP socket timeout")));
}

async function sendSmtpAlert(subject, message) {
  const host = String(process.env.RAGV6_ALERT_SMTP_HOST || "");
  const port = Number(process.env.RAGV6_ALERT_SMTP_PORT || 465);
  const user = String(process.env.RAGV6_ALERT_SMTP_USER || "");
  const password = String(process.env.RAGV6_ALERT_SMTP_PASS || "");
  const from = String(process.env.RAGV6_ALERT_EMAIL_FROM || user || "ragv6-monitor@localhost");
  const recipients = String(process.env.RAGV6_ALERT_EMAIL_TO || "").split(",").map((value) => value.trim()).filter(Boolean);
  if (!host || !recipients.length) throw new Error("SMTP host or recipient is not configured");
  const configuredMode = String(process.env.RAGV6_ALERT_SMTP_SECURE || "").toLowerCase();
  const mode = ["true", "tls", "ssl", "1"].includes(configuredMode)
    ? "tls"
    : (["starttls", "upgrade"].includes(configuredMode) || (!configuredMode && port === 587) ? "starttls" : "plain");
  let socket = mode === "tls"
    ? tls.connect({ host, port, servername: host, rejectUnauthorized: true })
    : net.connect({ host, port });
  await connectSocket(socket, mode === "tls" ? "secureConnect" : "connect");
  let reader = smtpResponseReader(socket);
  const send = async (line, codes) => {
    socket.write(`${line}\r\n`);
    const response = await reader.next();
    assertSmtpResponse(response, codes);
    return response;
  };
  try {
    assertSmtpResponse(await reader.next(), [220]);
    await send(`EHLO ${String(process.env.RAGV6_ALERT_SMTP_HELO || "ragv6-monitor.local")}`, [250]);
    if (mode === "starttls") {
      await send("STARTTLS", [220]);
      reader.stop();
      socket = tls.connect({ socket, servername: host, rejectUnauthorized: true });
      await connectSocket(socket, "secureConnect");
      reader = smtpResponseReader(socket);
      await send(`EHLO ${String(process.env.RAGV6_ALERT_SMTP_HELO || "ragv6-monitor.local")}`, [250]);
    }
    if (user) {
      await send("AUTH LOGIN", [334]);
      await send(Buffer.from(user).toString("base64"), [334]);
      await send(Buffer.from(password).toString("base64"), [235]);
    }
    await send(`MAIL FROM:<${from}>`, [250]);
    for (const recipient of recipients) await send(`RCPT TO:<${recipient}>`, [250, 251]);
    await send("DATA", [354]);
    const safeMessage = String(message).replace(/\r?\n/g, "\r\n").replace(/^\./gm, "..");
    const encodedSubject = `=?UTF-8?B?${Buffer.from(subject).toString("base64")}?=`;
    socket.write([
      `From: ${from}`,
      `To: ${recipients.join(", ")}`,
      `Subject: ${encodedSubject}`,
      `Date: ${new Date().toUTCString()}`,
      "MIME-Version: 1.0",
      "Content-Type: text/plain; charset=UTF-8",
      "Content-Transfer-Encoding: 8bit",
      "",
      safeMessage,
      ".",
      "",
    ].join("\r\n"));
    assertSmtpResponse(await reader.next(), [250]);
    await send("QUIT", [221]);
  } finally {
    reader.stop();
    socket.end();
  }
}

async function postAlertWebhook(url, payload) {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), 10000);
  try {
    const response = await fetch(url, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
      signal: controller.signal,
    });
    if (!response.ok) throw new Error(`HTTP ${response.status}: ${(await response.text()).slice(0, 200)}`);
  } finally {
    clearTimeout(timer);
  }
}

async function dispatchOperationalAlert(alertKey, severity, message) {
  const channels = alertChannelConfiguration();
  const configured = Object.entries(channels).filter(([, value]) => value.configured).map(([name]) => name);
  const delivery = [];
  const subject = `[RAGV6 ${severity.toUpperCase()}] ${alertKey}`;
  if (channels.email.configured) {
    try {
      await sendSmtpAlert(subject, message);
      delivery.push({ channel: "email", ok: true });
    } catch (error) {
      delivery.push({ channel: "email", ok: false, error: error.message.slice(0, 240) });
    }
  }
  if (channels.wecom.configured) {
    try {
      await postAlertWebhook(process.env.RAGV6_ALERT_WECOM_WEBHOOK, { msgtype: "text", text: { content: `${subject}\n${message}` } });
      delivery.push({ channel: "wecom", ok: true });
    } catch (error) {
      delivery.push({ channel: "wecom", ok: false, error: error.message.slice(0, 240) });
    }
  }
  if (channels.feishu.configured) {
    try {
      await postAlertWebhook(process.env.RAGV6_ALERT_FEISHU_WEBHOOK, { msg_type: "text", content: { text: `${subject}\n${message}` } });
      delivery.push({ channel: "feishu", ok: true });
    } catch (error) {
      delivery.push({ channel: "feishu", ok: false, error: error.message.slice(0, 240) });
    }
  }
  if (!configured.length) delivery.push({ channel: "none", ok: false, error: "告警通道尚未配置" });
  monitorDb.prepare(`
    INSERT INTO monitor_alerts(alert_key, severity, message, channels_json, delivery_json, created_at)
    VALUES (?, ?, ?, ?, ?, ?)
  `).run(alertKey, severity, message, JSON.stringify(configured), JSON.stringify(delivery), Date.now());
  writeServiceLog(
    delivery.some((row) => row.ok) ? "WARN" : "ERROR",
    `alert key=${alertKey} channels=${configured.join(",") || "none"} message=${message.replace(/\s+/g, " ").slice(0, 400)}`,
  );
}

async function maybeDispatchAlert(alertKey, severity, message) {
  const cooldown = Math.max(60000, Number(process.env.RAGV6_ALERT_COOLDOWN_MS || 600000));
  const latest = monitorDb.prepare(`
    SELECT created_at FROM monitor_alerts WHERE alert_key = ? ORDER BY created_at DESC LIMIT 1
  `).get(alertKey);
  if (latest && Date.now() - Number(latest.created_at) < cooldown) return;
  await dispatchOperationalAlert(alertKey, severity, message);
}

async function checkOperationalAlerts() {
  const since = Date.now() - 5 * 60 * 1000;
  const minimumSamples = Math.max(1, Number(process.env.RAGV6_ALERT_MIN_SAMPLES || 5));
  const recent = monitorDb.prepare(`
    SELECT category, status_code, duration_ms, model_failed, rag_no_evidence, client_type
    FROM monitor_requests WHERE completed_at >= ?
  `).all(since);
  const operationalRows = recent.filter((row) => row.client_type !== "qa");
  const chats = operationalRows.filter((row) => row.category === "chat");
  const health = await backendHealth();
  monitorState.backendFailureStreak = health.reachable ? 0 : monitorState.backendFailureStreak + 1;
  if (monitorState.backendFailureStreak >= 2) {
    await maybeDispatchAlert("rag_backend_unreachable", "critical", `RAG 检索服务连续 ${monitorState.backendFailureStreak} 次健康检查失败：${health.error || health.status || "unknown"}`);
  }
  if (operationalRows.length >= minimumSamples) {
    const errors = operationalRows.filter((row) => Number(row.status_code) >= 400).length;
    const errorRate = errors / operationalRows.length * 100;
    const threshold = Number(process.env.RAGV6_ALERT_ERROR_RATE || 20);
    if (errorRate >= threshold) {
      await maybeDispatchAlert("http_error_rate", "warning", `最近 5 分钟 HTTP 错误率 ${errorRate.toFixed(2)}%，超过阈值 ${threshold}%；样本 ${operationalRows.length}。`);
    }
    const p95 = percentile(operationalRows.map((row) => Number(row.duration_ms || 0)), 0.95);
    const p95Threshold = Number(process.env.RAGV6_ALERT_P95_MS || 60000);
    if (p95 >= p95Threshold) {
      await maybeDispatchAlert("p95_latency", "warning", `最近 5 分钟 P95 响应时间 ${p95} ms，超过阈值 ${p95Threshold} ms；样本 ${operationalRows.length}。`);
    }
  }
  if (chats.length >= minimumSamples) {
    const modelFailureRate = chats.reduce((sum, row) => sum + Number(row.model_failed || 0), 0) / chats.length * 100;
    const modelThreshold = Number(process.env.RAGV6_ALERT_MODEL_FAILURE_RATE || 10);
    if (modelFailureRate >= modelThreshold) {
      await maybeDispatchAlert("model_failure_rate", "critical", `最近 5 分钟模型失败率 ${modelFailureRate.toFixed(2)}%，超过阈值 ${modelThreshold}%；客服请求 ${chats.length}。`);
    }
    const noEvidenceRate = chats.reduce((sum, row) => sum + Number(row.rag_no_evidence || 0), 0) / chats.length * 100;
    const evidenceThreshold = Number(process.env.RAGV6_ALERT_NO_EVIDENCE_RATE || 40);
    if (noEvidenceRate >= evidenceThreshold) {
      await maybeDispatchAlert("rag_no_evidence_rate", "warning", `最近 5 分钟 RAG 无证据率 ${noEvidenceRate.toFixed(2)}%，超过阈值 ${evidenceThreshold}%；客服请求 ${chats.length}。`);
    }
  }
}

function applyMonitorRetention() {
  const cutoff = Date.now() - MONITOR_RETENTION_DAYS * 86400000;
  monitorDb.prepare("DELETE FROM monitor_requests WHERE completed_at < ?").run(cutoff);
  monitorDb.prepare("DELETE FROM monitor_feedback WHERE created_at < ?").run(cutoff);
  monitorDb.prepare("DELETE FROM monitor_alerts WHERE created_at < ?").run(cutoff);
}

function commonHeaders(extra = {}) {
  return {
    "Access-Control-Allow-Origin": "*",
    "Access-Control-Allow-Headers": "Content-Type, Authorization, X-Request-Id",
    "Access-Control-Allow-Methods": "GET, HEAD, POST, DELETE, OPTIONS",
    ...extra,
  };
}

function sendBuffer(res, status, body, contentType, extra = {}) {
  res.writeHead(status, commonHeaders({
    "Content-Type": contentType,
    "Content-Length": body.length,
    ...extra,
  }));
  res.end(body);
}

function sendJson(res, status, payload, extra = {}) {
  sendBuffer(res, status, Buffer.from(JSON.stringify(payload)), "application/json; charset=utf-8", extra);
}

function sendText(res, status, value, contentType = "text/plain; charset=utf-8", extra = {}) {
  sendBuffer(res, status, Buffer.from(String(value)), contentType, extra);
}

function redirect(res, location, status = 302) {
  res.writeHead(status, commonHeaders({ Location: location, "Content-Length": "0" }));
  res.end();
}

async function sendFile(req, res, filePath, cacheControl = "no-store") {
  try {
    const stat = await fsp.stat(filePath);
    if (!stat.isFile()) return sendText(res, 404, "Not Found");
    const type = MIME[path.extname(filePath).toLowerCase()] || "application/octet-stream";
    const range = String(req.headers.range || "");
    if (range) {
      const match = range.match(/^bytes=(\d*)-(\d*)$/);
      if (match) {
        const start = match[1] ? Number(match[1]) : 0;
        const end = match[2] ? Number(match[2]) : stat.size - 1;
        if (start >= 0 && end >= start && end < stat.size) {
          res.writeHead(206, commonHeaders({
            "Content-Type": type,
            "Content-Length": end - start + 1,
            "Content-Range": `bytes ${start}-${end}/${stat.size}`,
            "Accept-Ranges": "bytes",
            "Cache-Control": cacheControl,
          }));
          if (req.method === "HEAD") return res.end();
          return fs.createReadStream(filePath, { start, end }).pipe(res);
        }
      }
    }
    const acceptsGzip = req.method === "GET"
      && /(?:^|,)\s*gzip\s*(?:,|$)/i.test(String(req.headers["accept-encoding"] || ""))
      && /^(?:text\/|application\/(?:javascript|json))/i.test(type)
      && stat.size >= 1024;
    if (acceptsGzip) {
      res.writeHead(200, commonHeaders({
        "Content-Type": type,
        "Content-Encoding": "gzip",
        Vary: "Accept-Encoding",
        "Cache-Control": cacheControl,
      }));
      return fs.createReadStream(filePath)
        .pipe(zlib.createGzip({ level: 4 }))
        .pipe(res);
    }
    res.writeHead(200, commonHeaders({
      "Content-Type": type,
      "Content-Length": stat.size,
      "Accept-Ranges": "bytes",
      "Cache-Control": cacheControl,
    }));
    if (req.method === "HEAD") return res.end();
    fs.createReadStream(filePath).pipe(res);
  } catch (error) {
    if (error && error.code === "ENOENT") return sendText(res, 404, "Not Found");
    throw error;
  }
}

async function readBody(req, maxBytes = 20 * 1024 * 1024) {
  const chunks = [];
  let size = 0;
  for await (const chunk of req) {
    size += chunk.length;
    if (size > maxBytes) throw new Error("request body too large");
    chunks.push(chunk);
  }
  return Buffer.concat(chunks);
}

function safeFileStem(value) {
  const stem = String(value || "").trim().replace(/[\\/:*?"<>|]/g, "_").replace(/\s+/g, " ");
  if (!stem || stem === "." || stem === "..") return "";
  return stem.slice(0, 96);
}

async function readJson(filePath, fallback = null) {
  try {
    return JSON.parse(await fsp.readFile(filePath, "utf8"));
  } catch (error) {
    if (error && error.code === "ENOENT") return fallback;
    throw error;
  }
}

async function readJsonRequest(req) {
  try {
    return JSON.parse((await readBody(req)).toString("utf8") || "{}");
  } catch {
    const error = new Error("invalid JSON");
    error.statusCode = 400;
    throw error;
  }
}

function wecomConfigured() {
  return Boolean(WECOM.enabled && WECOM.token && WECOM.encodingAesKey && WECOM.corpId);
}

function wecomMissingConfig() {
  const missing = [];
  if (!WECOM.enabled) missing.push("WECOM_ENABLED");
  if (!WECOM.token) missing.push("WECOM_TOKEN");
  if (!WECOM.encodingAesKey) missing.push("WECOM_ENCODING_AES_KEY");
  if (!WECOM.corpId) missing.push("WECOM_CORP_ID");
  if (!WECOM.agentId) missing.push("WECOM_AGENT_ID");
  if (!WECOM.secret) missing.push("WECOM_SECRET");
  return missing;
}

function wecomSendAuthorized(req) {
  if (!WECOM.sendApiKey) return false;
  const header = String(req.headers.authorization || "");
  const supplied = header.startsWith("Bearer ") ? header.slice(7) : String(req.headers["x-raysource-api-key"] || "");
  const left = Buffer.from(supplied);
  const right = Buffer.from(WECOM.sendApiKey);
  return left.length === right.length && crypto.timingSafeEqual(left, right);
}

function wecomAesKey() {
  const value = WECOM.encodingAesKey.endsWith("=") ? WECOM.encodingAesKey : `${WECOM.encodingAesKey}=`;
  return Buffer.from(value, "base64");
}

function wecomSignature(timestamp, nonce, encrypted) {
  return crypto.createHash("sha1").update([WECOM.token, timestamp, nonce, encrypted].sort().join(""), "utf8").digest("hex");
}

function wecomDecrypt(encrypted) {
  const decipher = crypto.createDecipheriv("aes-256-cbc", wecomAesKey(), wecomAesKey().subarray(0, 16));
  decipher.setAutoPadding(false);
  const padded = Buffer.concat([decipher.update(Buffer.from(encrypted, "base64")), decipher.final()]);
  const pad = padded[padded.length - 1];
  const plain = pad > 0 && pad <= 32 ? padded.subarray(0, padded.length - pad) : padded;
  const xml = plain.subarray(16);
  const length = xml.readUInt32BE(0);
  return { message: xml.subarray(4, 4 + length).toString("utf8"), corpId: xml.subarray(4 + length).toString("utf8") };
}

function wecomEncrypt(message) {
  const random = crypto.randomBytes(16);
  const body = Buffer.from(String(message), "utf8");
  const corp = Buffer.from(WECOM.corpId, "utf8");
  const raw = Buffer.concat([random, Buffer.from([(body.length >>> 24) & 255, (body.length >>> 16) & 255, (body.length >>> 8) & 255, body.length & 255]), body, corp]);
  const pad = 32 - (raw.length % 32);
  const padded = Buffer.concat([raw, Buffer.alloc(pad, pad)]);
  const cipher = crypto.createCipheriv("aes-256-cbc", wecomAesKey(), wecomAesKey().subarray(0, 16));
  cipher.setAutoPadding(false);
  return Buffer.concat([cipher.update(padded), cipher.final()]).toString("base64");
}

function wecomXmlValue(xml, tag) {
  const match = String(xml || "").match(new RegExp(`<${tag}><!\\[CDATA\\[(.*?)\\]\\]><\\/${tag}>`, "s"))
    || String(xml || "").match(new RegExp(`<${tag}>(.*?)<\\/${tag}>`, "s"));
  return match ? match[1] : "";
}

function wecomXmlResponse(encrypted, timestamp, nonce) {
  return `<xml><Encrypt><![CDATA[${encrypted}]]></Encrypt><MsgSignature><![CDATA[${wecomSignature(timestamp, nonce, encrypted)}]]></MsgSignature><TimeStamp>${timestamp}</TimeStamp><Nonce><![CDATA[${nonce}]]></Nonce></xml>`;
}

function wecomSignatureMatches(actual, expected) {
  const left = Buffer.from(String(actual));
  const right = Buffer.from(String(expected));
  return left.length === right.length && crypto.timingSafeEqual(left, right);
}

function parseSseDonePayload(raw) {
  let event = "";
  for (const line of String(raw || "").split(/\r?\n/)) {
    if (line.startsWith("event:")) event = line.slice(6).trim();
    if (event === "done" && line.startsWith("data:")) {
      try { return JSON.parse(line.slice(5).trim()); } catch { return null; }
    }
  }
  return null;
}

async function invokeUnifiedChannelChat({ channel, question, sessionId, useHistoryContext = false }) {
  const response = await fetch(`http://127.0.0.1:${CONFIG.port}/ragv6-api/chat`, {
    method: "POST",
    headers: {
      "content-type": "application/json",
      "x-rag-channel": channel,
      "x-request-id": `${channel}-${crypto.randomUUID()}`,
    },
    body: JSON.stringify({
      question,
      session_id: sessionId,
      images: [],
      use_history_context: Boolean(useHistoryContext),
      history_context: "",
    }),
  });
  const raw = await response.text();
  const done = parseSseDonePayload(raw);
  if (!response.ok || !done?.answer) throw new Error(`unified chat failed (${response.status})`);
  return done;
}

async function sendWecomText(toUser, content) {
  const tokenResponse = await fetch(`https://qyapi.weixin.qq.com/cgi-bin/gettoken?corpid=${encodeURIComponent(WECOM.corpId)}&corpsecret=${encodeURIComponent(WECOM.secret)}`);
  const tokenData = await tokenResponse.json();
  if (!tokenResponse.ok || tokenData.errcode) throw new Error("wecom access token failed");
  const sendResponse = await fetch(`https://qyapi.weixin.qq.com/cgi-bin/message/send?access_token=${encodeURIComponent(tokenData.access_token)}`, {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify({ touser: toUser, msgtype: "text", agentid: Number(WECOM.agentId), text: { content } }),
  });
  const result = await sendResponse.json();
  if (!sendResponse.ok || result.errcode) throw new Error(`wecom send failed (${result.errcode || sendResponse.status})`);
  return result;
}

async function wecomCallback(req, res, url) {
  if (!WECOM.enabled) return sendJson(res, 404, { code: 404, msg: "企业微信接口未启用" });
  if (!wecomConfigured()) return sendJson(res, 503, { code: 503, msg: "企业微信接口未完成服务端配置" });
  const timestamp = String(url.searchParams.get("timestamp") || "");
  const nonce = String(url.searchParams.get("nonce") || "");
  const signature = String(url.searchParams.get("msg_signature") || "");
  if (!timestamp || !nonce || !signature) return sendText(res, 400, "missing wecom signature parameters");
  if (req.method === "GET") {
    const echostr = String(url.searchParams.get("echostr") || "");
    if (!echostr || !wecomSignatureMatches(signature, wecomSignature(timestamp, nonce, echostr))) return sendText(res, 401, "invalid signature");
    const decrypted = wecomDecrypt(echostr);
    return sendText(res, 200, decrypted.message);
  }
  const raw = (await readBody(req, 256 * 1024)).toString("utf8");
  const encrypted = wecomXmlValue(raw, "Encrypt");
  if (!encrypted || signature !== wecomSignature(timestamp, nonce, encrypted)) return sendText(res, 401, "invalid signature");
  const decrypted = wecomDecrypt(encrypted);
  const fromUser = wecomXmlValue(decrypted.message, "FromUserName");
  const msgType = wecomXmlValue(decrypted.message, "MsgType");
  const content = wecomXmlValue(decrypted.message, "Content").trim();
  writeServiceLog("INFO", `wecom callback from=${fromUser} msg_type=${msgType}`);
  if (msgType === "text" && fromUser && content && WECOM.agentId && WECOM.secret) {
    setImmediate(async () => {
      try {
        const done = await invokeUnifiedChannelChat({
          channel: "wecom",
          question: content,
          sessionId: `wecom:${fromUser}`,
          useHistoryContext: true,
        });
        await sendWecomText(fromUser, done.answer);
      } catch (error) {
        writeServiceLog("ERROR", `wecom async chat failed from=${fromUser} error=${error.message}`);
      }
    });
  }
  // Acknowledge immediately so WeCom does not retry while RAG is running.
  return sendText(res, 200, "success");
}

async function wecomSendMessage(req, res) {
  if (!WECOM.enabled) return sendJson(res, 404, { code: 404, msg: "企业微信接口未启用" });
  if (!wecomSendAuthorized(req)) return sendJson(res, 401, { code: 401, msg: "企业微信主动发送接口需要服务端 API Key" });
  if (!WECOM.enabled) return sendJson(res, 404, { code: 404, msg: "企业微信接口未启用" });
  if (!WECOM.corpId || !WECOM.agentId || !WECOM.secret) return sendJson(res, 503, { code: 503, msg: "企业微信主动发送参数未配置" });
  const body = await readJsonRequest(req);
  const toUser = String(body.to_user || body.touser || "").trim();
  const content = String(body.content || "").trim();
  if (!toUser || !content) return sendJson(res, 400, { code: 400, msg: "to_user 和 content 为必填项" });
  const tokenResponse = await fetch(`https://qyapi.weixin.qq.com/cgi-bin/gettoken?corpid=${encodeURIComponent(WECOM.corpId)}&corpsecret=${encodeURIComponent(WECOM.secret)}`);
  const tokenData = await tokenResponse.json();
  if (!tokenResponse.ok || tokenData.errcode) return sendJson(res, 502, { code: 502, msg: "获取企业微信 access_token 失败", wecom: tokenData });
  const sendResponse = await fetch(`https://qyapi.weixin.qq.com/cgi-bin/message/send?access_token=${encodeURIComponent(tokenData.access_token)}`, {
    method: "POST", headers: { "content-type": "application/json" },
    body: JSON.stringify({ touser: toUser, msgtype: "text", agentid: Number(WECOM.agentId), text: { content } }),
  });
  const result = await sendResponse.json();
  return sendJson(res, sendResponse.ok && !result.errcode ? 200 : 502, { code: sendResponse.ok && !result.errcode ? 0 : 502, msg: result.errmsg || "success", data: result });
}

async function listChunkManuals() {
  const entries = await fsp.readdir(MANUAL_SECTIONS_DIR, { withFileTypes: true });
  const chunks = await readJson(RETRIEVAL_CHUNKS_FILE, []);
  const chunkRows = Array.isArray(chunks) ? chunks : (chunks?.retrieval_chunks || []);
  const countByManual = new Map();
  for (const row of chunkRows) {
    const manual = String(row?.product || "").trim();
    if (manual) countByManual.set(manual, (countByManual.get(manual) || 0) + 1);
  }
  const manuals = [];
  for (const entry of entries) {
    if (!entry.isFile() || path.extname(entry.name).toLowerCase() !== ".json") continue;
    const filePath = path.join(MANUAL_SECTIONS_DIR, entry.name);
    const doc = await readJson(filePath, null);
    if (!doc || !Array.isArray(doc.sections)) continue;
    const manual = String(doc.manual || path.basename(entry.name, ".json"));
    manuals.push({
      manual,
      file: entry.name,
      section_count: doc.sections.length,
      retrieval_chunk_count: countByManual.get(manual) || 0,
      updated_at: (await fsp.stat(filePath)).mtime.toISOString(),
    });
  }
  return manuals.sort((a, b) => a.manual.localeCompare(b.manual, "zh-CN"));
}

async function resolveChunkManual(manual) {
  const wanted = String(manual || "").trim();
  if (!wanted) return null;
  const match = (await listChunkManuals()).find((item) => item.manual === wanted);
  return match || null;
}

async function backupChunkFile(filePath, label) {
  await fsp.mkdir(CHUNK_BACKUP_DIR, { recursive: true });
  const stamp = new Date().toISOString().replace(/[:.]/g, "-");
  const base = path.basename(filePath, path.extname(filePath));
  const destination = path.join(CHUNK_BACKUP_DIR, `${base}.${label}.${stamp}${path.extname(filePath)}`);
  try { await fsp.copyFile(filePath, destination); } catch (error) { if (error.code !== "ENOENT") throw error; }
  return destination;
}

function smartSplitManual(text, options = {}) {
  const target = Math.max(240, Math.min(2200, Number(options.target_chars) || 800));
  const overlap = Math.max(0, Math.min(300, Number(options.overlap_chars) || 100));
  const clean = String(text || "").replace(/\r\n?/g, "\n").replace(/\u0000/g, "").trim();
  if (!clean) return [];
  const lines = clean.split("\n");
  const isHeading = (line) => {
    const value = line.trim();
    return /^#{1,6}\s+/.test(value)
      || /^(?:第[一二三四五六七八九十百\d]+[章节部分]|\d+(?:\.\d+){0,4}[、.．\s])/.test(value)
      || (value.length >= 2 && value.length <= 48 && !/[。；;，,：:]$/.test(value));
  };
  const sections = [];
  let heading = "未命名章节";
  let documentTitle = "";
  let buffer = [];
  let startLine = 1;
  const flush = (endLine) => {
    const value = buffer.join("\n").replace(/\n{3,}/g, "\n\n").trim();
    if (!value) return;
    sections.push({ heading, text: value, start_line: startLine, end_line: endLine });
    buffer = [];
  };
  lines.forEach((line, index) => {
    if (isHeading(line)) {
      if (buffer.join("\n").trim()) flush(index);
      const value = line.trim().replace(/^#{1,6}\s*/, "");
      const markdownLevel = line.match(/^(#{1,6})\s+/)?.[1].length || 2;
      if (markdownLevel === 1) {
        documentTitle = value;
        heading = value;
      } else {
        heading = documentTitle ? `${documentTitle} / ${value}` : value;
      }
      startLine = index + 1;
      return;
    }
    buffer.push(line);
  });
  flush(lines.length);

  const chunks = [];
  for (const section of sections) {
    const paragraphs = section.text.split(/\n\s*\n/).map((item) => item.trim()).filter(Boolean);
    let current = "";
    let lineOffset = section.start_line;
    for (const paragraph of paragraphs) {
      const joined = current ? `${current}\n\n${paragraph}` : paragraph;
      if (current && joined.length > target) {
        chunks.push({ ...section, text: current, start_line: lineOffset, end_line: lineOffset + current.split("\n").length - 1 });
        const tail = overlap ? current.slice(-overlap).replace(/^\S*\s*/, "").trim() : "";
        current = tail ? `${tail}\n\n${paragraph}` : paragraph;
        lineOffset += Math.max(1, current.split("\n").length - 1);
      } else current = joined;
    }
    if (current) chunks.push({ ...section, text: current, start_line: lineOffset, end_line: section.end_line });
  }
  return chunks.map((chunk, index) => ({
    id: `draft-${String(index + 1).padStart(3, "0")}`,
    heading: chunk.heading,
    text: chunk.text,
    char_len: chunk.text.length,
    start_line: chunk.start_line,
    end_line: chunk.end_line,
    quality: chunk.text.length < 140 ? "short" : (chunk.text.length > target * 1.4 ? "long" : "ready"),
  }));
}

async function handleChunkManagerApi(req, res, pathname, url) {
  const prefix = "/ragv6-api/chunks";
  let suffix = pathname.slice(prefix.length);
  let upstreamPath = "";
  let body = null;

  if (req.method === "GET" && suffix === "/manuals") {
    upstreamPath = "/admin/chunks/manuals";
  } else if (req.method === "GET" && suffix === "/status") {
    upstreamPath = "/admin/chunks/status";
  } else if (req.method === "GET" && suffix === "/backups") {
    upstreamPath = "/admin/chunks/backups";
  } else if (req.method === "GET" && suffix.startsWith("/jobs/")) {
    upstreamPath = `/admin/chunks/jobs/${encodeURIComponent(suffix.slice("/jobs/".length))}`;
  } else if (req.method === "GET" && suffix.startsWith("/manual/")) {
    upstreamPath = `/admin/chunks/manual/${encodeURIComponent(decodeURIComponent(suffix.slice("/manual/".length)))}`;
  } else if (req.method === "GET" && suffix === "" && url.searchParams.get("manual")) {
    upstreamPath = `/admin/chunks/manual/${encodeURIComponent(String(url.searchParams.get("manual")))}`;
  } else if (req.method === "POST" && ["/preview", "/publish", "/rebuild", "/rollback", "/search-test"].includes(suffix)) {
    upstreamPath = `/admin/chunks${suffix}`;
    body = await readBody(req, 36 * 1024 * 1024);
  } else if (req.method === "POST" && suffix === "/split") {
    const legacy = await readJsonRequest(req);
    body = Buffer.from(JSON.stringify({
      manual: String(legacy.manual || "切分预览"),
      filename: String(legacy.filename || "manual.md"),
      text: String(legacy.text || ""),
      options: legacy.options || {},
    }));
    upstreamPath = "/admin/chunks/preview";
  } else if (req.method === "POST" && suffix === "/import") {
    const legacy = await readJsonRequest(req);
    body = Buffer.from(JSON.stringify({
      manual: String(legacy.manual || ""),
      filename: String(legacy.filename || `${legacy.manual || "manual"}.md`),
      text: String(legacy.source || legacy.text || ""),
      options: legacy.options || {},
      replace_existing: Boolean(legacy.replace_existing),
      rebuild_index: Boolean(legacy.rebuild_index),
    }));
    upstreamPath = "/admin/chunks/publish";
  } else if (req.method === "POST" && suffix === "/save") {
    return sendJson(res, 409, {
      code: 409,
      msg: "为防止 section_chunks、retrieval_chunks 与 FAISS 不一致，已禁用单文件直写；请从源手册重新预览并发布。",
      data: null,
    });
  } else {
    return false;
  }

  let upstream;
  try {
    upstream = await backendFetch(upstreamPath, {
      method: req.method,
      headers: body ? { "Content-Type": "application/json" } : {},
      body,
    });
  } catch (error) {
    return sendJson(res, 502, {
      code: 502,
      msg: `chunk service unavailable: ${error.message}`,
      data: null,
    });
  }
  const payload = Buffer.from(await upstream.arrayBuffer());
  return sendBuffer(
    res,
    upstream.status,
    payload,
    upstream.headers.get("content-type") || "application/json; charset=utf-8",
  );
}

async function handleChunkSqlApi(req, res, pathname) {
  const prefix = "/ragv6-api/chunk-sql";
  const suffix = pathname.slice(prefix.length);
  const allowed = new Set(["/status", "/sync", "/plan"]);
  if (!allowed.has(suffix)) return false;
  if (suffix === "/status" && req.method !== "GET") return false;
  if (suffix !== "/status" && req.method !== "POST") return false;
  const body = req.method === "POST" ? await readBody(req, 256 * 1024) : null;
  let upstream;
  try {
    upstream = await backendFetch(`/admin/chunk-sql${suffix}`, {
      method: req.method,
      headers: body ? { "Content-Type": "application/json" } : {},
      body,
    });
  } catch (error) {
    return sendJson(res, 502, {
      code: 502,
      msg: `chunk SQL service unavailable: ${error.message}`,
      data: null,
    });
  }
  const payload = Buffer.from(await upstream.arrayBuffer());
  return sendBuffer(
    res,
    upstream.status,
    payload,
    upstream.headers.get("content-type") || "application/json; charset=utf-8",
  );
}

function writeSse(res, event, data) {
  res.write(`event: ${event}\ndata: ${JSON.stringify(data)}\n\n`);
}

function publicChatProgress(stage) {
  const value = String(stage || "").trim().toLowerCase();
  if (value === "accepted" || value === "start" || value === "analyze" || value === "input") {
    return { stage: "analyze", message: "正在分析问题" };
  }
  if (value === "route" || value === "classify" || value === "classification" || value === "scope") {
    return { stage: "scope", message: "正在确认产品与问题范围" };
  }
  if (value === "retrieve" || value === "search" || value === "tool" || value === "knowledge" || value === "rerank" || value === "evidence") {
    return { stage: "knowledge", message: "正在匹配相关手册资料" };
  }
  if (value === "model" || value === "generate" || value === "finalize" || value === "compose") {
    return { stage: "compose", message: "正在整理并核对答案" };
  }
  if (value === "done" || value === "complete") {
    return { stage: "complete", message: "答案整理完成" };
  }
  return { stage: "analyze", message: "正在处理问题" };
}

function recordProgress(requestId, stage, message, startedAt) {
  if (!requestId) return;
  const row = progressStore.get(requestId) || { created: startedAt || Date.now(), events: [] };
  row.events.push({
    t: Date.now() / 1000,
    stage: String(stage || "info"),
    message: String(message || ""),
    elapsed: Math.round((Date.now() - row.created) / 100) / 10,
  });
  row.updated = Date.now();
  progressStore.set(requestId, row);
  if (progressStore.size > 200) {
    [...progressStore.entries()]
      .sort((left, right) => left[1].updated - right[1].updated)
      .slice(0, 50)
      .forEach(([key]) => progressStore.delete(key));
  }
}

function stripImageList(answer) {
  return String(answer || "").replace(/,\s*\[(?:\s*"[^"]+"\s*,?)+\s*\]\s*$/, "").trim();
}

function programAnswerIntro(question, answer) {
  const text = String(answer || "").trim();
  const subject = String(question || "").replace(/\s+/g, " ").trim();
  if (!text || !subject) return text;
  const prefix = `关于“${subject}”的问题，回答如下：`;
  if (text.startsWith(prefix) || /^关于[“\"].*?[”\"]的问题，回答如下：/.test(text)) return text;
  return `${prefix}\n\n${text}`;
}

function resolveImageFile(imageId) {
  const stem = String(imageId || "").trim().replace(/\.(?:jpg|jpeg|png|webp|gif|bmp)$/i, "");
  for (const extension of [".jpg", ".jpeg", ".png", ".webp", ".gif", ".bmp"]) {
    if (fs.existsSync(path.join(IMAGE_DIR, `${stem}${extension}`))) return `${stem}${extension}`;
  }
  return `${stem}.jpg`;
}

function currentModelProfile() {
  const active = MODEL_PROFILES.find((profile) => profile.id === activeProfileId) || MODEL_PROFILES[2];
  return { code: 0, msg: "success", data: { active, profiles: MODEL_PROFILES } };
}

function hasQuestionMedia(question) {
  return /https?:\/\/[^\s"'<>]+\.(?:png|jpe?g|webp|gif|bmp)(?:[?#][^\s"'<>]*)?/i.test(String(question || ""));
}

async function serviceFetch(origin, route, options = {}) {
  const target = new URL(route, origin);
  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), CONFIG.requestTimeoutMs);
  try {
    return await fetch(target, {
      ...options,
      signal: controller.signal,
      headers: {
        Authorization: `Bearer ${CONFIG.apiToken}`,
        ...(options.headers || {}),
      },
    });
  } finally {
    clearTimeout(timeout);
  }
}

async function backendFetch(route, options = {}) {
  return serviceFetch(CONFIG.apiOrigin, route, options);
}

async function chatBackendFetch(route, options = {}, forceLegacy = false, chatBackend = currentChatBackend()) {
  const origin = forceLegacy ? CONFIG.apiOrigin : chatBackend.origin;
  return serviceFetch(origin, route, options);
}

async function handleChatInternal(req, res) {
  const raw = await readBody(req);
  let payload;
  try {
    payload = JSON.parse(raw.toString("utf8") || "{}");
  } catch {
    return sendJson(res, 400, { code: 400, msg: "invalid JSON", data: null });
  }
  const originalQuestion = String(payload.question || "").trim();
  if (!originalQuestion) return sendJson(res, 422, { code: 422, msg: "question is required", data: null });
  const requestId = req.monitorContext?.requestId || normalizeRequestId(req.headers["x-request-id"] || payload.request_id);
  const startedAt = Date.now();
  const requestedProfile = MODEL_PROFILES.find((profile) => profile.model === payload.model || profile.id === payload.model);
  const requestedModel = requestedProfile?.model || String(payload.model || "gpt-5.6-terra");
  // Do not alter the selector's labels, icons, availability, or descriptions.
  // It is an outward-facing shell; the actual generation route is unified so
  // every visible selection has identical Terra Medium behavior.
  const effectiveTextModel = UNIFIED_RUNTIME_MODEL;
  const images = Array.isArray(payload.images) ? payload.images : [];
  const sessionId = String(payload.session_id || `ragv6_session_${Date.now()}`);
  const memorySession = await getMemorySession(sessionId);
  const detected = explicitQuestionProduct(originalQuestion);
  // Product scope is determined from evidence in the current question only.
  // A selected product in the UI and a product from a prior turn are display /
  // conversation state, not facts about this request.  Let the retrieval
  // service perform corpus-wide routing whenever the current question does
  // not itself identify a product.
  const resolvedProduct = detected.ambiguous ? "" : (detected.product || "");
  const requestedHistory = Boolean(payload.use_history_context);
  // A product inferred from a previous turn is allowed only when the caller
  // explicitly opts into history. With the switch off, every request remains
  // a standalone corpus-wide retrieval exactly as before.
  const historicalProduct = requestedHistory && !detected.ambiguous
    ? String(memorySession.lastProduct || "").trim()
    : "";
  const serviceScope = !resolvedProduct && !historicalProduct && memorySession.lastMode === "customer"
    ? "__service__"
    : "";
  const memoryScope = resolvedProduct || historicalProduct || serviceScope;
  const contextProduct = resolvedProduct || historicalProduct;
  const clientHistoryProduct = String(payload.history_product || "").trim();
  const requestedMemoryEpoch = String(payload.memory_epoch || "base").trim();
  const memoryEpoch = /^[A-Za-z0-9_-]{1,100}$/.test(requestedMemoryEpoch) ? requestedMemoryEpoch : "base";
  const memoryBucketKey = `${memoryScope}\u0000${memoryEpoch}`;
  const clientHistory = compactMemoryText(payload.history_context, PRODUCT_MEMORY_MAX_CHARS);
  const serverTurns = memoryScope ? (memorySession.buckets.get(memoryBucketKey) || []) : [];
  const historyContext = requestedHistory && memoryScope && !detected.ambiguous
    ? (clientHistoryProduct === contextProduct && clientHistory ? clientHistory : summarizeProductTurns(serverTurns))
    : "";
  // A newly attached image is independent product evidence. Do not force it
  // into the previous turn's product scope or reuse that scope's backend
  // session, otherwise an earlier generator/manual can override the picture.
  const imageIdentityRequest = images.length > 0;
  const contextPacket = !imageIdentityRequest && requestedHistory && memoryScope && !detected.ambiguous
    ? buildContextPacket({
        turns: serverTurns,
        product: contextProduct,
        currentQuestion: originalQuestion,
        clientPacket: payload.context_packet,
      })
    : {};
  const contextPacketUsed = Boolean(
    contextPacket.summary
    || Object.keys(contextPacket.entities || {}).length
    || (contextPacket.media_facts || []).length
    || (contextPacket.user_constraints || []).length
    || (contextPacket.recent_turns || []).length
  );
  const recommendedAnswer = findRecommendedAnswer(recommendedAnswerIndex, {
    question: originalQuestion,
    images,
    use_history_context: payload.use_history_context,
    history_context: payload.history_context,
    context_packet: payload.context_packet,
  });
  const recommendedQuestionKey = normalizeRecommendedQuestion(originalQuestion);
  const previousProduct = memorySession.lastProduct;
  const contextSwitched = Boolean(previousProduct && resolvedProduct && previousProduct !== resolvedProduct);
  const upstreamSessionId = imageIdentityRequest
    ? `${sessionId}:image:${requestId}`
    : scopedProductSessionId(sessionId, memoryScope, memoryEpoch, requestId);
  const upstreamPayload = {
    question: originalQuestion,
    model: effectiveTextModel,
    reasoning_effort: UNIFIED_RUNTIME_REASONING,
    images,
    // The retrieval service keeps its own in-memory history by session_id.
    // Product-scope that ID as well, otherwise A -> B -> A would leak B into A
    // even though the browser and gateway buckets were isolated.
    session_id: upstreamSessionId,
    forced_product: imageIdentityRequest ? null : contextProduct,
    use_history_context: imageIdentityRequest ? false : Boolean(historyContext || contextPacketUsed),
    history_context: imageIdentityRequest ? "" : historyContext,
    context_packet: contextPacket,
    history_product: imageIdentityRequest ? "" : contextProduct,
    stream: true,
  };
  writeServiceLog("INFO", `chat model wrapper request_id=${requestId} requested=${requestedModel} effective=${UNIFIED_RUNTIME_MODEL} reasoning=${UNIFIED_RUNTIME_REASONING}`);
  writeServiceLog(
    "INFO",
    `recommended answer lookup request_id=${requestId} hit=${Boolean(recommendedAnswer)} cache_size=${recommendedAnswerIndex.size} question_key=${crypto.createHash("sha256").update(recommendedQuestionKey).digest("hex").slice(0, 12)}`,
  );
  if (req.monitorContext) {
    req.monitorContext.requestId = requestId;
    req.monitorContext.category = "chat";
    req.monitorContext.product = resolvedProduct.slice(0, 160);
    req.monitorContext.model = upstreamPayload.model;
  }
  // Image requests now have a dedicated product-identification short circuit
  // on canonical 8014.  Do not send them to the legacy media backend.
  const forceLegacyChat = false;
  const chatBackend = currentChatBackend();
  const effectiveChatBackend = forceLegacyChat ? "legacy-media" : chatBackend.mode;
  writeServiceLog("INFO", `chat start request_id=${requestId} model=${upstreamPayload.model} image_count=${upstreamPayload.images.length} backend=${effectiveChatBackend}`);
  recordProgress(
    requestId,
    "input",
    "问题已接收",
    startedAt,
  );
  recordProgress(
    requestId,
    "policy",
    "对话范围设置已应用",
    startedAt,
  );
  recordProgress(
    requestId,
    "scope",
    "正在确认产品与问题范围",
    startedAt,
  );
  recordProgress(
    requestId,
    "model",
    "回答服务已准备",
    startedAt,
  );

  let upstream;
  try {
    upstream = await chatBackendFetch("/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json", "X-Request-Id": requestId },
      body: JSON.stringify(upstreamPayload),
    }, forceLegacyChat, chatBackend);
  } catch (error) {
    if (req.monitorContext) req.monitorContext.modelFailed = 1;
    writeServiceLog("ERROR", `chat upstream unavailable request_id=${requestId} error=${error.message}`);
    return sendJson(res, 502, { code: 502, msg: `retrieval service unavailable: ${error.message}`, data: null });
  }
  if (!upstream.ok || !upstream.body) {
    if (req.monitorContext) req.monitorContext.modelFailed = 1;
    const detail = await upstream.text();
    writeServiceLog("ERROR", `chat upstream rejected request_id=${requestId} status=${upstream.status} detail=${detail.replace(/\s+/g, " ").slice(0, 300)}`);
    return sendJson(res, upstream.status || 502, { code: upstream.status || 502, msg: detail.slice(0, 800), data: null });
  }
  const upstreamContentType = String(upstream.headers.get("content-type") || "").toLowerCase();
  if (!upstreamContentType.includes("text/event-stream")) {
    if (req.monitorContext) req.monitorContext.modelFailed = 1;
    await upstream.text().catch(() => "");
    writeServiceLog("ERROR", `chat upstream returned unexpected content-type request_id=${requestId} content_type=${upstreamContentType || "missing"}`);
    return sendJson(res, 502, {
      code: 502,
      msg: "retrieval service is restarting; please retry shortly",
      data: null,
    });
  }

  res.writeHead(200, commonHeaders({
    "Content-Type": "text/event-stream; charset=utf-8",
    "Cache-Control": "no-cache, no-transform",
    "X-Accel-Buffering": "no",
  }));
  res.flushHeaders?.();
  recordProgress(requestId, "connected", "知识服务已连接", startedAt);
  writeSse(res, "status", {
    stage: "analyze",
    message: "正在分析问题",
    request_id: requestId,
  });
  const publicStageRank = {
    analyze: 0,
    scope: 1,
    knowledge: 2,
    compose: 3,
    complete: 4,
  };
  let lastPublicStage = "analyze";
  let lastPublicRank = publicStageRank.analyze;
  const heartbeat = setInterval(() => {
    if (!res.writableEnded) res.write(`: heartbeat ${Date.now()}\n\n`);
  }, 15000);
  const decoder = new TextDecoder();
  let buffer = "";
  let latestAuditTrace = null;
  let cachedFallbackSent = false;
  const finishRecommendedAnswer = (reason) => {
    if (!recommendedAnswer || cachedFallbackSent || res.writableEnded) return;
    cachedFallbackSent = true;
    const cacheMode = recommendedAnswer.answerMode || "manual";
    const finalData = {
      request_id: requestId,
      answer: programAnswerIntro(originalQuestion, recommendedAnswer.answer),
      session_id: sessionId,
      timestamp: Math.floor(Date.now() / 1000),
      source: "recommended-answer-cache",
      answer_source: recommendedAnswer.source,
      recommended_cache_id: recommendedAnswer.cacheId,
      answer_mode: cacheMode,
      mode_label: cacheMode === "customer" ? "Customer service" : "V6 manual",
      routing: { mode: cacheMode, confidence: null, reason },
      product: contextProduct || resolvedProduct || "V6",
      question: originalQuestion,
      effective_question: originalQuestion,
      history_context: "",
      history_context_used: false,
      context_packet_version: null,
      context_retrieval_hint: "auto",
      context_product: contextProduct || resolvedProduct || "",
      context_turns: 0,
      context_switched: false,
      images: [],
      manuals: contextProduct ? [contextProduct] : [],
      sources: [],
      image_descriptions: [],
      elapsed: Math.round((Date.now() - startedAt) / 1000 * 1000) / 1000,
      retrieval_trace: hasMeaningfulAuditTrace(latestAuditTrace) ? latestAuditTrace : null,
    };
    if (req.monitorContext) {
      req.monitorContext.product = String(finalData.product).slice(0, 160);
      req.monitorContext.answerMode = cacheMode;
      req.monitorContext.ragNoEvidence = finalData.retrieval_trace ? 0 : 1;
    }
    recordProgress(requestId, "done", "recommended answer ready", startedAt);
    writeSse(res, "status", { stage: "complete", message: "Answer ready", request_id: requestId });
    writeSse(res, "done", finalData);
    writeServiceLog("INFO", `recommended answer done request_id=${requestId} cache_id=${recommendedAnswer.cacheId} reason=${reason} duration_ms=${Date.now() - startedAt}`);
  };
  const recommendedFallbackTimer = recommendedAnswer
    ? setTimeout(() => {
        finishRecommendedAnswer("recommended_cache_deadline");
        upstream.body?.cancel().catch(() => {});
      }, RECOMMENDED_ANSWER_MAX_WAIT_MS)
    : null;
  try {
    for await (const chunk of upstream.body) {
      buffer += decoder.decode(chunk, { stream: true });
      const frames = buffer.split(/\r?\n\r?\n/);
      buffer = frames.pop() || "";
      for (const frame of frames) {
        let event = "message";
        const dataLines = [];
        for (const line of frame.split(/\r?\n/)) {
          if (line.startsWith("event:")) event = line.slice(6).trim();
          if (line.startsWith("data:")) dataLines.push(line.slice(5).trimStart());
        }
        if (!dataLines.length) continue;
        const data = JSON.parse(dataLines.join("\n"));
        if (event === "status") {
          const publicStatus = {
            ...publicChatProgress(data.stage),
            request_id: data.request_id || requestId,
          };
          const publicRank = publicStageRank[publicStatus.stage] ?? lastPublicRank;
          if (
            publicRank < lastPublicRank
            || (publicRank === lastPublicRank && publicStatus.stage === lastPublicStage)
          ) {
            continue;
          }
          lastPublicRank = publicRank;
          lastPublicStage = publicStatus.stage;
          recordProgress(
            requestId,
            publicStatus.stage,
            publicStatus.message,
            startedAt,
          );
          writeSse(res, "status", publicStatus);
        } else if (event === "audit") {
          // Safe retrieval rankings/evidence can arrive before answer generation
          // finishes, allowing the right sidebar to update in real time.
          const auditTrace = data?.retrieval_trace || data?.trace || data;
          if (hasMeaningfulAuditTrace(auditTrace)) {
            latestAuditTrace = mergeAuditTrace(latestAuditTrace, auditTrace);
          }
          writeSse(res, "audit", data);
          if (recommendedAnswer) {
            finishRecommendedAnswer("retrieval_audit_ready");
            upstream.body?.cancel().catch(() => {});
            return;
          }
        } else if (event === "delta" || event === "error") {
          if (event === "error" && req.monitorContext) req.monitorContext.modelFailed = 1;
          if (event === "error") writeServiceLog("ERROR", `chat stream error request_id=${requestId} message=${String(data.message || "").replace(/\s+/g, " ").slice(0, 300)}`);
          // For an exact recommended-question hit, retain real routing and
          // retrieval audit but do not expose model draft text that will be
          // replaced by the reviewed answer in the final done event.
          if (event !== "delta" || !recommendedAnswer) writeSse(res, event, data);
        } else if (event === "done") {
          const mode = data.route === "service" ? "customer" : "manual";
          let cachedAnswer = null;
          const refusal = data.route === "clarify";
          const images = (data.pics || []).map((name) => ({ name, file: resolveImageFile(name) }));
          const sources = Array.isArray(data.sources) ? data.sources : [];
          const manuals = [...new Set(sources.map((source) => String(source?.manual || "").trim()).filter(Boolean))];
          // The retrieval service is authoritative after a stale UI product
          // scope is discarded. Do not label a Toothbrush/WaveRunner answer as
          // the previously selected refrigerator in the public response.
          // A concise follow-up can legitimately produce no retained source
          // row after answer/source projection. In history mode, preserve the
          // already-confirmed product instead of degrading it to the generic
          // "V6" label and poisoning the next turn's memory bucket.
          const responseManual = manuals[0]
            || (refusal ? String(data.product || "待确认产品") : "")
            || contextProduct || resolvedProduct || "V6";
          const responseProduct = manuals.length
            ? uiProductForManual(responseManual)
            : (contextProduct || uiProductForManual(responseManual));
          recordProgress(
            requestId,
            "route",
            "产品与问题范围确认完成",
            startedAt,
          );
          recordProgress(
            requestId,
            "images",
            "回答所需图示整理完成",
            startedAt,
          );
          recordProgress(
            requestId,
            "grounding",
            "答案已按当前问题范围完成核对",
            startedAt,
          );
          const finalAuditTrace = mergeAuditTrace(latestAuditTrace, data.retrieval_trace);
          const reviewedAnswer = Boolean(data.reviewed_answer);
          // The retrieval service owns reviewed answers. The gateway may keep
          // its cache for compatibility, but it must never replace a reviewed
          // backend result or expose answer-source bookkeeping as a retrieval
          // stage in the right-side audit.
          if (recommendedAnswer) {
            cachedAnswer = recommendedAnswer;
          }
          const finalData = {
            request_id: requestId,
            answer: programAnswerIntro(
              originalQuestion,
              cachedAnswer ? cachedAnswer.answer : stripImageList(data.answer),
            ),
            session_id: sessionId,
            timestamp: Math.floor(Date.now() / 1000),
            source: cachedAnswer
              ? "recommended-answer-cache"
              : "v6-retrieval-api",
            answer_source: cachedAnswer
              ? cachedAnswer.source
              : "live-rag",
            recommended_cache_id: cachedAnswer?.cacheId || null,
            answer_mode: mode,
            mode_label: mode === "customer" ? "Customer service" : "V6 manual",
            routing: {
              mode,
              confidence: finalAuditTrace?.answer_confidence?.level || null,
              reason: "retrieval_service",
            },
            answer_confidence: finalAuditTrace?.answer_confidence || null,
            reviewed_answer: reviewedAnswer,
            product: responseProduct,
            question: originalQuestion,
            effective_question: originalQuestion,
            history_context: "",
            history_context_used: Boolean(historyContext || contextPacketUsed),
            context_packet_version: contextPacketUsed ? 1 : null,
            context_retrieval_hint: contextPacketUsed ? contextPacket.retrieval_hint : "auto",
            context_product: responseProduct,
            context_turns: (historyContext || contextPacketUsed)
              ? Math.max(
                  serverTurns.length,
                  Math.ceil((contextPacket.recent_turns || []).length / 2),
                  1,
                )
              : 0,
            context_switched: contextSwitched,
            images,
            manuals: manuals.length ? manuals : (contextProduct ? [contextProduct] : []),
            sources,
            image_descriptions: Array.isArray(data.image_descriptions) ? data.image_descriptions : [],
            elapsed: data.elapsed,
            // Structured, browser-safe RAG audit information. The backend
            // deliberately excludes prompts, credentials and private context.
            retrieval_trace: hasMeaningfulAuditTrace(finalAuditTrace)
              ? finalAuditTrace
              : null,
          };
          if (req.monitorContext) {
            req.monitorContext.product = String(finalData.product || payload.ui_product || "").trim().slice(0, 160);
            req.monitorContext.answerMode = mode;
            req.monitorContext.evidenceCount = sources.length;
            req.monitorContext.imageCount = images.length;
            req.monitorContext.ragNoEvidence = sources.length ? 0 : 1;
          }
          const completedScope = mode === "customer" ? "__service__" : responseProduct;
          // A product-clarification or no-evidence refusal is not a factual
          // answer. Never save it as session memory or let it become the next
          // turn's implicit product scope.
          if (!refusal && completedScope && !detected.ambiguous) {
            const completedBucketKey = `${completedScope}\u0000${memoryEpoch}`;
            const turns = memorySession.buckets.get(completedBucketKey) || [];
            const inheritedComponent = String(contextPacket?.entities?.component || "").trim();
            const responseComponent = inheritedComponent && PRONOUN_FOLLOWUP_RE.test(originalQuestion)
              ? inheritedComponent
              : componentFromAnswerSources(originalQuestion, sources);
            finalData.context_component = responseComponent || null;
            turns.push({
              question: originalQuestion,
              answer: finalData.answer,
              product: responseProduct,
              component: responseComponent,
              imageDescriptions: finalData.image_descriptions,
            });
            memorySession.buckets.set(completedBucketKey, turns.slice(-PRODUCT_MEMORY_MAX_TURNS));
            if (mode === "manual" && responseProduct) memorySession.lastProduct = responseProduct;
            memorySession.lastMode = mode;
            memorySession.updatedAt = Date.now();
            await persistMemorySession(sessionId, memorySession);
          }
          recordProgress(requestId, "done", "答案整理完成", startedAt);
          writeServiceLog(
            "INFO",
            `chat done request_id=${requestId} product=${String(finalData.product || "").replace(/\s+/g, " ").slice(0, 120)} evidence_count=${sources.length} image_count=${images.length} answer_source=${finalData.answer_source} duration_ms=${Date.now() - startedAt}`,
          );
          if (lastPublicRank < publicStageRank.complete) {
            lastPublicRank = publicStageRank.complete;
            lastPublicStage = "complete";
            writeSse(res, "status", {
              stage: "complete",
              message: "答案整理完成",
              request_id: requestId,
            });
          }
          writeSse(res, "done", finalData);
        }
      }
    }
  } catch (error) {
    if (cachedFallbackSent) return;
    if (req.monitorContext) req.monitorContext.modelFailed = 1;
    recordProgress(requestId, "error", error.message, startedAt);
    writeServiceLog("ERROR", `chat exception request_id=${requestId} error=${error.stack || error.message}`);
    writeSse(res, "error", { message: error.message });
  } finally {
    if (recommendedFallbackTimer) clearTimeout(recommendedFallbackTimer);
    clearInterval(heartbeat);
    res.end();
  }
}

async function handleChat(req, res) {
  const requestId = normalizeRequestId(req.headers["x-request-id"]);
  const lease = await acquireChatSlot(req, requestId);
  if (!lease) {
    if (req.monitorContext) req.monitorContext.modelFailed = 1;
    res.setHeader("Retry-After", "5");
    return sendJson(res, 429, {
      code: 429,
      msg: "当前回答请求较多，请 5 秒后重试",
      data: { request_id: requestId, capacity: chatCapacitySnapshot() },
    });
  }
  res.setHeader("X-RAG-Channel", lease.channel);
  res.setHeader("X-RAG-Queue-Wait-Ms", String(lease.waitMs));
  res.setHeader("X-RAG-Admission-Backend", lease.distributedBackend);
  if (req.monitorContext) {
    req.monitorContext.channel = lease.channel;
    if (lease.waitMs) req.monitorContext.queueWaitMs = lease.waitMs;
  }
  try {
    return await handleChatInternal(req, res);
  } finally {
    releaseChatSlot(lease);
  }
}

async function proxyChatJson(req, res, rawOverride = null) {
  const raw = rawOverride === null ? await readBody(req) : rawOverride;
  const chatBackend = currentChatBackend();
  const channel = chatRequestChannel(req);
  let upstream;
  try {
    upstream = await chatBackendFetch("/chat", {
      method: req.method,
      headers: {
        "Content-Type": req.headers["content-type"] || "application/json",
        // Preserve the gateway's channel decision at the retrieval service.
        // This is intentionally a transport header, not user-controlled JSON:
        // only QQ is allowed to select the QQ latency path downstream.
        "X-RAG-Channel": channel,
      },
      body: raw.length ? raw : undefined,
    }, false, chatBackend);
  } catch (error) {
    return sendJson(res, 502, { code: 502, msg: error.message, data: null });
  }
  const body = Buffer.from(await upstream.arrayBuffer());
  sendBuffer(res, upstream.status, body, upstream.headers.get("content-type") || "application/json; charset=utf-8");
}

async function handleJsonChat(req, res) {
  const requestId = normalizeRequestId(req.headers["x-request-id"]);
  const lease = await acquireChatSlot(req, requestId);
  if (!lease) {
    if (req.monitorContext) req.monitorContext.modelFailed = 1;
    res.setHeader("Retry-After", "5");
    return sendJson(res, 429, {
      code: 429,
      msg: "\u5f53\u524d\u56de\u7b54\u8bf7\u6c42\u8f83\u591a\uff0c\u8bf7 5 \u79d2\u540e\u91cd\u8bd5",
      data: { request_id: requestId, capacity: chatCapacitySnapshot() },
    });
  }
  res.setHeader("X-RAG-Channel", lease.channel);
  res.setHeader("X-RAG-Queue-Wait-Ms", String(lease.waitMs));
  res.setHeader("X-RAG-Admission-Backend", lease.distributedBackend);
  if (req.monitorContext) {
    req.monitorContext.channel = lease.channel;
    if (lease.waitMs) req.monitorContext.queueWaitMs = lease.waitMs;
  }
  try {
    // QQ uses the non-streaming JSON contract.  Exact recommended-question
    // hits already have the reviewed answer and original manual picture IDs;
    // serve those locally instead of paying for a second retrieval/model
    // round.  Keep this branch QQ-only so the web SSE audit path remains the
    // canonical live-RAG path and all non-cache QQ questions are unchanged.
    let raw = null;
    if (lease.channel === "qq") {
      raw = await readBody(req);
      let payload = null;
      try {
        payload = JSON.parse(raw.toString("utf8") || "{}");
      } catch {
        return sendJson(res, 400, { code: 400, msg: "invalid JSON", data: null });
      }
      const cached = findRecommendedAnswer(recommendedAnswerIndex, payload, { allowFuzzy: true });
      const cachedPics = Array.isArray(cached?.pics) ? cached.pics : [];
      const cachedAnswer = String(cached?.answer || "").trim();
      const anchorCount = (cachedAnswer.match(/<PIC>/gi) || []).length;
      if (
        cached
        && cachedAnswer
        && cachedPics.length === anchorCount
        && payload.stream !== true
      ) {
        const sessionId = String(payload.session_id || `ragv6_session_${Date.now()}`);
        if (req.monitorContext) {
          req.monitorContext.answerMode = cached.answerMode;
          req.monitorContext.answerSource = "qq-recommended-answer-cache";
          req.monitorContext.evidenceCount = 0;
          req.monitorContext.imageCount = cachedPics.length;
          req.monitorContext.ragNoEvidence = 0;
        }
        writeServiceLog(
          "INFO",
          `qq exact cache hit request_id=${requestId} cache_id=${cached.cacheId} pics=${cachedPics.length}`,
        );
        return sendJson(res, 200, {
          code: 0,
          msg: "success",
          data: {
            answer: programAnswerIntro(payload.question, cachedAnswer),
            session_id: sessionId,
            timestamp: Math.floor(Date.now() / 1000),
            pics: cachedPics,
            image_descriptions: [],
            reviewed_answer: true,
            answer_source: "qq-recommended-answer-cache",
            cache_id: cached.cacheId,
          },
        });
      }
    }
    return await proxyChatJson(req, res, raw);
  } finally {
    releaseChatSlot(lease);
  }
}

async function proxyJson(req, res, backendPath) {
  const raw = await readBody(req);
  let upstream;
  try {
    upstream = await backendFetch(backendPath, {
      method: req.method,
      headers: { "Content-Type": req.headers["content-type"] || "application/json" },
      body: raw.length ? raw : undefined,
    });
  } catch (error) {
    return sendJson(res, 502, { code: 502, msg: error.message, data: null });
  }
  const body = Buffer.from(await upstream.arrayBuffer());
  sendBuffer(res, upstream.status, body, upstream.headers.get("content-type") || "application/json; charset=utf-8");
}

function normalizeManualLocator(value) {
  return String(value || "")
    .toLowerCase()
    .replace(/\u624b\u518c|\u8bf4\u660e\u4e66|\u7528\u6237\u6307\u5357|\u4ea7\u54c1|manual|user\s*guide/gi, "")
    .replace(/[\s\-_/|,.()\[\]{}<>"'`]+/g, "");
}

async function getManifest() {
  if (!manualManifest) manualManifest = JSON.parse(await fsp.readFile(MANUAL_MANIFEST, "utf8")).manuals || [];
  return manualManifest;
}

async function findManual(name) {
  const wanted = normalizeManualLocator(name);
  if (wanted.length < 2) return null;
  let best = null;
  let score = 0;
  for (const manual of await getManifest()) {
    // Older manifests only contain title/file. Treat both as first-class
    // locators, then include optional aliases supplied by newer manifests.
    const candidates = [
      manual.title,
      path.basename(String(manual.file || ""), path.extname(String(manual.file || ""))),
      ...(manual.aliases || []),
      ...(MANUAL_TITLE_ALIASES[manual.title] || []),
    ];
    for (const alias of candidates) {
      const candidate = normalizeManualLocator(alias);
      if (!candidate) continue;
      const current = candidate === wanted ? 1000 : (candidate.includes(wanted) || wanted.includes(candidate) ? 650 + Math.min(candidate.length, wanted.length) : 0);
      if (current > score) {
        best = manual;
        score = current;
      }
    }
  }
  return score >= 300 ? best : null;
}

function pushCaptionIndex(index, imageId, item) {
  if (!imageId || !item) return;
  const candidates = index.get(imageId) || [];
  const duplicate = candidates.some((candidate) => (
    candidate.product === item.product && candidate.content === item.content
  ));
  if (!duplicate) candidates.push(item);
  index.set(imageId, candidates);
}

function resolveCaptionCandidate(candidates, product) {
  if (!candidates?.length) return null;
  const wanted = normalizeManualLocator(product);
  const productMatches = wanted
    ? candidates.filter((candidate) => normalizeManualLocator(candidate.product) === wanted)
    : [];
  if (productMatches.length === 1) return productMatches[0];
  return candidates.length === 1 ? candidates[0] : null;
}

function sourceFigureContext(section) {
  const clean = (value) => String(value || "")
    .replace(/<PIC>/gi, " ")
    .replace(/\s+/g, " ")
    .trim();
  const heading = clean(section?.heading);
  const text = clean(section?.text);
  const context = heading && text.startsWith(heading)
    ? text
    : [heading, text].filter(Boolean).join("。 ");
  return context ? `手册图示上下文：${context.slice(0, 420)}` : "";
}

async function ensureCaptionIndexes() {
  if (captions && captionImageIndex && sectionCaptionIndex) return;

  captions = JSON.parse(await fsp.readFile(CAPTIONS_FILE, "utf8")).items || {};
  captionImageIndex = new Map();
  sectionCaptionIndex = new Map();
  for (const item of Object.values(captions)) {
    const content = String(item?.content || item?.short_caption || "").trim();
    if (!content) continue;
    pushCaptionIndex(captionImageIndex, String(item.image_id || ""), {
      product: String(item.product || ""),
      content,
    });
  }

  // The caption export and raw manual sections use different product naming
  // schemes. Index the source `pic_captions` too, so every source figure has
  // a deterministic fallback even when an export row is absent.
  const sectionFiles = await fsp.readdir(MANUAL_SECTIONS_DIR, { withFileTypes: true });
  for (const entry of sectionFiles) {
    if (!entry.isFile() || path.extname(entry.name).toLowerCase() !== ".json") continue;
    let manual;
    try {
      manual = JSON.parse(await fsp.readFile(path.join(MANUAL_SECTIONS_DIR, entry.name), "utf8"));
    } catch {
      continue;
    }
    const sourceProduct = String(manual?.manual || "").trim();
    for (const section of Array.isArray(manual?.sections) ? manual.sections : []) {
      const pics = Array.isArray(section?.pics) ? section.pics : [];
      const sourceCaptions = Array.isArray(section?.pic_captions) ? section.pic_captions : [];
      pics.forEach((pic, index) => {
        const content = String(sourceCaptions[index] || "").trim() || sourceFigureContext(section);
        if (!content) return;
        pushCaptionIndex(sectionCaptionIndex, String(pic || ""), {
          product: sourceProduct,
          content,
        });
      });
    }
  }
}

async function getCaption(product, imageId) {
  await ensureCaptionIndexes();
  let item = captions[`${product}|${imageId}`];
  if (!item) {
    // The chat UI may expose a short display name (for example, "发电机")
    // while caption records use the canonical manual name ("发电机手册").
    // Resolve only an unambiguous normalized product match for this image.
    const wanted = normalizeManualLocator(product);
    const matches = Object.values(captions).filter((candidate) => (
      String(candidate?.image_id || "") === imageId
      && normalizeManualLocator(candidate?.product) === wanted
    ));
    if (matches.length === 1) item = matches[0];
  }
  const direct = item ? String(item.content || item.short_caption || "").trim() : "";
  if (direct) return direct;

  // Image ids are unique across the current manual corpus. This avoids a
  // brittle dependency on inconsistent localized product names in old data.
  const exported = resolveCaptionCandidate(captionImageIndex.get(imageId), product);
  if (exported?.content) return exported.content;
  const source = resolveCaptionCandidate(sectionCaptionIndex.get(imageId), product);
  return source?.content || null;
}

async function route(req, res) {
  if (req.method === "OPTIONS") {
    res.writeHead(204, commonHeaders({ "Content-Length": "0" }));
    return res.end();
  }
  const url = new URL(req.url, `http://${req.headers.host || "localhost"}`);
  // Raysource is the public namespace. Keep the historical RAGv6 namespace
  // internally so existing browser bundles remain compatible during rollout.
  const pathname = decodeURIComponent(url.pathname).replace(/^\/raysource-api(?=\/|$)/i, "/ragv6-api");
  const lower = pathname.toLowerCase();

  if ((req.method === "GET" || req.method === "HEAD") && lower === "/ragv6-api/account/me") {
    const user = authenticatedUser(req);
    return sendJson(res, 200, {
      code: 0,
      data: user
        ? { mode: "account", user }
        : {
          mode: "guest",
          user: null,
          guest_history: {
            enabled: true,
            identity: "ip_hmac",
            retention_days: GUEST_HISTORY_RETENTION_DAYS,
          },
        },
    });
  }

  if (req.method === "POST" && (lower === "/ragv6-api/account/register" || lower === "/ragv6-api/account/login")) {
    if (!verifySameSiteRequest(req)) return sendJson(res, 403, { code: 403, msg: "拒绝跨站账号请求" });
    try {
      const body = JSON.parse((await readBody(req, 32 * 1024)).toString("utf8") || "{}");
      const credentials = validateCredentials(body.username, body.password);
      if (!consumeAuthAttempt(req, credentials.username)) {
        return sendJson(res, 429, { code: 429, msg: "该账号登录尝试过于频繁，请稍后重试" });
      }
      let userRow;
      if (lower.endsWith("/register")) {
        const existing = accountDb.prepare("SELECT id FROM users WHERE username = ? COLLATE NOCASE").get(credentials.username);
        if (existing) return sendJson(res, 409, { code: 409, msg: "该账号已存在" });
        const salt = crypto.randomBytes(16).toString("hex");
        const now = Date.now();
        const id = crypto.randomUUID();
        accountDb.prepare(`
          INSERT INTO users(id, username, display_name, password_salt, password_hash, role, created_at)
          VALUES (?, ?, ?, ?, ?, 'customer', ?)
        `).run(id, credentials.username, credentials.username, salt, passwordDigest(credentials.password, salt), now);
        userRow = accountDb.prepare("SELECT id, username, display_name, role, created_at FROM users WHERE id = ?").get(id);
      } else {
        const stored = accountDb.prepare(`
          SELECT id, username, display_name, role, created_at, password_salt, password_hash
          FROM users WHERE username = ? COLLATE NOCASE AND role <> 'guest_ip'
        `).get(credentials.username);
        const expected = stored?.password_hash || passwordDigest(credentials.password, "00000000000000000000000000000000");
        const candidate = stored ? passwordDigest(credentials.password, stored.password_salt) : expected.replace(/^./, expected[0] === "0" ? "1" : "0");
        const valid = Boolean(stored) && crypto.timingSafeEqual(Buffer.from(expected, "hex"), Buffer.from(candidate, "hex"));
        if (!valid) return sendJson(res, 401, { code: 401, msg: "账号或密码不正确" });
        userRow = stored;
      }
      const session = createLoginSession(req, userRow.id);
      writeServiceLog("INFO", `account login username=${userRow.username} ip=${requestIp(req)}`);
      return sendJson(res, 200, { code: 0, data: { mode: "account", user: publicUser(userRow) } }, {
        "Set-Cookie": session.cookie,
        "Cache-Control": "no-store",
      });
    } catch (error) {
      return sendJson(res, 400, { code: 400, msg: error.message });
    }
  }

  if (req.method === "POST" && lower === "/ragv6-api/account/logout") {
    if (!verifySameSiteRequest(req)) return sendJson(res, 403, { code: 403, msg: "拒绝跨站账号请求" });
    const token = parseCookies(req)[SESSION_COOKIE];
    if (token) accountDb.prepare("DELETE FROM sessions WHERE token_hash = ?").run(hashSessionToken(token));
    return sendJson(res, 200, { code: 0, data: { mode: "guest" } }, {
      "Set-Cookie": sessionCookie(req, "", 0),
      "Cache-Control": "no-store",
    });
  }

  if (lower === "/ragv6-api/account/conversations") {
    const principal = historyPrincipal(req);
    if (req.method === "GET" || req.method === "HEAD") {
      const rows = accountDb.prepare(`
        SELECT c.id, c.title, c.created_at, c.updated_at, COUNT(m.id) AS message_count
        FROM conversations c
        LEFT JOIN conversation_messages m ON m.conversation_id = c.id
        WHERE c.user_id = ?
        GROUP BY c.id
        ORDER BY c.updated_at DESC
        LIMIT 100
      `).all(principal.id);
      return sendJson(res, 200, {
        code: 0,
        data: { conversations: rows, owner_mode: principal.mode },
      }, { "Cache-Control": "no-store" });
    }
    if (req.method === "POST") {
      if (!verifySameSiteRequest(req)) return sendJson(res, 403, { code: 403, msg: "拒绝跨站请求" });
      const body = JSON.parse((await readBody(req, 32 * 1024)).toString("utf8") || "{}");
      const now = Date.now();
      const conversation = { id: crypto.randomUUID(), title: safeConversationTitle(body.title), created_at: now, updated_at: now };
      accountDb.prepare(`
        INSERT INTO conversations(id, user_id, title, created_at, updated_at)
        VALUES (?, ?, ?, ?, ?)
      `).run(conversation.id, principal.id, conversation.title, now, now);
      return sendJson(res, 201, {
        code: 0,
        data: { conversation, owner_mode: principal.mode },
      }, { "Cache-Control": "no-store" });
    }
  }

  const conversationMatch = lower.match(/^\/ragv6-api\/account\/conversations\/([a-f0-9-]{36})$/);
  if (conversationMatch) {
    const principal = historyPrincipal(req);
    const conversation = accountDb.prepare(`
      SELECT id, title, created_at, updated_at FROM conversations WHERE id = ? AND user_id = ?
    `).get(conversationMatch[1], principal.id);
    if (!conversation) return sendJson(res, 404, { code: 404, msg: "咨询记录不存在" });
    if (req.method === "GET" || req.method === "HEAD") {
      const messages = accountDb.prepare(`
        SELECT id, role, content, payload_json, created_at
        FROM conversation_messages WHERE conversation_id = ? ORDER BY id ASC LIMIT 500
      `).all(conversation.id).map((message) => {
        let payload = null;
        try {
          payload = message.payload_json ? JSON.parse(message.payload_json) : null;
        } catch {}
        return { id: message.id, role: message.role, content: message.content, payload, created_at: message.created_at };
      });
      return sendJson(res, 200, { code: 0, data: { conversation, messages } }, { "Cache-Control": "no-store" });
    }
    if (req.method === "DELETE") {
      if (!verifySameSiteRequest(req)) return sendJson(res, 403, { code: 403, msg: "拒绝跨站请求" });
      accountDb.prepare("DELETE FROM conversations WHERE id = ? AND user_id = ?").run(conversation.id, principal.id);
      return sendJson(res, 200, { code: 0, data: { deleted: conversation.id } });
    }
  }

  const messageMatch = lower.match(/^\/ragv6-api\/account\/conversations\/([a-f0-9-]{36})\/messages$/);
  if (messageMatch && req.method === "POST") {
    if (!verifySameSiteRequest(req)) return sendJson(res, 403, { code: 403, msg: "拒绝跨站请求" });
    const principal = historyPrincipal(req);
    const conversation = accountDb.prepare("SELECT id FROM conversations WHERE id = ? AND user_id = ?").get(messageMatch[1], principal.id);
    if (!conversation) return sendJson(res, 404, { code: 404, msg: "咨询记录不存在" });
    try {
      const body = JSON.parse((await readBody(req, 512 * 1024)).toString("utf8") || "{}");
      const role = body.role === "assistant" ? "assistant" : body.role === "user" ? "user" : "";
      const content = String(body.content || "").trim().slice(0, 50000);
      const payloadJson = body.payload == null ? null : JSON.stringify(body.payload);
      if (!role || !content) return sendJson(res, 400, { code: 400, msg: "消息内容无效" });
      if (payloadJson && Buffer.byteLength(payloadJson) > 400 * 1024) {
        return sendJson(res, 413, { code: 413, msg: "消息附加数据过大" });
      }
      const now = Date.now();
      const result = accountDb.prepare(`
        INSERT INTO conversation_messages(conversation_id, role, content, payload_json, created_at)
        VALUES (?, ?, ?, ?, ?)
      `).run(conversation.id, role, content, payloadJson, now);
      accountDb.prepare("UPDATE conversations SET updated_at = ? WHERE id = ?").run(now, conversation.id);
      return sendJson(res, 201, { code: 0, data: { id: Number(result.lastInsertRowid), created_at: now } });
    } catch (error) {
      return sendJson(res, 400, { code: 400, msg: error.message });
    }
  }

  if (req.method === "POST" && lower === "/ragv6-api/feedback") {
    if (!verifySameSiteRequest(req)) return sendJson(res, 403, { code: 403, msg: "拒绝跨站反馈请求" });
    try {
      const body = JSON.parse((await readBody(req, 32 * 1024)).toString("utf8") || "{}");
      const answerRequestId = String(body.request_id || "").trim();
      const action = String(body.action || "").trim().toLowerCase();
      const allowed = new Set(["solved", "unsolved", "transfer", "ticket_open", "ticket_submit"]);
      if (!REQUEST_ID_PATTERN.test(answerRequestId)) return sendJson(res, 400, { code: 400, msg: "无效的回答 Request ID" });
      if (!allowed.has(action)) return sendJson(res, 400, { code: 400, msg: "无效的客服反馈动作" });
      const monitoredRequest = monitorDb.prepare(`
        SELECT product FROM monitor_requests WHERE request_id = ? AND category = 'chat'
      `).get(answerRequestId);
      const product = String(body.product || monitoredRequest?.product || "").trim().slice(0, 160);
      if (action === "solved" || action === "unsolved") {
        monitorDb.prepare(`
          DELETE FROM monitor_feedback
          WHERE request_id = ? AND action IN ('solved', 'unsolved')
        `).run(answerRequestId);
      }
      monitorDb.prepare(`
        INSERT OR IGNORE INTO monitor_feedback(request_id, product, action, created_at)
        VALUES (?, ?, ?, ?)
      `).run(answerRequestId, product, action, Date.now());
      writeServiceLog("INFO", `service action request_id=${answerRequestId} action=${action} product=${product.replace(/\s+/g, " ").slice(0, 120)}`);
      return sendJson(res, 201, { code: 0, data: { request_id: answerRequestId, action } }, { "Cache-Control": "no-store" });
    } catch (error) {
      return sendJson(res, 400, { code: 400, msg: error.message });
    }
  }

  if (lower === "/internal-monitor") return redirect(res, "/internal-monitor/");
  if (lower.startsWith("/internal-monitor/") || lower.startsWith("/internal-monitor-api/") || lower.startsWith("/rag/monitor/")) {
    // The monitor is intentionally public: the site owner requested passwordless
    // access through the existing Cloudflare tunnel.
    if ((req.method === "GET" || req.method === "HEAD") && (lower === "/internal-monitor/" || lower === "/rag/monitor/")) {
      return sendFile(req, res, path.join(MONITOR_DIR, "index.html"));
    }
    const monitorAsset = lower.match(/^\/(?:internal-monitor|rag\/monitor)\/(app\.js|styles\.css)$/);
    if ((req.method === "GET" || req.method === "HEAD") && monitorAsset) {
      return sendFile(req, res, path.join(MONITOR_DIR, monitorAsset[1]));
    }
    if ((req.method === "GET" || req.method === "HEAD") && lower === "/internal-monitor-api/snapshot") {
      return sendJson(res, 200, await monitorSnapshot());
    }
    if (req.method === "POST" && lower === "/internal-monitor-api/command") {
      try {
        const payload = JSON.parse((await readBody(req, 32 * 1024)).toString("utf8") || "{}");
        return sendJson(res, 200, { code: 0, output: await runMonitorCommand(payload.command) });
      } catch (error) {
        return sendJson(res, 400, { code: 400, error: error.message });
      }
    }
  }

  if ((req.method === "GET" || req.method === "HEAD") && lower === "/livez") {
    return sendJson(res, 200, {
      status: "ok",
      service: "ragv6-web-client",
      pid: process.pid,
      uptime_seconds: Math.round(process.uptime()),
    }, { "Cache-Control": "no-store" });
  }
  if ((req.method === "GET" || req.method === "HEAD") && lower === "/readyz") {
    const [backend, shared] = await Promise.all([backendHealth(), sharedState.readiness()]);
    const ready = Boolean(backend.reachable && shared.ready);
    return sendJson(res, ready ? 200 : 503, {
      status: ready ? "ready" : "not_ready",
      backend,
      shared_state: shared,
    }, { "Cache-Control": "no-store" });
  }
  if ((req.method === "GET" || req.method === "HEAD") && lower === "/metrics") {
    return sendText(
      res,
      200,
      prometheusMetrics(),
      "text/plain; version=0.0.4; charset=utf-8",
      { "Cache-Control": "no-store" },
    );
  }
  if ((req.method === "GET" || req.method === "HEAD") && lower === "/health") {
    let backend = { reachable: false };
    try {
      const response = await fetch(new URL("/health", CONFIG.apiOrigin));
      backend = { reachable: response.ok, status: response.status, data: await response.json() };
    } catch (error) {
      backend = { reachable: false, error: error.message };
    }
    return sendJson(res, 200, {
      status: "ok",
      service: "ragv6-web-client",
      api_origin: CONFIG.apiOrigin,
      chat_backend: currentChatBackend(),
      backend,
      chat_capacity: chatCapacitySnapshot(),
      shared_state: sharedState.health(),
    });
  }
  if ((req.method === "GET" || req.method === "HEAD") && lower === "/manifest.json") {
    return sendFile(req, res, path.join(UI_DIR, "manifest.json"), "public, max-age=3600");
  }
  if ((req.method === "GET" || req.method === "HEAD") && lower === "/service-worker.js") {
    return sendFile(req, res, path.join(UI_DIR, "service-worker.js"), "no-cache");
  }
  if ((req.method === "GET" || req.method === "HEAD") && lower === "/downloads/raysource-android.apk") {
    // APK updates must take effect immediately; a stale binary can look like
    // a successful download while silently keeping users on an older build.
    return sendFile(req, res, ANDROID_APP_FILE, "no-store, no-cache, must-revalidate, max-age=0");
  }
  if ((req.method === "GET" || req.method === "HEAD") && lower === "/") return sendFile(req, res, path.join(UI_DIR, "index.html"));
  if ((req.method === "GET" || req.method === "HEAD") && (lower === "/ragv6" || lower.startsWith("/ragv6/"))) {
    return redirect(res, `${url.pathname.replace(/^\/ragv6/i, "/rag")}${url.search}`);
  }
  if ((req.method === "GET" || req.method === "HEAD") && (lower === "/rag" || lower === "/rag/")) return redirect(res, "/");
  if ((req.method === "GET" || req.method === "HEAD") && lower === "/rag/monitor") return redirect(res, "/rag/monitor/");

  if ((req.method === "GET" || req.method === "HEAD") && (lower === "/rag/chunk-manager" || lower === "/rag/chunk-manager/")) {
    return sendFile(req, res, path.join(CHUNK_MANAGER_DIR, "index.html"));
  }
  const chunkManagerAsset = lower.match(/^\/rag\/chunk-manager\/(app\.js|styles\.css)$/);
  if ((req.method === "GET" || req.method === "HEAD") && chunkManagerAsset) {
    return sendFile(req, res, path.join(CHUNK_MANAGER_DIR, chunkManagerAsset[1]));
  }

  const uiAsset = lower.match(/^\/(?:rag\/)?(audit-contract\.js|app\.js|styles\.css|answers\.json|manifest\.json|service-worker\.js|ray-source-(?:logo|mark|icon-192|icon-512)\.png|model-[a-z0-9-]+\.svg)$/);
  if ((req.method === "GET" || req.method === "HEAD") && uiAsset) return sendFile(req, res, path.join(UI_DIR, uiAsset[1]));

  if ((req.method === "GET" || req.method === "HEAD") && lower === "/rag/manual-index") return redirect(res, "/rag/manual-index/");
  // The index is deliberately small and changes independently from manual pages.
  // Revalidate it so visitors immediately receive the current navigation shell.
  if ((req.method === "GET" || req.method === "HEAD") && lower === "/rag/manual-index/") {
    return sendFile(req, res, MANUAL_INDEX, "public, max-age=300, stale-while-revalidate=86400");
  }
  if ((req.method === "GET" || req.method === "HEAD") && lower === "/rag/manual-locate") return redirect(res, `/rag/manual-locate/${url.search}`);
  if ((req.method === "GET" || req.method === "HEAD") && lower === "/rag/manual-locate/") {
    const manual = await findManual(url.searchParams.get("manual"));
    if (manual && /^[a-z0-9-]+\.html$/i.test(String(manual.file || ""))) {
      return sendFile(
        req,
        res,
        path.join(MANUALS_DIR, manual.file),
        "public, max-age=3600, stale-while-revalidate=86400",
      );
    }
    // Old links without a resolvable manual keep the complete-directory fallback.
    return sendFile(req, res, MANUAL_LOCATOR, "public, max-age=300, stale-while-revalidate=86400");
  }
  if ((req.method === "GET" || req.method === "HEAD") && lower === "/rag/manual-index/navigator.js") return sendFile(req, res, path.join(MANUAL_ROOT, "navigator.js"));
  if ((req.method === "GET" || req.method === "HEAD") && lower === "/rag/manual-view/progressive-loader.js") return sendFile(req, res, path.join(MANUAL_ROOT, "progressive-loader.js"));
  if ((req.method === "GET" || req.method === "HEAD") && lower === "/rag/manual-view/manifest.json") return sendFile(req, res, MANUAL_MANIFEST, "public, max-age=3600");

  if ((req.method === "GET" || req.method === "HEAD") && lower === "/rag/manual-view/") {
    const manual = await findManual(url.searchParams.get("manual"));
    if (!manual || !/^[a-z0-9-]+\.html$/i.test(String(manual.file || ""))) return redirect(res, `/rag/manual-index/${url.search}`);
    // Keep existing source links working too. Older answers point at this
    // route, but the individual manual file cannot interpret locator hashes.
    return sendFile(req, res, MANUAL_LOCATOR, "no-cache");
  }
  const manualPage = pathname.match(/^\/rag\/manual-view\/manuals\/(manual-\d{2}\.html)$/i);
  if ((req.method === "GET" || req.method === "HEAD") && manualPage) return sendFile(req, res, path.join(MANUALS_DIR, manualPage[1]), "public, max-age=3600");

  const imageMatch = pathname.match(/^\/manual-images\/([^/]+)$/)
    || pathname.match(/^\/rag\/manual-images\/([^/]+)$/i)
    || pathname.match(/^\/rag\/(?:manual-view\/)?\u624b\u518c\/\u63d2\u56fe\/([^/]+)$/i);
  if ((req.method === "GET" || req.method === "HEAD") && imageMatch) {
    const name = imageMatch[1];
    const extension = path.extname(name).toLowerCase();
    if (!/^[A-Za-z0-9._-]+$/.test(name) || name.includes("..") || !MIME[extension]?.startsWith("image/")) return sendText(res, 400, "Invalid image name");
    let imagePath = path.join(IMAGE_DIR, name);
    if (!fs.existsSync(imagePath)) {
      // Some source manuals label PNG figures as .jpg. Keep citation URLs
      // stable and resolve only a same-basename image within the asset folder.
      const base = path.basename(name, extension);
      imagePath = [".jpg", ".jpeg", ".png", ".gif", ".webp", ".bmp"]
        .map((candidateExtension) => path.join(IMAGE_DIR, `${base}${candidateExtension}`))
        .find((candidate) => fs.existsSync(candidate));
    }
    if (!imagePath) return sendText(res, 404, "Image not found");
    return sendFile(req, res, imagePath, "public, max-age=86400");
  }
  if ((req.method === "GET" || req.method === "HEAD") && lower === "/rag/image-caption") {
    const product = String(url.searchParams.get("product") || "").trim();
    const imageId = String(url.searchParams.get("image") || "").trim();
    const noStore = { "Cache-Control": "no-store, no-cache, must-revalidate, max-age=0" };
    if (!product || !/^[A-Za-z0-9_-]+$/.test(imageId)) return sendJson(res, 400, { code: 400, msg: "invalid caption request" }, noStore);
    return sendJson(res, 200, { code: 0, msg: "success", data: { caption: await getCaption(product, imageId) } }, noStore);
  }

  if ((req.method === "GET" || req.method === "POST") && lower === "/api/wecom/callback") {
    return wecomCallback(req, res, url);
  }
  if (req.method === "POST" && lower === "/api/wecom/send-message") return wecomSendMessage(req, res);
  if ((req.method === "GET" || req.method === "HEAD") && lower === "/api/wecom/health") {
    const missing = wecomMissingConfig();
    return sendJson(res, 200, {
      code: 0,
      enabled: WECOM.enabled,
      callback_configured: Boolean(WECOM.token && WECOM.encodingAesKey && WECOM.corpId),
      send_configured: Boolean(WECOM.corpId && WECOM.agentId && WECOM.secret),
      send_auth_configured: Boolean(WECOM.sendApiKey),
      missing_config: missing,
      callback_url: "/api/wecom/callback",
      configured_parts: {
        corp_id: Boolean(WECOM.corpId),
        agent_id: Boolean(WECOM.agentId),
        token: Boolean(WECOM.token),
        aes_key: Boolean(WECOM.encodingAesKey),
        send_api_key: Boolean(WECOM.sendApiKey),
      },
    });
  }

  // Public REST endpoint used by competition judges and external clients.
  // Authentication to the retrieval service remains server-side.
  if (req.method === "POST" && lower === "/chat") {
    // The web UI sends stream=true and identifies itself explicitly. Keep
    // public non-web /chat clients on the legacy JSON proxy, while routing the
    // browser's customer-service path through the same SSE handler as manual
    // questions.
    if (String(req.headers["x-client-type"] || "").toLowerCase() === "web") {
      return handleChat(req, res);
    }
    return handleJsonChat(req, res);
  }
  if (req.method === "POST" && lower === "/ragv6-api/chat") return handleChat(req, res);
  if (lower.startsWith("/ragv6-api/chunks")) {
    // Preserve the original path here. Manual names are case-sensitive, so
    // forwarding the lower-cased router path turns "Air Fryer" into
    // "air fryer" and makes the retrieval service return 404.
    const result = await handleChunkManagerApi(req, res, pathname, url);
    if (result !== false) return result;
  }
  if (lower.startsWith("/ragv6-api/chunk-sql")) {
    const result = await handleChunkSqlApi(req, res, pathname);
    if (result !== false) return result;
  }
  if (req.method === "POST" && lower === "/ragv6-api/translate") return proxyJson(req, res, "/translate");
  if (req.method === "POST" && lower === "/ragv6-api/retrieve") return proxyJson(req, res, "/retrieve");
  if (req.method === "GET" && lower === "/ragv6-api/progress") {
    const requestId = String(url.searchParams.get("request_id") || "");
    const row = progressStore.get(requestId) || { created: Date.now(), updated: Date.now(), events: [] };
    return sendJson(res, 200, { code: 0, msg: "success", data: { request_id: requestId, events: row.events, total_elapsed: Math.round(((row.updated || Date.now()) - row.created) / 100) / 10 } });
  }
  if (req.method === "GET" && lower === "/ragv6-api/model-profile") return sendJson(res, 200, currentModelProfile());
  if (req.method === "POST" && lower === "/ragv6-api/model-profile/switch") {
    try {
      const profileId = JSON.parse((await readBody(req)).toString("utf8") || "{}").profile_id;
      const profile = MODEL_PROFILES.find((item) => item.id === profileId);
      if (profile?.available !== false) activeProfileId = profileId;
    } catch {}
    return sendJson(res, 200, currentModelProfile());
  }

  return sendText(res, 404, "Not Found");
}

const server = http.createServer((req, res) => {
  const requestStartedAt = Date.now();
  const pathname = monitorPathname(req.url);
  const requestId = normalizeRequestId(req.headers["x-request-id"]);
  const trackRequest = !pathname.startsWith("/internal-monitor-api");
  req.monitorContext = {
    requestId,
    startedAt: requestStartedAt,
    category: requestCategory(pathname),
    product: "",
    model: "",
    answerMode: "",
    evidenceCount: 0,
    imageCount: 0,
    modelFailed: 0,
    ragNoEvidence: 0,
  };
  res.setHeader("X-Request-Id", requestId);
  if (trackRequest) monitorState.activeRequests += 1;
  let requestFinished = false;
  const finishRequest = () => {
    if (requestFinished || !trackRequest) return;
    requestFinished = true;
    monitorState.activeRequests = Math.max(0, monitorState.activeRequests - 1);
    recordMonitorRequest(req, res.statusCode || 0, Date.now() - requestStartedAt);
  };
  res.once("finish", finishRequest);
  res.once("close", finishRequest);
  route(req, res).catch((error) => {
    if (pathname === "/ragv6-api/chat") req.monitorContext.modelFailed = 1;
    writeServiceLog("ERROR", `request_id=${requestId} ${error.stack || error.message}`);
    console.error(error);
    if (!res.headersSent) sendJson(res, 500, { code: 500, msg: error.message, data: null });
    else res.end();
  });
});

async function startServer() {
  const shared = await sharedState.start();
  server.listen(CONFIG.port, CONFIG.host, () => {
    applyMonitorRetention();
    cleanupGuestHistory(Date.now(), true);
    const chatBackend = currentChatBackend();
    writeServiceLog("INFO", `Listening on http://${CONFIG.host}:${CONFIG.port}/rag/; retrieval=${CONFIG.apiOrigin}; chat_backend=${chatBackend.mode}; chat_origin=${chatBackend.origin}`);
    writeServiceLog("INFO", `shared state mode=${shared.mode} ready=${shared.ready} required=${shared.required} instance=${shared.instance_id}`);
    console.log(`RAGv6 web client: http://${CONFIG.host}:${CONFIG.port}/rag/`);
    console.log(`Retrieval API: ${CONFIG.apiOrigin}`);
    const missingWecom = wecomMissingConfig();
    writeServiceLog("INFO", `wecom enabled=${WECOM.enabled} callback_configured=${Boolean(WECOM.token && WECOM.encodingAesKey && WECOM.corpId)} send_configured=${Boolean(WECOM.corpId && WECOM.agentId && WECOM.secret)} missing=${missingWecom.join(",") || "none"}`);
  });
}

void startServer().catch((error) => {
  writeServiceLog("ERROR", `gateway startup failed: ${error.stack || error.message}`);
  console.error(error);
  process.exitCode = 1;
});

let shutdownStarted = false;
async function shutdown(signal) {
  if (shutdownStarted) return;
  shutdownStarted = true;
  writeServiceLog("INFO", `shutdown requested signal=${signal}`);
  const forceTimer = setTimeout(() => process.exit(1), 10_000);
  forceTimer.unref?.();
  server.close(async () => {
    await sharedState.stop();
    clearTimeout(forceTimer);
    process.exit(0);
  });
}

process.once("SIGINT", () => { void shutdown("SIGINT"); });
process.once("SIGTERM", () => { void shutdown("SIGTERM"); });

const alertTimer = setInterval(() => {
  checkOperationalAlerts().catch((error) => writeServiceLog("ERROR", `alert evaluator failed: ${error.stack || error.message}`));
}, 60000);
alertTimer.unref();
const retentionTimer = setInterval(() => {
  try {
    applyMonitorRetention();
    cleanupGuestHistory(Date.now(), true);
  } catch (error) {
    writeServiceLog("ERROR", `retention cleanup failed: ${error.message}`);
  }
}, 24 * 60 * 60 * 1000);
retentionTimer.unref();
