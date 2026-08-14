"use strict";

const crypto = require("node:crypto");
const { createClient } = require("redis");

const ACQUIRE_LEASE_SCRIPT = `
local global_key = KEYS[1]
local session_key = KEYS[2]
local request_key = KEYS[3]
local owner = ARGV[1]
local now_ms = tonumber(ARGV[2])
local expires_ms = tonumber(ARGV[3])
local limit = tonumber(ARGV[4])
local ttl_ms = tonumber(ARGV[5])
local has_session = ARGV[6] == '1'
local has_request = ARGV[7] == '1'

redis.call('ZREMRANGEBYSCORE', global_key, '-inf', now_ms)
if has_request and redis.call('EXISTS', request_key) == 1 then
  return -2
end
if has_session and redis.call('EXISTS', session_key) == 1 then
  return -3
end
if redis.call('ZCARD', global_key) >= limit then
  return 0
end

redis.call('ZADD', global_key, expires_ms, owner)
redis.call('PEXPIRE', global_key, math.max(ttl_ms * 2, 1000))
if has_session then
  local session_set = redis.call('SET', session_key, owner, 'PX', ttl_ms, 'NX')
  if not session_set then
    redis.call('ZREM', global_key, owner)
    return -3
  end
end
if has_request then
  local request_set = redis.call('SET', request_key, owner, 'PX', ttl_ms, 'NX')
  if not request_set then
    redis.call('ZREM', global_key, owner)
    if has_session and redis.call('GET', session_key) == owner then
      redis.call('DEL', session_key)
    end
    return -2
  end
end
return 1
`;

const RENEW_LEASE_SCRIPT = `
local global_key = KEYS[1]
local session_key = KEYS[2]
local request_key = KEYS[3]
local owner = ARGV[1]
local expires_ms = tonumber(ARGV[2])
local ttl_ms = tonumber(ARGV[3])
local has_session = ARGV[4] == '1'
local has_request = ARGV[5] == '1'

if not redis.call('ZSCORE', global_key, owner) then
  return 0
end
if has_session and redis.call('GET', session_key) ~= owner then
  return -1
end
if has_request and redis.call('GET', request_key) ~= owner then
  return -2
end
redis.call('ZADD', global_key, expires_ms, owner)
redis.call('PEXPIRE', global_key, math.max(ttl_ms * 2, 1000))
if has_session then redis.call('PEXPIRE', session_key, ttl_ms) end
if has_request then redis.call('PEXPIRE', request_key, ttl_ms) end
return 1
`;

const RELEASE_LEASE_SCRIPT = `
local global_key = KEYS[1]
local session_key = KEYS[2]
local request_key = KEYS[3]
local owner = ARGV[1]
local has_session = ARGV[2] == '1'
local has_request = ARGV[3] == '1'

redis.call('ZREM', global_key, owner)
if has_session and redis.call('GET', session_key) == owner then
  redis.call('DEL', session_key)
end
if has_request and redis.call('GET', request_key) == owner then
  redis.call('DEL', request_key)
end
return 1
`;

function envFlag(value, fallback = false) {
  if (value == null || String(value).trim() === "") return fallback;
  return /^(1|true|yes|on)$/i.test(String(value).trim());
}

function positiveInteger(value, fallback, minimum = 1) {
  const parsed = Number(value);
  return Number.isFinite(parsed) ? Math.max(minimum, Math.floor(parsed)) : fallback;
}

function sleep(ms) {
  return new Promise((resolve) => {
    const timer = setTimeout(resolve, Math.max(0, ms));
    timer.unref?.();
  });
}

function safeError(error) {
  return String(error?.message || error || "unknown")
    .replace(/redis:\/\/[^@\s]+@/gi, "redis://***@")
    .replace(/\s+/g, " ")
    .slice(0, 200);
}

function cloneJson(value) {
  if (value == null) return value;
  return JSON.parse(JSON.stringify(value));
}

class SharedState {
  constructor(options = {}) {
    this.enabled = options.enabled ?? envFlag(process.env.RAGV6_REDIS_ENABLED, true);
    this.required = options.required ?? envFlag(process.env.RAGV6_REDIS_REQUIRED, false);
    this.url = String(options.url || process.env.RAGV6_REDIS_URL || "redis://127.0.0.1:6379");
    this.namespace = String(options.namespace || process.env.RAGV6_REDIS_NAMESPACE || "raysource:v1")
      .replace(/[^A-Za-z0-9:_-]/g, "_")
      .slice(0, 80);
    this.connectTimeoutMs = positiveInteger(
      options.connectTimeoutMs ?? process.env.RAGV6_REDIS_CONNECT_TIMEOUT_MS,
      1800,
      100,
    );
    this.instanceId = String(options.instanceId || crypto.randomUUID());
    this.publicInstanceId = crypto.createHash("sha256").update(this.instanceId).digest("hex").slice(0, 12);
    this.client = options.client || null;
    this.ownsClient = !options.client;
    this.connectPromise = null;
    this.local = new Map();
    this.activeLeases = new Map();
    this.status = {
      mode: this.enabled ? "initializing" : "memory",
      connected: false,
      ready: false,
      degraded: !this.enabled,
      lastError: "",
      lastReadyAt: 0,
      reconnects: 0,
    };
    this.stats = {
      operations: 0,
      redisHits: 0,
      memoryFallbacks: 0,
      errors: 0,
      leaseAcquired: 0,
      leaseRejectedGlobal: 0,
      leaseRejectedSession: 0,
      leaseRejectedDuplicate: 0,
      leaseRenewFailures: 0,
      leaseReleased: 0,
    };
  }

  key(suffix) {
    return `${this.namespace}:${String(suffix || "").replace(/^:+/, "")}`;
  }

  opaque(value) {
    return crypto.createHash("sha256").update(String(value || "")).digest("hex");
  }

  async start() {
    if (!this.enabled) return this.health();
    if (this.connectPromise) return this.connectPromise;
    if (!this.client) {
      this.client = createClient({
        url: this.url,
        socket: {
          connectTimeout: this.connectTimeoutMs,
          reconnectStrategy: (retries) => Math.min(2500, 100 + retries * 150),
        },
        disableOfflineQueue: true,
      });
    }
    this.client.on?.("error", (error) => {
      this.status.ready = false;
      this.status.connected = Boolean(this.client?.isOpen);
      this.status.mode = "memory";
      this.status.degraded = true;
      this.status.lastError = safeError(error);
      this.stats.errors += 1;
    });
    this.client.on?.("ready", () => {
      this.status.ready = true;
      this.status.connected = true;
      this.status.mode = "redis";
      this.status.degraded = false;
      this.status.lastError = "";
      this.status.lastReadyAt = Date.now();
    });
    this.client.on?.("reconnecting", () => {
      this.status.reconnects += 1;
      this.status.ready = false;
      this.status.degraded = true;
    });
    this.client.on?.("end", () => {
      this.status.connected = false;
      this.status.ready = false;
      this.status.mode = "memory";
      this.status.degraded = true;
    });

    this.connectPromise = (async () => {
      try {
        const connection = this.client.isOpen ? Promise.resolve() : this.client.connect();
        await Promise.race([
          connection,
          new Promise((_, reject) => {
            const timer = setTimeout(() => reject(new Error("Redis connect timeout")), this.connectTimeoutMs + 250);
            timer.unref?.();
          }),
        ]);
        await this.client.ping();
        this.status.connected = true;
        this.status.ready = true;
        this.status.mode = "redis";
        this.status.degraded = false;
        this.status.lastError = "";
        this.status.lastReadyAt = Date.now();
      } catch (error) {
        this.status.connected = Boolean(this.client?.isOpen);
        this.status.ready = false;
        this.status.mode = "memory";
        this.status.degraded = true;
        this.status.lastError = safeError(error);
      }
      return this.health();
    })();
    return this.connectPromise;
  }

  async stop() {
    for (const lease of this.activeLeases.values()) lease.stopRenewal();
    this.activeLeases.clear();
    if (this.ownsClient && this.client?.isOpen) {
      try { await this.client.quit(); } catch { this.client.disconnect?.(); }
    }
  }

  redisReady() {
    return Boolean(this.enabled && this.client?.isReady);
  }

  rememberLocal(key, value, ttlMs) {
    this.local.set(key, {
      value: cloneJson(value),
      expiresAt: Date.now() + Math.max(1, ttlMs),
    });
  }

  localValue(key) {
    const row = this.local.get(key);
    if (!row) return null;
    if (row.expiresAt <= Date.now()) {
      this.local.delete(key);
      return null;
    }
    return cloneJson(row.value);
  }

  async getJson(suffix) {
    const key = this.key(suffix);
    this.stats.operations += 1;
    if (this.redisReady()) {
      try {
        const raw = await this.client.get(key);
        if (raw != null) {
          const value = JSON.parse(raw);
          this.stats.redisHits += 1;
          return value;
        }
        return null;
      } catch (error) {
        this.stats.errors += 1;
        this.status.lastError = safeError(error);
      }
    }
    this.stats.memoryFallbacks += 1;
    return this.localValue(key);
  }

  async setJson(suffix, value, ttlSeconds = 3600) {
    const key = this.key(suffix);
    const ttlMs = positiveInteger(Number(ttlSeconds) * 1000, 3600000, 1);
    this.stats.operations += 1;
    this.rememberLocal(key, value, ttlMs);
    if (!this.redisReady()) {
      this.stats.memoryFallbacks += 1;
      return { backend: "memory", stored: true };
    }
    try {
      await this.client.set(key, JSON.stringify(value), { PX: ttlMs });
      return { backend: "redis", stored: true };
    } catch (error) {
      this.stats.errors += 1;
      this.stats.memoryFallbacks += 1;
      this.status.lastError = safeError(error);
      return { backend: "memory", stored: true };
    }
  }

  async delete(suffix) {
    const key = this.key(suffix);
    this.local.delete(key);
    if (!this.redisReady()) return false;
    try {
      return Boolean(await this.client.del(key));
    } catch (error) {
      this.stats.errors += 1;
      this.status.lastError = safeError(error);
      return false;
    }
  }

  leaseKeys(sessionId, requestId) {
    return [
      this.key("admission:active"),
      sessionId ? this.key(`admission:session:${this.opaque(sessionId)}`) : this.key("admission:none:session"),
      requestId ? this.key(`admission:request:${this.opaque(requestId)}`) : this.key("admission:none:request"),
    ];
  }

  async tryRedisLease({ owner, sessionId, requestId, limit, ttlMs }) {
    const now = Date.now();
    const result = Number(await this.client.eval(ACQUIRE_LEASE_SCRIPT, {
      keys: this.leaseKeys(sessionId, requestId),
      arguments: [
        owner,
        String(now),
        String(now + ttlMs),
        String(limit),
        String(ttlMs),
        sessionId ? "1" : "0",
        requestId ? "1" : "0",
      ],
    }));
    if (result === 1) return { acquired: true, reason: "acquired" };
    if (result === -2) return { acquired: false, reason: "duplicate_request" };
    if (result === -3) return { acquired: false, reason: "session_busy" };
    return { acquired: false, reason: "global_busy" };
  }

  memoryOnlyLease(meta) {
    const lease = {
      backend: "memory-fallback",
      owner: `memory:${this.instanceId}:${crypto.randomUUID()}`,
      waitMs: 0,
      released: false,
      release: async () => { lease.released = true; },
    };
    this.stats.memoryFallbacks += 1;
    return lease;
  }

  async acquireAdmission(meta = {}) {
    const started = Date.now();
    const sessionId = String(meta.sessionId || "").slice(0, 240);
    const requestId = String(meta.requestId || "").slice(0, 240);
    const limit = positiveInteger(meta.limit, 4, 1);
    const ttlMs = positiveInteger(meta.ttlMs, 180000, 5000);
    const waitTimeoutMs = positiveInteger(meta.waitTimeoutMs, 120000, 0);
    if (!this.redisReady()) return this.memoryOnlyLease(meta);

    const owner = `${this.instanceId}:${crypto.randomUUID()}`;
    const deadline = started + waitTimeoutMs;
    let lastReason = "global_busy";
    while (true) {
      let result;
      try {
        result = await this.tryRedisLease({ owner, sessionId, requestId, limit, ttlMs });
      } catch (error) {
        this.stats.errors += 1;
        this.status.lastError = safeError(error);
        return this.memoryOnlyLease(meta);
      }
      if (result.acquired) break;
      lastReason = result.reason;
      if (lastReason === "duplicate_request") {
        this.stats.leaseRejectedDuplicate += 1;
        return null;
      }
      if (Date.now() >= deadline) {
        if (lastReason === "session_busy") this.stats.leaseRejectedSession += 1;
        else this.stats.leaseRejectedGlobal += 1;
        return null;
      }
      await sleep(Math.min(80, Math.max(10, deadline - Date.now())));
    }

    this.stats.leaseAcquired += 1;
    const keys = this.leaseKeys(sessionId, requestId);
    let released = false;
    const renewEveryMs = Math.max(2000, Math.floor(ttlMs / 3));
    const renewal = setInterval(async () => {
      if (released || !this.redisReady()) return;
      try {
        const result = Number(await this.client.eval(RENEW_LEASE_SCRIPT, {
          keys,
          arguments: [owner, String(Date.now() + ttlMs), String(ttlMs), sessionId ? "1" : "0", requestId ? "1" : "0"],
        }));
        if (result !== 1) this.stats.leaseRenewFailures += 1;
      } catch (error) {
        this.stats.leaseRenewFailures += 1;
        this.status.lastError = safeError(error);
      }
    }, renewEveryMs);
    renewal.unref?.();

    const lease = {
      backend: "redis",
      owner,
      waitMs: Date.now() - started,
      released: false,
      stopRenewal: () => clearInterval(renewal),
      release: async () => {
        if (released) return;
        released = true;
        lease.released = true;
        clearInterval(renewal);
        this.activeLeases.delete(owner);
        try {
          if (this.redisReady()) {
            await this.client.eval(RELEASE_LEASE_SCRIPT, {
              keys,
              arguments: [owner, sessionId ? "1" : "0", requestId ? "1" : "0"],
            });
          }
        } catch (error) {
          this.stats.errors += 1;
          this.status.lastError = safeError(error);
        } finally {
          this.stats.leaseReleased += 1;
        }
      },
    };
    this.activeLeases.set(owner, lease);
    return lease;
  }

  async readiness() {
    if (!this.enabled) return { ready: !this.required, mode: "memory", required: this.required };
    if (this.redisReady()) {
      try {
        await this.client.ping();
        return { ready: true, mode: "redis", required: this.required };
      } catch (error) {
        this.status.lastError = safeError(error);
      }
    }
    return { ready: !this.required, mode: "memory", required: this.required };
  }

  health() {
    return {
      enabled: this.enabled,
      required: this.required,
      mode: this.redisReady() ? "redis" : "memory",
      connected: Boolean(this.client?.isOpen),
      ready: this.redisReady(),
      degraded: this.enabled && !this.redisReady(),
      namespace: this.namespace,
      instance_id: this.publicInstanceId,
      last_error: this.status.lastError,
      last_ready_at: this.status.lastReadyAt || null,
      reconnects: this.status.reconnects,
      active_distributed_leases: this.activeLeases.size,
      operations: this.stats.operations,
      redis_hits: this.stats.redisHits,
      memory_fallbacks: this.stats.memoryFallbacks,
      errors: this.stats.errors,
      lease_acquired: this.stats.leaseAcquired,
      lease_rejected_global: this.stats.leaseRejectedGlobal,
      lease_rejected_session: this.stats.leaseRejectedSession,
      lease_rejected_duplicate: this.stats.leaseRejectedDuplicate,
      lease_renew_failures: this.stats.leaseRenewFailures,
      lease_released: this.stats.leaseReleased,
    };
  }
}

module.exports = { SharedState };
