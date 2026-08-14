"use strict";

const assert = require("node:assert/strict");
const test = require("node:test");
const { SharedState } = require("./shared-state");

class FakeRedisClient {
  constructor() {
    this.isOpen = true;
    this.isReady = true;
    this.values = new Map();
    this.active = new Map();
  }

  on() {}
  async ping() { return "PONG"; }
  async get(key) { return this.values.get(key) ?? null; }
  async set(key, value) { this.values.set(key, value); return "OK"; }
  async del(key) { return Number(this.values.delete(key)); }

  purge(now) {
    for (const [owner, expires] of this.active.entries()) {
      if (expires <= now) this.active.delete(owner);
    }
  }

  async eval(script, { keys, arguments: args }) {
    if (script.includes("ZREMRANGEBYSCORE")) {
      const [owner, nowRaw, expiresRaw, limitRaw, _ttl, hasSession, hasRequest] = args;
      const now = Number(nowRaw);
      this.purge(now);
      if (hasRequest === "1" && this.values.has(keys[2])) return -2;
      if (hasSession === "1" && this.values.has(keys[1])) return -3;
      if (this.active.size >= Number(limitRaw)) return 0;
      this.active.set(owner, Number(expiresRaw));
      if (hasSession === "1") this.values.set(keys[1], owner);
      if (hasRequest === "1") this.values.set(keys[2], owner);
      return 1;
    }
    if (script.includes("ZSCORE")) {
      const [owner, expiresRaw] = args;
      if (!this.active.has(owner)) return 0;
      this.active.set(owner, Number(expiresRaw));
      return 1;
    }
    if (script.includes("ZREM")) {
      const [owner, hasSession, hasRequest] = args;
      this.active.delete(owner);
      if (hasSession === "1" && this.values.get(keys[1]) === owner) this.values.delete(keys[1]);
      if (hasRequest === "1" && this.values.get(keys[2]) === owner) this.values.delete(keys[2]);
      return 1;
    }
    throw new Error("unexpected script");
  }
}

test("stores JSON in explicit local-memory degradation mode", async () => {
  const state = new SharedState({ enabled: false, namespace: "test:memory" });
  await state.setJson("session:1", { product: "冰箱", turns: 2 }, 60);
  assert.deepEqual(await state.getJson("session:1"), { product: "冰箱", turns: 2 });
  const health = state.health();
  assert.equal(health.mode, "memory");
  assert.equal(health.ready, false);
  assert.ok(health.memory_fallbacks >= 2);
});

test("enforces a shared concurrency limit across gateway instances", async () => {
  const client = new FakeRedisClient();
  const firstState = new SharedState({ client, enabled: true, instanceId: "first", namespace: "test:limit" });
  const secondState = new SharedState({ client, enabled: true, instanceId: "second", namespace: "test:limit" });
  const first = await firstState.acquireAdmission({ requestId: "r1", sessionId: "s1", limit: 1, waitTimeoutMs: 0, ttlMs: 5000 });
  const blocked = await secondState.acquireAdmission({ requestId: "r2", sessionId: "s2", limit: 1, waitTimeoutMs: 0, ttlMs: 5000 });
  assert.equal(first.backend, "redis");
  assert.equal(blocked, null);
  assert.equal(secondState.health().lease_rejected_global, 1);
  await first.release();
  const next = await secondState.acquireAdmission({ requestId: "r2", sessionId: "s2", limit: 1, waitTimeoutMs: 0, ttlMs: 5000 });
  assert.equal(next.backend, "redis");
  await next.release();
});

test("serializes sessions and rejects duplicate request ids across instances", async () => {
  const client = new FakeRedisClient();
  const firstState = new SharedState({ client, enabled: true, instanceId: "first", namespace: "test:identity" });
  const secondState = new SharedState({ client, enabled: true, instanceId: "second", namespace: "test:identity" });
  const first = await firstState.acquireAdmission({ requestId: "same-request", sessionId: "same-session", limit: 3, waitTimeoutMs: 0, ttlMs: 5000 });
  const duplicate = await secondState.acquireAdmission({ requestId: "same-request", sessionId: "other-session", limit: 3, waitTimeoutMs: 0, ttlMs: 5000 });
  const sameSession = await secondState.acquireAdmission({ requestId: "other-request", sessionId: "same-session", limit: 3, waitTimeoutMs: 0, ttlMs: 5000 });
  assert.equal(duplicate, null);
  assert.equal(sameSession, null);
  assert.equal(secondState.health().lease_rejected_duplicate, 1);
  assert.equal(secondState.health().lease_rejected_session, 1);
  await first.release();
});
