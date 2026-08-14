"use strict";

const assert = require("node:assert/strict");
const test = require("node:test");
const { ChannelConcurrencyManager } = require("./channel-concurrency");

function manager(overrides = {}) {
  return new ChannelConcurrencyManager({
    maxConcurrency: 2,
    queueTimeoutMs: 1_000,
    sessionQueueLimit: 1,
    channels: {
      web: { queueLimit: 2, weight: 2 },
      wecom: { queueLimit: 2, weight: 1 },
      qq: { queueLimit: 2, weight: 1 },
      api: { queueLimit: 1, weight: 1 },
    },
    ...overrides,
  });
}

test("enforces one shared concurrency limit and bounded channel queues", async () => {
  const capacity = manager();
  const first = await capacity.acquire({ channel: "web", requestId: "w1" });
  const second = await capacity.acquire({ channel: "qq", requestId: "q1" });
  const queued = capacity.acquire({ channel: "wecom", requestId: "c1" });
  const extra = capacity.acquire({ channel: "wecom", requestId: "c2" });
  const rejected = await capacity.acquire({ channel: "wecom", requestId: "c3" });

  assert.equal(rejected, null);
  assert.equal(capacity.snapshot().active, 2);
  assert.equal(capacity.snapshot().channels.wecom.queued, 2);

  first.release();
  const third = await queued;
  assert.equal(third.channel, "wecom");
  second.release();
  const fourth = await extra;
  third.release();
  fourth.release();
  assert.equal(capacity.snapshot().active, 0);
});

test("serializes the same session without blocking other sessions", async () => {
  const capacity = manager();
  const first = await capacity.acquire({ channel: "wecom", requestId: "c1", sessionId: "same" });
  const sameSession = capacity.acquire({ channel: "wecom", requestId: "c2", sessionId: "same" });
  const other = await capacity.acquire({ channel: "qq", requestId: "q1", sessionId: "other" });

  assert.equal(capacity.snapshot().active, 2);
  assert.equal(capacity.snapshot().channels.wecom.queued, 1);
  other.release();
  assert.equal(capacity.snapshot().active, 1);
  first.release();

  const next = await sameSession;
  assert.equal(next.sessionId, "same");
  next.release();
});

test("uses weighted round-robin when several channels are waiting", async () => {
  const capacity = manager({ maxConcurrency: 1 });
  const first = await capacity.acquire({ channel: "web", requestId: "w1" });
  const nextWeb = capacity.acquire({ channel: "web", requestId: "w2" });
  const nextWecom = capacity.acquire({ channel: "wecom", requestId: "c1" });
  const nextQq = capacity.acquire({ channel: "qq", requestId: "q1" });

  first.release();
  const webLease = await nextWeb;
  webLease.release();
  const wecomLease = await nextWecom;
  wecomLease.release();
  const qqLease = await nextQq;
  qqLease.release();

  assert.equal(capacity.snapshot().channels.web.completed, 2);
  assert.equal(capacity.snapshot().channels.wecom.completed, 1);
  assert.equal(capacity.snapshot().channels.qq.completed, 1);
});

test("expires stale queue entries and rejects duplicate request ids", async () => {
  const capacity = manager({ maxConcurrency: 1, queueTimeoutMs: 25 });
  const active = await capacity.acquire({ channel: "web", requestId: "same-id" });
  const duplicate = await capacity.acquire({ channel: "qq", requestId: "same-id" });
  const expired = await capacity.acquire({ channel: "qq", requestId: "queued-id" });

  assert.equal(duplicate, null);
  assert.equal(expired, null);
  assert.equal(capacity.snapshot().channels.qq.rejected_duplicate, 1);
  assert.equal(capacity.snapshot().channels.qq.timed_out, 1);
  active.release();
});

test("limits queued work from one session", async () => {
  const capacity = manager({ maxConcurrency: 1, sessionQueueLimit: 1 });
  const active = await capacity.acquire({ channel: "qq", requestId: "q1", sessionId: "session" });
  const queued = capacity.acquire({ channel: "qq", requestId: "q2", sessionId: "session" });
  const rejected = await capacity.acquire({ channel: "qq", requestId: "q3", sessionId: "session" });

  assert.equal(rejected, null);
  assert.equal(capacity.snapshot().channels.qq.rejected_session, 1);
  active.release();
  const next = await queued;
  next.release();
});
