"use strict";

const test = require("node:test");
const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");

const root = __dirname;
const server = fs.readFileSync(path.join(root, "server.js"), "utf8");
const env = fs.readFileSync(path.join(root, ".env"), "utf8");
const activeBackend = JSON.parse(
  fs.readFileSync(path.join(root, "backend-switch", "backend-active.json"), "utf8"),
);

test("production gateway is locked to the single canonical 8014 runtime", () => {
  assert.match(server, /CANONICAL_RETRIEVAL_ORIGIN = "http:\/\/127\.0\.0\.1:8014"/);
  assert.doesNotMatch(server, /http:\/\/127\.0\.0\.1:8011/);
  assert.match(env, /^RAGV6_API_ORIGIN=http:\/\/127\.0\.0\.1:8014$/m);
  assert.match(env, /^RAGV6_CHAT_ORIGIN=http:\/\/127\.0\.0\.1:8014$/m);
  assert.deepEqual(activeBackend, {
    mode: "vnext-fast",
    origin: "http://127.0.0.1:8014",
  });
});

test("WeCom text callbacks use unified chat with history enabled by default", () => {
  assert.match(server, /invokeUnifiedChannelChat\(\{/);
  assert.match(server, /channel:\s*"wecom"/);
  assert.match(server, /sessionId:\s*`wecom:\$\{fromUser\}`/);
  assert.match(server, /useHistoryContext:\s*true/);
  assert.match(server, /return sendText\(res, 200, "success"\)/);
});

test("public gateway never exposes reviewed-answer bookkeeping as retrieval", () => {
  assert.doesNotMatch(server, /benchmark_answer_fallback/);
  assert.doesNotMatch(server, /exact_reference_answer/);
  assert.match(server, /const reviewedAnswer = Boolean\(data\.reviewed_answer\)/);
  assert.match(server, /latestAuditTrace = mergeAuditTrace\(latestAuditTrace, auditTrace\)/);
});
