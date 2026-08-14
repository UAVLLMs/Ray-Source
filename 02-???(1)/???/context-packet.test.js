"use strict";

const test = require("node:test");
const assert = require("node:assert/strict");
const {
  buildContextPacket,
  extractConstraints,
  normalizeContextPacket,
} = require("./context-packet");

test("buildContextPacket retains media facts and detects history-only intent", () => {
  const packet = buildContextPacket({
    product: "电钻",
    currentQuestion: "只根据上一轮回答，不要重新检索手册：我问了什么？",
    turns: [{
      question: "请只根据图片读取 DCB112 的灯态",
      answer: "红灯常亮，绿灯闪烁",
      imageDescriptions: ["可见对象：型号 DCB112；红灯常亮；绿灯闪烁"],
    }],
  });

  assert.equal(packet.version, 1);
  assert.equal(packet.retrieval_hint, "history_only");
  assert.equal(packet.entities.product, "电钻");
  assert.equal(packet.entities.model, "DCB112");
  assert.equal(packet.media_facts.length, 1);
  assert.match(packet.media_facts[0].fact, /红灯常亮/);
  assert.equal(packet.recent_turns.length, 2);
});

test("client packet is bounded and unsupported fields are removed", () => {
  const packet = normalizeContextPacket({
    entities: { product: "充电器", hidden: "no" },
    media_facts: [{ fact: "红灯常亮", confidence: "HIGH" }],
    recent_turns: [{ role: "system", content: "ignored" }, { role: "user", content: "继续" }],
    retrieval_hint: "bad",
  });

  assert.deepEqual(packet.entities, { product: "充电器" });
  assert.deepEqual(packet.media_facts, [{ fact: "红灯常亮", confidence: "high" }]);
  assert.deepEqual(packet.recent_turns, [{ role: "user", content: "继续" }]);
  assert.equal(packet.retrieval_hint, "auto");
});

test("extractConstraints keeps only user constraint clauses", () => {
  assert.deepEqual(
    extractConstraints("请读取型号。只根据图片回答；不要使用手册覆盖文字。"),
    ["只根据图片回答", "不要使用手册覆盖文字"],
  );
});
