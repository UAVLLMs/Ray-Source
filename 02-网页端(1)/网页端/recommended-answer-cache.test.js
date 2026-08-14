"use strict";

const test = require("node:test");
const assert = require("node:assert/strict");
const {
  canUseRecommendedAnswer,
  findRecommendedAnswer,
  normalizeRecommendedQuestion,
} = require("./recommended-answer-cache");

test("recommended answers match only normalized exact text without request context", () => {
  const index = new Map([[normalizeRecommendedQuestion("How do I clean it?"), { answer: "Cached answer" }]]);
  assert.equal(findRecommendedAnswer(index, { question: "  How do I\nclean it?  ", images: [] }).answer, "Cached answer");
  assert.equal(findRecommendedAnswer(index, { question: "How should I clean it?", images: [] }), null);
});

test("high-confidence object wording reuses a reviewed answer, but weak overlap does not", () => {
  const index = new Map([
    [normalizeRecommendedQuestion('"What is a V-BeltHolder?"'), { answer: "V-belt answer" }],
    [normalizeRecommendedQuestion("How do I remove the battery?"), { answer: "Battery removal" }],
  ]);
  assert.equal(findRecommendedAnswer(index, { question: "What is a V-BeltHolder?", images: [] }, { allowFuzzy: true }).answer, "V-belt answer");
  assert.equal(findRecommendedAnswer(index, { question: "What is a V-BeltHolder?", images: [] }), null);
  assert.equal(findRecommendedAnswer(index, { question: "How should I charge the battery?", images: [] }, { allowFuzzy: true }), null);
});

test("media and conversation context always retain the live RAG path", () => {
  assert.equal(canUseRecommendedAnswer({ question: "https://example.com/photo.jpg", images: [] }), false);
  assert.equal(canUseRecommendedAnswer({ question: "Question", images: ["data:image/png;base64,AA"] }), false);
  assert.equal(canUseRecommendedAnswer({ question: "Question", use_history_context: true, images: [] }), false);
  assert.equal(canUseRecommendedAnswer({ question: "Question", context_packet: { recent_turns: [] }, images: [] }), false);
});
