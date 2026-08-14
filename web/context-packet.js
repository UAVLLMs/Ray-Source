"use strict";

const CONTEXT_PACKET_VERSION = 1;
const MAX_RECENT_TURNS = 8;
const MAX_MEDIA_FACTS = 12;
const MAX_CONSTRAINTS = 8;
const HISTORY_ONLY_RE = /(?:只|仅).{0,12}(?:根据|使用).{0,12}(?:上一轮|上轮|历史|刚才|前面).{0,20}(?:回答|复述|说明)|(?:不要|无需|不用).{0,12}(?:重新)?(?:检索|搜索|查(?:询)?手册)/i;

function compact(value, limit) {
  const text = String(value || "").replace(/\s+/g, " ").trim();
  return text.length <= limit ? text : `${text.slice(0, Math.max(0, limit - 1)).trim()}…`;
}

function uniqueStrings(values, limit, count) {
  const rows = [];
  const seen = new Set();
  for (const value of Array.isArray(values) ? values : []) {
    const text = compact(value, limit);
    const key = text.toLowerCase();
    if (!text || seen.has(key)) continue;
    rows.push(text);
    seen.add(key);
    if (rows.length >= count) break;
  }
  return rows;
}

function normalizeContextPacket(value) {
  if (!value || typeof value !== "object" || Array.isArray(value)) return {};
  const entities = {};
  for (const key of ["product", "model", "component", "symptom"]) {
    const text = compact(value.entities?.[key], 160);
    if (text) entities[key] = text;
  }
  const mediaFacts = [];
  const factSeen = new Set();
  for (const item of Array.isArray(value.media_facts) ? value.media_facts : []) {
    const fact = compact(typeof item === "object" ? item.fact : item, 360);
    const key = fact.toLowerCase();
    if (!fact || factSeen.has(key)) continue;
    const row = { fact };
    const source = compact(typeof item === "object" ? item.source : "", 120);
    const confidence = compact(typeof item === "object" ? item.confidence : "", 20).toLowerCase();
    if (source) row.source = source;
    if (["high", "medium", "low"].includes(confidence)) row.confidence = confidence;
    mediaFacts.push(row);
    factSeen.add(key);
    if (mediaFacts.length >= MAX_MEDIA_FACTS) break;
  }
  const recentTurns = [];
  for (const item of Array.isArray(value.recent_turns) ? value.recent_turns : []) {
    const role = String(item?.role || "").toLowerCase();
    const content = compact(item?.content, 420);
    if (!["user", "assistant"].includes(role) || !content) continue;
    recentTurns.push({ role, content });
    if (recentTurns.length >= MAX_RECENT_TURNS) break;
  }
  const retrievalHint = ["auto", "history_only", "required"].includes(value.retrieval_hint)
    ? value.retrieval_hint
    : "auto";
  const packet = { version: CONTEXT_PACKET_VERSION, retrieval_hint: retrievalHint };
  const summary = compact(value.summary, 700);
  if (summary) packet.summary = summary;
  if (Object.keys(entities).length) packet.entities = entities;
  if (mediaFacts.length) packet.media_facts = mediaFacts;
  const constraints = uniqueStrings(value.user_constraints, 220, MAX_CONSTRAINTS);
  if (constraints.length) packet.user_constraints = constraints;
  if (recentTurns.length) packet.recent_turns = recentTurns;
  return packet;
}

function extractConstraints(question) {
  return uniqueStrings(
    String(question || "")
      .split(/[。！？!?；;\n]+/)
      .filter((part) => /(?:只|仅|不要|不得|必须|无需|不用|优先|禁止)/.test(part)),
    220,
    MAX_CONSTRAINTS,
  );
}

function extractModel(turns) {
  const text = turns.slice(-4).map((turn) => turn.question || "").join(" ");
  const match = text.match(/\b[A-Z]{2,}[A-Z0-9-]*\d[A-Z0-9-]*\b/i);
  return match ? match[0].toUpperCase() : "";
}

function buildContextPacket({ turns, product, currentQuestion, clientPacket }) {
  const boundedTurns = (Array.isArray(turns) ? turns : []).slice(-4);
  const client = normalizeContextPacket(clientPacket);
  const recentTurns = [];
  const mediaFacts = [];
  const constraints = [];
  boundedTurns.forEach((turn, index) => {
    const source = `turn_${index + 1}`;
    const question = compact(turn.question, 420);
    const answer = compact(turn.answer, 420);
    if (question) recentTurns.push({ role: "user", content: question });
    if (answer) recentTurns.push({ role: "assistant", content: answer });
    constraints.push(...extractConstraints(question));
    const imageDescriptions = Array.isArray(turn.imageDescriptions) ? turn.imageDescriptions : [];
    const confidenceMatch = imageDescriptions.join(" ").match(/视觉识别置信度[：:]\s*(high|medium|low)/i);
    const visualConfidence = confidenceMatch ? confidenceMatch[1].toLowerCase() : "medium";
    for (const fact of imageDescriptions) {
      mediaFacts.push({ fact: compact(fact, 360), source: `${source}.image`, confidence: "high" });
      mediaFacts.at(-1).confidence = visualConfidence;
    }
  });
  const mergedRecent = recentTurns.length ? recentTurns : (client.recent_turns || []);
  const mergedFacts = mediaFacts.length ? mediaFacts : (client.media_facts || []);
  const mergedConstraints = uniqueStrings(
    [...constraints, ...(client.user_constraints || [])],
    220,
    MAX_CONSTRAINTS,
  );
  const entities = { ...(client.entities || {}) };
  if (product) entities.product = compact(product, 160);
  const component = [...boundedTurns]
    .reverse()
    .map((turn) => compact(turn?.component, 160))
    .find(Boolean);
  if (component) entities.component = component;
  const model = extractModel(boundedTurns);
  if (model) entities.model = model;
  const lastQuestion = boundedTurns.length ? compact(boundedTurns.at(-1).question, 180) : "";
  const summary = compact(
    client.summary || [
      product && `产品：${product}`,
      component && `当前部件：${component}`,
      lastQuestion && `最近问题：${lastQuestion}`,
    ].filter(Boolean).join("；"),
    700,
  );
  return normalizeContextPacket({
    version: CONTEXT_PACKET_VERSION,
    summary,
    entities,
    media_facts: mergedFacts,
    user_constraints: mergedConstraints,
    recent_turns: mergedRecent,
    retrieval_hint: HISTORY_ONLY_RE.test(String(currentQuestion || "")) ? "history_only" : "auto",
  });
}

module.exports = {
  HISTORY_ONLY_RE,
  buildContextPacket,
  extractConstraints,
  normalizeContextPacket,
};
