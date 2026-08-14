"use strict";

const fs = require("fs");
const path = require("path");

const DEFAULT_CACHE_FILE = path.join(__dirname, "data", "recommended-answer-cache.json");

function normalizeRecommendedQuestion(value) {
  return String(value || "")
    .normalize("NFKC")
    .replace(/\r\n?/g, "\n")
    .trim()
    .replace(/\s+/g, " ")
    .toLocaleLowerCase("en-US");
}

// Fixed manual answers are imported from OCR-oriented source material.  Some
// source rows retain a title marker but lose the line break after it, e.g.
// `# 程序显示 程序显示区…`.  This is deterministic table normalization at
// cache-load time, not a model or UI guess: only proven manual title/body
// boundaries receive an inserted newline and every other answer character,
// including `<PIC>`, is retained verbatim.
function fixedManualTitleBoundary(fragment) {
  const line = String(fragment || "").split("\n", 1)[0];
  if (!line || line.trimStart() !== line) return null;
  const repeated = line.match(/^(.{2,36}?) \1(?=\S)/u);
  if (repeated) return repeated[1].length;
  const suffixRepeat = line.match(/^(.{2,36}?(?:功能|说明|数据|操作|设置|安装|维护|清洁|警告|注意事项|概览|部件|规格|步骤|存放|模式|显示|运行|安全|准备|故障排除)) (.+)$/u);
  if (suffixRepeat) {
    const title = suffixRepeat[1];
    const stem = title.replace(/(?:功能|说明|数据|操作|设置|安装|维护|清洁|警告|注意事项|概览|部件|规格|步骤|存放|模式|显示|运行|安全|准备|故障排除)$/u, "");
    if (stem && suffixRepeat[2].startsWith(stem)) return title.length;
  }
  const namedSection = line.match(/^(.{2,48}?(?:功能|说明|数据|操作|设置|安装|维护|清洁|保养|调节|调整|运行|模式|显示|按钮|部件|组件|装备|安全|警告|注意事项|概览|规格|步骤|存放|连接|排水|洗涤剂|洗涤块|亮碟剂|滤网|系统|程序|电池|温度|餐具|物品|建议|高度|停机|介绍|使用|检查|更换|拆卸|组装|充电|开机|关机|故障排除)) (?=\S)/u);
  if (namedSection) return namedSection[1].length;
  const productSubject = line.match(/^(.{2,64}?) (?=(?:机器|产品|设备|洗碗机|空调|冰箱|健身单车|健身追踪器|控制台|新产品|本机|本产品|本设备|该机|该产品|该设备|您的|你(?:的)?))/u);
  if (productSubject) return productSubject[1].length;
  const inlineList = line.match(/^(.{2,96}?)(?=(?:[・•]|<PIC>|[0-9]+[.、]))/u);
  if (inlineList) return inlineList[1].length;
  const direct = line.match(/^(.{2,44}?) (?=(?:本|该|此|这|通过|使用|按(?:下)?|请|将|可|需|为|在|从|要|如|若|当|对于|以下|图|[0-9]+[.、]|[-•*]|<PIC>))/u);
  if (direct) return direct[1].length;
  const english = line.match(/^([A-Z][A-Za-z0-9 /&'()_-]{2,64}) (?=(?:The|This|Your|To|When|If|Before|After|Use|Check|Press|Remove|Install|Do|Never|Push|As|One|Severe|Risk|<PIC>|[0-9]+\.))/);
  return english ? english[1].length : null;
}

function normalizeFixedManualHeadings(answer, entry) {
  if (String(entry?.answer_mode || "manual") !== "manual" || String(entry?.product || "") === "客服售后") return answer;
  const text = String(answer || "");
  const markers = [...text.matchAll(/(?<!#)(#{1,6}) +/g)];
  const insertions = new Set();
  markers.forEach((marker, index) => {
    const markerStart = marker.index || 0;
    const markerEnd = markerStart + marker[0].length;
    if (markerStart > 0 && text[markerStart - 1] !== "\n") insertions.add(markerStart);
    const nextStart = index + 1 < markers.length ? markers[index + 1].index : text.length;
    const boundary = fixedManualTitleBoundary(text.slice(markerEnd, nextStart));
    if (boundary !== null && text[markerEnd + boundary] !== "\n") insertions.add(markerEnd + boundary);
  });
  if (!insertions.size) return text;
  const points = [...insertions].sort((left, right) => left - right);
  let output = "";
  let cursor = 0;
  for (const point of points) {
    output += text.slice(cursor, point) + "\n";
    cursor = point;
  }
  return output + text.slice(cursor);
}

// A recommended answer is allowed to accelerate only a high-confidence
// standalone question.  These stop words remove the grammatical shell while
// retaining the product object and operation words (for example, "glove
// compartment", "jetski", and "use").  This is deliberately conservative:
// one shared noun such as "battery" is not enough to reuse a reviewed answer.
const ENGLISH_CACHE_STOP_WORDS = new Set([
  "a", "an", "and", "are", "as", "at", "be", "by", "can", "could", "do", "does",
  "for", "from", "had", "has", "have", "how", "i", "if", "in", "is", "it", "me",
  "my", "of", "on", "or", "please", "should", "so", "that", "the", "their", "them",
  "this", "to", "what", "when", "where", "which", "who", "why", "will", "with", "you",
  "your", "want", "would", "introduce", "tell", "show", "explain", "realize",
]);

function cacheContentTokens(value) {
  const normalized = normalizeRecommendedQuestion(value)
    .replace(/[“”‘’"'`]/g, " ")
    .replace(/[-_/()[\]{}:;,!?\.]+/g, " ");
  if (!/[a-z0-9]/i.test(normalized)) return [];
  return [...new Set((normalized.match(/[a-z0-9]+/gi) || [])
    .map((token) => token.toLocaleLowerCase("en-US"))
    .filter((token) => (token.length > 1 || token === "v") && !ENGLISH_CACHE_STOP_WORDS.has(token)))];
}

function editDistanceAtMostOne(left, right) {
  if (left === right) return true;
  if (Math.abs(left.length - right.length) > 1) return false;
  let differences = 0;
  let i = 0;
  let j = 0;
  while (i < left.length && j < right.length) {
    if (left[i] === right[j]) {
      i += 1;
      j += 1;
      continue;
    }
    differences += 1;
    if (differences > 1) return false;
    if (left.length > right.length) i += 1;
    else if (right.length > left.length) j += 1;
    else {
      i += 1;
      j += 1;
    }
  }
  return differences + (left.length - i) + (right.length - j) <= 1;
}

const FUZZY_CJK_STOP_WORDS = new Set([
  "请", "请问", "我", "想", "要", "想要", "你", "您", "这", "这款", "这个", "那个",
  "我的", "有", "吗", "呢", "么", "的", "了", "与", "和", "及", "在", "中", "里", "上",
  "下", "前", "后", "时", "是", "能", "可以", "是否", "哪些", "什么", "怎么", "怎样", "如何",
  "为什么", "为何", "介绍", "说明", "了解", "一下", "分别", "需要", "应该", "该",
]);

const FUZZY_CONCEPT_REPLACEMENTS = [
  [/(?:洗涤剂|洗涤块|洗涤粉|洗涤液|detergent tablets?|dishwasher tablets?)/gi, " detergent "],
  [/(?:洗碗机|dish\s*washer|dishwasher)/gi, " dishwasher "],
  [/(?:水上摩托|摩托艇|jet\s*ski|jetski|watercraft)/gi, " jetski "],
  [/(?:上艇|登艇|上船|board(?:ing)?)/gi, " board "],
  [/(?:转弯|转向|turning|steering|turn)/gi, " turn "],
  [/(?:添加|加入|放入|add(?:ing)?|put\s+in)/gi, " add "],
  [/(?:清洁|清洗|擦拭|clean(?:ing)?|wash(?:ing)?)/gi, " clean "],
  [/(?:安装|装配|组装|install(?:ing)?|assemble)/gi, " install "],
  [/(?:拆卸|取下|移除|remove|detach)/gi, " remove "],
  [/(?:充电|charging|charge)/gi, " charge "],
  [/(?:电池电量|电池|battery|power\s+level)/gi, " battery "],
  [/(?:指示灯|指示器|指示标志|indicator|status\s+light)/gi, " indicator "],
  [/(?:扶手|把手|手柄|handgrip|handle)/gi, " handle "],
  [/(?:烤架|烤网|grill\s*rack|rack)/gi, " grill_rack "],
  [/(?:餐具篮|刀叉篮|cutlery\s*basket)/gi, " cutlery_basket "],
  [/(?:冰箱|冰柜|refrigerator|fridge)/gi, " refrigerator "],
];

function fuzzyConceptTokens(value) {
  let normalized = normalizeRecommendedQuestion(value)
    .replace(/[“”"'`]/g, " ")
    .replace(/\s+/g, " ");
  for (const [pattern, replacement] of FUZZY_CONCEPT_REPLACEMENTS) {
    normalized = normalized.replace(pattern, replacement);
  }
  const tokens = [];
  for (const segment of normalized.replace(/[-_/()[\]{}:;,!?\.]+/g, " ").match(/[a-z0-9_]+|[\u3400-\u9fff]+/gi) || []) {
    if (/^[a-z0-9_]+$/i.test(segment)) {
      const token = segment.toLocaleLowerCase("en-US");
      if ((token.length > 1 || token === "v") && !ENGLISH_CACHE_STOP_WORDS.has(token)) tokens.push(token);
      continue;
    }
    if (FUZZY_CJK_STOP_WORDS.has(segment)) continue;
    if (segment.length <= 4) tokens.push(segment);
    for (let index = 0; index < segment.length - 1; index += 1) {
      const bigram = segment.slice(index, index + 2);
      if (!FUZZY_CJK_STOP_WORDS.has(bigram)) tokens.push(bigram);
    }
  }
  return [...new Set(tokens)];
}

function fuzzyCacheMatch(index, question) {
  const queryTokens = fuzzyConceptTokens(question);
  if (queryTokens.length < 2) return null;
  const candidates = [];
  for (const [key, answer] of index.entries()) {
    const candidateTokens = fuzzyConceptTokens(key);
    if (candidateTokens.length < 2) continue;
    const matchedQuery = new Set();
    const matchedCandidate = new Set();
    for (const [queryIndex, queryToken] of queryTokens.entries()) {
      const candidateIndex = candidateTokens.findIndex((candidateToken) => (
        queryToken === candidateToken
        || (queryToken.length >= 4 && candidateToken.length >= 4 && editDistanceAtMostOne(queryToken, candidateToken))
      ));
      if (candidateIndex >= 0) {
        matchedQuery.add(queryIndex);
        matchedCandidate.add(candidateIndex);
      }
    }
    const shared = matchedQuery.size;
    const queryCoverage = shared / queryTokens.length;
    const candidateCoverage = matchedCandidate.size / candidateTokens.length;
    const f1 = queryCoverage + candidateCoverage > 0
      ? (2 * queryCoverage * candidateCoverage) / (queryCoverage + candidateCoverage)
      : 0;
    // The query must be covered almost completely. The candidate may contain
    // extra sub-questions because some reviewed rows combine related asks.
    if (shared >= 2 && queryCoverage >= 0.82 && f1 >= 0.34) {
      candidates.push({ answer, score: 0.7 * queryCoverage + 0.3 * f1, shared });
    }
  }
  candidates.sort((left, right) => right.score - left.score || right.shared - left.shared);
  if (!candidates.length) return null;
  // Never guess when two reviewed answers are similarly plausible.
  if (candidates.length > 1 && candidates[0].score - candidates[1].score < 0.08) return null;
  return candidates[0].answer;
}

function hasRemoteMediaUrl(question) {
  return /https?:\/\/\S+/i.test(String(question || ""));
}

function hasHistoryContext(payload) {
  if (payload?.use_history_context || String(payload?.history_context || "").trim()) return true;
  const packet = payload?.context_packet;
  return Boolean(packet && typeof packet === "object" && Object.keys(packet).length);
}

function canUseRecommendedAnswer(payload) {
  return Boolean(
    payload
    && (!Array.isArray(payload.images) || payload.images.length === 0)
    && !hasRemoteMediaUrl(payload.question)
    && !hasHistoryContext(payload),
  );
}

function loadRecommendedAnswerCache(filePath = DEFAULT_CACHE_FILE) {
  try {
    const payload = JSON.parse(fs.readFileSync(filePath, "utf8"));
    if (payload?.schema_version !== 1 || !Array.isArray(payload.entries)) return new Map();
    const index = new Map();
    for (const entry of payload.entries) {
      const question = normalizeRecommendedQuestion(entry?.question);
      let answer = String(entry?.answer || "").trim();
      let pics = Array.isArray(entry.pics)
        ? entry.pics.map((value) => String(value || "").trim()).filter(Boolean)
        : [];
      // Older reviewed exports appended the image-id JSON array to the answer
      // string instead of filling `pics`.  Normalize that legacy shape once at
      // load time so QQ can serve the reviewed text and pictures together.
      const legacyPics = answer.match(/,\s*(\[(?:\s*"[^"]+"\s*,?)+\])\s*$/);
      if (legacyPics) {
        try {
          const parsedPics = JSON.parse(legacyPics[1]);
          if (!pics.length && Array.isArray(parsedPics)) {
            pics = parsedPics.map((value) => String(value || "").trim()).filter(Boolean);
          }
          answer = answer.slice(0, legacyPics.index).trim();
          if (String(entry?.question || "").trim().startsWith('"') && answer.startsWith('"')) {
            answer = answer.slice(1).trim();
          }
        } catch {
          // Keep the original answer if a malformed legacy suffix is found.
        }
      }
      answer = normalizeFixedManualHeadings(answer, entry);
      if (!entry?.cache_eligible || !question || !answer || index.has(question)) continue;
      index.set(question, Object.freeze({
        cacheId: String(entry.cache_id || ""),
        answer,
        pics: Object.freeze(pics),
        answerMode: String(entry.answer_mode || "manual"),
        source: String(entry.source || "recommended-answer-cache"),
      }));
    }
    return index;
  } catch {
    // A cache is optional acceleration. Invalid or absent data must never
    // block the normal RAG path.
    return new Map();
  }
}

function findRecommendedAnswer(index, payload, { allowFuzzy = false } = {}) {
  if (!canUseRecommendedAnswer(payload)) return null;
  const exact = index.get(normalizeRecommendedQuestion(payload.question));
  if (exact) return exact;
  if (!allowFuzzy) return null;
  // First try punctuation/quote/hyphen-insensitive content matching, then a
  // single-character typo.  Ambiguous matches are rejected instead of
  // gambling with a reviewed answer.
  return fuzzyCacheMatch(index, payload.question);
}

module.exports = {
  canUseRecommendedAnswer,
  findRecommendedAnswer,
  loadRecommendedAnswerCache,
  normalizeFixedManualHeadings,
  normalizeRecommendedQuestion,
};
