(function initAuditContract(root, factory) {
  const api = factory();
  if (typeof module === "object" && module.exports) module.exports = api;
  if (root) root.RagAuditContract = api;
}(typeof globalThis !== "undefined" ? globalThis : this, function buildAuditContract() {
  "use strict";

  const KEYED_ARRAYS = new Set([
    "retrieval.candidates",
    "evidence.selected",
    "answer_evidence_alignment.matched_chunks",
    "events",
    "media_ingest.errors",
    "visual_preroute.vector_trace.hits",
  ]);

  function isObject(value) {
    return Boolean(value) && typeof value === "object" && !Array.isArray(value);
  }

  function clone(value) {
    if (Array.isArray(value)) return value.map(clone);
    if (!isObject(value)) return value;
    return Object.fromEntries(Object.entries(value).map(([key, item]) => [key, clone(item)]));
  }

  function nonEmptyText(value) {
    return typeof value === "string" && value.trim().length > 0;
  }

  function objectHasFields(value) {
    return isObject(value) && Object.values(value).some((item) => (
      nonEmptyText(item)
      || typeof item === "number"
      || item === true
      || (Array.isArray(item) && item.length > 0)
      || objectHasFields(item)
    ));
  }

  function hasMeaningfulAuditTrace(trace) {
    if (!isObject(trace)) return false;
    const query = isObject(trace.query) ? trace.query : {};
    const retrieval = isObject(trace.retrieval) ? trace.retrieval : {};
    const evidence = isObject(trace.evidence) ? trace.evidence : {};
    return Boolean(
      nonEmptyText(trace.execution_path)
      || nonEmptyText(trace.mode)
      || nonEmptyText(trace.status)
      || nonEmptyText(trace.original_query)
      || nonEmptyText(query.original)
      || nonEmptyText(query.sparse)
      || nonEmptyText(query.semantic)
      || (Array.isArray(retrieval.candidates) && retrieval.candidates.length)
      || (Array.isArray(evidence.selected) && evidence.selected.length)
      || objectHasFields(trace.route)
      || objectHasFields(trace.product_route)
      || objectHasFields(trace.classifier)
      || objectHasFields(trace.media_ingest)
      || objectHasFields(trace.visual_preroute)
      || objectHasFields(trace.manual_mode_input)
      || objectHasFields(trace.timings)
      || objectHasFields(trace.generation_evidence_budget)
      || objectHasFields(trace.structural_picture_binding)
      || objectHasFields(trace.answer_evidence_alignment)
    );
  }

  function arrayItemKey(path, item, index) {
    if (!isObject(item)) return `${typeof item}:${String(item)}`;
    if (path === "retrieval.candidates") return String(item.chunk_id ?? item.matched_chunk_id ?? index);
    if (path === "evidence.selected") return String(item.chunk_id ?? index);
    if (path === "answer_evidence_alignment.matched_chunks") return String(item.chunk_id ?? index);
    if (path === "events") return String(item.kind || item.name || `${item.stage || ""}:${index}`);
    if (path === "media_ingest.errors") return String(item.url || item.error || index);
    if (path === "visual_preroute.vector_trace.hits") return String(item.image_id || `${item.product || ""}:${index}`);
    return String(index);
  }

  function mergeArrays(previous, incoming, path) {
    if (!incoming.length) return clone(previous);
    if (!previous.length || !KEYED_ARRAYS.has(path)) return clone(incoming);

    const incomingKeys = new Set(incoming.map((item, index) => arrayItemKey(path, item, index)));
    const previousByKey = new Map(previous.map((item, index) => [arrayItemKey(path, item, index), item]));
    const merged = incoming.map((item, index) => {
      const key = arrayItemKey(path, item, index);
      return mergeValue(previousByKey.get(key), item, `${path}[]`);
    });
    for (let index = 0; index < previous.length; index += 1) {
      const item = previous[index];
      if (!incomingKeys.has(arrayItemKey(path, item, index))) merged.push(clone(item));
    }
    return merged;
  }

  function mergeValue(previous, incoming, path) {
    if (incoming === undefined || incoming === null || incoming === "") return clone(previous);
    if (Array.isArray(incoming)) {
      return mergeArrays(Array.isArray(previous) ? previous : [], incoming, path);
    }
    if (!isObject(incoming)) return incoming;

    const base = isObject(previous) ? clone(previous) : {};
    for (const [key, value] of Object.entries(incoming)) {
      const childPath = path ? `${path}.${key}` : key;
      base[key] = mergeValue(base[key], value, childPath);
    }
    return base;
  }

  function mergeAuditTrace(previous, incoming) {
    if (!isObject(incoming) || !Object.keys(incoming).length) return clone(previous) || null;
    return mergeValue(isObject(previous) ? previous : {}, incoming, "");
  }

  function auditViewKind(trace, record) {
    const value = isObject(trace) ? trace : {};
    const executionPath = String(value.execution_path || "").toLowerCase();
    const classifierRoute = String((value.classifier || {}).route || "").toLowerCase();
    const summaryMode = String((record?.summary || {}).modeKey || "").toLowerCase();
    if (
      executionPath.includes("service")
      || classifierRoute === "service"
      || summaryMode === "customer"
    ) return "customer_service";

    const media = value.media_ingest || {};
    const visual = value.visual_preroute || {};
    const manualInput = value.manual_mode_input || {};
    const mediaCount = Number(media.input_image_count || 0)
      + Number(media.resolved_image_count || 0)
      + (Array.isArray(media.discovered_urls) ? media.discovered_urls.length : 0);
    if (
      record?.requestKind === "visual_manual"
      || executionPath.includes("image_")
      || executionPath.includes("link_")
      || Boolean(manualInput.has_image || manualInput.has_link)
      || mediaCount > 0
      || visual.used === true
      || objectHasFields(visual.vector_trace)
    ) return "visual_manual";
    return "manual_rag";
  }

  return {
    auditViewKind,
    hasMeaningfulAuditTrace,
    mergeAuditTrace,
  };
}));
