"use strict";

const test = require("node:test");
const assert = require("node:assert/strict");
const {
  auditViewKind,
  hasMeaningfulAuditTrace,
  mergeAuditTrace,
} = require("./public/ragv6-ui/audit-contract");

test("empty final fields never erase live retrieval rankings", () => {
  const live = {
    query: { original: "question", sparse: "term", semantic: "question" },
    retrieval: { candidates: [{ chunk_id: "42", bm25_raw: 8.4, dense_cosine: 0.77 }] },
    evidence: { selected: [{ chunk_id: "42", tier: "core" }] },
  };
  const merged = mergeAuditTrace(live, {
    execution_path: "lightweight_rag",
    query: { sparse: "" },
    retrieval: { candidates: [] },
    evidence: { selected: [] },
  });

  assert.equal(merged.query.sparse, "term");
  assert.equal(merged.retrieval.candidates[0].dense_cosine, 0.77);
  assert.equal(merged.evidence.selected[0].chunk_id, "42");
});

test("partial final trace enriches the matching candidate", () => {
  const merged = mergeAuditTrace(
    { retrieval: { candidates: [{ chunk_id: "42", bm25_raw: 8.4 }] } },
    { retrieval: { candidates: [{ chunk_id: "42", final_rank: 1 }] } },
  );
  assert.deepEqual(merged.retrieval.candidates, [{ chunk_id: "42", bm25_raw: 8.4, final_rank: 1 }]);
});

test("answer-to-manual alignment survives live and final audit merging", () => {
  const live = {
    mode: "BM25 + Dense + RRF + rerank + evidence_replay",
    answer_evidence_alignment: {
      applied: true,
      matched_chunks: [{ chunk_id: "5329", heading: "使用前调节" }],
    },
  };
  const merged = mergeAuditTrace(live, {
    answer_evidence_alignment: {
      matched_chunks: [{ chunk_id: "5330", heading: "骑行姿势" }],
    },
  });

  assert.equal(hasMeaningfulAuditTrace(merged), true);
  assert.deepEqual(
    merged.answer_evidence_alignment.matched_chunks.map((item) => item.chunk_id),
    ["5330", "5329"],
  );
});

test("service and visual traces are meaningful and classified separately", () => {
  const service = {
    execution_path: "lightweight_service",
    classifier: { route: "service", strategy: "local_rule" },
  };
  const visual = {
    execution_path: "image_product_dual_terra_low_vector",
    visual_preroute: { vector_trace: { top_score: 0.84 } },
  };

  assert.equal(hasMeaningfulAuditTrace(service), true);
  assert.equal(hasMeaningfulAuditTrace(visual), true);
  assert.equal(auditViewKind(service), "customer_service");
  assert.equal(auditViewKind(visual), "visual_manual");
  assert.equal(auditViewKind({ execution_path: "lightweight_rag" }), "manual_rag");
});
