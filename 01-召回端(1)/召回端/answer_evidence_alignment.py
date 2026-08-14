"""Deterministic final-answer to manual-evidence alignment.

This module is deliberately independent from retrieval ranking.  BM25, Dense,
RRF and rerank scores describe how candidate chunks were retrieved; they must
never be rewritten merely because a reviewed answer is replayed.  The aligner
adds a separate, auditable layer that identifies which original manual chunks
support the visible answer, using exact picture anchors and conservative
same-language lexical overlap.
"""

from __future__ import annotations

import math
import re
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Iterable


_PIC_MARKER_RE = re.compile(r"<PIC(?::[^>]+)?>|\[\[PIC:[^\]]+\]\]", re.IGNORECASE)
_SOURCE_STOP_WORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "can", "do", "for",
    "from", "how", "in", "is", "it", "of", "on", "or", "the", "this", "to",
    "use", "using", "what", "when", "with", "you", "your",
}


def _clean_text(value: str) -> str:
    without_pics = _PIC_MARKER_RE.sub(" ", str(value or ""))
    return re.sub(r"[^a-z0-9\u3400-\u9fff%]+", " ", without_pics.lower()).strip()


def _compact(value: str) -> str:
    return re.sub(r"\s+", "", _clean_text(value))


def _tokens(value: str) -> frozenset[str]:
    cleaned = _clean_text(value)
    result = {
        word
        for word in re.findall(r"[a-z0-9][a-z0-9.+%/-]*", cleaned)
        if len(word) >= 2 and word not in _SOURCE_STOP_WORDS
    }
    for sequence in re.findall(r"[\u3400-\u9fff]+", cleaned):
        if len(sequence) == 1:
            continue
        if len(sequence) <= 4:
            result.add(sequence)
        result.update(sequence[index:index + 2] for index in range(len(sequence) - 1))
    return frozenset(result)


def _ngrams(value: str, size: int = 3) -> frozenset[str]:
    compact = _compact(value)
    return frozenset(
        compact[index:index + size]
        for index in range(max(0, len(compact) - size + 1))
    )


def _picture_ids(chunk: dict[str, Any]) -> tuple[str, ...]:
    values: list[str] = []
    for field in ("pics", "linked_pics", "evidence_pics"):
        for item in chunk.get(field) or []:
            value = str(item or "").strip()
            if value and value not in values:
                values.append(value)
    return tuple(values)


def _anchor_picture_markers(text: str, picture_ids: Iterable[str]) -> str:
    anchored = str(text or "")
    for picture_id in picture_ids:
        anchored = re.sub(
            r"<PIC(?::[^>]+)?>",
            f"[[PIC:{picture_id}]]",
            anchored,
            count=1,
            flags=re.IGNORECASE,
        )
    return anchored


def _answer_blocks(answer: str) -> list[str]:
    blocks: list[str] = []
    for raw in re.split(r"\n\s*\n+", str(answer or "")):
        block = _PIC_MARKER_RE.sub(" ", raw).strip()
        block = re.sub(r"^[#*_`\s]+|[#*_`\s]+$", "", block).strip()
        if len(_compact(block)) >= 8:
            blocks.append(block)
    return blocks


@dataclass(frozen=True)
class _IndexedChunk:
    document_order: int
    chunk_id: str
    product: str
    heading: str
    text: str
    anchored_text: str
    direct_pics: tuple[str, ...]
    all_pics: tuple[str, ...]
    compact: str
    tokens: frozenset[str]
    grams: frozenset[str]
    parent_section_id: Any


class AnswerEvidenceAligner:
    """Pre-index manual chunks and conservatively align a visible answer."""

    def __init__(self, chunks: Iterable[dict[str, Any]]) -> None:
        self._chunks: list[_IndexedChunk] = []
        self._by_product: dict[str, list[int]] = defaultdict(list)
        self._by_picture: dict[str, list[int]] = defaultdict(list)
        for document_order, chunk in enumerate(chunks):
            text = str(chunk.get("text") or "").strip()
            chunk_id = str(chunk.get("chunk_id") or "").strip()
            product = str(chunk.get("product") or "").strip()
            if not text or not chunk_id or not product:
                continue
            direct_pics = tuple(
                dict.fromkeys(
                    str(item or "").strip()
                    for item in (chunk.get("pics") or [])
                    if str(item or "").strip()
                )
            )
            all_pics = _picture_ids(chunk)
            indexed = _IndexedChunk(
                document_order=document_order,
                chunk_id=chunk_id,
                product=product,
                heading=str(chunk.get("heading") or "").strip(),
                text=text,
                anchored_text=_anchor_picture_markers(text, direct_pics),
                direct_pics=direct_pics,
                all_pics=all_pics,
                compact=_compact(text),
                tokens=_tokens(text),
                grams=_ngrams(text),
                parent_section_id=chunk.get("parent_section_id"),
            )
            index = len(self._chunks)
            self._chunks.append(indexed)
            self._by_product[product].append(index)
            for picture_id in all_pics:
                self._by_picture[picture_id.casefold()].append(index)

    @staticmethod
    def _score(block: str, chunk: _IndexedChunk) -> dict[str, Any]:
        block_compact = _compact(block)
        block_tokens = _tokens(block)
        block_grams = _ngrams(block)
        token_hits = len(block_tokens & chunk.tokens)
        gram_hits = len(block_grams & chunk.grams)
        token_recall = token_hits / max(1, len(block_tokens))
        gram_recall = gram_hits / max(1, len(block_grams))
        gram_precision = gram_hits / max(1, len(chunk.grams))
        exact = bool(
            min(len(block_compact), len(chunk.compact)) >= 8
            and (block_compact in chunk.compact or chunk.compact in block_compact)
        )
        balanced = math.sqrt(gram_recall * gram_precision) if gram_hits else 0.0
        score = gram_recall * 0.55 + token_recall * 0.25 + balanced * 0.20
        if exact:
            score = max(score, 0.96)
        return {
            "score": score,
            "exact": exact,
            "token_hits": token_hits,
            "gram_hits": gram_hits,
            "token_recall": token_recall,
            "gram_recall": gram_recall,
        }

    def align(
        self,
        *,
        answer: str,
        picture_ids: Iterable[str] = (),
        preferred_products: Iterable[str] = (),
        max_chunks: int = 8,
    ) -> dict[str, Any]:
        requested_pics = list(dict.fromkeys(
            str(item or "").strip() for item in picture_ids if str(item or "").strip()
        ))
        preferred = [
            product for product in dict.fromkeys(str(item or "").strip() for item in preferred_products)
            if product in self._by_product
        ]
        picture_candidate_ids = {
            index
            for picture_id in requested_pics
            for index in self._by_picture.get(picture_id.casefold(), [])
        }
        picture_products = list(dict.fromkeys(
            self._chunks[index].product for index in sorted(picture_candidate_ids)
        ))
        candidate_products = list(dict.fromkeys([*picture_products, *preferred]))
        if candidate_products:
            candidate_ids = {
                index for product in candidate_products for index in self._by_product.get(product, [])
            }
        else:
            candidate_ids = set(range(len(self._chunks)))

        selected: dict[int, dict[str, Any]] = {}
        matched_pics: list[str] = []
        missing_pics: list[str] = []

        def select(index: int, reason: str, score: float = 1.0, block_index: int | None = None) -> None:
            record = selected.setdefault(index, {
                "score": 0.0,
                "reasons": [],
                "answer_blocks": [],
            })
            record["score"] = max(float(record["score"]), float(score))
            if reason not in record["reasons"]:
                record["reasons"].append(reason)
            if block_index is not None and block_index not in record["answer_blocks"]:
                record["answer_blocks"].append(block_index)

        # A source picture ID is the strongest possible structural link. Prefer
        # the chunk that owns the picture directly over a neighboring linked-pic
        # expansion, then retain source order.
        for picture_id in requested_pics:
            matches = list(self._by_picture.get(picture_id.casefold(), []))
            if not matches:
                missing_pics.append(picture_id)
                continue
            matches.sort(key=lambda index: (
                picture_id.casefold() not in {
                    item.casefold() for item in self._chunks[index].direct_pics
                },
                self._chunks[index].product not in preferred,
                self._chunks[index].document_order,
            ))
            select(matches[0], f"picture:{picture_id}", score=1.0)
            matched_pics.append(picture_id)

        blocks = _answer_blocks(answer)
        matched_blocks: set[int] = set()
        for block_index, block in enumerate(blocks):
            scored = [
                (index, self._score(block, self._chunks[index]))
                for index in candidate_ids
            ]
            if not scored:
                continue
            best_index, best = max(
                scored,
                key=lambda item: (float(item[1]["score"]), -self._chunks[item[0]].document_order),
            )
            strong = bool(
                best["exact"]
                or (
                    best["gram_hits"] >= 3
                    and best["score"] >= 0.14
                    and (best["gram_recall"] >= 0.16 or best["token_recall"] >= 0.25)
                )
            )
            if strong:
                select(
                    best_index,
                    "literal_overlap",
                    score=float(best["score"]),
                    block_index=block_index,
                )
                matched_blocks.add(block_index)
                # A long answer paragraph can faithfully combine several
                # adjacent source chunks under one manual heading.  Greedily
                # retain only same-heading chunks that add new literal grams;
                # this recovers a complete continuous source unit without
                # admitting neighboring topics or generic full-manual matches.
                if len(_compact(block)) >= 120:
                    block_grams = set(_ngrams(block))
                    covered_grams = block_grams & set(self._chunks[best_index].grams)
                    continuation_pool = [
                        (index, score, (block_grams & set(self._chunks[index].grams)) - covered_grams)
                        for index, score in scored
                        if index != best_index
                        and self._chunks[index].heading == self._chunks[best_index].heading
                        and score["gram_hits"] >= 5
                    ]
                    for _ in range(4):
                        if not continuation_pool:
                            break
                        continuation_pool.sort(
                            key=lambda item: (len(item[2]), float(item[1]["score"])),
                            reverse=True,
                        )
                        next_index, next_score, new_grams = continuation_pool.pop(0)
                        min_new_grams = max(5, round(len(block_grams) * 0.02))
                        if len(new_grams) < min_new_grams:
                            break
                        select(
                            next_index,
                            "literal_overlap_continuation",
                            score=float(next_score["score"]),
                            block_index=block_index,
                        )
                        covered_grams.update(new_grams)
                        continuation_pool = [
                            (index, score, (block_grams & set(self._chunks[index].grams)) - covered_grams)
                            for index, score, _unused in continuation_pool
                        ]

        # If paragraph boundaries were lost in serialization, one conservative
        # whole-answer match still provides a useful source without expanding to
        # weak lexical neighbors.
        if not selected and answer and candidate_ids:
            scored = [
                (index, self._score(answer, self._chunks[index]))
                for index in candidate_ids
            ]
            best_index, best = max(scored, key=lambda item: float(item[1]["score"]))
            if best["exact"] or (best["gram_hits"] >= 4 and best["score"] >= 0.12):
                select(best_index, "whole_answer_overlap", score=float(best["score"]))

        mandatory_ids = {
            index
            for index, match in selected.items()
            if any(str(reason).startswith("picture:") for reason in match["reasons"])
        }
        optional_ids = sorted(
            (index for index in selected if index not in mandatory_ids),
            key=lambda index: (
                -float(selected[index]["score"]),
                self._chunks[index].document_order,
            ),
        )
        effective_limit = max(1, int(max_chunks), len(mandatory_ids))
        retained_ids = mandatory_ids | set(optional_ids[:max(0, effective_limit - len(mandatory_ids))])
        ranked_ids = sorted(retained_ids, key=lambda index: self._chunks[index].document_order)
        matched_chunks: list[dict[str, Any]] = []
        for index in ranked_ids:
            chunk = self._chunks[index]
            match = selected[index]
            matched_chunks.append({
                "chunk_id": chunk.chunk_id,
                "product": chunk.product,
                "heading": chunk.heading,
                "matched_content": chunk.anchored_text,
                "content": chunk.anchored_text,
                "pics": list(chunk.direct_pics),
                "parent_section_id": chunk.parent_section_id,
                "document_order": chunk.document_order,
                "evidence_role": "answer_aligned",
                "alignment_score": round(float(match["score"]), 6),
                "match_reasons": list(match["reasons"]),
                "answer_blocks": sorted(match["answer_blocks"]),
            })

        return {
            "applied": bool(str(answer or "").strip()),
            "method": "literal_overlap_and_picture_anchor",
            "matched_chunks": matched_chunks,
            "matched_chunk_count": len(matched_chunks),
            "answer_block_coverage": {
                "total": len(blocks),
                "matched": len(matched_blocks),
            },
            "picture_coverage": {
                "required": requested_pics,
                "matched": matched_pics,
                "missing": missing_pics,
            },
            "candidate_products": candidate_products,
        }


def public_alignment_trace(alignment: dict[str, Any]) -> dict[str, Any]:
    """Remove source body text from the live SSE audit snapshot."""

    result = {key: value for key, value in alignment.items() if key != "matched_chunks"}
    result["matched_chunks"] = [
        {
            key: value
            for key, value in chunk.items()
            if key not in {"content", "matched_content"}
        }
        for chunk in alignment.get("matched_chunks", [])
    ]
    return result
