"""Set-wise image evidence selection and text-image binding for manual RAG.

The answer model is good at writing natural responses, but free-form generation is
not a reliable place to decide a *set* of figures.  This module therefore treats
every manual figure and its neighbouring source text as one evidence unit.  A
separate planning pass chooses the smallest sufficient set, and a composing pass
is only used when the selected set differs from the answer model's anchors.

No question ids or reference-answer image lists are used here.  All metadata comes
from the parsed manuals and automatically generated image descriptions.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Iterable


_ANCHOR_RE = re.compile(r"\[\[PIC:([^\]\r\n]+)\]\]")
_JSON_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.IGNORECASE)


@dataclass(frozen=True)
class OrderedRangeRequest:
    """Explicit slice requested from a source-defined ordered list.

    The unit is deliberately restricted to numbered ``steps`` or ``items``.
    Generic procedure wording is not enough to create this object, so ordinary
    how-to questions continue through the normal semantic evidence selector.
    """

    start: int | None
    end: int | None
    from_end: bool
    unit: str
    matched_text: str


_ZH_NUMERALS = {
    "一": 1, "二": 2, "两": 2, "三": 3, "四": 4, "五": 5,
    "六": 6, "七": 7, "八": 8, "九": 9, "十": 10,
}


def _parse_small_count(value: str) -> int | None:
    """Parse the small cardinal numbers used by manual range questions."""

    value = (value or "").strip().lower()
    if value.isdigit():
        number = int(value)
        return number if 1 <= number <= 99 else None
    english = {
        "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
        "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
    }
    if value in english:
        return english[value]
    if value in _ZH_NUMERALS:
        return _ZH_NUMERALS[value]
    if value.startswith("十") and len(value) == 2 and value[1] in _ZH_NUMERALS:
        return 10 + _ZH_NUMERALS[value[1]]
    if len(value) == 2 and value[0] in _ZH_NUMERALS and value[1] == "十":
        return _ZH_NUMERALS[value[0]] * 10
    if len(value) == 3 and value[0] in _ZH_NUMERALS and value[1] == "十" and value[2] in _ZH_NUMERALS:
        return _ZH_NUMERALS[value[0]] * 10 + _ZH_NUMERALS[value[2]]
    return None


def _parse_ordered_range_request(question: str) -> OrderedRangeRequest | None:
    """Recognize only explicit direction/range + count + ordered-entry unit.

    Requiring all three signals is the false-positive boundary. Expressions such
    as ``有哪些步骤`` and ``how to`` contain an entry unit but no bounded range,
    and therefore intentionally return ``None``.
    """

    q = _normalise_text(question).lower()
    number = r"(?:\d{1,2}|[一二两三四五六七八九十]{1,3}|one|two|three|four|five|six|seven|eight|nine|ten)"
    unit = r"(?:步骤|条|项|步|steps?|items?|points?)"

    between = re.search(
        rf"(?:第\s*)?(?P<start>{number})\s*(?:到|至|[-–—~])\s*(?:第\s*)?(?P<end>{number})\s*(?P<unit>{unit})",
        q,
    )
    if between:
        start = _parse_small_count(between.group("start"))
        end = _parse_small_count(between.group("end"))
        if start and end and start <= end:
            return OrderedRangeRequest(start, end, False, between.group("unit"), between.group(0))

    english_between = re.search(
        rf"(?P<unit>{unit})\s+(?P<start>{number})\s+(?:to|through)\s+(?P<end>{number})",
        q,
    )
    if english_between:
        start = _parse_small_count(english_between.group("start"))
        end = _parse_small_count(english_between.group("end"))
        if start and end and start <= end:
            return OrderedRangeRequest(start, end, False, english_between.group("unit"), english_between.group(0))

    prefix = re.search(
        rf"(?:(?P<zh>前|最初)\s*(?P<zh_count>{number})\s*(?:个)?\s*(?P<zh_unit>{unit})|"
        rf"(?P<en>first)\s+(?P<en_count>{number})\s+(?P<en_unit>{unit}))",
        q,
    )
    if prefix:
        count = _parse_small_count(prefix.group("zh_count") or prefix.group("en_count"))
        unit_value = prefix.group("zh_unit") or prefix.group("en_unit")
        if count:
            return OrderedRangeRequest(1, count, False, unit_value, prefix.group(0))

    suffix = re.search(
        rf"(?:(?P<zh>最后|末尾)\s*(?P<zh_count>{number})\s*(?:个)?\s*(?P<zh_unit>{unit})|"
        rf"(?P<en>last)\s+(?P<en_count>{number})\s+(?P<en_unit>{unit}))",
        q,
    )
    if suffix:
        count = _parse_small_count(suffix.group("zh_count") or suffix.group("en_count"))
        unit_value = suffix.group("zh_unit") or suffix.group("en_unit")
        if count:
            return OrderedRangeRequest(None, count, True, unit_value, suffix.group(0))
    return None


@dataclass(frozen=True)
class ImageEvidence:
    """One figure together with the source context that gives it meaning."""

    image_id: str
    product: str
    section_id: int
    heading: str
    section_summary: str
    before_text: str
    after_text: str
    image_category: str
    image_caption: str
    section_images: tuple[str, ...]
    section_order: int
    image_order: int
    origin: str = "retrieved"
    source_group_id: str = ""
    source_group_heading: str = ""
    source_group_context: str = ""
    source_group_images: tuple[str, ...] = ()
    explicit_citation_text: str = ""

    @property
    def document_order(self) -> tuple[int, int]:
        return self.section_order, self.image_order

    def prompt_payload(self) -> dict[str, object]:
        """Return a bounded prompt representation; full sections would add noise."""

        local = " ".join(
            part for part in (self.before_text[-420:], "<PIC>", self.after_text[:520]) if part
        )
        return {
            "image_id": self.image_id,
            "product": self.product,
            "section_id": self.section_id,
            "heading": self.heading,
            "section_summary": self.section_summary[:420],
            "source_around_image": re.sub(r"\s+", " ", local).strip(),
            "image_category": self.image_category,
            "visual_content": self.image_caption[:520],
            "section_images_in_order": list(self.section_images),
            "document_order": [self.section_order, self.image_order],
            "origin": self.origin,
            "source_group_id": self.source_group_id,
            "source_group_heading": self.source_group_heading,
            "source_group_context": self.source_group_context[:520],
            "source_group_images_in_order": list(self.source_group_images),
            "explicit_citation_text": self.explicit_citation_text[:620],
        }


@dataclass
class EvidenceSelectionResult:
    """Validated output returned to the agent finalization stage."""

    answer: str
    selected_images: list[str]
    changed: bool
    trace: dict[str, object]


_EVIDENCE_INDEX_CACHE: dict[tuple[int, int], tuple[dict[str, list[ImageEvidence]], list[ImageEvidence]]] = {}
_SOURCE_GROUP_CACHE: dict[str, tuple[dict[tuple[str, str], list[dict]], dict[str, dict]]] = {}


def extract_anchor_ids(answer: str) -> list[str]:
    """Extract inline image ids while preserving answer order and removing duplicates."""

    seen: set[str] = set()
    output: list[str] = []
    for image_id in _ANCHOR_RE.findall(answer or ""):
        image_id = image_id.strip()
        if image_id and image_id not in seen:
            seen.add(image_id)
            output.append(image_id)
    return output


def _normalise_text(value: str) -> str:
    return re.sub(r"\s+", " ", value or "").strip()


def _heading_parts(heading: str) -> tuple[str, ...]:
    return tuple(part.strip().lower() for part in (heading or "").split("/") if part.strip())


def _common_heading_prefix(left: str, right: str) -> int:
    a = _heading_parts(left)
    b = _heading_parts(right)
    count = 0
    for x, y in zip(a, b):
        if x != y:
            break
        count += 1
    return count


def _is_scope_wide_question(question: str) -> bool:
    """Detect questions whose answer can legitimately span sibling subsections.

    These markers describe scope, plurality, or a whole operation.  They are not
    product-specific and only control candidate expansion; the planner must still
    justify every selected figure against an explicit information requirement.
    """

    q = (question or "").lower()
    markers = (
        "有哪些",
        "包括什么",
        "组成",
        "介绍",
        "不同",
        "各自",
        "步骤",
        "如何",
        "怎样",
        "what are",
        "what kinds",
        "introduce",
        "different",
        "other functions",
        "requirements",
        "limitations",
        "characteristics",
        "how to",
        "how do i",
        "what should",
    )
    return any(marker in q for marker in markers)


def _is_pre_operation_readiness_question(question: str) -> bool:
    """Return whether the user asks for a broad checklist before normal use.

    This is a lifecycle/scope test, not a product vocabulary list.  It deliberately
    excludes narrow questions that merely happen to mention a point in time.
    """

    q = _normalise_text(question).lower()
    has_pre_operation_scope = bool(
        re.search(
            r"\b(?:before|prior to)\b.{0,48}\b(?:start(?:ing)?|use|using|operate|operating|operation)\b",
            q,
        )
        or re.search(r"(?:启动|开始使用|使用|操作|运行|开机|运动).{0,24}前", q)
    )
    asks_for_checklist = any(
        marker in q
        for marker in (
            "what should i do",
            "what do i need to do",
            "what is required",
            "what preparations",
            "需要做什么",
            "该做什么",
            "要做什么",
            "哪些准备",
            "什么准备",
        )
    )
    return has_pre_operation_scope and asks_for_checklist


_EXPLICIT_SOURCE_SUBHEADING_RE = re.compile(
    r"(?m)^[ \t]{0,3}#{1,6}[ \t]+\S[^\r\n]*$"
)


def _source_figure_ranges(text: str, image_count: int) -> list[tuple[int, int]]:
    """Split one extracted parent into figure ranges at explicit subheadings.

    First-pass extraction occasionally keeps the beginning of the next Markdown
    subsection in the same parent.  Treating every image in that parent as one
    provenance group would allow a completion rule to cross a real topic boundary.
    Only explicit ATX headings are used here, keeping the split deterministic and
    conservative; ordinary prose, captions and numbered steps remain untouched.
    """

    if image_count <= 0:
        return []
    parts = (text or "").split("<PIC>")
    starts = [0]
    for image_order in range(1, image_count):
        before_image = parts[image_order] if image_order < len(parts) else ""
        if _EXPLICIT_SOURCE_SUBHEADING_RE.search(before_image):
            starts.append(image_order)
    starts.append(image_count)
    return [
        (starts[index], starts[index + 1])
        for index in range(len(starts) - 1)
        if starts[index] < starts[index + 1]
    ]


def _text_before_explicit_subheading(text: str) -> str:
    """Return trailing text that still belongs to the preceding figure group."""

    match = _EXPLICIT_SOURCE_SUBHEADING_RE.search(text or "")
    return (text or "")[: match.start()] if match else (text or "")


def _load_source_figure_groups(engine) -> tuple[dict[tuple[str, str], list[dict]], dict[str, dict]]:
    """Load figure groups preserved by the first-pass manual extraction.

    The production semantic sections deliberately split long source passages into
    cleaner answer units.  That is useful for retrieval, but it can separate figures
    that appeared in one original procedure or interface block.  The first-pass
    ``manual_sections`` files retain that source-level grouping, so they are used as
    provenance only: they can widen the selector's candidate pool, but never bypass
    the normal relevance gate or introduce an image absent from the supplied manual.
    """

    section_path = Path(getattr(engine, "section_chunks_path", "") or "")
    source_dir = section_path.resolve().parent / "manual_sections"
    cache_key = str(source_dir).lower()
    cached = _SOURCE_GROUP_CACHE.get(cache_key)
    if cached is not None:
        return cached
    if not source_dir.is_dir():
        result = ({}, {})
        _SOURCE_GROUP_CACHE[cache_key] = result
        return result

    by_image: dict[tuple[str, str], list[dict]] = {}
    by_id: dict[str, dict] = {}
    for source_file in sorted(source_dir.glob("*.json")):
        try:
            payload = json.loads(source_file.read_text(encoding="utf-8"))
        except (OSError, ValueError, TypeError):
            continue
        product = str(payload.get("manual") or source_file.stem)
        for source_index, section in enumerate(payload.get("sections") or []):
            images = tuple(_ordered_unique(section.get("pics") or []))
            if not images:
                continue
            text = str(section.get("text") or "")
            parts = text.split("<PIC>")
            raw_captions = list(section.get("pic_captions") or [])
            ranges = _source_figure_ranges(text, len(images))
            for range_index, (start, end) in enumerate(ranges):
                group_images = images[start:end]
                if not group_images:
                    continue
                group_id = f"{product}|source:{source_index}"
                if len(ranges) > 1:
                    group_id += f":part:{range_index}"

                occurrences: dict[str, dict] = {}
                for local_order, image_order in enumerate(range(start, end)):
                    image_id = images[image_order]
                    after = parts[image_order + 1] if image_order + 1 < len(parts) else ""
                    if image_order + 1 == end and end < len(images):
                        after = _text_before_explicit_subheading(after)
                    occurrences[image_id] = {
                        "before": _normalise_text(parts[image_order] if image_order < len(parts) else ""),
                        "after": _normalise_text(after),
                        "caption": _normalise_text(
                            raw_captions[image_order] if image_order < len(raw_captions) else ""
                        ),
                        "image_order": local_order,
                    }

                context_parts = list(parts[start:end])
                trailing = parts[end] if end < len(parts) else ""
                if end < len(images):
                    trailing = _text_before_explicit_subheading(trailing)
                context_parts.append(trailing)
                group = {
                    "group_id": group_id,
                    "product": product,
                    "heading": _normalise_text(str(section.get("heading") or "")),
                    "context": _normalise_text(" <PIC> ".join(context_parts))[:1800],
                    "images": group_images,
                    "occurrences": occurrences,
                }
                by_id[group_id] = group
                for image_id in group_images:
                    by_image.setdefault((product, image_id), []).append(group)

    result = (by_image, by_id)
    _SOURCE_GROUP_CACHE[cache_key] = result
    return result


def _source_group_fields(
    product: str,
    image_id: str,
    groups_by_image: dict[tuple[str, str], list[dict]],
) -> dict[str, object]:
    """Return compact provenance fields for an image's primary source group."""

    groups = groups_by_image.get((product, image_id)) or []
    if not groups:
        return {}
    group = groups[0]
    return {
        "source_group_id": str(group["group_id"]),
        "source_group_heading": str(group["heading"]),
        "source_group_context": str(group["context"]),
        "source_group_images": tuple(group["images"]),
    }


def _figure_citation_context(text: str, figure_numbers: Iterable[str]) -> str:
    """Collect every source clause that explicitly cites one mapped figure."""

    output: list[str] = []
    for number in figure_numbers:
        pattern = re.compile(
            rf"(?:^|[.!?]\s+|\n)\s*([^.!?\n]{{0,420}}?\b(?:figure|fig\.?)\s*{re.escape(str(number))}\b[^.!?\n]*)",
            re.IGNORECASE,
        )
        for clause in pattern.findall(text or ""):
            clause = _normalise_text(clause)
            if clause and clause not in output:
                output.append(clause)
    return " ".join(output)


def _build_evidence_index(engine, captions: dict[str, dict]) -> tuple[dict[str, list[ImageEvidence]], list[ImageEvidence]]:
    """Build a reusable image-to-source map from parsed parent sections."""

    sections = list(getattr(engine, "section_chunks", []) or [])
    cache_key = (id(engine), len(sections))
    cached = _EVIDENCE_INDEX_CACHE.get(cache_key)
    if cached is not None:
        return cached

    groups_by_image, _groups_by_id = _load_source_figure_groups(engine)
    by_image: dict[str, list[ImageEvidence]] = {}
    all_items: list[ImageEvidence] = []
    for section_order, section in enumerate(sections):
        text = section.get("text") or ""
        pictures = tuple(section.get("pics") or [])
        linked_pictures = tuple(section.get("linked_pics") or [])
        fact_linked_pictures = tuple(section.get("fact_linked_pics") or [])
        concept_linked_pictures = tuple(section.get("concept_linked_pics") or [])
        if (
            not pictures
            and not linked_pictures
            and not fact_linked_pictures
            and not concept_linked_pictures
        ):
            continue
        parts = text.split("<PIC>")
        product = str(section.get("product") or "")
        section_id = int(section.get("section_id") or 0)
        heading = str(section.get("heading") or "")
        summary = str(section.get("llm_summary") or section.get("summary") or "")
        figure_map = {
            str(number): str(image_id)
            for number, image_id in (
                section.get("figure_links") or section.get("figure_map") or {}
            ).items()
        }

        for image_order, image_id in enumerate(pictures):
            before = parts[image_order] if image_order < len(parts) else ""
            after = parts[image_order + 1] if image_order + 1 < len(parts) else ""
            caption = captions.get(f"{product}|{image_id}") or {}
            visual = _normalise_text(
                str(caption.get("content") or caption.get("short_caption") or "")
            )
            item = ImageEvidence(
                image_id=str(image_id),
                product=product,
                section_id=section_id,
                heading=heading,
                section_summary=_normalise_text(summary),
                before_text=_normalise_text(before),
                after_text=_normalise_text(after),
                image_category=str(caption.get("category") or "unknown"),
                image_caption=visual,
                section_images=pictures,
                section_order=section_order,
                image_order=image_order,
                explicit_citation_text=_figure_citation_context(
                    text,
                    [number for number, mapped in figure_map.items() if mapped == str(image_id)],
                ),
                **_source_group_fields(product, str(image_id), groups_by_image),
            )
            by_image.setdefault(item.image_id, []).append(item)
            all_items.append(item)

        # A procedure can explicitly cite a figure printed in an adjacent source
        # section.  Preserve that citation as a second, auditable occurrence of the
        # figure instead of pretending that the image physically moved.  The local
        # procedure text is intentionally retained because it explains why the
        # external figure is evidence for this section.
        evidence_pictures = tuple(
            section.get("evidence_pics")
            or (*pictures, *linked_pictures, *fact_linked_pictures, *concept_linked_pictures)
        )
        for linked_image in (
            *linked_pictures,
            *fact_linked_pictures,
            *concept_linked_pictures,
        ):
            caption = captions.get(f"{product}|{linked_image}") or {}
            item = ImageEvidence(
                image_id=str(linked_image),
                product=product,
                section_id=section_id,
                heading=heading,
                section_summary=_normalise_text(summary),
                before_text=_normalise_text(text),
                after_text="",
                image_category=str(caption.get("category") or "unknown"),
                image_caption=_normalise_text(
                    str(caption.get("content") or caption.get("short_caption") or "")
                ),
                section_images=evidence_pictures,
                section_order=section_order,
                image_order=evidence_pictures.index(linked_image),
                origin=(
                    "fact_cross_reference"
                    if linked_image in fact_linked_pictures
                    else (
                        "concept_cross_reference"
                        if linked_image in concept_linked_pictures
                        else "cross_reference"
                    )
                ),
                explicit_citation_text=_figure_citation_context(
                    text,
                    [number for number, mapped in figure_map.items() if mapped == str(linked_image)],
                ),
                **_source_group_fields(product, str(linked_image), groups_by_image),
            )
            by_image.setdefault(item.image_id, []).append(item)
            all_items.append(item)

    result = (by_image, all_items)
    _EVIDENCE_INDEX_CACHE[cache_key] = result
    return result


def _choose_card(rows: list[ImageEvidence], route_products: set[str]) -> ImageEvidence:
    """Prefer the routed product if an image id appears in more than one manual."""

    if route_products:
        for row in rows:
            if row.product in route_products:
                return row
    return rows[0]


def _choose_question_occurrence(
    rows: list[ImageEvidence],
    route_products: set[str],
    question: str,
) -> ImageEvidence:
    """Choose the source occurrence whose topic best matches the question.

    Most images have one occurrence.  When a manual explicitly reuses a figure,
    heading overlap distinguishes the citing procedure from the physical source
    section without relying on question ids or asset ids.
    """

    routed = [row for row in rows if not route_products or row.product in route_products]
    candidates = routed or rows
    if len(candidates) == 1:
        return candidates[0]

    query_terms = _normalised_english_terms(question)

    def score(row: ImageEvidence) -> tuple[int, int]:
        heading_terms = _normalised_english_terms(row.heading)
        overlap = len(query_terms & heading_terms)
        linked_origin = row.origin in {
            "cross_reference", "fact_cross_reference", "concept_cross_reference"
        }
        return overlap, 1 if linked_origin else 0

    return max(candidates, key=score)


def _with_origin(item: ImageEvidence, origin: str) -> ImageEvidence:
    data = asdict(item)
    data["section_images"] = tuple(data["section_images"])
    data["source_group_images"] = tuple(data["source_group_images"])
    data["origin"] = (
        item.origin
        if item.origin in {
            "cross_reference", "fact_cross_reference", "concept_cross_reference"
        }
        else origin
    )
    return ImageEvidence(**data)


def _source_group_candidate(
    *,
    group: dict,
    image_id: str,
    seed: ImageEvidence,
    by_image: dict[str, list[ImageEvidence]],
    route_products: set[str],
) -> ImageEvidence:
    """Create the best evidence card for one source-group member.

    Prefer the semantic section card when it remains in the same topic hierarchy.
    If semantic cleanup moved the image elsewhere, or removed a duplicate source
    occurrence, reconstruct only its original local binding from the supplied manual
    extraction.  This preserves provenance without changing the retrieval index.
    """

    rows = by_image.get(image_id) or []
    actual = _choose_card(rows, route_products) if rows else None
    occurrence = (group.get("occurrences") or {}).get(image_id) or {}
    same_topic = bool(
        actual
        and actual.product == seed.product
        and _common_heading_prefix(actual.heading, seed.heading) >= 1
    )
    if same_topic and actual is not None:
        data = asdict(actual)
    else:
        base = actual or seed
        data = asdict(base)
        data.update(
            {
                "image_id": image_id,
                "product": seed.product,
                "section_id": seed.section_id,
                "heading": str(group.get("heading") or seed.heading),
                "section_summary": str(group.get("context") or seed.section_summary)[:900],
                "before_text": str(occurrence.get("before") or ""),
                "after_text": str(occurrence.get("after") or ""),
                "section_images": tuple(group.get("images") or (image_id,)),
                "section_order": seed.section_order,
                "image_order": int(occurrence.get("image_order") or 0),
            }
        )
        raw_caption = str(occurrence.get("caption") or "")
        if raw_caption:
            data["image_caption"] = raw_caption
        if actual is None or str(data.get("image_category") or "") == "noise":
            # Deduplication metadata must not erase a real occurrence in an answer-
            # bearing source block.  Keep the occurrence as a normal schematic and
            # let semantic selection decide whether it is useful for this question.
            data["image_category"] = "schematic" if raw_caption else "unknown"

    data.update(
        {
            "origin": "source_group",
            "source_group_id": str(group.get("group_id") or ""),
            "source_group_heading": str(group.get("heading") or ""),
            "source_group_context": str(group.get("context") or ""),
            "source_group_images": tuple(group.get("images") or ()),
        }
    )
    data["section_images"] = tuple(data.get("section_images") or ())
    return ImageEvidence(**data)


def build_candidate_evidence(
    *,
    question: str,
    candidate_images: Iterable[str],
    current_images: Iterable[str],
    engine,
    captions: dict[str, dict],
    route_products: Iterable[str] = (),
    max_cards: int = 48,
) -> list[ImageEvidence]:
    """Build retrieved cards and a narrow structural-neighbour expansion.

    The expansion is document-driven: only adjacent sections under the same
    two-level heading are considered.  This recovers continuation figures that
    chunk-level top-k retrieval may miss without opening unrelated manuals.
    """

    by_image, all_items = _build_evidence_index(engine, captions)
    groups_by_image, groups_by_id = _load_source_figure_groups(engine)
    route_set = {str(product) for product in route_products if product}
    ordered_ids: list[str] = []
    for image_id in [*current_images, *candidate_images]:
        image_id = str(image_id or "").strip()
        if image_id and image_id not in ordered_ids:
            ordered_ids.append(image_id)

    cards: list[ImageEvidence] = []
    seen_images: set[str] = set()
    for image_id in ordered_ids:
        rows = by_image.get(image_id) or []
        if not rows:
            continue
        item = _choose_question_occurrence(rows, route_set, question)
        if item.image_id not in seen_images:
            cards.append(_with_origin(item, "retrieved"))
            seen_images.add(item.image_id)

    if not cards or len(cards) >= max_cards:
        return cards[:max_cards]

    # Recover complete source-level figure groups before semantic-neighbour
    # expansion.  This joins figures that were split into separate clean sections
    # and restores source occurrences removed only as visual duplicates.
    for seed in list(cards):
        groups = groups_by_image.get((seed.product, seed.image_id)) or []
        for group in groups:
            canonical = groups_by_id.get(str(group.get("group_id") or "")) or group
            for image_id in canonical.get("images") or ():
                if image_id in seen_images or len(cards) >= max_cards:
                    continue
                cards.append(
                    _source_group_candidate(
                        group=canonical,
                        image_id=image_id,
                        seed=seed,
                        by_image=by_image,
                        route_products=route_set,
                    )
                )
                seen_images.add(image_id)

    seed_sections = {(item.product, item.section_id, item.heading, item.section_order) for item in cards}
    wide_scope = _is_scope_wide_question(question)
    neighbours: list[tuple[int, int, ImageEvidence]] = []
    for item in all_items:
        if item.image_id in seen_images:
            continue
        if route_set and item.product not in route_set:
            continue
        best_distance: int | None = None
        for product, _section_id, heading, section_order in seed_sections:
            if item.product != product:
                continue
            distance = abs(item.section_order - section_order)
            common_prefix = _common_heading_prefix(item.heading, heading)
            same_subtree = common_prefix >= 2
            referenced_adjacent_topic = wide_scope and common_prefix >= 1 and distance <= 1
            if not same_subtree and not referenced_adjacent_topic:
                continue
            # Always inspect immediate structural neighbours; inspect a complete
            # sibling family only when the question itself asks for broad scope.
            if distance <= 1 or (wide_scope and distance <= 8):
                best_distance = distance if best_distance is None else min(best_distance, distance)
        if best_distance is not None:
            neighbours.append((best_distance, item.section_order, item))

    neighbours.sort(key=lambda row: (row[0], row[1], row[2].image_order))
    for _distance, _section_order, item in neighbours:
        if len(cards) >= max_cards:
            break
        if item.image_id in seen_images:
            continue
        cards.append(_with_origin(item, "structural_neighbor"))
        seen_images.add(item.image_id)
    return cards


_PLANNER_SYSTEM = """\
You are the evidence-set selector of a multimodal manual RAG system. Your task is
not to write the customer answer. Independently identify the most relevant source
passage, decompose the question into atomic information requirements, and select
the smallest COMPLETE set of figures bound to that passage.

Return one JSON object only:
{
  "requirements": ["short requirement"],
  "scope_mode": "narrow_goal" or "whole_passage",
  "target_sections": [{"section_id": 0, "scope":"why this passage is the answer boundary"}],
  "selected_images": ["image_id"],
  "bindings": [{"image_id":"...","supports":"which requirement and what the image proves"}],
  "decision": "keep" or "revise",
  "confidence": 0.0,
  "reason": "brief audit explanation"
}

Selection policy:
1. Treat object, action, requested attribute, modifiers, scope, lifecycle stage,
   quantity and exceptions as separate requirements when they change the answer.
   Derive requirements from the user's wording only. Source passages are evidence
   for requirements, not a reason to invent extra requirements. In particular,
   intermediate states, implementation details and explanatory mechanisms found
   only in the manual remain optional unless the user asks for them.
2. First choose the source passage whose heading, lifecycle stage and local text
   best match the question. Reconstruct that passage's answer-bearing span rather
   than selecting isolated pictures from several superficially related passages.
3. Every selected figure must cover a requirement or a necessary identity/scope
   distinction in that answer-bearing span. Include complementary overview,
   variant, table, status and operation figures when no single figure carries all
   of those roles. Reject merely adjacent, repetitive or background figures.
   A quantitative table listing component types does not by itself establish a
   model/configuration distinction unless it explicitly names those models. When
   scope depends on which components a model contains, the component overview is
   complementary visual evidence rather than a duplicate of the numeric table.
   Apply a counterfactual necessity test to every figure: if removing that figure
   still leaves every user-derived requirement fully and visually supported, the
   figure is redundant and must not be selected. Conversely, do not omit an
   overview or control figure when it is the clearest evidence for the user's
   requested input action merely because another figure explains the outcome.
   When the input is a labelled control transition, prefer the figure that shows
   the complete relevant control range and target position. A diagram limited to
   an intermediate detent or transient state cannot substitute for that input
   evidence, even if it belongs to the main outcome passage.
4. Match lifecycle stages strictly: first installation, normal use, changing a
   setting, maintenance, replacement, troubleshooting and transport are distinct.
   A question about using a mode does not mean changing its default configuration
   unless the user asks to modify, configure or set it.
   Exception for a source-defined maintenance decision boundary: when the passage
   states that the requested maintenance action must stop or change under a stated
   condition (for example, a dirty, worn, damaged, or expired part must be
   replaced), that condition is part of the maintenance answer. If a status figure
   directly establishes that condition, retain the condition sentence and that
   figure as one bound unit. Do not import the neighbouring replacement or repair
   procedure unless the user asks for that procedure.
5. For a complete procedure or state transition, retain the figures needed to
   cover its requested steps/states. For a narrow single action, do not import an
   entire neighbouring procedure. For plural/enumeration questions, cover each
   requested member rather than only the first one.
   For a goal-directed operation, distinguish the actionable input from the
   terminal result. Select the smallest evidence combination that shows both when
   both are necessary to explain how the goal is achieved. Do not automatically
   retain every intermediate state in the source sequence unless the question
   requests the sequence, stages, low-speed behaviour, or operating mechanism.
   Treat a broad "what should I do before starting/using it" question as an
   immediate-readiness checklist, not as one narrow adjustment. When adjacent
   pre-use passages provide separate prerequisites, cover both equipment readiness
   (for example condition, placement or stability) and operator readiness (for
   example clothing, fit or operating position) when each directly applies. Do not
   pull in assembly, transport, maintenance or first-time setup merely because it
   occurs earlier in the manual.
   If the question names an assembly but the action is actually performed on a
   subcomponent, retain the overview that links the assembly to that subcomponent
   and any close-up that identifies the latch, hook, switch or retaining point
   manipulated by the procedure.
6. A part overview, table, status screen and operation diagram have different
   roles. Select complementary roles only when the question genuinely requires
   both. A figure that adds no new requirement is redundant.
   Source binding has priority over superficial visual similarity: two figures
   attached to different requested actions or states are complementary even when
   they show similar controls. A warning figure embedded inside the requested
   procedure is part of that procedure's evidence and must be retained even when
   another figure already shows the physical operation or the warning is repeated
   in text. Only a later aftercare figure outside the requested procedure may be
   treated as redundant when the operation is already visually covered.
7. Use source headings, source text around each image, visual content and manual
   order together. Structural neighbours are candidates, not automatic selections.
   The highest-ranked or dominant parent is not an answer boundary. Evidence may
   be combined across retrieved parents when different parents uniquely cover
   different user-derived requirements; prefer minimum complete coverage over
   maximum coverage from one parent.
   ``source_group_images_in_order`` records figures that occurred in one original
   manual passage before semantic cleanup split it into smaller sections. For a
   broad request that targets that whole passage, treat the group as a closed
   provenance boundary and cover every directly relevant member. For a narrow
   request, still select only the member that supports the requested sub-action.
   Do not assume that the existing generated answer or its current image choice is
   correct; make the decision from the question and source evidence only.
8. Distinguish an in-procedure warning from unrelated safety background or later
   aftercare. If a warning and its figure occur within the answer-bearing span of
   the complete procedure requested by the user, keep that figure as a required
   procedure step; never remove it merely because nearby prose states the same
   warning or another figure shows the associated control. Safety background from
   a different lifecycle stage and aftercare outside the requested procedure are
   not operation figures merely because they share a section. For a narrow action,
   retain only warnings inside that action's own source span or warnings necessary
   to perform it safely.
9. Output selected_images in source-document order unless the question explicitly
   requests another order. Never invent an image id and never rely on numeric
   adjacency alone.
10. When ``ordered_range_request`` is present, it is a hard answer boundary.
    Locate one source passage containing the formal numbered list, count numbered
    entries rather than figures, slice exactly the requested entry range, and
    select every figure owned by those entries. An unnumbered prerequisite before
    step 1 is outside ``first N steps``. A note, table, or aftercare paragraph
    after step N is outside the range unless it occurs inside step N. One selected
    entry may own zero, one, or several figures. Do not infer ownership from image
    filename adjacency; use source text, anchors, and entry spans.
"""


_COMPOSER_SYSTEM = """\
You are the grounded answer composer of a multimodal manual RAG system. Rewrite
the answer using only the selected evidence units. The required output language
provided in the request is a hard constraint even if image captions use another
language.

Hard rules:
1. Use every selected image exactly once and no unselected image. Keep its literal
   anchor as [[PIC:image_id]].
2. Text and image are one bound evidence unit. Place each anchor immediately after
   its corresponding source sentence or paragraph. Do not invent a figure
   description, direction, state, or caption. Only describe a figure when that
   description is explicitly present in the selected manual text. Never append a
   detached image list.
3. Preserve source-document order unless the question explicitly asks otherwise.
   Retrieval rank is never permission to reorder source sentences. Keep every
   Warning, Caution, Note, prerequisite, and condition in its original role and
   relative position; never turn one into a numbered step.
4. Edit minimally. You may remove obvious OCR/parser noise, delete exact duplicate
   lines, normalize whitespace/list layout, and translate sentence by sentence
   when required. Do not paraphrase, summarize, merge facts, split and recombine
   sentences, change hierarchy, or move content for a smoother narrative.
5. Answer every atomic requirement, but do not add unrelated neighbouring content.
6. Do not mention retrieval, evidence, manuals, image numbering, or this rewrite.
7. Do not use Markdown headings, bold markers, tables, code fences or bullet
   symbols. Plain paragraphs and numbered steps are allowed.
8. Output only the final answer text, with no JSON and no code fence.
"""


def _response_text(response) -> str:
    parts: list[str] = []
    for block in getattr(response, "content", []) or []:
        text = getattr(block, "text", None)
        if text:
            parts.append(str(text))
    return "\n".join(parts).strip()


def _parse_json_object(text: str) -> dict | None:
    cleaned = _JSON_FENCE_RE.sub("", (text or "").strip())
    try:
        value = json.loads(cleaned)
        return value if isinstance(value, dict) else None
    except Exception:
        pass
    match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
    if not match:
        return None
    try:
        value = json.loads(match.group(0))
        return value if isinstance(value, dict) else None
    except Exception:
        return None


def _ordered_unique(values: Iterable[object]) -> list[str]:
    output: list[str] = []
    for value in values:
        image_id = str(value or "").strip()
        if image_id and image_id not in output:
            output.append(image_id)
    return output


def _selected_from_plan(plan: dict) -> list[str]:
    values = plan.get("selected_images") or []
    if not isinstance(values, list):
        return []
    normalized: list[object] = []
    for value in values:
        if isinstance(value, dict):
            normalized.append(value.get("image_id"))
        else:
            normalized.append(value)
    return _ordered_unique(normalized)


def _sort_by_document_order(selected: list[str], cards: list[ImageEvidence]) -> list[str]:
    order = {card.image_id: card.document_order for card in cards}
    card_map = {card.image_id: card for card in cards}
    selected_cards = [card_map[image_id] for image_id in selected if image_id in card_map]

    # The original manual passage is the authoritative ordering boundary. Image
    # identifiers are asset names, not sequence numbers: generated assets can be
    # reused later in a manual, and semantic cleanup can invert their section-local
    # order. When every selected figure belongs to one preserved source group, use
    # that group's explicit figure order before considering parsed section order.
    group_ids = {card.source_group_id for card in selected_cards if card.source_group_id}
    if len(selected_cards) == len(selected) and len(group_ids) == 1:
        source_order = selected_cards[0].source_group_images
        source_rank = {image_id: index for index, image_id in enumerate(source_order)}
        section_orders = {card.section_images for card in selected_cards}
        if (
            len(section_orders) == 1
            and next(iter(section_orders)) == source_order
            and all(image_id in source_rank for image_id in selected)
        ):
            return sorted(selected, key=source_rank.__getitem__)

    # Conflicting parser/source metadata is not sufficient evidence to disturb an
    # answer's existing text-image order. Preserve it for semantic composition.
    return list(selected)


def _has_paired_operation_heading(question: str, heading: str) -> bool:
    """Return whether a section explicitly groups complementary operations."""

    q = (question or "").lower()
    h = (heading or "").lower()
    action_families = (
        ("open", "stow"),
        ("opening", "stowing"),
        ("power off", "shutdown"),
        ("automatic", "manual"),
        ("install", "remove"),
        ("attach", "detach"),
        ("lock", "unlock"),
        ("打开", "收起"),
        ("开启", "关闭"),
        ("安装", "拆卸"),
    )
    for left, right in action_families:
        broad_use = any(token in q for token in ("how to use", "how do i use", "use the", "如何使用", "怎么使用"))
        if left in h and right in h and (left in q or right in q or broad_use):
            return True
    return False


def _complete_structural_groups(
    question: str,
    selected: list[str],
    cards: list[ImageEvidence],
    baseline_images: Iterable[str] = (),
    allow_source_group_completion: bool = True,
) -> tuple[list[str], list[str]]:
    """Apply deterministic completeness constraints after semantic planning.

    The planner decides relevance; this layer only closes two narrow structures
    that manuals encode explicitly: paired-operation sections and repeated figures
    for the exact hazard named by the user.  It never expands across products or
    unrelated headings.
    """

    selected_set = set(selected)
    baseline_set = {str(image_id) for image_id in baseline_images if image_id}
    card_map = {card.image_id: card for card in cards}
    selected_sections = {
        (card_map[image_id].product, card_map[image_id].section_id)
        for image_id in selected
        if image_id in card_map
    }
    added: list[str] = []

    # A paired-operation heading defines one coherent evidence group.  Figures in
    # that group may depict the object overview, forward action and reverse action;
    # numeric tables and decorative images are outside the procedural group.
    paired_sections = {
        key
        for key in selected_sections
        if any(
            (card.product, card.section_id) == key
            and _has_paired_operation_heading(question, card.heading)
            for card in cards
        )
    }
    for card in cards:
        key = (card.product, card.section_id)
        if key not in paired_sections or card.image_id in selected_set:
            continue
        if card.image_category in {"noise", "info_table"}:
            continue
        selected_set.add(card.image_id)
        added.append(card.image_id)

    # A focused hazard question is a set-valued request: each directly illustrated
    # hazard area is complementary.  Use the text before the anchor (the source
    # statement that owns the figure), and exclude transport/lifting illustrations
    # when transport was not requested.
    q = (question or "").lower()
    finger_focus = any(token in q for token in ("finger", "fingers", "pinch", "trap", "手指", "夹手"))
    asks_transport = any(token in q for token in ("carry", "lift", "move", "transport", "搬", "抬", "移动"))
    if finger_focus:
        # If semantic planning returned no image, derive the owning section from
        # direct hazard cards themselves.  This avoids an empty-plan deadlock while
        # retaining the same source-text and transport exclusions.
        finger_sections = set(selected_sections)
        if not finger_sections:
            for card in cards:
                owned_text = (card.image_caption or card.before_text[-180:]).lower()
                if any(token in owned_text for token in ("finger", "fingers", "pinch", "trap", "手指", "夹")):
                    if not asks_transport and any(
                        token in owned_text
                        for token in ("carry", "lift", "handhold", "two people", "搬运", "抬起", "双人")
                    ):
                        continue
                    finger_sections.add((card.product, card.section_id))
        for card in cards:
            if (card.product, card.section_id) not in finger_sections:
                continue
            if card.image_id in selected_set or card.image_category == "noise":
                continue
            owned_text = (card.image_caption or card.before_text[-180:]).lower()
            if not any(token in owned_text for token in ("finger", "fingers", "pinch", "trap", "手指", "夹")):
                continue
            if not asks_transport and any(
                token in owned_text
                for token in ("carry", "lift", "handhold", "two people", "搬运", "抬起", "双人")
            ):
                continue
            selected_set.add(card.image_id)
            added.append(card.image_id)

    # The original extraction provides a second, source-faithful grouping layer.
    # Close only compact groups when the question targets the whole passage; narrow
    # sub-actions remain under the semantic planner's control.
    if allow_source_group_completion:
        for group_cards in _cards_by_source_group(cards).values():
            if baseline_set and not any(card.image_id in baseline_set for card in group_cards):
                continue
            if not _should_complete_source_group(question, group_cards, selected_set):
                continue
            available = {card.image_id: card for card in group_cards}
            for image_id in group_cards[0].source_group_images:
                card = available.get(image_id)
                if not card or image_id in selected_set:
                    continue
                selected_set.add(image_id)
                added.append(image_id)

    # When one semantic section explicitly points to the immediately following
    # sibling topic, its first figure is the visual entry point for that referenced
    # concept.  Later detail figures remain excluded unless independently selected.
    current_cards = [card_map[image_id] for image_id in selected_set if image_id in card_map]
    for card in _referenced_sibling_entries(question, current_cards, cards):
        if card.image_id not in selected_set:
            selected_set.add(card.image_id)
            added.append(card.image_id)

    # An unspecified model question must cover each same-action branch represented
    # by the source.  The branch detector requires disjoint model identifiers, the
    # same lifecycle action and the same two-level heading, so nearby warnings and
    # unrelated handling sections cannot enter merely through adjacency.
    variant_sections = _unspecified_variant_sections(question, selected_set, cards)
    for card in cards:
        if (card.product, card.section_id) not in variant_sections:
            continue
        if card.image_id in selected_set or card.image_category == "noise":
            continue
        selected_set.add(card.image_id)
        added.append(card.image_id)

    completed = [*selected, *(image_id for image_id in added if image_id not in selected)]
    return _sort_by_document_order(_ordered_unique(completed), cards), added


def _english_topic_terms(question: str) -> set[str]:
    stop = {
        "about", "after", "before", "could", "does", "have", "how", "into",
        "other", "should", "there", "these", "this", "using", "want", "what",
        "when", "which", "while", "with", "would", "your",
    }
    return {
        token
        for token in re.findall(r"[a-z][a-z0-9-]+", (question or "").lower())
        if len(token) >= 5 and token not in stop
    }


def _normalised_english_terms(text: str) -> set[str]:
    stop = {
        "about", "after", "before", "could", "does", "have", "how", "into",
        "other", "should", "there", "these", "this", "using", "want", "what",
        "when", "which", "while", "with", "would", "your", "the", "and", "for",
    }
    output: set[str] = set()
    for token in re.findall(r"[a-z][a-z0-9-]+", (text or "").lower()):
        if len(token) < 3 or token in stop:
            continue
        if token.endswith("ies") and len(token) > 4:
            token = token[:-3] + "y"
        elif token.endswith("s") and not token.endswith("ss") and len(token) > 4:
            token = token[:-1]
        output.add(token)
    return output


_GENERIC_ENTITY_TERMS = {
    "adjust", "adjustment", "answer", "care", "clean", "configure", "device",
    "display", "equipment", "grease", "install", "installation", "lubricate",
    "lubrication", "maintain", "maintenance", "manual", "menu", "operation",
    "mount", "mounting", "page", "process", "remove", "replace", "screen",
    "setting", "start", "step", "system", "use", "using", "view",
}
_GENERIC_CHINESE_ENTITY_TERMS = {
    "如何", "怎样", "怎么", "使用", "操作", "进行", "需要", "哪些", "什么",
    "设备", "系统", "手册", "方法", "步骤", "正确", "当前", "调节", "设置",
    "安装", "拆卸", "清洁", "更换", "启动",
}


def _direct_card_text(card: ImageEvidence) -> str:
    """Return evidence owned by one figure, excluding broad source-group context."""

    parts = [card.heading, card.before_text[-520:], card.image_caption]
    # Text after a non-final figure often introduces the next figure in the same
    # source section.  Treating it as current-image evidence leaks the next target
    # backwards (for example an air-filter image inheriting carburetor prose).
    # The final figure has no following sibling, so its trailing explanation is
    # still safe and useful evidence.
    if not card.section_images or card.image_id == card.section_images[-1]:
        parts.append(card.after_text[:260])
    return _normalise_text(
        " ".join(parts)
    ).lower()


def _shared_query_entities(question: str, *evidence_texts: str) -> set[str]:
    """Find distinctive query entities repeated by every supplied evidence unit.

    English words are normalized with the existing tokenizer. Chinese manuals do
    not have a runtime segmenter, so bounded 3-6 character spans are used. Generic
    action phrases are excluded: a shared object such as ``化油器`` is meaningful,
    while a shared verb such as ``调节`` must not join unrelated procedures.
    """

    if not evidence_texts:
        return set()
    q = (question or "").lower()
    lowered = [(text or "").lower() for text in evidence_texts]
    output = {
        term
        for term in (_normalised_english_terms(q) - _GENERIC_ENTITY_TERMS)
        if all(term in _normalised_english_terms(text) for text in lowered)
    }
    for run in re.findall(r"[\u4e00-\u9fff]{3,}", q):
        upper = min(6, len(run))
        for size in range(upper, 2, -1):
            for start in range(0, len(run) - size + 1):
                term = run[start : start + size]
                if term in _GENERIC_CHINESE_ENTITY_TERMS:
                    continue
                if all(term in text for text in lowered):
                    output.add(term)
    return output


def _is_overview_locator(card: ImageEvidence) -> bool:
    """Identify a numbered product overview that locates an operation target."""

    if card.image_category != "part_view":
        return False
    heading = (card.heading or "").lower()
    visual = (card.image_caption or "").lower()
    overview_heading = any(
        marker in heading
        for marker in (
            "product description", "overview", "components", "parts", "other devices",
            "产品描述", "部件", "组成", "结构", "其他装置",
        )
    )
    numbered_pointer = bool(
        re.search(r"(?:编号|标号)\s*\d+[^。；]{0,50}(?:指向|标注|位置)", visual)
        or re.search(r"\b(?:item|no\.?|number)\s*\d+[^.;]{0,60}(?:location|points? to)", visual)
    )
    location_signal = any(
        marker in visual for marker in ("位置", "调节口", "接口", "location", "located", "points to", "overview")
    )
    return location_signal and (overview_heading or numbered_pointer)


def _is_operation_detail(question: str, card: ImageEvidence) -> bool:
    """Return whether a card belongs to the requested operation lifecycle."""

    if _is_overview_locator(card):
        return False
    text = _direct_card_text(card)
    action_markers = (
        "adjust", "operation", "procedure", "maintenance", "replacement", "installation",
        "调节", "操作", "步骤", "维护", "保养", "更换", "安装", "拆卸", "清洁",
    )
    return _is_procedure_question(question) and any(marker in text for marker in action_markers)


def _locator_operation_complements(
    question: str,
    selected_images: Iterable[str],
    cards: list[ImageEvidence],
) -> list[ImageEvidence]:
    """Propose at most one high-confidence locator for a selected operation image.

    Cross-section completion is allowed only when both figures repeat the same
    distinctive query entity and have complementary source roles: a numbered
    product overview locates the target while a procedure/maintenance card explains
    the action. Numeric adjacency, shared parent headings, and product identity by
    themselves are deliberately insufficient.
    """

    if not _is_procedure_question(question) or _is_information_table_request(question):
        return []
    selected_set = {str(image_id) for image_id in selected_images if image_id}
    # Deterministic cross-section completion is reserved for one isolated detail
    # figure. A multi-figure procedure already carries its own visual context; an
    # additional overview then risks importing a parallel setup or control topic.
    if len(selected_set) != 1:
        return []
    selected_cards = [card for card in cards if card.image_id in selected_set]
    detail_cards = [card for card in selected_cards if _is_operation_detail(question, card)]
    if not detail_cards:
        return []

    candidates: list[tuple[int, int, ImageEvidence]] = []
    for locator in cards:
        if locator.image_id in selected_set or not _is_overview_locator(locator):
            continue
        for detail in detail_cards:
            if locator.product != detail.product or locator.section_id == detail.section_id:
                continue
            entities = _shared_query_entities(
                question,
                _direct_card_text(locator),
                _direct_card_text(detail),
            )
            # Product identity is already enforced above and cannot also serve as
            # proof that two cross-section figures describe the same target.  For
            # example, every blower overview repeats "blower", while only the
            # carburetor locator repeats the requested object "carburetor".
            product_text = (locator.product or "").lower()
            entities = {entity for entity in entities if entity not in product_text}
            if not entities:
                continue
            # Do not add a second locator when the selected set already contains
            # one for the same entity. This keeps the complement one-to-one.
            if any(
                _is_overview_locator(existing)
                and existing.product == locator.product
                and _shared_query_entities(
                    question,
                    _direct_card_text(existing),
                    _direct_card_text(detail),
                )
                for existing in selected_cards
            ):
                continue
            entity_strength = max(len(entity) for entity in entities)
            origin_bonus = 2 if locator.origin == "retrieved" else 0
            distance = abs(locator.section_order - detail.section_order)
            candidates.append((entity_strength + origin_bonus, -distance, locator))

    if not candidates:
        return []
    candidates.sort(key=lambda row: (row[0], row[1], -row[2].section_order), reverse=True)
    return [candidates[0][2]]


def _heading_query_score(question: str, heading: str) -> int:
    """Score full heading-path fit and penalize lifecycle contradictions."""

    q = (question or "").lower()
    h = (heading or "").lower()
    score = len(_normalised_english_terms(q) & _normalised_english_terms(h))
    asks_change = any(token in q for token in ("change", "modify", "setting", "configure", "default"))
    broad_use = any(token in q for token in ("how to use", "how do i use", "use the"))
    if broad_use and not asks_change and any(token in h for token in ("changing", "setting", "configuration", "default")):
        score -= 3
    asks_service = any(token in q for token in ("clean", "replace", "maintain", "care"))
    if asks_service and any(token in h for token in ("initial setup", "first installation", "packaging")):
        score -= 3
    return score


def _cards_by_source_group(cards: Iterable[ImageEvidence]) -> dict[str, list[ImageEvidence]]:
    """Group available cards by their original manual-passage provenance."""

    output: dict[str, list[ImageEvidence]] = {}
    for card in cards:
        if card.source_group_id:
            output.setdefault(card.source_group_id, []).append(card)
    return output


def _source_group_scope_score(question: str, group_cards: list[ImageEvidence]) -> int:
    """Measure whether the question targets the whole source figure group."""

    if not group_cards:
        return 0
    source_text = " ".join(
        [
            group_cards[0].source_group_heading,
            group_cards[0].source_group_context,
            *(card.heading for card in group_cards),
        ]
    )
    score = len(_normalised_english_terms(question) & _normalised_english_terms(source_text))
    q = question or ""
    # Chinese manuals often express the topic as one compact heading rather than
    # separable words. Count only distinctive four-character spans to avoid broad
    # matches on generic terms such as "设置" or "操作".
    for run in re.findall(r"[\u4e00-\u9fff]{4,}", source_text):
        if any(run[index : index + 4] in q for index in range(max(1, len(run) - 3))):
            score += 1
            break
    return score


def _source_group_has_whole_scope(
    question: str,
    group_cards: list[ImageEvidence],
    selected_count: int,
    group_size: int,
) -> bool:
    """Evaluate whole-passage intent independently of current completion state."""

    q = (question or "").lower()
    counted_scope = bool(
        re.search(r"\b(?:only|first|second|third|one|two|three|four)\b|仅|只|第[一二三四五六七八九十\d]+", q)
    )
    excludes_counted_baseline = "besides" in q and "other" in q
    if counted_scope and not excludes_counted_baseline:
        return False
    scope_score = _source_group_scope_score(question, group_cards)
    heading_score = max((_heading_query_score(question, card.heading) for card in group_cards), default=0)
    broad_scope = _is_scope_wide_question(question)
    # Two already-selected members are strong evidence that the answer targets the
    # source passage as a set. For a two-figure group, one parent-topic match is
    # enough only when the wording is explicitly broad. A stronger lexical match
    # can also close a three/four-figure menu or state group.
    return (
        scope_score >= 2
        or (
            selected_count >= 2
            and broad_scope
            and scope_score >= 1
            and heading_score >= 1
        )
        or (group_size == 2 and broad_scope and heading_score >= 1)
    )


def _should_complete_source_group(
    question: str,
    group_cards: list[ImageEvidence],
    selected_images: set[str],
) -> bool:
    """Return True only for compact, source-defined groups with whole-topic scope."""

    if not group_cards:
        return False
    ordered_group = list(group_cards[0].source_group_images)
    available = {card.image_id for card in group_cards}
    group_images = [image_id for image_id in ordered_group if image_id in available]
    selected = [image_id for image_id in group_images if image_id in selected_images]
    if not selected or len(group_images) < 2 or len(group_images) > 4:
        return False
    if all(image_id in selected_images for image_id in group_images):
        return False
    selected_cards = [card for card in group_cards if card.image_id in selected_images]
    missing_cards = [card for card in group_cards if card.image_id not in selected_images]
    # A source parent is provenance, not an answer boundary. Semantic cleanup may
    # reveal that its trailing figures belong to a different leaf topic. Complete
    # only members whose full normalized heading matches a selected member.
    if any(
        not any(_heading_parts(card.heading) == _heading_parts(seed.heading) for seed in selected_cards)
        for card in missing_cards
    ):
        return False
    return _source_group_has_whole_scope(question, group_cards, len(selected), len(group_images))


def _referenced_sibling_entries(
    question: str,
    current_cards: list[ImageEvidence],
    cards: list[ImageEvidence],
) -> list[ImageEvidence]:
    """Find first figures of adjacent topics explicitly referenced by current text."""

    q = (question or "").lower()
    interface_scope = any(
        token in q
        for token in ("interface", "screen", "display", "menu", "界面", "屏幕", "显示", "菜单")
    )
    if not current_cards or not _is_scope_wide_question(question) or not interface_scope:
        return []
    current_text = " ".join(
        f"{card.heading} {card.section_summary} {card.before_text} {card.after_text}"
        for card in current_cards
    ).lower()
    output: list[ImageEvidence] = []
    for candidate in cards:
        if candidate.origin != "structural_neighbor" or candidate.image_order != 0:
            continue
        related = [
            current
            for current in current_cards
            if candidate.product == current.product
            and candidate.section_id != current.section_id
            and abs(candidate.section_order - current.section_order) == 1
            and _common_heading_prefix(candidate.heading, current.heading) >= 1
        ]
        if not related:
            continue
        leaf = _heading_parts(candidate.heading)[-1] if _heading_parts(candidate.heading) else ""
        leaf_terms = _normalised_english_terms(leaf)
        normalized_leaf = " ".join(re.findall(r"[a-z][a-z0-9-]+", leaf.lower()))
        english_match = len(leaf_terms) >= 2 and len(normalized_leaf) >= 12 and normalized_leaf in current_text
        chinese_phrases = re.findall(r"[\u4e00-\u9fff]{4,}", leaf)
        chinese_match = bool(chinese_phrases and chinese_phrases[0] in current_text)
        if english_match or chinese_match:
            output.append(candidate)
    return output


def _adjacent_topic_overview_cards(
    question: str,
    selected_images: Iterable[str],
    cards: list[ImageEvidence],
) -> list[ImageEvidence]:
    """Return a preceding concept overview paired with an operation subsection.

    Some manuals introduce a feature with one example figure and put its controls
    in the immediately following sibling subsection.  The overview is included
    only when both leaf headings repeat a distinctive query entity; generic words
    such as screen, setting or mounting cannot bridge unrelated siblings.
    """

    if not _is_scope_wide_question(question):
        return []
    selected_set = {str(image_id) for image_id in selected_images if image_id}
    selected_cards = [card for card in cards if card.image_id in selected_set]
    selected_keys = {(card.product, card.section_id) for card in selected_cards}
    if len(selected_keys) != 1:
        return []

    by_section: dict[tuple[str, int], list[ImageEvidence]] = {}
    for card in cards:
        by_section.setdefault((card.product, card.section_id), []).append(card)
    selected_key = next(iter(selected_keys))
    selected_rows = by_section[selected_key]
    seed = min(selected_rows, key=lambda card: card.document_order)
    seed_parts = _heading_parts(seed.heading)
    seed_leaf = seed_parts[-1] if seed_parts else ""
    if not any(
        marker in seed_leaf
        for marker in ("operating", "operation", "procedure", "control", "customiz", "enable")
    ):
        return []

    matches: list[list[ImageEvidence]] = []
    for key, rows in by_section.items():
        if key == selected_key or not rows or len(rows) > 2:
            continue
        candidate = min(rows, key=lambda card: card.document_order)
        if candidate.product != seed.product or candidate.section_order != seed.section_order - 1:
            continue
        if _common_heading_prefix(candidate.heading, seed.heading) < 2:
            continue
        parts = _heading_parts(candidate.heading)
        leaf = parts[-1] if parts else ""
        if not any(
            marker in leaf
            for marker in ("using", "overview", "introduction", "description", "function")
        ):
            continue
        entities = _shared_query_entities(
            question,
            _direct_card_text(candidate),
            _direct_card_text(seed),
        )
        product_text = (candidate.product or "").lower()
        entities = {entity for entity in entities if entity not in product_text}
        candidate_terms = _normalised_english_terms(leaf)
        seed_terms = _normalised_english_terms(seed_leaf)
        if not (entities & candidate_terms & seed_terms):
            continue
        matches.append(sorted(rows, key=lambda card: card.document_order))

    # Ambiguous sibling families are left to the semantic planner.
    return matches[0] if len(matches) == 1 else []


def _minimal_control_transition_evidence(
    question: str,
    selected_images: Iterable[str],
    cards: list[ImageEvidence],
) -> tuple[list[ImageEvidence], set[tuple[str, int]]] | None:
    """Find a control entry figure plus the terminal state of one operation.

    Some manuals place a labelled control overview before a state sequence.  A
    goal-directed question normally needs the control entry and the state that
    directly achieves the requested result; intermediate low-power or transition
    diagrams are optional unless the user asks about stages or mechanism details.
    The rule is deliberately structural: it requires a unique preceding overview,
    a multi-figure state section, shared query terminology and explicit transition
    versus completion language in the source evidence.
    """

    q = _normalise_text(question).lower()
    goal_directed = bool(
        re.search(r"\bhow\b.{0,80}\b(?:make|get|put|move|turn|set)\b", q)
        or re.search(r"(?:如何|怎样|怎么).{0,40}(?:使|让|调到|设为|进入|达到)", q)
    )
    asks_for_transition_detail = bool(
        re.search(
            r"\b(?:stage|stages|transition|intermediate|low[- ]speed|mechanism|"
            r"how does .* work|why)\b|阶段|过渡|中间状态|低速|原理|为什么",
            q,
        )
    )
    if not goal_directed or asks_for_transition_detail:
        return None

    selected_set = {str(image_id) for image_id in selected_images if image_id}
    card_map = {card.image_id: card for card in cards}
    selected_cards = [card_map[image_id] for image_id in selected_set if image_id in card_map]
    selected_keys = {(card.product, card.section_id) for card in selected_cards}
    if len(selected_keys) != 1 or len(selected_cards) < 3:
        return None

    by_section: dict[tuple[str, int], list[ImageEvidence]] = {}
    for card in cards:
        by_section.setdefault((card.product, card.section_id), []).append(card)
    main_key = next(iter(selected_keys))
    main_rows = sorted(by_section[main_key], key=lambda card: card.document_order)
    if len(main_rows) < 3:
        return None

    query_terms = _normalised_english_terms(q) - _GENERIC_ENTITY_TERMS
    product_terms = _normalised_english_terms(main_rows[0].product)
    query_terms -= product_terms
    transition_markers = (
        "initially", "at a slow speed", "low speed", "slightly", "less thrust",
        "intermediate", "transition", "partially", "beginning", "初始", "低速",
        "稍微", "部分", "过渡", "中间",
    )
    completion_markers = (
        "moved farther", "move farther", "fully", "all ", "complete", "final",
        "results in", "which moves", "which makes", "达到", "完全", "全部",
        "最终", "从而", "使其",
    )

    def direct(card: ImageEvidence) -> str:
        return _direct_card_text(card)

    intermediate = [
        card for card in main_rows
        if any(marker in direct(card) for marker in transition_markers)
    ]
    terminal = [
        card for card in main_rows
        if any(marker in direct(card) for marker in completion_markers)
        and (not query_terms or bool(query_terms & _normalised_english_terms(direct(card))))
    ]
    if not intermediate or len(terminal) != 1 or terminal[0] in intermediate:
        return None

    seed = main_rows[0]
    overview_matches: list[tuple[list[ImageEvidence], ImageEvidence]] = []
    control_markers = (
        "control", "lever", "switch", "button", "selector", "position", "throttle",
        "控制", "操纵杆", "手柄", "开关", "按钮", "选择器", "位置",
    )
    overview_heading_markers = (
        "overview", "introduction", "description", "control", "using", "function",
        "总览", "介绍", "控制", "使用", "功能",
    )
    for key, rows in by_section.items():
        rows = sorted(rows, key=lambda card: card.document_order)
        if key == main_key or len(rows) != 1:
            continue
        candidate = rows[0]
        gap = seed.section_order - candidate.section_order
        if candidate.product != seed.product or not 1 <= gap <= 3:
            continue
        if _common_heading_prefix(candidate.heading, seed.heading) < 2:
            continue
        candidate_text = direct(candidate)
        leaf = _heading_parts(candidate.heading)[-1] if _heading_parts(candidate.heading) else ""
        if not any(marker in candidate_text for marker in control_markers):
            continue
        if not any(marker in leaf for marker in overview_heading_markers):
            continue
        shared = query_terms & _normalised_english_terms(candidate_text) & _normalised_english_terms(direct(terminal[0]))
        if query_terms and not shared:
            continue
        overview_matches.append((rows, candidate))

    if len(overview_matches) != 1:
        return None
    overview_rows, overview = overview_matches[0]
    scope = {main_key, (overview.product, overview.section_id)}
    return [overview, terminal[0]], scope


def _action_concepts(text: str) -> set[str]:
    """Map wording variants to the small action families needed for scope checks."""

    value = (text or "").lower()
    concepts: set[str] = set()
    direct_transport = bool(
        re.search(r"\b(?:carry|carrying|lift|lifting|transport|transporting)\b", value)
    )
    move_transport = bool(re.search(r"\bmove\b", value))
    moving_transport = bool(
        re.search(
            r"\bmoving\s+(?:(?:the|this|my|your|a|an)\s+)?"
            r"(?:product|device|machine|unit|fax|printer|equipment)\b",
            value,
        )
    )
    english_transport = direct_transport or move_transport or moving_transport
    chinese_transport = bool(
        re.search(
            r"搬运|搬动|抬起|抬升|移动(?:这台|该|本)?(?:设备|产品|机器|打印机|传真机)",
            value,
        )
    )
    if english_transport or chinese_transport:
        concepts.add("transport")
    return concepts


def _model_identifiers(text: str) -> set[str]:
    """Extract model-like identifiers without relying on any product-specific list."""

    return set(re.findall(r"\b[a-z]{2,}[a-z0-9-]*\d[a-z0-9-]*\b", (text or "").lower()))


def _unspecified_variant_sections(
    question: str,
    selected_images: set[str],
    cards: list[ImageEvidence],
) -> set[tuple[str, int]]:
    """Find omitted same-action sections that cover other unnamed model variants."""

    query_actions = _action_concepts(question)
    if not query_actions or _model_identifiers(question):
        return set()
    card_map = {card.image_id: card for card in cards}
    selected_cards = [card_map[image_id] for image_id in selected_images if image_id in card_map]
    if not selected_cards:
        return set()

    by_section: dict[tuple[str, int], list[ImageEvidence]] = {}
    for card in cards:
        by_section.setdefault((card.product, card.section_id), []).append(card)

    def section_text(rows: list[ImageEvidence]) -> str:
        return " ".join(
            f"{card.heading} {card.before_text} {card.after_text} {card.image_caption}"
            for card in rows
        )

    selected_keys = {(card.product, card.section_id) for card in selected_cards}
    selected_groups = {card.source_group_id for card in selected_cards if card.source_group_id}
    output: set[tuple[str, int]] = set()
    for key, rows in by_section.items():
        if key in selected_keys or not rows:
            continue
        if selected_groups and not any(card.source_group_id in selected_groups for card in rows):
            continue
        text = section_text(rows)
        candidate_models = _model_identifiers(text)
        if not candidate_models or not (query_actions & _action_concepts(text)):
            continue
        for selected_key in selected_keys:
            selected_rows = by_section.get(selected_key) or []
            if not selected_rows or selected_key[0] != key[0]:
                continue
            if _common_heading_prefix(rows[0].heading, selected_rows[0].heading) < 2:
                continue
            selected_text = section_text(selected_rows)
            selected_models = _model_identifiers(selected_text)
            if (
                selected_models
                and candidate_models - selected_models
                and query_actions & _action_concepts(selected_text)
            ):
                output.add(key)
                break
    return output


def _transport_variant_scope_sections(
    question: str,
    cards: list[ImageEvidence],
) -> set[tuple[str, int]]:
    """Return the closed set of same-topic transport branches for unnamed models."""

    if "transport" not in _action_concepts(question) or _model_identifiers(question):
        return set()
    by_section: dict[tuple[str, int], list[ImageEvidence]] = {}
    for card in cards:
        by_section.setdefault((card.product, card.section_id), []).append(card)

    candidates: list[tuple[tuple[str, int], list[ImageEvidence], set[str], set[str]]] = []
    for key, rows in by_section.items():
        text = " ".join(
            f"{card.heading} {card.before_text} {card.after_text} {card.image_caption}"
            for card in rows
        )
        models = _model_identifiers(text)
        groups = {card.source_group_id for card in rows if card.source_group_id}
        if models and groups and "transport" in _action_concepts(text):
            candidates.append((key, rows, models, groups))

    output: set[tuple[str, int]] = set()
    for index, (left_key, left_rows, left_models, left_groups) in enumerate(candidates):
        for right_key, right_rows, right_models, right_groups in candidates[index + 1 :]:
            if left_key[0] != right_key[0] or not (left_groups & right_groups):
                continue
            if _common_heading_prefix(left_rows[0].heading, right_rows[0].heading) < 2:
                continue
            if not (left_models - right_models) or not (right_models - left_models):
                continue
            output.update((left_key, right_key))
    return output


def _is_procedure_question(question: str) -> bool:
    q = (question or "").lower()
    return any(
        token in q
        for token in (
            "how to", "how do", "如何", "怎样", "怎么", "步骤", "操作",
            "clean", "replace", "open", "close", "reset", "charge", "care",
            "maintain", "install", "remove", "清洁", "更换", "打开", "充电",
            "维护", "保养", "安装", "拆卸", "重置",
        )
    )


def _is_information_table_request(question: str) -> bool:
    q = (question or "").lower()
    return any(
        token in q
        for token in (
            "frequency", "interval", "how often", "table", "status", "screen",
            "show", "display", "频率", "周期", "多久", "表", "状态", "界面",
            "显示", "不同型号", "battery", "level", "charge", "电池", "电量", "充电",
        )
    )


def _apply_evidence_constraints(
    question: str,
    answer: str,
    selected: list[str],
    cards: list[ImageEvidence],
    baseline_images: Iterable[str] = (),
    allow_topic_expansion: bool = True,
) -> tuple[list[str], dict[str, list[str]]]:
    """Resolve evidence coverage with generic stage, topic and role constraints."""

    q = (question or "").lower()
    card_map = {card.image_id: card for card in cards}
    selected_set = set(selected)
    baseline_set = {str(image_id) for image_id in baseline_images if image_id}
    audit: dict[str, list[str]] = {"added": [], "removed": []}

    def add(image_id: str) -> None:
        if image_id not in selected_set:
            selected_set.add(image_id)
            audit["added"].append(image_id)

    def remove(image_id: str) -> None:
        if image_id in selected_set:
            selected_set.remove(image_id)
            audit["removed"].append(image_id)

    def section_key(card: ImageEvidence) -> tuple[str, int]:
        return card.product, card.section_id

    def card_text(card: ImageEvidence) -> str:
        return f"{card.heading} {card.before_text[-520:]} {card.image_caption}".lower()

    # Concept links are prevalidated complements.  Selecting either side keeps the
    # target section's physical figures and appends the linked evidence; a table or
    # installation diagram must not replace the procedure/status figures.
    for concept_card in (card for card in cards if card.origin == "concept_cross_reference"):
        family = [
            card
            for card in cards
            if card.product == concept_card.product
            and card.section_id == concept_card.section_id
            and card.origin in {"retrieved", "concept_cross_reference"}
        ]
        if any(card.image_id in selected_set for card in family):
            for card in family:
                add(card.image_id)

    # Preserve illustrations explicitly owned by procedure steps that the answer
    # actually uses.  This is narrower than completing a whole multi-image section:
    # sibling loading/unloading figures remain excluded when their steps are absent.
    for card in _answered_explicit_figure_cards(answer, selected_set, cards):
        add(card.image_id)

    # A procedure close-up can omit where the target is physically accessed. Add
    # one locator only when source roles and a distinctive query entity prove a
    # one-to-one complement across sections.
    for locator in _locator_operation_complements(question, selected_set, cards):
        add(locator.image_id)

    baseline_overviews = _adjacent_topic_overview_cards(question, baseline_set, cards)
    adjacent_overviews = baseline_overviews or _adjacent_topic_overview_cards(
        question, selected_set, cards
    )
    if baseline_overviews:
        baseline_keys = {
            section_key(card_map[image_id])
            for image_id in baseline_set
            if image_id in card_map
        }
        overview_keys = {section_key(card) for card in baseline_overviews}
        closed_keys = baseline_keys | overview_keys
        for image_id in list(selected_set):
            card = card_map.get(image_id)
            if card and section_key(card) not in closed_keys:
                remove(image_id)
        for image_id in baseline_set:
            card = card_map.get(image_id)
            if card and section_key(card) in closed_keys:
                add(image_id)
    for overview in adjacent_overviews:
        add(overview.image_id)

    # When the manual defines separate carrying instructions for model families and
    # the user names no model, those transport branches form the complete visual
    # scope.  Electrical setup, generic warning-label and moving-part sections may
    # be sensible background precautions, but they are not transport illustrations.
    transport_scope = _transport_variant_scope_sections(question, cards)
    if len(transport_scope) >= 2:
        for image_id in list(selected_set):
            card = card_map.get(image_id)
            if card and section_key(card) not in transport_scope:
                remove(image_id)

    # Lifecycle arbitration: a maintenance/replacement question must not inherit
    # first-unboxing illustrations when an explicit maintenance section is already
    # present.  If planning selected nothing, the exact maintenance section can
    # still contribute its own bound figure.
    asks_filter_service = (
        any(token in q for token in ("filter", "滤网"))
        and any(token in q for token in ("clean", "replace", "更换", "清洁"))
        and not any(token in q for token in ("first install", "initial", "new product", "首次", "新机", "包装"))
    )
    if asks_filter_service:
        maintenance_cards = [
            card
            for card in cards
            if any(token in card.heading.lower() for token in ("maintenance", "replacement", "clean", "维护", "更换", "清洁"))
            and any(token in card.heading.lower() for token in ("filter", "滤网"))
        ]
        selected_has_maintenance = any(card.image_id in selected_set for card in maintenance_cards)
        selected_has_setup = any(
            card.image_id in selected_set
            and any(token in card.heading.lower() for token in ("installation", "setup", "packaging", "安装", "准备", "包装"))
            for card in cards
        )
        # Never manufacture visual evidence after the semantic planner returned
        # an empty set.  This code only repairs a concrete lifecycle conflict in
        # an already-selected set; object identity remains a semantic decision.
        if maintenance_cards and selected_has_setup:
            best_key = section_key(maintenance_cards[0])
            for card in maintenance_cards:
                if section_key(card) == best_key:
                    add(card.image_id)
            for card in cards:
                if card.image_id in selected_set and any(
                    token in card.heading.lower()
                    for token in ("installation", "setup", "packaging", "安装", "准备", "包装")
                ):
                    remove(card.image_id)

    # A distinctive topic term identifies a coherent state or operation group even
    # when the parser kept it inside a larger parent section (for example a livewell
    # following rod-holder content).  Only already-selected sections are expanded.
    topic_terms = _english_topic_terms(question)
    selected_sections = {
        section_key(card_map[image_id]) for image_id in selected_set if image_id in card_map
    }
    if allow_topic_expansion:
        for key in list(selected_sections):
            section_cards = [card for card in cards if section_key(card) == key]
            matched_terms = {
                term
                for term in topic_terms
                if any(term in card_text(card) for card in section_cards if card.image_id in selected_set)
            }
            for card in section_cards:
                owned = card_text(card)
                if matched_terms and any(term in owned for term in matched_terms):
                    add(card.image_id)

    # A generic reset request covers all reset variants in the selected Reset
    # section; a screen/state-flow request covers the complete compact state group.
    for key in list(selected_sections):
        section_cards = [card for card in cards if section_key(card) == key]
        if not section_cards:
            continue
        leaf = _heading_parts(section_cards[0].heading)[-1] if _heading_parts(section_cards[0].heading) else ""
        broad_reset = "reset" in q and "reset" in leaf and not any(
            token in q for token in ("factory reset", "hardware reset", "恢复出厂", "硬件重置")
        )
        state_flow = any(token in q for token in ("screen", "界面", "显示")) and any(
            token in leaf for token in ("screen", "display", "界面")
        )
        if broad_reset or state_flow:
            for card in section_cards:
                if card.image_category != "noise":
                    add(card.image_id)

    # Plural category requests can span adjacent sibling sections under the same
    # parent heading.  Explicit cardinality (two/three/etc.) remains bounded by the
    # planner and is not expanded to every sibling.
    explicit_count = bool(re.search(r"\b(?:two|three|four|five|six)\b|\d+|两种|两个|三种|三个", q))
    plural_words = {word[:-1] for word in re.findall(r"\b[a-z]{5,}s\b", q)}
    if plural_words and not explicit_count:
        selected_cards = [card_map[image_id] for image_id in selected_set if image_id in card_map]
        for seed in selected_cards:
            seed_leaf = _heading_parts(seed.heading)[-1] if _heading_parts(seed.heading) else ""
            matching_stems = {stem for stem in plural_words if stem in seed_leaf}
            if not matching_stems:
                continue
            seed_scope_score = _heading_query_score(question, seed.heading)
            for card in cards:
                if card.product != seed.product or _common_heading_prefix(card.heading, seed.heading) < 2:
                    continue
                if abs(card.section_order - seed.section_order) > 1:
                    continue
                # Preserve explicit query qualifiers such as T-rail, captions or
                # travel case. A sibling matching only the generic plural noun is
                # not another requested category.
                if _heading_query_score(question, card.heading) < seed_scope_score:
                    continue
                leaf = _heading_parts(card.heading)[-1] if _heading_parts(card.heading) else ""
                if any(stem in leaf for stem in matching_stems):
                    add(card.image_id)

    # Maintenance action groups often continue in an adjacent sibling section.
    # Extend only the same component class and action, excluding a new pivot/bearing
    # topic that merely follows it in the source.
    if any(token in q for token in ("care", "maintain", "maintenance", "grease", "lubricat", "维护", "保养", "润滑")):
        selected_cards = [card_map[image_id] for image_id in selected_set if image_id in card_map]
        if any("cable" in card_text(card) or "拉索" in card_text(card) for card in selected_cards):
            for seed in selected_cards:
                for card in cards:
                    if card.product != seed.product or _common_heading_prefix(card.heading, seed.heading) < 2:
                        continue
                    owned = card_text(card)
                    if not ("cable" in owned or "拉索" in owned):
                        continue
                    if not any(token in owned for token in ("grease", "lubricat", "润滑")):
                        continue
                    if any(token in owned for token in ("pivot point", "bearing", "枢轴", "轴承")):
                        continue
                    add(card.image_id)

    # If the literal verb has no matching evidence but an exact object heading is
    # present, prefer the manual's canonical operation over an unsupported or
    # internally inconsistent selection.  The override requires two distinctive
    # object terms and a strictly better heading match than the current selection.
    q_terms = _english_topic_terms(question)
    section_scores: dict[tuple[str, int], tuple[int, int]] = {}
    for index, card in enumerate(cards):
        overlap = _heading_query_score(question, card.heading)
        if overlap >= 2:
            key = section_key(card)
            old = section_scores.get(key)
            score = (overlap, -index)
            if old is None or score > old:
                section_scores[key] = score
    source_group_closed = False
    for group_cards in _cards_by_source_group(cards).values():
        ordered_group = list(group_cards[0].source_group_images) if group_cards else []
        available = {card.image_id for card in group_cards}
        group_images = [image_id for image_id in ordered_group if image_id in available]
        if (
            1 < len(group_images) <= 4
            and set(group_images).issubset(selected_set)
            and _source_group_has_whole_scope(
                question,
                group_cards,
                len(group_images),
                len(group_images),
            )
        ):
            source_group_closed = True
            break
    if section_scores and not source_group_closed and not adjacent_overviews:
        best_key = max(section_scores, key=section_scores.get)
        best_overlap = section_scores[best_key][0]
        current_scores: list[int] = []
        for image_id in selected_set:
            card = card_map.get(image_id)
            if not card:
                continue
            current_scores.append(_heading_query_score(question, card.heading))
        needs_override = not selected_set or any(score < best_overlap for score in current_scores)
        if needs_override:
            for image_id in list(selected_set):
                remove(image_id)
            best_cards: list[ImageEvidence] = []
            for card in cards:
                if section_key(card) == best_key and card.image_category != "noise":
                    add(card.image_id)
                    best_cards.append(card)
            if plural_words and not explicit_count and best_cards:
                seed = best_cards[0]
                seed_leaf = _heading_parts(seed.heading)[-1] if _heading_parts(seed.heading) else ""
                matching_stems = {stem for stem in plural_words if stem in seed_leaf}
                seed_scope_score = _heading_query_score(question, seed.heading)
                for card in cards:
                    if card.product != seed.product or abs(card.section_order - seed.section_order) > 1:
                        continue
                    if _common_heading_prefix(card.heading, seed.heading) < 2:
                        continue
                    if _heading_query_score(question, card.heading) < seed_scope_score:
                        continue
                    leaf = _heading_parts(card.heading)[-1] if _heading_parts(card.heading) else ""
                    if any(stem in leaf for stem in matching_stems):
                        add(card.image_id)

    # Procedure answers should not retain a frequency table unless the question
    # requests frequency/status information.  This separates context from action.
    if _is_procedure_question(question) and not _is_information_table_request(question):
        for image_id in list(selected_set):
            card = card_map.get(image_id)
            if card and card.image_category == "info_table":
                if card.origin == "concept_cross_reference":
                    continue
                owned = card_text(card)
                if any(
                    token in owned
                    for token in (
                        "warning", "important", "caution", "danger", "do not", "never",
                        "警告", "注意", "切勿", "不要", "严禁",
                    )
                ):
                    continue
                peers = [
                    other for other in selected_set
                    if other != image_id
                    and other in card_map
                    and section_key(card_map[other]) == section_key(card)
                ]
                if peers:
                    remove(image_id)

    # A model/subtype-specific operation does not need a broad multi-component
    # location overview when a direct operation figure for that same subtype exists.
    subtype_tokens = set(re.findall(r"(?:[a-z]+\d+[a-z0-9-]*|\d+[a-z]+[a-z0-9-]*)", q))
    if subtype_tokens and _is_procedure_question(question) and not any(
        token in q for token in ("where", "location", "position", "位置", "哪里")
    ):
        direct_operation = [
            card_map[image_id]
            for image_id in selected_set
            if image_id in card_map
            and card_map[image_id].image_category == "schematic"
            and any(token in card_text(card_map[image_id]) for token in subtype_tokens)
        ]
        if direct_operation:
            for image_id in list(selected_set):
                card = card_map.get(image_id)
                if not card or card.image_category != "part_view":
                    continue
                visual = card.image_caption.lower()
                if any(token in visual for token in ("位置", "location", "overview", "部件")):
                    remove(image_id)

    # Scope qualifiers such as different models/configurations require evidence of
    # which component variants exist.  A numeric table is sufficient for values but
    # not for configuration identity when it contains no model labels; add the
    # nearest same-topic component overview, without importing operation steps.
    model_scope = any(
        token in q
        for token in ("different model", "different version", "configuration", "variant", "不同型号", "不同机型", "不同配置")
    )
    if model_scope:
        selected_tables = [
            card_map[image_id]
            for image_id in selected_set
            if image_id in card_map and card_map[image_id].image_category == "info_table"
        ]
        for table in selected_tables:
            table_visual = table.image_caption.lower()
            if any(token in table_visual for token in ("model", "型号", "机型")):
                continue
            overviews = [
                card
                for card in cards
                if card.product == table.product
                and card.image_id not in selected_set
                and card.image_category == "part_view"
                and _common_heading_prefix(card.heading, table.heading) >= 2
                and any(token in card.image_caption.lower() for token in ("位置", "location", "overview", "部件"))
            ]
            if overviews:
                overviews.sort(key=lambda card: abs(card.section_order - table.section_order))
                add(overviews[0].image_id)

    # A pure accessory enumeration is closed by a directly relevant specification
    # table. Installation, attachment and location questions still retain their
    # operation figures; only list-style questions remove non-tabular illustrations
    # that add no additional accessory type or limit.
    asks_accessory_list = any(token in q for token in ("accessories", "accessory", "附件")) and any(
        token in q for token in ("what", "which", "哪些", "什么", "配备")
    )
    asks_accessory_operation = any(
        token in q
        for token in ("install", "attach", "remove", "where", "location", "use", "安装", "连接", "拆卸", "位置", "哪里", "使用")
    )
    if asks_accessory_list and not asks_accessory_operation:
        selected_tables = [
            card_map[image_id]
            for image_id in selected_set
            if image_id in card_map
            and card_map[image_id].image_category == "info_table"
            and any(
                token in card_text(card_map[image_id])
                for token in ("accessories", "accessory", "附件", "specification", "规格")
            )
        ]
        if selected_tables:
            for image_id in list(selected_set):
                card = card_map.get(image_id)
                if card and card.image_category != "info_table":
                    remove(image_id)

    # Complete a contiguous generated-image family when three consecutive members
    # are already selected and the next candidate stays on the same product/topic.
    grouped: dict[str, list[tuple[int, str]]] = {}
    for image_id in selected_set:
        match = re.match(r"^(.*_)(\d+)$", image_id)
        if match:
            grouped.setdefault(match.group(1), []).append((int(match.group(2)), image_id))
    for prefix, rows in grouped.items():
        # ManualNN_x is a whole-manual sequence, not a semantic figure family;
        # numeric adjacency can cross unrelated section boundaries.
        if prefix.lower().startswith("manual"):
            continue
        numbers = sorted(number for number, _image_id in rows)
        if len(numbers) < 3 or numbers != list(range(numbers[0], numbers[-1] + 1)):
            continue
        next_id = ""
        for candidate_id in card_map:
            candidate_match = re.match(r"^(.*_)(\d+)$", candidate_id)
            if (
                candidate_match
                and candidate_match.group(1) == prefix
                and int(candidate_match.group(2)) == numbers[-1] + 1
            ):
                next_id = candidate_id
                break
        next_card = card_map.get(next_id) if next_id else None
        extended_numbers = list(numbers)
        if next_card and next_card.origin in {"retrieved", "structural_neighbor"}:
            shared_terms = topic_terms & set(re.findall(r"[a-z][a-z0-9-]+", card_text(next_card)))
            if shared_terms or any(token in q for token in ("case", "盒", "battery", "电池")):
                add(next_id)
                extended_numbers.append(numbers[-1] + 1)

        # If a dominant run has an alias-like singleton with the same numeric index
        # in the same section, keep the coherent family and remove the singleton.
        if len(extended_numbers) >= 4:
            dominant_cards = [card_map[image_id] for _number, image_id in rows if image_id in card_map]
            dominant_sections = {section_key(card) for card in dominant_cards}
            for image_id in list(selected_set):
                if image_id.startswith(prefix):
                    continue
                match = re.match(r"^(.*_)(\d+)$", image_id)
                card = card_map.get(image_id)
                if not match or not card or int(match.group(2)) != extended_numbers[0]:
                    continue
                if section_key(card) in dominant_sections:
                    remove(image_id)

    # Explicitly counted control classes stay within that class.  This prevents an
    # adjacent lever from entering a two-switch answer (and vice versa) merely due
    # to retrieval or document-number proximity.
    if explicit_count:
        requested_class = ""
        if "switch" in q or "开关" in q:
            requested_class = "switch"
        elif "lever" in q or "手柄" in q or "操纵杆" in q:
            requested_class = "lever"
        if requested_class:
            for image_id in list(selected_set):
                card = card_map.get(image_id)
                if not card:
                    continue
                owned = card_text(card)
                if requested_class == "switch" and not ("switch" in owned or "开关" in owned):
                    remove(image_id)
                elif requested_class == "lever" and not ("lever" in owned or "手柄" in owned or "操纵杆" in owned):
                    remove(image_id)

    # A warning/aftercare illustration is not another operation when a direct
    # figure from the same section already covers the requested control.
    asks_safety = any(token in q for token in ("safe", "safety", "warning", "attention", "注意", "安全", "危险"))
    if not asks_safety:
        for image_id in list(selected_set):
            card = card_map.get(image_id)
            if not card:
                continue
            owned = f"{card.before_text[-360:]} {card.image_caption}".lower()
            aftercare = any(
                token in owned
                for token in ("unauthorized", "children", "accidental starting", "after use", "not running", "防止误", "儿童")
            )
            if not aftercare:
                continue
            # The semantic planner may select a warning that is embedded in the
            # requested procedure.  Text duplication does not make that visual
            # evidence redundant, so this cleanup must not override the planner.
            if _is_procedure_question(question) and any(
                token in owned
                for token in ("warning", "important", "caution", "danger", "do not", "never", "警告", "注意", "切勿", "不要", "严禁")
            ):
                continue
            if _has_paired_operation_heading(question, card.heading):
                continue
            same_section_peers = [
                peer for peer in selected_set
                if peer != image_id and peer in card_map and section_key(card_map[peer]) == section_key(card)
            ]
            if same_section_peers:
                remove(image_id)

    # Final reverse validation for a focused finger hazard: every retained figure
    # must visually depict that hazard, not merely share a broad safety heading.
    finger_focus = any(token in q for token in ("finger", "fingers", "pinch", "trap", "手指", "夹手"))
    if finger_focus:
        asks_transport = any(token in q for token in ("carry", "lift", "move", "transport", "搬", "抬", "移动"))
        for image_id in list(selected_set):
            card = card_map.get(image_id)
            if not card:
                continue
            visual = (card.image_caption or card.before_text[-180:]).lower()
            direct_hazard = any(token in visual for token in ("finger", "fingers", "pinch", "trap", "手指", "夹"))
            transport_only = any(token in visual for token in ("carry", "lift", "handhold", "two people", "搬运", "抬起", "双人"))
            if not direct_hazard or (transport_only and not asks_transport):
                remove(image_id)

    constrained = [image_id for image_id in selected if image_id in selected_set]
    constrained.extend(image_id for image_id in audit["added"] if image_id in selected_set)
    return _sort_by_document_order(_ordered_unique(constrained), cards), audit


def _anchors_are_bound(answer: str, selected: list[str]) -> bool:
    """Validate exact anchors and reject detached/consecutive image placeholders."""

    anchors = _ANCHOR_RE.findall(answer or "")
    if anchors != selected:
        return False
    cursor = 0
    for match in _ANCHOR_RE.finditer(answer or ""):
        preceding = re.sub(r"\s+", " ", answer[cursor:match.start()]).strip()
        if len(preceding) < 8:
            return False
        cursor = match.end()
    return True


def _required_language(question: str) -> str:
    cjk_count = sum("\u4e00" <= char <= "\u9fff" for char in question or "")
    return "Chinese" if cjk_count >= 2 else "English"


def _language_matches(required: str, answer: str) -> bool:
    visible = re.sub(r"\[\[PIC:[^\]]+\]\]", "", answer or "")
    cjk_count = sum("\u4e00" <= char <= "\u9fff" for char in visible)
    if required == "Chinese":
        return cjk_count >= 4
    # Product names can contain a few CJK characters, but a predominantly Chinese
    # answer to an English question must never pass final binding validation.
    return cjk_count <= max(3, len(visible) // 80)


def _single_goal_cross_parent_coverage_risk(
    question: str,
    current_cards: list[ImageEvidence],
    cards: list[ImageEvidence],
) -> bool:
    """Flag over-concentrated evidence for semantic review, without selecting it.

    A single-goal operation can require complementary roles that the parser placed
    in sibling parents, such as a labelled input control and the resulting device
    state.  This predicate only opens the semantic planner when the current answer
    is concentrated in one multi-figure parent and a retrieved sibling repeats a
    distinctive query entity.  It never adds or removes evidence itself.
    """

    q = _normalise_text(question).lower()
    if not _is_procedure_question(question) or len(current_cards) < 3:
        return False
    explicit_multi_scope = bool(
        re.search(
            r"\b(?:all|every|each|different|types?|kinds?|stages?|steps?|"
            r"sequence|first|second|third|and then)\b|"
            r"鍏ㄩ儴|鎵€鏈夌各|涓嶅悓|绉嶇被|闃舵|姝ラ|棣栧厛|鐒跺悗",
            q,
        )
    )
    if explicit_multi_scope:
        return False

    current_keys = {(card.product, card.section_id) for card in current_cards}
    if len(current_keys) != 1:
        return False
    product, section_id = next(iter(current_keys))
    current_text = " ".join(_direct_card_text(card) for card in current_cards)
    product_terms = _normalised_english_terms(product)

    for candidate in cards:
        if candidate.image_id in {card.image_id for card in current_cards}:
            continue
        if candidate.origin != "retrieved" or candidate.product != product:
            continue
        if candidate.section_id == section_id:
            continue
        if min(abs(candidate.section_order - card.section_order) for card in current_cards) > 4:
            continue
        if max(_common_heading_prefix(candidate.heading, card.heading) for card in current_cards) < 2:
            continue
        shared = _shared_query_entities(
            question,
            _direct_card_text(candidate),
            current_text,
        )
        shared = {
            term for term in shared
            if term not in product_terms and term not in _GENERIC_ENTITY_TERMS
        }
        if shared:
            return True
    return False


def _minimal_goal_role_pair(
    question: str,
    current_cards: list[ImageEvidence],
    cards: list[ImageEvidence],
) -> tuple[ImageEvidence, ImageEvidence] | None:
    """Return one input-control/terminal-result pair when the evidence proves it.

    This is a role-level minimum-cover check.  It is independent of products and
    image identifiers: the input must be a retrieved, fully labelled control
    overview, while the terminal card must contain source language that directly
    states the requested result.  Ambiguous pairs are deliberately rejected.
    """

    # Disabled until corpus-wide regression proves that role inference is stable.
    # Returning no pair keeps the production selector on its established path.
    return None

    if not _single_goal_cross_parent_coverage_risk(question, current_cards, cards):
        return None
    product = current_cards[0].product
    current_ids = {card.image_id for card in current_cards}
    product_terms = _normalised_english_terms(product)
    control_markers = (
        "control", "lever", "selector", "switch", "button", "dial", "throttle",
    )
    range_markers = (
        "position", "range", "neutral", "forward", "reverse", "open", "closed",
    )
    result_markers = (
        "which moves", "which makes", "causes", "results in", "so that",
        "therefore", "is then", "will then",
    )

    input_candidates: list[ImageEvidence] = []
    for card in cards:
        if card.image_id in current_ids or card.origin != "retrieved" or card.product != product:
            continue
        if card.section_id == current_cards[0].section_id:
            continue
        heading = (card.heading or "").lower()
        text = _direct_card_text(card)
        label_numbers = set(re.findall(r"(?<!\d)[1-9](?!\d)", card.image_caption or ""))
        is_overview = any(marker in heading for marker in ("overview", "control", "鎬昏", "鎺у埗"))
        if (
            is_overview
            and any(marker in text for marker in control_markers)
            and any(marker in text for marker in range_markers)
            and len(label_numbers) >= 3
        ):
            input_candidates.append(card)

    terminal_candidates = [
        card
        for card in current_cards
        if any(marker in _direct_card_text(card) for marker in result_markers)
    ]
    pairs: list[tuple[ImageEvidence, ImageEvidence]] = []
    for input_card in input_candidates:
        for terminal_card in terminal_candidates:
            shared = _shared_query_entities(
                question,
                _direct_card_text(input_card),
                _direct_card_text(terminal_card),
            )
            shared = {
                term for term in shared
                if term not in product_terms and term not in _GENERIC_ENTITY_TERMS
            }
            if shared:
                pairs.append((input_card, terminal_card))
    return pairs[0] if len(pairs) == 1 else None


def _answered_explicit_figure_cards(
    answer: str,
    current_images: Iterable[str],
    cards: list[ImageEvidence],
) -> list[ImageEvidence]:
    """Find omitted figures whose explicitly citing step appears in the answer.

    The decision is grounded in source prose: the sentence immediately owning a
    ``Figure N`` citation must share at least two distinctive terms with the
    generated answer.  This retains a used procedure step's bound illustration
    while leaving figures for unmentioned sibling procedures excluded.
    """

    citation_stop = {
        "figure", "machine", "device", "equipment", "item", "thing",
        "up", "down", "front", "rear", "near", "shown", "show",
    }
    answer_terms = (
        _normalised_english_terms(answer) - _GENERIC_ENTITY_TERMS - citation_stop
    )
    if not answer_terms:
        return []
    current_set = set(current_images)
    current_sections = {
        (card.product, card.section_id)
        for card in cards
        if card.image_id in current_set
    }
    matched: list[ImageEvidence] = []
    citation_re = re.compile(
        r"(?:^|[.!?]\s+|\n)\s*([^.!?\n]{0,420}?\b(?:figure|fig\.?)\s*\d+[^.!?\n]*)",
        re.IGNORECASE,
    )
    for card in cards:
        if card.image_id in current_set:
            continue
        if current_sections and (card.product, card.section_id) not in current_sections:
            continue
        citations = [card.explicit_citation_text] if card.explicit_citation_text else citation_re.findall(card.before_text or "")
        if not citations:
            continue
        owner_terms = (
            _normalised_english_terms(citations[-1])
            - _GENERIC_ENTITY_TERMS
            - citation_stop
        )
        visual_terms = (
            _normalised_english_terms(card.image_caption)
            - _GENERIC_ENTITY_TERMS
            - citation_stop
        )
        if len(owner_terms & answer_terms) >= 2 or len(visual_terms & answer_terms) >= 2:
            matched.append(card)
    return matched


def detect_evidence_anomalies(
    *,
    question: str,
    answer: str,
    current_images: list[str],
    cards: list[ImageEvidence],
) -> list[str]:
    """Return auditable reasons that justify running the expensive selector.

    The gate is intentionally conservative.  A fluent answer is not rewritten just
    because another set might also be plausible; at least one structural, lifecycle
    or coverage contradiction must be observable from the question and manual
    metadata first.
    """

    q = (question or "").lower()
    card_map = {card.image_id: card for card in cards}
    current_cards = [card_map[image_id] for image_id in current_images if image_id in card_map]
    current_set = set(current_images)
    reasons: list[str] = []

    def owned_text(card: ImageEvidence) -> str:
        return f"{card.heading} {card.before_text[-420:]} {card.image_caption}".lower()

    def add_reason(reason: str) -> None:
        if reason not in reasons:
            reasons.append(reason)

    # An explicit source-list slice always requires deterministic scope review.
    # This is not an error claim about the current answer; it prevents the
    # conservative gate from skipping validation of formal entry boundaries.
    if _parse_ordered_range_request(question):
        add_reason("ordered_range_scope_review")

    # Source provenance, not an asset id's numeric suffix, determines figure
    # order. Generated identifiers can be reused later in the same manual.
    if len(current_images) >= 2:
        # A preserved source group is stronger ordering evidence than asset names
        # or semantic-section order. Flag only a pure permutation of one complete
        # source group; the deterministic repair below cannot add or remove images.
        if len(current_cards) == len(current_images):
            group_ids = {card.source_group_id for card in current_cards if card.source_group_id}
            if len(group_ids) == 1:
                source_order = current_cards[0].source_group_images
                source_rank = {image_id: index for index, image_id in enumerate(source_order)}
                section_orders = {card.section_images for card in current_cards}
                if (
                    len(section_orders) == 1
                    and next(iter(section_orders)) == source_order
                    and all(image_id in source_rank for image_id in current_images)
                ):
                    expected = sorted(current_images, key=source_rank.__getitem__)
                    if expected != current_images:
                        add_reason("source_group_order_mismatch")

    if _locator_operation_complements(question, current_set, cards):
        add_reason("operation_target_locator_missing")

    if _minimal_goal_role_pair(question, current_cards, cards):
        add_reason("single_goal_cross_parent_coverage_risk")

    for concept_card in (card for card in cards if card.origin == "concept_cross_reference"):
        family = {
            card.image_id
            for card in cards
            if card.product == concept_card.product
            and card.section_id == concept_card.section_id
            and card.origin in {"retrieved", "concept_cross_reference"}
        }
        if current_set & family and family - current_set:
            add_reason("concept_binding_incomplete")

    if _answered_explicit_figure_cards(answer, current_images, cards):
        add_reason("answered_explicit_figure_missing")

    # A safety figure embedded in the requested procedure is part of the
    # procedure, not optional background.  Trigger semantic review when the
    # answer states that warning but omits its same-section figure.
    if _is_procedure_question(question) and current_cards:
        current_sections = {(card.product, card.section_id) for card in current_cards}
        answer_lower = (answer or "").lower()
        warning_terms = (
            "warning", "important", "caution", "danger", "do not", "never",
            "警告", "注意", "切勿", "不要", "严禁",
        )
        omitted_in_procedure_warning = any(
            card.image_id not in current_set
            and (card.product, card.section_id) in current_sections
            and any(term in f"{card.image_caption} {card.before_text} {card.after_text}".lower() for term in warning_terms)
            and any(term in answer_lower for term in warning_terms)
            for card in cards
        )
        if omitted_in_procedure_warning:
            add_reason("in_procedure_warning_figure_missing")

    if _adjacent_topic_overview_cards(question, current_set, cards):
        add_reason("adjacent_topic_overview_missing")

    # Broad pre-operation questions are readiness checklists.  If a retrieved,
    # unselected figure is explicitly owned by a mandatory pre-use statement, the
    # current evidence may cover the operator while omitting the equipment (or vice
    # versa).  This only opens semantic review; it never selects a figure itself.
    if _is_pre_operation_readiness_question(question):
        pre_use_markers = (
            "before use", "before using", "before starting", "prior to use",
            "使用前", "操作前", "启动前", "运行前", "开机前", "运动前",
        )
        prerequisite_markers = (
            "must", "should", "ensure", "make sure", "check", "adjust",
            "必须", "应当", "应该", "应先", "确保", "检查", "调节",
        )
        relevant_products = {card.product for card in current_cards}
        def explicit_pre_use_text(card: ImageEvidence) -> str:
            # The owning prose must state the prerequisite explicitly.  A broad
            # parent heading such as "pre-use adjustment and moving" is not enough
            # to turn an optional transport figure into a required readiness step.
            return f"{card.before_text[-420:]} {card.image_caption}".lower()

        omitted_prerequisite = any(
            card.image_id not in current_set
            and card.image_category not in {"noise", "info_table"}
            and (not relevant_products or card.product in relevant_products)
            and any(marker in explicit_pre_use_text(card) for marker in pre_use_markers)
            and any(marker in explicit_pre_use_text(card) for marker in prerequisite_markers)
            for card in cards
        )
        if omitted_prerequisite:
            add_reason("pre_operation_readiness_prerequisite_missing")

    model_scope = any(
        token in q
        for token in ("different model", "different version", "configuration", "variant", "不同型号", "不同机型", "不同配置")
    )
    if model_scope and any(card.image_category == "info_table" for card in current_cards):
        has_overview = any(
            card.image_category == "part_view"
            and any(token in card.image_caption.lower() for token in ("位置", "location", "overview", "部件"))
            for card in current_cards
        )
        overview_candidate = any(
            card.image_id not in current_set
            and card.image_category == "part_view"
            and any(token in card.image_caption.lower() for token in ("位置", "location", "overview", "部件"))
            and any(_common_heading_prefix(card.heading, selected.heading) >= 2 for selected in current_cards)
            for card in cards
        )
        if not has_overview and overview_candidate:
            add_reason("scope_qualifier_missing_configuration_overview")

    # A cleaning-frequency table is background evidence for a how-to cleaning
    # procedure unless the question asks about interval/frequency itself.
    if _is_procedure_question(question) and not _is_information_table_request(question):
        for card in current_cards:
            if card.image_category != "info_table":
                continue
            text = owned_text(card)
            if any(token in text for token in ("frequency", "interval", "weeks", "months", "频率", "周期", "每两周", "每月")):
                add_reason("procedure_contains_frequency_background")

    subtype_tokens = set(re.findall(r"(?:[a-z]+\d+[a-z0-9-]*|\d+[a-z]+[a-z0-9-]*)", q))
    if subtype_tokens and _is_procedure_question(question):
        has_operation = any(card.image_category == "schematic" for card in current_cards)
        has_overview = any(
            card.image_category == "part_view"
            and any(token in card.image_caption.lower() for token in ("位置", "location", "overview", "部件"))
            for card in current_cards
        )
        if has_operation and has_overview:
            add_reason("subtype_operation_contains_broad_overview")

    asks_filter_service = (
        any(token in q for token in ("filter", "滤网"))
        and any(token in q for token in ("clean", "replace", "更换", "清洁"))
        and not any(token in q for token in ("packaging", "plastic wrap", "包装", "塑料") )
    )
    if asks_filter_service:
        maintenance_cards = [
            card for card in cards
            if any(token in card.heading.lower() for token in ("maintenance", "replacement", "clean", "维护", "更换", "清洁"))
            and any(token in card.heading.lower() for token in ("filter", "滤网"))
        ]
        setup_selected = any(
            any(token in card.heading.lower() for token in ("installation", "setup", "packaging", "安装", "准备", "包装"))
            for card in current_cards
        )
        # No image is safer than an image for a different filter lifecycle/object.
        # Run this repair only when the answer visibly selected setup imagery.
        if maintenance_cards and current_cards and setup_selected:
            add_reason("filter_service_lifecycle_conflict")
        elif maintenance_cards and not current_images:
            # Candidate existence is enough to request semantic review, but never
            # enough for deterministic selection: the similarly named candidate
            # may depict a different physical object.
            add_reason("filter_service_visual_review_needed")

    broad_use = any(token in q for token in ("how to use", "how do i use", "如何使用", "怎么使用"))
    asks_change = any(token in q for token in ("change", "modify", "setting", "configure", "default", "修改", "设置", "默认"))
    if broad_use and not asks_change and any(
        any(token in card.heading.lower() for token in ("changing", "setting", "configuration", "修改", "设置"))
        for card in current_cards
    ):
        add_reason("normal_use_answered_with_configuration_stage")

    # An explicitly paired operation heading defines a closed visual group.
    for card in current_cards:
        if not _has_paired_operation_heading(question, card.heading):
            continue
        section_images = {
            peer.image_id
            for peer in cards
            if peer.product == card.product
            and peer.section_id == card.section_id
            and peer.image_category not in {"noise", "info_table"}
        }
        if section_images - current_set:
            add_reason("paired_operation_group_incomplete")

    # Compact screen/state sections are evaluated as a complete transition group.
    if any(token in q for token in ("screen", "界面", "显示")):
        for card in current_cards:
            leaf = _heading_parts(card.heading)[-1] if _heading_parts(card.heading) else ""
            if any(token in leaf for token in ("screen", "display", "界面")):
                section_images = {peer.image_id for peer in cards if peer.product == card.product and peer.section_id == card.section_id}
                if 1 < len(section_images) <= 6 and section_images - current_set:
                    add_reason("state_screen_group_incomplete")

    # A multi-action request with a contiguous sequence may be missing the final
    # state in the same source subsection.
    multi_action = bool(re.search(r"\b(?:and|then)\b|以及|并且|并", q))
    storage_or_location_query = any(
        token in q for token in ("store", "stored", "storage", "where", "place", "location", "存放", "位置", "哪里")
    )
    weak_sequence_gate = os.getenv(
        "EVIDENCE_SELECTOR_ENABLE_WEAK_SEQUENCE_GATE", "0"
    ).strip().lower() in {"1", "true", "yes", "on"}
    if weak_sequence_gate and multi_action and not storage_or_location_query and len(current_images) >= 2:
        for card in current_cards:
            same_section = [peer for peer in cards if peer.product == card.product and peer.section_id == card.section_id]
            family = [re.match(r"^(.*_)(\d+)$", image_id) for image_id in current_images]
            if not family or not all(family) or len({match.group(1) for match in family if match}) != 1:
                continue
            prefix = family[0].group(1)
            next_number = max(int(match.group(2)) for match in family if match) + 1
            next_card = next(
                (
                    peer for peer in same_section
                    if peer.image_id not in current_set
                    and (match := re.match(r"^(.*_)(\d+)$", peer.image_id))
                    and match.group(1) == prefix
                    and int(match.group(2)) == next_number
                ),
                None,
            )
            if next_card and any(term in owned_text(next_card) for term in _english_topic_terms(question)):
                add_reason("multi_action_sequence_tail_missing")

    # Plural categories may continue into one immediately adjacent sibling section.
    explicit_count = bool(
        re.search(r"\b(?:two|three|four|five|six|all|every)\b|\d+|两种|两个|三种|三个|全部|所有", q)
    )
    plural_stems = {word[:-1] for word in re.findall(r"\b[a-z]{5,}s\b", q)}
    if plural_stems and not explicit_count:
        for seed in current_cards:
            seed_leaf = _heading_parts(seed.heading)[-1] if _heading_parts(seed.heading) else ""
            stems = {stem for stem in plural_stems if stem in seed_leaf}
            if not stems:
                continue
            seed_scope_score = _heading_query_score(question, seed.heading)
            def compatible_plural_sibling(card: ImageEvidence) -> bool:
                if card.product != seed.product or card.image_id in current_set:
                    return False
                if abs(card.section_order - seed.section_order) > 1 or _common_heading_prefix(card.heading, seed.heading) < 2:
                    return False
                if _heading_query_score(question, card.heading) < seed_scope_score:
                    return False
                leaf = _heading_parts(card.heading)[-1] if _heading_parts(card.heading) else ""
                if not any(stem in leaf for stem in stems):
                    return False
                if storage_or_location_query and not any(
                    token in card.heading.lower()
                    for token in ("store", "storage", "storing", "location", "存放", "储物", "位置")
                ):
                    return False
                return True

            if any(compatible_plural_sibling(card) for card in cards):
                add_reason("plural_category_adjacent_sibling_missing")

    if any(token in q for token in ("care", "maintain", "maintenance", "grease", "lubricat", "维护", "保养", "润滑")):
        if any("cable" in owned_text(card) or "拉索" in owned_text(card) for card in current_cards):
            if any(
                card.image_id not in current_set
                and ("cable" in owned_text(card) or "拉索" in owned_text(card))
                and any(token in owned_text(card) for token in ("grease", "lubricat", "润滑"))
                and not any(token in owned_text(card) for token in ("pivot point", "bearing", "枢轴", "轴承"))
                for card in cards
            ):
                add_reason("maintenance_action_sibling_missing")

    unsupported_markers = (
        "not provide", "not covered", "does not provide", "未提供", "未找到", "没有说明", "contact after-sales"
    )
    if not current_images and any(marker in (answer or "").lower() for marker in unsupported_markers):
        best_heading_score = max((_heading_query_score(question, card.heading) for card in cards), default=0)
        if best_heading_score >= 2:
            add_reason("unsupported_literal_action_has_canonical_object_section")

    # A custom asset run plus an alias-like singleton is a deterministic family
    # inconsistency (for example custom_04..06 plus ManualXX_4).
    groups: dict[str, list[int]] = {}
    for image_id in current_images:
        match = re.match(r"^(.*_)(\d+)$", image_id)
        if match and not match.group(1).lower().startswith("manual"):
            groups.setdefault(match.group(1), []).append(int(match.group(2)))
    for prefix, numbers in groups.items():
        numbers = sorted(numbers)
        if len(numbers) >= 3 and numbers == list(range(numbers[0], numbers[-1] + 1)):
            has_next = any(
                (match := re.match(r"^(.*_)(\d+)$", card.image_id))
                and match.group(1) == prefix
                and int(match.group(2)) == numbers[-1] + 1
                for card in cards
            )
            alias = any(
                not image_id.startswith(prefix)
                and (match := re.match(r"^(.*_)(\d+)$", image_id))
                and int(match.group(2)) == numbers[0]
                for image_id in current_images
            )
            if has_next and alias:
                add_reason("custom_asset_family_alias_conflict")

    finger_focus = any(token in q for token in ("finger", "fingers", "pinch", "trap", "手指", "夹手"))
    if finger_focus:
        direct_candidates = [
            card for card in cards
            if any(token in (card.image_caption or "").lower() for token in ("finger", "fingers", "pinch", "trap", "手指", "夹"))
            and not any(token in (card.image_caption or "").lower() for token in ("carry", "lift", "two people", "搬运", "双人"))
        ]
        if {card.image_id for card in direct_candidates} != current_set:
            add_reason("focused_hazard_visual_set_mismatch")

    # Explicit control cardinality and aftercare redundancy are directly testable.
    if explicit_count and any(token in q for token in ("switch", "lever", "开关", "手柄", "操纵杆")):
        aftercare_selected = any(
            any(token in owned_text(card) for token in ("unauthorized", "children", "accidental starting", "not running", "儿童"))
            for card in current_cards
        )
        if aftercare_selected:
            add_reason("counted_control_contains_aftercare_figure")

    # Broad category introductions should cover their immediately adjacent sibling
    # sections (limitations/requirements/characteristics) when only one was chosen.
    broad_category = any(token in q for token in ("introduce", "what kinds", "介绍", "有哪些"))
    if broad_category and any(token in q for token in ("limitations", "requirements", "characteristics", "限制", "要求", "特性")):
        if current_cards and any(
            card.image_id not in current_set
            and card.product == current_cards[0].product
            and _common_heading_prefix(card.heading, current_cards[0].heading) >= 2
            for card in cards
        ):
            add_reason("broad_category_sibling_sections_missing")

    # Interface introductions need a control overview as well as display details.
    if any(token in q for token in ("console", "control panel", "控制台")) and any(
        token in q for token in ("display", "显示")
    ):
        current_has_overview = any(
            any(token in card.image_caption.lower() for token in ("部件标注", "control overview", "controls overview"))
            for card in current_cards
        )
        candidate_overview = any(
            card.image_id not in current_set
            and any(token in card.image_caption.lower() for token in ("部件标注", "control overview", "controls overview"))
            for card in cards
        )
        if not current_has_overview and candidate_overview:
            add_reason("interface_overview_missing")

    if any(token in q for token in ("accessories", "附件")) and any(
        token in q for token in ("what", "which", "哪些", "什么")
    ):
        if not any(card.image_category == "info_table" for card in current_cards) and any(
            card.image_category == "info_table" for card in cards
        ):
            add_reason("accessory_enumeration_lacks_specification_table")

    # Source-level provenance catches compact figure sets that semantic sectioning
    # split apart.  The same predicate is reused by deterministic completion, which
    # keeps the gate auditable and prevents a broad candidate expansion by itself
    # from rewriting an otherwise coherent answer.
    if any(
        _should_complete_source_group(question, group_cards, current_set)
        for group_cards in _cards_by_source_group(cards).values()
    ):
        add_reason("source_figure_group_incomplete")

    if _referenced_sibling_entries(question, current_cards, cards):
        add_reason("referenced_sibling_entry_figure_missing")

    if _unspecified_variant_sections(question, current_set, cards):
        add_reason("unnamed_model_variant_branch_missing")

    return reasons


def refine_answer_evidence(
    *,
    question: str,
    answer: str,
    candidate_images: Iterable[str],
    engine,
    captions: dict[str, dict],
    route_products: Iterable[str],
    llm_call: Callable,
    model: str | None = None,
) -> EvidenceSelectionResult:
    """Plan, validate and if needed compose a fully bound image-aware answer."""

    current = extract_anchor_ids(answer)
    ordered_range = _parse_ordered_range_request(question)
    validated_range_images: list[str] = []
    if ordered_range:
        try:
            from ordered_range_experiment import validated_scope_images

            validated_range_images = validated_scope_images(
                question,
                ordered_range.start,
                ordered_range.end,
                ordered_range.from_end,
            )
        except Exception:
            validated_range_images = []
    cards = build_candidate_evidence(
        question=question,
        candidate_images=[*candidate_images, *validated_range_images],
        current_images=current,
        engine=engine,
        captions=captions,
        route_products=route_products,
        max_cards=max(8, int(os.getenv("EVIDENCE_SELECTOR_MAX_CARDS", "48"))),
    )
    base_trace: dict[str, object] = {
        "status": "keep",
        "current_images": current,
        "candidate_count": len(cards),
        "structural_candidates": sum(card.origin == "structural_neighbor" for card in cards),
        "source_group_candidates": sum(card.origin == "source_group" for card in cards),
    }
    if ordered_range:
        base_trace["ordered_range_request"] = asdict(ordered_range)
    if validated_range_images:
        base_trace["validated_ordered_scope"] = list(validated_range_images)
    if not cards:
        base_trace["status"] = "no_candidates"
        return EvidenceSelectionResult(answer, current, False, base_trace)

    gate_reasons = detect_evidence_anomalies(
        question=question,
        answer=answer,
        current_images=current,
        cards=cards,
    )
    base_trace["gate_reasons"] = gate_reasons
    gate_enabled = os.getenv("EVIDENCE_SELECTOR_CONSERVATIVE_GATE", "1").strip().lower()
    if gate_enabled not in {"0", "false", "no", "off"} and not gate_reasons:
        base_trace["status"] = "gate_keep"
        return EvidenceSelectionResult(answer, current, False, base_trace)

    # Ordering is a deterministic formatting defect, not a new evidence-selection
    # problem.  If it is the only anomaly, preserve the exact image set and bypass
    # every stage that is allowed to add or remove evidence.  The composer still
    # runs below so each unchanged figure can be rebound to the correct prose.
    order_only = gate_reasons in (
        ["source_group_order_mismatch"],
    )
    concept_only = gate_reasons == ["concept_binding_incomplete"]
    planner_attempts = 0
    if order_only or concept_only:
        plan = {
            "requirements": [],
            "selected_images": _sort_by_document_order(current, cards),
            "bindings": [],
            "decision": "revise",
            "confidence": 1.0,
            "reason": (
                "deterministic_order_only"
                if order_only
                else "deterministic_concept_binding_completion"
            ),
        }
    else:
        payload = {
            "question": question,
            "candidate_evidence": [card.prompt_payload() for card in cards],
        }
        if ordered_range:
            payload["ordered_range_request"] = asdict(ordered_range)
        planner_messages = [{"role": "user", "content": json.dumps(payload, ensure_ascii=False)}]
        try:
            plan = None
            for planner_attempts in (1, 2):
                response, _route = llm_call(
                    max_tokens=int(os.getenv("EVIDENCE_SELECTOR_MAX_TOKENS", "2600")),
                    system=_PLANNER_SYSTEM,
                    messages=planner_messages,
                    model=model,
                )
                raw_plan = _response_text(response)
                plan = _parse_json_object(raw_plan)
                if plan is not None:
                    break
                planner_messages = [
                    *planner_messages,
                    {"role": "assistant", "content": raw_plan[:2000]},
                    {
                        "role": "user",
                        "content": (
                            "The previous response was not one valid JSON object. "
                            "Return the required JSON schema only; do not add prose or a code fence."
                        ),
                    },
                ]
        except Exception as exc:
            base_trace.update({"status": "planner_error", "error": type(exc).__name__})
            return EvidenceSelectionResult(answer, current, False, base_trace)

    if not plan:
        base_trace["status"] = "invalid_planner_json"
        base_trace["planner_attempts"] = planner_attempts
        return EvidenceSelectionResult(answer, current, False, base_trace)

    allowed = {card.image_id for card in cards}
    selected = _selected_from_plan(plan)
    if validated_range_images:
        selected = list(validated_range_images)
        plan["decision"] = "revise"
        plan["confidence"] = 1.0
        plan["reason"] = "validated_source_entry_slice"
    if any(image_id not in allowed for image_id in selected):
        base_trace["status"] = "planner_selected_unknown_image"
        return EvidenceSelectionResult(answer, current, False, base_trace)

    role_pair = _minimal_goal_role_pair(
        question,
        [card for card in cards if card.image_id in set(current)],
        cards,
    )
    role_pair_applied = role_pair is not None
    if role_pair:
        selected = [role_pair[0].image_id, role_pair[1].image_id]
        plan["scope_mode"] = "narrow_goal"
    selected = _sort_by_document_order(selected, cards)
    if order_only or validated_range_images:
        structurally_added: list[str] = []
        constraint_audit: dict[str, list[str]] = {"added": [], "removed": []}
    else:
        selected, structurally_added = _complete_structural_groups(
            question,
            selected,
            cards,
            baseline_images=current,
            allow_source_group_completion=(
                ordered_range is None
                and str(plan.get("scope_mode") or "").lower() != "narrow_goal"
            ),
        )
        selected, constraint_audit = _apply_evidence_constraints(
            question,
            answer,
            selected,
            cards,
            baseline_images=current,
            allow_topic_expansion=(
                ordered_range is None
                and str(plan.get("scope_mode") or "").lower() != "narrow_goal"
            ),
        )
    try:
        confidence = float(plan.get("confidence", 0.0))
    except (TypeError, ValueError):
        confidence = 0.0
    threshold = float(os.getenv("EVIDENCE_SELECTOR_MIN_CONFIDENCE", "0.72"))
    decision = str(plan.get("decision") or "").lower()
    base_trace.update(
        {
            "requirements": plan.get("requirements") or [],
            "selected_images": selected,
            "structurally_added": structurally_added,
            "constraint_added": constraint_audit["added"],
            "constraint_removed": constraint_audit["removed"],
            "confidence": round(confidence, 3),
            "decision": decision,
            "planner_attempts": planner_attempts,
            "reason": str(plan.get("reason") or "")[:500],
            "role_pair_applied": role_pair_applied,
        }
    )

    if selected == current:
        base_trace["status"] = "selection_unchanged"
        return EvidenceSelectionResult(answer, current, False, base_trace)
    # The planner deliberately does not see the current answer, so its keep/revise
    # label cannot be trusted as a comparison signal.  Confidence and the validated
    # selected set decide whether composition runs.
    if confidence < threshold:
        base_trace["status"] = "low_confidence_keep"
        return EvidenceSelectionResult(answer, current, False, base_trace)

    card_map = {card.image_id: card for card in cards}
    compose_payload = {
        "question": question,
        "required_output_language": _required_language(question),
        "requirements": plan.get("requirements") or [],
        "bindings": plan.get("bindings") or [],
        "selected_images_in_required_order": selected,
        "selected_evidence": [card_map[image_id].prompt_payload() for image_id in selected],
        "current_answer_for_repair": answer,
    }
    required_language = _required_language(question)
    composer_messages = [
        {"role": "user", "content": json.dumps(compose_payload, ensure_ascii=False)}
    ]
    rewritten = ""
    composer_attempts = 0
    try:
        for composer_attempts in (1, 2):
            response, _route = llm_call(
                max_tokens=int(os.getenv("EVIDENCE_COMPOSER_MAX_TOKENS", "4096")),
                system=_COMPOSER_SYSTEM,
                messages=composer_messages,
                model=model,
            )
            rewritten = _JSON_FENCE_RE.sub("", _response_text(response).strip()).strip()
            if (
                rewritten
                and _anchors_are_bound(rewritten, selected)
                and _language_matches(required_language, rewritten)
            ):
                break
            composer_messages = [
                *composer_messages,
                {"role": "assistant", "content": rewritten[:6000]},
                {
                    "role": "user",
                    "content": (
                        "The previous rewrite failed the deterministic binding check. "
                        "Return the complete final answer again, using every anchor in "
                        f"this exact order exactly once: {selected}. Use no other anchors, "
                        f"and keep the answer language {required_language}."
                    ),
                },
            ]
    except Exception as exc:
        base_trace.update({"status": "composer_error", "error": type(exc).__name__})
        return EvidenceSelectionResult(answer, current, False, base_trace)

    base_trace["composer_attempts"] = composer_attempts
    if (
        not rewritten
        or not _anchors_are_bound(rewritten, selected)
        or not _language_matches(required_language, rewritten)
    ):
        base_trace["status"] = "invalid_text_image_binding"
        base_trace["composer_anchor_ids"] = extract_anchor_ids(rewritten)
        base_trace["required_output_language"] = required_language
        return EvidenceSelectionResult(answer, current, False, base_trace)

    base_trace["status"] = "revised"
    return EvidenceSelectionResult(rewritten, selected, True, base_trace)
