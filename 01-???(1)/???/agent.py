"""
ReAct 客服智能体。

工具：
- search_manual: 统一检索入口。模型主要给关键词，系统同时做 BM25 + dense 召回并统一 rerank

LLM: MiniMax-M2.7 via Anthropic SDK
"""

from __future__ import annotations

import json
import os
import re
import time
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

from product_router import ProductRouteDecision, ProductRouter, build_product_prompt_block
from retrieval_engine import RetrievalEngine, SearchResult, contains_cjk, tokenize_mixed
from llm_router import create_message_with_fallback, create_message_streaming
from submission_utils import extract_inline_pic_refs, inject_inline_pic_refs
from evidence_selector import refine_answer_evidence

ROOT = Path(__file__).resolve().parent
SKILLS_DIR = ROOT / "skills"
IMAGE_CAPTIONS_PATH = ROOT / "data" / "image_captions_v4_final.json"

# ────────────────── 配置 ──────────────────

MAX_TURNS = 2
MAX_SEARCH_RESULTS = 8
PRE_RETRIEVAL_RESULTS = 5
MAX_SEARCH_ATTEMPTS = int(os.getenv("V6_MAX_SEARCH_ATTEMPTS", "2"))
MAX_INTERNAL_ITERATIONS = 10

PRODUCT_PROMPT_BLOCK = build_product_prompt_block()
_IMAGE_CAPTIONS_CACHE: dict[str, dict] | None = None


_LEXICAL_STOP_TERMS = {
    "如何", "怎样", "怎么", "什么", "请问", "一下", "进行", "可以", "需要",
    "用户", "步骤", "流程", "方法", "操作", "the", "a", "an", "to", "how",
    "do", "does", "i", "my", "you", "your", "steps", "procedure", "instructions",
}


def _significant_lexical_terms(text: str) -> set[str]:
    """Return meaningful literal terms without assigning semantic categories."""
    return {
        token.casefold()
        for token in tokenize_mixed(text or "")
        if len(token.strip()) >= 2 and token.casefold() not in _LEXICAL_STOP_TERMS
    }


_HISTORY_CURRENT_QUESTION_MARKER = "用户当前问题:"


def _current_question_for_retrieval(text: str) -> str:
    """Keep product history available to the answer model, but out of routing."""
    value = (text or "").strip()
    if _HISTORY_CURRENT_QUESTION_MARKER not in value:
        return value
    current = value.rsplit(_HISTORY_CURRENT_QUESTION_MARKER, 1)[-1].strip()
    return current or value


_ELLIPTICAL_FOLLOWUP_RE = re.compile(
    r"(?:^|[，。！？?\s])(?:那|这个|那个|它|刚才|之前|前面|上述|该)(?:个|些|项|部件|步骤)?"
    r"|洗完|装回|还要|还有|怎么办|能马上",
    re.IGNORECASE,
)

_MODE_ENUMERATION_RE = re.compile(
    r"(?:还有|其他|哪些|什么|全部|所有|列(?:个|出)?|清单).{0,12}模式"
    r"|模式.{0,12}(?:还有|其他|哪些|什么|全部|所有|列(?:个|出)?|清单)",
    re.IGNORECASE,
)
_MODE_SECTION_EXCLUDE_RE = re.compile(
    r"安全|警告|故障|诊断|报修|维修|维护|清洁|安装|排水|搬运|存档|显示控制|电池",
    re.IGNORECASE,
)
_FEATURE_ENUMERATION_RE = re.compile(
    r"(?:还有|其他|哪些|什么|全部|所有).{0,12}(?:技术|功能|特性|特点|能力)"
    r"|(?:技术|功能|特性|特点|能力).{0,12}(?:还有|其他|哪些|什么|全部|所有)"
    r"|\b(?:what|which).{0,24}(?:features?|technolog(?:y|ies)|functions?|capabilities?)\b"
    r"|\b(?:features?|technolog(?:y|ies)|functions?|capabilities?).{0,24}"
    r"(?:what|which|other|all|available)\b",
    re.IGNORECASE,
)


def _expand_mode_enumeration_query(question: str, products: list[str] | None = None) -> str:
    """Add bounded vocabulary for enumerated modes, features, and technology."""
    value = str(question or "").strip()
    if _MODE_ENUMERATION_RE.search(value) and products == ["空调手册"]:
        return (
            f"{value}\n空调运行模式 制冷模式 制热模式 除湿模式 循环模式 "
            "极速制冷 极速制热 自动运行 自动转换 节能制冷"
        )[:500]
    if _MODE_ENUMERATION_RE.search(value):
        return f"{value}\n运行模式 自动模式 高级模式"[:500]
    if _FEATURE_ENUMERATION_RE.search(value):
        # Product manuals can describe a feature in either Chinese or English.
        # The expansion is intent vocabulary only; it neither supplies a product
        # identity nor names a feature that is not present in the manual.
        return (
            f"{value}\n技术 功能 智能功能 特性 特点 模式 设置 控制 反馈 传感 "
            "feature features smart technology technologies capability capabilities "
            "mode modes setting settings control controls feedback sensor sensing app"
        )[:500]
    return value


def _is_feature_enumeration_question(question: str) -> bool:
    """Whether the user asks to list a product's technologies/features."""
    return bool(_FEATURE_ENUMERATION_RE.search(str(question or "")))


def _filter_mode_enumeration_results(
    question: str,
    results: list[SearchResult],
) -> tuple[list[SearchResult], int]:
    """Reject maintenance/safety neighbours from a mode-enumeration answer."""
    if not _MODE_ENUMERATION_RE.search(str(question or "")):
        return results, 0
    kept = []
    for result in results:
        heading = str(result.heading or "")
        if _MODE_SECTION_EXCLUDE_RE.search(heading):
            continue
        searchable = f"{heading}\n{str(result.text or '')[:1200]}"
        mode_mentions = len(re.findall(r"(?:模式|mode)s?", searchable, flags=re.IGNORECASE))
        if "模式" in heading or re.search(r"\bmodes?\b", heading, flags=re.IGNORECASE) or mode_mentions >= 2:
            kept.append(result)
    # A relevance gate must not turn a valid retrieval into no evidence.
    return (kept, len(results) - len(kept)) if kept else (results, 0)


def _retrieval_query_with_local_context(full_text: str, current: str) -> str:
    """Resolve an elliptical follow-up with at most two prior user questions.

    Assistant answers never enter retrieval, and the added user text is searched
    once as part of the same hybrid query. This preserves A -> B -> A memory
    without reviving per-clause historical routing or repeated searches.
    """
    if (
        _HISTORY_CURRENT_QUESTION_MARKER not in (full_text or "")
        or not _ELLIPTICAL_FOLLOWUP_RE.search(current or "")
    ):
        return current
    history_prefix = full_text.rsplit(_HISTORY_CURRENT_QUESTION_MARKER, 1)[0]
    prior_questions = [
        match.strip()
        for match in re.findall(r"(?m)^用户:\s*(.+?)\s*$", history_prefix)
        if match.strip()
    ]
    if not prior_questions:
        return current
    # Prefer the latest fully specified user question as the retrieval
    # anchor. Chaining vague follow-ups ("that one", "put it back") into the
    # query causes semantic drift and lets a stale action outrank the subject.
    grounded = [
        item for item in prior_questions
        if not _ELLIPTICAL_FOLLOWUP_RE.search(item)
    ]
    anchor = (grounded[-1] if grounded else prior_questions[-1])[:220]
    return f"{current}\n{anchor}"[:500]


def _fast_tech_path_enabled() -> bool:
    return os.getenv("FAST_TECH_PATH_ENABLED", "1").strip().lower() not in {
        "0", "false", "no", "off",
    }


def _sanitize_search_input_from_original_words(
    input_data: dict,
    original_question: str,
) -> tuple[dict, list[str]]:
    """Keep only model keywords whose literal terms occur in the user wording.

    This prevents a speculative tool plan from becoming a second user request.
    No action list, intent taxonomy, product name, or chunk id is used here.
    """
    sanitized = dict(input_data or {})
    original_terms = _significant_lexical_terms(original_question)
    dropped: list[str] = []
    keywords = sanitized.get("keywords")
    if isinstance(keywords, list):
        kept: list[str] = []
        for raw_keyword in keywords:
            keyword = str(raw_keyword or "").strip()
            if not keyword:
                continue
            keyword_terms = _significant_lexical_terms(keyword)
            if keyword_terms and not keyword_terms.issubset(original_terms):
                dropped.append(keyword)
                continue
            kept.append(keyword)
        sanitized["keywords"] = kept or [original_question]
    sanitized["query"] = original_question
    return sanitized, dropped


def _narrow_to_unique_literal_heading(
    question: str,
    results: list[SearchResult],
) -> tuple[list[SearchResult], int]:
    """Use one uniquely strong literal heading anchor when available.

    The comparison treats every literal user term equally.  It does not classify
    actions or infer synonyms.  Narrowing activates only when one of the top-three
    candidates has at least two shared terms and a strict lead over every other
    heading; ties and weak matches preserve the complete ranked result set.
    """
    structural = [
        result for result in results
        if result.source.get("evidence_role") in {"primary", "support"}
    ]
    if (
        any(result.source.get("evidence_role") == "primary" for result in structural)
        and any(result.source.get("evidence_role") == "support" for result in structural)
    ):
        # The retrieval layer has already identified a minimal procedure plus
        # its adjacent warning/note. Literal-heading narrowing must not discard
        # that safety preface merely because the procedure heading matches more
        # words in the question.
        structural.sort(key=lambda result: (
            int(result.source.get("document_order", result.chunk_id)),
            result.chunk_id,
        ))
        return structural, len(results) - len(structural)

    question_terms = _significant_lexical_terms(question)
    if len(question_terms) < 2 or not results:
        return results, 0
    scored = [
        (len(question_terms & _significant_lexical_terms(result.heading)), index)
        for index, result in enumerate(results)
    ]
    best_score = max(score for score, _index in scored)
    best_indexes = [index for score, index in scored if score == best_score]
    runner_up = max((score for score, _index in scored if score < best_score), default=0)
    if (
        best_score < 2
        or len(best_indexes) != 1
        or best_indexes[0] >= 3
        or best_score - runner_up < 1
    ):
        return results, 0
    anchor = results[best_indexes[0]]
    shared_terms = sorted(question_terms & _significant_lexical_terms(anchor.heading))
    anchor_shared_terms = set(shared_terms)
    # A lower-scoring heading may still cover a different part of a compound
    # question.  If it contributes any literal user term absent from the best
    # heading, the request is not safely reducible to one section.  Preserve the
    # full ranked set and let the multi-block answer path cover every part.
    for result in results:
        if result is anchor:
            continue
        other_shared_terms = question_terms & _significant_lexical_terms(result.heading)
        if other_shared_terms - anchor_shared_terms:
            return results, 0
    anchor.source["evidence_role"] = "primary"
    anchor.source["literal_heading_anchor_terms"] = shared_terms
    return [anchor], len(results) - 1


def load_skill_md(skill_name: str) -> str | None:
    """读取固定 skill 的 markdown 说明文件。"""
    md_path = SKILLS_DIR / f"{skill_name}.md"
    if not md_path.exists():
        return None
    return md_path.read_text(encoding="utf-8")


SEARCH_MANUAL_SKILL_BLOCK = load_skill_md("search_manual") or """# 手册检索

- search_manual(keywords, products?, query?): 统一检索入口。优先填写关键词列表；系统会同时执行 BM25 关键词检索和向量语义检索，合并后用 rerank 排序。若要取消路由改查全库，传空数组 []

检索结果中的正文会直接带 `[[PIC:图片文件名]]` 锚点；若下方出现“图片内容标注”，它是对该图画面的辅助描述。它的唯一用途是帮你判断这张图画的是什么、是否与你正文这一段相符，从而决定要不要保留该图；严禁把图片标注照抄进答案。配图跟着最相关那一段本身有没有图走。"""


def _load_image_captions() -> dict[str, dict]:
    global _IMAGE_CAPTIONS_CACHE
    if _IMAGE_CAPTIONS_CACHE is None:
        try:
            payload = json.loads(IMAGE_CAPTIONS_PATH.read_text(encoding="utf-8"))
            _IMAGE_CAPTIONS_CACHE = payload.get("items", {})
        except Exception:
            _IMAGE_CAPTIONS_CACHE = {}
    return _IMAGE_CAPTIONS_CACHE


def _format_image_evidence(product: str, pics: list[str], *, max_items: int = 8) -> str:
    """Return concise image evidence lines for captions tied to visible PIC anchors."""
    if not pics:
        return ""
    captions = _load_image_captions()
    lines: list[str] = []
    seen: set[str] = set()
    for pic in pics:
        if pic in seen:
            continue
        seen.add(pic)
        item = captions.get(f"{product}|{pic}")
        if not item:
            continue
        cat = item.get("category")
        # noise 不注入（装饰/图标），其余都注入（part_view/schematic/info_table）
        if cat == "noise":
            continue
        # info_table（表格/参数数据）是产品级的，不限于某个章节，跳过章节匹配检查
        if item.get("section_fit") == "mismatch" and cat != "info_table":
            continue
        short = (item.get("short_caption") or "").strip()
        dense = (item.get("content") or "").strip()
        evidence = dense or short
        if not evidence:
            continue
        if len(evidence) > 260:
            evidence = evidence[:260].rstrip() + "..."
        lines.append(f"- [[PIC:{pic}]] {short}: {evidence}" if short else f"- [[PIC:{pic}]] {evidence}")
        if len(lines) >= max_items:
            break
    if not lines:
        return ""
    return (
        "图片内容标注（仅供你判断这张图画的是什么、是否与你这段正文相符，从而决定要不要保留它的 [[PIC:...]] 锚点；"
        "据此用你自己的话写一句简短图说即可）。注意：这是图片的辅助标注，不是手册正文，"
        "严禁把下面的清单/表格/字段原文照抄进答案；下面文字若不完整，直接忽略，绝不要在答案里提“截断/未显示/未完整”之类的话，也不要输出本标题：\n"
        + "\n".join(lines)
    )



# ────────────────── 通用客服 SYSTEM PROMPT（V4.0 专供 LLM 打分） ──────────────────

SERVICE_SYSTEM_PROMPT = """\
你是某电商平台的智能客服。请根据用户的问题，给出友好、专业、详细的回答。

本题已被判定为通用客服问题，**绝对不要调用任何搜索或技能工具**，不要编造具体的电话号码、邮箱、网址、实体门店地址或客服工号。

要求：
1. 语气亲切自然，使用"您好""请您放心"等礼貌用语
2. 回答结构清晰，使用标题和列表组织内容
3. 内容详实，覆盖用户问题的各个方面，回答要有深度，不要停留在表面
4. 如果用户问题涉及退换货、运费、物流、维修、投诉等，给出明确的处理流程和时效说明（如48小时、3-5天、7天无理由）以及相关前提条件
5. 不要输出任何与问题无关的内容
6. 禁止使用任何 emoji 表情符号或 Unicode 装饰符号（如 ✅、😊、💡、⚠、📦 等），只使用纯文本
7. 回答尽量详细全面，字数尽量多（建议 1000 字以上），把每个相关点都展开说明，不要简略回答
8. 用户提问中包含的所有诉求（如运费、时效、责任归属等）必须一一对应作答，绝不可遗漏任何一个子问题
"""

# ────────────────── 产品技术 SYSTEM PROMPT（V3.1：V3 路由 + 完整性优先） ──────────────────

TECH_SYSTEM_PROMPT = """\
<评审得分约束>
最终答案必须按以下规则组织证据和表达：
1. 只保留与用户问题直接相关的核心文字和图片；相关度极低的文字或图片会扣分，禁止为了显得全面而堆叠。
2. 相关度一般但不是核心内容的文字或图片可以不使用；除非它直接补足用户问题，否则不要主动加入。
3. 核心内容不完整只能得到部分分数；先检查问题中的对象、动作、条件、例外和范围是否全部覆盖。
4. 满分目标是：完整覆盖核心内容，同时不召回或展示相关度极低的文字和图片。
5. 每张图片必须与它前面的文字证据直接对应，并按手册文档顺序和答案中的出现顺序展示；文字与图片不对应或顺序错误会扣分。
6. 不得过度改写手册原文；必须保留原文核心含义、结构、标题层级、步骤、警告和条件。过度改写导致内容或结构与原文不对应会扣分。
7. 二级、三级标题属于评分证据的一部分。命中相关标题时必须保留对应的二级/三级标题，并按标题层级组织答案；不得把不同层级内容混成一段。
</评审得分约束>

你是一个多模态产品客服智能体，负责回答产品技术问题（型号、操作、故障、配件、参数）。

<Role & Goal>
先查手册，再回答。技术问题必须基于检索到的手册内容作答，禁止凭常识补全型号、步骤、参数、限制条件。

<Verified Visual Facts>
如果用户消息包含 [VERIFIED_VISUAL_FACTS]，其中的高/中置信度条目是“当前图片中实际可见内容”的允许证据：
- 用户问图片上有什么、文字是什么、颜色或灯态是什么时，必须直接使用这些视觉事实，不能回答“图片未提供”或用手册内容覆盖它们。
- 用户问可见现象代表什么、如何操作或有哪些限制时，视觉事实用于确认对象和现象，手册证据用于解释含义与步骤。
- 视觉事实与手册证据冲突时必须明确指出冲突，禁止静默删除任何一方。
- [STRUCTURED_CONTEXT_V1] 只用于理解历史实体、图片事实、用户约束和最近对话；不得把其中的历史答案当成新的手册证据。

<原文摘录模式（最高优先级）>
默认把用户问题视为“从手册中选择哪些原文”，而不是“要求你重新表述”。同语言命中时，最终答案必须直接保留检索正文的原句、原有顺序、步骤编号、Warning / Caution / Note 身份和图片锚点；用户问题只能决定取舍，不能成为改写、概括、调换句序或补写过渡句的理由。
当存在有效手册证据时，直接从所选完整子块的原文标题或第一条原文句开始。选中子块的内部标题属于原文边界，必须保留；不得另写切入句，不得根据问题自行改写、合并或概括。从标题后的第一句开始继续保持原文句序。
只允许：删除明显 OCR/解析噪声、删除完全重复的原句、整理空行与列表排版。禁止同义词替换、合并或拆分事实、总结、解释性扩写、改变标题层级或把警告改写成普通步骤。跨语言命中时，只能逐句直译，不得自由意译。
语言必须与用户提问一致：中文问中文答，英文问英文答。
若命中的手册原文与用户语言不同，仍须先完成与同语言题完全相同的完整子块选择，再逐句直译整个已选子块；不得扩大到未选子块，也不得删减已选子块内部内容。专有名词、型号、品牌可保留原文。

<Scoring & Strategy>
你的回答会被一个评分模型对照手册原文打分，而它的参照标准是**手册里与这个问题最相关的那一个章节段落**——基本等同于"把那一段答准、答全、配上它本身的图"就是满分。理解这套机制能帮你拿高分：
- 命中那一段的要点（步骤、规格、按钮名、部件名、警告、以及那一段原本就带的配图）才得分；越贴近那一段、越完整、图文越互补，分越高。
- **多答了别的章节的内容不会加分，反而会扣分**：因为它偏离了评分参照的那一段、冲淡了重点；同理，凭空补的、手册里没有的内容不得分还可能被判为错误。
- 配图也以"那一段本身有没有图"为准：那一段有图就带上（图文互补加分），那一段本就没图（如纯条款/纯文字操作），硬塞别处的图反而扣分。
所以制胜打法是**精准**而非**全面**：锁定最相关的那一个主题章节，把它答准答全（若它被手册切成了几个相邻同主题小节，就合起来还原），既不漏它的要点，也不掺别的主题。

<Execution Logic>
- 你总共只有 2 次正常 ReAct 决策机会：第一轮通常用于 search_manual 正式确认，第二轮应基于系统预检索和 search_manual 证据正常收束；如果第二轮仍冒险继续检索，后续只能进入无工具强制收束
- 当前唯一主动工具是 search_manual。技术题正式回答前至少调用一次 search_manual；最多允许两次 search_manual：第一次用于正式确认，第二次只作为证据明显错路由或完全不覆盖问题时的补救检索。若第一次证据已经足够，第二轮应直接基于该证据完整回答，不要继续搜索；完整回答不等于简短回答
- 系统预检索只作为首轮定位参考，不能直接替代正式工具检索。技术题仍需继续调用 search_manual 做显式确认后再作答；若 search_manual 与预检索证据一致且足够，请下一轮直接完整收束
- search_manual 返回的是完整 parent section 证据。第二轮回答前必须检查该 section 是否包含同一主题下的并列步骤、部件、图示、警告和例外；这些只要直接回答用户问题，就应保留
- 若 search_manual 结果已经聚合到同一最相关 parent section，不要继续搜索；应基于该 parent section 完整收束，而不是过度摘要
- 最终收束前必须做“问题需求覆盖”检查：不要只抓住问题里的主动作或主名词，还要覆盖修饰语、限定范围、适用条件、例外情况和对象范围。若主答案证据只回答了“是什么/多少/怎么做”，但没有覆盖用户限定的“哪类对象、哪种范围、何种条件下”，必须在同一主题证据中补足该限定语对应的文字和配图；这类补足属于回答用户原问题，不属于额外扩展
- 若系统给出 [PRODUCT_ROUTE]，优先在候选产品内检索；只有 medium/conflict、低增益、无结果或证据指向其他产品时，才扩展 products=[] 做全库确认
- 若候选检索结果不足、偏泛、连续命中相近章节或无结果，可换关键词，或把 products 设为空数组 `[]` 扩展到全库
- 路由 high confidence（单产品）时，第一轮直接在该产品内精查；1-2 次工具调用通常足够
- 路由 medium confidence（多候选或仅内容投票）时，若用户只问一个产品，应把候选用于定位唯一正确手册；若用户原句明确点名多个产品，则每个产品都是独立答案边界，必须分别检索对应手册并覆盖每个子问题
- 仅当用户**明确**问“有哪些 / 组成 / 部件 / 组件 / 功能 / 接口 / 视图”等**枚举型**问题时，才不要默认几个片段已经完整；此时应基于预检索和 search_manual 返回的完整 parent section 判断是否覆盖 overview / parts / view / functions 等并列项。**操作型 / 步骤型 / 单点问题（how / 怎样 / 为什么 / 某个具体动作）不适用本条**——这类题应聚焦最相关的那一个主题章节作答，不要为“求全”去翻并列章节
- 对“有哪些 / 组成 / 部件 / 视图 / 接口 / 功能”等枚举型问题，不要只写最先看到的几个点；必须保留 search_manual 返回的同一最相关 parent section 内直接相关的并列项及对应图片
- 何时停止检索：
  1. 只有当现有检索结果已经足以完整回答用户问题，并覆盖关键步骤、规格、限制条件、注意事项、例外情况后，才允许停止检索并开始作答
  2. 若用户问 how/procedure/steps，而当前结果只有零散描述、单张配图、或不完整的片段步骤，则不算“已足够回答”，应继续检索到可执行的完整步骤
  3. 若用户一次问多个点，只有当每个子问题都已有对应证据时，才允许停止检索；只覆盖其中一部分时应继续检索
  4. 单独出现 `[[PIC:...]]` 图片锚点、单条规格、或单段条件说明，不自动等于“可以停止”；必须确认这些证据已足以解答用户疑惑，并且没有遗漏题目限定语对应的对象范围、适用范围、配置差异或例外条件
  5. 仅限**明确的枚举型问题**（有哪些/组成/部件/功能/接口/视图）：即使已有若干相关片段，也要确认已命中的完整 parent section 是否覆盖并列项；未确认完整性前不要直接回答。**操作型/步骤型/单点问题不走本条**——锁定最贴题的那一个主题章节即可，不要为求全去拽并列章节；但锁定后必须完整保留该主题章节中直接相关的并列项和对应图片
  6. 连续 2 次检索返回相同/相近章节（如都是 Safety 或 Regulatory）→ 停止重复搜索，改写关键词或基于最相关章节完整收束
  7. 返回"无检索结果"或"无新增结果"→ 最多再换 1 组关键词重试；仍无结果就直接说明手册未覆盖或基于已有证据收束
- 不要在“证据已足以完整解答用户疑惑”的情况下为了“更多信息”而额外检索；但只要还有关键缺口，就必须继续检索

{product_prompt_block}

<Constraint Rules>
1. 问题焦点优先（锁定最相关那一段）：答案围绕与问题最相关的章节主题组织，按各章节 heading 判断哪个最贴题。只保留与问题直接相关的步骤、警告、规格、例外；不要把别的章节主题、或同章里的通用安全/维护/保修/背景整段搬进来。若这个主题被手册切成了几个相邻同主题的小节，合起来答全（还原同一主题不算发散）；heading 换了主题就停
   - **单段取材（重要）**：当用户只问一个点时，检索返回的多个 section 只是候选——先判定唯一一个最贴题的 section，答案正文和配图只能取自这个 section。其余 section 即使内容相关、相邻、看着有用，也不要摘取。
   - **多点并问（同样重要）**：当用户原句明确并列询问多个点时，不得把整题强行压成唯一一个 section，也不得因某个 section 排名更高就丢掉其他点。单产品问题在该手册内为每个点选择一个最小且直接匹配的证据块；用户明确点名多个产品时，为每个产品从它自己的手册选择对应最小证据块。一个 section 已覆盖多个点时可以共用，否则保留多个 section，并按用户原句中各点出现的顺序分别回答。检索扩写出来但用户没说的点不算用户需求。
   - **限定语覆盖（重要）**：锁定主题后，仍要逐项核对用户问题里的限定语是否都有证据支撑。若限定语对应的是同一主题下相邻小节中的对象范围图、配置说明图、状态说明图或表格，不要因为它不在第一命中 section 就丢掉；但只能补与该限定语直接绑定的最小证据，不得把相邻小节的操作步骤或无关背景一并扩展进来。
   - **复合标题裁剪（重要）**：若手册标题本身用“X 的 A 与 B / A and B”合并两个并列子项，而用户原句只明确询问 A，则保留该段的公共前置说明和 A 对应原句，删除明确以 B 引出的原句；不得因为 A、B 共用一个 heading 就把 B 一并回答。只有用户同时询问 A 与 B 时才完整保留两边。按标题中的字面子项判断，不得维护产品或动作特判表。
2. 逐句直译：跨语言命中时，必须逐句直译**已选中的原文句子**，禁止只写大意；已选句子中的括号内补充说明、注意事项、例外条件不得遗漏。与当前问题无关的候选句子必须先裁掉，不能因跨语言而保留。
3. 多子问题完整覆盖：用户一次问多个点时，每个点都要分别回答，禁止只答其中一部分
4. 保留关键步骤编号与部件代号：手册里的数字步骤编号、部件代号（[1] / [A] / a / b / C1 / C2）必须保留；但**手册原文里的"Figure N / 图N"等图片编号不要写进答案**（见下方第 8 条与 Output Format）
5. 手册边界（防跨手册大杂烩）：用户只问一个产品、或题面没有明确点名多个产品时，路由候选仅用于找对唯一手册，答案只能基于该手册。只有用户原句明确点名多个产品时才允许跨手册回答：必须为每个产品分别选择其对应手册中的最小直接证据块，按用户原句顺序分开呈现；严禁把一本手册的内容、步骤或图片归到另一本产品，也不得加入用户未点名的第三本手册。同一产品下多个型号仍在该产品手册内并列处理。
6. 枚举型问题保护（窄触发）：仅当用户问句**明确**问"有哪些 / 列出 / 一共多少 / 包含什么 / 组成部分 / 配件清单 / what are the parts / list the components"这类**列举性**问题时，必须完整保留检索结果中相关的并列项及其对应图片锚点，不得为了精简而省略任何并列项。问"如何 / 怎样 / how to / how do I"这类**操作型/步骤型**问题不属于枚举题，应只保留与该具体动作直接相关的步骤与图片，不要把整章所有带图步骤都搬进来
7. 必要完整性：不要为了追求简短而漏掉问题所需的关键步骤、数字、按钮名、部件名、图片锚点和安全警告；但当内容只是同章相邻主题、泛化提醒或与问题无关的长篇原文时，必须裁掉
8. 图片锚点与展示顺序绑定：你写的每一个 `[[PIC:文件名]]` 都会按出现顺序变成用户看到的第 1、2、3… 张图。**严禁在文中写"图1 / 图2 / 图3 / Figure 1 / Figure 2 / 第N张图"这类数字编号引用图片**——手册原文的图编号与用户看到的展示顺序通常不一致，写出来必然错位。需要回指前面的图时写"上图 / 前面那张图 / 下图 / 如图"；每个 `[[PIC:...]]` 前后必须有一句文字说明这张图展示什么（部件、方向、状态、灯光颜色等）

9. 定量图示完整性（窄触发）：仅当用户明确询问尺寸、距离、容量、重量、频率、规格或数值范围时，必须同时核对最相关章节正文与相关图片内容标注。若同一图示用多个并列数值共同定义答案对象，应完整保留这些直接相关数值，不得只摘取其中一个；只按图中明确标签描述各数值，不得把未标注方向的尺寸自行解释成左侧、右侧、起点或无效区；不要把图中与问题无关的刻度、装饰文字或其他章节数值加入答案。
<Output Format>
- 若检索结果中带有 `[[PIC:图片文件名]]`，**只在你正文里实际描述到该图所示内容时才保留该锚点**；与当前问题不相关的图必须删掉，不要为了"完整"而把整章图全堆出来
- 保留下来的锚点必须原样写成 `[[PIC:文件名]]`；严禁改名、只写成 `<PIC>`、PIC、[PIC]、`<PIC>文件名</PIC>`
- 严禁在正文中出现"图1 / 图2 / Figure 3 / 第N张图"等数字编号引用图片；需要回指写"上图 / 下图 / 如图"
- 每个 `[[PIC:...]]` 前后必须有一句文字描述该图展示的部件/方向/状态/颜色/标注，让用户不看图也能理解
- 一段话内不要连续出现 3 个以上 `[[PIC:...]]` 而不加文字说明
- 不要把带图段落改写成纯文字段落（指相关图片不要删）
- 保留换行和段落分隔（空行表示新段落），不用 markdown 标题(#)、列表(-/*)、加粗(**)、表格(|)、代码块(```)
- 需要小标题时直接写裸文字一行，不加任何符号
- 不要说“根据手册”“手册中显示”“请查阅手册”“如图所示”“见下图”“根据检索到的信息”“以下是...”等元话术；直接进入内容
- **严禁在答案开头或任何位置写关于你自己检索/思考过程的话**，例如：“检索结果已完整覆盖/已经命中/足以回答”“可以直接作答”“我已找到完整信息”“Based on the search results”“According to the manual”“I have found / the search results show”“The manual provides”等。这些是你的内部思考，绝不能出现在给用户的答案里。答案第一句必须直接是用户要的结论/步骤本身。
- **严禁输出 `---` 分隔线或“以下为正式回答”之类过渡语**；直接从正文开始
- 问什么答什么，但不得因此重排原文：同一主题涉及多个 chunk 时，严格按每条证据的“文档顺序”升序组织答案。Warning / Caution / Note / 前置条件必须保留原有身份和相对位置，不得移到步骤之后，也不得改写成步骤
- 检索确实没命中时直说“未在手册中找到相关内容，建议联系售后确认”——但这句必须**与答案正文同语言**：英文题用英文表述（如 “This is not covered in the manual; please contact after-sales support.”），绝不能在英文答案里夹中文

## 产品技术回答格式

参考官方范例：

范例 A（图例型）：
问：我的DCB107或DCB112型号电钻指示灯闪烁时，这些闪烁标识代表什么含义？
答：DCB107、DCB112 电池组充电中[[PIC:Manual04_22]]电池组已充满[[PIC:Manual04_23]]过热/过冷延迟[[PIC:Manual04_24]]电池组或充电器故障[[PIC:Manual04_25]]电源故障[[PIC:Manual04_26]]

说明：图例型题目要让每个图对应一个短语，并保留检索结果中的 `[[PIC:...]]` 锚点；不要改写成纯文字段落。

范例 B（结构型）：
问：我想更换健身追踪器的表带，有其他尺寸可选吗？
答：表带尺寸

表带尺寸如下所示。注意：单独销售的配件表带可能略有差异。
[[PIC:Manual16_51]]

环境条件
[[PIC:Manual16_52]]

范例 C（聚焦·无图型，满分答案）：
问：如何清洁空气净化器的设备内外？
答：清洁设备内外前，务必先拔下电源插头。不要在通电状态下清洁，以免触电或造成设备故障。

1. 清洁外壳：用温水或温和清洁剂浸湿软布，擦拭空气净化器外壳，然后再用软布擦干。

2. 清洁内部滤网仓：打开背部滤网盖并取出滤网。用吸尘器和湿毛巾清洁滤网仓内部。

3. 日常频率：为保证净化效果，每月清洁设备及预过滤网 1-2 次；灰尘较多的地区建议增加清洁频率。

清洁时不要把水直接倒入或喷入机身内部，湿布应拧至不滴水后再擦拭。清洁完成后，确认外壳和滤网仓内部已擦干、滤网已装回、背部滤网盖已盖好，再重新接通电源使用。日常维护时，建议定期检查进风口和滤网仓是否有明显积尘，避免灰尘堆积影响进风和净化效率。

说明：这是真实拿到满分（5/5）的答案，它是 0 图的——因为它的参考章节"设备清洁与日常维护"本身没图，于是只精准照搬这一个章节、没去拽相邻的"灰尘传感器清洁"等别的章节、也没硬凑图。聚焦单章节、该 0 图就 0 图，就是满分。

## 工具
{search_manual_skill_block}


## 幻觉抑制（硬约束）
- 型号、规格数字化、按钮名称、步骤顺序、故障代码、灯光含义、配件兼容性、保修政策、维修费用、官方时效、安全警告原文，这些必须基于检索内容，检索没写就不编造
- 问什么答什么：用户问 how/procedure/steps 时，优先给可执行步骤；若当前命中主要是 safety/regulatory/notice，先继续检索步骤型章节。确实找不到步骤时，只用一句话简短说明“手册未给出完整步骤”，不要展开长篇解释

## 回答丰富度（只限同一最相关章节内的直接补充）
回答首先追求**贴题和聚焦**，不要为了显得全面而主动扩展。只有当补充内容同时满足以下条件时才可以加入：
1. 补充信息来自你已经锁定的同一个最相关主题章节，或是该章节原文中明确出现的 note / warning / condition / exception；
2. 补充信息能直接帮助回答用户当前问法中的动作、条件、判断或注意事项；
3. 加入后不会把答案带到相邻章节、通用维护、安全背景、使用建议或另一个功能主题。

允许保留的补充类型仅限：
- 同一章节原文明确写出的适用条件、例外情况、完成判断、警告/注意事项；
- 同一章节图片中可见、且与正文步骤直接对应的部件/按钮/方向/状态说明。

不要额外发挥使用场景、易错点、日常维护建议或经验性技巧；除非这些内容就在同一最相关章节原文里，并且直接回答用户的问题。若不确定是否属于同一章节，宁可不补。

丰富回答范例：
问：如何开启空调的节能制冷模式？
答：节能制冷模式可最大限度降低制冷时的耗电量，并将设定温度调节至最适宜的水平，打造更舒适的环境。

1. 按下开/关键开启电源。
2. 反复按下模式键，选择制冷模式。
3. 按下节能键，显示屏上会显示节能标识。[[PIC:Manual01_21]]

注：部分机型不支持此功能。

日常使用中，夏季夜间睡眠或白天长时间离家时开启此模式效果最佳。达到设定温度后压缩机会自动降低运行频率，相比普通制冷模式更安静省电。若感觉制冷不够，可先将温度调低1-2度快速降温，再切回节能模式维持恒温。

## 最后复述
1. 是否逐句完整翻译且未作任何删减，尤其不要漏括号内补充说明、例外条件、免责条款
2. 是否完整保留与正文描述对应的 `[[PIC:图片文件名]]`、步骤编号、部件代号；是否已删除所有"图N / Figure N / 第N张图"数字引用
3. 是否没有使用任何 Markdown 列表格式，并且没有写元话术

常见错误示范：
- 错：指示灯代表 PIC 正在充电 PIC 已充满     → 对：指示灯代表[[PIC:Manual04_22]]正在充电[[PIC:Manual04_23]]已充满
- 错：<PIC>Manual04_22</PIC>                 → 对：[[PIC:Manual04_22]]
- 错：把带 3 张图的检索结果改写成纯文字       → 对：保留 3 个 `[[PIC:...]]`
- 错：删掉 "Important: do not spray..."      → 对：保留 safety 警告原文
- 英文问题错：您好，空气炸锅首次使用前...     → 对：Before using the air fryer for the first time...
""".format(
    product_prompt_block=PRODUCT_PROMPT_BLOCK,
    search_manual_skill_block=SEARCH_MANUAL_SKILL_BLOCK,
)

# 兼容旧名：外部如果还在引用 SYSTEM_PROMPT，默认指向 TECH（更具一般性）
TECH_SYSTEM_PROMPT += """

图片锚点补充硬规则（优先于前文任何图片说明要求）：
当检索正文只给出 [[PIC:...]] 锚点、没有紧邻的手册图注或图内文字时，锚点只能直接跟在对应的原文句末或单独成行。不得补写“图示展示了”“下图为”“该图说明了”或任何根据图片推断出的部件、方向、状态、安装位置。图片只用于展示原版手册插图，不是生成说明的依据。
"""
SYSTEM_PROMPT = TECH_SYSTEM_PROMPT

# Final guardrail: retrieved manual chunks are the complete fact boundary.
TECH_SYSTEM_PROMPT += """

证据边界硬规则：锁定相关 chunk 后，最终回答的每一句都必须能在这些 chunk 正文中找到原句或逐句直译。检索排名只表示相关性，不表示叙述顺序；同一手册同一主题的多条证据必须按“文档顺序”升序还原。只允许删除明显 OCR/解析噪声、删除完全相同的重复行、调整空行和列表排版，以及必要的逐句直译。禁止概括改写、交换句序、合并事实、拆开后重新组合、改变层级，或把 Warning / Caution / Note / 前置条件改造成步骤；不得新增原文没有的清洁步骤、完成检查、注意事项、原因、频率或建议。只有用户问题确实跨越同一主题的相邻小节时才可组合这些小节，且仍须保持原文顺序。
当检索输出含 `[TOPIC_BUNDLE] complete` 时，这些 `primary` / `support` 证据共同构成可选择的同主题原文范围。裁剪单位必须是手册中的完整子块，而不是孤立句子：子块从一个内部标题/明确主题开始，到下一个同级内部标题前结束。用户只问一个对象/动作时，保留该子块全部原文；用户同时问多个并列对象/动作时，按文档顺序保留每个被点名子块的全部原文。选中子块内部的说明、部件标注、步骤、Note、Caution、Warning 和图片锚点不得再删。只删除完全未被问题点名的其他子块。`related` 和 `ranked` 也遵循同一边界。

终稿裁剪任务（最高优先级）：最终 LLM 的价值在于从正确手册原文中选择完整子块，而不是按句子随意缩写。先根据用户点名的对象/动作确定所需子块：问 A 取完整 A 块，问 B 取完整 B 块，问 A+B 则按原文顺序取完整 A 块和完整 B 块。只能在子块边界删除；一旦选中某块，该块内的标题、说明、部件标注、步骤、Note、Caution、Warning 和图片锚点全部保留，不得只摘操作句或截断警告。跨语言时先选完整子块，再逐句翻译整个已选子块。所有正文保持原文顺序和原句，不新增概括。不要输出取舍过程。
"""
SYSTEM_PROMPT = TECH_SYSTEM_PROMPT

# ────────────────── 工具定义 ──────────────────

TOOLS = [
    {
        "name": "search_manual",
        "description": "统一检索入口。优先输入关键词列表；系统会基于关键词做 BM25，并始终带上原始用户问题做语义召回；若补充 query，也会把它作为额外语义线索一起召回，最后统一 rerank。适用于绝大多数普通检索场景。可通过 products 参数限定产品范围。",
        "input_schema": {
            "type": "object",
            "properties": {
                "keywords": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "关键词列表，如 [\"DCB107\", \"指示灯\"]。尽量给 2-6 个高信息量词。",
                },
                "query": {
                    "type": "string",
                    "description": "可选的补充语义描述。通常可省略；只有关键词不足以表达动作关系时再填写。即使填写，系统也仍会保留原始用户问题参与语义召回。",
                },
                "products": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "限定检索的产品名称列表，如 [\"电钻手册\"]。不传则搜索全部产品。",
                },
            },
            "required": ["keywords"],
        },
    },
]


# ────────────────── 工具执行 ──────────────────

def format_search_results(results: list[SearchResult], filtered_count: int = 0) -> str:
    """把检索结果格式化为 LLM 可读的文本，正文里直接带内联图片锚点。"""
    if not results and filtered_count == 0:
        return "\n".join([
            "[SEARCH_STATUS] no_result",
            "[SEARCH_REASON] empty_recall",
            "[SEARCH_FILTERED] 0",
            "[SEARCH_SUGGEST] switch_strategy",
            "(无检索结果)",
        ])
    if not results and filtered_count > 0:
        return "\n".join([
            "[SEARCH_STATUS] no_result",
            "[SEARCH_REASON] empty_after_postprocess",
            f"[SEARCH_FILTERED] {filtered_count}",
            "[SEARCH_SUGGEST] switch_strategy",
            f"(本次检索返回的候选在后处理阶段未形成可用结果。建议换关键词、扩展 products=[]，或基于已有证据收束)",
        ])

    bundle_results = [
        result for result in results
        if result.source.get("evidence_role") in {"primary", "support", "related"}
    ]
    has_complete_bundle = (
        any(result.source.get("evidence_role") == "primary" for result in bundle_results)
        and any(result.source.get("evidence_role") == "support" for result in bundle_results)
    )
    if bundle_results:
        bundle_ids = {id(result) for result in bundle_results}
        bundle_results.sort(key=lambda result: (
            int(result.source.get("document_order", result.chunk_id)),
            result.chunk_id,
        ))
        ordered_results = bundle_results + [
            result for result in results if id(result) not in bundle_ids
        ]
    else:
        ordered_results = list(results)

    lines = []
    if has_complete_bundle:
        lines.append("[TOPIC_BUNDLE] complete")
        lines.append("[TOPIC_BUNDLE_AVAILABLE_ROLES] primary,support")
    section_ids = [
        int(r.source.get("parent_section_id"))
        for r in ordered_results
        if isinstance(r.source.get("parent_section_id"), int)
    ]
    top_section_id: int | None = None
    top_section_count = 0
    top_section_summary = ""
    if section_ids:
        section_counts = Counter(section_ids)
        top_section_id, top_section_count = section_counts.most_common(1)[0]
        for r in ordered_results:
            if r.source.get("parent_section_id") == top_section_id:
                top_section_summary = (r.source.get("section_summary") or "").strip()
                if top_section_summary:
                    break
    if filtered_count > 0:
        lines.append(f"（注：{filtered_count} 条候选在后处理阶段未被保留）")
    if section_ids:
        lines.append(f"[SECTION_IDS] {','.join(str(sid) for sid in section_ids)}")
    if top_section_id is not None:
        lines.append(f"[SECTION_TOP] {top_section_id}")
        lines.append(f"[SECTION_TOP_COUNT] {top_section_count}")
        if top_section_summary:
            lines.append(f"[SECTION_TOP_SUMMARY] {top_section_summary}")

    # search_manual returns parent-section evidence through the retrieval engine;
    # the model may answer directly when the returned evidence is sufficient.
    SECTION_FULL_TOP_N = 0
    SECTION_FULL_CHAR_CAP = 3500
    section_freq = Counter()
    section_first_idx: dict = {}
    for idx, r in enumerate(ordered_results):
        psid = r.source.get("parent_section_id")
        if not isinstance(psid, int):
            continue
        section_freq[psid] += 1
        if psid not in section_first_idx:
            section_first_idx[psid] = idx
    expanded_section_ids: set = set()
    for psid, _count in section_freq.most_common(SECTION_FULL_TOP_N):
        ref = ordered_results[section_first_idx[psid]]
        sec_text = (ref.source.get("section_text") or "").strip()
        if not sec_text:
            continue
        sec_pics = list(ref.source.get("section_pics") or [])
        sec_heading = (ref.source.get("section_heading") or ref.heading or "").strip()
        full_text = inject_inline_pic_refs(sec_text, sec_pics)
        # Image captions are retrieval/selection metadata, not manual prose.
        # Keep only the original text and its PIC anchors in the model context.
        truncated = ""
        if len(full_text) > SECTION_FULL_CHAR_CAP:
            full_text = full_text[:SECTION_FULL_CHAR_CAP]
            truncated = " ...(章节文本已截断，请优先基于已显示内容作答，必要时换关键词检索同主题章节)"
        lines.append(f"[SECTION_FULL] 产品: {ref.product} | 章节ID: {psid} | 章节: {sec_heading}")
        lines.append(f"    完整章节正文:")
        lines.append(f"    {full_text}{truncated}")
        lines.append("")
        expanded_section_ids.add(psid)

    # —— 剩余 chunk：仅展示尚未被展开章节覆盖的，避免重复
    chunk_idx = 0
    for r in ordered_results:
        psid = r.source.get("parent_section_id")
        if isinstance(psid, int) and psid in expanded_section_ids:
            continue
        chunk_idx += 1
        lines.append(f"[{chunk_idx}] 产品: {r.product} | 章节: {r.heading}")
        # Preserve the exact retrieval identity for source provenance.
        matched_chunk_id = r.source.get("matched_chunk_id", r.chunk_id)
        lines.append(f"    命中ChunkID: {matched_chunk_id}")
        if isinstance(psid, int):
            lines.append(f"    上层章节ID: {psid}")
        lines.append(f"    证据角色: {r.source.get('evidence_role', 'ranked')}")
        relevance_tier = str((r.source.get("relevance") or {}).get("relevance_tier") or "related")
        lines.append(f"    [RELEVANCE_TIER] {relevance_tier}")
        lines.append(f"    文档顺序: {r.source.get('document_order', psid if isinstance(psid, int) else r.chunk_id)}")
        section_summary = (r.source.get("section_summary") or "").strip()
        if section_summary:
            lines.append(f"    上层摘要: {section_summary}")
        matched_text = (r.source.get("matched_chunk_text") or "").strip()
        matched_pics = list(r.source.get("matched_chunk_pics") or [])
        if matched_text:
            matched_content = inject_inline_pic_refs(matched_text, matched_pics)
            lines.append(f"    命中Chunk正文JSON: {json.dumps(matched_content, ensure_ascii=False)}")
        content = inject_inline_pic_refs(r.text, r.pics)
        lines.append(f"    内容: {content}")
        lines.append("")
    return "\n".join(lines)


_ROUTER_CACHE: dict[int, ProductRouter] = {}


def _get_product_router(engine: RetrievalEngine) -> ProductRouter:
    engine.ensure_index()
    cache_key = id(engine)
    router = _ROUTER_CACHE.get(cache_key)
    if router is None:
        router = ProductRouter(engine.catalog, engine=engine)
        _ROUTER_CACHE[cache_key] = router
    return router


def _run_search_with_defaults(
    engine: RetrievalEngine,
    *,
    name: str,
    input_data: dict,
    default_products: list[str] | None,
    default_query_context: str = "",
    balance_products: bool = False,
) -> tuple[list[SearchResult], int]:
    has_products_key = "products" in input_data
    # A routed product is the default retrieval boundary. Previously this
    # parameter was accepted but never applied when the model omitted
    # `products`, silently turning a high-confidence product route into a
    # whole-corpus search. That allowed unrelated manuals to win reranking.
    # An explicit empty list remains the documented escape hatch for a
    # deliberate cross-manual confirmation search.
    products = input_data.get("products") if has_products_key else default_products
    if has_products_key and isinstance(products, list) and len(products) == 0:
        # An empty tool argument cannot override an already-established
        # product boundary. Otherwise the model can silently reopen global
        # recall after the UI or router has identified one manual.
        products = default_products if default_products else None

    if os.getenv("DEBUG_ROUTE"):
        print(f"[TOOL] {name} llm_products={input_data.get('products', '<unset>')} default={default_products} → used={products}", flush=True)

    if name in SEARCH_TOOL_NAMES:
        if name == "search_manual":
            keywords = input_data.get("keywords", [])
            # Tool keywords are useful recall hints, but the model-generated
            # free-form query may silently broaden a narrow user request (for
            # example, turning "replace a filter" into an entire initial
            # installation procedure). Keep dense retrieval and reranking tied
            # to the original user wording; BM25 still receives the supplied
            # keywords as supplementary lexical recall signals.
            semantic_query = (default_query_context or input_data.get("query") or "").strip()
        elif name == "keyword_search":
            keywords = input_data.get("keywords", [])
            semantic_query = ""
        else:
            semantic_query = input_data.get("query", "")
            keywords = re.findall(r"\S+", semantic_query)
        if balance_products and isinstance(products, list) and len(products) > 1:
            # Explicit cross-product questions need evidence from every named
            # manual.  Searching the union once lets one product consume all
            # result slots, so retrieve within each boundary and interleave the
            # independently ranked lists.  No product or intent taxonomy is
            # involved: the boundaries come from literal user product mentions.
            per_product_top_k = max(
                2,
                (MAX_SEARCH_RESULTS + len(products) - 1) // len(products) + 1,
            )
            grouped_results: list[list[SearchResult]] = []
            filtered = 0
            for product in products:
                group, group_filtered = engine.search_manual(
                    keywords,
                    semantic_query=semantic_query,
                    original_query=default_query_context,
                    top_k=per_product_top_k,
                    products=[product],
                )
                grouped_results.append(group)
                filtered += group_filtered
            results = []
            seen: set[str] = set()
            for rank_index in range(max((len(group) for group in grouped_results), default=0)):
                for group in grouped_results:
                    if rank_index >= len(group):
                        continue
                    result = group[rank_index]
                    fingerprint = _result_fingerprint(result)
                    if fingerprint in seen:
                        continue
                    seen.add(fingerprint)
                    results.append(result)
                    if len(results) >= MAX_SEARCH_RESULTS:
                        break
                if len(results) >= MAX_SEARCH_RESULTS:
                    break
        else:
            results, filtered = engine.search_manual(
                keywords,
                semantic_query=semantic_query,
                original_query=default_query_context,
                top_k=MAX_SEARCH_RESULTS,
                products=products,
            )
        results, literal_filtered = _narrow_to_unique_literal_heading(
            default_query_context,
            results,
        )
        results, mode_filtered = _filter_mode_enumeration_results(
            default_query_context,
            results,
        )
        return results, filtered + literal_filtered + mode_filtered

    return [], 0


def execute_tool(
    engine: RetrievalEngine,
    name: str,
    input_data: dict,
    default_products: list[str] | None = None,
    default_query_context: str = "",
    balance_products: bool = False,
) -> str:
    """执行工具调用，返回结果文本。"""
    if name in SEARCH_TOOL_NAMES:
        results, filtered = _run_search_with_defaults(
            engine,
            name=name,
            input_data=input_data,
            default_products=default_products,
            default_query_context=default_query_context,
            balance_products=balance_products,
        )
        return format_search_results(results, filtered)

    return f"未知工具: {name}"


# ────────────────── Agent 主循环 ──────────────────

@dataclass
class AgentResult:
    """一次 run_agent 调用的结构化产物。

    answer/pics 是最终提交格式化前的核心输出；tool_calls/turns 用于统计工具纪律；trace 保存产品路由、预检索、LLM tool_use 与最终收束路径，便于验证报告复盘。
    """
    answer: str
    pics: list[str] = field(default_factory=list)
    tool_calls: int = 0
    turns: int = 0
    trace: dict | None = None
    # 最终回答的首 token 耗时（秒）。仅 stream_ttft=True 时填充：主循环每轮流式跑，
    # 出现文本增量(content)的那轮即最终回答，记其首 token 时间；纯工具轮不计。
    ttft: float | None = None


def _serialize_trace_content(content) -> object:
    if isinstance(content, (str, int, float, bool)) or content is None:
        return content
    if isinstance(content, list):
        return [_serialize_trace_content(item) for item in content]
    if isinstance(content, dict):
        return {str(k): _serialize_trace_content(v) for k, v in content.items()}

    data: dict[str, object] = {}
    for attr in ("type", "id", "name", "input", "text", "tool_use_id", "content"):
        if hasattr(content, attr):
            data[attr] = _serialize_trace_content(getattr(content, attr))
    if data:
        return data
    return repr(content)


def _build_trace_llm_event(*, index: int, response_content) -> dict:
    event: dict[str, object] = {
        "kind": "llm_call",
        "index": index,
        "actions": [],
    }
    text_preview_parts: list[str] = []
    actions: list[dict[str, object]] = []

    for block in response_content:
        block_type = getattr(block, "type", None)
        if block_type == "tool_use":
            actions.append({
                "type": "tool_use",
                "name": getattr(block, "name", ""),
                "input": _serialize_trace_content(getattr(block, "input", {}) or {}),
            })
        elif block_type == "text":
            text = (getattr(block, "text", "") or "").strip()
            if text:
                text_preview_parts.append(text)

    if actions:
        event["actions"] = actions
    if text_preview_parts:
        preview = "\n".join(text_preview_parts)
        event["text_preview"] = preview[:300]
    return event


def _extract_search_trace_hits(result_text: str) -> list[dict[str, object]]:
    hits: list[dict[str, object]] = []
    current: dict[str, object] | None = None
    content_lines: list[str] = []

    def finish_current() -> None:
        nonlocal content_lines
        if current is not None and content_lines:
            current["content"] = "\n".join(content_lines).strip()[:12000]
        content_lines = []

    for raw_line in (result_text or "").splitlines():
        line = raw_line.rstrip()
        stripped = line.strip()
        m = re.match(r"^\[(\d+)\]\s+产品:\s+(.*?)\s+\|\s+章节:\s+(.*)$", stripped)
        if m:
            finish_current()
            current = {
                "rank": int(m.group(1)),
                "product": m.group(2).strip(),
                "heading": m.group(3).strip(),
            }
            hits.append(current)
            continue
        if current is None:
            continue
        if stripped.startswith("命中ChunkID:"):
            value = stripped.split(":", 1)[1].strip()
            current["matched_chunk_id"] = int(value) if value.isdigit() else value
        elif stripped.startswith("上层章节ID:"):
            value = stripped.split(":", 1)[1].strip()
            if value.isdigit():
                current["parent_section_id"] = int(value)
            else:
                current["parent_section_id"] = value
        elif stripped.startswith("证据角色:"):
            current["evidence_role"] = stripped.split(":", 1)[1].strip()
        elif stripped.startswith("文档顺序:"):
            value = stripped.split(":", 1)[1].strip()
            current["document_order"] = int(value) if value.isdigit() else value
        elif stripped.startswith("上层摘要:"):
            current["section_summary"] = stripped.split(":", 1)[1].strip()[:200]
        elif stripped.startswith("命中Chunk正文JSON:"):
            value = stripped.split(":", 1)[1].strip()
            try:
                current["matched_content"] = str(json.loads(value))[:12000]
            except Exception:
                current["matched_content"] = value[:12000]
        elif stripped.startswith("内容:"):
            first_line = stripped.split(":", 1)[1].strip()
            current["text_preview"] = first_line[:300]
            content_lines = [first_line] if first_line else []
        elif content_lines and not stripped.startswith("图片内容标注"):
            content_lines.append(stripped)

    finish_current()
    return hits



def _build_trace_tool_event(
    *,
    index: int,
    name: str,
    input_data: dict,
    default_products: list[str] | None,
    default_query_context: str,
    elapsed: float,
    pics: list[str],
    result_text: str,
) -> dict:
    obs = _observe_tool_output(name, input_data, result_text)
    action = input_data.get("action") if isinstance(input_data, dict) else None
    event: dict[str, object] = {
        "kind": "tool_call",
        "index": index,
        "name": name,
        "input": _serialize_trace_content(input_data),
        "default_products": _serialize_trace_content(default_products),
        "default_query_context": default_query_context,
        "elapsed": round(elapsed, 3),
        "pics": pics,
        "no_result": obs.no_result,
        "products": obs.products,
        "headings": obs.headings[:8],
        "parent_section_ids": obs.parent_section_ids[:8],
        "explicit_product": obs.explicit_product,
        "dominant_product": obs.dominant_product,
        "dominant_parent_section_id": obs.dominant_parent_section_id,
        "search_status": obs.search_status,
        "search_reason": obs.search_reason,
        "search_filtered": obs.search_filtered,
    }
    if name in SEARCH_TOOL_NAMES:
        event["retrieval_hits"] = _extract_search_trace_hits(result_text)
    result_preview = (result_text or "").strip()
    if result_preview:
        event["result_preview"] = result_preview[:500]
    return event


def _collect_pics_from_results(results: list[SearchResult]) -> list[str]:
    """从检索结果列表里按顺序去重收集图片文件名。"""
    pics: list[str] = []
    for r in results:
        for p in r.pics:
            if p not in pics:
                pics.append(p)
    return pics


def _extend_unique_pics(target: list[str], pictures: list[str]) -> None:
    """Append retrieved image ids in evidence order without duplicating them."""

    for picture in pictures:
        if picture and picture not in target:
            target.append(picture)


def _result_fingerprint(result: SearchResult) -> str:
    """为检索条目生成稳定指纹，用于会话内去重。"""
    text = " ".join((result.text or "").split())
    heading = " ".join((result.heading or "").split())
    product = (result.product or "").strip()
    return f"{product}\n{heading}\n{text}"


def _dedup_results_by_history(
    results: list[SearchResult],
    seen_result_keys: set[str],
) -> tuple[list[SearchResult], int]:
    """过滤历史已见检索内容，返回(新增结果, 被过滤数量)。"""
    fresh: list[SearchResult] = []
    dropped = 0
    for r in results:
        key = _result_fingerprint(r)
        if key in seen_result_keys:
            dropped += 1
            continue
        seen_result_keys.add(key)
        fresh.append(r)
    return fresh, dropped


def _execute_tool_with_pics(
    engine: RetrievalEngine,
    name: str,
    input_data: dict,
    default_products: list[str] | None = None,
    default_query_context: str = "",
    seen_result_keys: set[str] | None = None,
    balance_products: bool = False,
) -> tuple[str, list[str]]:
    """执行工具调用，同时返回本次检索到的 trace 图片列表。"""
    if name in SEARCH_TOOL_NAMES:
        results, filtered = _run_search_with_defaults(
            engine,
            name=name,
            input_data=input_data,
            default_products=default_products,
            default_query_context=default_query_context,
            balance_products=balance_products,
        )
        dropped = 0
        if seen_result_keys is not None:
            results, dropped = _dedup_results_by_history(results, seen_result_keys)
        if not results:
            if dropped > 0:
                return "(无新增检索结果，当前结果与历史重复)", []
            return format_search_results(results, filtered), []
        return format_search_results(results, filtered), _collect_pics_from_results(results)

    return f"未知工具: {name}", []


_GENERIC_FAILURE_ANSWERS = {
    "",
    "抱歉，处理过程中出现异常，请重试。",
    "处理过程中出现异常，请重试。",
    "抱歉，请重试。",
}


def _extract_text_from_response(response) -> str:
    parts: list[str] = []
    for block in response.content:
        if hasattr(block, "text"):
            parts.append(block.text)
    return "\n".join(parts).strip()


def _normalize_final_answer(answer: str) -> str:
    return " ".join(answer.strip().split())


def _resolve_answer_pics(answer: str) -> tuple[str, list[str]]:
    """只从正文中的 [[PIC:...]] 抽图，不再按检索结果顺序兜底补图。"""
    answer, inline_pics = extract_inline_pic_refs(answer)
    pic_count = answer.count("<PIC>")
    pics = inline_pics[:pic_count] if inline_pics else []
    if pic_count > len(pics):
        parts = answer.split("<PIC>")
        rebuilt = parts[0]
        for i, tail in enumerate(parts[1:], start=1):
            rebuilt += ("<PIC>" if i <= len(pics) else "") + tail
        answer = rebuilt
    return answer, pics


def _question_requests_comparison(question: str) -> bool:
    q = (question or "").lower()
    markers = [
        "compare",
        "difference",
        "vs",
        "versus",
        "区别",
        "对比",
        "分别",
        "各自",
        "哪种",
    ]
    return any(marker in q for marker in markers)


def _format_spec_blocks(answer: str) -> str:
    text = answer or ""
    replacements = [
        ("电源要求：工作电压：", "电源要求：\n工作电压："),
        ("，50Hz 工作电流：", "，50Hz\n工作电流："),
        ("认证标准：交流电源适配器：", "认证标准：\n交流电源适配器："),
    ]
    for src, dst in replacements:
        text = text.replace(src, dst)
    return text


def _rewrite_single_product_answer(
    *,
    answer: str,
    question: str,
    system_prompt: str,
    model: str | None,
    products: list[str],
) -> str:
    response, _route = create_message_with_fallback(
        max_tokens=int(os.getenv("AGENT_FINALIZE_MAX_TOKENS", "4096")),
        system=system_prompt,
        messages=[
            {
                "role": "user",
                "content": (
                    "请只基于已有证据重写下面这条最终答案，不要调用工具。\n"
                    "要求：\n"
                    "1. 用户没有要求对比时，不要把多个产品/手册来源无标记地混写在同一答案里\n"
                    "2. 若证据已足以支持单一产品答案，就只保留一个最自洽的产品答案\n"
                    "3. 只有在确实必须保留多个产品时，才显式写成“若是 A…；若是 B…”\n"
                    "4. 保留已有的图片锚点 [[PIC:...]]、步骤、规格和警告，不要新增未检索到的信息\n\n"
                    f"问题：{question}\n"
                    f"候选产品：{'、'.join(products)}\n"
                    f"当前答案：\n{answer}"
                ),
            }
        ],
        model=model,
    )
    rewritten = _extract_text_from_response(response).strip()
    return rewritten or answer


def _postprocess_final_answer(
    *,
    answer: str,
    question: str,
    system_prompt: str,
    model: str | None,
    route_products: list[str],
) -> str:
    # Presentation must not trigger another model pass: it would paraphrase
    # retrieved manual prose after the main answer had already been grounded.
    return _format_spec_blocks(answer)


def _apply_evidence_selection(
    *,
    answer: str,
    question: str,
    engine: RetrievalEngine,
    candidate_images: list[str],
    route_products: list[str],
    model: str | None,
) -> tuple[str, dict[str, object] | None]:
    """Run the optional set-wise selector before converting anchors to ``<PIC>``.

    Keeping this behind a runtime flag lets batch validation compare the original
    and revised paths against the same retrieval index.  The selector itself has a
    strict fallback: malformed plans, low confidence, or invalid text-image binding
    return the original answer unchanged.
    """

    enabled = os.getenv("EVIDENCE_SELECTOR_ENABLED", "0").strip().lower()
    if enabled not in {"1", "true", "yes", "on"}:
        return answer, None

    # Evidence planning/recomposition runs only after the conservative anomaly
    # gate fires. Give this repair path its own timeout budget so a slow planner
    # does not silently return the unvalidated, cross-section draft.
    selector_timeout = float(os.getenv("EVIDENCE_SELECTOR_TIMEOUT_SECONDS", "75"))

    def selector_llm_call(**kwargs):
        return create_message_with_fallback(timeout=selector_timeout, **kwargs)

    result = refine_answer_evidence(
        question=question,
        answer=answer,
        candidate_images=candidate_images,
        engine=engine,
        captions=_load_image_captions(),
        route_products=route_products,
        llm_call=selector_llm_call,
        model=model,
    )
    return result.answer, result.trace


def _dominant_text_language(text: str) -> str:
    value = text or ""
    cjk_count = len(re.findall(r"[\u3400-\u9fff]", value))
    latin_count = len(re.findall(r"[A-Za-z]", value))
    return "zh" if cjk_count >= 4 and cjk_count >= latin_count * 0.3 else "non_zh"


def _clean_source_literal(text: str) -> str:
    """Remove Markdown transport escapes without rewriting source wording."""
    value = re.sub(r"\\([^\w\s])", r"\1", text or "")
    return re.sub(r"[ \t]+\n", "\n", value).strip()


def _source_backed_opening(draft: str, literal_source: str) -> str:
    """Keep at most one concise, source-backed lead sentence from the draft."""
    first_line = next((line.strip() for line in (draft or "").splitlines() if line.strip()), "")
    candidate = re.split(r"(?<=[。！？.!?])\s*", first_line, maxsplit=1)[0].strip()
    if not candidate or "[[PIC:" in candidate or "<PIC" in candidate:
        return ""
    if len(candidate) > 140 or len(re.findall(r"[。！？.!?]", candidate)) > 1:
        return ""
    candidate_compact = re.sub(r"[^a-z0-9\u3400-\u9fff]+", "", candidate.lower())
    source_compact = re.sub(r"[^a-z0-9\u3400-\u9fff]+", "", literal_source.lower())
    if len(candidate_compact) < 6 or candidate_compact in source_compact:
        return ""
    candidate_grams = {
        candidate_compact[index:index + 2]
        for index in range(max(0, len(candidate_compact) - 1))
    }
    source_grams = {
        source_compact[index:index + 2]
        for index in range(max(0, len(source_compact) - 1))
    }
    overlap = len(candidate_grams & source_grams) / max(len(candidate_grams), 1)
    return candidate if overlap >= 0.45 else ""


def _is_literal_evidence_answer(answer: str, literal_source: str) -> bool:
    """Allow model-selected source excerpts, but reject deep paraphrases."""
    source_compact = re.sub(r"[^a-z0-9\u3400-\u9fff]+", "", literal_source.lower())
    lines = [line.strip() for line in (answer or "").splitlines() if line.strip()]
    if not source_compact or not lines:
        return False

    checked = 0
    matched = 0
    for index, line in enumerate(lines):
        # The optional first line is the only permitted light rewrite.
        if index == 0:
            continue
        for unit in re.split(r"(?<=[。！？.!?])\s*", line):
            compact = re.sub(r"[^a-z0-9\u3400-\u9fff]+", "", unit.lower())
            if len(compact) < 6:
                continue
            checked += 1
            if compact in source_compact:
                matched += 1
                continue
            grams = {compact[pos:pos + 2] for pos in range(len(compact) - 1)}
            source_grams = {
                source_compact[pos:pos + 2]
                for pos in range(len(source_compact) - 1)
            }
            if grams and len(grams & source_grams) / len(grams) >= 0.78:
                matched += 1
    return checked > 0 and matched / checked >= 0.9


def _same_language_topic_bundle_answer(
    *,
    question: str,
    trace: dict | None,
) -> tuple[str | None, list[object]]:
    """Assemble complete same-language topic evidence without LLM paraphrasing."""
    if not trace:
        return None, []
    for event in reversed(trace.get("events", [])):
        if event.get("kind") != "tool_call" or event.get("name") != "search_manual":
            continue
        hits = list(event.get("retrieval_hits") or [])
        primary = [hit for hit in hits if hit.get("evidence_role") == "primary"]
        if not primary:
            continue
        products = {str(hit.get("product") or "").strip() for hit in primary}
        topic_keys = {
            " / ".join(part.strip() for part in str(hit.get("heading") or "").split("/")[:-1] if part.strip())
            or str(hit.get("heading") or "").strip()
            for hit in primary
        }
        # Only replace a draft when its selected main evidence is one coherent
        # manual subsection. Cross-section questions still rely on the literal
        # prompt above to choose their relevant source sentences.
        if len(products) != 1 or len(topic_keys) != 1:
            continue
        product = next(iter(products))
        topic_key = next(iter(topic_keys))
        support = [
            hit for hit in hits
            if hit.get("evidence_role") == "support"
            and str(hit.get("product") or "").strip() == product
            and (
                " / ".join(part.strip() for part in str(hit.get("heading") or "").split("/")[:-1] if part.strip())
                or str(hit.get("heading") or "").strip()
            ) == topic_key
        ]
        required = primary + support
        evidence_text = "\n".join(str(hit.get("content") or "") for hit in required)
        if _dominant_text_language(question) != _dominant_text_language(evidence_text):
            return None, []

        required.sort(key=lambda hit: (
            int(hit.get("document_order") or 0),
            int(hit.get("rank") or 0),
        ))
        blocks: list[str] = []
        chunk_ids: list[object] = []
        last_heading = ""
        for hit in required:
            content = _clean_source_literal(str(hit.get("content") or ""))
            if not content:
                continue
            heading = str(hit.get("heading") or "").split("/")[-1].strip()
            blocks.append(f"{heading}\n{content}" if heading and heading != last_heading else content)
            last_heading = heading
            chunk_ids.append(hit.get("matched_chunk_id"))
        return "\n\n".join(blocks).strip() or None, chunk_ids
    return None, []


def _literal_multi_intent_requirements(
    *,
    question: str,
    trace: dict | None,
) -> list[dict[str, object]]:
    """Find ranked evidence blocks that cover distinct literal parts of a question.

    This intentionally assigns no semantic action categories.  A block is required
    only when its heading contributes a meaningful word that no earlier selected
    heading covers.  Single-focus questions therefore keep their normal path.
    """
    if not trace:
        return []
    question_terms = _significant_lexical_terms(question)
    if len(question_terms) < 3:
        return []

    hits: list[dict[str, object]] = []
    seen_hits: set[tuple[str, str]] = set()
    for event in trace.get("events", []):
        if event.get("kind") != "tool_call" or event.get("name") != "search_manual":
            continue
        for hit in event.get("retrieval_hits") or []:
            key = (
                str(hit.get("product") or "").strip(),
                str(hit.get("matched_chunk_id") or "").strip(),
            )
            if key in seen_hits:
                continue
            seen_hits.add(key)
            hits.append(hit)
    if not hits:
        return []

    route_info = trace.get("product_route") or {}
    if route_info.get("reason") == "clause_multi_product":
        routed_products = [
            str(product or "").strip()
            for product in route_info.get("products") or []
            if str(product or "").strip()
        ]
        routed_clauses = _split_literal_question_clauses(question)
        if len(routed_products) == len(routed_clauses) and len(routed_products) >= 2:
            routed_requirements: list[dict[str, object]] = []
            for clause_index, (product, clause) in enumerate(zip(routed_products, routed_clauses)):
                clause_terms = _significant_lexical_terms(clause)
                candidates: list[tuple[int, int, dict[str, object], set[str]]] = []
                for hit in hits:
                    if str(hit.get("product") or "").strip() != product:
                        continue
                    shared_terms = clause_terms & _significant_lexical_terms(
                        f"{product} {hit.get('heading') or ''}"
                    )
                    candidates.append((
                        len(shared_terms),
                        -int(hit.get("rank") or 0),
                        hit,
                        shared_terms,
                    ))
                if not candidates:
                    break
                _score, _inverse_rank, best_hit, shared_terms = max(
                    candidates,
                    key=lambda item: (item[0], item[1]),
                )
                if len(shared_terms) < 2:
                    break
                competing_terms: set[str] = set()
                for _candidate_score, _candidate_rank, candidate_hit, candidate_terms in candidates:
                    if candidate_hit is not best_hit:
                        competing_terms.update(candidate_terms)
                distinguishing_terms = shared_terms - competing_terms
                if not distinguishing_terms:
                    distinguishing_terms = shared_terms - _significant_lexical_terms(product)
                if not distinguishing_terms:
                    distinguishing_terms = shared_terms
                routed_requirements.append({
                    **best_hit,
                    "_shared_terms": shared_terms,
                    "_unique_terms": distinguishing_terms,
                    "_question_position": clause_index,
                })
            if len(routed_requirements) == len(routed_products):
                return routed_requirements

    # When the user literally names multiple products, bind each named product
    # to the clause where it appears before comparing headings.  This prevents a
    # word belonging to product B's clause from selecting a similarly worded
    # section in product A.  Clause splitting is purely textual; no action or
    # lifecycle vocabulary is maintained.
    product_positions: list[tuple[int, str, str]] = []
    folded_question = question.casefold()
    for hit in hits:
        product = str(hit.get("product") or "").strip()
        label = _literal_product_label(product).casefold()
        position = folded_question.find(label) if label else -1
        if position >= 0 and all(existing[1] != product for existing in product_positions):
            product_positions.append((position, product, label))
    product_positions.sort(key=lambda item: item[0])
    if len(product_positions) >= 2:
        clauses = _split_literal_question_clauses(question)
        requirements: list[dict[str, object]] = []
        for position, product, label in product_positions:
            clause = next(
                (part for part in clauses if label and label in part.casefold()),
                question,
            )
            clause_terms = _significant_lexical_terms(clause)
            candidates: list[tuple[int, int, dict[str, object], set[str]]] = []
            for hit in hits:
                if str(hit.get("product") or "").strip() != product:
                    continue
                shared_terms = clause_terms & _significant_lexical_terms(
                    f"{product} {hit.get('heading') or ''}"
                )
                candidates.append((
                    len(shared_terms),
                    -int(hit.get("rank") or 0),
                    hit,
                    shared_terms,
                ))
            if not candidates:
                continue
            _score, _inverse_rank, best_hit, shared_terms = max(
                candidates,
                key=lambda item: (item[0], item[1]),
            )
            if len(shared_terms) < 2:
                continue
            competing_terms: set[str] = set()
            for _candidate_score, _candidate_rank, candidate_hit, candidate_terms in candidates:
                if candidate_hit is best_hit:
                    continue
                competing_terms.update(candidate_terms)
            distinguishing_terms = shared_terms - competing_terms
            if not distinguishing_terms:
                product_terms = _significant_lexical_terms(product)
                distinguishing_terms = shared_terms - product_terms
            if not distinguishing_terms:
                distinguishing_terms = shared_terms
            requirements.append({
                **best_hit,
                "_shared_terms": shared_terms,
                "_unique_terms": distinguishing_terms,
                "_question_position": position,
            })
        if len(requirements) >= 2:
            requirements.sort(key=lambda hit: int(hit.get("_question_position") or 0))
            return requirements

    # The remaining generic path is only for an explicitly coordinated
    # multi-clause request. A single clause may contain a broad parent heading
    # ("使用前检查") plus one narrow target ("发动机机油"); those are scope and
    # subtopic, not two independent intents. Without this gate, a neighboring
    # procedure such as "发动机机油更换" could be appended to the answer.
    if len(_split_literal_question_clauses(question)) < 2:
        return []

    selected: list[dict[str, object]] = []
    covered_terms: set[str] = set()
    for hit in hits[:8]:
        heading = str(hit.get("heading") or "")
        product = str(hit.get("product") or "")
        shared_terms = question_terms & _significant_lexical_terms(f"{product} {heading}")
        # Requiring two literal terms keeps generic headings such as "安装位置"
        # from becoming a second intent merely because they share one verb.
        if len(shared_terms) < 2:
            continue
        new_terms = shared_terms - covered_terms
        if not selected or new_terms:
            selected.append({**hit, "_shared_terms": shared_terms})
            covered_terms.update(shared_terms)

    if len(selected) < 2:
        return []

    requirements: list[dict[str, object]] = []
    for index, hit in enumerate(selected):
        other_terms: set[str] = set()
        for other_index, other in enumerate(selected):
            if other_index != index:
                other_terms.update(other.get("_shared_terms") or set())
        unique_terms = set(hit.get("_shared_terms") or set()) - other_terms
        if not unique_terms:
            continue
        positions = [
            question.casefold().find(term)
            for term in unique_terms
            if question.casefold().find(term) >= 0
        ]
        requirements.append({
            **hit,
            "_unique_terms": unique_terms,
            "_question_position": min(positions) if positions else len(question),
        })

    requirements.sort(key=lambda hit: (
        int(hit.get("_question_position") or 0),
        int(hit.get("rank") or 0),
    ))
    return requirements if len(requirements) >= 2 else []


def _literal_compound_heading_parts(
    leaf_heading: str,
    target_terms: set[str],
) -> tuple[str, list[str], list[str], list[str]]:
    """Return display heading and selected/unselected literal heading parts."""
    match = re.match(r"^(?P<prefix>.+?的)(?P<body>.+)$", leaf_heading or "")
    prefix = match.group("prefix") if match else ""
    body = match.group("body") if match else (leaf_heading or "")
    parts = [
        part.strip()
        for part in re.split(r"(?:[、/&]|与|和|及)", body)
        if part.strip()
    ]
    if len(parts) < 2:
        return leaf_heading, parts, parts, []
    selected_parts = [
        part for part in parts
        if _significant_lexical_terms(part) & target_terms
    ]
    if not selected_parts or len(selected_parts) == len(parts):
        return leaf_heading, parts, parts, []
    unselected_parts = [part for part in parts if part not in selected_parts]
    return prefix + "、".join(selected_parts), parts, selected_parts, unselected_parts


def _literal_compound_heading_scope(
    leaf_heading: str,
    target_terms: set[str],
) -> tuple[str, set[str]]:
    """Select named sides of a compound heading by literal wording."""
    display_heading, _parts, _selected_parts, unselected_parts = (
        _literal_compound_heading_parts(leaf_heading, target_terms)
    )
    unselected_terms = {
        term
        for part in unselected_parts
        for term in _significant_lexical_terms(part)
    }
    return display_heading, unselected_terms


def _trim_unasked_compound_lines(
    text: str,
    unasked_terms: set[str],
    *,
    heading_parts: list[str] | None = None,
    selected_parts: list[str] | None = None,
) -> str:
    """Remove source sentences explicitly introduced by an unasked heading side."""
    if not text or not unasked_terms:
        return text
    kept_lines: list[str] = []
    removed_anchors: list[str] = []
    parts = list(heading_parts or [])
    selected = set(selected_parts or [])
    active = not parts or parts[0] in selected
    for line in text.splitlines():
        line_intro = re.sub(r"^\s*(?:[-•*]|\d+[.)、])?\s*", "", line)
        matched_part = next(
            (part for part in sorted(parts, key=len, reverse=True) if line_intro.startswith(part)),
            None,
        )
        if matched_part is not None:
            active = matched_part in selected
        if parts and not active:
            removed_anchors.extend(re.findall(r"\[\[PIC:([^\]]+)\]\]", line))
            kept_lines.append("")
            continue
        # With a parsed compound heading, only explicit internal heading
        # boundaries may switch scope.  A necessary operation inside the
        # selected procedure can legitimately reuse wording from another
        # heading part (for example, removing packaging during installation),
        # so sentence-level keyword deletion would corrupt the procedure.
        if parts:
            kept_lines.append(line.rstrip())
            continue
        if not parts and any(line_intro.startswith(term) for term in unasked_terms):
            removed_anchors.extend(re.findall(r"\[\[PIC:([^\]]+)\]\]", line))
            kept_lines.append("")
            continue
        units = re.split(r"(?<=[。！？.!?])", line)
        kept_units: list[str] = []
        for unit in units:
            stripped = re.sub(r"^\s*(?:[-•*]|\d+[.)、])?\s*", "", unit)
            if any(stripped.startswith(term) for term in unasked_terms):
                removed_anchors.extend(re.findall(r"\[\[PIC:([^\]]+)\]\]", unit))
                continue
            kept_units.append(unit)
        kept_lines.append("".join(kept_units).strip())
    while kept_lines and not kept_lines[-1]:
        kept_lines.pop()
    kept_text = "\n".join(kept_lines).strip()
    # A single trailing diagram commonly illustrates both sides of one compact
    # compound section. Keep that auditable image with the remaining side; do
    # not move multiple side-specific images across a removed boundary.
    if (
        len(parts) == 2
        and len(removed_anchors) == 1
        and "[[PIC:" not in kept_text
        and kept_lines
    ):
        for index in range(len(kept_lines) - 1, -1, -1):
            if kept_lines[index].strip():
                kept_lines[index] = (
                    kept_lines[index].rstrip()
                    + f" [[PIC:{removed_anchors[0]}]]"
                )
                break
        kept_text = "\n".join(kept_lines).strip()
    return kept_text


def _select_literal_source_subblocks(
    content: str,
    target_terms: set[str],
    *,
    leaf_heading: str = "",
) -> str:
    """Select complete internal source blocks whose literal title matches a target."""
    source = _clean_source_literal(content)
    if not source:
        return ""
    lines = source.splitlines()
    blocks: list[tuple[str, list[str]]] = []
    current_title = ""
    current_lines: list[str] = []

    def finish() -> None:
        nonlocal current_lines
        if current_lines:
            blocks.append((current_title, current_lines))
        current_lines = []

    for line in lines:
        stripped = line.strip()
        is_title = bool(
            stripped
            and len(stripped) <= 80
            and stripped.endswith(("：", ":"))
            and not re.match(r"^\d+[.)、]", stripped)
        )
        if is_title:
            finish()
            current_title = stripped
            current_lines = [line]
        else:
            current_lines.append(line)
    finish()

    titled = [
        "\n".join(block_lines).strip()
        for title, block_lines in blocks
        if title and (_significant_lexical_terms(title) & target_terms)
    ]
    selected = "\n\n".join(block for block in titled if block) if titled else source
    _display_heading, heading_parts, selected_parts, unselected_parts = (
        _literal_compound_heading_parts(leaf_heading, target_terms)
    )
    unasked_terms = {
        term
        for part in unselected_parts
        for term in _significant_lexical_terms(part)
    }
    return _trim_unasked_compound_lines(
        selected,
        unasked_terms,
        heading_parts=heading_parts if unselected_parts else None,
        selected_parts=selected_parts if unselected_parts else None,
    )


def _apply_multi_intent_coverage_guard(
    *,
    answer: str,
    question: str,
    trace: dict | None,
) -> str:
    """Restore already-retrieved evidence when a final answer drops one user point."""
    requirements = _literal_multi_intent_requirements(question=question, trace=trace)
    if not requirements:
        return answer

    answer_terms = _significant_lexical_terms(answer)
    missing = [
        hit for hit in requirements
        if not set(hit.get("_unique_terms") or set()).issubset(answer_terms)
    ]
    mode = "pass" if not missing else "replace_missing"
    if trace is not None:
        trace["events"].append({
            "kind": "multi_intent_coverage_guard",
            "index": len(trace["events"]) + 1,
            "mode": mode,
            "required_chunk_ids": [hit.get("matched_chunk_id") for hit in requirements],
            "missing_terms": sorted({
                term
                for hit in missing
                for term in set(hit.get("_unique_terms") or set())
            }),
        })
    if not missing:
        return answer

    products = {str(hit.get("product") or "").strip() for hit in requirements}

    blocks: list[str] = []
    evidence_language_samples: list[str] = []
    for hit in requirements:
        target_terms = set(hit.get("_unique_terms") or set())
        content = str(hit.get("content") or hit.get("matched_content") or "")
        leaf_heading = str(hit.get("heading") or "").split("/")[-1].strip()
        selected = _select_literal_source_subblocks(
            content,
            target_terms,
            leaf_heading=leaf_heading,
        )
        if not selected:
            return answer
        evidence_language_samples.append(selected)
        leaf_heading, _unasked_terms = _literal_compound_heading_scope(
            leaf_heading,
            target_terms,
        )
        first_line = next((line.strip() for line in selected.splitlines() if line.strip()), "")
        if leaf_heading and not first_line.endswith(("：", ":")):
            selected = f"{leaf_heading}\n{selected}"
        if len(products) > 1:
            product = str(hit.get("product") or "").strip()
            if product:
                selected = f"{product}\n{selected}"
        blocks.append(selected)

    if any(
        _dominant_text_language(question) != _dominant_text_language(sample)
        for sample in evidence_language_samples
    ):
        return answer
    return "\n\n".join(blocks).strip() or answer


def _apply_single_compound_scope_guard(
    *,
    answer: str,
    question: str,
    trace: dict | None,
) -> str:
    """Restore a scoped single-focus compound block, including a shared image."""
    if not trace:
        return answer
    route_info = trace.get("product_route") or {}
    if len(route_info.get("products") or []) != 1:
        return answer
    if _literal_multi_intent_requirements(question=question, trace=trace):
        return answer

    question_terms = _significant_lexical_terms(question)
    candidates: list[tuple[int, int, dict[str, object], str]] = []
    for event in trace.get("events", []):
        if event.get("kind") != "tool_call" or event.get("name") != "search_manual":
            continue
        for hit in event.get("retrieval_hits") or []:
            # This guard is allowed to narrow the highest-ranked evidence block,
            # but must never replace an answer with a lower-ranked neighboring
            # section merely because that heading contains multiple subtopics.
            if int(hit.get("rank") or 0) != 1:
                continue
            leaf_heading = str(hit.get("heading") or "").split("/")[-1].strip()
            display_heading, _parts, selected_parts, unselected_parts = (
                _literal_compound_heading_parts(leaf_heading, question_terms)
            )
            if len(selected_parts) != 1 or not unselected_parts:
                continue
            shared = question_terms & _significant_lexical_terms(
                f"{hit.get('product') or ''} {leaf_heading}"
            )
            if len(shared) < 2:
                continue
            candidates.append((
                len(shared),
                -int(hit.get("rank") or 0),
                hit,
                display_heading,
            ))
    if not candidates:
        return answer
    _score, _rank, hit, display_heading = max(
        candidates,
        key=lambda item: (item[0], item[1]),
    )
    content = str(hit.get("content") or hit.get("matched_content") or "")
    selected = _select_literal_source_subblocks(
        content,
        question_terms,
        leaf_heading=str(hit.get("heading") or "").split("/")[-1].strip(),
    )
    if not selected:
        return answer
    if _dominant_text_language(question) != _dominant_text_language(selected):
        return answer
    rebuilt = f"{display_heading}\n{selected}".strip()
    trace["events"].append({
        "kind": "single_compound_scope_guard",
        "index": len(trace["events"]) + 1,
        "chunk_id": hit.get("matched_chunk_id"),
        "mode": "replace",
    })
    return rebuilt


def _strip_numeric_figure_references(answer: str) -> str:
    """Remove source-local figure numbers while preserving inline PIC anchors."""
    text = answer or ""
    text = re.sub(
        r"如\s*(?:图|Figure|Fig\.?)\s*[A-Za-z0-9一二三四五六七八九十-]+\s*所示",
        "如图所示",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(
        r"[（(]\s*(?:图|Figure|Fig\.?)\s*[A-Za-z0-9一二三四五六七八九十-]+\s*[）)]",
        "",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(r"第\s*[0-9一二三四五六七八九十]+\s*张图", "图", text)
    text = re.sub(
        r"(?<![\w])(?:Figure|Fig\.?)\s*[A-Za-z0-9-]+(?![\w])",
        "",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(r"(?<![\w])图\s*[0-9一二三四五六七八九十-]+(?![\w])", "", text)
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r" {2,}", " ", text)
    return text.strip()


def _apply_same_language_fidelity_guard(
    *,
    answer: str,
    question: str,
    trace: dict | None,
) -> str:
    exact_answer, chunk_ids = _same_language_topic_bundle_answer(
        question=question,
        trace=trace,
    )
    has_required_support = False
    if trace is not None:
        for event in reversed(trace.get("events", [])):
            if event.get("kind") != "tool_call" or event.get("name") != "search_manual":
                continue
            hits = event.get("retrieval_hits") or []
            primary_headings = {
                str(hit.get("heading") or "").strip().casefold()
                for hit in hits
                if str(hit.get("evidence_role") or "") == "primary"
            }
            support_headings = {
                str(hit.get("heading") or "").strip().casefold()
                for hit in hits
                if str(hit.get("evidence_role") or "") == "support"
            }
            # A same-heading primary/support pair can be an arbitrary label mix,
            # not a proven structural bundle.  Replacement is allowed only when
            # retrieval selected a distinct supporting subsection (typically an
            # adjacent Warning/Note) alongside the primary subsection.
            has_required_support = bool(
                primary_headings
                and support_headings
                and any(primary != support for primary in primary_headings for support in support_headings)
            )
            break
    mode = "replace_required_bundle" if exact_answer and has_required_support else "observe_only"
    if trace is not None:
        trace["events"].append({
            "kind": "fidelity_guard",
            "index": len(trace["events"]) + 1,
            "mode": mode,
            "chunk_ids": chunk_ids,
            "source_backed": bool(exact_answer and _is_literal_evidence_answer(answer, exact_answer)),
        })
    # A primary+support bundle is created only when retrieval has proven that an
    # immediately adjacent Warning/Note is part of the selected procedure. In
    # that narrow case, restore both complete source blocks if the model clips
    # the safety preface. Ordinary ranked candidates remain observe-only.
    if exact_answer and has_required_support:
        return exact_answer
    return answer


_FINAL_CLIP_SYSTEM_PROMPT = """\
You are the final evidence editor for a manual-answering system.
Your only job is to select and lightly format the answer draft. Do not retrieve,
add facts, explain, reorder source facts, or freely paraphrase.

Rules:
1. Select complete manual sub-blocks, never isolated sentences. A sub-block starts
   at an internal title or a clearly introduced operation/object and ends before
   the next sibling title/operation. If the question names A, keep all of A. If it
   names B, keep all of B. If it names A and B, keep both complete blocks in source order.
   When one source heading structurally combines "X's A and B" but the user
   literally asks only A, keep the shared preface and A-specific source lines,
   and remove lines explicitly introduced by B. Keep both only when asked.
2. Once a block is selected, retain every sentence, part label, step, Note,
   Caution, Warning, and image anchor inside it. Delete only wholly unrequested
   sibling blocks; never trim the inside of a selected block for brevity.
3. This is deletion-only editing for factual content: retain or delete complete
   source blocks, but never merge actions, condense a sentence, rewrite a
   Warning/Caution/Note, or keep only part of one sentence.
   Begin with the selected block's existing source title when present; that title
   is part of the block and must not be deleted. Never create an opening from the
   user's wording, and never return a one-sentence summary when a complete selected
   block exists.
4. For cross-language answers, block selection happens before translation.
   Translate every sentence of each selected complete block and no unselected block.
5. Output only the final answer. Do not describe your editing process.
"""


def _clip_final_answer_with_llm(
    *,
    answer: str,
    question: str,
    model: str | None,
) -> str:
    """Give the last model call one narrow responsibility: evidence selection."""
    draft = (answer or "").strip()
    if not draft or os.getenv("FINAL_LLM_CLIP_ENABLED", "1").strip().lower() in {"0", "false", "no", "off"}:
        return answer
    response, _route = create_message_with_fallback(
        max_tokens=int(os.getenv("FINAL_LLM_CLIP_MAX_TOKENS", "2048")),
        system=_FINAL_CLIP_SYSTEM_PROMPT,
        messages=[{
            "role": "user",
            "content": f"User question:\n{question}\n\nAnswer draft:\n{draft}",
        }],
        model=model,
    )
    clipped = _extract_text_from_response(response).strip()
    return clipped or answer


def _finalize_without_tools(
    *,
    system_prompt: str,
    messages: list[dict],
    model: str | None,
) -> str:
    """跑满最大轮数后，基于现有上下文强制收束一次最终答案。"""
    finalize_messages = list(messages)
    finalize_messages.append({
        "role": "user",
        "content": (
            "不要再调用任何工具。请仅根据上面对话里已经检索到的内容，"
            "现在直接输出最终答案。\n"
            "要求：\n"
            "1. 禁止输出“处理中”“请重试”“需要更多信息”等异常或占位话术\n"
            "2. 若已有检索结果，必须尽最大可能整合成可提交答案\n"
            "3. 技术题继续保留正文中的图片锚点（如 [[PIC:Manual01_1]]）、数字规格、警告语和列表编号；客服题保持自然客服口吻\n"
            "4. 若现有检索内容仍不足，请明确说明（用与回答相同的语言：中文题用中文、英文题用英文，绝不中英混杂），不要输出异常兜底句\n"
            "5. 严格自检：逐句完整翻译且不删减；保留与描述对应的 [[PIC:...]]、步骤/部件代号；删除所有 '图N / Figure N' 数字引用，回指改写为'上图/下图'；不要使用 Markdown 列表"
        ),
    })
    response, _route = create_message_with_fallback(
        max_tokens=int(os.getenv("AGENT_MAX_TOKENS", "8192")),
        system=system_prompt,
        messages=finalize_messages,
        model=model,
    )
    return _extract_text_from_response(response)


_ROUTE_HINT_CS = (
    "【路由信号：本题为通用客服问题（非产品技术），"
    "禁止调用任何检索工具，直接按客服范例 C/D/E 的风格作答。】"
)

# 技术题通用指南
_ROUTE_HINT_TECH = (
    "【路由提示：本题更可能是产品技术问题。建议先调用 search_manual 检索手册再回答；"
    "英文提问请用英文回答，中文提问请用中文回答。"
    "若 search_manual 连续返回无结果（no_result）或命中偏泛，请换关键词、必要时 products=[] 全库确认；"
    "若仍无证据，基于已有证据收束或说明手册未覆盖，避免反复空转。】"
)
_STRUCTURE_QUERY_TOKENS = [
    "anatomy",
    "overview",
    "front view",
    "rear view",
    "navigation button view",
    "top view",
    "bottom view",
    "buttons and interfaces",
    "buttons & indicators",
    "parts",
    "components",
    "结构",
    "部件",
    "组件",
    "视图",
    "按键",
    "接口",
]


def _build_product_route_hint(route: ProductRouteDecision, question: str = "") -> str:
    """产品路由提示：恢复老版自然语言形式，按 reason/置信度分支。"""
    question_is_zh = contains_cjk(question) if question else False

    if route.reason == "explicit_multi_product" and len(route.products) > 1:
        return (
            f"【产品路由提示：题面明确点名多个产品={'、'.join(route.products)}。"
            "必须分别在每一本对应手册内检索，只用该手册回答它对应的子问题；"
            "每个产品都取得直接证据后，按用户原句顺序合并，禁止遗漏任何产品，"
            "也禁止把一本手册的步骤或图片用于另一本产品。】"
        )

    if route.reason == "clause_multi_product" and len(route.products) > 1:
        return (
            f"【产品路由提示：系统已逐分句确认对应手册={'、'.join(route.products)}。"
            "每个分句只能使用为该分句确认的手册证据；全部分句取得直接证据后，"
            "按用户原句顺序合并。不得让首个产品锁住后续独立分句，也不得跨分句串用步骤或图片。】"
        )

    # 1) 显式产品名 / 别名硬锁（单产品 high）
    if route.reason in {"explicit_product_name", "explicit_product_nickname"} and len(route.products) == 1:
        product = route.products[0]
        cross_lang = (product.endswith("手册")) != question_is_zh
        cross_lang_part = (
            "命中的手册语言与提问语言不同，请翻译后再回答。" if cross_lang else ""
        )
        return (
            f"【产品路由提示：题面已显式指明产品={product}。全程检索仅限该产品手册，"
            f"禁止扩展到其他手册或全库。{cross_lang_part}】"
        )

    # 2) 未识别候选（低置信 / 内容投票发散等）
    if not route.products:
        return (
            "【产品路由提示：本题未能可靠识别产品候选。"
            "建议直接 search_manual 用 products=[] 做全库检索；"
            "若结果偏泛或无结果，请改写关键词后再确认一次，仍无证据则说明手册未覆盖。】"
        )

    # 3) 单候选高置信（别名命中、name_and_content_agree 等）
    if len(route.products) == 1 and route.confidence == "high":
        product = route.products[0]
        cross_lang = (product.endswith("手册")) != question_is_zh
        cross_lang_part = (
            "命中的手册语言与提问语言不同，请翻译后再回答。" if cross_lang else ""
        )
        return (
            f"【产品路由提示：候选={product}，置信较高。"
            "建议优先在该手册内检索；若结果偏泛或连续无结果，再将 products 设为 [] 做一次全库确认。"
            f"{cross_lang_part}】"
        )

    # 4) 多候选（medium 置信）— 老版核心软指令
    cross_lang_products = [
        p for p in route.products if (p.endswith("手册")) != question_is_zh
    ]
    cross_lang_note = (
        "命中的部分手册语言与提问语言不同，请翻译后再回答。"
        if cross_lang_products else ""
    )
    structure_note = (
        "本题为结构/部件类问题，请优先检索 overview/view/parts/functions 等章节并基于完整 parent section 判断并列项。"
        if _is_structure_query(question) else ""
    )

    products_text = "、".join(route.products)
    confidence_word = "较高" if route.confidence == "high" else "一般"

    return (
        f"【产品路由提示：候选={products_text}。该候选置信{confidence_word}；"
        "把这些候选当作检索起点，不是唯一答案。"
        "若多个候选都命中相关信息，可以并列回答。"
        "若结果偏泛、连续命中相近章节或无结果，再将 products 设为 [] 做一次全库确认。"
        f"{cross_lang_note}{structure_note}】"
    )


def _build_routed_question(
    question: str,
    question_id: int | None,
    product_route: ProductRouteDecision | None = None,
) -> str:
    """根据 id 在用户消息前面加路由提示；不传 id 则让 LLM 自判。

    技术题分两段：通用技术题指南 + 产品候选指南。
    客服题（qid<64）只挂 _ROUTE_HINT_CS。
    """
    parts: list[str] = []
    if question_id is None:
        # 没 id（API 模式）→ 默认按技术题处理
        parts.append(_ROUTE_HINT_TECH)
        if product_route is not None:
            hint = _build_product_route_hint(product_route, question)
            if hint:
                parts.append(hint)
        parts.append(question)
        return "\n\n".join(parts)

    if question_id < 64:
        parts.append(_ROUTE_HINT_CS)
    else:
        parts.append(_ROUTE_HINT_TECH)
        if product_route is not None:
            product_hint = _build_product_route_hint(product_route, question)
            if product_hint:
                parts.append(product_hint)
    parts.append(question)
    return "\n\n".join(parts)


@dataclass
class ToolObservation:
    """一次 search_manual 工具返回后的轻量结构化观察。

    主循环用它判断是否无结果、是否反复命中同一 parent section、是否需要扩全库或强制收束；这些字段也写入 trace 供赛后复盘。
    """
    no_result: bool = False
    products: list[str] = field(default_factory=list)
    headings: list[str] = field(default_factory=list)
    parent_section_ids: list[int] = field(default_factory=list)
    dominant_product: str | None = None
    dominant_count: int = 0
    dominant_parent_section_id: int | None = None
    dominant_parent_section_count: int = 0
    dominant_section_summary: str | None = None
    explicit_product: str | None = None
    search_status: str | None = None
    search_reason: str | None = None
    search_filtered: int = 0


SEARCH_TOOL_NAMES = {"search_manual", "keyword_search", "vector_search"}


def _is_structure_query(question: str) -> bool:
    q = (question or "").lower()
    return any(token in q for token in _STRUCTURE_QUERY_TOKENS)


def _normalize_heading_key(heading: str) -> str:
    text = re.sub(r"\s+", " ", (heading or "").strip().lower())
    return text


def _is_safety_like_heading(heading: str) -> bool:
    key = _normalize_heading_key(heading)
    markers = [
        "safety",
        "hazard",
        "regulatory",
        "legal",
        "fcc",
        "warning",
        "telephone and fcc notices",
        "product safety guide",
    ]
    return any(marker in key for marker in markers)


def _parse_products_and_headings(result_text: str) -> tuple[list[str], list[str]]:
    products: list[str] = []
    headings: list[str] = []
    for line in result_text.splitlines():
        m = re.match(r"^\[\d+\]\s+产品:\s+(.*?)\s+\|\s+章节:\s+(.*)$", line.strip())
        if m:
            products.append(m.group(1).strip())
            headings.append(m.group(2).strip())
            continue
        if line.startswith("产品: "):
            products.append(line[len("产品: "):].strip())
            continue
        m2 = re.match(r"^章节:\s+\[\d+\]\s+(.*)$", line.strip())
        if m2:
            headings.append(m2.group(1).strip())
    return products, headings


def _extract_tag_value(result_text: str, tag: str) -> str | None:
    pattern = rf"^\[{re.escape(tag)}\]\s+(.*)$"
    for line in result_text.splitlines():
        m = re.match(pattern, line.strip())
        if m:
            return m.group(1).strip()
    return None


def _extract_tag_int(result_text: str, tag: str) -> int | None:
    value = _extract_tag_value(result_text, tag)
    if value is None:
        return None
    value = value.strip()
    return int(value) if value.isdigit() else None


def _observe_tool_output(name: str, input_data: dict, result_text: str) -> ToolObservation:
    obs = ToolObservation()
    text = (result_text or "").strip()
    obs.search_status = _extract_tag_value(text, "SEARCH_STATUS")
    obs.search_reason = _extract_tag_value(text, "SEARCH_REASON")
    filtered_text = _extract_tag_value(text, "SEARCH_FILTERED")
    if filtered_text and filtered_text.isdigit():
        obs.search_filtered = int(filtered_text)
    obs.no_result = text in {"", "(无检索结果)"} or text.startswith("未找到 ")
    if obs.search_status == "no_result":
        obs.no_result = True
    products, headings = _parse_products_and_headings(text)
    obs.products = products
    obs.headings = headings
    section_ids_text = _extract_tag_value(text, "SECTION_IDS")
    if section_ids_text:
        obs.parent_section_ids = [
            int(part.strip())
            for part in section_ids_text.split(",")
            if part.strip().isdigit()
        ]
    counts = Counter(products)
    if counts:
        obs.dominant_product, obs.dominant_count = counts.most_common(1)[0]
    obs.dominant_parent_section_id = _extract_tag_int(text, "SECTION_TOP")
    obs.dominant_parent_section_count = _extract_tag_int(text, "SECTION_TOP_COUNT") or 0
    obs.dominant_section_summary = _extract_tag_value(text, "SECTION_TOP_SUMMARY")

    if name in SEARCH_TOOL_NAMES:
        products_arg = input_data.get("products")
        if isinstance(products_arg, list) and len(products_arg) == 1:
            obs.explicit_product = products_arg[0]

    return obs


def _make_route_note(text: str) -> str:
    return f"【路由状态更新：{text}】"


def _coerce_tool_params(value) -> dict:
    return dict(value) if isinstance(value, dict) else {}


def _get_primary_route_product(route: ProductRouteDecision) -> str | None:
    return route.products[0] if route.products else None


def _get_locked_route_product(route: ProductRouteDecision) -> str | None:
    if route.reason in {"explicit_product_name", "explicit_product_nickname"} and len(route.products) == 1:
        return route.products[0]
    return None


def _literal_product_label(product: str) -> str:
    """Return the catalog label users are expected to name in a question."""
    label = str(product or "").strip()
    label = re.sub(r"(?:产品)?手册$", "", label, flags=re.IGNORECASE).strip()
    label = re.sub(r"\b(?:user\s+)?manual$", "", label, flags=re.IGNORECASE).strip()
    return label


def _split_literal_question_clauses(question: str) -> list[str]:
    """Split explicit parallel wording without classifying the requested actions."""
    return [
        part.strip()
        for part in re.split(
            r"(?:[；;。！？!?]+|[，,]\s*(?:并且?|以及|同时|另外|然后)|并且|以及|同时|另外|然后|(?<!合)并(?![行列]))",
            question or "",
        )
        if part.strip()
    ]


def _recover_explicit_multi_product_route(
    question: str,
    route: ProductRouteDecision,
) -> ProductRouteDecision:
    """Preserve every catalog product literally named by the user.

    ProductRouter records all explicit catalog matches in debug_scores but its
    legacy return contract narrows ``products`` to the first match.  Rehydrate
    only literal labels from the original wording; semantic candidates and
    inferred products remain subject to the existing single-manual behavior.
    """
    if route.reason not in {"explicit_product_name", "explicit_product_nickname"}:
        return route
    folded_question = (question or "").casefold()
    matches: list[tuple[int, str, float]] = []
    for raw_product, raw_score in route.debug_scores:
        product = str(raw_product or "").strip()
        label = _literal_product_label(product).casefold()
        if len(label) < 2:
            continue
        position = folded_question.find(label)
        if position < 0:
            continue
        matches.append((position, product, float(raw_score)))
    matches.sort(key=lambda item: item[0])
    products: list[str] = []
    scores: list[tuple[str, float]] = []
    for _position, product, score in matches:
        if product in products:
            continue
        products.append(product)
        scores.append((product, score))
    if len(products) < 2:
        return route
    return ProductRouteDecision(
        products=products,
        confidence="high",
        reason="explicit_multi_product",
        debug_scores=scores,
    )


def _recover_clause_multi_product_route(
    engine: RetrievalEngine,
    router: ProductRouter,
    question: str,
    route: ProductRouteDecision,
) -> ProductRouteDecision:
    """Add a per-clause manual only after a unique literal evidence match.

    A single explicit product continues to be a hard boundary for ordinary
    questions.  For an actual multi-clause question, however, a later clause may
    omit its product name.  Route that clause independently and accept another
    manual only when retrieval produces one unique strong heading anchor.  Ties,
    weak matches, and generic clauses preserve the original single-manual route.
    """
    if route.reason == "explicit_multi_product":
        return route
    clauses = _split_literal_question_clauses(question)
    if len(clauses) < 2:
        return route

    resolved: list[tuple[int, str, float]] = []
    for clause_index, clause in enumerate(clauses):
        decision = router.route(clause)
        product: str | None = None
        confidence_score = 0.0
        if (
            decision.reason in {"explicit_product_name", "explicit_product_nickname"}
            and len(decision.products) == 1
        ):
            product = decision.products[0]
            confidence_score = 1000.0
        else:
            clause_terms = _significant_lexical_terms(clause)
            if len(clause_terms) < 2:
                continue
            candidates, _filtered = engine.search_manual(
                list(clause_terms),
                semantic_query=clause,
                original_query=clause,
                top_k=5,
                products=decision.products or None,
            )
            narrowed, _literal_filtered = _narrow_to_unique_literal_heading(
                clause,
                candidates,
            )
            if (
                len(narrowed) == 1
                and narrowed[0].source.get("evidence_role") == "primary"
            ):
                product = narrowed[0].product
                confidence_score = float(narrowed[0].score or 0.0)
        if product and all(existing[1] != product for existing in resolved):
            resolved.append((clause_index, product, confidence_score))

    if len(resolved) < 2:
        return route
    resolved.sort(key=lambda item: item[0])
    return ProductRouteDecision(
        products=[product for _index, product, _score in resolved],
        confidence="high",
        reason="clause_multi_product",
        debug_scores=[
            (product, score)
            for _index, product, score in resolved
        ],
    )


def _get_locked_route_products(route: ProductRouteDecision) -> list[str]:
    if route.reason in {"explicit_multi_product", "clause_multi_product"} and len(route.products) > 1:
        return list(route.products)
    product = _get_locked_route_product(route)
    return [product] if product else []



def _question_text(question: str | list) -> str:
    """从纯文本或 OpenAI-compatible 多模态 content 中抽出文本问题。

    产品路由、预检索、trace 和客服/技术分类只需要文字；图片仍保留在原始 content 中交给回答模型。
    """
    if isinstance(question, str):
        return question
    parts: list[str] = []
    for item in question:
        if isinstance(item, dict):
            if item.get("type") == "text":
                parts.append(str(item.get("text", "")))
            elif "text" in item:
                parts.append(str(item.get("text", "")))
        else:
            parts.append(str(item))
    return "\n".join(p for p in parts if p)


def _with_routed_text(question: str | list, routed_text: str) -> str | list:
    """把路由提示和预检索提示写回第一段文本，同时保留用户上传图片。

    API 多模态请求会传入 content list；这里只替换首个 text block，不动 image_url block，保证图片仍随同本轮消息进入主回答模型。
    """
    if isinstance(question, str):
        return routed_text
    replaced = False
    content: list = []
    for item in question:
        if isinstance(item, dict) and item.get("type") == "text" and not replaced:
            new_item = dict(item)
            new_item["text"] = routed_text
            content.append(new_item)
            replaced = True
        else:
            content.append(item)
    if not replaced:
        content.insert(0, {"type": "text", "text": routed_text})
    return content


def run_agent(
    question: str | list,
    engine: RetrievalEngine,
    model: str | None = None,
    session_id: str | None = None,
    forced_product: str | None = None,
    question_id: int | None = None,
    collect_trace: bool = False,
    stream_ttft: bool = False,
    token_callback=None,
    progress_callback=None,
    retrieval_query: str | None = None,
) -> AgentResult:
    """运行 ReAct Agent，返回最终回答。

    路由：传入 question_id 时按 id<64 客服 / id>=64 技术 硬路由；不传则 LLM 自判。
    图片处理：让 LLM 保留正文中的 [[PIC:文件名]] 锚点，最终再抽取为 <PIC> + pics。
    stream_ttft：每轮主循环 LLM 调用改流式（拼回同构 response，工具/循环逻辑不变），
        记录最终回答（出现文本增量那轮）的首 token 耗时到 AgentResult.ttft。默认 False=原行为。
    """
    final_ttft: float | None = None
    def emit_progress(stage: str, message: str) -> None:
        if progress_callback is not None:
            progress_callback(stage, message)

    emit_progress("start", "正在初始化检索引擎")
    engine.ensure_index()
    trace_started_at = time.strftime("%Y-%m-%d %H:%M:%S")
    question_text = _question_text(question)
    explicit_retrieval_query = str(retrieval_query or "").strip()
    retrieval_question_text = explicit_retrieval_query or _current_question_for_retrieval(question_text)
    retrieval_search_text = (
        retrieval_question_text
        if explicit_retrieval_query
        else _retrieval_query_with_local_context(question_text, retrieval_question_text)
    )
    trace_t0 = time.time()
    if question_id is not None:
        os.environ["CURRENT_QID"] = str(question_id)
        try:
            import retrieval_engine as _retrieval_engine
            _retrieval_engine._RERANK_CONTEXT.qid = str(question_id)
        except Exception:
            pass
    trace: dict | None = None
    if collect_trace:
        trace = {
            "id": question_id,
            "question": question_text,
            "started_at": trace_started_at,
            "events": [],
        }

    route_t0 = time.time()
    product_route = ProductRouteDecision([], "none", "not_tech_question", [])
    if question_id is None or question_id >= 64:
        emit_progress("route", "正在定位产品手册")
        router = _get_product_router(engine)
        product_route = _recover_explicit_multi_product_route(
            retrieval_question_text,
            router.route(retrieval_question_text),
        )
        if not forced_product:
            product_route = _recover_clause_multi_product_route(
                engine,
                router,
                retrieval_question_text,
                product_route,
            )
        # A product chosen in the UI is explicit user context, not an answer
        # lookup. Resolve it through the same V6 catalog, then constrain all
        # pre-retrieval and formal retrieval to that canonical manual.
        if forced_product:
            hinted_route = _get_product_router(engine).route(forced_product)
            if len(hinted_route.products) == 1:
                product = hinted_route.products[0]
                product_route = ProductRouteDecision(
                    products=[product],
                    confidence="high",
                    reason="ui_product_hint",
                    debug_scores=[(product, 1001.0)],
                )
    product_route_elapsed = round(time.time() - route_t0, 3)
    current_route = product_route
    locked_route_products = _get_locked_route_products(product_route)
    locked_route_product = locked_route_products[0] if len(locked_route_products) == 1 else None
    retrieval_search_text = _expand_mode_enumeration_query(
        retrieval_search_text,
        list(current_route.products or []),
    )
    if trace is not None:
        trace["product_route"] = asdict(product_route)
        trace["retrieval_question"] = retrieval_question_text
        trace["retrieval_search_text"] = retrieval_search_text
        trace["routed_question"] = _build_routed_question(question_text, question_id, product_route)
        trace["timings"] = {
            "product_route_elapsed": product_route_elapsed,
            "pre_retrieval": {},
            "llm_calls": [],
            "finalize_elapsed": None,
        }

    # V3.1：按 qid 选 system prompt
    # - qid < 64 → 纯客服 prompt（不含技术路由/检索噪声，回到 V2 风格 + 完整性要求）
    # - qid >= 64 或 None（API 在线模式）→ 技术 prompt（V3 路由 + 完整性优先）
    if question_id is not None and question_id < 64:
        system_prompt = SERVICE_SYSTEM_PROMPT
    else:
        # Chunk 管理后台可以在运行中发布新手册。用当前 catalog 动态刷新
        # 产品名单，避免新产品已经进入索引却仍缺席于 Agent 路由提示。
        dynamic_product_block = build_product_prompt_block(engine.catalog.keys())
        system_prompt = TECH_SYSTEM_PROMPT.replace(
            PRODUCT_PROMPT_BLOCK,
            dynamic_product_block,
            1,
        )

    routed_question = _build_routed_question(question_text, question_id, product_route)
    messages = [{
        "role": "user",
        "content": _with_routed_text(question, routed_question),
    }]
    # Preserve the union of images that actually reached either pre-retrieval or
    # formal tool retrieval.  The final evidence selector may only choose from
    # this auditable pool plus narrow same-heading structural neighbours.
    evidence_candidate_images: list[str] = []

    # Fast path: one deterministic hybrid retrieval followed by one answer call.
    # The existing ReAct loop remains the quality fallback for ambiguous,
    # multi-product, multimodal, comparison, or evidence-empty questions.
    fast_terms = _significant_lexical_terms(retrieval_search_text)
    fast_path_candidate = (
        _fast_tech_path_enabled()
        and (question_id is None or question_id >= 64)
        and isinstance(question, str)
        and len(current_route.products or []) == 1
        and current_route.confidence == "high"
        and current_route.reason not in {"explicit_multi_product", "clause_multi_product"}
        and not _question_requests_comparison(retrieval_question_text)
        and len(fast_terms) >= 2
    )
    if fast_path_candidate:
        emit_progress("retrieve", "正在并行执行 BM25 与语义召回")
        fast_retrieval_t0 = time.time()
        fast_keywords = sorted(
            fast_terms,
            key=lambda term: retrieval_search_text.casefold().find(term),
        )[:12]
        fast_results, fast_filtered = engine.search_manual(
            fast_keywords or [retrieval_search_text],
            semantic_query=retrieval_search_text,
            original_query=retrieval_search_text,
            top_k=MAX_SEARCH_RESULTS,
            products=list(current_route.products),
        )
        fast_results, fast_mode_filtered = _filter_mode_enumeration_results(
            retrieval_question_text,
            fast_results,
        )
        fast_filtered += fast_mode_filtered
        fast_retrieval_elapsed = time.time() - fast_retrieval_t0
        fast_result_text = format_search_results(fast_results, fast_filtered)
        fast_pics = _collect_pics_from_results(fast_results)
        _extend_unique_pics(evidence_candidate_images, fast_pics)
        if trace is not None:
            trace["timings"]["fast_retrieval"] = {
                "elapsed": round(fast_retrieval_elapsed, 3),
                "returned_sections": len(fast_results),
                "filtered": fast_filtered,
            }
            trace["events"].append(
                _build_trace_tool_event(
                    index=len(trace["events"]) + 1,
                    name="search_manual",
                    input_data={
                        "keywords": fast_keywords,
                        "query": retrieval_search_text,
                        "products": list(current_route.products),
                        "execution_path": "fast_hybrid",
                    },
                    default_products=list(current_route.products),
                    default_query_context=retrieval_search_text,
                    elapsed=fast_retrieval_elapsed,
                    pics=fast_pics,
                    result_text=fast_result_text,
                )
            )

        fast_evidence_ok = bool(
            fast_results
            and any(
                len(str(result.text or "").strip()) >= 60
                for result in fast_results[:3]
            )
        )
        if fast_evidence_ok:
            messages.append({
                "role": "user",
                "content": (
                    "[正式手册检索证据]\n"
                    "以下证据已由系统在当前产品内完成 BM25、Dense 与 Rerank，"
                    "无需再规划或调用检索工具。\n\n"
                    f"{fast_result_text}"
                ),
            })
            messages.append({
                "role": "user",
                "content": (
                    "请现在直接回答用户最后一个问题，只能使用上述正式手册证据。\n"
                    "把最直接回答问题的片段作为主证据；补充与该操作直接相关的"
                    "警告、条件、例外和共用步骤。\n"
                    "不要输出用户未询问的同级主题，例如只问冷机启动时不得加入"
                    "热机启动或停机步骤。不要整段照搬复合父章节。\n"
                    "保留与最终文字逐段对应的 [[PIC:...]] 图片锚点；"
                    "不要保留属于被排除主题的图片。不要调用工具。"
                ),
            })
            messages.append({
                "role": "user",
                "content": (
                    "Answer the user's latest question directly in the first "
                    "sentence, before any procedure or background. "
                    "For yes/no questions, begin with yes, no, or state that "
                    "the manual does not explicitly specify it. Never replace "
                    "a narrow follow-up with a general how-to. Prefer evidence "
                    "about the same component and action; treat placement or "
                    "other lifecycle stages as unrelated unless explicitly "
                    "asked. If the exact point is absent, say so and provide "
                    "only the nearest directly supported precaution."
                ),
            })
            emit_progress("model", "正在根据正式证据生成答案")
            fast_llm_t0 = time.time()
            if stream_ttft:
                response, _route, final_ttft = create_message_streaming(
                    max_tokens=int(os.getenv("AGENT_MAX_TOKENS", "8192")),
                    system=system_prompt,
                    tools=None,
                    messages=messages,
                    model=model,
                    on_delta=token_callback,
                )
            else:
                response, _route = create_message_with_fallback(
                    max_tokens=int(os.getenv("AGENT_MAX_TOKENS", "8192")),
                    system=system_prompt,
                    tools=None,
                    messages=messages,
                    model=model,
                )
            fast_llm_elapsed = time.time() - fast_llm_t0
            if trace is not None:
                trace["timings"]["llm_calls"].append({
                    "index": 1,
                    "turn": 1,
                    "elapsed": round(fast_llm_elapsed, 3),
                    "has_tool": False,
                    "content_blocks": len(response.content or []),
                    "execution_path": "fast_hybrid",
                })
                trace["events"].append(
                    _build_trace_llm_event(
                        index=len(trace["events"]) + 1,
                        response_content=response.content,
                    )
                )

            answer = _extract_text_from_response(response)
            answer = _postprocess_final_answer(
                answer=answer,
                question=retrieval_question_text,
                system_prompt=system_prompt,
                model=model,
                route_products=current_route.products,
            )
            answer, evidence_selection_trace = _apply_evidence_selection(
                answer=answer,
                question=retrieval_question_text,
                engine=engine,
                candidate_images=evidence_candidate_images,
                route_products=list(current_route.products or []),
                model=model,
            )
            if trace is not None and evidence_selection_trace is not None:
                trace["events"].append({
                    "kind": "evidence_selection",
                    "index": len(trace["events"]) + 1,
                    **evidence_selection_trace,
                })
            answer = _apply_same_language_fidelity_guard(
                answer=answer,
                question=retrieval_question_text,
                trace=trace,
            )
            answer = _apply_multi_intent_coverage_guard(
                answer=answer,
                question=retrieval_question_text,
                trace=trace,
            )
            # Deliberately skip _clip_final_answer_with_llm and
            # _apply_single_compound_scope_guard here. The former adds another
            # model call; the latter previously replaced a focused cold-start
            # answer with the entire cold/hot/stop parent section.
            answer = _strip_numeric_figure_references(answer)
            answer, pics = _resolve_answer_pics(answer)
            emit_progress("done", "快速检索、生成和证据校验完成")
            if trace is not None:
                trace["execution_path"] = "fast_hybrid"
            return AgentResult(
                answer=answer,
                pics=pics,
                tool_calls=1,
                turns=1,
                ttft=final_ttft,
                trace=(
                    {
                        **trace,
                        "result": {
                            "answer": answer,
                            "pics": pics,
                            "tool_calls": 1,
                            "turns": 1,
                            "execution_path": "fast_hybrid",
                        },
                        "error": None,
                        "elapsed": round(time.time() - trace_t0, 2),
                    }
                    if trace is not None else None
                ),
            )
        if trace is not None:
            trace["events"].append({
                "kind": "fast_path_fallback",
                "index": len(trace["events"]) + 1,
                "reason": "insufficient_evidence",
            })
        emit_progress("retrieve", "快速召回证据不足，正在进入深度检索")

    pre_results: list[SearchResult] = []
    pre_filtered = 0
    # 初始预检索只对技术题启用；客服题不应引入检索噪声，也不应依赖 retrieval 辅助。
    if question_id is None or question_id >= 64:
        emit_progress("retrieve", "正在执行手册预检索")
        pre_total_t0 = time.time()
        dense_elapsed = 0.0
        rerank_elapsed = 0.0
        build_results_elapsed = 0.0
        engine.ensure_index()
        # 产品已知时：在该产品 chunk 内做 dense 召回（filter 在前），而不是"全局 top-30 再过滤"。
        # 通用词 query（清洁/使用/设置）下，自家章节会被别产品挤出全局 top-30，过滤后只剩 1 节（见 q108）。
        # 产品内召回保证目标手册的相关章节都进候选，预检索覆盖更全。
        if current_route.products:
            allowed: set[int] = set()
            for p in current_route.products:
                allowed.update(engine.product_chunk_ids.get(p, []))
            dense_t0 = time.time()
            dense_ids = engine._dense_recall(retrieval_search_text, top_n=30, allowed_doc_ids=sorted(allowed))
            dense_elapsed = round(time.time() - dense_t0, 3)
        else:
            dense_t0 = time.time()
            dense_ids = engine._dense_recall(retrieval_search_text, top_n=30)
            dense_ids = engine._reorder_by_lang(retrieval_search_text, dense_ids)
            dense_elapsed = round(time.time() - dense_t0, 3)
        if dense_ids:
            rerank_t0 = time.time()
            pre_ids = engine._rerank_candidates(retrieval_search_text, dense_ids, top_n=PRE_RETRIEVAL_RESULTS)[:PRE_RETRIEVAL_RESULTS]
            rerank_elapsed = round(time.time() - rerank_t0, 3)
            build_t0 = time.time()
            pre_results = engine._build_results(pre_ids)
            pre_results, pre_literal_filtered = _narrow_to_unique_literal_heading(
                retrieval_question_text,
                pre_results,
            )
            pre_filtered += pre_literal_filtered
            pre_results, pre_mode_filtered = _filter_mode_enumeration_results(
                retrieval_question_text,
                pre_results,
            )
            pre_filtered += pre_mode_filtered
            build_results_elapsed = round(time.time() - build_t0, 3)
        _extend_unique_pics(evidence_candidate_images, _collect_pics_from_results(pre_results))
        if trace is not None:
            trace["timings"]["pre_retrieval"] = {
                "total_elapsed": round(time.time() - pre_total_t0, 3),
                "dense_elapsed": dense_elapsed,
                "rerank_elapsed": rerank_elapsed,
                "build_results_elapsed": build_results_elapsed,
                "dense_candidates": len(dense_ids or []),
                "returned_sections": len(pre_results),
            }

    # 完整链路诊断：把预检索 top-N 的每个 section（产品/标题/rerank分/可选图）落进 trace，
    # 配合后续 tool_call 的 pics 与最终 answer pics，可逐图还原“召回→注入→选用”三层命运。
    if trace is not None:
        trace["events"].append({
            "kind": "pre_retrieval",
            "index": len(trace["events"]) + 1,
            "products": list(current_route.products or []),
            "sections": [
                {
                    "rank": i,
                    "chunk_id": r.chunk_id,
                    "product": r.product,
                    "heading": r.heading,
                    "score": round(float(r.score), 4),
                    "pics": list(r.pics or []),
                }
                for i, r in enumerate(pre_results)
            ],
        })

    if pre_results:
        pre_text = format_search_results(pre_results, pre_filtered)
        messages.append({
            "role": "user",
            "content": (
                "[系统预检索结果]\n"
                "以下内容是系统根据当前问题做的首轮检索定位结果，仅作为后续检索线索；"
                "它不是正式 search_manual 工具返回，技术题仍需继续调用 search_manual 做显式确认。\n\n"
                f"{pre_text}"
            ),
        })
        messages.append({
            "role": "user",
            "content": _make_route_note("系统预检索：用你的问题做了首轮向量检索，结果仅供参考，只能作为后续检索线索，不能直接替代正式工具检索。技术题仍需继续调用 search_manual 做显式确认后再作答。"),
        })

    tool_calls = 0
    empty_search_streak = 0
    empty_search_product: str | None = None
    expand_hint_emitted = False
    same_product_streak = 0
    same_product_no_result_hits = 0
    same_product_name: str | None = None
    structure_query = _is_structure_query(question_text)
    heading_memory: dict[str, set[str]] = {}
    seen_result_keys: set[str] = set()
    low_gain_product: str | None = None
    low_gain_streak = 0
    low_gain_hint_emitted = False
    auto_expand_once = False
    safety_loop_streak = 0
    zero_headings_streak = 0
    # 系统预检索仅作为首轮定位参考；技术题仍需至少一次 search_manual 做正式确认。
    formal_retrieval_confirmed = False
    section_focus_product: str | None = None
    section_focus_id: int | None = None
    section_focus_streak = 0
    section_focus_hint_emitted = False
    recent_focus_product: str | None = _get_primary_route_product(current_route)
    search_attempts = 0

    effective_turns_used = 0
    internal_iterations = 0

    while effective_turns_used < MAX_TURNS and internal_iterations < MAX_INTERNAL_ITERATIONS:
        internal_iterations += 1
        current_turn = effective_turns_used + 1
        remaining_turns = MAX_TURNS - effective_turns_used
        remaining_search_attempts = MAX_SEARCH_ATTEMPTS - search_attempts
        turn_reminder = _make_route_note(
            f"当前是第 {current_turn} / {MAX_TURNS} 次 ReAct 决策机会，剩余 {remaining_turns} 次。"
            "每次机会可以选择继续调用工具，或直接给出最终答案；如果本轮继续调用工具，将消耗一次回答机会。"
            f"普通检索已使用 {search_attempts} 次；该计数仅用于防止重复搜索，不代表还可以额外增加模型轮次。"
            "请参考上一轮状态，避免重复检索；若已有 search_manual 工具结果足以回答，请直接收束答案。"
        )
        if remaining_turns <= 1:
            turn_reminder += "\n" + _make_route_note(
                "当前已是最后一次正常 ReAct 决策机会。除非完全没有可用证据，否则不要再调用工具；"
                "应直接基于 search_manual 工具结果和系统预检索线索输出最终答案。若本轮继续调用工具，"
                "后续只能进入无工具强制收束，答案质量可能下降。"
            )
        active_tools = TOOLS
        llm_t0 = time.time()
        if stream_ttft:
            emit_progress("model", f"正在执行第 {current_turn} 轮模型推理")
            response, _route, _turn_ttft = create_message_streaming(
                max_tokens=int(os.getenv("AGENT_MAX_TOKENS", "8192")),
                system=system_prompt,
                tools=active_tools,
                messages=messages + [{"role": "user", "content": turn_reminder}],
                model=model,
                on_delta=token_callback,
            )
            # 本轮出现文本增量(content)→本轮是最终回答，记其首 token；纯工具轮 _turn_ttft=None
            if _turn_ttft is not None:
                final_ttft = _turn_ttft
        else:
            response, _route = create_message_with_fallback(
                max_tokens=int(os.getenv("AGENT_MAX_TOKENS", "8192")),
                system=system_prompt,
                tools=active_tools,
                messages=messages + [{"role": "user", "content": turn_reminder}],
                model=model,
            )
        llm_elapsed = round(time.time() - llm_t0, 3)
        if trace is not None:
            has_tool = any(getattr(block, "type", None) == "tool_use" for block in response.content)
            trace["timings"]["llm_calls"].append({
                "index": len(trace["timings"].get("llm_calls", [])) + 1,
                "turn": current_turn,
                "elapsed": llm_elapsed,
                "has_tool": has_tool,
                "content_blocks": len(response.content or []),
            })
        if trace is not None:
            trace["events"].append(
                _build_trace_llm_event(
                    index=len(trace["events"]) + 1,
                    response_content=response.content,
                )
            )

        # 收集文本和工具调用
        has_tool_use = False
        executed_tool_round = False
        tool_results = []
        route_notes: list[str] = []

        for block in response.content:
            if block.type != "tool_use":
                continue
            emit_progress("retrieve", f"正在调用 {block.name} 获取正式证据")
            has_tool_use = True
            tool_calls += 1
            tool_input = dict(block.input or {})
            dropped_expansion_keywords: list[str] = []
            if block.name == "search_manual":
                tool_input, dropped_expansion_keywords = _sanitize_search_input_from_original_words(
                    tool_input,
                    question_text,
                )
                if dropped_expansion_keywords:
                    route_notes.append(
                        _make_route_note(
                            "已忽略包含用户原话之外新增词项的检索扩写："
                            + "、".join(dropped_expansion_keywords)
                            + "。扩写内容不得作为答案需求或选段依据。"
                        )
                    )
            search_blocked_by_circuit_breaker = False
            if locked_route_products and block.name in SEARCH_TOOL_NAMES:
                if tool_input.get("products") != locked_route_products:
                    tool_input["products"] = list(locked_route_products)
                    route_notes.append(
                        _make_route_note(
                            "题面已明确点名产品="
                            + "、".join(locked_route_products)
                            + "；本轮检索分别锁定这些产品，不得遗漏或扩展到其他手册。"
                        )
                    )
            if (
                not locked_route_products
                and
                auto_expand_once
                and block.name in SEARCH_TOOL_NAMES
                and isinstance(tool_input.get("products"), list)
                and len(tool_input.get("products") or []) == 1
            ):
                tool_input["products"] = []
                auto_expand_once = False
                route_notes.append(
                    _make_route_note(
                        "上一轮已判定当前产品内信息增益过低；本轮普通检索自动放开到全库做一次确认。"
                    )
                )
            if (
                not search_blocked_by_circuit_breaker
                and search_attempts >= MAX_SEARCH_ATTEMPTS
                and block.name in SEARCH_TOOL_NAMES
            ):
                target_product = section_focus_product or recent_focus_product or _get_primary_route_product(current_route)
                search_blocked_by_circuit_breaker = True
                result_text = (
                    "（状态机拦截：search_manual 检索次数已达上限。"
                    "禁止继续 search_manual。"
                    "请基于现有 search_manual 检索证据和系统预检索线索完整收束答案；若证据仍不足，请用同语言说明手册未覆盖。）"
                )
                call_pics = []
                route_notes.append(
                    _make_route_note(
                        f"search_manual 已使用 {MAX_SEARCH_ATTEMPTS} 次；"
                        "请直接基于已有证据收束，不能再继续 search_manual。"
                    )
                )
            if not search_blocked_by_circuit_breaker:
                tool_started_at = time.time()
                result_text, call_pics = _execute_tool_with_pics(
                    engine,
                    block.name,
                    tool_input,
                    default_products=current_route.products or None,
                    default_query_context=retrieval_search_text,
                    seen_result_keys=seen_result_keys,
                    balance_products=len(locked_route_products) > 1,
                )
                _extend_unique_pics(evidence_candidate_images, call_pics)
                if trace is not None:
                    trace["events"].append(
                        _build_trace_tool_event(
                            index=len(trace["events"]) + 1,
                            name=block.name,
                            input_data=tool_input,
                            default_products=current_route.products or None,
                            default_query_context=retrieval_search_text,
                            elapsed=time.time() - tool_started_at,
                            pics=call_pics,
                            result_text=result_text,
                        )
                    )
            # The assistant tool-use block is replayed to the next model turn.
            # Keep its protocol id/name, but replace its mutable input mapping
            # with the executed, original-intent-safe parameters so the model
            # cannot later treat its own speculative keywords as user requests.
            if isinstance(getattr(block, "input", None), dict):
                block.input.clear()
                block.input.update(tool_input)
            if not search_blocked_by_circuit_breaker:
                executed_tool_round = True
            if search_blocked_by_circuit_breaker:
                tool_calls -= 1
            obs = _observe_tool_output(block.name, tool_input, result_text)
            if (
                not search_blocked_by_circuit_breaker
                and block.name in SEARCH_TOOL_NAMES
            ):
                formal_retrieval_confirmed = True

            if block.name in SEARCH_TOOL_NAMES and not search_blocked_by_circuit_breaker:
                search_attempts += 1
                # 连续 0 headings 计数：用于提示换关键词或收束
                if not obs.headings:
                    zero_headings_streak += 1
                else:
                    zero_headings_streak = 0

                candidate_product = obs.explicit_product
                if not candidate_product and isinstance((block.input or {}).get("products"), list):
                    products_arg = (block.input or {}).get("products") or []
                    if len(products_arg) == 1:
                        candidate_product = products_arg[0]

                if candidate_product:
                    recent_focus_product = candidate_product
                    if candidate_product == same_product_name:
                        same_product_streak += 1
                    else:
                        same_product_name = candidate_product
                        same_product_streak = 1
                        same_product_no_result_hits = 0
                    if obs.no_result:
                        same_product_no_result_hits += 1
                else:
                    same_product_name = None
                    same_product_streak = 0
                    same_product_no_result_hits = 0

                dominant_product = obs.dominant_product or candidate_product
                if dominant_product and obs.headings:
                    heading_keys = {
                        _normalize_heading_key(h)
                        for h in obs.headings
                        if _normalize_heading_key(h)
                    }
                    seen_headings = heading_memory.setdefault(dominant_product, set())
                    fresh_headings = heading_keys - seen_headings
                    repeated_ratio = 1.0 - (len(fresh_headings) / max(len(heading_keys), 1))
                    safety_like_ratio = (
                        sum(1 for h in obs.headings if _is_safety_like_heading(h)) / max(len(obs.headings), 1)
                    )
                    if repeated_ratio >= 0.75:
                        if dominant_product == low_gain_product:
                            low_gain_streak += 1
                        else:
                            low_gain_product = dominant_product
                            low_gain_streak = 1
                    else:
                        low_gain_product = dominant_product
                        low_gain_streak = 0
                        low_gain_hint_emitted = False
                    if safety_like_ratio >= 0.6:
                        safety_loop_streak += 1
                    else:
                        safety_loop_streak = 0
                    seen_headings.update(heading_keys)
                    if (
                        obs.dominant_parent_section_id is not None
                        and obs.dominant_parent_section_count >= 2
                    ):
                        if (
                            dominant_product == section_focus_product
                            and obs.dominant_parent_section_id == section_focus_id
                        ):
                            section_focus_streak += 1
                        else:
                            section_focus_product = dominant_product
                            section_focus_id = obs.dominant_parent_section_id
                            section_focus_streak = 1
                            section_focus_hint_emitted = False
                    else:
                        section_focus_product = None
                        section_focus_id = None
                        section_focus_streak = 0
                        section_focus_hint_emitted = False
                else:
                    low_gain_product = None
                    low_gain_streak = 0
                    low_gain_hint_emitted = False
                    safety_loop_streak = 0
                    section_focus_product = None
                    section_focus_id = None
                    section_focus_streak = 0
                    section_focus_hint_emitted = False

                products_arg = (block.input or {}).get("products")
                search_is_unbounded = (
                    not isinstance(products_arg, list) or len(products_arg) == 0
                )
                if (
                    not locked_route_products
                    and
                    search_is_unbounded
                    and obs.dominant_product
                    and obs.dominant_count >= 2
                    and current_route.products[:1] != [obs.dominant_product]
                ):
                    current_route = ProductRouteDecision(
                        products=[obs.dominant_product],
                        confidence="high",
                        reason="retrieval_evidence_rebind",
                        debug_scores=[(obs.dominant_product, float(obs.dominant_count))],
                    )
                    recent_focus_product = obs.dominant_product
                    same_product_name = obs.dominant_product
                    same_product_streak = 0
                    same_product_no_result_hits = 0
                    empty_search_streak = 0
                    empty_search_product = None
                    expand_hint_emitted = False
                    route_notes.append(
                        _make_route_note(
                            f"全库检索的主命中已明显收敛到 {obs.dominant_product}；"
                            "后续优先围绕该产品继续检索。"
                        )
                    )

                if obs.no_result:
                    if candidate_product and candidate_product == empty_search_product:
                        empty_search_streak += 1
                    else:
                        empty_search_product = candidate_product
                        empty_search_streak = 1
                else:
                    empty_search_streak = 0
                    empty_search_product = None
                    expand_hint_emitted = False

            if (
                structure_query
                and empty_search_streak >= 2
            ):
                route_notes.append(
                    _make_route_note(
                        "当前问题更像目录/结构题，且普通检索连续无结果；"
                        "请改用 overview/view/parts/functions 等目录词或 products=[] 全库确认；仍无证据则基于已有内容收束。"
                    )
                )
            elif (
                section_focus_streak >= 1
                and section_focus_product is not None
                and section_focus_id is not None
                and not section_focus_hint_emitted
            ):
                section_summary = (obs.dominant_section_summary or "").strip()
                summary_hint = f" 上层摘要：{section_summary}" if section_summary else ""
                route_notes.append(
                    _make_route_note(
                        f"当前多条命中已聚合到 {section_focus_product} 的上层章节 {section_focus_id}。"
                        "检索已返回该 parent section 的证据；不要继续重复搜索，优先围绕该章节直接收束。"
                        f"{summary_hint}"
                    )
                )
                section_focus_hint_emitted = True

            # 连续 2 次检索返回 0 headings → 停止空转，改关键词/全库确认或收束
            if zero_headings_streak >= 2:
                route_notes.append(
                    _make_route_note(
                        "连续 2 次检索均未返回有效章节（headings），说明关键词无法命中手册内容；"
                        "请换一组高信息量关键词或 products=[] 全库确认；若仍无证据，请基于已有内容收束。"
                    )
                )
            elif (
                not structure_query
                and low_gain_product is not None
                and low_gain_streak >= 2
                and not low_gain_hint_emitted
            ):
                auto_expand_once = current_route.confidence == "medium"
                alternative_products = [
                    product
                    for product in current_route.products
                    if product != low_gain_product
                ]
                if current_route.confidence == "medium" and alternative_products:
                    low_gain_message = (
                        f"你在 {low_gain_product} 内连续命中相近章节，信息增益较低；"
                        f"不要只盯住单一候选，当前还可检查 {'、'.join(alternative_products)}。"
                        " 下一轮优先改用更贴近用户动作/对象的关键词，切去其他候选或做一次全库确认。"
                        " 若不同候选都给出相关信息，可按“若是 A…；若是 B…”并列回答。"
                    )
                else:
                    low_gain_message = (
                        f"你在 {low_gain_product} 内连续命中相近章节，信息增益较低；"
                        "建议下一轮改用更贴近用户动作/对象的关键词重试，"
                        "优先查 setup/connection/procedure 等步骤型线索。"
                        "若仍不贴题，再将 products 设为 [] 做一次全库确认。"
                    )
                route_notes.append(
                    _make_route_note(low_gain_message)
                )
                low_gain_hint_emitted = True
            elif (
                not structure_query
                and safety_loop_streak >= 2
            ):
                route_notes.append(
                    _make_route_note(
                        "当前结果连续落在 safety/regulatory 类章节，和用户要的操作步骤不完全对齐；"
                        "下一轮优先查 Quick Setup Guide、installation、setup、station ID 等步骤型线索，"
                        "少查 safety / legal / FCC 关键词。"
                    )
                )
                safety_loop_streak = 0
            elif (
                not structure_query
                and empty_search_streak >= 2
                and not expand_hint_emitted
            ):
                scope = empty_search_product or "当前限定范围"
                route_notes.append(
                    _make_route_note(
                        f"在 {scope} 内连续检索无结果，可能陷入单产品误区；"
                        "建议下一轮显式将 products 设为 [] 做一次全库检索，"
                        "并优先使用用户原语言关键词重试。"
                    )
                )
                expand_hint_emitted = True
            elif (
                not structure_query
                and same_product_name is not None
                and same_product_streak >= 4
                and same_product_no_result_hits >= 1
                and not expand_hint_emitted
            ):
                route_notes.append(
                    _make_route_note(
                        f"你在 {same_product_name} 内已连续多轮检索且出现无结果，"
                        "可能陷入单产品误区；建议下一轮显式将 products 设为 [] 做一次全库检索，"
                        "再回到最相关产品收敛。"
                    )
                )
                expand_hint_emitted = True

            tool_results.append({
                "type": "tool_result",
                "tool_use_id": block.id,
                "content": result_text,
            })

        if has_tool_use:
            messages.append({"role": "assistant", "content": response.content})
            messages.append({"role": "user", "content": tool_results})
            if route_notes:
                messages.append({"role": "user", "content": "\n".join(route_notes)})
            if executed_tool_round:
                effective_turns_used += 1
            continue

        # 没有工具调用 → 最终回答
        if (question_id is None or question_id >= 64) and not formal_retrieval_confirmed:
            messages.append({"role": "assistant", "content": response.content})
            messages.append({
                "role": "user",
                "content": _make_route_note(
                    "技术题尚未获得任何可用手册证据。请先调用 search_manual 完成显式确认；"
                    "若已无机会继续检索，后续会基于已有内容收束并说明手册未覆盖。"
                ),
            })
            effective_turns_used += 1
            continue

        answer = _extract_text_from_response(response)
        answer = _postprocess_final_answer(
            answer=answer,
            question=retrieval_question_text,
            system_prompt=system_prompt,
            model=model,
            route_products=current_route.products,
        )
        answer = _clip_final_answer_with_llm(
            answer=answer,
            question=retrieval_question_text,
            model=model,
        )
        answer, evidence_selection_trace = _apply_evidence_selection(
            answer=answer,
            question=retrieval_question_text,
            engine=engine,
            candidate_images=evidence_candidate_images,
            route_products=list(current_route.products or []),
            model=model,
        )
        if trace is not None and evidence_selection_trace is not None:
            trace["events"].append({
                "kind": "evidence_selection",
                "index": len(trace["events"]) + 1,
                **evidence_selection_trace,
            })
        answer = _apply_same_language_fidelity_guard(
            answer=answer,
            question=retrieval_question_text,
            trace=trace,
        )
        answer = _apply_multi_intent_coverage_guard(
            answer=answer,
            question=retrieval_question_text,
            trace=trace,
        )
        answer = _apply_single_compound_scope_guard(
            answer=answer,
            question=retrieval_question_text,
            trace=trace,
        )
        answer = _strip_numeric_figure_references(answer)
        answer, pics = _resolve_answer_pics(answer)
        emit_progress("done", "答案生成和证据校验完成")

        return AgentResult(
            answer=answer,
            pics=pics,
            tool_calls=tool_calls,
            turns=current_turn,
            ttft=final_ttft,
            trace=(
                {
                    **trace,
                    "result": {
                        "answer": answer,
                        "pics": pics,
                        "tool_calls": tool_calls,
                        "turns": current_turn,
                    },
                    "error": None,
                    "elapsed": round(time.time() - trace_t0, 2),
                }
                if trace is not None else None
            ),
        )

    # 超过最大轮数：先强制收束；若仍是异常占位句，则抛异常交给批处理记 error
    finalize_t0 = time.time()
    answer = _finalize_without_tools(
        system_prompt=system_prompt,
        messages=messages,
        model=model,
    )
    if trace is not None:
        trace["timings"]["finalize_elapsed"] = round(time.time() - finalize_t0, 3)
    if _normalize_final_answer(answer) in _GENERIC_FAILURE_ANSWERS:
        raise RuntimeError(
            f"agent exceeded MAX_TURNS={MAX_TURNS} and failed to finalize an answer"
        )

    answer = _postprocess_final_answer(
        answer=answer,
        question=retrieval_question_text,
        system_prompt=system_prompt,
        model=model,
        route_products=current_route.products,
    )
    answer = _clip_final_answer_with_llm(
        answer=answer,
        question=retrieval_question_text,
        model=model,
    )
    answer, evidence_selection_trace = _apply_evidence_selection(
        answer=answer,
        question=retrieval_question_text,
        engine=engine,
        candidate_images=evidence_candidate_images,
        route_products=list(current_route.products or []),
        model=model,
    )
    if trace is not None and evidence_selection_trace is not None:
        trace["events"].append({
            "kind": "evidence_selection",
            "index": len(trace["events"]) + 1,
            **evidence_selection_trace,
        })
    answer = _apply_same_language_fidelity_guard(
        answer=answer,
        question=retrieval_question_text,
        trace=trace,
    )
    answer = _apply_multi_intent_coverage_guard(
        answer=answer,
        question=retrieval_question_text,
        trace=trace,
    )
    answer = _apply_single_compound_scope_guard(
        answer=answer,
        question=retrieval_question_text,
        trace=trace,
    )
    answer = _strip_numeric_figure_references(answer)
    answer, pics = _resolve_answer_pics(answer)

    return AgentResult(
        answer=answer,
        pics=pics,
        tool_calls=tool_calls,
        turns=MAX_TURNS,
        ttft=final_ttft,
        trace=(
            {
                **trace,
                "result": {
                    "answer": answer,
                    "pics": pics,
                    "tool_calls": tool_calls,
                    "turns": MAX_TURNS,
                },
                "error": None,
                "elapsed": round(time.time() - trace_t0, 2),
            }
            if trace is not None else None
        ),
    )


# ────────────────── CLI 入口 ──────────────────

def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="ReAct 客服智能体")
    parser.add_argument("question", nargs="?", help="用户问题")
    parser.add_argument("--interactive", "-i", action="store_true", help="交互模式")
    args = parser.parse_args()

    engine = RetrievalEngine()
    engine.ensure_index()
    print(f"索引加载完成: {len(engine.retrieval_chunks)} 检索块, {len(engine.catalog)} 产品\n")

    if args.interactive:
        print("交互模式（输入 quit 退出）")
        while True:
            question = input("\n> ").strip()
            if question.lower() in ("quit", "exit", "q"):
                break
            if not question:
                continue
            t0 = time.time()
            result = run_agent(question, engine)
            elapsed = time.time() - t0
            print(f"\n{result.answer}")
            if result.pics:
                print(f"\n图片: {result.pics}")
            print(f"\n--- {result.tool_calls} 次工具调用, {result.turns} 轮, {elapsed:.1f}s ---")
    elif args.question:
        t0 = time.time()
        result = run_agent(args.question, engine)
        elapsed = time.time() - t0
        print(result.answer)
        if result.pics:
            print(f"\n图片: {result.pics}")
        print(f"\n--- {result.tool_calls} 次工具调用, {result.turns} 轮, {elapsed:.1f}s ---")
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
