"""Query Planner 实现（GitHub issue #6）。

独立、低耦合模块：**不导入** LanceDB / NetworkX / RRF / ContextBundle，只产出
``QueryPlan`` 数据契约（定义于 ``query_plan_models``）。

两级规划：
- Level 1（必执行，确定性）：规范化 + 型号/错误码/路径/数字单位提取 +
  复用 #2 ``lexical_tokenizer`` + 意图识别 + 默认 filters/context_mode。
  原始查询始终保留为 ``semantic_queries[0]``。
- Level 2（条件式 LLM rewrite）：仅当指代/口语/多对象/意图不确定/低召回时触发；
  通过 ``RewriteProvider`` Protocol 注入，默认 ``NullRewriteProvider``（无 LLM 可用）；
  约束校验失败 / 超时 / 异常自动回退 Level 1；最多 2 条额外语义查询。
"""
from __future__ import annotations

import logging
import re
import threading
import unicodedata
from dataclasses import replace
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from lexical_tokenizer import fts_terms, extract_exact_terms, load_lexicon
from query_plan_models import (
    QueryIntent,
    QueryPlan,
    PlannerContext,
    RetrievalFeedback,
    NullRewriteProvider,
)

logger = logging.getLogger(__name__)

PLANNER_SCHEMA_VERSION = "qp-1"

# 意图关键词（按优先级顺序匹配）
_INTENT_KEYWORDS: List[Tuple[QueryIntent, List[str]]] = [
    (QueryIntent.GLOBAL, [r"全局", r"概述", r"总结.*所有", r"一共有多少", r"总体", r"报告", r"全景", r"全部"]),
    (QueryIntent.COMPARISON, [r"对比", r"比较", r"\bvs\b", r"versus", r"区别", r"差异", r"不同", r"优劣", r"对比"]),
    (QueryIntent.RELATION, [r"为什么", r"为何", r"如何.*影响", r"导致", r"关联", r"依赖", r"关系", r"引起", r"机理", r"根因"]),
    (QueryIntent.PROCEDURE, [r"如何", r"怎么", r"步骤", r"安装", r"校准", r"配置", r"流程", r"方法", r"怎样", r"设置", r"操作"]),
]

_CONTEXT_MODE_MAP = {
    QueryIntent.LOOKUP: "section",
    QueryIntent.PROCEDURE: "parent_section",
    QueryIntent.COMPARISON: "multiple_sections",
    QueryIntent.RELATION: "evidence",
    QueryIntent.GLOBAL: "global",
}

_DEFAULT_FILTERS = {
    QueryIntent.LOOKUP: {},
    QueryIntent.PROCEDURE: {},
    QueryIntent.COMPARISON: {},
    QueryIntent.RELATION: {},
    QueryIntent.GLOBAL: {},
}

_PRONOUN_RE = re.compile(r"(它|这个|那个|上一个|上一个|前面|之前|其|该|前者|后者|这东西|那东西|上述)")
_RELAX_WORDS = [
    r"大雨(中|环境|下|时)?", r"小雨(中|环境|下|时)?", r"严格(地|的)?", r"仅仅",
    r"只(是)?", r"必须", r"一定(要)?", r"明显", r"大概", r"可能",
]
_RELAX_RE = re.compile("|".join(f"(?:{w})" for w in _RELAX_WORDS))
_HOOK_ENHANCE_MARKERS = re.compile(r"(关键词[:：]|关键词提取|扩展查询|中英互译|增强查询|query\s*[:：])", re.IGNORECASE)
_NUMERIC_UNIT_RE = re.compile(r"\d+(?:\.\d+)?\s*(?:GHz|MHz|kHz|Hz|V|A|W|mm|cm|m|km|kg|°C|℃|%|ms|us|µs|ns|dB|MB|GB)?")
_NEGATION_RE = re.compile(r"(不|没|无|非|禁止|排除|避免|不要|未)")


class DefaultQueryPlanner:
    """issue #6 的默认 Query Planner 实现。"""

    def __init__(self, project_root: Optional[Path] = None,
                 rewrite_provider: Optional[Any] = None,
                 config: Optional[Dict[str, Any]] = None):
        self.project_root = Path(project_root) if project_root else None
        self.rewrite_provider = rewrite_provider or NullRewriteProvider()
        self.config: Dict[str, Any] = {
            "rewrite": _env("QUERY_PLANNER_REWRITE", "auto"),
            "max_semantic_queries": int(_env("QUERY_PLANNER_MAX_SEMANTIC_QUERIES", "3")),
            "max_retries": int(_env("QUERY_PLANNER_MAX_RETRIES", "1")),
            "rewrite_timeout": float(_env("QUERY_PLANNER_REWRITE_TIMEOUT_SECONDS", "8")),
            "min_query_chars": int(_env("QUERY_PLANNER_MIN_QUERY_CHARS", "6")),
        }
        if config:
            self.config.update(config)
        self.lexicon = load_lexicon(self.project_root) if self.project_root else set()
        self.synonyms = self._load_synonyms(self.project_root) if self.project_root else {}
        self._lexicon_hash = _hash_file(self.project_root / "lexicon.txt") if self.project_root else ""
        self._tokenizer_hash = _hash_str("fts_terms-v1|extract_exact-v1")

    # ---------- 配置 / 资源 ----------
    def _load_synonyms(self, root: Path) -> Dict[str, List[str]]:
        """极简解析项目级 query_synonyms（支持 `term: [a, b]` 或 `term: a, b`）。"""
        f = root / "query_synonyms.yaml"
        out: Dict[str, List[str]] = {}
        if not f.exists():
            return out
        try:
            for line in f.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line or line.startswith("#") or ":" not in line:
                    continue
                key, val = line.split(":", 1)
                key = key.strip()
                val = val.strip().strip("[]")
                alts = [a.strip().strip('"').strip("'") for a in val.split(",") if a.strip()]
                if key and alts:
                    out[key] = alts
        except Exception as e:  # noqa: BLE001
            logger.warning("query_synonyms 解析失败: %s", e)
        return out

    # ---------- Level 1：确定性规划 ----------
    def _normalize(self, q: str) -> str:
        q = unicodedata.normalize("NFKC", q)
        q = q.replace(" ", " ")
        q = re.sub(r"\s+", " ", q).strip()
        return q

    def _extract_exact(self, q: str) -> List[str]:
        terms = list(extract_exact_terms(q))
        terms += re.findall(r"--[a-zA-Z][\w-]+", q)  # CLI 参数也属硬约束
        seen, out = set(), []
        for t in terms:
            if t not in seen:
                seen.add(t)
                out.append(t)
        return out

    def _expand_synonyms(self, terms: List[str]) -> List[str]:
        out = list(terms)
        for t in terms:
            for alt in self.synonyms.get(t, []):
                if alt not in out:
                    out.append(alt)
            for k, vs in self.synonyms.items():  # 反向：alt→key
                if t in vs and k not in out:
                    out.append(k)
        return out

    def _detect_intent(self, q: str, entities: List[str]) -> Tuple[str, str]:
        for intent, pats in _INTENT_KEYWORDS:
            for p in pats:
                if re.search(p, q, re.IGNORECASE):
                    return intent.value, f"命中关键词/{p}"
        if len(entities) >= 2:
            return QueryIntent.COMPARISON.value, "多实体→comparison"
        return QueryIntent.LOOKUP.value, "默认 lookup"

    def _relation_intent(self, q: str, intent: str) -> Optional[str]:
        if intent == QueryIntent.RELATION.value:
            return "cause_or_influence"
        if re.search(r"(导致|关联|依赖|关系|引起)", q):
            return "cause_or_influence"
        return None

    def _should_rewrite_l2(self, normalized: str, intent: str, plan: QueryPlan,
                           ctx: PlannerContext) -> bool:
        cfg = self.config["rewrite"]
        if cfg == "off":
            return False
        if cfg == "force":
            return True
        # auto：仅在确有必要时触发，保证稳定/可复现
        if _PRONOUN_RE.search(normalized) and ctx.conversation_text:
            return True
        if len(normalized) < self.config["min_query_chars"] or not plan.exact_terms:
            return True
        if intent in (QueryIntent.COMPARISON.value, QueryIntent.RELATION.value) \
                and len(plan.entities) >= 2:
            return True
        return False

    def plan(self, query: str, context: Optional[PlannerContext] = None) -> QueryPlan:
        ctx = context or PlannerContext()
        original = query
        normalized = self._normalize(query)
        exact = self._extract_exact(normalized)
        lexical = self._expand_synonyms(list(fts_terms(normalized, self.lexicon)))
        entities = list(dict.fromkeys(list(exact) + list(ctx.known_entities)))
        intent, reason = self._detect_intent(normalized, entities)
        relation_intent = self._relation_intent(normalized, intent)
        context_mode = _CONTEXT_MODE_MAP[QueryIntent(intent)]
        filters = dict(_DEFAULT_FILTERS[QueryIntent(intent)])
        if ctx.page_types:
            filters["page_type"] = list(ctx.page_types)

        hook_enh = _HOOK_ENHANCE_MARKERS.search(original) is not None
        warnings: Tuple[str, ...] = ()
        if hook_enh:
            warnings = (("hook_injected_enhanced_query",),)

        plan = QueryPlan(
            original_query=original,
            normalized_query=normalized,
            intent=intent,
            routing_reason=reason,
            semantic_queries=(original,),
            lexical_terms=tuple(lexical),
            exact_terms=tuple(exact),
            entities=tuple(entities),
            relation_intent=relation_intent,
            filters=filters,
            context_mode=context_mode,
            rewrite_used=False,
            rewrite_provider="null",
            rewrite_confidence=1.0,
            preserved_constraints=tuple(exact),
            warnings=warnings,
            planner_schema_version=PLANNER_SCHEMA_VERSION,
            tokenizer_hash=self._tokenizer_hash,
            lexicon_hash=self._lexicon_hash,
            hook_injected_enhanced=hook_enh,
        )
        if self._should_rewrite_l2(normalized, intent, plan, ctx):
            plan = self._apply_rewrite(plan, ctx, None)
        return plan

    # ---------- Level 2：条件式 LLM rewrite ----------
    def _call_with_timeout(self, fn, *args):
        holder: Dict[str, Any] = {}

        def _run():
            try:
                holder["v"] = fn(*args)
            except Exception as e:  # noqa: BLE001
                holder["e"] = e

        t = threading.Thread(target=_run, daemon=True)
        t.start()
        t.join(self.config["rewrite_timeout"])
        if "e" in holder:
            raise holder["e"]
        if t.is_alive():
            raise TimeoutError("rewrite provider timed out")
        return holder.get("v")

    def _constraints_preserved(self, original: str, queries: List[str]):
        combined = " ".join(queries)
        protected = set(self._extract_exact(original))
        protected |= set(_NUMERIC_UNIT_RE.findall(original))
        missing = [t for t in protected if t and t not in combined]
        neg = _NEGATION_RE.findall(original)
        neg_missing = [n for n in set(neg) if n not in combined]
        ok = (not missing) and (not neg_missing)
        return ok, missing, neg_missing

    def _apply_rewrite(self, plan: QueryPlan, ctx: PlannerContext,
                       retry_feedback: Optional[RetrievalFeedback]) -> QueryPlan:
        provider = self.rewrite_provider
        try:
            result = self._call_with_timeout(
                provider.rewrite, plan.original_query, plan, ctx, retry_feedback)
        except Exception as e:  # noqa: BLE001
            return self._fallback(plan, f"rewrite exception: {e}")
        if not isinstance(result, dict):
            return self._fallback(plan, "rewrite returned non-dict")

        extra = list(result.get("semantic_queries") or [])
        combined = [plan.original_query] + extra
        ok, missing, neg_missing = self._constraints_preserved(plan.original_query, combined)
        if not ok:
            return self._fallback(plan, f"constraint loss: missing={missing} neg_missing={neg_missing}")

        maxq = self.config["max_semantic_queries"]
        chosen = [plan.original_query] + extra[: max(0, maxq - 1)]
        semantic = tuple(dict.fromkeys(chosen))

        entities = list(plan.entities)
        if result.get("entities"):
            entities = list(dict.fromkeys(entities + list(result["entities"])))
        relation_intent = result.get("relation_intent", plan.relation_intent)

        return replace(
            plan,
            semantic_queries=semantic,
            rewrite_used=True,
            rewrite_provider=getattr(provider, "name", type(provider).__name__),
            rewrite_confidence=float(result.get("confidence", 0.8)),
            entities=tuple(entities),
            relation_intent=relation_intent,
            preserved_constraints=tuple(
                result.get("preserved_constraints") or list(plan.preserved_constraints)),
            rewrite_source=("retry" if retry_feedback is not None else "llm"),
            warnings=plan.warnings + (("rewrite_applied",) if not plan.rewrite_used else ()),
        )

    def _fallback(self, plan: QueryPlan, reason: str) -> QueryPlan:
        return replace(plan, warnings=plan.warnings + ((f"rewrite_fallback: {reason}",)))

    # ---------- 低召回重试 ----------
    def plan_retry(self, previous: QueryPlan, feedback: RetrievalFeedback,
                   context: Optional[PlannerContext] = None) -> Optional[QueryPlan]:
        ctx = context or PlannerContext()
        if previous.retry_attempt >= self.config["max_retries"]:
            return None

        original = previous.original_query
        # 缩写展开 / 同义上位 / 中英互译
        extra_terms: List[str] = []
        for t in list(previous.lexical_terms) + list(previous.exact_terms):
            extra_terms.extend(self.synonyms.get(t, []))
            for k, vs in self.synonyms.items():
                if t in vs and k not in extra_terms:
                    extra_terms.append(k)

        # 去除可能过强但非用户硬约束的检索词（不触碰 exact_terms / 型号 / 数值）
        relaxed = _RELAX_RE.sub(" ", previous.normalized_query)
        relaxed = re.sub(r"\s+", " ", relaxed).strip()

        # 补充从 conversation context 已解析出的实体（保留原始 exact_terms）
        entities = list(dict.fromkeys(list(previous.entities) + list(ctx.known_entities)))
        general_sq = relaxed if relaxed and relaxed != original else original
        semantic = tuple(dict.fromkeys(
            [original, general_sq] + [s for s in previous.semantic_queries if s != original]))

        lexical = list(dict.fromkeys(list(previous.lexical_terms) + extra_terms))
        return replace(
            previous,
            semantic_queries=semantic[: self.config["max_semantic_queries"]],
            lexical_terms=tuple(lexical),
            exact_terms=tuple(previous.exact_terms),  # 硬约束保留
            entities=tuple(entities),
            retry_attempt=previous.retry_attempt + 1,
            rewrite_used=previous.rewrite_used,
            rewrite_source="retry",
            routing_reason=previous.routing_reason + "|retry",
            warnings=previous.warnings + (("low_recall_retry",),),
        )


def _env(name: str, default: str) -> str:
    import os
    return os.environ.get(name, default)


def _hash_file(p: Path) -> str:
    try:
        import hashlib
        return hashlib.sha256(Path(p).read_bytes()).hexdigest()[:12]
    except Exception:  # noqa: BLE001
        return ""


def _hash_str(s: str) -> str:
    import hashlib
    return hashlib.sha256(s.encode("utf-8")).hexdigest()[:12]
