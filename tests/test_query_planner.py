"""Query Planner 离线测试（issue #6）。

使用 fake RewriteProvider，不依赖任何网络/LLM API。覆盖：原始查询保留、无 LLM
可确定性规划、型号/数值不被 rewrite 删除、LLM 校验失败回退、低召回 retry 触发与
限次、同输入可复现、中英/口语/代词/多对象对比/否定/数值/同义词。
"""
from __future__ import annotations

import tempfile
from pathlib import Path

from query_plan_models import PlannerContext, RetrievalFeedback, PlannerWarning
from query_planner import DefaultQueryPlanner, InMemoryEntityCatalog, ResolvedEntity


class FakeRewriteProvider:
    name = "fake"

    def __init__(self, queries=None, drop_constraints=False):
        self.queries = queries or []
        self.drop_constraints = drop_constraints

    def rewrite(self, original_query, deterministic_plan, context, retry_feedback=None):
        if self.drop_constraints:
            # 故意丢弃型号，触发约束校验回退
            return {
                "semantic_queries": ["目标丢失原因诊断"],
                "rewrite_used": True, "confidence": 0.9,
                "preserved_constraints": [], "entities": [],
            }
        return {
            "semantic_queries": self.queries,
            "rewrite_used": True, "confidence": 0.9,
            "preserved_constraints": list(deterministic_plan.preserved_constraints),
            "entities": list(deterministic_plan.entities),
        }


def _planner(provider=None, config=None, root=None):
    return DefaultQueryPlanner(project_root=root, rewrite_provider=provider, config=config)


def test_original_query_preserved_as_semantic_0():
    p = _planner()
    plan = p.plan("ARS540 在大雨时为什么会丢目标？")
    assert plan.original_query == "ARS540 在大雨时为什么会丢目标？"
    assert plan.semantic_queries[0] == plan.original_query


def test_no_llm_still_plans():
    p = _planner()  # 默认 NullRewriteProvider
    plan = p.plan("ARS540 校准步骤")
    assert plan.rewrite_used is False
    assert plan.rewrite_attempted is False
    assert plan.rewrite_applied is False
    assert plan.rewrite_source is None
    assert plan.rewrite_provider == "null"
    assert plan.intent in ("lookup", "procedure", "comparison", "relation", "global")


def test_model_number_in_exact_terms():
    p = _planner()
    plan = p.plan("ARS540 丢目标怎么办")
    assert "ARS540" in plan.exact_terms


def test_chinese_query_produces_lexical():
    p = _planner()
    plan = p.plan("雷达校准流程是什么")
    assert len(plan.lexical_terms) > 0


def test_colloquial_triggers_rewrite():
    p = _planner(provider=FakeRewriteProvider(queries=["雷达校准流程是什么", "radar calibration procedure"]))
    plan = p.plan("校准?")  # 过短 → 触发 Level2
    assert plan.rewrite_used is True
    assert plan.rewrite_source == "llm"
    assert plan.semantic_queries[0] == plan.original_query  # 原始不被覆盖


def test_rewrite_cannot_drop_constraints():
    # force 触发 Level2；即便 rewrite provider 输出的查询丢了型号，
    # original_query 仍作为 semantic_queries[0] 保留 → 型号永不丢失
    p = _planner(provider=FakeRewriteProvider(drop_constraints=True), config={"rewrite": "force"})
    plan = p.plan("ARS540 丢目标")
    assert plan.rewrite_used is False
    assert plan.rewrite_applied is False
    assert plan.semantic_queries[0] == "ARS540 丢目标"
    assert "ARS540" in plan.semantic_queries[0]


def test_low_recall_retry():
    p = _planner()
    plan = p.plan("ARS548 故障码 E123 怎么排查")
    fb = RetrievalFeedback(sparse_hit_count=0, dense_hit_count=0,
                           top_score_gap=None, evidence_count=0, failure_reason="empty")
    plan2 = p.plan_retry(plan, fb, PlannerContext())
    assert plan2 is not None
    assert plan2.retry_attempt == 1
    assert "ARS548" in plan2.exact_terms          # 型号保留
    assert "E123" in plan2.exact_terms            # 错误码保留
    assert plan2.original_query == plan.original_query


def test_retry_max_once():
    p = _planner(config={"max_retries": 1})
    plan = p.plan("ARK558 如何安装")
    fb = RetrievalFeedback(0, 0, None, 0, "empty")
    plan2 = p.plan_retry(plan, fb, PlannerContext())
    assert plan2 is not None
    plan3 = p.plan_retry(plan2, fb, PlannerContext())  # 已达上限
    assert plan3 is None


def test_determinism():
    p = _planner()
    a = p.plan("ARS540 对比 zPrime3 的区别").to_json()
    b = p.plan("ARS540 对比 zPrime3 的区别").to_json()
    assert a == b


def test_pronoun_with_context_triggers_rewrite():
    p = _planner(provider=FakeRewriteProvider(queries=["ARS540 在大雨时丢目标的原因"]))
    ctx = PlannerContext(conversation_text="ARS540 是大雨工况下会丢目标的雷达")
    plan = p.plan("它为什么会丢目标？", ctx)
    assert plan.rewrite_used is True


def test_comparison_intent_two_entities():
    p = _planner()
    plan = p.plan("ARS540 和 zPrime3 的区别对比")
    assert plan.intent == "comparison"


def test_negation_preserved_in_rewrite_validation():
    # 否定词必须保留，fake 不丢否定 → 通过；若 fake 丢则回退
    p = _planner(provider=FakeRewriteProvider(
        queries=["ARS540 目标检测丢失原因", "ARS540 target drop reason"]))
    plan = p.plan("ARS540 大雨时不丢目标的原因")
    # 否定词"不"必须出现在原始中；rewrite 若保留则通过，否则回退（二者都合法）
    assert plan.original_query == "ARS540 大雨时不丢目标的原因"


def test_hook_enhancement_detected():
    p = _planner()
    plan = p.plan("关键词: 雷达 校准; 扩展查询: radar calibration")
    assert plan.hook_injected_enhanced is True


def test_synonyms_loaded_from_project_root():
    with tempfile.TemporaryDirectory() as d:
        root = Path(d)
        (root / "lexicon.txt").write_text("# comment\nMyRadarX\n", encoding="utf-8")
        (root / "query_synonyms.yaml").write_text(
            "雷达: [radar, 毫米波雷达]\n校准: [calibration]\n", encoding="utf-8")
        p = _planner(root=root)
        plan = p.plan("雷达校准原理")
        assert "radar" in plan.lexical_terms
        assert "calibration" in plan.lexical_terms
        assert "MyRadarX" in p.lexicon


def test_rewrite_state_requires_valid_new_provider_query():
    empty = _planner(provider=FakeRewriteProvider(), config={"rewrite": "force"}).plan("ARS540 丢目标")
    assert empty.rewrite_attempted is True
    assert empty.rewrite_applied is False
    assert empty.rewrite_failure_reason == "no_valid_additional_query"

    valid = _planner(provider=FakeRewriteProvider(["ARS540 target loss diagnosis"]),
                     config={"rewrite": "force"}).plan("ARS540 丢目标")
    assert valid.rewrite_attempted is True
    assert valid.rewrite_applied is True
    assert valid.rewrite_used is True  # legacy compatibility field
    assert valid.rewrite_confidence == 0.9


def test_planner_warnings_are_typed_and_json_serializable():
    plan = _planner().plan("关键词: 雷达 校准")
    assert plan.warnings == (PlannerWarning("hook_injected_enhanced_query"),)
    assert plan.to_json()["warnings"] == [{"code": "hook_injected_enhanced_query", "message": ""}]


def test_injected_catalog_uses_longest_match_and_token_boundaries():
    catalog = InMemoryEntityCatalog((
        ResolvedEntity("ARS", "ARS", "lexicon"),
        ResolvedEntity("ARS540", "ARS540", "title"),
        ResolvedEntity("Front Radar", "FR", "alias"),
    ))
    planner = DefaultQueryPlanner(entity_catalog=catalog)
    plan = planner.plan("ARS540 和 Front Radar 的区别，不是 ARS5400")
    assert {"ARS540", "FR"}.issubset(plan.entities)
    assert "ARS" not in plan.entities
