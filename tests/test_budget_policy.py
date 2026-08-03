"""预算契约测试（issue #14 follow-up：ContextBundle budget contract）。

契约：

    effective = min(base × intent multiplier, hard_max_tokens if supplied)
    max_context_tokens == effective_budget_tokens
    token_count        <= effective_budget_tokens

预算由 ``query.hybrid_search()`` 唯一决定并通过 Bundle 暴露；下游（eval / 宿主）
只能读 Bundle 字段，不得复制 ``_CONTEXT_MODE_MAP``。
"""
from pathlib import Path

import pytest

from models import ChunkHit, ContextBundle
from query import (
    BUDGET_POLICY, hybrid_search, resolve_budget, result_to_json,
)
from query_planner import DefaultQueryPlanner


COMPARISON_QUERY = "对比 Columbus Front Radar 与 Corner Radar 的探测距离区别"


def _planner(root):
    return DefaultQueryPlanner(project_root=root)


# ---------------------------------------------------------------------------
# 1) 纯策略解析（无 IO）
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("context_mode,expected_mult", [
    ("multiple_sections", 1.4),   # comparison
    ("section", 1.0),             # lookup
    ("parent_section", 1.0),      # procedure
    ("evidence", 1.0),            # relation
    ("global", 1.0),
    ("unknown_mode", 1.0),        # 未知 mode 退化为 1.0
])
def test_resolve_budget_multipliers(context_mode, expected_mult):
    mode, mult, planned, effective = resolve_budget(context_mode, 4096)
    assert mult == expected_mult
    assert planned == int(4096 * expected_mult)
    assert effective == planned          # 未提供 hard cap → effective == planned
    assert mode in {"snippet", "summary", "full"}


def test_resolve_budget_comparison_is_5734():
    _, mult, planned, effective = resolve_budget("multiple_sections", 4096)
    assert (mult, planned, effective) == (1.4, 5734, 5734)


@pytest.mark.parametrize("override", ["full", "snippet", "summary"])
def test_mode_override_forces_multiplier_one(override):
    mode, mult, planned, effective = resolve_budget(
        "multiple_sections", 4096, mode_override=override)
    assert mode == override
    assert mult == 1.0
    assert planned == effective == 4096


def test_hard_cap_clamps_planned_budget():
    _, mult, planned, effective = resolve_budget(
        "multiple_sections", 4096, hard_max_tokens=4096)
    assert (mult, planned, effective) == (1.4, 5734, 4096)


def test_hard_cap_above_planned_is_noop():
    _, _, planned, effective = resolve_budget(
        "multiple_sections", 4096, hard_max_tokens=100000)
    assert planned == effective == 5734


# ---------------------------------------------------------------------------
# 2) Bundle 数据契约
# ---------------------------------------------------------------------------

def test_legacy_construction_keeps_invariant():
    """只传 max_context_tokens 的老调用点也满足 effective == max_context_tokens。"""
    b = ContextBundle(query="q", mode="snippet", max_context_tokens=200, token_count=180)
    assert b.effective_budget_tokens == 200
    assert b.requested_base_budget_tokens == 200
    assert b.budget_multiplier == 1.0
    assert b.budget_contract_violations() == []


def test_within_expanded_budget_is_not_a_violation():
    """token_count 超过基础预算但不超 effective → 合法，不算 overflow。"""
    b = ContextBundle(query="q", mode="snippet", token_count=5000)
    b.apply_budget(base_tokens=4096, multiplier=1.4, effective_tokens=5734)
    assert b.max_context_tokens == 5734
    assert b.token_count > b.requested_base_budget_tokens
    assert b.budget_contract_violations() == []


def test_over_effective_budget_is_reported():
    b = ContextBundle(query="q", mode="snippet", token_count=6000)
    b.apply_budget(base_tokens=4096, multiplier=1.4, effective_tokens=5734)
    assert any("token_count" in v for v in b.budget_contract_violations())


def test_effective_over_hard_cap_is_reported():
    b = ContextBundle(query="q", mode="snippet", token_count=10)
    b.apply_budget(base_tokens=4096, multiplier=1.4, effective_tokens=5734,
                   hard_max_tokens=4096)
    assert any("hard_max_tokens" in v for v in b.budget_contract_violations())


def test_max_context_tokens_drift_is_reported():
    b = ContextBundle(query="q", mode="snippet", token_count=10)
    b.apply_budget(base_tokens=4096, multiplier=1.4, effective_tokens=5734)
    b.max_context_tokens = 4096  # 模拟契约漂移
    assert any("max_context_tokens" in v for v in b.budget_contract_violations())


# ---------------------------------------------------------------------------
# 3) 端到端：hybrid_search 分配预算 + JSON 暴露（**不依赖 embedding 模型**）
#
#    用一个内存假 repository 跑通整条 hybrid_search 链路，验证预算字段与
#    #14 契约（多 section / 双路 evidence / 引用字段）在扩展预算下仍保留。
#    假对象的 repository 端口全部用 getattr(..., fallback) 容错路径，因此
#    multiple_sections 在缺少 get_parent_section 时自动回退到 evidence 文本。
# ---------------------------------------------------------------------------

_PAGE_BODIES = {
    "cfr100": ("Columbus Front Radar CFR-100 探测距离 250 m，频段 77 GHz，接口 CAN FD。"
               "该雷达用于前向探测，支持 AEB 与 ACC 功能。") * 200,
    "ccr100": ("Columbus Corner Radar CCR-100 探测距离 150 m，视场角 ±75°，接口 CAN FD。"
               "该雷达用于角雷达，支持 BSD 与 LCA 功能。") * 200,
}


class _FakeWiki:
    """Minimal in-memory stand-in for WikiIndex — no embeddings, no index."""

    def __init__(self):
        self.index_dir = Path("/tmp/fake_index")

    @staticmethod
    def count_tokens(text):
        return max(1, len(text) // 4)

    def _hits(self):
        hits = []
        for page_id, body in _PAGE_BODIES.items():
            for i in range(2):
                hits.append(ChunkHit(
                    chunk_id=f"{page_id}::{i}", page_id=page_id,
                    path=f"Wiki/{page_id}.md", title=page_id,
                    page_type="specs", section_path=["root", page_id],
                    heading=page_id, chunk_kind="dense", text=body,
                    channel="fts", score=1.0 - i * 0.1))
        return hits

    def search_fts_terms(self, lexical_terms, exact_terms, k=20):
        return self._hits()

    def search_vector(self, semantic_query, k=20):
        return self._hits()

    def get_page_sources(self, page_id):
        return [f"raw/{page_id}.docx"]

    def get_image_meta(self, path):
        return None

    def get_parent_section(self, chunk_id):
        return []   # → assemble_context 回退到 evidence 文本（multiple_sections）

    def get_chunk(self, chunk_id):
        return None

    def get_neighbors(self, chunk_id):
        return []

    def read_page(self, page_id):
        return _PAGE_BODIES.get(page_id, "")


def _planner_none():
    return DefaultQueryPlanner()

def test_comparison_expands_budget_end_to_end():
    wi = _FakeWiki()
    res = hybrid_search(wi, COMPARISON_QUERY, _planner_none(), k=5, max_tokens=4096)
    b = res.bundle
    assert res.plan.intent == "comparison"
    assert res.plan.context_mode == "multiple_sections"
    assert b.requested_base_budget_tokens == 4096
    assert b.budget_multiplier == 1.4
    assert b.effective_budget_tokens == 5734
    assert b.max_context_tokens == 5734
    assert b.hard_max_tokens is None
    assert b.budget_policy == BUDGET_POLICY
    assert b.budget_contract_violations() == []
    # #14 契约不回退：多 section 取材 + 双路 evidence/引用字段保留
    assert b.items, "comparison 查询应有上下文条目"
    assert all(it.scope == "multiple_sections" for it in b.items)
    assert any(it.evidence for it in b.items)
    assert any(it.sources for it in b.items)


@pytest.mark.parametrize("query,intent", [
    ("Columbus Front Radar 探测距离是多少", "lookup"),
    ("如何校准 Columbus Front Radar", "procedure"),
    ("为什么 Corner Radar 视场角更大", "relation"),
])
def test_non_comparison_intents_keep_base_budget(query, intent):
    wi = _FakeWiki()
    res = hybrid_search(wi, query, _planner_none(), k=5, max_tokens=4096)
    b = res.bundle
    assert res.plan.intent == intent
    assert b.budget_multiplier == 1.0
    assert b.effective_budget_tokens == b.requested_base_budget_tokens == 4096
    assert b.max_context_tokens == 4096
    assert b.budget_contract_violations() == []


@pytest.mark.parametrize("override", ["full", "snippet"])
def test_explicit_mode_override_disables_expansion(override):
    wi = _FakeWiki()
    res = hybrid_search(wi, COMPARISON_QUERY, _planner_none(), k=5, max_tokens=4096,
                        mode_override=override)
    b = res.bundle
    assert b.mode == override
    assert b.budget_multiplier == 1.0
    assert b.effective_budget_tokens == 4096
    assert b.budget_contract_violations() == []


def test_hard_cap_binds_comparison():
    """--hard-max-tokens 4096：仍按 multiple_sections 取材，但上限被压回 4096。"""
    wi = _FakeWiki()
    res = hybrid_search(wi, COMPARISON_QUERY, _planner_none(), k=5, max_tokens=4096,
                        hard_max_tokens=4096)
    b = res.bundle
    assert res.plan.context_mode == "multiple_sections"
    assert all(it.scope == "multiple_sections" for it in b.items)
    assert b.budget_multiplier == 1.4          # 策略仍生效
    assert b.effective_budget_tokens == 4096   # 但被硬上限压回
    assert b.max_context_tokens == 4096
    assert b.hard_max_tokens == 4096
    assert b.token_count <= 4096
    assert b.budget_contract_violations() == []


def test_hard_cap_truncation_is_explicit():
    """硬上限极小 → 必须显式标注截断/省略，而不是静默丢内容。"""
    wi = _FakeWiki()
    res = hybrid_search(wi, COMPARISON_QUERY, _planner_none(), k=5, max_tokens=4096,
                        hard_max_tokens=8)
    b = res.bundle
    assert b.effective_budget_tokens == 8
    assert b.token_count <= 8
    assert b.budget_contract_violations() == []
    marked = any(it.truncated and it.truncation_reason for it in b.items) or b.omitted_items
    assert marked, "预算被硬上限压缩后必须显式标注截断或省略"


def test_json_budget_fields_match_bundle():
    wi = _FakeWiki()
    res = hybrid_search(wi, COMPARISON_QUERY, _planner_none(), k=5, max_tokens=4096)
    payload = result_to_json(res)
    b = res.bundle
    assert payload["requested_base_budget_tokens"] == b.requested_base_budget_tokens == 4096
    assert payload["budget_multiplier"] == b.budget_multiplier == 1.4
    assert payload["effective_budget_tokens"] == b.effective_budget_tokens == 5734
    assert payload["hard_max_tokens"] is None
    assert payload["budget_policy"] == BUDGET_POLICY
    assert payload["max_context_tokens"] == b.max_context_tokens == 5734
    assert payload["token_count"] == b.token_count


def test_cli_exposes_hard_max_tokens_flag():
    from query import _build_arg_parser
    args = _build_arg_parser().parse_args(["root", "q", "--hard-max-tokens", "4096"])
    assert args.hard_max_tokens == 4096
    assert _build_arg_parser().parse_args(["root", "q"]).hard_max_tokens is None


# ---------------------------------------------------------------------------
# 4) eval 不得复制预算规则
# ---------------------------------------------------------------------------

def test_eval_does_not_duplicate_budget_policy():
    src = (Path(__file__).resolve().parent.parent / "eval" / "run_eval.py").read_text(
        encoding="utf-8")
    assert "_CONTEXT_MODE_MAP =" not in src, "eval 不得复制预算倍率表"
    assert "import _CONTEXT_MODE_MAP" not in src
    assert "effective_budget_tokens" in src, "eval 必须按 effective budget 判定 overflow"
    assert "budget_contract_violations" in src, "eval 必须做契约漂移检查"


def test_global_reports_recount_selected_text_before_hard_budget():
    """Stored report counts are evidence only; query selection recounts actual text."""
    from obsidian_wiki.application.community_report_service import CommunityReportService
    from obsidian_wiki.domain.community_report_models import CommunityReportStatus

    class NamedCounter:
        identity = "named-budget-counter/v1"

        def __init__(self):
            self.calls = 0

        def count(self, text):
            self.calls += 1
            # The fake deliberately has a stable, named contract: build,
            # artifact validation, and query-time selection must all measure
            # the same report text.  The call count proves selection still
            # invokes the counter instead of trusting stored token_count.
            return len(text.split())

    class Store:
        active = None

        def stage(self, build_id, reports, manifest):
            self.staged = (tuple(reports), manifest)

        def read_staged(self, build_id):
            return self.staged

        def activate(self, build_id):
            self.active = self.staged

        def read_active(self):
            return self.active

    from obsidian_wiki.domain.community_report_models import GraphEdge, GraphSnapshotState, PageSnapshot

    class Graph:
        def read(self):
            return GraphSnapshotState(
                pages=(
                    PageSnapshot("Wiki/a.md", "a-hash"),
                    PageSnapshot("Wiki/b.md", "b-hash"),
                ),
                edges=(GraphEdge("Wiki/a.md", "Wiki/b.md", ("related",), 1.0),),
                communities=((7, ("Wiki/a.md", "Wiki/b.md")),),
            )

    counter = NamedCounter()
    service = CommunityReportService(Store(), Graph(), counter)
    service.build()
    outcome = service.retrieve(query_terms=("community",), k=1, max_tokens=3)

    assert outcome.status is CommunityReportStatus.FRESH
    assert outcome.reports == ()
    assert outcome.stale_reasons == ("selected community reports exceed the effective token budget",)
    # build + staged validation + query validation + selection recount
    assert counter.calls == 4
