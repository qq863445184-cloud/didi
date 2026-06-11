from time import time

from app.my_memory_system import (
    MemoryRecord,
    WorkingMemoryStore,
    combined_score,
    importance_factor,
    recency_factor,
)


def test_importance_factor_maps_to_expected_range():
    # 重要性 0 -> 下限 0.8；1 -> 0.8 + 0.4 = 1.2；0.5 -> 1.0
    assert importance_factor(0.0) == 0.8
    assert abs(importance_factor(1.0) - 1.2) < 1e-9
    assert abs(importance_factor(0.5) - 1.0) < 1e-9
    # 越界值被夹紧
    assert importance_factor(-5.0) == 0.8
    assert abs(importance_factor(9.0) - 1.2) < 1e-9


def test_recency_factor_halves_each_half_life():
    now = 10_000.0
    half_life = 1000.0
    # 刚发生：~1.0
    assert abs(recency_factor(now, now, half_life) - 1.0) < 1e-9
    # 过了一个半衰期：0.5
    assert abs(recency_factor(now - 1000.0, now, half_life) - 0.5) < 1e-9
    # 两个半衰期：0.25
    assert abs(recency_factor(now - 2000.0, now, half_life) - 0.25) < 1e-9
    # 缺时间信息：退化为 1.0
    assert recency_factor(None, now, half_life) == 1.0
    assert recency_factor(now, None, half_life) == 1.0


def test_combined_score_without_recency_is_base_times_importance():
    # recency_weight=0 时退化为 base × importance_factor
    score = combined_score(2.0, 1.0, recency_weight=0.0)
    assert abs(score - 2.0 * 1.2) < 1e-9


def test_combined_score_penalizes_older_memory():
    now = 10_000.0
    half_life = 1000.0
    fresh = combined_score(
        1.0, 0.5, created_at=now, now=now,
        half_life_seconds=half_life, recency_weight=0.5,
    )
    stale = combined_score(
        1.0, 0.5, created_at=now - 3000.0, now=now,
        half_life_seconds=half_life, recency_weight=0.5,
    )
    # 同样的基础分和重要性，越旧的最终分越低
    assert fresh > stale


def test_working_memory_ranks_recent_over_old_when_overlap_equal():
    store = WorkingMemoryStore(half_life_seconds=1000.0, recency_weight=0.5)
    now = time()

    # 两条内容关键词重叠相同、重要性相同，只有时间不同
    old = MemoryRecord(content="用户在讨论 Agent 记忆系统", importance=0.5)
    old.created_at = now - 5000.0  # 远早于现在
    recent = MemoryRecord(content="用户在讨论 Agent 记忆系统", importance=0.5)
    recent.created_at = now
    store.add(old)
    store.add(recent)

    results = store.search("Agent 记忆系统", limit=2)

    assert len(results) == 2
    # 近期那条应排在前面
    assert results[0].record.created_at == recent.created_at
    assert results[0].score > results[1].score


def test_working_memory_ranks_important_over_trivial_when_overlap_equal():
    store = WorkingMemoryStore(recency_weight=0.0)  # 关掉时间因子，单看重要性
    high = MemoryRecord(content="用户是后端负责人", importance=0.9)
    low = MemoryRecord(content="用户是后端负责人", importance=0.2)
    store.add(low)
    store.add(high)

    results = store.search("后端负责人", limit=2)

    assert results[0].record.importance == 0.9
    assert results[0].score > results[1].score
