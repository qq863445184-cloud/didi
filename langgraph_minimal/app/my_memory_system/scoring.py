from __future__ import annotations

"""Unified scoring for all memory stores.

把四种记忆的打分统一到一个公式骨架，对齐第八章的设计：

    final = base_score × recency_blend × importance_factor × access_factor

- base_score      检索本身的相关性（关键词重叠 / 向量相似度 / 图命中）。
- importance_factor  重要的记忆更容易被想起：0.8 + 重要性 × 0.4，范围 [0.8, 1.2]。
- recency_blend   最近的记忆更鲜活：把时间近因按 recency_weight 混进基础分，
                  近因系数用指数半衰减 0.5 ** (age / half_life)。
- access_factor   被反复检索的记忆被强化：1 + access_gain × (1 - 1/(1+count))，
                  随访问次数饱和增长，模拟海马体重放，“常用的不会被忘”。

各层只需提供 base_score 和自己的 recency 配置，公式本身集中在这里，
避免每个 store 各写一套、口径不一致。
"""


def clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))


def importance_factor(importance: float, *, floor: float = 0.8, span: float = 0.4) -> float:
    """Map importance in [0,1] to a multiplier in [floor, floor+span]."""

    return floor + clamp01(importance) * span


def recency_factor(
    created_at: float | None,
    now: float | None,
    half_life_seconds: float,
) -> float:
    """Exponential time decay in (0, 1]; 1.0 when timing info is missing.

    half_life_seconds 是“记忆鲜活度减半”的时间：经过一个半衰期，
    近因系数从 1.0 衰减到 0.5，再一个半衰期到 0.25，以此类推。
    """

    if not created_at or not now or half_life_seconds <= 0:
        return 1.0
    age = now - created_at
    if age <= 0:
        return 1.0
    return 0.5 ** (age / half_life_seconds)


def access_factor(access_count: int, *, gain: float = 0.5) -> float:
    """Saturating boost for frequently retrieved memories; >= 1.0.

    用 1 + gain × (1 - 1/(1+count)) 做饱和增长：
    - count=0  -> 1.0（从未命中，不加成，向后兼容）
    - count=1  -> 1 + gain×0.5
    - count→∞  -> 1 + gain（上限封顶，避免热门记忆无限霸榜）
    """

    if access_count <= 0 or gain <= 0:
        return 1.0
    return 1.0 + gain * (1.0 - 1.0 / (1.0 + access_count))


def combined_score(
    base_score: float,
    importance: float,
    *,
    created_at: float | None = None,
    now: float | None = None,
    half_life_seconds: float = 0.0,
    recency_weight: float = 0.0,
    access_count: int = 0,
    access_gain: float = 0.5,
) -> float:
    """Combine relevance, recency, importance and access into one score.

    recency_weight=0 时退化为不考虑时间；access_count=0 时退化为不考虑访问，
    因此旧调用方（不传这两个参数）行为完全不变。
    """

    score = base_score
    if recency_weight > 0 and half_life_seconds > 0:
        recency = recency_factor(created_at, now, half_life_seconds)
        blend = (1.0 - recency_weight) + recency_weight * recency
        score = score * blend
    return score * importance_factor(importance) * access_factor(
        access_count, gain=access_gain
    )
