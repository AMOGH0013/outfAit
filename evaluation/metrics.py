from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import math

from app.services.outfit_engine import cosine_similarity


@dataclass(frozen=True)
class ModeMetrics:
    mean_score: float
    diversity: float
    repetition_rate: float
    color_entropy: float
    wear_through: float


def _shannon_entropy(counts: Mapping[str, int]) -> float:
    total = sum(int(v) for v in counts.values())
    if total <= 0:
        return 0.0
    h = 0.0
    for v in counts.values():
        p = float(v) / float(total)
        if p <= 1e-12:
            continue
        h -= p * math.log(p, 2)
    return round(h, 4)


def _l2_normalize(vec: list[float]) -> list[float]:
    norm = math.sqrt(sum(float(x) * float(x) for x in vec))
    if norm <= 1e-12:
        return vec
    return [float(x) / norm for x in vec]


def _outfit_embedding(
    item_ids: Sequence[str],
    embedding_by_item_id: Mapping[str, Sequence[float] | None],
) -> list[float] | None:
    embs = []
    for item_id in item_ids:
        e = embedding_by_item_id.get(item_id)
        if e and isinstance(e, (list, tuple)) and len(e) > 0:
            embs.append([float(x) for x in e])
    if not embs:
        return None
    dim = len(embs[0])
    if any(len(e) != dim for e in embs):
        return None
    avg = [0.0 for _ in range(dim)]
    for e in embs:
        for i, x in enumerate(e):
            avg[i] += x
    avg = [x / float(len(embs)) for x in avg]
    return _l2_normalize(avg)


def compute_metrics(
    *,
    day_rows: list[dict],
    embedding_by_item_id: Mapping[str, Sequence[float] | None],
    color_by_item_id: Mapping[str, str | None],
) -> ModeMetrics:
    if not day_rows:
        return ModeMetrics(mean_score=0.0, diversity=0.0, repetition_rate=0.0, color_entropy=0.0, wear_through=0.0)

    # mean_score
    scores = [float(r.get("final_score") or 0.0) for r in day_rows]
    mean_score = sum(scores) / float(len(scores))

    # wear_through
    worn = sum(1 for r in day_rows if (r.get("feedback") or "").strip().lower() == "worn")
    wear_through = worn / float(len(day_rows))

    # repetition_rate
    uses: list[str] = []
    for r in day_rows:
        for iid in (r.get("item_ids") or []):
            uses.append(str(iid))
    total_uses = len(uses)
    unique_uses = len(set(uses))
    repetition_rate = 0.0 if total_uses == 0 else (1.0 - (unique_uses / float(total_uses)))

    # diversity: average embedding distance between consecutive days (1 - cosine similarity)
    outfit_vecs: list[list[float] | None] = []
    for r in day_rows:
        item_ids = [str(x) for x in (r.get("item_ids") or [])]
        outfit_vecs.append(_outfit_embedding(item_ids, embedding_by_item_id))
    distances = []
    for prev, cur in zip(outfit_vecs, outfit_vecs[1:]):
        if not prev or not cur:
            continue
        distances.append(1.0 - cosine_similarity(prev, cur))
    diversity = sum(distances) / float(len(distances)) if distances else 0.0

    # color_entropy
    color_counts: dict[str, int] = {}
    for r in day_rows:
        for iid in (r.get("item_ids") or []):
            c = (color_by_item_id.get(str(iid)) or "").strip().lower()
            if not c:
                continue
            color_counts[c] = int(color_counts.get(c, 0)) + 1
    color_entropy = _shannon_entropy(color_counts)

    return ModeMetrics(
        mean_score=round(mean_score, 4),
        diversity=round(diversity, 4),
        repetition_rate=round(repetition_rate, 4),
        color_entropy=color_entropy,
        wear_through=round(wear_through, 4),
    )


def _rolling_mean(values: list[float], window: int) -> list[float]:
    if window <= 1:
        return [float(v) for v in values]
    out: list[float] = []
    acc = 0.0
    q: list[float] = []
    for v in values:
        v = float(v)
        q.append(v)
        acc += v
        if len(q) > window:
            acc -= q.pop(0)
        out.append(acc / float(len(q)))
    return out


def compute_time_series(
    *,
    day_rows: list[dict],
    embedding_by_item_id: Mapping[str, Sequence[float] | None],
    color_by_item_id: Mapping[str, str | None],
    window: int = 7,
) -> list[dict[str, Any]]:
    """
    Return per-day time series with rolling (moving-average) trend lines.

    Fields per row:
      - date
      - final_score
      - feedback
      - distance_prev (1 - cosine similarity) between consecutive outfit embeddings
      - ma_score, ma_diversity, ma_repetition_rate, ma_color_entropy, cum_wear_through
    """
    if not day_rows:
        return []

    dates = [str(r.get("date")) for r in day_rows]
    scores = [float(r.get("final_score") or 0.0) for r in day_rows]
    feedbacks = [(r.get("feedback") or "").strip().lower() for r in day_rows]

    # Compute per-day outfit embedding vectors
    outfit_vecs: list[list[float] | None] = []
    for r in day_rows:
        item_ids = [str(x) for x in (r.get("item_ids") or [])]
        outfit_vecs.append(_outfit_embedding(item_ids, embedding_by_item_id))

    # distance_prev (day 0 has None)
    dist_prev: list[float] = []
    dist_prev.append(0.0)
    for prev, cur in zip(outfit_vecs, outfit_vecs[1:]):
        if not prev or not cur:
            dist_prev.append(0.0)
        else:
            dist_prev.append(1.0 - cosine_similarity(prev, cur))

    # Rolling diversity: rolling mean of dist_prev excluding day0 zeros still OK as trend.
    ma_div = _rolling_mean(dist_prev, window=window)

    # Rolling repetition rate + entropy: computed on sliding window
    ma_rep: list[float] = []
    ma_ent: list[float] = []
    for i in range(len(day_rows)):
        j0 = max(0, i - window + 1)
        window_rows = day_rows[j0 : i + 1]

        uses: list[str] = []
        color_counts: dict[str, int] = {}
        for wr in window_rows:
            for iid in (wr.get("item_ids") or []):
                iid = str(iid)
                uses.append(iid)
                c = (color_by_item_id.get(iid) or "").strip().lower()
                if c:
                    color_counts[c] = int(color_counts.get(c, 0)) + 1

        total_uses = len(uses)
        unique_uses = len(set(uses))
        rep = 0.0 if total_uses == 0 else (1.0 - (unique_uses / float(total_uses)))
        ma_rep.append(rep)
        ma_ent.append(_shannon_entropy(color_counts))

    ma_score = _rolling_mean(scores, window=window)

    cum_wear = []
    worn_count = 0
    for i, fb in enumerate(feedbacks):
        if fb == "worn":
            worn_count += 1
        cum_wear.append(worn_count / float(i + 1))

    series: list[dict[str, Any]] = []
    for i in range(len(day_rows)):
        series.append(
            {
                "date": dates[i],
                "final_score": round(scores[i], 4),
                "feedback": feedbacks[i],
                "distance_prev": round(float(dist_prev[i]), 4),
                "ma_score": round(float(ma_score[i]), 4),
                "ma_diversity": round(float(ma_div[i]), 4),
                "ma_repetition_rate": round(float(ma_rep[i]), 4),
                "ma_color_entropy": round(float(ma_ent[i]), 4),
                "cum_wear_through": round(float(cum_wear[i]), 4),
            }
        )
    return series


def compute_stability(
    series: list[dict[str, Any]],
    *,
    key: str,
    tail_days: int = 21,
) -> float:
    """
    Simple stability proxy: average absolute day-to-day change of a trend metric over the last N days.
    Lower is more stable.
    """
    if not series:
        return 0.0
    tail = series[-tail_days:] if len(series) > tail_days else series
    vals = [float(r.get(key) or 0.0) for r in tail]
    if len(vals) < 2:
        return 0.0
    diffs = [abs(b - a) for a, b in zip(vals, vals[1:])]
    return round(sum(diffs) / float(len(diffs)), 6)
