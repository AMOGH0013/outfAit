from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Literal, Sequence
from uuid import uuid4, UUID

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.pool import StaticPool

from app.database import Base
from app.models.user import User
from app.models.body_profile import BodyProfile
from app.models.wardrobe import WardrobeItem
from app.models.outfit import Outfit, OutfitItem

from app.api.outfits import _build_pair_biases, _get_recent_item_embeddings
from app.api.feedback import create_feedback, FeedbackCreate
from app.services.outfit_engine import cosine_similarity, fit_preference_score, generate_weekly_plan
from app.services.style_profile import compute_user_style_profile

from evaluation.metrics import compute_metrics, compute_time_series, compute_stability
from evaluation.report import generate_report


Mode = Literal["baseline", "personalized"]


TOP_TYPES = ["shirt", "tshirt", "kurta", "hoodie"]
BOTTOM_TYPES = ["jeans", "trousers", "chinos", "shorts"]
NEUTRALS = ["black", "navy", "gray", "white", "beige", "brown"]
COLOR_POOL = [
    "black", "navy", "gray", "white", "beige", "brown",
    "red", "blue", "green", "yellow", "pink", "purple", "olive", "orange",
]


def _stable_int(s: str) -> int:
    return int(hashlib.sha256(s.encode("utf-8")).hexdigest(), 16)


def _choose_from(seq: Sequence[Any], key: str) -> Any:
    if not seq:
        raise ValueError("Cannot choose from empty sequence")
    return seq[_stable_int(key) % len(seq)]


def _load_embedding_library(db_path: Path) -> dict[str, list[dict[str, Any]]]:
    """
    Load real embeddings from an existing SQLite DB (usually `wardrobe.db`).
    Returns groups: {"top": [...], "bottom": [...], "any": [...]}.
    """
    if not db_path.exists():
        raise FileNotFoundError(f"Embedding source DB not found: {db_path}")

    conn = sqlite3.connect(str(db_path))
    try:
        cur = conn.cursor()
        cur.execute(
            "SELECT category, item_type, color, embedding FROM wardrobe_items WHERE embedding IS NOT NULL"
        )
        rows = cur.fetchall()
    finally:
        conn.close()

    any_rows: list[dict[str, Any]] = []
    top_rows: list[dict[str, Any]] = []
    bottom_rows: list[dict[str, Any]] = []

    for category, item_type, color, embedding_txt in rows:
        if not embedding_txt:
            continue
        try:
            emb = json.loads(embedding_txt) if isinstance(embedding_txt, str) else embedding_txt
        except Exception:
            continue
        if not isinstance(emb, list) or len(emb) < 10:
            continue
        rec = {
            "category": (category or "").strip().lower(),
            "item_type": (item_type or "").strip().lower(),
            "color": (color or "").strip().lower(),
            "embedding": emb,
        }
        any_rows.append(rec)
        if rec["category"] == "top":
            top_rows.append(rec)
        elif rec["category"] == "bottom":
            bottom_rows.append(rec)

    groups = {"top": top_rows, "bottom": bottom_rows, "any": any_rows}
    if not groups["any"]:
        raise RuntimeError(
            "No embeddings found in source DB. Scan a few items first so wardrobe_items.embedding is populated."
        )
    return groups


def _make_engine() -> Any:
    return create_engine(
        "sqlite+pysqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
        echo=False,
    )


def _make_session(engine) -> Session:
    SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)
    return SessionLocal()


def _allowed_types_for_category(category: str) -> list[str]:
    return BOTTOM_TYPES if category == "bottom" else TOP_TYPES


def _generate_wardrobe_seed(
    *,
    user_id: str,
    style_bias: dict,
    emb_groups: dict[str, list[dict[str, Any]]],
    n_tops: int,
    n_bottoms: int,
) -> list[dict[str, Any]]:
    preferred_colors = [c.strip().lower() for c in (style_bias.get("preferred_colors") or []) if isinstance(c, str)]
    preferred_types = [t.strip().lower() for t in (style_bias.get("preferred_item_types") or []) if isinstance(t, str)]
    avoid_colors = set(c.strip().lower() for c in (style_bias.get("avoid_colors") or []) if isinstance(c, str))

    def color_for(idx: int, category: str) -> str:
        pool = []
        # strongly bias toward preferred colors, then neutrals, then rest (excluding avoid)
        pool.extend([c for c in preferred_colors if c not in avoid_colors] * 3)
        pool.extend([c for c in NEUTRALS if c not in avoid_colors] * 2)
        pool.extend([c for c in COLOR_POOL if c not in avoid_colors])
        return _choose_from(pool, f"{user_id}:{category}:color:{idx}")

    def type_for(idx: int, category: str) -> str:
        allowed = _allowed_types_for_category(category)
        pool = []
        pool.extend([t for t in preferred_types if t in allowed] * 3)
        pool.extend(allowed)
        return _choose_from(pool, f"{user_id}:{category}:type:{idx}")

    def embedding_for(idx: int, category: str, item_type: str) -> list[float]:
        # Prefer matching category/item_type, then category, then any.
        cat_rows = emb_groups.get(category) or []
        type_rows = [r for r in cat_rows if r.get("item_type") == item_type and isinstance(r.get("embedding"), list)]
        pool = type_rows or cat_rows or (emb_groups.get("any") or [])
        rec = _choose_from(pool, f"{user_id}:{category}:{item_type}:emb:{idx}")
        return rec["embedding"]

    seeds: list[dict[str, Any]] = []
    for i in range(n_tops):
        item_type = type_for(i, "top")
        color = color_for(i, "top")
        seeds.append(
            {
                "category": "top",
                "item_type": item_type,
                "color": color,
                "color_palette": [color] + [c for c in NEUTRALS if c != color][:2],
                "embedding": embedding_for(i, "top", item_type),
            }
        )
    for i in range(n_bottoms):
        item_type = type_for(i, "bottom")
        # bottoms default to more neutrals unless user explicitly prefers bright colors
        color = _choose_from(
            (preferred_colors if preferred_colors else NEUTRALS) + NEUTRALS,
            f"{user_id}:bottom:color:{i}",
        )
        seeds.append(
            {
                "category": "bottom",
                "item_type": item_type,
                "color": color,
                "color_palette": [color] + [c for c in NEUTRALS if c != color][:2],
                "embedding": embedding_for(i, "bottom", item_type),
            }
        )
    return seeds


def _simulate_feedback(
    *,
    rng: random.Random,
    user_profile: dict,
    style_bias: dict,
    outfit_items: list[WardrobeItem],
    recent_worn_embeddings: list[list[float]],
    noise_rate: float,
) -> str:
    preferred_colors = set(c.strip().lower() for c in (style_bias.get("preferred_colors") or []) if isinstance(c, str))
    avoid_colors = set(c.strip().lower() for c in (style_bias.get("avoid_colors") or []) if isinstance(c, str))

    colors = [(it.color or "").strip().lower() for it in outfit_items]
    colors = [c for c in colors if c]

    # Fit mismatch: reuse production helper
    user_ctx = SimpleNamespace(
        id=uuid4(),
        body_shape=None,
        forbidden_items=None,
        fit_preference=user_profile.get("fit_preference"),
        style_profile=None,
    )
    fit = fit_preference_score(user_ctx, outfit_items)
    fit_mismatch = fit < -0.01

    # Embedding similarity vs recently worn
    max_sim = 0.0
    for it in outfit_items:
        if not it.embedding:
            continue
        for prev in recent_worn_embeddings or []:
            s = cosine_similarity(it.embedding, prev)
            if s > max_sim:
                max_sim = s

    # Deterministic rule-based behavior
    if any(c in avoid_colors for c in colors):
        action = "disliked"
    elif any(c in preferred_colors for c in colors):
        action = "liked"
    elif max_sim >= 0.90:
        action = "skipped"
    elif fit_mismatch:
        action = "disliked"
    else:
        action = "worn" if rng.random() < 0.30 else "liked"

    # Controlled noise (10-15%): occasionally override behavior
    if rng.random() < noise_rate:
        r = rng.random()
        if r < 0.50:
            action = "liked"
        elif r < 0.70:
            action = "worn"
        elif r < 0.90:
            action = "skipped"
        else:
            action = "disliked"

    return action


@contextmanager
def freeze_time(frozen: datetime):
    """
    Monkeypatch `datetime.now()` in a few production modules so evaluation can simulate day-by-day behavior
    without modifying production code.
    """
    from datetime import datetime as _RealDateTime

    def _mk_dt_class(dt: datetime):
        class _FrozenDateTime(_RealDateTime):  # type: ignore[misc]
            @classmethod
            def now(cls, tz=None):  # noqa: N805
                if tz is None:
                    return dt
                return dt.astimezone(tz)

        return _FrozenDateTime

    frozen_cls = _mk_dt_class(frozen)

    modules = []
    import app.services.outfit_engine as m_outfit_engine
    import app.api.outfits as m_api_outfits
    import app.services.style_profile as m_style_profile
    import app.api.feedback as m_api_feedback
    import app.models.feedback as m_model_feedback

    modules.extend([m_outfit_engine, m_api_outfits, m_style_profile, m_api_feedback, m_model_feedback])

    saved = {m: getattr(m, "datetime") for m in modules}
    try:
        for m in modules:
            setattr(m, "datetime", frozen_cls)
        yield
    finally:
        for m, old in saved.items():
            setattr(m, "datetime", old)


@dataclass(frozen=True)
class RunResult:
    user_id: str
    mode: Mode
    day_rows: list[dict]
    metrics: dict
    series: list[dict]
    stability: dict


def run_user_mode(
    *,
    user_def: dict,
    wardrobe_seed: list[dict[str, Any]],
    mode: Mode,
    start_date: date,
    days: int,
    seed: int,
    noise_rate: float,
    window: int,
) -> RunResult:
    engine = _make_engine()
    Base.metadata.create_all(bind=engine)
    db = _make_session(engine)
    try:
        # Create user + profile
        user = User(
            email=f"{user_def['user_id']}@synthetic.local",
            password_hash="synthetic",
            body_shape=user_def["profile"].get("body_shape"),
        )
        db.add(user)
        db.flush()

        profile = BodyProfile(
            user_id=user.id,
            body_shape=user_def["profile"].get("body_shape"),
            fit_preference=user_def["profile"].get("fit_preference"),
            skin_tone=user_def["profile"].get("skin_tone"),
            user_name=user_def["user_id"],
        )
        db.add(profile)
        db.flush()

        # Insert wardrobe
        for idx, s in enumerate(wardrobe_seed):
            item = WardrobeItem(
                user_id=user.id,
                image_url=f"/synthetic/{user_def['user_id']}/{idx}.png",
                mask_url=None,
                item_type=s["item_type"],
                category=s["category"],
                color=s["color"],
                color_palette=s.get("color_palette"),
                embedding=s.get("embedding"),
                is_active=True,
            )
            db.add(item)
        db.commit()

        # Build embedding/color maps for metrics
        items = db.query(WardrobeItem).filter(WardrobeItem.user_id == user.id).all()
        embedding_by_id = {str(i.id): i.embedding for i in items}
        color_by_id = {str(i.id): i.color for i in items}

        rng = random.Random(_stable_int(f"{seed}:{user_def['user_id']}:{mode}"))

        day_rows: list[dict] = []
        for day_idx in range(days):
            d = start_date + timedelta(days=day_idx)
            sim_dt = datetime.combine(d, time(12, 0), tzinfo=timezone.utc)

            with freeze_time(sim_dt):
                wardrobe_items = (
                    db.query(WardrobeItem)
                    .filter(
                        WardrobeItem.user_id == user.id,
                        WardrobeItem.is_active.is_(True),
                        WardrobeItem.item_type != "unknown",
                    )
                    .all()
                )

                if mode == "personalized":
                    pair_biases = _build_pair_biases(db, user.id)
                    recent_embeddings = _get_recent_item_embeddings(db, user.id)
                    sp = compute_user_style_profile(db, user.id)
                    user_ctx = SimpleNamespace(
                        id=user.id,
                        body_shape=user.body_shape,
                        forbidden_items=None,
                        fit_preference=profile.fit_preference,
                        style_profile={
                            **sp.to_public(),
                            "liked_centroid": sp.liked_centroid,
                            "disliked_centroid": sp.disliked_centroid,
                        },
                    )
                else:
                    pair_biases = None
                    recent_embeddings = None
                    user_ctx = SimpleNamespace(
                        id=user.id,
                        body_shape=None,
                        forbidden_items=None,
                        fit_preference=None,
                        style_profile=None,
                    )

                plan = generate_weekly_plan(
                    user=user_ctx,
                    wardrobe_items=wardrobe_items,
                    days=1,
                    pair_biases=pair_biases,
                    recent_item_embeddings=recent_embeddings,
                )
                if not plan:
                    break
                best = plan[0]
                picked: list[WardrobeItem] = best["items"]

                outfit = Outfit(
                    user_id=user.id,
                    outfit_date=d,
                    score=best["score"],
                    explanation=json.dumps(best["explanation_data"]),
                )
                db.add(outfit)
                db.flush()
                for it in picked:
                    db.add(OutfitItem(outfit_id=outfit.id, wardrobe_item_id=it.id))
                db.commit()

                # Feedback simulation uses *recent worn* embeddings (same as production diversity uses)
                recent_worn_embeddings = _get_recent_item_embeddings(db, user.id)
                action = _simulate_feedback(
                    rng=rng,
                    user_profile=user_def["profile"],
                    style_bias=user_def["style_bias"],
                    outfit_items=picked,
                    recent_worn_embeddings=recent_worn_embeddings,
                    noise_rate=noise_rate,
                )

                # Apply feedback through existing handler (direct call)
                create_feedback(
                    payload=FeedbackCreate(outfit_id=outfit.id, action=action),
                    db=db,
                    current_user=user,
                )

                db.commit()

                day_rows.append(
                    {
                        "date": str(d),
                        "outfit_id": str(outfit.id),
                        "item_ids": [str(i.id) for i in picked],
                        "final_score": float(best["score"]),
                        "feedback": action,
                    }
                )

        metrics = compute_metrics(
            day_rows=day_rows,
            embedding_by_item_id=embedding_by_id,
            color_by_item_id=color_by_id,
        )
        series = compute_time_series(
            day_rows=day_rows,
            embedding_by_item_id=embedding_by_id,
            color_by_item_id=color_by_id,
            window=window,
        )
        stability = {
            "ma_score_tail_volatility": compute_stability(series, key="ma_score"),
            "ma_diversity_tail_volatility": compute_stability(series, key="ma_diversity"),
            "cum_wear_tail_volatility": compute_stability(series, key="cum_wear_through"),
        }
        return RunResult(
            user_id=user_def["user_id"],
            mode=mode,
            day_rows=day_rows,
            metrics=metrics.__dict__,
            series=series,
            stability=stability,
        )
    finally:
        db.close()


def _write_user_csv(out_path: Path, day_rows: list[dict]) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["date", "outfit_id", "item_ids", "final_score", "feedback"])
        for r in day_rows:
            w.writerow([r["date"], r["outfit_id"], "|".join(r["item_ids"]), r["final_score"], r["feedback"]])


def _write_timeseries_csv(out_path: Path, series: list[dict]) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if not series:
        return
    cols = list(series[0].keys())
    with out_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for r in series:
            w.writerow([r.get(c) for c in cols])


def _mean(xs: list[float]) -> float:
    return sum(xs) / float(len(xs)) if xs else 0.0


def _std(xs: list[float]) -> float:
    if len(xs) < 2:
        return 0.0
    m = _mean(xs)
    var = sum((x - m) ** 2 for x in xs) / float(len(xs) - 1)
    return var ** 0.5


def main() -> int:
    parser = argparse.ArgumentParser(description="Offline synthetic A/B evaluation for personalization.")
    parser.add_argument("--days", type=int, default=21)
    parser.add_argument("--start-date", type=str, default="2026-02-01")
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--noise", type=float, default=0.12, help="noise rate (0.10-0.15 recommended)")
    parser.add_argument("--window", type=int, default=7, help="moving-average window size for trends")
    parser.add_argument("--replicates", type=int, default=1, help="repeat runs to estimate variance (deterministic)")
    parser.add_argument("--embedding-source-db", type=str, default="wardrobe.db")
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    eval_dir = root / "evaluation"

    users_path = eval_dir / "synthetic_users.json"
    users = json.loads(users_path.read_text(encoding="utf-8"))

    emb_groups = _load_embedding_library(root / args.embedding_source_db)

    start_d = date.fromisoformat(args.start_date)
    days = int(args.days)
    seed = int(args.seed)
    noise = float(args.noise)
    window = max(1, int(args.window))
    replicates = max(1, int(args.replicates))

    all_results: dict[str, Any] = {
        "start_date": str(start_d),
        "days": days,
        "seed": seed,
        "noise": noise,
        "window": window,
        "replicates": replicates,
        "users": [],
    }

    for u in users:
        wardrobe_seed = _generate_wardrobe_seed(
            user_id=u["user_id"],
            style_bias=u["style_bias"],
            emb_groups=emb_groups,
            n_tops=16,
            n_bottoms=8,
        )

        base_runs: list[RunResult] = []
        pers_runs: list[RunResult] = []

        for r in range(replicates):
            rep_seed = seed + r * 101
            base_runs.append(
                run_user_mode(
                    user_def=u,
                    wardrobe_seed=wardrobe_seed,
                    mode="baseline",
                    start_date=start_d,
                    days=days,
                    seed=rep_seed,
                    noise_rate=noise,
                    window=window,
                )
            )
            pers_runs.append(
                run_user_mode(
                    user_def=u,
                    wardrobe_seed=wardrobe_seed,
                    mode="personalized",
                    start_date=start_d,
                    days=days,
                    seed=rep_seed,
                    noise_rate=noise,
                    window=window,
                )
            )

        # Save replicate-0 detailed CSVs (keeps output size reasonable)
        base0 = base_runs[0]
        pers0 = pers_runs[0]
        _write_user_csv(eval_dir / f"{u['user_id']}_baseline.csv", base0.day_rows)
        _write_user_csv(eval_dir / f"{u['user_id']}_personalized.csv", pers0.day_rows)
        _write_timeseries_csv(eval_dir / f"{u['user_id']}_baseline_timeseries.csv", base0.series)
        _write_timeseries_csv(eval_dir / f"{u['user_id']}_personalized_timeseries.csv", pers0.series)

        def agg(metric_key: str, runs: list[RunResult]) -> dict[str, float]:
            xs = [float(rr.metrics[metric_key]) for rr in runs]
            return {"mean": round(_mean(xs), 4), "std": round(_std(xs), 4)}

        baseline_agg = {k: agg(k, base_runs) for k in base0.metrics.keys()}
        personalized_agg = {k: agg(k, pers_runs) for k in pers0.metrics.keys()}

        def lift_stat(metric_key: str) -> dict[str, float]:
            xs = [float(p.metrics[metric_key]) - float(b.metrics[metric_key]) for b, p in zip(base_runs, pers_runs)]
            m = _mean(xs)
            s = _std(xs)
            se = s / (len(xs) ** 0.5) if len(xs) > 1 else 0.0
            ci = 1.96 * se if se else 0.0
            return {"mean": round(m, 4), "std": round(s, 4), "ci95": round(ci, 4)}

        lift = {
            "score_lift": lift_stat("mean_score"),
            "diversity_lift": lift_stat("diversity"),
            "repetition_rate_lift": lift_stat("repetition_rate"),
            "color_entropy_lift": lift_stat("color_entropy"),
            "wear_through_lift": lift_stat("wear_through"),
        }

        stability = {
            "baseline": base0.stability,
            "personalized": pers0.stability,
            "lift": {
                k: round(float(pers0.stability.get(k, 0.0)) - float(base0.stability.get(k, 0.0)), 6)
                for k in set(base0.stability.keys()) | set(pers0.stability.keys())
            },
        }

        all_results["users"].append(
            {
                "user_id": u["user_id"],
                "baseline": base0.metrics,
                "personalized": pers0.metrics,
                "baseline_agg": baseline_agg,
                "personalized_agg": personalized_agg,
                "lift": lift,
                "stability": stability,
                "baseline_series": base0.series,
                "personalized_series": pers0.series,
            }
        )

    summary_path = eval_dir / "summary.json"
    summary_path.write_text(json.dumps(all_results, indent=2), encoding="utf-8")
    generate_report(summary_path=summary_path, out_path=eval_dir / "report.md")
    print(f"Wrote {summary_path}")
    print(f"Wrote {eval_dir / 'report.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
