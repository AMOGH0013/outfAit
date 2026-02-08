from __future__ import annotations

import json
from pathlib import Path
from typing import Any


_ROOT = Path(__file__).resolve().parents[1]
_EVAL_DIR = _ROOT / "evaluation"


def _fmt(x):
    if isinstance(x, float):
        return f"{x:.4f}"
    return str(x)


_SPARK_CHARS = " .:-=+*#%@"


def _sparkline(values: list[float], width: int = 42) -> str:
    if not values:
        return ""
    if width <= 0:
        width = len(values)
    if len(values) > width:
        # even downsample
        step = (len(values) - 1) / float(width - 1) if width > 1 else 1.0
        sampled = [values[int(round(i * step))] for i in range(width)]
    else:
        sampled = values[:]
    lo = min(sampled)
    hi = max(sampled)
    if abs(hi - lo) < 1e-12:
        return _SPARK_CHARS[0] * len(sampled)
    out = []
    for v in sampled:
        t = (v - lo) / (hi - lo)
        idx = int(round(t * (len(_SPARK_CHARS) - 1)))
        idx = max(0, min(idx, len(_SPARK_CHARS) - 1))
        out.append(_SPARK_CHARS[idx])
    return "".join(out)


def _lift_mean(x: Any) -> float:
    # supports both old float lifts and new {mean,std,ci95} lifts
    if isinstance(x, (int, float)):
        return float(x)
    if isinstance(x, dict) and "mean" in x:
        try:
            return float(x["mean"])
        except Exception:
            return 0.0
    return 0.0


def _lift_ci95(x: Any) -> float | None:
    if isinstance(x, dict) and "ci95" in x:
        try:
            return float(x["ci95"])
        except Exception:
            return None
    return None


def generate_report(summary_path: Path | None = None, out_path: Path | None = None) -> Path:
    summary_path = summary_path or (_EVAL_DIR / "summary.json")
    out_path = out_path or (_EVAL_DIR / "report.md")

    data = json.loads(summary_path.read_text(encoding="utf-8"))
    users = data.get("users", [])

    lines = []
    lines.append("# Offline A/B Evaluation (Synthetic Users)")
    lines.append("")
    lines.append(f"- days: `{data.get('days')}`")
    lines.append(f"- start_date: `{data.get('start_date')}`")
    lines.append(f"- seed: `{data.get('seed')}`")
    if "noise" in data:
        lines.append(f"- noise: `{data.get('noise')}`")
    if "window" in data:
        lines.append(f"- rolling window: `{data.get('window')}d`")
    if "replicates" in data:
        lines.append(f"- replicates: `{data.get('replicates')}`")
    lines.append("")

    lines.append("## Per-user metrics")
    lines.append("")
    rep = int(data.get("replicates") or 1)
    if rep > 1:
        lines.append("| user_id | mean_score base (mean+/-std) | mean_score pers (mean+/-std) | score_lift (mean+/-ci95) | diversity_lift (mean+/-ci95) | repetition_lift (mean+/-ci95) | wear_through_lift (mean+/-ci95) |")
        lines.append("|---|---:|---:|---:|---:|---:|---:|")
    else:
        lines.append("| user_id | mean_score (base) | mean_score (pers) | score_lift | diversity_lift | repetition_lift | wear_through_lift |")
        lines.append("|---|---:|---:|---:|---:|---:|---:|")

    lifts = {"score": [], "diversity": [], "repetition": [], "wear": []}
    for u in users:
        uid = u["user_id"]
        base = u["baseline"]
        pers = u["personalized"]
        base_agg = u.get("baseline_agg") or {}
        pers_agg = u.get("personalized_agg") or {}
        lift = u["lift"]
        score_l = lift.get("score_lift")
        div_l = lift.get("diversity_lift")
        rep_l = lift.get("repetition_rate_lift")
        wear_l = lift.get("wear_through_lift")

        def cell(x: Any) -> str:
            mean = _lift_mean(x)
            ci = _lift_ci95(x)
            if rep > 1 and ci is not None:
                return f"{mean:.4f} (+/-{ci:.4f})"
            return f"{mean:.4f}"

        def mean_std_cell(agg: Any, fallback: float) -> str:
            if isinstance(agg, dict) and "mean" in agg and "std" in agg:
                try:
                    return f"{float(agg['mean']):.4f} (+/-{float(agg['std']):.4f})"
                except Exception:
                    return f"{fallback:.4f}"
            return f"{fallback:.4f}"

        if rep > 1:
            base_ms = mean_std_cell((base_agg.get("mean_score") if isinstance(base_agg, dict) else None), float(base["mean_score"]))
            pers_ms = mean_std_cell((pers_agg.get("mean_score") if isinstance(pers_agg, dict) else None), float(pers["mean_score"]))
        else:
            base_ms = _fmt(base["mean_score"])
            pers_ms = _fmt(pers["mean_score"])

        lines.append(
            f"| {uid} | {base_ms} | {pers_ms} | {cell(score_l)} | "
            f"{cell(div_l)} | {cell(rep_l)} | {cell(wear_l)} |"
        )
        lifts["score"].append(_lift_mean(score_l))
        lifts["diversity"].append(_lift_mean(div_l))
        lifts["repetition"].append(_lift_mean(rep_l))
        lifts["wear"].append(_lift_mean(wear_l))

    def avg(xs):
        return sum(xs) / float(len(xs)) if xs else 0.0

    lines.append("")
    lines.append("## Summary")
    lines.append("")
    lines.append(f"- Avg score lift: `{avg(lifts['score']):.4f}`")
    lines.append(f"- Avg diversity lift: `{avg(lifts['diversity']):.4f}` (higher = more visually diverse day-to-day)")
    lines.append(f"- Avg repetition lift: `{avg(lifts['repetition']):.4f}` (negative = fewer repeats; positive = more repeats)")
    lines.append(f"- Avg wear-through lift: `{avg(lifts['wear']):.4f}`")
    lines.append("")
    lines.append("### Interpretation")
    lines.append("- If diversity lift is positive and repetition lift is negative, personalization is improving variety.")
    lines.append("- If score lift is positive, personalization is improving the engine's own scoring objective.")
    lines.append("")

    window = int(data.get("window") or 7)
    lines.append(f"## Trends (rolling {window}d)")
    lines.append("")
    lines.append("Sparklines show trend lines from day 1 to day N. Baseline vs Personalized use independent scaling per line.")
    lines.append("")

    # Early vs late snapshots at trend endpoints
    early_lifts: dict[str, list[float]] = {"ma_score": [], "ma_diversity": [], "ma_repetition_rate": [], "cum_wear_through": []}
    late_lifts: dict[str, list[float]] = {"ma_score": [], "ma_diversity": [], "ma_repetition_rate": [], "cum_wear_through": []}

    for u in users:
        uid = u["user_id"]
        base_series = u.get("baseline_series") or []
        pers_series = u.get("personalized_series") or []

        def get_vals(series: list[dict], key: str) -> list[float]:
            out = []
            for r in series:
                try:
                    out.append(float(r.get(key) or 0.0))
                except Exception:
                    out.append(0.0)
            return out

        b_score = get_vals(base_series, "ma_score")
        p_score = get_vals(pers_series, "ma_score")
        b_div = get_vals(base_series, "ma_diversity")
        p_div = get_vals(pers_series, "ma_diversity")
        b_rep = get_vals(base_series, "ma_repetition_rate")
        p_rep = get_vals(pers_series, "ma_repetition_rate")
        b_wear = get_vals(base_series, "cum_wear_through")
        p_wear = get_vals(pers_series, "cum_wear_through")

        def at(vals: list[float], idx: int) -> float:
            if not vals:
                return 0.0
            idx = max(0, min(idx, len(vals) - 1))
            return float(vals[idx])

        early_idx = min(max(0, window - 1), max(0, len(base_series) - 1))
        late_idx = max(0, len(base_series) - 1)

        early = {
            "ma_score": (at(b_score, early_idx), at(p_score, early_idx)),
            "ma_diversity": (at(b_div, early_idx), at(p_div, early_idx)),
            "ma_repetition_rate": (at(b_rep, early_idx), at(p_rep, early_idx)),
            "cum_wear_through": (at(b_wear, early_idx), at(p_wear, early_idx)),
        }
        late = {
            "ma_score": (at(b_score, late_idx), at(p_score, late_idx)),
            "ma_diversity": (at(b_div, late_idx), at(p_div, late_idx)),
            "ma_repetition_rate": (at(b_rep, late_idx), at(p_rep, late_idx)),
            "cum_wear_through": (at(b_wear, late_idx), at(p_wear, late_idx)),
        }

        for k in early_lifts.keys():
            early_lifts[k].append(float(early[k][1] - early[k][0]))
            late_lifts[k].append(float(late[k][1] - late[k][0]))

        lines.append("<details>")
        lines.append(f"<summary>{uid}</summary>")
        lines.append("")
        lines.append(f"- ma_score: base `{_sparkline(b_score)}` | pers `{_sparkline(p_score)}`")
        lines.append(f"- cum_wear_through: base `{_sparkline(b_wear)}` | pers `{_sparkline(p_wear)}`")
        lines.append(f"- ma_diversity: base `{_sparkline(b_div)}` | pers `{_sparkline(p_div)}`")
        lines.append(f"- ma_repetition_rate: base `{_sparkline(b_rep)}` | pers `{_sparkline(p_rep)}`")
        lines.append("")
        lines.append(f"Snapshot at day {early_idx+1} (end of first rolling window) vs day {late_idx+1}:")
        lines.append(
            f"- early lift (pers-base): score {early['ma_score'][1]-early['ma_score'][0]:.4f}, "
            f"div {early['ma_diversity'][1]-early['ma_diversity'][0]:.4f}, "
            f"rep {early['ma_repetition_rate'][1]-early['ma_repetition_rate'][0]:.4f}, "
            f"wear {early['cum_wear_through'][1]-early['cum_wear_through'][0]:.4f}"
        )
        lines.append(
            f"- late lift (pers-base): score {late['ma_score'][1]-late['ma_score'][0]:.4f}, "
            f"div {late['ma_diversity'][1]-late['ma_diversity'][0]:.4f}, "
            f"rep {late['ma_repetition_rate'][1]-late['ma_repetition_rate'][0]:.4f}, "
            f"wear {late['cum_wear_through'][1]-late['cum_wear_through'][0]:.4f}"
        )

        stab = u.get("stability") or {}
        bstab = stab.get("baseline") or {}
        pstab = stab.get("personalized") or {}
        lines.append("")
        lines.append("Stability (lower tail volatility is more stable):")
        lines.append(f"- baseline: `{bstab}`")
        lines.append(f"- personalized: `{pstab}`")
        lines.append("")
        lines.append("</details>")
        lines.append("")

    def _avg(xs: list[float]) -> float:
        return sum(xs) / float(len(xs)) if xs else 0.0

    lines.append("## Early vs Late lift (averaged)")
    lines.append("")
    lines.append(f"- Early avg lift (day {window}): score `{_avg(early_lifts['ma_score']):.4f}`, diversity `{_avg(early_lifts['ma_diversity']):.4f}`, repetition `{_avg(early_lifts['ma_repetition_rate']):.4f}`, wear `{_avg(early_lifts['cum_wear_through']):.4f}`")
    lines.append(f"- Late avg lift (day N): score `{_avg(late_lifts['ma_score']):.4f}`, diversity `{_avg(late_lifts['ma_diversity']):.4f}`, repetition `{_avg(late_lifts['ma_repetition_rate']):.4f}`, wear `{_avg(late_lifts['cum_wear_through']):.4f}`")
    lines.append("")

    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out_path


if __name__ == "__main__":
    path = generate_report()
    print(f"Wrote {path}")
