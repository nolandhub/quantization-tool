# ABOUTME: Summary of a run: Δ pp and recovery % against the baseline (only numbers qtool computes itself),
# ABOUTME: the summary table printed at the end and summary.json written next to the results.
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

from qtool.console import fmt_gb
from qtool.parsers import Metric


def delta_pp(value: float, base: float) -> float:
    return (value - base) * 100


def recovery(value: float, base: float) -> float | None:
    return value / base * 100 if base > 0 else None


@dataclass
class Row:
    label: str
    status: str  # OK | FAILED | SKIPPED
    metrics: dict[str, Metric] = field(default_factory=dict)  # primary metric per task
    size_bytes: int | None = None
    note: str = ""


def _score(m: Metric | None) -> str:
    if m is None:
        return "—"
    return f"{m.value:.3f} ± {m.stderr:.3f}" if m.stderr is not None else f"{m.value:.3f}"


def vs_baseline(m: Metric | None, base: Metric | None) -> tuple[str, str]:
    if m is None or base is None:
        return "n/a", "n/a"
    rec = recovery(m.value, base.value)
    return f"{delta_pp(m.value, base.value):+.1f} pp".replace("-", "−"), "n/a" if rec is None else f"{rec:.1f}%"


def summary_lines(rows: list[Row], task: str, baseline: str | None) -> list[str]:
    base_row = next((r for r in rows if r.label == baseline), None)
    base = base_row.metrics.get(task) if base_row else None
    w = max([16] + [len(r.label) + 2 for r in rows])
    out = [f"  {'scheme':<{w}}{'status':<9}{'accuracy':<16}{'Δ vs base':<12}{'recovery':<11}size"]
    for r in rows:
        m = r.metrics.get(task)
        delta, rec = ("—", "—") if r.label == baseline else vs_baseline(m, base)
        size = fmt_gb(r.size_bytes) + (f" ({r.note})" if r.note else "")
        out.append(f"  {r.label:<{w}}{r.status:<9}{_score(m):<16}{delta:<12}{rec:<11}{size}")
    return out


def summary_title(task: str, rows: list[Row], limit: int | None) -> str:
    metric = next((r.metrics[task] for r in rows if task in r.metrics), None)
    what = f"{task} {metric.filter}" if metric and metric.filter != "none" else task
    return f"Summary · {what} ({'full' if limit is None else f'limit {limit}'})"


def write_summary(path: Path, rows: list[Row], meta: dict) -> None:
    data = {"meta": meta, "rows": [{**asdict(r), "metrics": {t: asdict(m) for t, m in r.metrics.items()}} for r in rows]}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=1))
