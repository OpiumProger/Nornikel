"""Расчёт метрик и экспертная классификация руды."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

TALC_THRESHOLD_PCT = 10.0


@dataclass
class OreMetrics:
    talc_pct: float
    sulfide_pct: float
    ordinary_pct: float
    fine_pct: float
    ordinary_share: float  # ordinary / (ordinary + fine)
    fine_share: float
    talc_method: str


@dataclass
class OreResult:
    ore_class: str
    ore_class_ru: str
    summary: str
    metrics: OreMetrics


CLASS_RU = {
    "otalkovannaya": "оталькованная",
    "ryadovaya": "рядовая",
    "trudnoobogatimaya": "труднообогатимая",
}


def mask_area_pct(mask: np.ndarray) -> float:
    return float(np.count_nonzero(mask)) / mask.size * 100


def compute_metrics(
    talc_mask: np.ndarray,
    ordinary_mask: np.ndarray,
    fine_mask: np.ndarray,
    talc_method: str,
) -> OreMetrics:
    talc_pct = mask_area_pct(talc_mask)
    ordinary_pct = mask_area_pct(ordinary_mask)
    fine_pct = mask_area_pct(fine_mask)
    sulfide_pct = ordinary_pct + fine_pct

    sulfide_sum = ordinary_pct + fine_pct
    if sulfide_sum > 0:
        ordinary_share = ordinary_pct / sulfide_sum * 100
        fine_share = fine_pct / sulfide_sum * 100
    else:
        ordinary_share = fine_share = 50.0

    return OreMetrics(
        talc_pct=round(talc_pct, 2),
        sulfide_pct=round(sulfide_pct, 2),
        ordinary_pct=round(ordinary_pct, 2),
        fine_pct=round(fine_pct, 2),
        ordinary_share=round(ordinary_share, 1),
        fine_share=round(fine_share, 1),
        talc_method=talc_method,
    )


def classify_ore(metrics: OreMetrics) -> OreResult:
    if metrics.talc_pct > TALC_THRESHOLD_PCT:
        ore_class = "otalkovannaya"
    elif metrics.ordinary_share >= metrics.fine_share:
        ore_class = "ryadovaya"
    else:
        ore_class = "trudnoobogatimaya"

    ru = CLASS_RU[ore_class]
    summary = (
        f"Руда классифицирована как {ru}: "
        f"содержание талька — {metrics.talc_pct:.1f}%, "
        f"преобладание {'обычных' if metrics.ordinary_share >= metrics.fine_share else 'тонких'} "
        f"срастаний — {max(metrics.ordinary_share, metrics.fine_share):.0f}%"
    )

    return OreResult(ore_class=ore_class, ore_class_ru=ru, summary=summary, metrics=metrics)
