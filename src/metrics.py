"""Расчёт метрик и экспертная классификация руды (логика ТЗ)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

TALC_THRESHOLD_PCT = 10.0

CLASS_RU = {
    "otalkovannaya": "оталькованная",
    "ryadovaya": "рядовая",
    "trudnoobogatimaya": "труднообогатимая",
}


@dataclass
class OreMetrics:
    talc_pct: float
    sulfide_pct: float
    ordinary_pct: float
    fine_pct: float
    ordinary_share: float  # % среди сульфидов
    fine_share: float
    talc_method: str


@dataclass
class OreResult:
    ore_class: str
    ore_class_ru: str
    summary: str
    metrics: OreMetrics
    rules_class: str | None = None
    rules_class_ru: str | None = None
    classifier_confidence: float | None = None
    classifier_probabilities: dict[str, float] | None = None
    classification_method: str = "rules"


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


def classify_ore_by_rules(metrics: OreMetrics) -> tuple[str, str]:
    """Экспертная логика из ТЗ."""
    if metrics.talc_pct > TALC_THRESHOLD_PCT:
        ore_class = "otalkovannaya"
    elif metrics.ordinary_share >= metrics.fine_share:
        ore_class = "ryadovaya"
    else:
        ore_class = "trudnoobogatimaya"
    return ore_class, CLASS_RU[ore_class]


def build_summary(ore_class_ru: str, metrics: OreMetrics) -> str:
    """Текстовый вывод в формате ТЗ."""
    if metrics.fine_share >= metrics.ordinary_share:
        intergrowth, share = "тонких", metrics.fine_share
    else:
        intergrowth, share = "обычных", metrics.ordinary_share
    return (
        f"Руда классифицирована как {ore_class_ru}: "
        f"содержание талька — {metrics.talc_pct:.1f}%, "
        f"преобладание {intergrowth} срастаний — {share:.0f}%"
    )


def format_metrics_table(metrics: OreMetrics) -> str:
    """Таблица метрик для отчёта / консоли."""
    lines = [
        "| Метрика | Значение |",
        "|---------|----------|",
        f"| Доля сульфидов (всего) | {metrics.sulfide_pct:.2f}% |",
        f"| Обычные срастания | {metrics.ordinary_pct:.2f}% ({metrics.ordinary_share:.1f}% среди сульфидов) |",
        f"| Тонкие срастания | {metrics.fine_pct:.2f}% ({metrics.fine_share:.1f}% среди сульфидов) |",
        f"| Тальк | {metrics.talc_pct:.2f}% |",
        f"| Метод талька | {metrics.talc_method} |",
    ]
    return "\n".join(lines)


def classify_ore(metrics: OreMetrics) -> OreResult:
    ore_class, ru = classify_ore_by_rules(metrics)
    return OreResult(
        ore_class=ore_class,
        ore_class_ru=ru,
        summary=build_summary(ru, metrics),
        metrics=metrics,
        rules_class=ore_class,
        rules_class_ru=ru,
        classification_method="rules",
    )


def classify_ore_result(
    metrics: OreMetrics,
    classifier_result: dict | None = None,
) -> OreResult:
    """
    Финальный класс — по экспертным правилам ТЗ.
    CNN (если передан) — только для сравнения.
    """
    ore_class, ru = classify_ore_by_rules(metrics)
    result = OreResult(
        ore_class=ore_class,
        ore_class_ru=ru,
        summary=build_summary(ru, metrics),
        metrics=metrics,
        rules_class=ore_class,
        rules_class_ru=ru,
        classification_method="rules",
    )
    if classifier_result:
        result.classifier_confidence = float(classifier_result.get("confidence", 0))
        result.classifier_probabilities = classifier_result.get("probabilities")
    return result


def classify_ore_with_cnn(
    metrics: OreMetrics,
    classifier_result: dict,
) -> OreResult:
    """Обратная совместимость: правила — финал, CNN — в полях сравнения."""
    return classify_ore_result(metrics, classifier_result)
