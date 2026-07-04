"""Формирование JSON/PDF отчётов по результатам анализа."""

from __future__ import annotations

import json
from datetime import datetime
from io import BytesIO
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image as PILImage

from src.metrics import CLASS_RU, format_classifier_note


def build_report_payload(
    analysis: dict[str, Any],
    source_name: str,
    talc_model: Path | None = None,
    use_unet: bool = True,
    talc_threshold: float | str = "auto",
    talc_heuristic: str = "legacy",
    unet_max_side: int | None = None,
    classifier_model: Path | None = None,
    use_classifier: bool = False,
) -> dict[str, Any]:
    """Возвращает JSON-сериализуемый отчёт без изображений и numpy-массивов."""
    result = analysis["result"]
    metrics = result.metrics
    classifier = analysis.get("classifier") or {}

    return {
        "file": source_name,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "ore_class": result.ore_class,
        "ore_class_ru": result.ore_class_ru,
        "summary": result.summary,
        "classification_method": result.classification_method,
        "rules_class": result.rules_class,
        "rules_class_ru": result.rules_class_ru,
        "classifier_class": classifier.get("pred_label"),
        "classifier_class_ru": classifier.get("pred_label_ru"),
        "classifier_confidence": result.classifier_confidence,
        "classifier_probabilities": result.classifier_probabilities,
        "parameters": {
            "use_unet": use_unet,
            "talc_model": str(talc_model) if use_unet and talc_model else None,
            "talc_threshold": talc_threshold,
            "talc_threshold_used": analysis.get("talc_threshold_used"),
            "talc_heuristic": talc_heuristic,
            "unet_max_side": unet_max_side,
            "use_classifier": use_classifier,
            "classifier_model": str(classifier_model) if use_classifier and classifier_model else None,
        },
        "metrics": {
            "talc_pct": metrics.talc_pct,
            "sulfide_pct": metrics.sulfide_pct,
            "ordinary_pct": metrics.ordinary_pct,
            "fine_pct": metrics.fine_pct,
            "ordinary_share": metrics.ordinary_share,
            "fine_share": metrics.fine_share,
            "talc_method": metrics.talc_method,
        },
    }


def report_json_bytes(payload: dict[str, Any]) -> bytes:
    return json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8")


def _image_bytes_bgr(img_bgr: np.ndarray) -> BytesIO:
    ok, buf = cv2.imencode(".png", img_bgr)
    if not ok:
        raise ValueError("Не удалось подготовить изображение для PDF")
    return BytesIO(buf.tobytes())


def _padded_logo_bytes(logo_path: Path, padding: int = 28) -> BytesIO:
    logo = PILImage.open(logo_path).convert("RGBA")
    padded = PILImage.new(
        "RGBA",
        (logo.width + padding * 2, logo.height + padding * 2),
        (255, 255, 255, 0),
    )
    padded.paste(logo, (padding, padding), logo)

    out = BytesIO()
    padded.save(out, format="PNG")
    out.seek(0)
    return out


def _register_font() -> str:
    """Подключает шрифт с кириллицей, если он есть в системе."""
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont

    candidates = [
        "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/Library/Fonts/Arial Unicode.ttf",
        "/Library/Fonts/Arial.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/local/share/fonts/dejavu/DejaVuSans.ttf",
    ]
    for font_path in candidates:
        if Path(font_path).exists():
            pdfmetrics.registerFont(TTFont("ReportFont", font_path))
            return "ReportFont"
    return "Helvetica"


def create_pdf_report(
    payload: dict[str, Any],
    overlay_bgr: np.ndarray,
    logo_path: Path | None = None,
) -> bytes:
    """Создаёт PDF-отчёт с заключением, метриками и цветной маской."""
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.lib.units import cm
    from reportlab.platypus import Image, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

    font_name = _register_font()
    buffer = BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        rightMargin=1.6 * cm,
        leftMargin=1.6 * cm,
        topMargin=1.4 * cm,
        bottomMargin=1.4 * cm,
    )

    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        "TitleRu",
        parent=styles["Title"],
        fontName=font_name,
        fontSize=18,
        leading=22,
    )
    body_style = ParagraphStyle(
        "BodyRu",
        parent=styles["BodyText"],
        fontName=font_name,
        fontSize=10,
        leading=14,
    )

    metrics = payload["metrics"]
    params = payload.get("parameters") or {}
    threshold_used = params.get("talc_threshold_used")
    unet_max_side = params.get("unet_max_side")

    def _table_style() -> TableStyle:
        return TableStyle(
            [
                ("FONTNAME", (0, 0), (-1, -1), font_name),
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1f4e79")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("GRID", (0, 0), (-1, -1), 0.4, colors.grey),
                ("BACKGROUND", (0, 1), (-1, -1), colors.HexColor("#f7f9fb")),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("LEFTPADDING", (0, 0), (-1, -1), 7),
                ("RIGHTPADDING", (0, 0), (-1, -1), 7),
            ]
        )

    def _make_table(rows: list[list[str]]) -> Table:
        table = Table(rows, colWidths=[8.2 * cm, 7.2 * cm])
        table.setStyle(_table_style())
        return table

    cnn_class = payload.get("classifier_class_ru") or payload.get("classifier_class") or "не запускалась"
    cnn_conf = payload.get("classifier_confidence")
    cnn_conf_str = f"{float(cnn_conf):.3f}" if cnn_conf is not None else "—"
    classification_method = payload.get("classification_method", "rules")

    metrics_rows = [
        ["Показатель", "Значение"],
        ["Класс руды", payload["ore_class_ru"]],
        ["Класс по CNN-классификатору", cnn_class],
        ["Тальк, %", f"{metrics['talc_pct']:.2f}"],
        ["Сульфиды всего, %", f"{metrics['sulfide_pct']:.2f}"],
        ["Обычные срастания, %", f"{metrics['ordinary_pct']:.2f}"],
        ["Тонкие срастания, %", f"{metrics['fine_pct']:.2f}"],
        ["Доля обычных среди сульфидов, %", f"{metrics['ordinary_share']:.1f}"],
        ["Доля тонких среди сульфидов, %", f"{metrics['fine_share']:.1f}"],
        ["Метод определения талька", metrics["talc_method"]],
        ["Выбранная эвристика талька", str(params.get("talc_heuristic", "legacy"))],
        [
            "Размер U-Net-инференса",
            "полный размер" if unet_max_side is None else f"до {unet_max_side}px",
        ],
        [
            "Порог U-Net",
            f"{float(threshold_used):.3f}" if threshold_used is not None else "не использовался",
        ],
        ["U-Net включена", "да" if params.get("use_unet") else "нет"],
        ["CNN включена", "да" if params.get("use_classifier") else "нет"],
        ["Уверенность CNN", cnn_conf_str],
    ]
    metrics_table = _make_table(metrics_rows)

    cnn_probs = payload.get("classifier_probabilities") or {}
    cnn_table = None
    if cnn_probs:
        cnn_rows = [["Класс CNN", "Вероятность"]]
        for label, prob in cnn_probs.items():
            label_ru = CLASS_RU.get(str(label), str(label))
            cnn_rows.append([label_ru, f"{float(prob):.3f}"])
        cnn_table = _make_table(cnn_rows)

    overlay = Image(_image_bytes_bgr(overlay_bgr), width=15.6 * cm, height=10.5 * cm, kind="proportional")
    story = []
    if logo_path and logo_path.exists():
        story.extend(
            [
                Image(_padded_logo_bytes(logo_path), width=5.2 * cm, height=1.7 * cm, kind="proportional"),
                Spacer(1, 0.35 * cm),
            ]
        )

    story.extend(
        [
            Paragraph("Отчёт по анализу руды", title_style),
            Spacer(1, 0.25 * cm),
            Paragraph(f"Файл: {payload['file']}", body_style),
            Paragraph(f"Дата анализа: {payload['created_at']}", body_style),
            Spacer(1, 0.25 * cm),
            Paragraph(payload["summary"], body_style),
            Spacer(1, 0.15 * cm),
            Paragraph(
                f"Метод классификации: {classification_method}. "
                "Класс по экспертным правилам — в summary выше.",
                body_style,
            ),
            Spacer(1, 0.45 * cm),
            Paragraph("Метрики", body_style),
            Spacer(1, 0.15 * cm),
            metrics_table,
        ]
    )

    if cnn_probs:
        story.extend(
            [
                Spacer(1, 0.15 * cm),
                Paragraph(
                    format_classifier_note(cnn_class, float(cnn_conf or 0)),
                    body_style,
                ),
            ]
        )

    if cnn_table is not None:
        story.extend(
            [
                Spacer(1, 0.35 * cm),
                Paragraph("CNN-классификатор", body_style),
                Spacer(1, 0.15 * cm),
                cnn_table,
            ]
        )

    story.extend(
        [
            Spacer(1, 0.45 * cm),
            Paragraph("Цветовая маска: зелёный — обычные срастания, красный — тонкие, синий — тальк.", body_style),
            Spacer(1, 0.2 * cm),
            overlay,
        ]
    )
    doc.build(story)
    return buffer.getvalue()
