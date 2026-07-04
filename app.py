"""Локальный веб-интерфейс для анализа OM-снимков руды.

Запуск:
    streamlit run app.py
"""

from __future__ import annotations

import base64
import html
import tempfile
from io import BytesIO
from pathlib import Path

import cv2
import streamlit as st
import streamlit.components.v1 as components
from PIL import Image, ImageOps

from analyze import analyze_image
from src.metrics import format_classifier_note
from src.report import build_report_payload, create_pdf_report, report_json_bytes


PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_TALC_MODEL = PROJECT_DIR / "models" / "best_talc_unet.pth"
DEFAULT_CLASSIFIER_MODEL = PROJECT_DIR / "models" / "best_ore_resnet18.pth"
LOGO_PATH = PROJECT_DIR / "nornickel-vector-logo-seeklogo" / "nornickel-seeklogo.png"
MAX_PREVIEW_SIDE = 1800

# Панорамные микрофото могут быть очень большими. Для отображения делаем
# отдельный preview, но анализ выполняется по исходному загруженному файлу.
Image.MAX_IMAGE_PIXELS = None


def _save_uploaded_file(uploaded_file) -> Path:
    suffix = Path(uploaded_file.name).suffix or ".png"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded_file.getbuffer())
        return Path(tmp.name)


def _bgr_to_rgb(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def _pil_to_data_uri(img: Image.Image) -> str:
    out = BytesIO()
    img.save(out, format="PNG")
    encoded = base64.b64encode(out.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def _resize_for_preview(img: Image.Image, max_side: int = MAX_PREVIEW_SIDE) -> Image.Image:
    img = img.copy()
    if max(img.size) > max_side:
        img.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
    return img


def _uploaded_to_data_uri(uploaded_file) -> tuple[str, tuple[int, int]]:
    img = Image.open(BytesIO(uploaded_file.getvalue())).convert("RGB")
    preview = _resize_for_preview(img)
    return _pil_to_data_uri(preview), preview.size


def _uploaded_preview_image(uploaded_file) -> Image.Image:
    img = Image.open(BytesIO(uploaded_file.getvalue())).convert("RGB")
    return _resize_for_preview(img)


def _array_to_data_uri(img) -> tuple[str, tuple[int, int]]:
    if img.ndim == 2:
        pil_img = Image.fromarray(img).convert("RGB")
    else:
        pil_img = Image.fromarray(img).convert("RGB")
    preview = _resize_for_preview(pil_img)
    return _pil_to_data_uri(preview), preview.size


def _logo_with_padding(path: Path, padding: int = 28) -> BytesIO:
    logo = Image.open(path).convert("RGBA")
    padded = ImageOps.expand(logo, border=padding, fill=(255, 255, 255, 0))
    out = BytesIO()
    padded.save(out, format="PNG")
    out.seek(0)
    return out


def _zoom_viewer(title: str, data_uri: str, image_size: tuple[int, int], height: int = 650) -> None:
    safe_title = html.escape(title)
    width, height_px = image_size
    components.html(
        f"""
        <div style="font-family: system-ui, -apple-system, BlinkMacSystemFont, sans-serif;">
          <div style="display:flex; gap:12px; align-items:center; margin-bottom:10px;">
            <b>{safe_title}</b>
            <button id="zoomOut" type="button">-</button>
            <input id="zoom" type="range" min="25" max="500" value="100" step="25" style="width:260px;">
            <button id="zoomIn" type="button">+</button>
            <button id="reset" type="button">Сброс</button>
            <span id="zoomLabel">100%</span>
          </div>
          <div style="font-size:13px; color:#666; margin-bottom:8px;">
            Колесо мыши над картинкой тоже меняет масштаб. Перетаскивайте изображение мышью, чтобы изучать детали.
          </div>
          <div id="viewport"
               style="height:{height}px; overflow:auto; border:1px solid #ddd; border-radius:10px; background:#f7f7f7; cursor:grab;">
            <img id="image" src="{data_uri}" style="width:{width}px; height:auto; display:block; user-select:none;">
          </div>
        </div>
        <script>
          const naturalWidth = {width};
          const naturalHeight = {height_px};
          const slider = document.getElementById("zoom");
          const label = document.getElementById("zoomLabel");
          const image = document.getElementById("image");
          const viewport = document.getElementById("viewport");
          const zoomOut = document.getElementById("zoomOut");
          const zoomIn = document.getElementById("zoomIn");
          const reset = document.getElementById("reset");

          function applyZoom(value) {{
            const zoom = Math.max(25, Math.min(500, Number(value)));
            slider.value = zoom;
            label.textContent = zoom + "%";
            image.style.width = Math.round(naturalWidth * zoom / 100) + "px";
          }}

          slider.addEventListener("input", () => applyZoom(slider.value));
          zoomOut.addEventListener("click", () => applyZoom(Number(slider.value) - 25));
          zoomIn.addEventListener("click", () => applyZoom(Number(slider.value) + 25));
          reset.addEventListener("click", () => {{
            applyZoom(100);
            viewport.scrollLeft = 0;
            viewport.scrollTop = 0;
          }});

          viewport.addEventListener("wheel", (event) => {{
            event.preventDefault();
            const delta = event.deltaY < 0 ? 25 : -25;
            applyZoom(Number(slider.value) + delta);
          }}, {{ passive: false }});

          let dragging = false;
          let startX = 0;
          let startY = 0;
          let scrollLeft = 0;
          let scrollTop = 0;

          viewport.addEventListener("mousedown", (event) => {{
            dragging = true;
            viewport.style.cursor = "grabbing";
            startX = event.pageX - viewport.offsetLeft;
            startY = event.pageY - viewport.offsetTop;
            scrollLeft = viewport.scrollLeft;
            scrollTop = viewport.scrollTop;
          }});
          viewport.addEventListener("mouseleave", () => {{
            dragging = false;
            viewport.style.cursor = "grab";
          }});
          viewport.addEventListener("mouseup", () => {{
            dragging = false;
            viewport.style.cursor = "grab";
          }});
          viewport.addEventListener("mousemove", (event) => {{
            if (!dragging) return;
            event.preventDefault();
            const x = event.pageX - viewport.offsetLeft;
            const y = event.pageY - viewport.offsetTop;
            viewport.scrollLeft = scrollLeft - (x - startX);
            viewport.scrollTop = scrollTop - (y - startY);
          }});
        </script>
        """,
        height=height + 100,
    )


def _comparison_magnifier(
    original_uri: str,
    overlay_uri: str,
    image_size: tuple[int, int],
    zoom: float = 2.6,
    lens_size: int = 170,
) -> None:
    width, height_px = image_size
    # Streamlit components do not auto-resize to their HTML content.
    # Use a generous estimate because the app runs in wide layout.
    estimated_card_width = 900
    estimated_image_height = int(estimated_card_width * height_px / max(width, 1))
    component_height = min(1800, max(980, estimated_image_height + 360))
    components.html(
        f"""
        <style>
          .magnifier-grid {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 18px;
            font-family: system-ui, -apple-system, BlinkMacSystemFont, sans-serif;
          }}
          .magnifier-card {{
            border: 1px solid #ddd;
            border-radius: 12px;
            padding: 12px 12px 22px 12px;
            background: #fff;
          }}
          .magnifier-title {{
            font-weight: 700;
            margin-bottom: 10px;
          }}
          .magnifier-wrap {{
            position: relative;
            overflow: hidden;
            border: 1px solid #cfcfcf;
            border-radius: 10px;
            background: #cfcfcf;
            box-sizing: border-box;
            padding: 2px;
          }}
          .magnifier-wrap::after {{
            content: "";
            position: absolute;
            inset: 2px;
            border: 1px solid #cfcfcf;
            border-radius: 8px;
            pointer-events: none;
            z-index: 20;
          }}
          .magnifier-wrap img {{
            width: 100%;
            height: auto;
            display: block;
            object-fit: contain;
            border-radius: 8px;
            user-select: none;
            pointer-events: none;
          }}
          .lens {{
            width: {lens_size}px;
            height: {lens_size}px;
            border: 3px solid white;
            border-radius: 50%;
            box-shadow: 0 6px 22px rgba(0,0,0,0.38);
            position: absolute;
            display: none;
            pointer-events: none;
            background-repeat: no-repeat;
            z-index: 10;
          }}
          .hint {{
            margin: 8px 0 14px;
            color: #666;
            font-size: 13px;
          }}
          @media (max-width: 900px) {{
            .magnifier-grid {{
              grid-template-columns: 1fr;
            }}
          }}
        </style>

        <div class="hint">
          Наведите мышь на исходное изображение или цветовую маску: появится лупа с увеличенным фрагментом.
        </div>
        <div class="magnifier-grid">
          <div class="magnifier-card">
            <div class="magnifier-title">Исходное изображение</div>
            <div class="magnifier-wrap" data-src="{original_uri}">
              <img src="{original_uri}" alt="Исходное изображение">
              <div class="lens"></div>
            </div>
          </div>
          <div class="magnifier-card">
            <div class="magnifier-title">Цветовая маска</div>
            <div class="magnifier-wrap" data-src="{overlay_uri}">
              <img src="{overlay_uri}" alt="Цветовая маска">
              <div class="lens"></div>
            </div>
          </div>
        </div>

        <script>
          const zoom = {zoom};
          const lensSize = {lens_size};

          document.querySelectorAll(".magnifier-wrap").forEach((wrap) => {{
            const img = wrap.querySelector("img");
            const lens = wrap.querySelector(".lens");
            const src = wrap.dataset.src;

            function moveLens(event) {{
              const rect = img.getBoundingClientRect();
              const x = Math.max(0, Math.min(event.clientX - rect.left, rect.width));
              const y = Math.max(0, Math.min(event.clientY - rect.top, rect.height));

              lens.style.display = "block";
              lens.style.left = `${{x - lensSize / 2}}px`;
              lens.style.top = `${{y - lensSize / 2}}px`;
              lens.style.backgroundImage = `url("${{src}}")`;
              lens.style.backgroundSize = `${{rect.width * zoom}}px ${{rect.height * zoom}}px`;
              lens.style.backgroundPosition = `-${{x * zoom - lensSize / 2}}px -${{y * zoom - lensSize / 2}}px`;
            }}

            wrap.addEventListener("mousemove", moveLens);
            wrap.addEventListener("mouseenter", moveLens);
            wrap.addEventListener("mouseleave", () => {{
              lens.style.display = "none";
            }});
          }});
        </script>
        """,
        height=component_height,
    )


st.set_page_config(
    page_title="Анализ руды",
    layout="wide",
)

if LOGO_PATH.exists():
    st.image(_logo_with_padding(LOGO_PATH), width=190)

st.title("Анализ OM-снимков руды")
st.caption(
    "Загрузите TIFF/PNG/JPEG изображение. Система построит цветовую маску, "
    "посчитает доли фаз и сформирует отчёт."
)

with st.sidebar:
    st.header("Параметры анализа")
    use_unet = st.checkbox(
        "Использовать U-Net для талька",
        value=DEFAULT_TALC_MODEL.exists(),
        help="Если модель не найдена или отключена, используется эвристика по тёмным/синим областям.",
    )
    talc_model = st.text_input("Путь к модели талька", value=str(DEFAULT_TALC_MODEL))
    auto_talc_threshold = st.checkbox(
        "Автоматический порог U-Net",
        value=True,
        help="Берёт калиброванный порог из models/talc_unet_config.json и адаптирует его под снимок.",
    )
    if auto_talc_threshold:
        talc_threshold: float | str = "auto"
    else:
        talc_threshold = st.slider("Порог талька", min_value=0.01, max_value=0.5, value=0.05, step=0.01)
    talc_heuristic = "legacy"
    unet_max_side_value = st.select_slider(
        "Размер для U-Net-инференса",
        options=[0, 768, 1024, 1536, 2048, 3072, 4096],
        value=1536,
        format_func=lambda value: "Полный размер" if value == 0 else f"до {value}px",
        help="Меньше = быстрее на CPU. 0 запускает U-Net на исходном размере без уменьшения.",
    )
    unet_max_side = None if unet_max_side_value == 0 else int(unet_max_side_value)

    st.divider()
    st.header("Дополнительная CNN-проверка")
    use_classifier = st.checkbox(
        "Запустить CNN-классификатор",
        value=DEFAULT_CLASSIFIER_MODEL.exists(),
        help="CNN-классификатор зачастую даёт более значимый ориентир при классификации руды.",
    )
    classifier_model = st.text_input("Путь к CNN-модели", value=str(DEFAULT_CLASSIFIER_MODEL))

    talc_model_path = Path(talc_model) if talc_model else None
    classifier_model_path = Path(classifier_model) if classifier_model else None
    if use_unet and talc_model_path and talc_model_path.exists():
        st.success("U-Net: будет запущен инференс")
    elif use_unet:
        st.warning("U-Net модель не найдена, будет fallback на эвристику")
    else:
        st.info("U-Net отключена, будет эвристика")

    if use_classifier and classifier_model_path and classifier_model_path.exists():
        st.success("CNN: будет запущен инференс")
    elif use_classifier:
        st.warning("CNN модель не найдена, CNN будет пропущена")

    st.divider()
    st.markdown(
        "**Легенда маски**  \n"
        "Зелёный — обычные срастания  \n"
        "Красный — тонкие срастания  \n"
        "Синий — тальк"
    )

uploaded_file = st.file_uploader(
    "Загрузите изображение",
    type=["png", "jpg", "jpeg", "tif", "tiff"],
)

if uploaded_file is None:
    st.info("Загрузите изображение, чтобы запустить анализ.")
    st.stop()

is_new_upload = st.session_state.get("last_uploaded_name") != uploaded_file.name
run = st.button("Анализировать", type="primary", width="stretch")

if not run and ("last_analysis" not in st.session_state or is_new_upload):
    st.image(_uploaded_preview_image(uploaded_file), caption="Загруженное изображение (preview)", width="stretch")
    st.stop()

if run:
    input_path = _save_uploaded_file(uploaded_file)
    model_path = Path(talc_model) if talc_model else None
    cnn_model_path = Path(classifier_model) if classifier_model else None

    with st.spinner("Идёт анализ изображения..."):
        try:
            analysis = analyze_image(
                input_path,
                talc_model=model_path,
                classifier_model=cnn_model_path,
                talc_threshold=talc_threshold,
                talc_heuristic=talc_heuristic,
                unet_max_side=unet_max_side,
                use_unet=use_unet,
                use_classifier=use_classifier,
            )
        except Exception as exc:
            st.error(f"Не удалось выполнить анализ: {exc}")
            st.stop()

        payload = build_report_payload(
            analysis,
            source_name=uploaded_file.name,
            talc_model=model_path,
            use_unet=use_unet,
            talc_threshold=talc_threshold,
            talc_heuristic=talc_heuristic,
            unet_max_side=unet_max_side,
            classifier_model=cnn_model_path,
            use_classifier=use_classifier,
        )
        pdf_bytes = create_pdf_report(payload, analysis["overlay"], logo_path=LOGO_PATH)
        json_bytes = report_json_bytes(payload)

    st.session_state["last_analysis"] = analysis
    st.session_state["last_payload"] = payload
    st.session_state["last_pdf"] = pdf_bytes
    st.session_state["last_json"] = json_bytes
    st.session_state["last_uploaded_name"] = uploaded_file.name

analysis = st.session_state["last_analysis"]
payload = st.session_state["last_payload"]
metrics = payload["metrics"]

st.success(payload["summary"])
st.caption(f"Метод классификации: {payload['classification_method']}. Класс по экспертным правилам — в summary выше.")
threshold_used = payload["parameters"].get("talc_threshold_used")
if threshold_used is not None:
    st.caption(f"Фактически использованный порог U-Net для талька: {float(threshold_used):.3f}")
if payload.get("classifier_probabilities"):
    cnn_class_ru = payload.get("classifier_class_ru") or payload.get("classifier_class")
    st.info(
        format_classifier_note(
            cnn_class_ru,
            float(payload.get("classifier_confidence") or 0),
        )
    )

col1, col2, col3, col4 = st.columns(4)
col1.metric("Тальк", f"{metrics['talc_pct']:.2f}%")
col2.metric("Сульфиды всего", f"{metrics['sulfide_pct']:.2f}%")
col3.metric("Обычные", f"{metrics['ordinary_pct']:.2f}%")
col4.metric("Тонкие", f"{metrics['fine_pct']:.2f}%")

if payload.get("classifier_probabilities"):
    cnn_col1, cnn_col2 = st.columns(2)
    cnn_col1.metric("Класс по CNN-классификатору", payload.get("classifier_class_ru") or payload.get("classifier_class"))
    cnn_col2.metric("Уверенность CNN", f"{float(payload.get('classifier_confidence') or 0):.3f}")

st.subheader("Исходное изображение и цветовая маска")
original_uri, original_size = _uploaded_to_data_uri(uploaded_file)
overlay_uri, _ = _array_to_data_uri(_bgr_to_rgb(analysis["overlay"]))
_comparison_magnifier(original_uri, overlay_uri, original_size)

st.subheader("Метрики")
st.dataframe(
    [
        {"Показатель": "Класс руды", "Значение": payload["ore_class_ru"]},
        {
            "Показатель": "Класс по CNN-классификатору",
            "Значение": payload.get("classifier_class_ru") or payload.get("classifier_class") or "не запускалась",
        },
        {"Показатель": "Тальк, %", "Значение": f"{metrics['talc_pct']:.2f}"},
        {"Показатель": "Сульфиды всего, %", "Значение": f"{metrics['sulfide_pct']:.2f}"},
        {"Показатель": "Обычные срастания, %", "Значение": f"{metrics['ordinary_pct']:.2f}"},
        {"Показатель": "Тонкие срастания, %", "Значение": f"{metrics['fine_pct']:.2f}"},
        {"Показатель": "Доля обычных среди сульфидов, %", "Значение": f"{metrics['ordinary_share']:.1f}"},
        {"Показатель": "Доля тонких среди сульфидов, %", "Значение": f"{metrics['fine_share']:.1f}"},
        {"Показатель": "Метод определения талька", "Значение": metrics["talc_method"]},
        {"Показатель": "Выбранная эвристика талька", "Значение": payload["parameters"]["talc_heuristic"]},
        {
            "Показатель": "Размер U-Net-инференса",
            "Значение": (
                "полный размер"
                if payload["parameters"].get("unet_max_side") is None
                else f"до {payload['parameters']['unet_max_side']}px"
            ),
        },
        {
            "Показатель": "Порог U-Net",
            "Значение": f"{float(threshold_used):.3f}" if threshold_used is not None else "не использовался",
        },
    ],
    hide_index=True,
    width="stretch",
)

if payload.get("classifier_probabilities"):
    st.subheader("CNN-классификатор")
    st.dataframe(
        [
            {"Класс": label, "Вероятность": f"{prob:.3f}"}
            for label, prob in payload["classifier_probabilities"].items()
        ],
        hide_index=True,
        width="stretch",
    )

st.subheader("Скачать отчёт")
download_json, download_pdf = st.columns(2)
with download_json:
    st.download_button(
        "Скачать JSON",
        data=st.session_state["last_json"],
        file_name=f"{Path(uploaded_file.name).stem}_report.json",
        mime="application/json",
        width="stretch",
    )
with download_pdf:
    st.download_button(
        "Скачать PDF",
        data=st.session_state["last_pdf"],
        file_name=f"{Path(uploaded_file.name).stem}_report.pdf",
        mime="application/pdf",
        width="stretch",
    )
