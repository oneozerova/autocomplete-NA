"""Minimal Streamlit UI: one field with instant inline ghost text, TAB to accept.

Completion runs entirely in the browser (a compact model is baked into the page),
so ghost text appears with zero network round-trips — sub-millisecond per
keystroke. The full Python `Completer` remains the integration API; this page is
just its fast demo front-end.

    streamlit run app.py           # build the web model first: python -m src.export_web
"""
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parent
WEB_MODEL = ROOT / "models" / "web_model.json"
TEMPLATE = ROOT / "frontend" / "ghost.html"
MERGED = ROOT / "frontend" / "prompt.html"

st.set_page_config(page_title="RU автодополнение", page_icon="✍️", layout="centered")


@st.cache_data(show_spinner="Загрузка модели…")
def build(template: Path) -> str:
    if not WEB_MODEL.exists():
        st.stop()
    model_json = WEB_MODEL.read_text(encoding="utf-8")
    html = template.read_text(encoding="utf-8")
    return html.replace("/*__MODEL__*/", model_json)


st.markdown("#### Промпт для генерации изображения")

if not WEB_MODEL.exists():
    st.error("Нет web-модели. Соберите её: `python -m src.export_web`")
else:
    st.iframe(build(TEMPLATE), height=110)
    st.caption(
        "Дополнение окончаний работает в браузере (мгновенно), с учётом контекста "
        "и грамматического согласования. Примеры: «красивые девушк», «в тёмном лес», "
        "«воины сто», «женщина котор»."
    )

    st.divider()
    st.markdown("#### Объединённое поле: ghost-подсказки + инлайн-параметры")
    st.caption(
        "То же браузерное дополнение окончаний (Tab — принять), но числа, проценты, "
        "цвета, углы и пресеты в тексте становятся редактируемыми виджетами прямо "
        "в промпте — как в концепте Create Game."
    )
    st.iframe(build(MERGED), height=200)
