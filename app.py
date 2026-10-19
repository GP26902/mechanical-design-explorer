"""Mechanical Design Explorer - base shell.

Auto-discovers every chapter in chapters/ (see common/discovery.py). To add a chapter, drop one
file into chapters/ (copy chapters/_template.py). No edits to this file are needed.
"""
from pathlib import Path

import streamlit as st

from common.discovery import discover_chapters
from common.ui import render_safely, show_disclaimer, unit_selector

CHAPTERS_DIR = Path(__file__).parent / "chapters"

st.set_page_config(page_title="Mechanical Design Explorer", page_icon="🛠️", layout="wide")

chapters = discover_chapters(CHAPTERS_DIR)
loaded = [c for c in chapters if c.ok]
broken = [c for c in chapters if not c.ok]

st.sidebar.title("🛠️ Mechanical Design Explorer")
if not loaded:
    st.sidebar.info("No chapters found.")
    st.title("Mechanical Design Explorer")
    st.info("Add a chapter file to the `chapters/` folder (copy `_template.py`).")
else:
    choice = st.sidebar.selectbox("Chapter", loaded, format_func=lambda c: c.title)
    unit_selector()
    st.title(choice.title)
    st.caption(choice.description)
    show_disclaimer()
    render_safely(choice.render)

for c in broken:
    st.sidebar.error(c.error)
