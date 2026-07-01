"""
website/app.py — FireDetect AI — Main Router
Run: streamlit run website/app.py
"""

import streamlit as st
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

st.set_page_config(
    page_title="FireDetect AI — Human Detection System",
    page_icon="🔥",
    layout="wide",
    initial_sidebar_state="expanded",
)

from utils import load_css, sidebar_nav
from pages.home import render as home
from pages.detect_page import render as detect
from pages.dashboard import render as dashboard
from pages.how_it_works import render as how_it_works

load_css()
page = sidebar_nav()

if   "Home"      in page: home()
elif "Detection" in page: detect()
elif "Dashboard" in page: dashboard()
elif "How"       in page: how_it_works()
