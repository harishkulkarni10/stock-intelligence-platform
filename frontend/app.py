"""Minimal Streamlit surface for /analyze."""

from __future__ import annotations

import os

import requests
import streamlit as st

API_URL = os.getenv("API_URL", os.getenv("API_BASE_URL", "http://localhost:8000"))

st.set_page_config(page_title="Stock Intelligence Platform", layout="wide")
st.title("Stock Intelligence Platform")
st.caption("Stage 2 analyze UI — forecast + news + report agents")

ticker = st.text_input("Ticker", value="NVDA").strip().upper()
if st.button("Analyze", type="primary") and ticker:
    with st.spinner("Running forecast + agents..."):
        response = requests.post(
            f"{API_URL.rstrip('/')}/analyze",
            json={"ticker": ticker},
            timeout=600,
        )
    if response.status_code != 200:
        st.error(f"{response.status_code}: {response.text}")
    else:
        payload = response.json()
        st.write("Status:", payload.get("status"), "| Cached:", payload.get("cached"))
        st.write(
            "Recommendation:",
            payload.get("recommendation"),
            "| Confidence:",
            payload.get("confidence"),
        )
        preds = payload.get("predictions") or {}
        st.subheader("Forecast")
        st.json(preds.get("forecast") or preds)
        st.subheader("Performance analysis")
        st.write(payload.get("performance_analysis") or "")
        st.subheader("News summary")
        st.write(payload.get("news_summary") or "")
        st.subheader("Final report")
        st.markdown(payload.get("final_report") or payload.get("detail") or "")
