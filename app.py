import os
from pathlib import Path
from dotenv import load_dotenv

# 1. Load environment configuration from explicit .env path
env_path = Path(__file__).resolve().parent / ".env"
load_dotenv(dotenv_path=env_path, override=True)

import streamlit as st
import sqlite3
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import ta
from langchain_core.messages import HumanMessage

# Import compiled LangGraph workflow and helper from apis
from workflow import financial_graph
from apis import format_symbol

# ----------------------------------------------------
# 1. Database Setup (SQLite)
# ----------------------------------------------------
conn = sqlite3.connect("watchlist.db", check_same_thread=False)
c = conn.cursor()
c.execute("CREATE TABLE IF NOT EXISTS watchlist (symbol TEXT PRIMARY KEY)")
conn.commit()

# ----------------------------------------------------
# 2. Page Config & Title
# ----------------------------------------------------
st.set_page_config(page_title="Financial Research Agent (Track A)", page_icon="📈", layout="wide")
st.title("📈 Financial Research Agent")
st.caption("Track A: Financial Stock Analysis with LangGraph & Streamlit")

# ----------------------------------------------------
# 3. Sidebar: Watchlist & SEBI Notice
# ----------------------------------------------------
with st.sidebar:
    st.sidebar.header("📌 Watchlist (SQLite)")
    
    # Add new ticker
    new_ticker = st.text_input("Add Ticker (e.g. RELIANCE, TCS, INFY):")
    if st.button("Add Stock", use_container_width=True) and new_ticker:
        sym = format_symbol(new_ticker)
        c.execute("INSERT OR IGNORE INTO watchlist VALUES (?)", (sym,))
        conn.commit()
        st.rerun()

    # Load saved tickers from database
    c.execute("SELECT symbol FROM watchlist")
    saved_stocks = [r[0] for r in c.fetchall()]
    if not saved_stocks:
        saved_stocks = ["RELIANCE.NS", "TCS.NS", "INFY.NS"]
        for s in saved_stocks:
            c.execute("INSERT OR IGNORE INTO watchlist VALUES (?)", (s,))
        conn.commit()

    selected_stock = st.selectbox("Select Active Stock", saved_stocks)

    # Delete ticker
    if st.button("Delete Stock", use_container_width=True):
        c.execute("DELETE FROM watchlist WHERE symbol = ?", (selected_stock,))
        conn.commit()
        st.rerun()

    st.markdown("---")
    st.sidebar.markdown("**SEBI Disclaimer**: Educational and analytical research only. Not financial advisory.")

# ----------------------------------------------------
# 4. Candlestick & RSI Chart Display
# ----------------------------------------------------
st.subheader(f"📊 Price Chart & Technicals: {selected_stock}")

try:
    ticker_obj = yf.Ticker(selected_stock)
    df = ticker_obj.history(period="6mo")

    if not df.empty:
        # Calculate technical indicators
        df["SMA_20"] = ta.trend.sma_indicator(df["Close"], window=20)
        df["SMA_50"] = ta.trend.sma_indicator(df["Close"], window=50)
        df["RSI"] = ta.momentum.rsi(df["Close"], window=14)

        # Create Plotly subplots (Row 1: Candlestick + SMAs, Row 2: RSI)
        fig = make_subplots(
            rows=2, cols=1, 
            shared_xaxes=True, 
            vertical_spacing=0.08, 
            row_heights=[0.7, 0.3],
            subplot_titles=(f"{selected_stock} Price Chart (INR)", "14-Day RSI")
        )

        fig.add_trace(go.Candlestick(
            x=df.index, open=df["Open"], high=df["High"], low=df["Low"], close=df["Close"], name="Price"
        ), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df["SMA_20"], line=dict(color="orange", width=1.5), name="SMA 20"), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df["SMA_50"], line=dict(color="blue", width=1.5), name="SMA 50"), row=1, col=1)
        
        # RSI chart with 70/30 thresholds
        fig.add_trace(go.Scatter(x=df.index, y=df["RSI"], line=dict(color="purple", width=1.5), name="RSI"), row=2, col=1)
        fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)

        fig.update_layout(height=450, margin=dict(l=10, r=10, t=30, b=10), xaxis_rangeslider_visible=False)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning(f"No price data found for {selected_stock}.")
except Exception as e:
    st.error(f"Error rendering chart: {e}")

# ----------------------------------------------------
# 5. Agentic AI Research Chat Panel
# ----------------------------------------------------
st.markdown("---")
st.subheader("🤖 Financial Research Assistant")

# Store chat history in session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = [
        {"role": "assistant", "content": f"Hello! I'm your Financial Research assistant. I can help you explore Indian equities by pulling fundamentals, technicals, or sentiment for {selected_stock}."}
    ]

# Display past messages
for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Preset prompt buttons
col1, col2, col3 = st.columns(3)
preset_prompt = None
if col1.button(f"🔍 Full Audit for {selected_stock}"):
    preset_prompt = f"Perform a comprehensive financial analysis on {selected_stock}, including fundamentals, technicals, and sentiment."
if col2.button(f"📈 RSI & SMA for {selected_stock}"):
    preset_prompt = f"What are the current RSI and moving averages for {selected_stock}?"
if col3.button(f"📰 News for {selected_stock}"):
    preset_prompt = f"Fetch recent news and sentiment analysis for {selected_stock}."

# User chat input
user_prompt = st.chat_input("Ask about an Indian stock...") or preset_prompt

if user_prompt:
    st.session_state.chat_history.append({"role": "user", "content": user_prompt})
    with st.chat_message("user"):
        st.markdown(user_prompt)

    with st.chat_message("assistant"):
        with st.spinner("Agent running financial research tools..."):
            try:
                # Invoke LangGraph agent
                result = financial_graph.invoke({"messages": [HumanMessage(content=user_prompt)]})
                reply = result["messages"][-1].content
                st.markdown(reply)
                st.session_state.chat_history.append({"role": "assistant", "content": reply})
            except Exception as err:
                error_msg = f"Agent Execution Note: {err}"
                st.error(error_msg)
                st.session_state.chat_history.append({"role": "assistant", "content": error_msg})
