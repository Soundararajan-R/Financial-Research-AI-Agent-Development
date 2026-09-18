import os
import yfinance as yf
import pandas as pd
import ta
import requests
from textblob import TextBlob
from langchain_core.tools import tool

# Helper function to format Indian stock tickers for NSE (.NS)
def format_symbol(symbol: str) -> str:
    symbol = symbol.strip().upper()
    if not (symbol.endswith(".NS") or symbol.endswith(".BO")):
        symbol += ".NS"  # Default to National Stock Exchange (NSE)
    return symbol

# Tool 1: Fetch fundamental ratios using yfinance
@tool
def get_stock_fundamentals(symbol: str) -> dict:
    """Fetches fundamental financial metrics for Indian stocks (e.g. RELIANCE, TCS, INFY)."""
    sym = format_symbol(symbol)
    ticker = yf.Ticker(sym)
    info = ticker.info
    summary_text = str(info.get("longBusinessSummary", "N/A"))
    return {
        "symbol": sym,
        "current_price": info.get("currentPrice") or info.get("regularMarketPrice"),
        "pe_ratio": info.get("trailingPE", "N/A"),
        "market_cap": info.get("marketCap", "N/A"),
        "debt_to_equity": info.get("debtToEquity", "N/A"),
        "summary": summary_text[:300] + "..." if len(summary_text) > 300 else summary_text
    }

# Tool 2: Calculate SMA 20, SMA 50, and 14-day RSI using technical analysis (ta) library
@tool
def get_technical_indicators(symbol: str) -> dict:
    """Calculates SMA 20, SMA 50, and 14-day RSI for a given Indian stock."""
    sym = format_symbol(symbol)
    ticker = yf.Ticker(sym)
    df = ticker.history(period="6mo")
    if df.empty:
        return {"error": "No historical price data found."}
    
    # Calculate indicators using ta library
    df["SMA_20"] = ta.trend.sma_indicator(df["Close"], window=20)
    df["SMA_50"] = ta.trend.sma_indicator(df["Close"], window=50)
    df["RSI"] = ta.momentum.rsi(df["Close"], window=14)
    
    valid_df = df.dropna(subset=["SMA_20", "SMA_50", "RSI", "Close"])
    latest = valid_df.iloc[-1] if not valid_df.empty else df.iloc[-1]
    return {
        "symbol": sym,
        "latest_close": round(float(latest["Close"]), 2) if pd.notna(latest["Close"]) else "N/A",
        "SMA_20": round(float(latest["SMA_20"]), 2) if pd.notna(latest["SMA_20"]) else "N/A",
        "SMA_50": round(float(latest["SMA_50"]), 2) if pd.notna(latest["SMA_50"]) else "N/A",
        "RSI": round(float(latest["RSI"]), 2) if pd.notna(latest["RSI"]) else "N/A"
    }

# Tool 3: Fetch recent news and calculate sentiment score with TextBlob
@tool
def get_stock_news_sentiment(company_name: str) -> dict:
    """Fetches recent news and analyzes market sentiment score (-1 to 1)."""
    api_key = os.getenv("NEWS_API_KEY")
    url = f"https://newsapi.org/v2/everything?q={company_name}&language=en&sortBy=publishedAt&pageSize=5&apiKey={api_key}"
    
    try:
        res = requests.get(url).json()
        articles = res.get("articles", [])
        if not articles:
            return {"sentiment": "Neutral", "score": 0.0, "headlines": []}
        
        scores = []
        headlines = []
        for a in articles:
            title = a.get("title", "")
            headlines.append(title)
            scores.append(TextBlob(title).sentiment.polarity)
            
        avg_score = sum(scores) / len(scores) if scores else 0
        sentiment_label = "Bullish" if avg_score > 0.1 else ("Bearish" if avg_score < -0.1 else "Neutral")
        return {"sentiment": sentiment_label, "score": round(avg_score, 2), "headlines": headlines[:3]}
    except Exception as e:
        return {"sentiment": "Neutral", "score": 0.0, "error": str(e)}
