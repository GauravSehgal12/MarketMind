import yfinance as yf
import requests
import pandas as pd

from app.config import settings


BASE_URL = "https://finnhub.io/api/v1"


def get_quote(symbol: str) -> dict:
    """Get current market quote from Finnhub."""

    url = f"{BASE_URL}/quote"

    params = {
        "symbol": symbol,
        "token": settings.finnhub_api_key,
    }

    response = requests.get(
        url,
        params=params,
        timeout=10,
    )

    response.raise_for_status()

    return response.json()


def get_historical_data(
    symbol: str,
    period: str = "2y",
) -> pd.DataFrame:
    """Get historical OHLCV data."""

    df = yf.download(
        symbol,
        period=period,
        interval="1d",
        auto_adjust=False,
        progress=False,
    )

    if df is None or df.empty:
        raise ValueError(
            f"No historical data found for {symbol}"
        )

    # Handle yfinance MultiIndex columns
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df = df.reset_index()

    df = df.rename(
        columns={
            "Date": "timestamp",
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Close": "close",
            "Adj Close": "adj_close",
            "Volume": "volume",
        }
    )

    return df