import requests
import pandas as pd

from app.config import settings


BASE_URL = "https://finnhub.io/api/v1"


def get_quote(symbol: str) -> dict:
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
    resolution: str,
    from_timestamp: int,
    to_timestamp: int,
) -> pd.DataFrame:

    url = f"{BASE_URL}/stock/candle"

    params = {
        "symbol": symbol,
        "resolution": resolution,
        "from": from_timestamp,
        "to": to_timestamp,
        "token": settings.finnhub_api_key,
    }

    response = requests.get(
        url,
        params=params,
        timeout=10,
    )

    response.raise_for_status()

    data = response.json()

    if data.get("s") != "ok":
        raise ValueError(
            f"Finnhub returned an unsuccessful response: {data}"
        )

    return pd.DataFrame({
        "timestamp": pd.to_datetime(data["t"], unit="s"),
        "open": data["o"],
        "high": data["h"],
        "low": data["l"],
        "close": data["c"],
        "volume": data["v"],
    })