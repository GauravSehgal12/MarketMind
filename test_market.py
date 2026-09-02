import requests

from app.config import settings


url = "https://finnhub.io/api/v1/quote"

params = {
    "symbol": "NVDA",
    "token": settings.finnhub_api_key,
}

response = requests.get(
    url,
    params=params,
    timeout=10,
)

print("Status:", response.status_code)
print("Response:", response.text)