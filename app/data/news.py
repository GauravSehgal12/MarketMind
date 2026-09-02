import hashlib
from datetime import datetime, timedelta

from newsapi import NewsApiClient

from app.config import settings


def create_article_id(url: str) -> str:
    """
    Create a stable unique ID for an article
    using its URL.
    """
    return hashlib.sha256(
        url.encode("utf-8")
    ).hexdigest()


def is_relevant_to_stock(
    article: dict,
    symbol: str,
    company_name: str,
) -> bool:
    """
    Check whether an article is actually
    relevant to the requested stock.
    """

    title = (
        article.get("title") or ""
    ).lower()

    description = (
        article.get("description") or ""
    ).lower()

    content = (
        article.get("content") or ""
    ).lower()

    text = (
        title
        + " "
        + description
        + " "
        + content
    )

    symbol = symbol.lower()
    company_name = company_name.lower()

    # Strong relevance indicators
    strong_keywords = [
        company_name,
        f"{symbol} stock",
        f"{company_name} stock",
        f"{company_name} shares",
        f"{company_name} earnings",
        f"{company_name} revenue",
        f"{company_name} gpu",
        f"{company_name} chips",
    ]

    for keyword in strong_keywords:

        if keyword in text:
            return True

    return False


def get_news(
    symbol: str,
    company_name: str,
    days: int = 7,
    page_size: int = 100,
) -> list[dict]:
    """
    Fetch financial news related to a stock/company.
    """

    if not settings.news_api_key:
        raise ValueError(
            "NEWS_API_KEY is not configured in .env"
        )

    newsapi = NewsApiClient(
        api_key=settings.news_api_key
    )

    # -----------------------------
    # Date range
    # -----------------------------

    to_date = datetime.utcnow()

    from_date = (
        to_date - timedelta(days=days)
    )

    # -----------------------------
    # Search query
    # -----------------------------

    query = (
        f'"{company_name}" OR "{symbol}"'
    )

    # -----------------------------
    # Fetch articles
    # -----------------------------

    response = newsapi.get_everything(
        q=query,
        from_param=from_date.strftime(
            "%Y-%m-%dT%H:%M:%S"
        ),
        to=to_date.strftime(
            "%Y-%m-%dT%H:%M:%S"
        ),
        language="en",
        sort_by="publishedAt",
        page_size=page_size,
    )

    if response.get("status") != "ok":
        raise RuntimeError(
            f"NewsAPI error: {response}"
        )

    # -----------------------------
    # Process articles
    # -----------------------------

    articles = []

    for article in response.get(
        "articles", []
    ):

        url = article.get("url")

        if not url:
            continue

        # -------------------------
        # Relevance filtering
        # -------------------------

        if not is_relevant_to_stock(
            article,
            symbol,
            company_name,
        ):
            continue

        source = article.get(
            "source",
            {},
        )

        article_data = {
            "article_id": create_article_id(
                url
            ),

            "symbol": symbol,

            "source": source.get(
                "name"
            ),

            "title": article.get(
                "title"
            ),

            "description": article.get(
                "description"
            ),

            "content": article.get(
                "content"
            ),

            "url": url,

            "published_at": datetime.fromisoformat(
    article["publishedAt"].replace(
        "Z",
        "+00:00"
    )
).replace(tzinfo=None),
        }

        articles.append(
            article_data
        )

    return articles