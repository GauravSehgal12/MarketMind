from datetime import datetime, timedelta, timezone
import hashlib

from newsapi import NewsApiClient
from sqlalchemy import create_engine, text

from app.config import settings


def ingest_historical_news(
    symbol: str = "NVDA",
    company_name: str = "NVIDIA",
    days: int = 30,
    window_days: int = 3,
):
    """
    Fetch historical NVIDIA news in small date windows and
    store relevant articles in PostgreSQL.

    Small windows avoid NewsAPI's 100-result developer-plan
    limitation.
    """

    if not settings.news_api_key:
        raise ValueError(
            "NEWS_API_KEY is not configured."
        )

    newsapi = NewsApiClient(
        api_key=settings.news_api_key
    )

    end_date = datetime.now(timezone.utc)
    start_date = end_date - timedelta(days=days)

    print("\nHistorical News Ingestion")
    print("=========================")
    print(f"From: {start_date.date()}")
    print(f"To:   {end_date.date()}")

    # --------------------------------------------------
    # Search query
    # --------------------------------------------------

    query = (
        f'"{company_name}" OR "{symbol}"'
    )

    print(f"\nQuery: {query}")
    print("Filtering: title")

    # --------------------------------------------------
    # Database
    # --------------------------------------------------

    engine = create_engine(
        settings.database_url
    )

    insert_sql = text(
        """
        INSERT INTO news_articles (
            article_id,
            symbol,
            source,
            title,
            description,
            content,
            url,
            published_at
        )
        VALUES (
            :article_id,
            :symbol,
            :source,
            :title,
            :description,
            :content,
            :url,
            :published_at
        )
        ON CONFLICT (article_id)
        DO NOTHING
        """
    )

    total_fetched = 0
    total_relevant = 0
    total_inserted = 0
    total_skipped = 0

    current_start = start_date

    # --------------------------------------------------
    # Fetch small windows
    # --------------------------------------------------

    while current_start < end_date:

        current_end = min(
            current_start
            + timedelta(days=window_days),
            end_date,
        )

        print(
            f"\nFetching:"
            f" {current_start.date()}"
            f" -> {current_end.date()}"
        )

        try:

            response = newsapi.get_everything(
                q=query,
                from_param=current_start.strftime(
                    "%Y-%m-%dT%H:%M:%S"
                ),
                to=current_end.strftime(
                    "%Y-%m-%dT%H:%M:%S"
                ),
                language="en",
                sort_by="publishedAt",
                page_size=100,
                page=1,
            )

        except Exception as e:

            print(
                f"API error for window "
                f"{current_start.date()} -> "
                f"{current_end.date()}"
            )

            print(e)

            current_start = current_end
            continue

        if response.get("status") != "ok":

            print(
                "NewsAPI returned an error:"
            )

            print(response)

            current_start = current_end
            continue

        articles = response.get(
            "articles",
            []
        )

        total_fetched += len(articles)

        print(
            f"API returned: {len(articles)}"
        )

        # --------------------------------------------------
        # Local relevance filtering
        # --------------------------------------------------

        relevant_articles = []

        for article in articles:

            title = (
                article.get("title")
                or ""
            ).lower()

            # Remove removed/unusable titles
            if not title:
                continue

            # Require NVIDIA/NVDA in title
            if (
                "nvidia" not in title
                and "nvda" not in title
            ):
                continue

            relevant_articles.append(
                article
            )

        total_relevant += len(
            relevant_articles
        )

        print(
            f"Relevant titles: "
            f"{len(relevant_articles)}"
        )

        # --------------------------------------------------
        # Insert
        # --------------------------------------------------

        with engine.begin() as connection:

            for article in relevant_articles:

                url = article.get("url")

                if not url:
                    total_skipped += 1
                    continue

                article_id = hashlib.sha256(
                    url.encode("utf-8")
                ).hexdigest()

                source = article.get(
                    "source"
                )

                if isinstance(
                    source,
                    dict
                ):
                    source_name = (
                        source.get("name")
                    )
                else:
                    source_name = (
                        str(source)
                        if source
                        else None
                    )

                published_at = article.get(
                    "publishedAt"
                )

                if not published_at:
                    total_skipped += 1
                    continue

                try:

                    published_at = (
                        datetime.fromisoformat(
                            published_at.replace(
                                "Z",
                                "+00:00"
                            )
                        )
                        .replace(
                            tzinfo=None
                        )
                    )

                except ValueError:

                    total_skipped += 1
                    continue

                result = connection.execute(
                    insert_sql,
                    {
                        "article_id":
                            article_id,

                        "symbol":
                            symbol,

                        "source":
                            source_name,

                        "title":
                            article.get(
                                "title"
                            ),

                        "description":
                            article.get(
                                "description"
                            ),

                        "content":
                            article.get(
                                "content"
                            ),

                        "url":
                            url,

                        "published_at":
                            published_at,
                    },
                )

                if result.rowcount == 1:
                    total_inserted += 1
                else:
                    total_skipped += 1

        current_start = current_end

    # --------------------------------------------------
    # Summary
    # --------------------------------------------------

    print(
        "\n========================="
    )

    print(
        "Historical News Complete"
    )

    print(
        "========================="
    )

    print(
        f"API articles fetched: "
        f"{total_fetched}"
    )

    print(
        f"Relevant articles: "
        f"{total_relevant}"
    )

    print(
        f"Inserted: "
        f"{total_inserted}"
    )

    print(
        f"Skipped/duplicates: "
        f"{total_skipped}"
    )