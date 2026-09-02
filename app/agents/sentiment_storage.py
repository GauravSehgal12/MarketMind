from sqlalchemy import text

from app.database.connection import engine
from app.agents.sentiment_agent import analyze_sentiment


def analyze_pending_articles(limit: int = 20) -> int:
    """
    Analyze news articles that do not have sentiment yet
    and store the sentiment results in PostgreSQL.
    """

    # Get unanalyzed articles
    with engine.connect() as connection:
        result = connection.execute(
            text("""
                SELECT
                    id,
                    title,
                    description,
                    content
                FROM news_articles
                WHERE sentiment_score IS NULL
                ORDER BY published_at DESC
                LIMIT :limit
            """),
            {"limit": limit},
        )

        articles = result.mappings().all()

    print(f"Articles pending sentiment: {len(articles)}")

    processed = 0

    for article in articles:

        try:
            sentiment = analyze_sentiment(
                title=article["title"] or "",
                description=article["description"],
                content=article["content"],
            )

            with engine.begin() as connection:
                connection.execute(
                    text("""
                        UPDATE news_articles
                        SET
                            sentiment_score = :score,
                            sentiment_label = :label
                        WHERE id = :id
                    """),
                    {
                        "score": sentiment["sentiment_score"],
                        "label": sentiment["sentiment_label"],
                        "id": article["id"],
                    },
                )

            processed += 1

            print(
                f"✓ Article {article['id']} | "
                f"{sentiment['sentiment_label']} | "
                f"{sentiment['sentiment_score']:.3f}"
            )

        except Exception as e:

            print(
                f"✗ Article {article['id']} failed: {e}"
            )

    return processed