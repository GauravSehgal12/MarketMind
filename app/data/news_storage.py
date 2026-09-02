from sqlalchemy import text

from app.database.connection import engine


def save_news_articles(articles: list[dict]) -> int:
    """
    Save news articles into PostgreSQL.

    Existing articles are ignored using article_id.
    """

    if not articles:
        return 0

    query = text("""
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
    """)

    inserted = 0

    with engine.begin() as connection:

        for article in articles:

            result = connection.execute(
                query,
                article
            )

            inserted += result.rowcount

    return inserted