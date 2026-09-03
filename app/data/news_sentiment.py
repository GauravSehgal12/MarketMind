from sqlalchemy import create_engine, text

from app.config import settings
from app.agents.sentiment_agent import analyze_sentiment


def process_news_sentiment():

    engine = create_engine(
        settings.database_url
    )

    processed = 0
    failed = 0

    with engine.begin() as connection:

        rows = connection.execute(
            text(
                """
                SELECT
                    id,
                    title,
                    description,
                    content
                FROM news_articles
                WHERE sentiment_score IS NULL
                ORDER BY published_at
                """
            )
        ).mappings().all()

        print(
            f"\nArticles requiring sentiment: "
            f"{len(rows)}"
        )

        for article in rows:

            article_id = article["id"]

            try:

                # ------------------------------------------------
                # Build text
                # ------------------------------------------------

                title = (
                    article["title"]
                    or ""
                )

                description = (
                    article["description"]
                    or ""
                )

                content = (
                    article["content"]
                    or ""
                )

                text_to_analyze = (
                    f"Title: {title}\n\n"
                    f"Description: {description}\n\n"
                    f"Content: {content}"
                )

                # ------------------------------------------------
                # Sentiment
                # ------------------------------------------------

                result = analyze_sentiment(
                    text_to_analyze
                )

                # ------------------------------------------------
                # Save
                # ------------------------------------------------

                connection.execute(
                    text(
                        """
                        UPDATE news_articles
                        SET
                            sentiment_score =
                                :score,
                            sentiment_label =
                                :label
                        WHERE id = :id
                        """
                    ),
                    {
                        "score": result["score"],
                        "label": result["label"],
                        "id": article_id,
                    }
                )

                processed += 1

                print(
                    f"Article {article_id}: "
                    f"{result['score']} | "
                    f"{result['label']}"
                )

            except Exception as error:

                failed += 1

                print(
                    f"Failed article "
                    f"{article_id}: "
                    f"{error}"
                )

    print(
        "\n======================="
    )

    print(
        "Sentiment Analysis Done"
    )

    print(
        "======================="
    )

    print(
        f"Processed: {processed}"
    )

    print(
        f"Failed:    {failed}"
    )


if __name__ == "__main__":

    process_news_sentiment()