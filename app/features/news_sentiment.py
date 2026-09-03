from sqlalchemy import create_engine, text
import pandas as pd

from app.config import settings
from app.agents.sentiment_agent import analyze_sentiment


def score_unscored_news():
    """
    Score all news articles that do not yet have sentiment.
    """

    engine = create_engine(
        settings.database_url
    )

    select_sql = text(
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

    update_sql = text(
        """
        UPDATE news_articles
        SET
            sentiment_score = :score,
            sentiment_label = :label
        WHERE id = :id
        """
    )

    with engine.begin() as connection:

        rows = connection.execute(
            select_sql
        ).mappings().all()

        print(
            f"Unscored articles: {len(rows)}"
        )

        scored = 0
        failed = 0

        for article in rows:

            title = article["title"] or ""
            description = article["description"] or ""
            content = article["content"] or ""

            text_to_analyze = (
                f"Title: {title}\n"
                f"Description: {description}\n"
                f"Content: {content}"
            )

            try:

                result = analyze_sentiment(
                    text_to_analyze
                )

                print(
                    "\nRaw sentiment result:"
                )
                print(result)

                # ==================================
                # Handle dictionary response
                # ==================================

                if isinstance(result, dict):

                    score = result.get(
                        "score"
                    )

                    label = result.get(
                        "label"
                    )

                # ==================================
                # Handle tuple response
                # ==================================

                elif isinstance(
                    result,
                    tuple
                ):

                    score = result[0]
                    label = result[1]

                # ==================================
                # Handle list response
                # ==================================

                elif isinstance(
                    result,
                    list
                ):

                    score = result[0]
                    label = result[1]

                else:

                    raise ValueError(
                        f"Unexpected sentiment "
                        f"response type: "
                        f"{type(result)}"
                    )

                # ==================================
                # Validate result
                # ==================================

                if score is None:

                    raise ValueError(
                        f"Sentiment result has "
                        f"no score: {result}"
                    )

                if label is None:

                    raise ValueError(
                        f"Sentiment result has "
                        f"no label: {result}"
                    )

                score = float(score)

                label = str(
                    label
                ).lower()

                # ==================================
                # Store result
                # ==================================

                connection.execute(
                    update_sql,
                    {
                        "id": article["id"],
                        "score": score,
                        "label": label,
                    },
                )

                scored += 1

                print(
                    f"SUCCESS | "
                    f"{score:.2f} | "
                    f"{label} | "
                    f"{title[:80]}"
                )

            except Exception as exc:

                failed += 1

                print(
                    f"FAILED article "
                    f"{article['id']}: "
                    f"{exc}"
                )

    print(
        "\n============================"
    )

    print(
        "Sentiment Scoring Complete"
    )

    print(
        "============================"
    )

    print(
        f"Scored:  {scored}"
    )

    print(
        f"Failed:  {failed}"
    )