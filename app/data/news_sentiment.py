import json
import time

from sqlalchemy import create_engine, text
from groq import Groq

from app.config import settings


# ============================================================
# GROQ CLIENT
# ============================================================

client = Groq(
    api_key=settings.groq_api_key
)


# ============================================================
# SENTIMENT ANALYSIS
# ============================================================

def analyze_sentiment(
    title: str,
    description: str | None = None,
    max_retries: int = 3,
):
    """
    Analyze financial-news sentiment using Groq.

    Returns:
        {
            "score": float,
            "label": str
        }

    Score:
        -1.0 = very negative
         0.0 = neutral
         1.0 = very positive
    """

    description = description or ""

    text_to_analyze = (
        f"Headline: {title}\n"
        f"Description: {description}"
    )

    prompt = f"""
You are a financial sentiment analysis system.

Analyze the following news article from the perspective
of its likely impact on the mentioned company/stock.

Return ONLY valid JSON.

The JSON must contain exactly two fields:

"score": a number between -1 and 1
"label": one of "positive", "negative", or "neutral"

Do not include markdown.
Do not include explanations.
Do not include additional fields.

Example:
{{"score": 0.75, "label": "positive"}}

Article:
{text_to_analyze}
"""

    for attempt in range(1, max_retries + 1):

        try:

            print(
                f"\nCalling Groq "
                f"(attempt {attempt}/{max_retries})..."
            )

            response = client.chat.completions.create(
                model=getattr(
                    settings,
                    "groq_model",
                    "llama-3.3-70b-versatile",
                ),
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a financial sentiment "
                            "classification system. "
                            "Return only valid JSON."
                        ),
                    },
                    {
                        "role": "user",
                        "content": prompt,
                    },
                ],
                temperature=0,
                max_tokens=100,
            )

            raw_response = (
                response.choices[0]
                .message
                .content
            )

            print("\nRaw Groq response:")
            print("------------------")
            print(repr(raw_response))

            if not raw_response:
                raise ValueError(
                    "Groq returned an empty response"
                )

            cleaned = raw_response.strip()

            # ------------------------------------------------
            # Remove markdown fences if model adds them
            # ------------------------------------------------

            if cleaned.startswith("```"):

                cleaned = (
                    cleaned
                    .replace("```json", "")
                    .replace("```JSON", "")
                    .replace("```", "")
                    .strip()
                )

            print("\nCleaned response:")
            print("-----------------")
            print(cleaned)

            # ------------------------------------------------
            # Parse JSON
            # ------------------------------------------------

            result = json.loads(cleaned)

            print("\nParsed result:")
            print(result)

            # ------------------------------------------------
            # Validate result
            # ------------------------------------------------

            if "score" not in result:
                raise ValueError(
                    "Missing 'score' field"
                )

            if "label" not in result:
                raise ValueError(
                    "Missing 'label' field"
                )

            score = float(result["score"])
            label = str(result["label"]).lower().strip()

            if not -1 <= score <= 1:
                raise ValueError(
                    f"Invalid sentiment score: {score}"
                )

            if label not in {
                "positive",
                "negative",
                "neutral",
            }:
                raise ValueError(
                    f"Invalid sentiment label: {label}"
                )

            return {
                "score": score,
                "label": label,
            }

        except Exception as e:

            print("\n!!!!!!!!!!!!!!!!!!!!!!!!")
            print("SENTIMENT ERROR")
            print("!!!!!!!!!!!!!!!!!!!!!!!!")
            print(
                f"Error type: {type(e).__name__}"
            )
            print(f"Error: {e}")

            if attempt < max_retries:

                wait_time = 2 ** attempt

                print(
                    f"Retrying in {wait_time} seconds..."
                )

                time.sleep(wait_time)

    # --------------------------------------------------------
    # Safe fallback
    # --------------------------------------------------------

    print("\nAll Groq attempts failed.")

    return {
        "score": 0.0,
        "label": "neutral",
    }


# ============================================================
# PROCESS NEWS ARTICLES
# ============================================================

def process_news_sentiment(
    symbol: str = "NVDA",
):
    """
    Process all articles that do not yet have sentiment.
    """

    engine = create_engine(
        settings.database_url
    )

    select_sql = text(
        """
        SELECT
            article_id,
            title,
            description
        FROM news_articles
        WHERE symbol = :symbol
          AND sentiment_score IS NULL
        ORDER BY published_at
        """
    )

    update_sql = text(
        """
        UPDATE news_articles
        SET
            sentiment_score = :score,
            sentiment_label = :label
        WHERE article_id = :article_id
        """
    )

    processed = 0
    failed = 0

    print("\n")
    print("=" * 60)
    print("NEWS SENTIMENT ANALYSIS")
    print("=" * 60)

    with engine.begin() as connection:

        articles = connection.execute(
            select_sql,
            {
                "symbol": symbol
            },
        ).fetchall()

        print(
            f"Articles requiring sentiment: "
            f"{len(articles)}"
        )

        for index, article in enumerate(
            articles,
            start=1,
        ):

            article_id = article.article_id
            title = article.title
            description = article.description

            print("\n" + "-" * 60)

            print(
                f"Article {index}/{len(articles)}"
            )

            print(
                f"Title: {title}"
            )

            try:

                result = analyze_sentiment(
                    title=title,
                    description=description,
                )

                connection.execute(
                    update_sql,
                    {
                        "score": result["score"],
                        "label": result["label"],
                        "article_id": article_id,
                    },
                )

                processed += 1

                print(
                    f"Result: "
                    f"{result['score']} | "
                    f"{result['label']}"
                )

            except Exception as e:

                failed += 1

                print(
                    f"Failed article "
                    f"{index}: {e}"
                )

    print("\n")
    print("=" * 60)
    print("Sentiment Analysis Done")
    print("=" * 60)

    print(
        f"Processed: {processed}"
    )

    print(
        f"Failed:    {failed}"
    )


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    process_news_sentiment(
        symbol="NVDA"
    )