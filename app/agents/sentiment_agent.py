import json

from groq import Groq

from app.config import settings


client = Groq(
    api_key=settings.groq_api_key
)


def analyze_sentiment(
    title: str,
    description: str | None = None,
    content: str | None = None,
) -> dict:

    article_text = f"""
Title:
{title}

Description:
{description or ""}

Content:
{content or ""}
"""

    prompt = f"""
You are a financial sentiment analysis agent.

Analyze this news article from the perspective
of its potential impact on the company's stock price.

Return ONLY valid JSON.

The JSON must have exactly these fields:

{{
    "sentiment_score": 0.0,
    "sentiment_label": "neutral",
    "confidence": 0.0,
    "reason": "short explanation"
}}

Rules:

- sentiment_score must be between -1.0 and 1.0
- -1.0 = extremely negative
- 0.0 = neutral
- 1.0 = extremely positive

- sentiment_label must be:
  "positive", "neutral", or "negative"

- confidence must be between 0.0 and 1.0

- Focus on potential stock-market impact

- Do not give investment advice

- Return JSON only

Article:

{article_text}
"""

    response = client.chat.completions.create(
        model="openai/gpt-oss-120b",
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a financial sentiment "
                    "analysis agent. Return only JSON."
                ),
            },
            {
                "role": "user",
                "content": prompt,
            },
        ],
        temperature=0,
    )

    result = response.choices[0].message.content

    if not result:
        raise ValueError(
            "Groq returned an empty response"
        )

    result = result.strip()

    # Remove markdown code fences if returned
    if result.startswith("```"):
        result = result.replace(
            "```json", ""
        ).replace(
            "```", ""
        ).strip()

    data = json.loads(result)

    score = float(
        data["sentiment_score"]
    )

    confidence = float(
        data["confidence"]
    )

    label = data["sentiment_label"]

    if not -1 <= score <= 1:
        raise ValueError(
            "sentiment_score must be between -1 and 1"
        )

    if not 0 <= confidence <= 1:
        raise ValueError(
            "confidence must be between 0 and 1"
        )

    if label not in {
        "positive",
        "neutral",
        "negative",
    }:
        raise ValueError(
            "Invalid sentiment_label"
        )

    return {
        "sentiment_score": score,
        "sentiment_label": label,
        "confidence": confidence,
        "reason": data.get(
            "reason",
            ""
        ),
    }