import json
import re
import time

from groq import Groq

from app.config import settings


# ---------------------------------------------------------
# Groq client
# ---------------------------------------------------------

client = Groq(
    api_key=settings.groq_api_key
)


# ---------------------------------------------------------
# Model
# ---------------------------------------------------------

MODEL_NAME = "openai/gpt-oss-20b"


# ---------------------------------------------------------
# Sentiment analysis
# ---------------------------------------------------------

def analyze_sentiment(
    title: str,
    description: str | None = None,
    content: str | None = None,
    max_retries: int = 3,
) -> dict:

    description = description or ""
    content = content or ""

    # Keep input reasonably small.
    text = f"""
TITLE:
{title}

DESCRIPTION:
{description}

CONTENT:
{content}
"""

    text = text[:12000]

    prompt = f"""
You are a financial news sentiment classifier.

Analyze the sentiment of the following news article specifically
with respect to NVIDIA (NVDA) and its stock/business outlook.

Return ONLY valid JSON.

The JSON must have exactly these two fields:

{{
  "score": 0.0,
  "label": "neutral"
}}

Rules:

1. score must be a number between -1.0 and 1.0.
2. -1.0 means extremely negative.
3. 0.0 means neutral.
4. 1.0 means extremely positive.
5. label must be exactly one of:
   "positive"
   "negative"
   "neutral"

Do not include markdown.
Do not include explanations.
Do not include additional fields.

ARTICLE:
{text}
"""

    for attempt in range(1, max_retries + 1):

        try:

            print(
                f"\nCalling Groq "
                f"(attempt {attempt}/{max_retries})..."
            )

            response = client.chat.completions.create(
                model=MODEL_NAME,

                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a precise financial "
                            "sentiment classifier. "
                            "Follow the requested output "
                            "format exactly."
                        ),
                    },
                    {
                        "role": "user",
                        "content": prompt,
                    },
                ],

                temperature=0,

                # IMPORTANT:
                # Do NOT use response_format=json_object
                # here.
                #
                # We parse and validate the JSON ourselves.
            )

            raw = response.choices[0].message.content

            print("\nRaw Groq response:")
            print("------------------")
            print(repr(raw))

            if not raw:
                raise ValueError(
                    "Groq returned an empty response"
                )

            cleaned = clean_json_response(raw)

            print("\nCleaned response:")
            print("-----------------")
            print(cleaned)

            result = parse_sentiment(cleaned)

            print("\nParsed result:")
            print(result)

            return result

        except Exception as error:

            print(
                "\n!!!!!!!!!!!!!!!!!!!!!!!!"
            )

            print(
                "SENTIMENT ERROR"
            )

            print(
                "!!!!!!!!!!!!!!!!!!!!!!!!"
            )

            print(
                f"Error type: "
                f"{type(error).__name__}"
            )

            print(
                f"Error: {error}"
            )

            if attempt < max_retries:

                wait_time = 2 ** attempt

                print(
                    f"\nRetrying in "
                    f"{wait_time} seconds..."
                )

                time.sleep(wait_time)

            else:

                print(
                    "\nAll Groq attempts failed."
                )

    # -----------------------------------------------------
    # Safe fallback
    # -----------------------------------------------------

    return {
        "score": 0.0,
        "label": "neutral",
    }


# ---------------------------------------------------------
# Clean model output
# ---------------------------------------------------------

def clean_json_response(raw: str) -> str:

    cleaned = raw.strip()

    # Remove markdown code fences.
    cleaned = re.sub(
        r"^```(?:json)?\s*",
        "",
        cleaned,
        flags=re.IGNORECASE,
    )

    cleaned = re.sub(
        r"\s*```$",
        "",
        cleaned,
    )

    cleaned = cleaned.strip()

    # Find the JSON object if the model added
    # accidental text around it.
    start = cleaned.find("{")
    end = cleaned.rfind("}")

    if start != -1 and end != -1:
        cleaned = cleaned[start:end + 1]

    return cleaned


# ---------------------------------------------------------
# Parse and validate sentiment
# ---------------------------------------------------------

def parse_sentiment(
    response: str,
) -> dict:

    result = json.loads(response)

    if not isinstance(result, dict):
        raise ValueError(
            "Groq response is not a JSON object"
        )

    if "score" not in result:
        raise ValueError(
            "Missing 'score' field"
        )

    if "label" not in result:
        raise ValueError(
            "Missing 'label' field"
        )

    # ----------------------------------------------
    # Validate score
    # ----------------------------------------------

    try:

        score = float(
            result["score"]
        )

    except (TypeError, ValueError):

        raise ValueError(
            "Score must be numeric"
        )

    score = max(
        -1.0,
        min(1.0, score)
    )

    # ----------------------------------------------
    # Validate label
    # ----------------------------------------------

    label = str(
        result["label"]
    ).lower().strip()

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