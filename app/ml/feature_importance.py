import pandas as pd
import matplotlib.pyplot as plt

from app.features.combined import create_feature_dataset
from app.ml.train_context import (
    FEATURE_COLUMNS,
    TARGET_COLUMN,
    create_model,
)


def analyze_feature_importance(
    symbol: str,
    period: str = "2y",
):

    df = create_feature_dataset(
        symbol=symbol,
        period=period,
    )

    df = df.dropna(
        subset=FEATURE_COLUMNS + [TARGET_COLUMN]
    ).reset_index(drop=True)

    X = df[FEATURE_COLUMNS]
    y = df[TARGET_COLUMN]

    model = create_model()

    model.fit(X, y)

    importance = pd.DataFrame({
        "feature": FEATURE_COLUMNS,
        "importance": model.feature_importances_,
    })

    importance = importance.sort_values(
        "importance",
        ascending=False,
    ).reset_index(drop=True)

    print("\nFeature Importance")
    print("==================")

    print(
        importance.to_string(index=False)
    )

    # Plot top 15
    top_features = importance.head(15)

    plt.figure(figsize=(10, 7))

    plt.barh(
        top_features["feature"][::-1],
        top_features["importance"][::-1],
    )

    plt.xlabel("Importance")
    plt.ylabel("Feature")
    plt.title(
        f"XGBoost Feature Importance - {symbol}"
    )

    plt.tight_layout()
    plt.show()

    return importance