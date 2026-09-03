import pandas as pd

from sklearn.metrics import accuracy_score, roc_auc_score
from xgboost import XGBClassifier

from app.features.combined import create_feature_dataset
from app.features.news_merge import add_news_features


# ============================================================
# TECHNICAL + MARKET FEATURES
# ============================================================

TECHNICAL_MARKET_FEATURES = [

    # NVDA price / technical
    "close",
    "return_1d",
    "return_5d",
    "sma_20",
    "sma_50",
    "ema_20",
    "rsi_14",
    "macd",
    "macd_signal",
    "macd_histogram",
    "atr_14",
    "volatility_20",
    "volume_change",

    # Market context
    "spy_return_1d",
    "spy_return_5d",
    "spy_volatility_20",

    "qqq_return_1d",
    "qqq_return_5d",

    "vix_level",
    "vix_change",

    # Relative performance
    "nvda_vs_spy_1d",
    "nvda_vs_qqq_1d",

    "market_breadth_score",
]


# ============================================================
# NEWS FEATURES
#
# IMPORTANT:
# These are LAGGED features.
#
# News from day T is used to predict day T+1.
# ============================================================

NEWS_FEATURES = [

    "news_sentiment_mean_lag1",
    "news_article_count_lag1",
    "positive_news_ratio_lag1",
    "negative_news_ratio_lag1",
    "news_available_lag1",
]


# ============================================================
# ALL FEATURES
# ============================================================

FEATURE_COLUMNS = (
    TECHNICAL_MARKET_FEATURES
    + NEWS_FEATURES
)


TARGET_COLUMN = "next_day_direction"


# ============================================================
# MODEL
# ============================================================

def create_model():

    return XGBClassifier(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.03,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=42,
    )


# ============================================================
# WALK-FORWARD VALIDATION
# ============================================================

def walk_forward_news_validation(
    symbol: str = "NVDA",
    period: str = "2y",
    n_splits: int = 5,
):

    print("\n")
    print("=" * 60)
    print("LOADING FEATURE DATASET")
    print("=" * 60)

    # --------------------------------------------------------
    # Technical + market dataset
    # --------------------------------------------------------

    df = create_feature_dataset(
        symbol=symbol,
        period=period,
    )

    print(
        f"Technical/market rows: {len(df)}"
    )

    # --------------------------------------------------------
    # Add news features
    # --------------------------------------------------------

    print("\nAdding news features...")

    df = add_news_features(
        df,
        symbol=symbol,
    )

    print(
        f"Rows after news merge: {len(df)}"
    )

    # --------------------------------------------------------
    # Sort chronologically
    # --------------------------------------------------------

    df = (
        df
        .sort_values("timestamp")
        .reset_index(drop=True)
    )

    # --------------------------------------------------------
    # Check columns
    # --------------------------------------------------------

    missing_columns = [
        column
        for column in FEATURE_COLUMNS + [TARGET_COLUMN]
        if column not in df.columns
    ]

    if missing_columns:

        raise ValueError(
            "Missing required columns:\n"
            + "\n".join(missing_columns)
        )

    # --------------------------------------------------------
    # Fill missing news values
    # --------------------------------------------------------

    for column in NEWS_FEATURES:

        df[column] = (
            df[column]
            .fillna(0)
        )

    # --------------------------------------------------------
    # Drop rows with missing technical features
    # --------------------------------------------------------

    df = df.dropna(
        subset=(
            TECHNICAL_MARKET_FEATURES
            + [TARGET_COLUMN]
        )
    ).reset_index(drop=True)

    # --------------------------------------------------------
    # Minimum data check
    # --------------------------------------------------------

    if len(df) < 100:

        raise ValueError(
            "Not enough data for validation."
        )

    # --------------------------------------------------------
    # X / y
    # --------------------------------------------------------

    X = df[
        FEATURE_COLUMNS
    ]

    y = df[
        TARGET_COLUMN
    ]

    total_rows = len(df)

    initial_train_size = int(
        total_rows * 0.60
    )

    remaining_rows = (
        total_rows
        - initial_train_size
    )

    test_size = (
        remaining_rows
        // n_splits
    )

    if test_size < 1:

        raise ValueError(
            "Test size is too small."
        )

    # ========================================================
    # DATASET INFORMATION
    # ========================================================

    print("\n")
    print("=" * 60)
    print("NEWS-AUGMENTED DATASET")
    print("=" * 60)

    print(
        f"Total usable rows: {total_rows}"
    )

    print(
        f"Initial training rows: "
        f"{initial_train_size}"
    )

    print(
        f"Test rows per fold: "
        f"{test_size}"
    )

    print(
        f"Total features: "
        f"{len(FEATURE_COLUMNS)}"
    )

    print(
        f"Technical/market features: "
        f"{len(TECHNICAL_MARKET_FEATURES)}"
    )

    print(
        f"News features: "
        f"{len(NEWS_FEATURES)}"
    )

    # ========================================================
    # NEWS COVERAGE
    # ========================================================

    news_rows = (
        df["news_available_lag1"] == 1
    ).sum()

    news_coverage = (
        news_rows / len(df)
    )

    print(
        f"\nRows containing prior-day news: "
        f"{news_rows}"
    )

    print(
        f"News coverage: "
        f"{news_coverage:.2%}"
    )

    # ========================================================
    # SHOW NEWS ROWS
    # ========================================================

    news_df = df[
        df["news_available_lag1"] == 1
    ]

    if len(news_df) > 0:

        print("\nNews-covered dates:")
        print(
            news_df[
                [
                    "timestamp",
                    "news_sentiment_mean_lag1",
                    "news_article_count_lag1",
                    "positive_news_ratio_lag1",
                    "negative_news_ratio_lag1",
                ]
            ].to_string(
                index=False
            )
        )

    else:

        print(
            "\nWARNING:"
        )

        print(
            "No rows currently contain "
            "prior-day news."
        )

        print(
            "The news features will therefore "
            "not provide meaningful predictive "
            "signal yet."
        )

    # ========================================================
    # WALK FORWARD
    # ========================================================

    results = []

    print("\n")
    print("=" * 60)
    print("WALK-FORWARD VALIDATION")
    print("=" * 60)

    for fold in range(n_splits):

        train_end = (
            initial_train_size
            + fold * test_size
        )

        test_start = train_end

        if fold == n_splits - 1:

            test_end = total_rows

        else:

            test_end = (
                test_start
                + test_size
            )

        # ----------------------------------------------------
        # Split
        # ----------------------------------------------------

        X_train = X.iloc[
            :train_end
        ]

        y_train = y.iloc[
            :train_end
        ]

        X_test = X.iloc[
            test_start:test_end
        ]

        y_test = y.iloc[
            test_start:test_end
        ]

        # ----------------------------------------------------
        # Train
        # ----------------------------------------------------

        model = create_model()

        model.fit(
            X_train,
            y_train,
        )

        # ----------------------------------------------------
        # Predictions
        # ----------------------------------------------------

        predictions = model.predict(
            X_test
        )

        probabilities = (
            model.predict_proba(
                X_test
            )[:, 1]
        )

        # ----------------------------------------------------
        # Metrics
        # ----------------------------------------------------

        accuracy = accuracy_score(
            y_test,
            predictions,
        )

        if len(
            y_test.unique()
        ) > 1:

            auc = roc_auc_score(
                y_test,
                probabilities,
            )

        else:

            auc = float("nan")

        # ----------------------------------------------------
        # News coverage in test
        # ----------------------------------------------------

        test_news_rows = (
            df.iloc[
                test_start:test_end
            ][
                "news_available_lag1"
            ] == 1
        ).sum()

        test_news_coverage = (
            test_news_rows
            / len(X_test)
        )

        # ----------------------------------------------------
        # Save results
        # ----------------------------------------------------

        results.append(
            {
                "fold": fold + 1,
                "train_size": len(X_train),
                "test_size": len(X_test),
                "accuracy": accuracy,
                "roc_auc": auc,
                "news_rows": test_news_rows,
                "news_coverage": (
                    test_news_coverage
                ),
            }
        )

        # ----------------------------------------------------
        # Print
        # ----------------------------------------------------

        print(
            f"\nFold {fold + 1}"
        )

        print(
            f"Train: {len(X_train)}"
        )

        print(
            f"Test:  {len(X_test)}"
        )

        print(
            f"News coverage: "
            f"{test_news_coverage:.2%}"
        )

        print(
            f"Accuracy: "
            f"{accuracy:.4f}"
        )

        print(
            f"ROC-AUC: "
            f"{auc:.4f}"
        )

    # ========================================================
    # FINAL RESULTS
    # ========================================================

    results_df = pd.DataFrame(
        results
    )

    mean_accuracy = (
        results_df["accuracy"]
        .mean()
    )

    mean_auc = (
        results_df["roc_auc"]
        .mean()
    )

    print("\n")
    print("=" * 60)
    print("FINAL NEWS MODEL RESULTS")
    print("=" * 60)

    print(
        f"Mean Accuracy: "
        f"{mean_accuracy:.4f}"
    )

    print(
        f"Mean ROC-AUC: "
        f"{mean_auc:.4f}"
    )

    print("\nFold Results")

    print(
        results_df.to_string(
            index=False
        )
    )

    # ========================================================
    # BASELINE COMPARISON
    # ========================================================

    baseline_accuracy = 0.5523
    baseline_auc = 0.6121

    accuracy_change = (
        mean_accuracy
        - baseline_accuracy
    )

    auc_change = (
        mean_auc
        - baseline_auc
    )

    print("\n")
    print("=" * 60)
    print("BASELINE vs NEWS MODEL")
    print("=" * 60)

    print(
        f"Baseline Accuracy: "
        f"{baseline_accuracy:.4f}"
    )

    print(
        f"News Model Accuracy: "
        f"{mean_accuracy:.4f}"
    )

    print(
        f"Accuracy Change: "
        f"{accuracy_change:+.4f}"
    )

    print()

    print(
        f"Baseline ROC-AUC: "
        f"{baseline_auc:.4f}"
    )

    print(
        f"News Model ROC-AUC: "
        f"{mean_auc:.4f}"
    )

    print(
        f"ROC-AUC Change: "
        f"{auc_change:+.4f}"
    )

    # ========================================================
    # INTERPRETATION
    # ========================================================

    print("\n")
    print("=" * 60)
    print("INTERPRETATION")
    print("=" * 60)

    if news_coverage < 0.10:

        print(
            "WARNING: News coverage is below 10%."
        )

        print(
            "Do NOT use this result to judge "
            "whether news sentiment is useful."
        )

        print(
            "Historical news coverage needs "
            "to be increased first."
        )

    elif auc_change > 0.02:

        print(
            "News appears to add meaningful "
            "predictive signal."
        )

    elif auc_change > 0:

        print(
            "News provides a small positive "
            "improvement."
        )

    elif auc_change > -0.02:

        print(
            "News does not materially improve "
            "the model."
        )

    else:

        print(
            "News appears to hurt predictive "
            "performance."
        )

    return results_df


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    walk_forward_news_validation(
        symbol="NVDA",
        period="2y",
        n_splits=5,
    )