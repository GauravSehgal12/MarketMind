import pandas as pd

from sklearn.metrics import accuracy_score, roc_auc_score
from xgboost import XGBClassifier

from app.features.combined import create_feature_dataset


FEATURE_COLUMNS = [
    # NVDA price / technical features
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

TARGET_COLUMN = "next_day_direction"


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


def walk_forward_context_validation(
    symbol: str,
    period: str = "2y",
    n_splits: int = 5,
):

    print("\nLoading feature dataset...")

    df = create_feature_dataset(
        symbol=symbol,
        period=period,
    )

    df = df.sort_values(
        "timestamp"
    ).reset_index(drop=True)

    df = df.dropna(
        subset=FEATURE_COLUMNS + [
            TARGET_COLUMN
        ]
    ).reset_index(drop=True)

    if len(df) < 100:
        raise ValueError(
            "Not enough data for validation."
        )

    X = df[FEATURE_COLUMNS]
    y = df[TARGET_COLUMN]

    total_rows = len(df)

    initial_train_size = int(
        total_rows * 0.60
    )

    remaining_rows = (
        total_rows - initial_train_size
    )

    test_size = remaining_rows // n_splits

    results = []

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

    print("\nWalk-Forward Validation")
    print("========================")

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
                test_start + test_size
            )

        X_train = X.iloc[:train_end]
        y_train = y.iloc[:train_end]

        X_test = X.iloc[
            test_start:test_end
        ]

        y_test = y.iloc[
            test_start:test_end
        ]

        model = create_model()

        model.fit(
            X_train,
            y_train,
        )

        predictions = model.predict(
            X_test
        )

        probabilities = model.predict_proba(
            X_test
        )[:, 1]

        accuracy = accuracy_score(
            y_test,
            predictions,
        )

        if len(y_test.unique()) > 1:

            auc = roc_auc_score(
                y_test,
                probabilities,
            )

        else:

            auc = float("nan")

        results.append({
            "fold": fold + 1,
            "train_size": len(X_train),
            "test_size": len(X_test),
            "accuracy": accuracy,
            "roc_auc": auc,
        })

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
            f"Accuracy: {accuracy:.4f}"
        )

        print(
            f"ROC-AUC: {auc:.4f}"
        )

    results_df = pd.DataFrame(results)

    print("\n========================")
    print("Overall Results")
    print("========================")

    print(
        f"Mean Accuracy: "
        f"{results_df['accuracy'].mean():.4f}"
    )

    print(
        f"Mean ROC-AUC: "
        f"{results_df['roc_auc'].mean():.4f}"
    )

    return results_df