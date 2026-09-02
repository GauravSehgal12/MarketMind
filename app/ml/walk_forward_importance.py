import pandas as pd

from sklearn.inspection import permutation_importance

from app.features.combined import create_feature_dataset
from app.ml.train_context import (
    FEATURE_COLUMNS,
    TARGET_COLUMN,
    create_model,
)


def walk_forward_feature_importance(
    symbol: str,
    period: str = "2y",
    n_splits: int = 5,
    n_repeats: int = 10,
):

    df = create_feature_dataset(
        symbol=symbol,
        period=period,
    )

    df = (
        df.sort_values("timestamp")
        .dropna(
            subset=FEATURE_COLUMNS + [TARGET_COLUMN]
        )
        .reset_index(drop=True)
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

    all_results = []

    print("\nWalk-Forward Permutation Importance")
    print("====================================")

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

        X_test = X.iloc[test_start:test_end]
        y_test = y.iloc[test_start:test_end]

        model = create_model()

        model.fit(
            X_train,
            y_train,
        )

        # Permutation importance
        result = permutation_importance(
            model,
            X_test,
            y_test,
            scoring="roc_auc",
            n_repeats=n_repeats,
            random_state=42,
        )

        fold_importance = pd.DataFrame({
            "feature": FEATURE_COLUMNS,
            "importance_mean": (
                result["importances_mean"]
            ),
            "importance_std": (
                result["importances_std"]
            ),
        })

        fold_importance["fold"] = (
            fold + 1
        )

        all_results.append(
            fold_importance
        )

        print(
            f"\nFold {fold + 1}"
        )

        print(
            fold_importance
            .sort_values(
                "importance_mean",
                ascending=False,
            )
            .head(10)
            .to_string(index=False)
        )

    # Combine all folds
    all_results_df = pd.concat(
        all_results,
        ignore_index=True,
    )

    # Aggregate across folds
    summary = (
        all_results_df
        .groupby("feature")
        .agg(
            mean_importance=(
                "importance_mean",
                "mean",
            ),
            std_importance=(
                "importance_mean",
                "std",
            ),
            avg_permutation_std=(
                "importance_std",
                "mean",
            ),
        )
        .reset_index()
    )

    summary = summary.sort_values(
        "mean_importance",
        ascending=False,
    ).reset_index(drop=True)

    print("\n")
    print("====================================")
    print("Overall Feature Importance")
    print("====================================")

    print(
        summary.to_string(index=False)
    )

    return summary, all_results_df