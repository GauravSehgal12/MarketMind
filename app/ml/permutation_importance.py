import pandas as pd

from sklearn.inspection import permutation_importance
from sklearn.metrics import roc_auc_score

from app.features.combined import create_feature_dataset
from app.ml.train_context import (
    FEATURE_COLUMNS,
    TARGET_COLUMN,
    create_model,
)


def analyze_permutation_importance(
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

    # Keep chronological split
    split = int(len(df) * 0.8)

    X_train = X.iloc[:split]
    y_train = y.iloc[:split]

    X_test = X.iloc[split:]
    y_test = y.iloc[split:]

    model = create_model()

    model.fit(
        X_train,
        y_train,
    )

    baseline_probability = model.predict_proba(
        X_test
    )[:, 1]

    baseline_auc = roc_auc_score(
        y_test,
        baseline_probability,
    )

    print(
        f"\nBaseline ROC-AUC: "
        f"{baseline_auc:.4f}"
    )

    result = permutation_importance(
        model,
        X_test,
        y_test,
        scoring="roc_auc",
        n_repeats=10,
        random_state=42,
    )

    importance = pd.DataFrame({
        "feature": FEATURE_COLUMNS,
        "importance_mean": result["importances_mean"],
        "importance_std": result["importances_std"],
    })

    importance = importance.sort_values(
        "importance_mean",
        ascending=False,
    ).reset_index(drop=True)

    print("\nPermutation Importance")
    print("======================")

    print(
        importance.to_string(index=False)
    )

    return importance