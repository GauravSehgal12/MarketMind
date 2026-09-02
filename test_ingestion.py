from app.ml.permutation_importance import (
    analyze_permutation_importance,
)


importance = analyze_permutation_importance(
    symbol="NVDA",
    period="2y",
)