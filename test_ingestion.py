from app.ml.new_models import (
    walk_forward_news_validation
)


results = walk_forward_news_validation(
    symbol="NVDA",
    period="2y",
    n_splits=5,
)

print("\n")
print("Validation completed.")