from app.ml.final_model import evaluate_final_model


results = evaluate_final_model(
    symbol="NVDA",
    period="2y",
    n_splits=5,
)

print("\n")
print(results)