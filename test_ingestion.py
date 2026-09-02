from app.agent.sentiment_storage import analyze_pending_articles


print("Starting sentiment storage...")
print("============================")


processed = analyze_pending_articles(
    limit=20
)


print("\n============================")
print(f"Articles processed: {processed}")