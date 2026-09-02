from app.data.news import get_news
from app.data.news_storage import save_news_articles


# Fetch NVDA news
articles = get_news(
    symbol="NVDA",
    company_name="NVIDIA",
    days=7,
    page_size=100,
)

print("Articles fetched:", len(articles))

# Save articles to PostgreSQL
inserted = save_news_articles(articles)

print("Articles inserted:", inserted)