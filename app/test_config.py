from config import settings

print("Configuration loaded successfully!")
print("Finnhub configured:", bool(settings.finnhub_api_key))
print("FRED configured:", bool(settings.fred_api_key))
print("Groq configured:", bool(settings.groq_api_key))
print("Database configured:", bool(settings.database_url))