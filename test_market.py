from app.data.market import get_historical_data


df = get_historical_data("NVDA", period="2y")

print(df.head())
print()

print("Number of rows:", len(df))
print()

print("Columns:")
print(df.columns.tolist())
print()

print("Latest data:")
print(df.tail())