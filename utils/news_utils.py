import requests
import pandas as pd
import os



def fetch_latest_news(topic, max_articles=10):
    api_key = os.getenv("NEWS_API_KEY")
    url = "https://newsapi.org/v2/everything"

    params = {
        "q": topic,
        "pageSize": max_articles,
        "sortBy": "publishedAt",
        "language": "en",
        "apiKey": api_key
    }

    response = requests.get(url, params=params)
    data = response.json()

    articles = []

    for article in data.get("articles", []):

        articles.append({
            "title": article.get("title", ""),
            "body": article.get("content") or article.get("description", ""),
            "date": article.get("publishedAt", ""),
            "source": article.get("source", {}).get("name", "")
        })

    return pd.DataFrame(articles)