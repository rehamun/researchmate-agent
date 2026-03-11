import requests
import pandas as pd
import os

NEWS_API_KEY = os.getenv("NEWS_API_KEY")


def fetch_latest_news(topic, max_articles=10):

    url = "https://newsapi.org/v2/everything"

    params = {
        "q": topic,
        "pageSize": max_articles,
        "sortBy": "publishedAt",
        "language": "en",
        "apiKey": NEWS_API_KEY
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