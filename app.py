import os
import streamlit as st
import pandas as pd

from utils.news_utils import fetch_latest_news
from utils.pdf_utils import chunk_pages
from utils.rag_utils import build_chunk_index
from utils.agent_utils import (
    analyze_news_sentiment,
    answer_question_with_sources
)

st.set_page_config(page_title="NewsMate Agent", layout="wide")

try:
    if "OPENAI_API_KEY" in st.secrets:
        os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]
    if "OPENAI_MODEL" in st.secrets:
        os.environ["OPENAI_MODEL"] = st.secrets["OPENAI_MODEL"]
    if "EMBEDDING_MODEL" in st.secrets:
        os.environ["EMBEDDING_MODEL"] = st.secrets["EMBEDDING_MODEL"]
except Exception:
    pass


if "news_df" not in st.session_state:
    st.session_state.news_df = pd.DataFrame()

if "sentiment_df" not in st.session_state:
    st.session_state.sentiment_df = pd.DataFrame()

if "indexed_chunks" not in st.session_state:
    st.session_state.indexed_chunks = []

if "processed" not in st.session_state:
    st.session_state.processed = False


st.sidebar.title("News Topic Setup")

topic = st.sidebar.text_input(
    "Enter Topic",
    placeholder="Example: Artificial Intelligence"
)

top_k_chunks = st.sidebar.slider(
    "Top retrieved chunks for Q&A",
    min_value=3,
    max_value=10,
    value=6
)

st.title("NewsMate Agent")
st.caption("AI Agent for News Sentiment Analysis and Question Answering")

process_btn = st.button("Fetch Latest News")


if process_btn:

    if not topic.strip():
        st.error("Please enter a topic.")

    else:

        with st.spinner("Fetching news articles..."):

            news_df = fetch_latest_news(topic)

        if news_df.empty:
            st.error("No news articles found.")

        else:

            st.session_state.news_df = news_df

            with st.spinner("Analyzing sentiment..."):

                sentiment_df = analyze_news_sentiment(news_df)

            st.session_state.sentiment_df = sentiment_df

            articles = []

            for idx, row in news_df.iterrows():

                articles.append({
                    "page_number": idx,
                    "text": row["title"] + ". " + row["body"]
                })

            chunks = chunk_pages(articles, chunk_size=1200, overlap=200)

            indexed_chunks = build_chunk_index(chunks)

            for chunk in indexed_chunks:
                chunk["source"] = "news"

            st.session_state.indexed_chunks = indexed_chunks
            st.session_state.processed = True

            st.success("News processed successfully.")


if st.session_state.processed:

    st.markdown("---")

    tab1, tab2, tab3 = st.tabs([
        "News Articles",
        "Sentiment Analysis",
        "Q&A"
    ])

    with tab1:

        st.subheader("Collected News Articles")

        st.dataframe(
            st.session_state.news_df,
            use_container_width=True
        )


    with tab2:

        st.subheader("Sentiment Comparison")

        st.dataframe(
            st.session_state.sentiment_df,
            use_container_width=True
        )


    with tab3:

        st.subheader("Ask Questions About the News")

        user_question = st.text_input(
            "Enter your question",
            placeholder="Example: What are the main concerns about AI?"
        )

        if st.button("Answer Question"):

            if not user_question.strip():
                st.warning("Please enter a question.")

            else:

                with st.spinner("Searching articles and generating answer..."):

                    answer, sources = answer_question_with_sources(
                        question=user_question,
                        indexed_chunks=st.session_state.indexed_chunks,
                        analyses=[],
                        top_k=top_k_chunks
                    )

                st.markdown("### Answer")
                st.write(answer)

                st.markdown("### Retrieved Source Chunks")

                for source in sources:

                    with st.expander(
                        f"Chunk {source.get('chunk_id')} | Score {source.get('score',0):.4f}"
                    ):
                        st.write(source["text"])

else:
    st.info("Enter a topic and click 'Fetch Latest News'.")